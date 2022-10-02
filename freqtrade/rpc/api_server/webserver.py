import asyncio
import logging
from ipaddress import IPv4Address
from threading import Thread
from typing import Any, Dict

import orjson
import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
# Look into alternatives
from janus import Queue as ThreadedQueue
from starlette.responses import JSONResponse

from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.rpc.api_server.uvicorn_threaded import UvicornServer
from freqtrade.rpc.api_server.ws import ChannelManager
from freqtrade.rpc.rpc import RPC, RPCException, RPCHandler


logger = logging.getLogger(__name__)


class FTJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        """
        Use rapidjson for responses
        Handles NaN and Inf / -Inf in a javascript way by default.
        """
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY)


class ApiServer(RPCHandler):

    __instance = None
    __initialized = False

    _rpc: RPC
    # Backtesting type: Backtesting
    _bt = None
    _bt_data = None
    _bt_timerange = None
    _bt_last_config: Config = {}
    _has_rpc: bool = False
    _bgtask_running: bool = False
    _config: Config = {}
    # Exchange - only available in webserver mode.
    _exchange = None
    # websocket message queue stuff
    _ws_channel_manager = None
    _ws_thread = None
    _ws_loop = None

    def __new__(cls, *args, **kwargs):
        """
        This class is a singleton.
        We'll only have one instance of it around.
        """
        if ApiServer.__instance is None:
            ApiServer.__instance = object.__new__(cls)
            ApiServer.__initialized = False
        return ApiServer.__instance

    def __init__(self, config: Config, standalone: bool = False) -> None:
        ApiServer._config = config
        if self.__initialized and (standalone or self._standalone):
            return
        self._standalone: bool = standalone
        self._server = None
        self._ws_queue = None
        self._ws_background_task = None

        ApiServer.__initialized = True

        api_config = self._config['api_server']

        ApiServer._ws_channel_manager = ChannelManager()

        self.app = FastAPI(title="Freqtrade API",
                           docs_url='/docs' if api_config.get('enable_openapi', False) else None,
                           redoc_url=None,
                           default_response_class=FTJSONResponse,
                           )
        self.configure_app(self.app, self._config)
        self.start_api()

    def add_rpc_handler(self, rpc: RPC):
        """
        Attach rpc handler
        """
        if not self._has_rpc:
            ApiServer._rpc = rpc
            ApiServer._has_rpc = True
        else:
            # This should not happen assuming we didn't mess up.
            raise OperationalException('RPC Handler already attached.')

    def cleanup(self) -> None:
        """ Cleanup pending module resources """
        ApiServer._has_rpc = False
        del ApiServer._rpc
        if self._server and not self._standalone:
            logger.info("Stopping API Server")
            self._server.cleanup()

        if self._ws_thread and self._ws_loop:
            logger.info("Stopping API Server background tasks")

            if self._ws_background_task:
                # Cancel the queue task
                self._ws_background_task.cancel()

            self._ws_thread.join()

        self._ws_thread = None
        self._ws_loop = None
        self._ws_background_task = None

    @classmethod
    def shutdown(cls):
        cls.__initialized = False
        del cls.__instance
        cls.__instance = None
        cls._has_rpc = False
        cls._rpc = None

    def send_msg(self, msg: Dict[str, str]) -> None:
        if self._ws_queue:
            sync_q = self._ws_queue.sync_q
            sync_q.put(msg)

    def handle_rpc_exception(self, request, exc):
        logger.exception(f"API Error calling: {exc}")
        return JSONResponse(
            status_code=502,
            content={'error': f"Error querying {request.url.path}: {exc.message}"}
        )

    def configure_app(self, app: FastAPI, config):
        from freqtrade.rpc.api_server.api_auth import http_basic_or_jwt_token, router_login
        from freqtrade.rpc.api_server.api_backtest import router as api_backtest
        from freqtrade.rpc.api_server.api_v1 import router as api_v1
        from freqtrade.rpc.api_server.api_v1 import router_public as api_v1_public
        from freqtrade.rpc.api_server.api_ws import router as ws_router
        from freqtrade.rpc.api_server.web_ui import router_ui

        app.include_router(api_v1_public, prefix="/api/v1")

        app.include_router(api_v1, prefix="/api/v1",
                           dependencies=[Depends(http_basic_or_jwt_token)],
                           )
        app.include_router(api_backtest, prefix="/api/v1",
                           dependencies=[Depends(http_basic_or_jwt_token)],
                           )
        app.include_router(ws_router, prefix="/api/v1")
        app.include_router(router_login, prefix="/api/v1", tags=["auth"])
        # UI Router MUST be last!
        app.include_router(router_ui, prefix='')

        app.add_middleware(
            CORSMiddleware,
            allow_origins=config['api_server'].get('CORS_origins', []),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app.add_exception_handler(RPCException, self.handle_rpc_exception)

    def start_message_queue(self):
        if self._ws_thread:
            return

        # Create a new loop, as it'll be just for the background thread
        self._ws_loop = asyncio.new_event_loop()

        # Start the thread
        self._ws_thread = Thread(target=self._ws_loop.run_forever)
        self._ws_thread.start()

        # Finally, submit the coro to the thread
        self._ws_background_task = asyncio.run_coroutine_threadsafe(
            self._broadcast_queue_data(), loop=self._ws_loop)

    async def _broadcast_queue_data(self):
        # Instantiate the queue in this coroutine so it's attached to our loop
        self._ws_queue = ThreadedQueue()
        async_queue = self._ws_queue.async_q

        try:
            while True:
                logger.debug("Getting queue messages...")
                # Get data from queue
                message = await async_queue.get()
                logger.debug(f"Found message of type: {message.get('type')}")
                # Broadcast it
                await self._ws_channel_manager.broadcast(message)
                # Limit messages per sec.
                # Could cause problems with queue size if too low, and
                # problems with network traffik if too high.
                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            pass

        # For testing, shouldn't happen when stable
        except Exception as e:
            logger.exception(f"Exception happened in background task: {e}")

        finally:
            # Disconnect channels and stop the loop on cancel
            await self._ws_channel_manager.disconnect_all()
            self._ws_loop.stop()

    def start_api(self):
        """
        Start API ... should be run in thread.
        """
        rest_ip = self._config['api_server']['listen_ip_address']
        rest_port = self._config['api_server']['listen_port']

        logger.info(f'Starting HTTP Server at {rest_ip}:{rest_port}')
        if not IPv4Address(rest_ip).is_loopback:
            logger.warning("SECURITY WARNING - Local Rest Server listening to external connections")
            logger.warning("SECURITY WARNING - This is insecure please set to your loopback,"
                           "e.g 127.0.0.1 in config.json")

        if not self._config['api_server'].get('password'):
            logger.warning("SECURITY WARNING - No password for local REST Server defined. "
                           "Please make sure that this is intentional!")

        if (self._config['api_server'].get('jwt_secret_key', 'super-secret')
                in ('super-secret, somethingrandom')):
            logger.warning("SECURITY WARNING - `jwt_secret_key` seems to be default."
                           "Others may be able to log into your bot.")

        logger.info('Starting Local Rest Server.')
        verbosity = self._config['api_server'].get('verbosity', 'error')

        uvconfig = uvicorn.Config(self.app,
                                  port=rest_port,
                                  host=rest_ip,
                                  use_colors=False,
                                  log_config=None,
                                  access_log=True if verbosity != 'error' else False,
                                  )
        try:
            self._server = UvicornServer(uvconfig)
            if self._standalone:
                self._server.run()
            else:
                self.start_message_queue()
                self._server.run_in_thread()
        except Exception:
            logger.exception("Api server failed to start.")
