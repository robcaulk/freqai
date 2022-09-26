# pragma pylint: disable=W0603
"""
Cryptocurrency Exchanges support
"""
import asyncio
import http
import inspect
import logging
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from math import ceil
from threading import Lock
from typing import Any, Coroutine, Dict, List, Literal, Optional, Tuple, Union

import arrow
import ccxt
import ccxt.async_support as ccxt_async
from cachetools import TTLCache
from ccxt import ROUND_DOWN, ROUND_UP, TICK_SIZE, TRUNCATE, decimal_to_precision
from dateutil import parser
from pandas import DataFrame

from freqtrade.constants import (DEFAULT_AMOUNT_RESERVE_PERCENT, NON_OPEN_EXCHANGE_STATES, BuySell,
                                 Config, EntryExit, ListPairsWithTimeframes, MakerTaker,
                                 PairWithTimeframe)
from freqtrade.data.converter import ohlcv_to_dataframe, trades_dict_to_list
from freqtrade.enums import OPTIMIZE_MODES, CandleType, MarginMode, TradingMode
from freqtrade.exceptions import (DDosProtection, ExchangeError, InsufficientFundsError,
                                  InvalidOrderException, OperationalException, PricingError,
                                  RetryableOrderError, TemporaryError)
from freqtrade.exchange.common import (API_FETCH_ORDER_RETRY_COUNT, BAD_EXCHANGES,
                                       EXCHANGE_HAS_OPTIONAL, EXCHANGE_HAS_REQUIRED,
                                       SUPPORTED_EXCHANGES, remove_credentials, retrier,
                                       retrier_async)
from freqtrade.misc import (chunks, deep_merge_dicts, file_dump_json, file_load_json,
                            safe_value_fallback2)
from freqtrade.plugins.pairlist.pairlist_helpers import expand_pairlist
from freqtrade.util import FtPrecise


CcxtModuleType = Any


logger = logging.getLogger(__name__)


# Workaround for adding samesite support to pre 3.8 python
# Only applies to python3.7, and only on certain exchanges (kraken)
# Replicates the fix from starlette (which is actually causing this problem)
http.cookies.Morsel._reserved["samesite"] = "SameSite"  # type: ignore


class Exchange:

    # Parameters to add directly to buy/sell calls (like agreeing to trading agreement)
    _params: Dict = {}

    # Additional parameters - added to the ccxt object
    _ccxt_params: Dict = {}

    # Dict to specify which options each exchange implements
    # This defines defaults, which can be selectively overridden by subclasses using _ft_has
    # or by specifying them in the configuration.
    _ft_has_default: Dict = {
        "stoploss_on_exchange": False,
        "order_time_in_force": ["GTC"],
        "time_in_force_parameter": "timeInForce",
        "ohlcv_params": {},
        "ohlcv_candle_limit": 500,
        "ohlcv_has_history": True,  # Some exchanges (Kraken) don't provide history via ohlcv
        "ohlcv_partial_candle": True,
        "ohlcv_require_since": False,
        # Check https://github.com/ccxt/ccxt/issues/10767 for removal of ohlcv_volume_currency
        "ohlcv_volume_currency": "base",  # "base" or "quote"
        "tickers_have_quoteVolume": True,
        "tickers_have_price": True,
        "trades_pagination": "time",  # Possible are "time" or "id"
        "trades_pagination_arg": "since",
        "l2_limit_range": None,
        "l2_limit_range_required": True,  # Allow Empty L2 limit (kucoin)
        "mark_ohlcv_price": "mark",
        "mark_ohlcv_timeframe": "8h",
        "ccxt_futures_name": "swap",
        "fee_cost_in_contracts": False,  # Fee cost needs contract conversion
        "needs_trading_fees": False,  # use fetch_trading_fees to cache fees
        "order_props_in_contracts": ['amount', 'cost', 'filled', 'remaining'],
    }
    _ft_has: Dict = {}
    _ft_has_futures: Dict = {}

    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = [
        # TradingMode.SPOT always supported and not required in this list
    ]

    def __init__(self, config: Config, validate: bool = True,
                 load_leverage_tiers: bool = False) -> None:
        """
        Initializes this module with the given config,
        it does basic validation whether the specified exchange and pairs are valid.
        :return: None
        """
        self._api: ccxt.Exchange
        self._api_async: ccxt_async.Exchange = None
        self._markets: Dict = {}
        self._trading_fees: Dict[str, Any] = {}
        self._leverage_tiers: Dict[str, List[Dict]] = {}
        # Lock event loop. This is necessary to avoid race-conditions when using force* commands
        # Due to funding fee fetching.
        self._loop_lock = Lock()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._config: Config = {}

        self._config.update(config)

        # Holds last candle refreshed time of each pair
        self._pairs_last_refresh_time: Dict[PairWithTimeframe, int] = {}
        # Timestamp of last markets refresh
        self._last_markets_refresh: int = 0

        # Cache for 10 minutes ...
        self._cache_lock = Lock()
        self._fetch_tickers_cache: TTLCache = TTLCache(maxsize=2, ttl=60 * 10)
        # Cache values for 1800 to avoid frequent polling of the exchange for prices
        # Caching only applies to RPC methods, so prices for open trades are still
        # refreshed once every iteration.
        self._exit_rate_cache: TTLCache = TTLCache(maxsize=100, ttl=1800)
        self._entry_rate_cache: TTLCache = TTLCache(maxsize=100, ttl=1800)

        # Holds candles
        self._klines: Dict[PairWithTimeframe, DataFrame] = {}

        # Holds all open sell orders for dry_run
        self._dry_run_open_orders: Dict[str, Any] = {}
        remove_credentials(config)

        if config['dry_run']:
            logger.info('Instance is running with dry_run enabled')
        logger.info(f"Using CCXT {ccxt.__version__}")
        exchange_config = config['exchange']
        self.log_responses = exchange_config.get('log_responses', False)

        # Leverage properties
        self.trading_mode: TradingMode = config.get('trading_mode', TradingMode.SPOT)
        self.margin_mode: MarginMode = (
            MarginMode(config.get('margin_mode'))
            if config.get('margin_mode')
            else MarginMode.NONE
        )
        self.liquidation_buffer = config.get('liquidation_buffer', 0.05)

        # Deep merge ft_has with default ft_has options
        self._ft_has = deep_merge_dicts(self._ft_has, deepcopy(self._ft_has_default))
        if self.trading_mode == TradingMode.FUTURES:
            self._ft_has = deep_merge_dicts(self._ft_has_futures, self._ft_has)
        if exchange_config.get('_ft_has_params'):
            self._ft_has = deep_merge_dicts(exchange_config.get('_ft_has_params'),
                                            self._ft_has)
            logger.info("Overriding exchange._ft_has with config params, result: %s", self._ft_has)

        # Assign this directly for easy access
        self._ohlcv_partial_candle = self._ft_has['ohlcv_partial_candle']

        self._trades_pagination = self._ft_has['trades_pagination']
        self._trades_pagination_arg = self._ft_has['trades_pagination_arg']

        # Initialize ccxt objects
        ccxt_config = self._ccxt_config
        ccxt_config = deep_merge_dicts(exchange_config.get('ccxt_config', {}), ccxt_config)
        ccxt_config = deep_merge_dicts(exchange_config.get('ccxt_sync_config', {}), ccxt_config)

        self._api = self._init_ccxt(exchange_config, ccxt_kwargs=ccxt_config)

        ccxt_async_config = self._ccxt_config
        ccxt_async_config = deep_merge_dicts(exchange_config.get('ccxt_config', {}),
                                             ccxt_async_config)
        ccxt_async_config = deep_merge_dicts(exchange_config.get('ccxt_async_config', {}),
                                             ccxt_async_config)
        self._api_async = self._init_ccxt(
            exchange_config, ccxt_async, ccxt_kwargs=ccxt_async_config)

        logger.info(f'Using Exchange "{self.name}"')

        if validate:
            # Initial markets load
            self._load_markets()
            self.validate_config(config)
            self.required_candle_call_count = self.validate_required_startup_candles(
                config.get('startup_candle_count', 0), config.get('timeframe', ''))

        # Converts the interval provided in minutes in config to seconds
        self.markets_refresh_interval: int = exchange_config.get(
            "markets_refresh_interval", 60) * 60

        if self.trading_mode != TradingMode.SPOT and load_leverage_tiers:
            self.fill_leverage_tiers()
        self.additional_exchange_init()

    def __del__(self):
        """
        Destructor - clean up async stuff
        """
        self.close()

    def close(self):
        logger.debug("Exchange object destroyed, closing async loop")
        if (self._api_async and inspect.iscoroutinefunction(self._api_async.close)
                and self._api_async.session):
            logger.debug("Closing async ccxt session.")
            self.loop.run_until_complete(self._api_async.close())

    def validate_config(self, config):
        # Check if timeframe is available
        self.validate_timeframes(config.get('timeframe'))

        # Check if all pairs are available
        self.validate_stakecurrency(config['stake_currency'])
        if not config['exchange'].get('skip_pair_validation'):
            self.validate_pairs(config['exchange']['pair_whitelist'])
        self.validate_ordertypes(config.get('order_types', {}))
        self.validate_order_time_in_force(config.get('order_time_in_force', {}))
        self.validate_trading_mode_and_margin_mode(self.trading_mode, self.margin_mode)
        self.validate_pricing(config['exit_pricing'])
        self.validate_pricing(config['entry_pricing'])

    def _init_ccxt(self, exchange_config: Dict[str, Any], ccxt_module: CcxtModuleType = ccxt,
                   ccxt_kwargs: Dict = {}) -> ccxt.Exchange:
        """
        Initialize ccxt with given config and return valid
        ccxt instance.
        """
        # Find matching class for the given exchange name
        name = exchange_config['name']

        if not is_exchange_known_ccxt(name, ccxt_module):
            raise OperationalException(f'Exchange {name} is not supported by ccxt')

        ex_config = {
            'apiKey': exchange_config.get('key'),
            'secret': exchange_config.get('secret'),
            'password': exchange_config.get('password'),
            'uid': exchange_config.get('uid', ''),
        }
        if ccxt_kwargs:
            logger.info('Applying additional ccxt config: %s', ccxt_kwargs)
        if self._ccxt_params:
            # Inject static options after the above output to not confuse users.
            ccxt_kwargs = deep_merge_dicts(self._ccxt_params, ccxt_kwargs)
        if ccxt_kwargs:
            ex_config.update(ccxt_kwargs)
        try:

            api = getattr(ccxt_module, name.lower())(ex_config)
        except (KeyError, AttributeError) as e:
            raise OperationalException(f'Exchange {name} is not supported') from e
        except ccxt.BaseError as e:
            raise OperationalException(f"Initialization of ccxt failed. Reason: {e}") from e

        self.set_sandbox(api, exchange_config, name)

        return api

    @property
    def _ccxt_config(self) -> Dict:
        # Parameters to add directly to ccxt sync/async initialization.
        if self.trading_mode == TradingMode.MARGIN:
            return {
                "options": {
                    "defaultType": "margin"
                }
            }
        elif self.trading_mode == TradingMode.FUTURES:
            return {
                "options": {
                    "defaultType": self._ft_has["ccxt_futures_name"]
                }
            }
        else:
            return {}

    @property
    def name(self) -> str:
        """exchange Name (from ccxt)"""
        return self._api.name

    @property
    def id(self) -> str:
        """exchange ccxt id"""
        return self._api.id

    @property
    def timeframes(self) -> List[str]:
        return list((self._api.timeframes or {}).keys())

    @property
    def markets(self) -> Dict:
        """exchange ccxt markets"""
        if not self._markets:
            logger.info("Markets were not loaded. Loading them now..")
            self._load_markets()
        return self._markets

    @property
    def precisionMode(self) -> int:
        """exchange ccxt precisionMode"""
        return self._api.precisionMode

    def additional_exchange_init(self) -> None:
        """
        Additional exchange initialization logic.
        .api will be available at this point.
        Must be overridden in child methods if required.
        """
        pass

    def _log_exchange_response(self, endpoint, response) -> None:
        """ Log exchange responses """
        if self.log_responses:
            logger.info(f"API {endpoint}: {response}")

    def ohlcv_candle_limit(
            self, timeframe: str, candle_type: CandleType, since_ms: Optional[int] = None) -> int:
        """
        Exchange ohlcv candle limit
        Uses ohlcv_candle_limit_per_timeframe if the exchange has different limits
        per timeframe (e.g. bittrex), otherwise falls back to ohlcv_candle_limit
        :param timeframe: Timeframe to check
        :param candle_type: Candle-type
        :param since_ms: Starting timestamp
        :return: Candle limit as integer
        """
        return int(self._ft_has.get('ohlcv_candle_limit_per_timeframe', {}).get(
            timeframe, self._ft_has.get('ohlcv_candle_limit')))

    def get_markets(self, base_currencies: List[str] = [], quote_currencies: List[str] = [],
                    spot_only: bool = False, margin_only: bool = False, futures_only: bool = False,
                    tradable_only: bool = True,
                    active_only: bool = False) -> Dict[str, Any]:
        """
        Return exchange ccxt markets, filtered out by base currency and quote currency
        if this was requested in parameters.
        """
        markets = self.markets
        if not markets:
            raise OperationalException("Markets were not loaded.")

        if base_currencies:
            markets = {k: v for k, v in markets.items() if v['base'] in base_currencies}
        if quote_currencies:
            markets = {k: v for k, v in markets.items() if v['quote'] in quote_currencies}
        if tradable_only:
            markets = {k: v for k, v in markets.items() if self.market_is_tradable(v)}
        if spot_only:
            markets = {k: v for k, v in markets.items() if self.market_is_spot(v)}
        if margin_only:
            markets = {k: v for k, v in markets.items() if self.market_is_margin(v)}
        if futures_only:
            markets = {k: v for k, v in markets.items() if self.market_is_future(v)}
        if active_only:
            markets = {k: v for k, v in markets.items() if market_is_active(v)}
        return markets

    def get_quote_currencies(self) -> List[str]:
        """
        Return a list of supported quote currencies
        """
        markets = self.markets
        return sorted(set([x['quote'] for _, x in markets.items()]))

    def get_pair_quote_currency(self, pair: str) -> str:
        """ Return a pair's quote currency (base/quote:settlement) """
        return self.markets.get(pair, {}).get('quote', '')

    def get_pair_base_currency(self, pair: str) -> str:
        """ Return a pair's base currency (base/quote:settlement) """
        return self.markets.get(pair, {}).get('base', '')

    def market_is_future(self, market: Dict[str, Any]) -> bool:
        return (
            market.get(self._ft_has["ccxt_futures_name"], False) is True and
            market.get('linear', False) is True
        )

    def market_is_spot(self, market: Dict[str, Any]) -> bool:
        return market.get('spot', False) is True

    def market_is_margin(self, market: Dict[str, Any]) -> bool:
        return market.get('margin', False) is True

    def market_is_tradable(self, market: Dict[str, Any]) -> bool:
        """
        Check if the market symbol is tradable by Freqtrade.
        Ensures that Configured mode aligns to
        """
        return (
            market.get('quote', None) is not None
            and market.get('base', None) is not None
            and (self.precisionMode != TICK_SIZE
                 # Too low precision will falsify calculations
                 or market.get('precision', {}).get('price') > 1e-11)
            and ((self.trading_mode == TradingMode.SPOT and self.market_is_spot(market))
                 or (self.trading_mode == TradingMode.MARGIN and self.market_is_margin(market))
                 or (self.trading_mode == TradingMode.FUTURES and self.market_is_future(market)))
        )

    def klines(self, pair_interval: PairWithTimeframe, copy: bool = True) -> DataFrame:
        if pair_interval in self._klines:
            return self._klines[pair_interval].copy() if copy else self._klines[pair_interval]
        else:
            return DataFrame()

    def get_contract_size(self, pair: str) -> float:
        if self.trading_mode == TradingMode.FUTURES:
            market = self.markets[pair]
            contract_size: float = 1.0
            if market['contractSize'] is not None:
                # ccxt has contractSize in markets as string
                contract_size = float(market['contractSize'])
            return contract_size
        else:
            return 1

    def _trades_contracts_to_amount(self, trades: List) -> List:
        if len(trades) > 0 and 'symbol' in trades[0]:
            contract_size = self.get_contract_size(trades[0]['symbol'])
            if contract_size != 1:
                for trade in trades:
                    trade['amount'] = trade['amount'] * contract_size
        return trades

    def _order_contracts_to_amount(self, order: Dict) -> Dict:
        if 'symbol' in order and order['symbol'] is not None:
            contract_size = self.get_contract_size(order['symbol'])
            if contract_size != 1:
                for prop in self._ft_has.get('order_props_in_contracts', []):
                    if prop in order and order[prop] is not None:
                        order[prop] = order[prop] * contract_size
        return order

    def _amount_to_contracts(self, pair: str, amount: float) -> float:

        contract_size = self.get_contract_size(pair)
        return amount_to_contracts(amount, contract_size)

    def _contracts_to_amount(self, pair: str, num_contracts: float) -> float:

        contract_size = self.get_contract_size(pair)
        return contracts_to_amount(num_contracts, contract_size)

    def amount_to_contract_precision(self, pair: str, amount: float) -> float:
        """
        Helper wrapper around amount_to_contract_precision
        """
        contract_size = self.get_contract_size(pair)

        return amount_to_contract_precision(amount, self.get_precision_amount(pair),
                                            self.precisionMode, contract_size)

    def set_sandbox(self, api: ccxt.Exchange, exchange_config: dict, name: str) -> None:
        if exchange_config.get('sandbox'):
            if api.urls.get('test'):
                api.urls['api'] = api.urls['test']
                logger.info("Enabled Sandbox API on %s", name)
            else:
                logger.warning(
                    f"No Sandbox URL in CCXT for {name}, exiting. Please check your config.json")
                raise OperationalException(f'Exchange {name} does not provide a sandbox api')

    def _load_async_markets(self, reload: bool = False) -> None:
        try:
            if self._api_async:
                self.loop.run_until_complete(
                    self._api_async.load_markets(reload=reload))

        except (asyncio.TimeoutError, ccxt.BaseError) as e:
            logger.warning('Could not load async markets. Reason: %s', e)
            return

    def _load_markets(self) -> None:
        """ Initialize markets both sync and async """
        try:
            self._markets = self._api.load_markets()
            self._load_async_markets()
            self._last_markets_refresh = arrow.utcnow().int_timestamp
            if self._ft_has['needs_trading_fees']:
                self._trading_fees = self.fetch_trading_fees()

        except ccxt.BaseError:
            logger.exception('Unable to initialize markets.')

    def reload_markets(self) -> None:
        """Reload markets both sync and async if refresh interval has passed """
        # Check whether markets have to be reloaded
        if (self._last_markets_refresh > 0) and (
                self._last_markets_refresh + self.markets_refresh_interval
                > arrow.utcnow().int_timestamp):
            return None
        logger.debug("Performing scheduled market reload..")
        try:
            self._markets = self._api.load_markets(reload=True)
            # Also reload async markets to avoid issues with newly listed pairs
            self._load_async_markets(reload=True)
            self._last_markets_refresh = arrow.utcnow().int_timestamp
            self.fill_leverage_tiers()
        except ccxt.BaseError:
            logger.exception("Could not reload markets.")

    def validate_stakecurrency(self, stake_currency: str) -> None:
        """
        Checks stake-currency against available currencies on the exchange.
        Only runs on startup. If markets have not been loaded, there's been a problem with
        the connection to the exchange.
        :param stake_currency: Stake-currency to validate
        :raise: OperationalException if stake-currency is not available.
        """
        if not self._markets:
            raise OperationalException(
                'Could not load markets, therefore cannot start. '
                'Please investigate the above error for more details.'
            )
        quote_currencies = self.get_quote_currencies()
        if stake_currency not in quote_currencies:
            raise OperationalException(
                f"{stake_currency} is not available as stake on {self.name}. "
                f"Available currencies are: {', '.join(quote_currencies)}")

    def validate_pairs(self, pairs: List[str]) -> None:
        """
        Checks if all given pairs are tradable on the current exchange.
        :param pairs: list of pairs
        :raise: OperationalException if one pair is not available
        :return: None
        """

        if not self.markets:
            logger.warning('Unable to validate pairs (assuming they are correct).')
            return
        extended_pairs = expand_pairlist(pairs, list(self.markets), keep_invalid=True)
        invalid_pairs = []
        for pair in extended_pairs:
            # Note: ccxt has BaseCurrency/QuoteCurrency format for pairs
            if self.markets and pair not in self.markets:
                raise OperationalException(
                    f'Pair {pair} is not available on {self.name} {self.trading_mode.value}. '
                    f'Please remove {pair} from your whitelist.')

                # From ccxt Documentation:
                # markets.info: An associative array of non-common market properties,
                # including fees, rates, limits and other general market information.
                # The internal info array is different for each particular market,
                # its contents depend on the exchange.
                # It can also be a string or similar ... so we need to verify that first.
            elif (isinstance(self.markets[pair].get('info'), dict)
                  and self.markets[pair].get('info', {}).get('prohibitedIn', False)):
                # Warn users about restricted pairs in whitelist.
                # We cannot determine reliably if Users are affected.
                logger.warning(f"Pair {pair} is restricted for some users on this exchange."
                               f"Please check if you are impacted by this restriction "
                               f"on the exchange and eventually remove {pair} from your whitelist.")
            if (self._config['stake_currency'] and
                    self.get_pair_quote_currency(pair) != self._config['stake_currency']):
                invalid_pairs.append(pair)
        if invalid_pairs:
            raise OperationalException(
                f"Stake-currency '{self._config['stake_currency']}' not compatible with "
                f"pair-whitelist. Please remove the following pairs: {invalid_pairs}")

    def get_valid_pair_combination(self, curr_1: str, curr_2: str) -> str:
        """
        Get valid pair combination of curr_1 and curr_2 by trying both combinations.
        """
        for pair in [f"{curr_1}/{curr_2}", f"{curr_2}/{curr_1}"]:
            if pair in self.markets and self.markets[pair].get('active'):
                return pair
        raise ExchangeError(f"Could not combine {curr_1} and {curr_2} to get a valid pair.")

    def validate_timeframes(self, timeframe: Optional[str]) -> None:
        """
        Check if timeframe from config is a supported timeframe on the exchange
        """
        if not hasattr(self._api, "timeframes") or self._api.timeframes is None:
            # If timeframes attribute is missing (or is None), the exchange probably
            # has no fetchOHLCV method.
            # Therefore we also show that.
            raise OperationalException(
                f"The ccxt library does not provide the list of timeframes "
                f"for the exchange {self.name} and this exchange "
                f"is therefore not supported. ccxt fetchOHLCV: {self.exchange_has('fetchOHLCV')}")

        if timeframe and (timeframe not in self.timeframes):
            raise OperationalException(
                f"Invalid timeframe '{timeframe}'. This exchange supports: {self.timeframes}")

        if timeframe and timeframe_to_minutes(timeframe) < 1:
            raise OperationalException("Timeframes < 1m are currently not supported by Freqtrade.")

    def validate_ordertypes(self, order_types: Dict) -> None:
        """
        Checks if order-types configured in strategy/config are supported
        """
        if any(v == 'market' for k, v in order_types.items()):
            if not self.exchange_has('createMarketOrder'):
                raise OperationalException(
                    f'Exchange {self.name} does not support market orders.')

        if (order_types.get("stoploss_on_exchange")
                and not self._ft_has.get("stoploss_on_exchange", False)):
            raise OperationalException(
                f'On exchange stoploss is not supported for {self.name}.'
            )

    def validate_pricing(self, pricing: Dict) -> None:
        if pricing.get('use_order_book', False) and not self.exchange_has('fetchL2OrderBook'):
            raise OperationalException(f'Orderbook not available for {self.name}.')
        if (not pricing.get('use_order_book', False) and (
                not self.exchange_has('fetchTicker')
                or not self._ft_has['tickers_have_price'])):
            raise OperationalException(f'Ticker pricing not available for {self.name}.')

    def validate_order_time_in_force(self, order_time_in_force: Dict) -> None:
        """
        Checks if order time in force configured in strategy/config are supported
        """
        if any(v.upper() not in self._ft_has["order_time_in_force"]
               for k, v in order_time_in_force.items()):
            raise OperationalException(
                f'Time in force policies are not supported for {self.name} yet.')

    def validate_required_startup_candles(self, startup_candles: int, timeframe: str) -> int:
        """
        Checks if required startup_candles is more than ohlcv_candle_limit().
        Requires a grace-period of 5 candles - so a startup-period up to 494 is allowed by default.
        """

        candle_limit = self.ohlcv_candle_limit(
            timeframe, self._config['candle_type_def'],
            int(date_minus_candles(timeframe, startup_candles).timestamp() * 1000)
            if timeframe else None)
        # Require one more candle - to account for the still open candle.
        candle_count = startup_candles + 1
        # Allow 5 calls to the exchange per pair
        required_candle_call_count = int(
            (candle_count / candle_limit) + (0 if candle_count % candle_limit == 0 else 1))
        if self._ft_has['ohlcv_has_history']:

            if required_candle_call_count > 5:
                # Only allow 5 calls per pair to somewhat limit the impact
                raise OperationalException(
                    f"This strategy requires {startup_candles} candles to start, "
                    "which is more than 5x "
                    f"the amount of candles {self.name} provides for {timeframe}.")
        elif required_candle_call_count > 1:
            raise OperationalException(
                f"This strategy requires {startup_candles} candles to start, which is more than "
                f"the amount of candles {self.name} provides for {timeframe}.")
        if required_candle_call_count > 1:
            logger.warning(f"Using {required_candle_call_count} calls to get OHLCV. "
                           f"This can result in slower operations for the bot. Please check "
                           f"if you really need {startup_candles} candles for your strategy")
        return required_candle_call_count

    def validate_trading_mode_and_margin_mode(
        self,
        trading_mode: TradingMode,
        margin_mode: Optional[MarginMode]  # Only None when trading_mode = TradingMode.SPOT
    ):
        """
        Checks if freqtrade can perform trades using the configured
        trading mode(Margin, Futures) and MarginMode(Cross, Isolated)
        Throws OperationalException:
            If the trading_mode/margin_mode type are not supported by freqtrade on this exchange
        """
        if trading_mode != TradingMode.SPOT and (
            (trading_mode, margin_mode) not in self._supported_trading_mode_margin_pairs
        ):
            mm_value = margin_mode and margin_mode.value
            raise OperationalException(
                f"Freqtrade does not support {mm_value} {trading_mode.value} on {self.name}"
            )

    def get_option(self, param: str, default: Any = None) -> Any:
        """
        Get parameter value from _ft_has
        """
        return self._ft_has.get(param, default)

    def exchange_has(self, endpoint: str) -> bool:
        """
        Checks if exchange implements a specific API endpoint.
        Wrapper around ccxt 'has' attribute
        :param endpoint: Name of endpoint (e.g. 'fetchOHLCV', 'fetchTickers')
        :return: bool
        """
        return endpoint in self._api.has and self._api.has[endpoint]

    def get_precision_amount(self, pair: str) -> Optional[float]:
        """
        Returns the amount precision of the exchange.
        :param pair: Pair to get precision for
        :return: precision for amount or None. Must be used in combination with precisionMode
        """
        return self.markets.get(pair, {}).get('precision', {}).get('amount', None)

    def get_precision_price(self, pair: str) -> Optional[float]:
        """
        Returns the price precision of the exchange.
        :param pair: Pair to get precision for
        :return: precision for price or None. Must be used in combination with precisionMode
        """
        return self.markets.get(pair, {}).get('precision', {}).get('price', None)

    def amount_to_precision(self, pair: str, amount: float) -> float:
        """
        Returns the amount to buy or sell to a precision the Exchange accepts

        """
        return amount_to_precision(amount, self.get_precision_amount(pair), self.precisionMode)

    def price_to_precision(self, pair: str, price: float) -> float:
        """
        Returns the price rounded up to the precision the Exchange accepts.
        Rounds up
        """
        return price_to_precision(price, self.get_precision_price(pair), self.precisionMode)

    def price_get_one_pip(self, pair: str, price: float) -> float:
        """
        Get's the "1 pip" value for this pair.
        Used in PriceFilter to calculate the 1pip movements.
        """
        precision = self.markets[pair]['precision']['price']
        if self.precisionMode == TICK_SIZE:
            return precision
        else:
            return 1 / pow(10, precision)

    def get_min_pair_stake_amount(
        self,
        pair: str,
        price: float,
        stoploss: float,
        leverage: Optional[float] = 1.0
    ) -> Optional[float]:
        return self._get_stake_amount_limit(pair, price, stoploss, 'min', leverage)

    def get_max_pair_stake_amount(self, pair: str, price: float, leverage: float = 1.0) -> float:
        max_stake_amount = self._get_stake_amount_limit(pair, price, 0.0, 'max')
        if max_stake_amount is None:
            # * Should never be executed
            raise OperationalException(f'{self.name}.get_max_pair_stake_amount should'
                                       'never set max_stake_amount to None')
        return max_stake_amount / leverage

    def _get_stake_amount_limit(
        self,
        pair: str,
        price: float,
        stoploss: float,
        limit: Literal['min', 'max'],
        leverage: Optional[float] = 1.0
    ) -> Optional[float]:

        isMin = limit == 'min'

        try:
            market = self.markets[pair]
        except KeyError:
            raise ValueError(f"Can't get market information for symbol {pair}")

        stake_limits = []
        limits = market['limits']
        if (limits['cost'][limit] is not None):
            stake_limits.append(
                self._contracts_to_amount(
                    pair,
                    limits['cost'][limit]
                )
            )

        if (limits['amount'][limit] is not None):
            stake_limits.append(
                self._contracts_to_amount(
                    pair,
                    limits['amount'][limit] * price
                )
            )

        if not stake_limits:
            return None if isMin else float('inf')

        # reserve some percent defined in config (5% default) + stoploss
        amount_reserve_percent = 1.0 + self._config.get('amount_reserve_percent',
                                                        DEFAULT_AMOUNT_RESERVE_PERCENT)
        amount_reserve_percent = (
            amount_reserve_percent / (1 - abs(stoploss)) if abs(stoploss) != 1 else 1.5
        )
        # it should not be more than 50%
        amount_reserve_percent = max(min(amount_reserve_percent, 1.5), 1)

        # The value returned should satisfy both limits: for amount (base currency) and
        # for cost (quote, stake currency), so max() is used here.
        # See also #2575 at github.
        return self._get_stake_amount_considering_leverage(
            max(stake_limits) * amount_reserve_percent,
            leverage or 1.0
        ) if isMin else min(stake_limits)

    def _get_stake_amount_considering_leverage(self, stake_amount: float, leverage: float) -> float:
        """
        Takes the minimum stake amount for a pair with no leverage and returns the minimum
        stake amount when leverage is considered
        :param stake_amount: The stake amount for a pair before leverage is considered
        :param leverage: The amount of leverage being used on the current trade
        """
        return stake_amount / leverage

    # Dry-run methods

    def create_dry_run_order(self, pair: str, ordertype: str, side: str, amount: float,
                             rate: float, leverage: float, params: Dict = {},
                             stop_loss: bool = False) -> Dict[str, Any]:
        order_id = f'dry_run_{side}_{datetime.now().timestamp()}'
        # Rounding here must respect to contract sizes
        _amount = self._contracts_to_amount(
            pair, self.amount_to_precision(pair, self._amount_to_contracts(pair, amount)))
        dry_order: Dict[str, Any] = {
            'id': order_id,
            'symbol': pair,
            'price': rate,
            'average': rate,
            'amount': _amount,
            'cost': _amount * rate,
            'type': ordertype,
            'side': side,
            'filled': 0,
            'remaining': _amount,
            'datetime': arrow.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'timestamp': arrow.utcnow().int_timestamp * 1000,
            'status': "closed" if ordertype == "market" and not stop_loss else "open",
            'fee': None,
            'info': {},
            'leverage': leverage
        }
        if stop_loss:
            dry_order["info"] = {"stopPrice": dry_order["price"]}
            dry_order["stopPrice"] = dry_order["price"]
            # Workaround to avoid filling stoploss orders immediately
            dry_order["ft_order_type"] = "stoploss"

        if dry_order["type"] == "market" and not dry_order.get("ft_order_type"):
            # Update market order pricing
            average = self.get_dry_market_fill_price(pair, side, amount, rate)
            dry_order.update({
                'average': average,
                'filled': _amount,
                'remaining': 0.0,
                'cost': (dry_order['amount'] * average) / leverage
            })
            # market orders will always incurr taker fees
            dry_order = self.add_dry_order_fee(pair, dry_order, 'taker')

        dry_order = self.check_dry_limit_order_filled(dry_order, immediate=True)

        self._dry_run_open_orders[dry_order["id"]] = dry_order
        # Copy order and close it - so the returned order is open unless it's a market order
        return dry_order

    def add_dry_order_fee(
        self,
        pair: str,
        dry_order: Dict[str, Any],
        taker_or_maker: MakerTaker,
    ) -> Dict[str, Any]:
        fee = self.get_fee(pair, taker_or_maker=taker_or_maker)
        dry_order.update({
            'fee': {
                'currency': self.get_pair_quote_currency(pair),
                'cost': dry_order['cost'] * fee,
                'rate': fee
            }
        })
        return dry_order

    def get_dry_market_fill_price(self, pair: str, side: str, amount: float, rate: float) -> float:
        """
        Get the market order fill price based on orderbook interpolation
        """
        if self.exchange_has('fetchL2OrderBook'):
            ob = self.fetch_l2_order_book(pair, 20)
            ob_type = 'asks' if side == 'buy' else 'bids'
            slippage = 0.05
            max_slippage_val = rate * ((1 + slippage) if side == 'buy' else (1 - slippage))

            remaining_amount = amount
            filled_amount = 0.0
            book_entry_price = 0.0
            for book_entry in ob[ob_type]:
                book_entry_price = book_entry[0]
                book_entry_coin_volume = book_entry[1]
                if remaining_amount > 0:
                    if remaining_amount < book_entry_coin_volume:
                        # Orderbook at this slot bigger than remaining amount
                        filled_amount += remaining_amount * book_entry_price
                        break
                    else:
                        filled_amount += book_entry_coin_volume * book_entry_price
                    remaining_amount -= book_entry_coin_volume
                else:
                    break
            else:
                # If remaining_amount wasn't consumed completely (break was not called)
                filled_amount += remaining_amount * book_entry_price
            forecast_avg_filled_price = max(filled_amount, 0) / amount
            # Limit max. slippage to specified value
            if side == 'buy':
                forecast_avg_filled_price = min(forecast_avg_filled_price, max_slippage_val)

            else:
                forecast_avg_filled_price = max(forecast_avg_filled_price, max_slippage_val)

            return self.price_to_precision(pair, forecast_avg_filled_price)

        return rate

    def _is_dry_limit_order_filled(self, pair: str, side: str, limit: float) -> bool:
        if not self.exchange_has('fetchL2OrderBook'):
            return True
        ob = self.fetch_l2_order_book(pair, 1)
        try:
            if side == 'buy':
                price = ob['asks'][0][0]
                logger.debug(f"{pair} checking dry buy-order: price={price}, limit={limit}")
                if limit >= price:
                    return True
            else:
                price = ob['bids'][0][0]
                logger.debug(f"{pair} checking dry sell-order: price={price}, limit={limit}")
                if limit <= price:
                    return True
        except IndexError:
            # Ignore empty orderbooks when filling - can be filled with the next iteration.
            pass
        return False

    def check_dry_limit_order_filled(
            self, order: Dict[str, Any], immediate: bool = False) -> Dict[str, Any]:
        """
        Check dry-run limit order fill and update fee (if it filled).
        """
        if (order['status'] != "closed"
                and order['type'] in ["limit"]
                and not order.get('ft_order_type')):
            pair = order['symbol']
            if self._is_dry_limit_order_filled(pair, order['side'], order['price']):
                order.update({
                    'status': 'closed',
                    'filled': order['amount'],
                    'remaining': 0,
                })

                self.add_dry_order_fee(
                    pair,
                    order,
                    'taker' if immediate else 'maker',
                )

        return order

    def fetch_dry_run_order(self, order_id) -> Dict[str, Any]:
        """
        Return dry-run order
        Only call if running in dry-run mode.
        """
        try:
            order = self._dry_run_open_orders[order_id]
            order = self.check_dry_limit_order_filled(order)
            return order
        except KeyError as e:
            from freqtrade.persistence import Order
            order = Order.order_by_id(order_id)
            if order:
                ccxt_order = order.to_ccxt_object()
                self._dry_run_open_orders[order_id] = ccxt_order
                return ccxt_order
            # Gracefully handle errors with dry-run orders.
            raise InvalidOrderException(
                f'Tried to get an invalid dry-run-order (id: {order_id}). Message: {e}') from e

    # Order handling

    def _lev_prep(self, pair: str, leverage: float, side: BuySell):
        if self.trading_mode != TradingMode.SPOT:
            self.set_margin_mode(pair, self.margin_mode)
            self._set_leverage(leverage, pair)

    def _get_params(
        self,
        side: BuySell,
        ordertype: str,
        leverage: float,
        reduceOnly: bool,
        time_in_force: str = 'GTC',
    ) -> Dict:
        params = self._params.copy()
        if time_in_force != 'GTC' and ordertype != 'market':
            param = self._ft_has.get('time_in_force_parameter', '')
            params.update({param: time_in_force.upper()})
        if reduceOnly:
            params.update({'reduceOnly': True})
        return params

    def create_order(
        self,
        *,
        pair: str,
        ordertype: str,
        side: BuySell,
        amount: float,
        rate: float,
        leverage: float,
        reduceOnly: bool = False,
        time_in_force: str = 'GTC',
    ) -> Dict:
        if self._config['dry_run']:
            dry_order = self.create_dry_run_order(
                pair, ordertype, side, amount, self.price_to_precision(pair, rate), leverage)
            return dry_order

        params = self._get_params(side, ordertype, leverage, reduceOnly, time_in_force)

        try:
            # Set the precision for amount and price(rate) as accepted by the exchange
            amount = self.amount_to_precision(pair, self._amount_to_contracts(pair, amount))
            needs_price = (ordertype != 'market'
                           or self._api.options.get("createMarketBuyOrderRequiresPrice", False))
            rate_for_order = self.price_to_precision(pair, rate) if needs_price else None

            if not reduceOnly:
                self._lev_prep(pair, leverage, side)

            order = self._api.create_order(
                pair,
                ordertype,
                side,
                amount,
                rate_for_order,
                params,
            )
            self._log_exchange_response('create_order', order)
            order = self._order_contracts_to_amount(order)
            return order

        except ccxt.InsufficientFunds as e:
            raise InsufficientFundsError(
                f'Insufficient funds to create {ordertype} {side} order on market {pair}. '
                f'Tried to {side} amount {amount} at rate {rate}.'
                f'Message: {e}') from e
        except ccxt.InvalidOrder as e:
            raise ExchangeError(
                f'Could not create {ordertype} {side} order on market {pair}. '
                f'Tried to {side} amount {amount} at rate {rate}. '
                f'Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not place {side} order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def stoploss_adjust(self, stop_loss: float, order: Dict, side: str) -> bool:
        """
        Verify stop_loss against stoploss-order value (limit or price)
        Returns True if adjustment is necessary.
        """
        raise OperationalException(f"stoploss is not implemented for {self.name}.")

    def _get_stop_order_type(self, user_order_type) -> Tuple[str, str]:

        available_order_Types: Dict[str, str] = self._ft_has["stoploss_order_types"]

        if user_order_type in available_order_Types.keys():
            ordertype = available_order_Types[user_order_type]
        else:
            # Otherwise pick only one available
            ordertype = list(available_order_Types.values())[0]
            user_order_type = list(available_order_Types.keys())[0]
        return ordertype, user_order_type

    def _get_stop_limit_rate(self, stop_price: float, order_types: Dict, side: str) -> float:
        # Limit price threshold: As limit price should always be below stop-price
        limit_price_pct = order_types.get('stoploss_on_exchange_limit_ratio', 0.99)
        if side == "sell":
            limit_rate = stop_price * limit_price_pct
        else:
            limit_rate = stop_price * (2 - limit_price_pct)

        bad_stop_price = ((stop_price <= limit_rate) if side ==
                          "sell" else (stop_price >= limit_rate))
        # Ensure rate is less than stop price
        if bad_stop_price:
            raise OperationalException(
                'In stoploss limit order, stop price should be more than limit price')
        return limit_rate

    def _get_stop_params(self, ordertype: str, stop_price: float) -> Dict:
        params = self._params.copy()
        # Verify if stopPrice works for your exchange!
        params.update({'stopPrice': stop_price})
        return params

    @retrier(retries=0)
    def stoploss(self, pair: str, amount: float, stop_price: float, order_types: Dict,
                 side: BuySell, leverage: float) -> Dict:
        """
        creates a stoploss order.
        requires `_ft_has['stoploss_order_types']` to be set as a dict mapping limit and market
            to the corresponding exchange type.

        The precise ordertype is determined by the order_types dict or exchange default.

        The exception below should never raise, since we disallow
        starting the bot in validate_ordertypes()

        This may work with a limited number of other exchanges, but correct working
            needs to be tested individually.
        WARNING: setting `stoploss_on_exchange` to True will NOT auto-enable stoploss on exchange.
            `stoploss_adjust` must still be implemented for this to work.
        """
        if not self._ft_has['stoploss_on_exchange']:
            raise OperationalException(f"stoploss is not implemented for {self.name}.")

        user_order_type = order_types.get('stoploss', 'market')
        ordertype, user_order_type = self._get_stop_order_type(user_order_type)

        stop_price_norm = self.price_to_precision(pair, stop_price)
        limit_rate = None
        if user_order_type == 'limit':
            limit_rate = self._get_stop_limit_rate(stop_price, order_types, side)
            limit_rate = self.price_to_precision(pair, limit_rate)

        if self._config['dry_run']:
            dry_order = self.create_dry_run_order(
                pair,
                ordertype,
                side,
                amount,
                stop_price_norm,
                stop_loss=True,
                leverage=leverage,
            )
            return dry_order

        try:
            params = self._get_stop_params(ordertype=ordertype, stop_price=stop_price_norm)
            if self.trading_mode == TradingMode.FUTURES:
                params['reduceOnly'] = True

            amount = self.amount_to_precision(pair, self._amount_to_contracts(pair, amount))

            self._lev_prep(pair, leverage, side)
            order = self._api.create_order(symbol=pair, type=ordertype, side=side,
                                           amount=amount, price=limit_rate, params=params)
            self._log_exchange_response('create_stoploss_order', order)
            order = self._order_contracts_to_amount(order)
            logger.info(f"stoploss {user_order_type} order added for {pair}. "
                        f"stop price: {stop_price}. limit: {limit_rate}")
            return order
        except ccxt.InsufficientFunds as e:
            raise InsufficientFundsError(
                f'Insufficient funds to create {ordertype} sell order on market {pair}. '
                f'Tried to sell amount {amount} at rate {limit_rate}. '
                f'Message: {e}') from e
        except ccxt.InvalidOrder as e:
            # Errors:
            # `Order would trigger immediately.`
            raise InvalidOrderException(
                f'Could not create {ordertype} sell order on market {pair}. '
                f'Tried to sell amount {amount} at rate {limit_rate}. '
                f'Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"Could not place stoploss order due to {e.__class__.__name__}. "
                f"Message: {e}") from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier(retries=API_FETCH_ORDER_RETRY_COUNT)
    def fetch_order(self, order_id: str, pair: str, params: Dict = {}) -> Dict:
        if self._config['dry_run']:
            return self.fetch_dry_run_order(order_id)
        try:
            order = self._api.fetch_order(order_id, pair, params=params)
            self._log_exchange_response('fetch_order', order)
            order = self._order_contracts_to_amount(order)
            return order
        except ccxt.OrderNotFound as e:
            raise RetryableOrderError(
                f'Order not found (pair: {pair} id: {order_id}). Message: {e}') from e
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(
                f'Tried to get an invalid order (pair: {pair} id: {order_id}). Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def fetch_stoploss_order(self, order_id: str, pair: str, params: Dict = {}) -> Dict:
        return self.fetch_order(order_id, pair, params)

    def fetch_order_or_stoploss_order(self, order_id: str, pair: str,
                                      stoploss_order: bool = False) -> Dict:
        """
        Simple wrapper calling either fetch_order or fetch_stoploss_order depending on
        the stoploss_order parameter
        :param order_id: OrderId to fetch order
        :param pair: Pair corresponding to order_id
        :param stoploss_order: If true, uses fetch_stoploss_order, otherwise fetch_order.
        """
        if stoploss_order:
            return self.fetch_stoploss_order(order_id, pair)
        return self.fetch_order(order_id, pair)

    def check_order_canceled_empty(self, order: Dict) -> bool:
        """
        Verify if an order has been cancelled without being partially filled
        :param order: Order dict as returned from fetch_order()
        :return: True if order has been cancelled without being filled, False otherwise.
        """
        return (order.get('status') in NON_OPEN_EXCHANGE_STATES
                and order.get('filled') == 0.0)

    @retrier
    def cancel_order(self, order_id: str, pair: str, params: Dict = {}) -> Dict:
        if self._config['dry_run']:
            try:
                order = self.fetch_dry_run_order(order_id)

                order.update({'status': 'canceled', 'filled': 0.0, 'remaining': order['amount']})
                return order
            except InvalidOrderException:
                return {}

        try:
            order = self._api.cancel_order(order_id, pair, params=params)
            self._log_exchange_response('cancel_order', order)
            order = self._order_contracts_to_amount(order)
            return order
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(
                f'Could not cancel order. Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not cancel order due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def cancel_stoploss_order(self, order_id: str, pair: str, params: Dict = {}) -> Dict:
        return self.cancel_order(order_id, pair, params)

    def is_cancel_order_result_suitable(self, corder) -> bool:
        if not isinstance(corder, dict):
            return False

        required = ('fee', 'status', 'amount')
        return all(corder.get(k, None) is not None for k in required)

    def cancel_order_with_result(self, order_id: str, pair: str, amount: float) -> Dict:
        """
        Cancel order returning a result.
        Creates a fake result if cancel order returns a non-usable result
        and fetch_order does not work (certain exchanges don't return cancelled orders)
        :param order_id: Orderid to cancel
        :param pair: Pair corresponding to order_id
        :param amount: Amount to use for fake response
        :return: Result from either cancel_order if usable, or fetch_order
        """
        try:
            corder = self.cancel_order(order_id, pair)
            if self.is_cancel_order_result_suitable(corder):
                return corder
        except InvalidOrderException:
            logger.warning(f"Could not cancel order {order_id} for {pair}.")
        try:
            order = self.fetch_order(order_id, pair)
        except InvalidOrderException:
            logger.warning(f"Could not fetch cancelled order {order_id}.")
            order = {'fee': {}, 'status': 'canceled', 'amount': amount, 'info': {}}

        return order

    def cancel_stoploss_order_with_result(self, order_id: str, pair: str, amount: float) -> Dict:
        """
        Cancel stoploss order returning a result.
        Creates a fake result if cancel order returns a non-usable result
        and fetch_order does not work (certain exchanges don't return cancelled orders)
        :param order_id: stoploss-order-id to cancel
        :param pair: Pair corresponding to order_id
        :param amount: Amount to use for fake response
        :return: Result from either cancel_order if usable, or fetch_order
        """
        corder = self.cancel_stoploss_order(order_id, pair)
        if self.is_cancel_order_result_suitable(corder):
            return corder
        try:
            order = self.fetch_stoploss_order(order_id, pair)
        except InvalidOrderException:
            logger.warning(f"Could not fetch cancelled stoploss order {order_id}.")
            order = {'fee': {}, 'status': 'canceled', 'amount': amount, 'info': {}}

        return order

    @retrier
    def get_balances(self) -> dict:

        try:
            balances = self._api.fetch_balance()
            # Remove additional info from ccxt results
            balances.pop("info", None)
            balances.pop("free", None)
            balances.pop("total", None)
            balances.pop("used", None)

            return balances
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get balance due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def fetch_positions(self, pair: str = None) -> List[Dict]:
        """
        Fetch positions from the exchange.
        If no pair is given, all positions are returned.
        :param pair: Pair for the query
        """
        if self._config['dry_run'] or self.trading_mode != TradingMode.FUTURES:
            return []
        try:
            symbols = []
            if pair:
                symbols.append(pair)
            positions: List[Dict] = self._api.fetch_positions(symbols)
            self._log_exchange_response('fetch_positions', positions)
            return positions
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get positions due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def fetch_trading_fees(self) -> Dict[str, Any]:
        """
        Fetch user account trading fees
        Can be cached, should not update often.
        """
        if (self._config['dry_run'] or self.trading_mode != TradingMode.FUTURES
                or not self.exchange_has('fetchTradingFees')):
            return {}
        try:
            trading_fees: Dict[str, Any] = self._api.fetch_trading_fees()
            self._log_exchange_response('fetch_trading_fees', trading_fees)
            return trading_fees
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not fetch trading fees due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def fetch_bids_asks(self, symbols: Optional[List[str]] = None, cached: bool = False) -> Dict:
        """
        :param cached: Allow cached result
        :return: fetch_tickers result
        """
        if not self.exchange_has('fetchBidsAsks'):
            return {}
        if cached:
            with self._cache_lock:
                tickers = self._fetch_tickers_cache.get('fetch_bids_asks')
            if tickers:
                return tickers
        try:
            tickers = self._api.fetch_bids_asks(symbols)
            with self._cache_lock:
                self._fetch_tickers_cache['fetch_bids_asks'] = tickers
            return tickers
        except ccxt.NotSupported as e:
            raise OperationalException(
                f'Exchange {self._api.name} does not support fetching bids/asks in batch. '
                f'Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not load bids/asks due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def get_tickers(self, symbols: Optional[List[str]] = None, cached: bool = False) -> Dict:
        """
        :param cached: Allow cached result
        :return: fetch_tickers result
        """
        if cached:
            with self._cache_lock:
                tickers = self._fetch_tickers_cache.get('fetch_tickers')
            if tickers:
                return tickers
        try:
            tickers = self._api.fetch_tickers(symbols)
            with self._cache_lock:
                self._fetch_tickers_cache['fetch_tickers'] = tickers
            return tickers
        except ccxt.NotSupported as e:
            raise OperationalException(
                f'Exchange {self._api.name} does not support fetching tickers in batch. '
                f'Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not load tickers due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    # Pricing info

    @retrier
    def fetch_ticker(self, pair: str) -> dict:
        try:
            if (pair not in self.markets or
                    self.markets[pair].get('active', False) is False):
                raise ExchangeError(f"Pair {pair} not available")
            data = self._api.fetch_ticker(pair)
            return data
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not load ticker due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @staticmethod
    def get_next_limit_in_list(limit: int, limit_range: Optional[List[int]],
                               range_required: bool = True):
        """
        Get next greater value in the list.
        Used by fetch_l2_order_book if the api only supports a limited range
        """
        if not limit_range:
            return limit

        result = min([x for x in limit_range if limit <= x] + [max(limit_range)])
        if not range_required and limit > result:
            # Range is not required - we can use None as parameter.
            return None
        return result

    @retrier
    def fetch_l2_order_book(self, pair: str, limit: int = 100) -> dict:
        """
        Get L2 order book from exchange.
        Can be limited to a certain amount (if supported).
        Returns a dict in the format
        {'asks': [price, volume], 'bids': [price, volume]}
        """
        limit1 = self.get_next_limit_in_list(limit, self._ft_has['l2_limit_range'],
                                             self._ft_has['l2_limit_range_required'])
        try:

            return self._api.fetch_l2_order_book(pair, limit1)
        except ccxt.NotSupported as e:
            raise OperationalException(
                f'Exchange {self._api.name} does not support fetching order book.'
                f'Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get order book due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _get_price_side(self, side: str, is_short: bool, conf_strategy: Dict) -> str:
        price_side = conf_strategy['price_side']

        if price_side in ('same', 'other'):
            price_map = {
                ('entry', 'long', 'same'): 'bid',
                ('entry', 'long', 'other'): 'ask',
                ('entry', 'short', 'same'): 'ask',
                ('entry', 'short', 'other'): 'bid',
                ('exit', 'long', 'same'): 'ask',
                ('exit', 'long', 'other'): 'bid',
                ('exit', 'short', 'same'): 'bid',
                ('exit', 'short', 'other'): 'ask',
            }
            price_side = price_map[(side, 'short' if is_short else 'long', price_side)]
        return price_side

    def get_rate(self, pair: str, refresh: bool,
                 side: EntryExit, is_short: bool,
                 order_book: Optional[dict] = None, ticker: Optional[dict] = None) -> float:
        """
        Calculates bid/ask target
        bid rate - between current ask price and last price
        ask rate - either using ticker bid or first bid based on orderbook
        or remain static in any other case since it's not updating.
        :param pair: Pair to get rate for
        :param refresh: allow cached data
        :param side: "buy" or "sell"
        :return: float: Price
        :raises PricingError if orderbook price could not be determined.
        """
        name = side.capitalize()
        strat_name = 'entry_pricing' if side == "entry" else 'exit_pricing'

        cache_rate: TTLCache = self._entry_rate_cache if side == "entry" else self._exit_rate_cache
        if not refresh:
            with self._cache_lock:
                rate = cache_rate.get(pair)
            # Check if cache has been invalidated
            if rate:
                logger.debug(f"Using cached {side} rate for {pair}.")
                return rate

        conf_strategy = self._config.get(strat_name, {})

        price_side = self._get_price_side(side, is_short, conf_strategy)

        price_side_word = price_side.capitalize()

        if conf_strategy.get('use_order_book', False):

            order_book_top = conf_strategy.get('order_book_top', 1)
            if order_book is None:
                order_book = self.fetch_l2_order_book(pair, order_book_top)
            logger.debug('order_book %s', order_book)
            # top 1 = index 0
            try:
                rate = order_book[f"{price_side}s"][order_book_top - 1][0]
            except (IndexError, KeyError) as e:
                logger.warning(
                    f"{pair} - {name} Price at location {order_book_top} from orderbook "
                    f"could not be determined. Orderbook: {order_book}"
                )
                raise PricingError from e
            logger.debug(f"{pair} - {name} price from orderbook {price_side_word}"
                         f"side - top {order_book_top} order book {side} rate {rate:.8f}")
        else:
            logger.debug(f"Using Last {price_side_word} / Last Price")
            if ticker is None:
                ticker = self.fetch_ticker(pair)
            ticker_rate = ticker[price_side]
            if ticker['last'] and ticker_rate:
                if side == 'entry' and ticker_rate > ticker['last']:
                    balance = conf_strategy.get('price_last_balance', 0.0)
                    ticker_rate = ticker_rate + balance * (ticker['last'] - ticker_rate)
                elif side == 'exit' and ticker_rate < ticker['last']:
                    balance = conf_strategy.get('price_last_balance', 0.0)
                    ticker_rate = ticker_rate - balance * (ticker_rate - ticker['last'])
            rate = ticker_rate

        if rate is None:
            raise PricingError(f"{name}-Rate for {pair} was empty.")
        with self._cache_lock:
            cache_rate[pair] = rate

        return rate

    def get_rates(self, pair: str, refresh: bool, is_short: bool) -> Tuple[float, float]:
        entry_rate = None
        exit_rate = None
        if not refresh:
            with self._cache_lock:
                entry_rate = self._entry_rate_cache.get(pair)
                exit_rate = self._exit_rate_cache.get(pair)
            if entry_rate:
                logger.debug(f"Using cached buy rate for {pair}.")
            if exit_rate:
                logger.debug(f"Using cached sell rate for {pair}.")

        entry_pricing = self._config.get('entry_pricing', {})
        exit_pricing = self._config.get('exit_pricing', {})
        order_book = ticker = None
        if not entry_rate and entry_pricing.get('use_order_book', False):
            order_book_top = max(entry_pricing.get('order_book_top', 1),
                                 exit_pricing.get('order_book_top', 1))
            order_book = self.fetch_l2_order_book(pair, order_book_top)
            entry_rate = self.get_rate(pair, refresh, 'entry', is_short, order_book=order_book)
        elif not entry_rate:
            ticker = self.fetch_ticker(pair)
            entry_rate = self.get_rate(pair, refresh, 'entry', is_short, ticker=ticker)
        if not exit_rate:
            exit_rate = self.get_rate(pair, refresh, 'exit',
                                      is_short, order_book=order_book, ticker=ticker)
        return entry_rate, exit_rate

    # Fee handling

    @retrier
    def get_trades_for_order(self, order_id: str, pair: str, since: datetime,
                             params: Optional[Dict] = None) -> List:
        """
        Fetch Orders using the "fetch_my_trades" endpoint and filter them by order-id.
        The "since" argument passed in is coming from the database and is in UTC,
        as timezone-native datetime object.
        From the python documentation:
            > Naive datetime instances are assumed to represent local time
        Therefore, calling "since.timestamp()" will get the UTC timestamp, after applying the
        transformation from local timezone to UTC.
        This works for timezones UTC+ since then the result will contain trades from a few hours
        instead of from the last 5 seconds, however fails for UTC- timezones,
        since we're then asking for trades with a "since" argument in the future.

        :param order_id order_id: Order-id as given when creating the order
        :param pair: Pair the order is for
        :param since: datetime object of the order creation time. Assumes object is in UTC.
        """
        if self._config['dry_run']:
            return []
        if not self.exchange_has('fetchMyTrades'):
            return []
        try:
            # Allow 5s offset to catch slight time offsets (discovered in #1185)
            # since needs to be int in milliseconds
            _params = params if params else {}
            my_trades = self._api.fetch_my_trades(
                pair, int((since.replace(tzinfo=timezone.utc).timestamp() - 5) * 1000),
                params=_params)
            matched_trades = [trade for trade in my_trades if trade['order'] == order_id]

            self._log_exchange_response('get_trades_for_order', matched_trades)

            matched_trades = self._trades_contracts_to_amount(matched_trades)

            return matched_trades
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get trades due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def get_order_id_conditional(self, order: Dict[str, Any]) -> str:
        return order['id']

    @retrier
    def get_fee(self, symbol: str, type: str = '', side: str = '', amount: float = 1,
                price: float = 1, taker_or_maker: MakerTaker = 'maker') -> float:
        try:
            if self._config['dry_run'] and self._config.get('fee', None) is not None:
                return self._config['fee']
            # validate that markets are loaded before trying to get fee
            if self._api.markets is None or len(self._api.markets) == 0:
                self._api.load_markets()

            return self._api.calculate_fee(symbol=symbol, type=type, side=side, amount=amount,
                                           price=price, takerOrMaker=taker_or_maker)['rate']
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get fee info due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @staticmethod
    def order_has_fee(order: Dict) -> bool:
        """
        Verifies if the passed in order dict has the needed keys to extract fees,
        and that these keys (currency, cost) are not empty.
        :param order: Order or trade (one trade) dict
        :return: True if the fee substructure contains currency and cost, false otherwise
        """
        if not isinstance(order, dict):
            return False
        return ('fee' in order and order['fee'] is not None
                and (order['fee'].keys() >= {'currency', 'cost'})
                and order['fee']['currency'] is not None
                and order['fee']['cost'] is not None
                )

    def calculate_fee_rate(
            self, fee: Dict, symbol: str, cost: float, amount: float) -> Optional[float]:
        """
        Calculate fee rate if it's not given by the exchange.
        :param fee: ccxt Fee dict - must contain cost / currency / rate
        :param symbol: Symbol of the order
        :param cost: Total cost of the order
        :param amount: Amount of the order
        """
        if fee.get('rate') is not None:
            return fee.get('rate')
        fee_curr = fee.get('currency')
        if fee_curr is None:
            return None
        fee_cost = float(fee['cost'])
        if self._ft_has['fee_cost_in_contracts']:
            # Convert cost via "contracts" conversion
            fee_cost = self._contracts_to_amount(symbol, fee['cost'])

        # Calculate fee based on order details
        if fee_curr == self.get_pair_base_currency(symbol):
            # Base currency - divide by amount
            return round(fee_cost / amount, 8)
        elif fee_curr == self.get_pair_quote_currency(symbol):
            # Quote currency - divide by cost
            return round(fee_cost / cost, 8) if cost else None
        else:
            # If Fee currency is a different currency
            if not cost:
                # If cost is None or 0.0 -> falsy, return None
                return None
            try:
                comb = self.get_valid_pair_combination(fee_curr, self._config['stake_currency'])
                tick = self.fetch_ticker(comb)

                fee_to_quote_rate = safe_value_fallback2(tick, tick, 'last', 'ask')
            except ExchangeError:
                fee_to_quote_rate = self._config['exchange'].get('unknown_fee_rate', None)
                if not fee_to_quote_rate:
                    return None
            return round((fee_cost * fee_to_quote_rate) / cost, 8)

    def extract_cost_curr_rate(self, fee: Dict, symbol: str, cost: float,
                               amount: float) -> Tuple[float, str, Optional[float]]:
        """
        Extract tuple of cost, currency, rate.
        Requires order_has_fee to run first!
        :param fee: ccxt Fee dict - must contain cost / currency / rate
        :param symbol: Symbol of the order
        :param cost: Total cost of the order
        :param amount: Amount of the order
        :return: Tuple with cost, currency, rate of the given fee dict
        """
        return (float(fee['cost']),
                fee['currency'],
                self.calculate_fee_rate(
                    fee,
                    symbol,
                    cost,
                    amount
                    )
                )

    # Historic data

    def get_historic_ohlcv(self, pair: str, timeframe: str,
                           since_ms: int, candle_type: CandleType,
                           is_new_pair: bool = False,
                           until_ms: int = None) -> List:
        """
        Get candle history using asyncio and returns the list of candles.
        Handles all async work for this.
        Async over one pair, assuming we get `self.ohlcv_candle_limit()` candles per call.
        :param pair: Pair to download
        :param timeframe: Timeframe to get data for
        :param since_ms: Timestamp in milliseconds to get history from
        :param until_ms: Timestamp in milliseconds to get history up to
        :param candle_type: '', mark, index, premiumIndex, or funding_rate
        :return: List with candle (OHLCV) data
        """
        pair, _, _, data = self.loop.run_until_complete(
            self._async_get_historic_ohlcv(pair=pair, timeframe=timeframe,
                                           since_ms=since_ms, until_ms=until_ms,
                                           is_new_pair=is_new_pair, candle_type=candle_type))
        logger.info(f"Downloaded data for {pair} with length {len(data)}.")
        return data

    def get_historic_ohlcv_as_df(self, pair: str, timeframe: str,
                                 since_ms: int, candle_type: CandleType) -> DataFrame:
        """
        Minimal wrapper around get_historic_ohlcv - converting the result into a dataframe
        :param pair: Pair to download
        :param timeframe: Timeframe to get data for
        :param since_ms: Timestamp in milliseconds to get history from
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        :return: OHLCV DataFrame
        """
        ticks = self.get_historic_ohlcv(pair, timeframe, since_ms=since_ms, candle_type=candle_type)
        return ohlcv_to_dataframe(ticks, timeframe, pair=pair, fill_missing=True,
                                  drop_incomplete=self._ohlcv_partial_candle)

    async def _async_get_historic_ohlcv(self, pair: str, timeframe: str,
                                        since_ms: int, candle_type: CandleType,
                                        is_new_pair: bool = False, raise_: bool = False,
                                        until_ms: Optional[int] = None
                                        ) -> Tuple[str, str, str, List]:
        """
        Download historic ohlcv
        :param is_new_pair: used by binance subclass to allow "fast" new pair downloading
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        """

        one_call = timeframe_to_msecs(timeframe) * self.ohlcv_candle_limit(
            timeframe, candle_type, since_ms)
        logger.debug(
            "one_call: %s msecs (%s)",
            one_call,
            arrow.utcnow().shift(seconds=one_call // 1000).humanize(only_distance=True)
        )
        input_coroutines = [self._async_get_candle_history(
            pair, timeframe, candle_type, since) for since in
            range(since_ms, until_ms or (arrow.utcnow().int_timestamp * 1000), one_call)]

        data: List = []
        # Chunk requests into batches of 100 to avoid overwelming ccxt Throttling
        for input_coro in chunks(input_coroutines, 100):

            results = await asyncio.gather(*input_coro, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    logger.warning(f"Async code raised an exception: {repr(res)}")
                    if raise_:
                        raise
                    continue
                else:
                    # Deconstruct tuple if it's not an exception
                    p, _, c, new_data = res
                    if p == pair and c == candle_type:
                        data.extend(new_data)
        # Sort data again after extending the result - above calls return in "async order"
        data = sorted(data, key=lambda x: x[0])
        return pair, timeframe, candle_type, data

    def _build_coroutine(self, pair: str, timeframe: str, candle_type: CandleType,
                         since_ms: Optional[int]) -> Coroutine:

        if (not since_ms
                and (self._ft_has["ohlcv_require_since"] or self.required_candle_call_count > 1)):
            # Multiple calls for one pair - to get more history
            one_call = timeframe_to_msecs(timeframe) * self.ohlcv_candle_limit(
                timeframe, candle_type, since_ms)
            move_to = one_call * self.required_candle_call_count
            now = timeframe_to_next_date(timeframe)
            since_ms = int((now - timedelta(seconds=move_to // 1000)).timestamp() * 1000)

        if since_ms:
            return self._async_get_historic_ohlcv(
                pair, timeframe, since_ms=since_ms, raise_=True, candle_type=candle_type)
        else:
            # One call ... "regular" refresh
            return self._async_get_candle_history(
                pair, timeframe, since_ms=since_ms, candle_type=candle_type)

    def refresh_latest_ohlcv(self, pair_list: ListPairsWithTimeframes, *,
                             since_ms: Optional[int] = None, cache: bool = True,
                             drop_incomplete: Optional[bool] = None
                             ) -> Dict[PairWithTimeframe, DataFrame]:
        """
        Refresh in-memory OHLCV asynchronously and set `_klines` with the result
        Loops asynchronously over pair_list and downloads all pairs async (semi-parallel).
        Only used in the dataprovider.refresh() method.
        :param pair_list: List of 2 element tuples containing pair, interval to refresh
        :param since_ms: time since when to download, in milliseconds
        :param cache: Assign result to _klines. Usefull for one-off downloads like for pairlists
        :param drop_incomplete: Control candle dropping.
            Specifying None defaults to _ohlcv_partial_candle
        :return: Dict of [{(pair, timeframe): Dataframe}]
        """
        logger.debug("Refreshing candle (OHLCV) data for %d pairs", len(pair_list))
        drop_incomplete = self._ohlcv_partial_candle if drop_incomplete is None else drop_incomplete
        input_coroutines = []
        cached_pairs = []
        # Gather coroutines to run
        for pair, timeframe, candle_type in set(pair_list):
            if (timeframe not in self.timeframes
                    and candle_type in (CandleType.SPOT, CandleType.FUTURES)):
                logger.warning(
                    f"Cannot download ({pair}, {timeframe}) combination as this timeframe is "
                    f"not available on {self.name}. Available timeframes are "
                    f"{', '.join(self.timeframes)}.")
                continue
            if ((pair, timeframe, candle_type) not in self._klines or not cache
                    or self._now_is_time_to_refresh(pair, timeframe, candle_type)):
                input_coroutines.append(self._build_coroutine(
                    pair, timeframe, candle_type=candle_type, since_ms=since_ms))

            else:
                logger.debug(
                    f"Using cached candle (OHLCV) data for {pair}, {timeframe}, {candle_type} ..."
                )
                cached_pairs.append((pair, timeframe, candle_type))

        results_df = {}
        # Chunk requests into batches of 100 to avoid overwelming ccxt Throttling
        for input_coro in chunks(input_coroutines, 100):
            async def gather_stuff():
                return await asyncio.gather(*input_coro, return_exceptions=True)

            with self._loop_lock:
                results = self.loop.run_until_complete(gather_stuff())

            for res in results:
                if isinstance(res, Exception):
                    logger.warning(f"Async code raised an exception: {repr(res)}")
                    continue
                # Deconstruct tuple (has 4 elements)
                pair, timeframe, c_type, ticks = res
                # keeping last candle time as last refreshed time of the pair
                if ticks:
                    self._pairs_last_refresh_time[(pair, timeframe, c_type)] = ticks[-1][0] // 1000
                # keeping parsed dataframe in cache
                ohlcv_df = ohlcv_to_dataframe(
                    ticks, timeframe, pair=pair, fill_missing=True,
                    drop_incomplete=drop_incomplete)
                results_df[(pair, timeframe, c_type)] = ohlcv_df
                if cache:
                    self._klines[(pair, timeframe, c_type)] = ohlcv_df
        # Return cached klines
        for pair, timeframe, c_type in cached_pairs:
            results_df[(pair, timeframe, c_type)] = self.klines(
                (pair, timeframe, c_type),
                copy=False
            )

        return results_df

    def _now_is_time_to_refresh(self, pair: str, timeframe: str, candle_type: CandleType) -> bool:
        # Timeframe in seconds
        interval_in_sec = timeframe_to_seconds(timeframe)

        return not (
            (self._pairs_last_refresh_time.get(
                (pair, timeframe, candle_type),
                0
            ) + interval_in_sec) >= arrow.utcnow().int_timestamp
        )

    @retrier_async
    async def _async_get_candle_history(
        self,
        pair: str,
        timeframe: str,
        candle_type: CandleType,
        since_ms: Optional[int] = None,
    ) -> Tuple[str, str, str, List]:
        """
        Asynchronously get candle history data using fetch_ohlcv
        :param candle_type: '', mark, index, premiumIndex, or funding_rate
        returns tuple: (pair, timeframe, ohlcv_list)
        """
        try:
            # Fetch OHLCV asynchronously
            s = '(' + arrow.get(since_ms // 1000).isoformat() + ') ' if since_ms is not None else ''
            logger.debug(
                "Fetching pair %s, interval %s, since %s %s...",
                pair, timeframe, since_ms, s
            )
            params = deepcopy(self._ft_has.get('ohlcv_params', {}))
            candle_limit = self.ohlcv_candle_limit(
                timeframe, candle_type=candle_type, since_ms=since_ms)

            if candle_type != CandleType.SPOT:
                params.update({'price': candle_type})
            if candle_type != CandleType.FUNDING_RATE:
                data = await self._api_async.fetch_ohlcv(
                    pair, timeframe=timeframe, since=since_ms,
                    limit=candle_limit, params=params)
            else:
                # Funding rate
                data = await self._api_async.fetch_funding_rate_history(
                    pair, since=since_ms,
                    limit=candle_limit)
                # Convert funding rate to candle pattern
                data = [[x['timestamp'], x['fundingRate'], 0, 0, 0, 0] for x in data]
            # Some exchanges sort OHLCV in ASC order and others in DESC.
            # Ex: Bittrex returns the list of OHLCV in ASC order (oldest first, newest last)
            # while GDAX returns the list of OHLCV in DESC order (newest first, oldest last)
            # Only sort if necessary to save computing time
            try:
                if data and data[0][0] > data[-1][0]:
                    data = sorted(data, key=lambda x: x[0])
            except IndexError:
                logger.exception("Error loading %s. Result was %s.", pair, data)
                return pair, timeframe, candle_type, []
            logger.debug("Done fetching pair %s, interval %s ...", pair, timeframe)
            return pair, timeframe, candle_type, data

        except ccxt.NotSupported as e:
            raise OperationalException(
                f'Exchange {self._api.name} does not support fetching historical '
                f'candle (OHLCV) data. Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not fetch historical candle (OHLCV) data '
                                 f'for pair {pair} due to {e.__class__.__name__}. '
                                 f'Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(f'Could not fetch historical candle (OHLCV) data '
                                       f'for pair {pair}. Message: {e}') from e

    # Fetch historic trades

    @retrier_async
    async def _async_fetch_trades(self, pair: str,
                                  since: Optional[int] = None,
                                  params: Optional[dict] = None) -> List[List]:
        """
        Asyncronously gets trade history using fetch_trades.
        Handles exchange errors, does one call to the exchange.
        :param pair: Pair to fetch trade data for
        :param since: Since as integer timestamp in milliseconds
        returns: List of dicts containing trades
        """
        try:
            # fetch trades asynchronously
            if params:
                logger.debug("Fetching trades for pair %s, params: %s ", pair, params)
                trades = await self._api_async.fetch_trades(pair, params=params, limit=1000)
            else:
                logger.debug(
                    "Fetching trades for pair %s, since %s %s...",
                    pair, since,
                    '(' + arrow.get(since // 1000).isoformat() + ') ' if since is not None else ''
                )
                trades = await self._api_async.fetch_trades(pair, since=since, limit=1000)
            trades = self._trades_contracts_to_amount(trades)
            return trades_dict_to_list(trades)
        except ccxt.NotSupported as e:
            raise OperationalException(
                f'Exchange {self._api.name} does not support fetching historical trade data.'
                f'Message: {e}') from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(f'Could not load trade history due to {e.__class__.__name__}. '
                                 f'Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(f'Could not fetch trade data. Msg: {e}') from e

    async def _async_get_trade_history_id(self, pair: str,
                                          until: int,
                                          since: Optional[int] = None,
                                          from_id: Optional[str] = None) -> Tuple[str, List[List]]:
        """
        Asyncronously gets trade history using fetch_trades
        use this when exchange uses id-based iteration (check `self._trades_pagination`)
        :param pair: Pair to fetch trade data for
        :param since: Since as integer timestamp in milliseconds
        :param until: Until as integer timestamp in milliseconds
        :param from_id: Download data starting with ID (if id is known). Ignores "since" if set.
        returns tuple: (pair, trades-list)
        """

        trades: List[List] = []

        if not from_id:
            # Fetch first elements using timebased method to get an ID to paginate on
            # Depending on the Exchange, this can introduce a drift at the start of the interval
            # of up to an hour.
            # e.g. Binance returns the "last 1000" candles within a 1h time interval
            # - so we will miss the first trades.
            t = await self._async_fetch_trades(pair, since=since)
            # DEFAULT_TRADES_COLUMNS: 0 -> timestamp
            # DEFAULT_TRADES_COLUMNS: 1 -> id
            from_id = t[-1][1]
            trades.extend(t[:-1])
        while True:
            t = await self._async_fetch_trades(pair,
                                               params={self._trades_pagination_arg: from_id})
            if t:
                # Skip last id since its the key for the next call
                trades.extend(t[:-1])
                if from_id == t[-1][1] or t[-1][0] > until:
                    logger.debug(f"Stopping because from_id did not change. "
                                 f"Reached {t[-1][0]} > {until}")
                    # Reached the end of the defined-download period - add last trade as well.
                    trades.extend(t[-1:])
                    break

                from_id = t[-1][1]
            else:
                break

        return (pair, trades)

    async def _async_get_trade_history_time(self, pair: str, until: int,
                                            since: Optional[int] = None) -> Tuple[str, List[List]]:
        """
        Asyncronously gets trade history using fetch_trades,
        when the exchange uses time-based iteration (check `self._trades_pagination`)
        :param pair: Pair to fetch trade data for
        :param since: Since as integer timestamp in milliseconds
        :param until: Until as integer timestamp in milliseconds
        returns tuple: (pair, trades-list)
        """

        trades: List[List] = []
        # DEFAULT_TRADES_COLUMNS: 0 -> timestamp
        # DEFAULT_TRADES_COLUMNS: 1 -> id
        while True:
            t = await self._async_fetch_trades(pair, since=since)
            if t:
                since = t[-1][0]
                trades.extend(t)
                # Reached the end of the defined-download period
                if until and t[-1][0] > until:
                    logger.debug(
                        f"Stopping because until was reached. {t[-1][0]} > {until}")
                    break
            else:
                break

        return (pair, trades)

    async def _async_get_trade_history(self, pair: str,
                                       since: Optional[int] = None,
                                       until: Optional[int] = None,
                                       from_id: Optional[str] = None) -> Tuple[str, List[List]]:
        """
        Async wrapper handling downloading trades using either time or id based methods.
        """

        logger.debug(f"_async_get_trade_history(), pair: {pair}, "
                     f"since: {since}, until: {until}, from_id: {from_id}")

        if until is None:
            until = ccxt.Exchange.milliseconds()
            logger.debug(f"Exchange milliseconds: {until}")

        if self._trades_pagination == 'time':
            return await self._async_get_trade_history_time(
                pair=pair, since=since, until=until)
        elif self._trades_pagination == 'id':
            return await self._async_get_trade_history_id(
                pair=pair, since=since, until=until, from_id=from_id
            )
        else:
            raise OperationalException(f"Exchange {self.name} does use neither time, "
                                       f"nor id based pagination")

    def get_historic_trades(self, pair: str,
                            since: Optional[int] = None,
                            until: Optional[int] = None,
                            from_id: Optional[str] = None) -> Tuple[str, List]:
        """
        Get trade history data using asyncio.
        Handles all async work and returns the list of candles.
        Async over one pair, assuming we get `self.ohlcv_candle_limit()` candles per call.
        :param pair: Pair to download
        :param since: Timestamp in milliseconds to get history from
        :param until: Timestamp in milliseconds. Defaults to current timestamp if not defined.
        :param from_id: Download data starting with ID (if id is known)
        :returns List of trade data
        """
        if not self.exchange_has("fetchTrades"):
            raise OperationalException("This exchange does not support downloading Trades.")

        with self._loop_lock:
            return self.loop.run_until_complete(
                self._async_get_trade_history(pair=pair, since=since,
                                              until=until, from_id=from_id))

    @retrier
    def _get_funding_fees_from_exchange(self, pair: str, since: Union[datetime, int]) -> float:
        """
        Returns the sum of all funding fees that were exchanged for a pair within a timeframe
        Dry-run handling happens as part of _calculate_funding_fees.
        :param pair: (e.g. ADA/USDT)
        :param since: The earliest time of consideration for calculating funding fees,
            in unix time or as a datetime
        """
        if not self.exchange_has("fetchFundingHistory"):
            raise OperationalException(
                f"fetch_funding_history() is not available using {self.name}"
            )

        if type(since) is datetime:
            since = int(since.timestamp()) * 1000   # * 1000 for ms

        try:
            funding_history = self._api.fetch_funding_history(
                symbol=pair,
                since=since
            )
            return sum(fee['amount'] for fee in funding_history)
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get funding fees due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def get_leverage_tiers(self) -> Dict[str, List[Dict]]:
        try:
            return self._api.fetch_leverage_tiers()
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not load leverage tiers due to {e.__class__.__name__}. Message: {e}'
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier_async
    async def get_market_leverage_tiers(self, symbol: str) -> Tuple[str, List[Dict]]:
        """ Leverage tiers per symbol """
        try:
            tier = await self._api_async.fetch_market_leverage_tiers(symbol)
            return symbol, tier
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not load leverage tiers for {symbol}'
                f' due to {e.__class__.__name__}. Message: {e}'
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def load_leverage_tiers(self) -> Dict[str, List[Dict]]:
        if self.trading_mode == TradingMode.FUTURES:
            if self.exchange_has('fetchLeverageTiers'):
                # Fetch all leverage tiers at once
                return self.get_leverage_tiers()
            elif self.exchange_has('fetchMarketLeverageTiers'):
                # Must fetch the leverage tiers for each market separately
                # * This is slow(~45s) on Okx, makes ~90 api calls to load all linear swap markets
                markets = self.markets
                symbols = []

                for symbol, market in markets.items():
                    if (self.market_is_future(market)
                            and market['quote'] == self._config['stake_currency']):
                        symbols.append(symbol)

                tiers: Dict[str, List[Dict]] = {}

                tiers_cached = self.load_cached_leverage_tiers(self._config['stake_currency'])
                if tiers_cached:
                    tiers = tiers_cached

                coros = [
                    self.get_market_leverage_tiers(symbol)
                    for symbol in sorted(symbols) if symbol not in tiers]

                # Be verbose here, as this delays startup by ~1 minute.
                if coros:
                    logger.info(
                        f"Initializing leverage_tiers for {len(symbols)} markets. "
                        "This will take about a minute.")
                else:
                    logger.info("Using cached leverage_tiers.")

                async def gather_results():
                    return await asyncio.gather(*input_coro, return_exceptions=True)

                for input_coro in chunks(coros, 100):

                    with self._loop_lock:
                        results = self.loop.run_until_complete(gather_results())

                    for symbol, res in results:
                        tiers[symbol] = res
                if len(coros) > 0:
                    self.cache_leverage_tiers(tiers, self._config['stake_currency'])
                logger.info(f"Done initializing {len(symbols)} markets.")

                return tiers
            else:
                return {}
        else:
            return {}

    def cache_leverage_tiers(self, tiers: Dict[str, List[Dict]], stake_currency: str) -> None:

        filename = self._config['datadir'] / "futures" / f"leverage_tiers_{stake_currency}.json"
        if not filename.parent.is_dir():
            filename.parent.mkdir(parents=True)
        data = {
            "updated": datetime.now(timezone.utc),
            "data": tiers,
        }
        file_dump_json(filename, data)

    def load_cached_leverage_tiers(self, stake_currency: str) -> Optional[Dict[str, List[Dict]]]:
        filename = self._config['datadir'] / "futures" / f"leverage_tiers_{stake_currency}.json"
        if filename.is_file():
            tiers = file_load_json(filename)
            updated = tiers.get('updated')
            if updated:
                updated_dt = parser.parse(updated)
                if updated_dt < datetime.now(timezone.utc) - timedelta(weeks=4):
                    logger.info("Cached leverage tiers are outdated. Will update.")
                    return None
            return tiers['data']
        return None

    def fill_leverage_tiers(self) -> None:
        """
        Assigns property _leverage_tiers to a dictionary of information about the leverage
        allowed on each pair
        """
        leverage_tiers = self.load_leverage_tiers()
        for pair, tiers in leverage_tiers.items():
            pair_tiers = []
            for tier in tiers:
                pair_tiers.append(self.parse_leverage_tier(tier))
            self._leverage_tiers[pair] = pair_tiers

    def parse_leverage_tier(self, tier) -> Dict:
        info = tier.get('info', {})
        return {
            'minNotional': tier['minNotional'],
            'maxNotional': tier['maxNotional'],
            'maintenanceMarginRate': tier['maintenanceMarginRate'],
            'maxLeverage': tier['maxLeverage'],
            'maintAmt': float(info['cum']) if 'cum' in info else None,
        }

    def get_max_leverage(self, pair: str, stake_amount: Optional[float]) -> float:
        """
        Returns the maximum leverage that a pair can be traded at
        :param pair: The base/quote currency pair being traded
        :stake_amount: The total value of the traders margin_mode in quote currency
        """

        if self.trading_mode == TradingMode.SPOT:
            return 1.0

        if self.trading_mode == TradingMode.FUTURES:

            # Checks and edge cases
            if stake_amount is None:
                raise OperationalException(
                    f'{self.name}.get_max_leverage requires argument stake_amount'
                )

            if pair not in self._leverage_tiers:
                # Maybe raise exception because it can't be traded on futures?
                return 1.0

            pair_tiers = self._leverage_tiers[pair]

            if stake_amount == 0:
                return self._leverage_tiers[pair][0]['maxLeverage']  # Max lev for lowest amount

            for tier_index in range(len(pair_tiers)):

                tier = pair_tiers[tier_index]
                lev = tier['maxLeverage']

                if tier_index < len(pair_tiers) - 1:
                    next_tier = pair_tiers[tier_index + 1]
                    next_floor = next_tier['minNotional'] / next_tier['maxLeverage']
                    if next_floor > stake_amount:  # Next tier min too high for stake amount
                        return min((tier['maxNotional'] / stake_amount), lev)
                        #
                        # With the two leverage tiers below,
                        # - a stake amount of 150 would mean a max leverage of (10000 / 150) = 66.66
                        # - stakes below 133.33 = max_lev of 75
                        # - stakes between 133.33-200 = max_lev of 10000/stake = 50.01-74.99
                        # - stakes from 200 + 1000 = max_lev of 50
                        #
                        # {
                        #     "min": 0,      # stake = 0.0
                        #     "max": 10000,  # max_stake@75 = 10000/75 = 133.33333333333334
                        #     "lev": 75,
                        # },
                        # {
                        #     "min": 10000,  # stake = 200.0
                        #     "max": 50000,  # max_stake@50 = 50000/50 = 1000.0
                        #     "lev": 50,
                        # }
                        #

                else:  # if on the last tier
                    if stake_amount > tier['maxNotional']:
                        # If stake is > than max tradeable amount
                        raise InvalidOrderException(f'Amount {stake_amount} too high for {pair}')
                    else:
                        return tier['maxLeverage']

            raise OperationalException(
                'Looped through all tiers without finding a max leverage. Should never be reached'
            )

        elif self.trading_mode == TradingMode.MARGIN:  # Search markets.limits for max lev
            market = self.markets[pair]
            if market['limits']['leverage']['max'] is not None:
                return market['limits']['leverage']['max']
            else:
                return 1.0  # Default if max leverage cannot be found
        else:
            return 1.0

    @retrier
    def _set_leverage(
        self,
        leverage: float,
        pair: Optional[str] = None,
        trading_mode: Optional[TradingMode] = None
    ):
        """
        Set's the leverage before making a trade, in order to not
        have the same leverage on every trade
        """
        if self._config['dry_run'] or not self.exchange_has("setLeverage"):
            # Some exchanges only support one margin_mode type
            return

        try:
            res = self._api.set_leverage(symbol=pair, leverage=leverage)
            self._log_exchange_response('set_leverage', res)
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not set leverage due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def get_interest_rate(self) -> float:
        """
        Retrieve interest rate - necessary for Margin trading.
        Should not call the exchange directly when used from backtesting.
        """
        return 0.0

    def funding_fee_cutoff(self, open_date: datetime):
        """
        :param open_date: The open date for a trade
        :return: The cutoff open time for when a funding fee is charged
        """
        return open_date.minute > 0 or open_date.second > 0

    @retrier
    def set_margin_mode(self, pair: str, margin_mode: MarginMode, params: dict = {}):
        """
        Set's the margin mode on the exchange to cross or isolated for a specific pair
        :param pair: base/quote currency pair (e.g. "ADA/USDT")
        """
        if self._config['dry_run'] or not self.exchange_has("setMarginMode"):
            # Some exchanges only support one margin_mode type
            return

        try:
            res = self._api.set_margin_mode(margin_mode.value, pair, params)
            self._log_exchange_response('set_margin_mode', res)
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not set margin mode due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _fetch_and_calculate_funding_fees(
        self,
        pair: str,
        amount: float,
        is_short: bool,
        open_date: datetime,
        close_date: Optional[datetime] = None
    ) -> float:
        """
        Fetches and calculates the sum of all funding fees that occurred for a pair
        during a futures trade.
        Only used during dry-run or if the exchange does not provide a funding_rates endpoint.
        :param pair: The quote/base pair of the trade
        :param amount: The quantity of the trade
        :param is_short: trade direction
        :param open_date: The date and time that the trade started
        :param close_date: The date and time that the trade ended
        """

        if self.funding_fee_cutoff(open_date):
            open_date += timedelta(hours=1)
        timeframe = self._ft_has['mark_ohlcv_timeframe']
        timeframe_ff = self._ft_has.get('funding_fee_timeframe',
                                        self._ft_has['mark_ohlcv_timeframe'])

        if not close_date:
            close_date = datetime.now(timezone.utc)
        open_timestamp = int(timeframe_to_prev_date(timeframe, open_date).timestamp()) * 1000
        # close_timestamp = int(close_date.timestamp()) * 1000

        mark_comb: PairWithTimeframe = (
            pair, timeframe, CandleType.from_string(self._ft_has["mark_ohlcv_price"]))

        funding_comb: PairWithTimeframe = (pair, timeframe_ff, CandleType.FUNDING_RATE)
        candle_histories = self.refresh_latest_ohlcv(
            [mark_comb, funding_comb],
            since_ms=open_timestamp,
            cache=False,
            drop_incomplete=False,
        )
        try:
            # we can't assume we always get histories - for example during exchange downtimes
            funding_rates = candle_histories[funding_comb]
            mark_rates = candle_histories[mark_comb]
        except KeyError:
            raise ExchangeError("Could not find funding rates.") from None

        funding_mark_rates = self.combine_funding_and_mark(
            funding_rates=funding_rates, mark_rates=mark_rates)

        return self.calculate_funding_fees(
            funding_mark_rates,
            amount=amount,
            is_short=is_short,
            open_date=open_date,
            close_date=close_date
        )

    @staticmethod
    def combine_funding_and_mark(funding_rates: DataFrame, mark_rates: DataFrame,
                                 futures_funding_rate: Optional[int] = None) -> DataFrame:
        """
        Combine funding-rates and mark-rates dataframes
        :param funding_rates: Dataframe containing Funding rates (Type FUNDING_RATE)
        :param mark_rates: Dataframe containing Mark rates (Type mark_ohlcv_price)
        :param futures_funding_rate: Fake funding rate to use if funding_rates are not available
        """
        if futures_funding_rate is None:
            return mark_rates.merge(
                funding_rates, on='date', how="inner", suffixes=["_mark", "_fund"])
        else:
            if len(funding_rates) == 0:
                # No funding rate candles - full fillup with fallback variable
                mark_rates['open_fund'] = futures_funding_rate
                return mark_rates.rename(
                        columns={'open': 'open_mark',
                                 'close': 'close_mark',
                                 'high': 'high_mark',
                                 'low': 'low_mark',
                                 'volume': 'volume_mark'})

            else:
                # Fill up missing funding_rate candles with fallback value
                combined = mark_rates.merge(
                    funding_rates, on='date', how="outer", suffixes=["_mark", "_fund"]
                    )
                combined['open_fund'] = combined['open_fund'].fillna(futures_funding_rate)
                return combined

    def calculate_funding_fees(
        self,
        df: DataFrame,
        amount: float,
        is_short: bool,
        open_date: datetime,
        close_date: Optional[datetime] = None,
        time_in_ratio: Optional[float] = None
    ) -> float:
        """
        calculates the sum of all funding fees that occurred for a pair during a futures trade
        :param df: Dataframe containing combined funding and mark rates
                   as `open_fund` and `open_mark`.
        :param amount: The quantity of the trade
        :param is_short: trade direction
        :param open_date: The date and time that the trade started
        :param close_date: The date and time that the trade ended
        :param time_in_ratio: Not used by most exchange classes
        """
        fees: float = 0

        if not df.empty:
            df = df[(df['date'] >= open_date) & (df['date'] <= close_date)]
            fees = sum(df['open_fund'] * df['open_mark'] * amount)

        # Negate fees for longs as funding_fees expects it this way based on live endpoints.
        return fees if is_short else -fees

    def get_funding_fees(
            self, pair: str, amount: float, is_short: bool, open_date: datetime) -> float:
        """
        Fetch funding fees, either from the exchange (live) or calculates them
        based on funding rate/mark price history
        :param pair: The quote/base pair of the trade
        :param is_short: trade direction
        :param amount: Trade amount
        :param open_date: Open date of the trade
        :return: funding fee since open_date
        :raies: ExchangeError if something goes wrong.
        """
        if self.trading_mode == TradingMode.FUTURES:
            if self._config['dry_run']:
                funding_fees = self._fetch_and_calculate_funding_fees(
                    pair, amount, is_short, open_date)
            else:
                funding_fees = self._get_funding_fees_from_exchange(pair, open_date)
            return funding_fees
        else:
            return 0.0

    def get_liquidation_price(
        self,
        pair: str,
        # Dry-run
        open_rate: float,   # Entry price of position
        is_short: bool,
        amount: float,  # Absolute value of position size
        stake_amount: float,
        wallet_balance: float,
        mm_ex_1: float = 0.0,  # (Binance) Cross only
        upnl_ex_1: float = 0.0,  # (Binance) Cross only
    ) -> Optional[float]:
        """
        Set's the margin mode on the exchange to cross or isolated for a specific pair
        """
        if self.trading_mode == TradingMode.SPOT:
            return None
        elif (self.trading_mode != TradingMode.FUTURES):
            raise OperationalException(
                f"{self.name} does not support {self.margin_mode} {self.trading_mode}")

        isolated_liq = None
        if self._config['dry_run'] or not self.exchange_has("fetchPositions"):

            isolated_liq = self.dry_run_liquidation_price(
                pair=pair,
                open_rate=open_rate,
                is_short=is_short,
                amount=amount,
                stake_amount=stake_amount,
                wallet_balance=wallet_balance,
                mm_ex_1=mm_ex_1,
                upnl_ex_1=upnl_ex_1
            )
        else:
            positions = self.fetch_positions(pair)
            if len(positions) > 0:
                pos = positions[0]
                isolated_liq = pos['liquidationPrice']

        if isolated_liq:
            buffer_amount = abs(open_rate - isolated_liq) * self.liquidation_buffer
            isolated_liq = (
                isolated_liq - buffer_amount
                if is_short else
                isolated_liq + buffer_amount
            )
            return isolated_liq
        else:
            return None

    def dry_run_liquidation_price(
        self,
        pair: str,
        open_rate: float,   # Entry price of position
        is_short: bool,
        amount: float,
        stake_amount: float,
        wallet_balance: float,  # Or margin balance
        mm_ex_1: float = 0.0,  # (Binance) Cross only
        upnl_ex_1: float = 0.0,  # (Binance) Cross only
    ) -> Optional[float]:
        """
        Important: Must be fetching data from cached values as this is used by backtesting!
        PERPETUAL:
         gateio: https://www.gate.io/help/futures/perpetual/22160/calculation-of-liquidation-price
         okex: https://www.okex.com/support/hc/en-us/articles/
            360053909592-VI-Introduction-to-the-isolated-mode-of-Single-Multi-currency-Portfolio-margin

        :param exchange_name:
        :param open_rate: Entry price of position
        :param is_short: True if the trade is a short, false otherwise
        :param amount: Absolute value of position size incl. leverage (in base currency)
        :param stake_amount: Stake amount - Collateral in settle currency.
        :param trading_mode: SPOT, MARGIN, FUTURES, etc.
        :param margin_mode: Either ISOLATED or CROSS
        :param wallet_balance: Amount of margin_mode in the wallet being used to trade
            Cross-Margin Mode: crossWalletBalance
            Isolated-Margin Mode: isolatedWalletBalance

        # * Not required by Gateio or OKX
        :param mm_ex_1:
        :param upnl_ex_1:
        """

        market = self.markets[pair]
        taker_fee_rate = market['taker']
        mm_ratio, _ = self.get_maintenance_ratio_and_amt(pair, stake_amount)

        if self.trading_mode == TradingMode.FUTURES and self.margin_mode == MarginMode.ISOLATED:

            if market['inverse']:
                raise OperationalException(
                    "Freqtrade does not yet support inverse contracts")

            value = wallet_balance / amount

            mm_ratio_taker = (mm_ratio + taker_fee_rate)
            if is_short:
                return (open_rate + value) / (1 + mm_ratio_taker)
            else:
                return (open_rate - value) / (1 - mm_ratio_taker)
        else:
            raise OperationalException(
                "Freqtrade only supports isolated futures for leverage trading")

    def get_maintenance_ratio_and_amt(
        self,
        pair: str,
        nominal_value: float = 0.0,
    ) -> Tuple[float, Optional[float]]:
        """
        Important: Must be fetching data from cached values as this is used by backtesting!
        :param pair: Market symbol
        :param nominal_value: The total trade amount in quote currency including leverage
        maintenance amount only on Binance
        :return: (maintenance margin ratio, maintenance amount)
        """

        if (self._config.get('runmode') in OPTIMIZE_MODES
                or self.exchange_has('fetchLeverageTiers')
                or self.exchange_has('fetchMarketLeverageTiers')):

            if pair not in self._leverage_tiers:
                raise InvalidOrderException(
                    f"Maintenance margin rate for {pair} is unavailable for {self.name}"
                )

            pair_tiers = self._leverage_tiers[pair]

            for tier in reversed(pair_tiers):
                if nominal_value >= tier['minNotional']:
                    return (tier['maintenanceMarginRate'], tier['maintAmt'])

            raise OperationalException("nominal value can not be lower than 0")
            # The lowest notional_floor for any pair in fetch_leverage_tiers is always 0 because it
            # describes the min amt for a tier, and the lowest tier will always go down to 0
        else:
            raise OperationalException(f"Cannot get maintenance ratio using {self.name}")


def is_exchange_known_ccxt(exchange_name: str, ccxt_module: CcxtModuleType = None) -> bool:
    return exchange_name in ccxt_exchanges(ccxt_module)


def is_exchange_officially_supported(exchange_name: str) -> bool:
    return exchange_name in SUPPORTED_EXCHANGES


def ccxt_exchanges(ccxt_module: CcxtModuleType = None) -> List[str]:
    """
    Return the list of all exchanges known to ccxt
    """
    return ccxt_module.exchanges if ccxt_module is not None else ccxt.exchanges


def available_exchanges(ccxt_module: CcxtModuleType = None) -> List[str]:
    """
    Return exchanges available to the bot, i.e. non-bad exchanges in the ccxt list
    """
    exchanges = ccxt_exchanges(ccxt_module)
    return [x for x in exchanges if validate_exchange(x)[0]]


def validate_exchange(exchange: str) -> Tuple[bool, str]:
    ex_mod = getattr(ccxt, exchange.lower())()
    if not ex_mod or not ex_mod.has:
        return False, ''
    missing = [k for k in EXCHANGE_HAS_REQUIRED if ex_mod.has.get(k) is not True]
    if missing:
        return False, f"missing: {', '.join(missing)}"

    missing_opt = [k for k in EXCHANGE_HAS_OPTIONAL if not ex_mod.has.get(k)]

    if exchange.lower() in BAD_EXCHANGES:
        return False, BAD_EXCHANGES.get(exchange.lower(), '')
    if missing_opt:
        return True, f"missing opt: {', '.join(missing_opt)}"

    return True, ''


def validate_exchanges(all_exchanges: bool) -> List[Tuple[str, bool, str]]:
    """
    :return: List of tuples with exchangename, valid, reason.
    """
    exchanges = ccxt_exchanges() if all_exchanges else available_exchanges()
    exchanges_valid = [
        (e, *validate_exchange(e)) for e in exchanges
    ]
    return exchanges_valid


def timeframe_to_seconds(timeframe: str) -> int:
    """
    Translates the timeframe interval value written in the human readable
    form ('1m', '5m', '1h', '1d', '1w', etc.) to the number
    of seconds for one timeframe interval.
    """
    return ccxt.Exchange.parse_timeframe(timeframe)


def timeframe_to_minutes(timeframe: str) -> int:
    """
    Same as timeframe_to_seconds, but returns minutes.
    """
    return ccxt.Exchange.parse_timeframe(timeframe) // 60


def timeframe_to_msecs(timeframe: str) -> int:
    """
    Same as timeframe_to_seconds, but returns milliseconds.
    """
    return ccxt.Exchange.parse_timeframe(timeframe) * 1000


def timeframe_to_prev_date(timeframe: str, date: datetime = None) -> datetime:
    """
    Use Timeframe and determine the candle start date for this date.
    Does not round when given a candle start date.
    :param timeframe: timeframe in string format (e.g. "5m")
    :param date: date to use. Defaults to now(utc)
    :returns: date of previous candle (with utc timezone)
    """
    if not date:
        date = datetime.now(timezone.utc)

    new_timestamp = ccxt.Exchange.round_timeframe(timeframe, date.timestamp() * 1000,
                                                  ROUND_DOWN) // 1000
    return datetime.fromtimestamp(new_timestamp, tz=timezone.utc)


def timeframe_to_next_date(timeframe: str, date: datetime = None) -> datetime:
    """
    Use Timeframe and determine next candle.
    :param timeframe: timeframe in string format (e.g. "5m")
    :param date: date to use. Defaults to now(utc)
    :returns: date of next candle (with utc timezone)
    """
    if not date:
        date = datetime.now(timezone.utc)
    new_timestamp = ccxt.Exchange.round_timeframe(timeframe, date.timestamp() * 1000,
                                                  ROUND_UP) // 1000
    return datetime.fromtimestamp(new_timestamp, tz=timezone.utc)


def date_minus_candles(
        timeframe: str, candle_count: int, date: Optional[datetime] = None) -> datetime:
    """
    subtract X candles from a date.
    :param timeframe: timeframe in string format (e.g. "5m")
    :param candle_count: Amount of candles to subtract.
    :param date: date to use. Defaults to now(utc)

    """
    if not date:
        date = datetime.now(timezone.utc)

    tf_min = timeframe_to_minutes(timeframe)
    new_date = timeframe_to_prev_date(timeframe, date) - timedelta(minutes=tf_min * candle_count)
    return new_date


def market_is_active(market: Dict) -> bool:
    """
    Return True if the market is active.
    """
    # "It's active, if the active flag isn't explicitly set to false. If it's missing or
    # true then it's true. If it's undefined, then it's most likely true, but not 100% )"
    # See https://github.com/ccxt/ccxt/issues/4874,
    # https://github.com/ccxt/ccxt/issues/4075#issuecomment-434760520
    return market.get('active', True) is not False


def amount_to_contracts(amount: float, contract_size: Optional[float]) -> float:
    """
    Convert amount to contracts.
    :param amount: amount to convert
    :param contract_size: contract size - taken from exchange.get_contract_size(pair)
    :return: num-contracts
    """
    if contract_size and contract_size != 1:
        return float(FtPrecise(amount) / FtPrecise(contract_size))
    else:
        return amount


def contracts_to_amount(num_contracts: float, contract_size: Optional[float]) -> float:
    """
    Takes num-contracts and converts it to contract size
    :param num_contracts: number of contracts
    :param contract_size: contract size - taken from exchange.get_contract_size(pair)
    :return: Amount
    """

    if contract_size and contract_size != 1:
        return float(FtPrecise(num_contracts) * FtPrecise(contract_size))
    else:
        return num_contracts


def amount_to_precision(amount: float, amount_precision: Optional[float],
                        precisionMode: Optional[int]) -> float:
    """
    Returns the amount to buy or sell to a precision the Exchange accepts
    Re-implementation of ccxt internal methods - ensuring we can test the result is correct
    based on our definitions.
    :param amount: amount to truncate
    :param amount_precision: amount precision to use.
                             should be retrieved from markets[pair]['precision']['amount']
    :param precisionMode: precision mode to use. Should be used from precisionMode
                          one of ccxt's DECIMAL_PLACES, SIGNIFICANT_DIGITS, or TICK_SIZE
    :return: truncated amount
    """
    if amount_precision is not None and precisionMode is not None:
        precision = int(amount_precision) if precisionMode != TICK_SIZE else amount_precision
        # precision must be an int for non-ticksize inputs.
        amount = float(decimal_to_precision(amount, rounding_mode=TRUNCATE,
                                            precision=precision,
                                            counting_mode=precisionMode,
                                            ))

    return amount


def amount_to_contract_precision(
        amount, amount_precision: Optional[float], precisionMode: Optional[int],
        contract_size: Optional[float]) -> float:
    """
    Returns the amount to buy or sell to a precision the Exchange accepts
    including calculation to and from contracts.
    Re-implementation of ccxt internal methods - ensuring we can test the result is correct
    based on our definitions.
    :param amount: amount to truncate
    :param amount_precision: amount precision to use.
                             should be retrieved from markets[pair]['precision']['amount']
    :param precisionMode: precision mode to use. Should be used from precisionMode
                          one of ccxt's DECIMAL_PLACES, SIGNIFICANT_DIGITS, or TICK_SIZE
    :param contract_size: contract size - taken from exchange.get_contract_size(pair)
    :return: truncated amount
    """
    if amount_precision is not None and precisionMode is not None:
        contracts = amount_to_contracts(amount, contract_size)
        amount_p = amount_to_precision(contracts, amount_precision, precisionMode)
        return contracts_to_amount(amount_p, contract_size)
    return amount


def price_to_precision(price: float, price_precision: Optional[float],
                       precisionMode: Optional[int]) -> float:
    """
    Returns the price rounded up to the precision the Exchange accepts.
    Partial Re-implementation of ccxt internal method decimal_to_precision(),
    which does not support rounding up
    TODO: If ccxt supports ROUND_UP for decimal_to_precision(), we could remove this and
    align with amount_to_precision().
    !!! Rounds up
    :param price: price to convert
    :param price_precision: price precision to use. Used from markets[pair]['precision']['price']
    :param precisionMode: precision mode to use. Should be used from precisionMode
                          one of ccxt's DECIMAL_PLACES, SIGNIFICANT_DIGITS, or TICK_SIZE
    :return: price rounded up to the precision the Exchange accepts

    """
    if price_precision is not None and precisionMode is not None:
        # price = float(decimal_to_precision(price, rounding_mode=ROUND,
        #                                    precision=price_precision,
        #                                    counting_mode=self.precisionMode,
        #                                    ))
        if precisionMode == TICK_SIZE:
            precision = FtPrecise(price_precision)
            price_str = FtPrecise(price)
            missing = price_str % precision
            if not missing == FtPrecise("0"):
                price = round(float(str(price_str - missing + precision)), 14)
        else:
            symbol_prec = price_precision
            big_price = price * pow(10, symbol_prec)
            price = ceil(big_price) / pow(10, symbol_prec)
    return price
