import logging
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.exceptions import HTTPException

from freqtrade import __version__
from freqtrade.constants import USERPATH_STRATEGIES
from freqtrade.data.history import get_datahandler
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exceptions import OperationalException
from freqtrade.rpc import RPC
from freqtrade.rpc.api_server.api_schemas import (AvailablePairs, Balances, BlacklistPayload,
                                                  BlacklistResponse, Count, Daily,
                                                  DeleteLockRequest, DeleteTrade, ForceEnterPayload,
                                                  ForceEnterResponse, ForceExitPayload, Health,
                                                  Locks, Logs, OpenTradeSchema, PairHistory,
                                                  PerformanceEntry, Ping, PlotConfig, Profit,
                                                  ResultMsg, ShowConfig, Stats, StatusMsg,
                                                  StrategyListResponse, StrategyResponse, SysInfo,
                                                  Version, WhitelistResponse)
from freqtrade.rpc.api_server.deps import get_config, get_exchange, get_rpc, get_rpc_optional
from freqtrade.rpc.rpc import RPCException


logger = logging.getLogger(__name__)

# API version
# Pre-1.1, no version was provided
# Version increments should happen in "small" steps (1.1, 1.12, ...) unless big changes happen.
# 1.11: forcebuy and forcesell accept ordertype
# 1.12: add blacklist delete endpoint
# 1.13: forcebuy supports stake_amount
# versions 2.xx -> futures/short branch
# 2.14: Add entry/exit orders to trade response
# 2.15: Add backtest history endpoints
# 2.16: Additional daily metrics
# 2.17: Forceentry - leverage, partial force_exit
# 2.20: Add websocket endpoints
API_VERSION = 2.20

# Public API, requires no auth.
router_public = APIRouter()
# Private API, protected by authentication
router = APIRouter()


@router_public.get('/ping', response_model=Ping)
def ping():
    """simple ping"""
    return {"status": "pong"}


@router.get('/version', response_model=Version, tags=['info'])
def version():
    """ Bot Version info"""
    return {"version": __version__}


@router.get('/balance', response_model=Balances, tags=['info'])
def balance(rpc: RPC = Depends(get_rpc), config=Depends(get_config)):
    """Account Balances"""
    return rpc._rpc_balance(config['stake_currency'], config.get('fiat_display_currency', ''),)


@router.get('/count', response_model=Count, tags=['info'])
def count(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_count()


@router.get('/performance', response_model=List[PerformanceEntry], tags=['info'])
def performance(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_performance()


@router.get('/profit', response_model=Profit, tags=['info'])
def profit(rpc: RPC = Depends(get_rpc), config=Depends(get_config)):
    return rpc._rpc_trade_statistics(config['stake_currency'],
                                     config.get('fiat_display_currency')
                                     )


@router.get('/stats', response_model=Stats, tags=['info'])
def stats(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_stats()


@router.get('/daily', response_model=Daily, tags=['info'])
def daily(timescale: int = 7, rpc: RPC = Depends(get_rpc), config=Depends(get_config)):
    return rpc._rpc_timeunit_profit(timescale, config['stake_currency'],
                                    config.get('fiat_display_currency', ''))


@router.get('/status', response_model=List[OpenTradeSchema], tags=['info'])
def status(rpc: RPC = Depends(get_rpc)):
    try:
        return rpc._rpc_trade_status()
    except RPCException:
        return []


# Using the responsemodel here will cause a ~100% increase in response time (from 1s to 2s)
# on big databases. Correct response model: response_model=TradeResponse,
@router.get('/trades', tags=['info', 'trading'])
def trades(limit: int = 500, offset: int = 0, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_trade_history(limit, offset=offset, order_by_id=True)


@router.get('/trade/{tradeid}', response_model=OpenTradeSchema, tags=['info', 'trading'])
def trade(tradeid: int = 0, rpc: RPC = Depends(get_rpc)):
    try:
        return rpc._rpc_trade_status([tradeid])[0]
    except (RPCException, KeyError):
        raise HTTPException(status_code=404, detail='Trade not found.')


@router.delete('/trades/{tradeid}', response_model=DeleteTrade, tags=['info', 'trading'])
def trades_delete(tradeid: int, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_delete(tradeid)


# TODO: Missing response model
@router.get('/edge', tags=['info'])
def edge(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_edge()


@router.get('/show_config', response_model=ShowConfig, tags=['info'])
def show_config(rpc: Optional[RPC] = Depends(get_rpc_optional), config=Depends(get_config)):
    state = ''
    strategy_version = None
    if rpc:
        state = rpc._freqtrade.state
        strategy_version = rpc._freqtrade.strategy.version()
    resp = RPC._rpc_show_config(config, state, strategy_version)
    resp['api_version'] = API_VERSION
    return resp


# /forcebuy is deprecated with short addition. use /forceentry instead
@router.post('/forceenter', response_model=ForceEnterResponse, tags=['trading'])
@router.post('/forcebuy', response_model=ForceEnterResponse, tags=['trading'])
def force_entry(payload: ForceEnterPayload, rpc: RPC = Depends(get_rpc)):
    ordertype = payload.ordertype.value if payload.ordertype else None

    trade = rpc._rpc_force_entry(payload.pair, payload.price, order_side=payload.side,
                                 order_type=ordertype, stake_amount=payload.stakeamount,
                                 enter_tag=payload.entry_tag or 'force_entry',
                                 leverage=payload.leverage)

    if trade:
        return ForceEnterResponse.parse_obj(trade.to_json())
    else:
        return ForceEnterResponse.parse_obj(
            {"status": f"Error entering {payload.side} trade for pair {payload.pair}."})


# /forcesell is deprecated with short addition. use /forceexit instead
@router.post('/forceexit', response_model=ResultMsg, tags=['trading'])
@router.post('/forcesell', response_model=ResultMsg, tags=['trading'])
def forceexit(payload: ForceExitPayload, rpc: RPC = Depends(get_rpc)):
    ordertype = payload.ordertype.value if payload.ordertype else None
    return rpc._rpc_force_exit(payload.tradeid, ordertype, amount=payload.amount)


@router.get('/blacklist', response_model=BlacklistResponse, tags=['info', 'pairlist'])
def blacklist(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_blacklist()


@router.post('/blacklist', response_model=BlacklistResponse, tags=['info', 'pairlist'])
def blacklist_post(payload: BlacklistPayload, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_blacklist(payload.blacklist)


@router.delete('/blacklist', response_model=BlacklistResponse, tags=['info', 'pairlist'])
def blacklist_delete(pairs_to_delete: List[str] = Query([]), rpc: RPC = Depends(get_rpc)):
    """Provide a list of pairs to delete from the blacklist"""

    return rpc._rpc_blacklist_delete(pairs_to_delete)


@router.get('/whitelist', response_model=WhitelistResponse, tags=['info', 'pairlist'])
def whitelist(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_whitelist()


@router.get('/locks', response_model=Locks, tags=['info', 'locks'])
def locks(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_locks()


@router.delete('/locks/{lockid}', response_model=Locks, tags=['info', 'locks'])
def delete_lock(lockid: int, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_delete_lock(lockid=lockid)


@router.post('/locks/delete', response_model=Locks, tags=['info', 'locks'])
def delete_lock_pair(payload: DeleteLockRequest, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_delete_lock(lockid=payload.lockid, pair=payload.pair)


@router.get('/logs', response_model=Logs, tags=['info'])
def logs(limit: Optional[int] = None):
    return RPC._rpc_get_logs(limit)


@router.post('/start', response_model=StatusMsg, tags=['botcontrol'])
def start(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_start()


@router.post('/stop', response_model=StatusMsg, tags=['botcontrol'])
def stop(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_stop()


@router.post('/stopentry', response_model=StatusMsg, tags=['botcontrol'])
@router.post('/stopbuy', response_model=StatusMsg, tags=['botcontrol'])
def stop_buy(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_stopentry()


@router.post('/reload_config', response_model=StatusMsg, tags=['botcontrol'])
def reload_config(rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_reload_config()


@router.get('/pair_candles', response_model=PairHistory, tags=['candle data'])
def pair_candles(
        pair: str, timeframe: str, limit: Optional[int] = None, rpc: RPC = Depends(get_rpc)):
    return rpc._rpc_analysed_dataframe(pair, timeframe, limit)


@router.get('/pair_history', response_model=PairHistory, tags=['candle data'])
def pair_history(pair: str, timeframe: str, timerange: str, strategy: str,
                 config=Depends(get_config), exchange=Depends(get_exchange)):
    # The initial call to this endpoint can be slow, as it may need to initialize
    # the exchange class.
    config = deepcopy(config)
    config.update({
        'strategy': strategy,
    })
    return RPC._rpc_analysed_history_full(config, pair, timeframe, timerange, exchange)


@router.get('/plot_config', response_model=PlotConfig, tags=['candle data'])
def plot_config(rpc: RPC = Depends(get_rpc)):
    return PlotConfig.parse_obj(rpc._rpc_plot_config())


@router.get('/strategies', response_model=StrategyListResponse, tags=['strategy'])
def list_strategies(config=Depends(get_config)):
    directory = Path(config.get(
        'strategy_path', config['user_data_dir'] / USERPATH_STRATEGIES))
    from freqtrade.resolvers.strategy_resolver import StrategyResolver
    strategies = StrategyResolver.search_all_objects(
        directory, False, config.get('recursive_strategy_search', False))
    strategies = sorted(strategies, key=lambda x: x['name'])

    return {'strategies': [x['name'] for x in strategies]}


@router.get('/strategy/{strategy}', response_model=StrategyResponse, tags=['strategy'])
def get_strategy(strategy: str, config=Depends(get_config)):
    if ":" in strategy:
        raise HTTPException(status_code=500, detail="base64 encoded strategies are not allowed.")

    config_ = deepcopy(config)
    from freqtrade.resolvers.strategy_resolver import StrategyResolver
    try:
        strategy_obj = StrategyResolver._load_strategy(strategy, config_,
                                                       extra_dir=config_.get('strategy_path'))
    except OperationalException:
        raise HTTPException(status_code=404, detail='Strategy not found')

    return {
        'strategy': strategy_obj.get_strategy_name(),
        'code': strategy_obj.__source__,
    }


@router.get('/available_pairs', response_model=AvailablePairs, tags=['candle data'])
def list_available_pairs(timeframe: Optional[str] = None, stake_currency: Optional[str] = None,
                         candletype: Optional[CandleType] = None, config=Depends(get_config)):

    dh = get_datahandler(config['datadir'], config.get('dataformat_ohlcv'))
    trading_mode: TradingMode = config.get('trading_mode', TradingMode.SPOT)
    pair_interval = dh.ohlcv_get_available_data(config['datadir'], trading_mode)

    if timeframe:
        pair_interval = [pair for pair in pair_interval if pair[1] == timeframe]
    if stake_currency:
        pair_interval = [pair for pair in pair_interval if pair[0].endswith(stake_currency)]
    if candletype:
        pair_interval = [pair for pair in pair_interval if pair[2] == candletype]
    else:
        candle_type = CandleType.get_default(trading_mode)
        pair_interval = [pair for pair in pair_interval if pair[2] == candle_type]

    pair_interval = sorted(pair_interval, key=lambda x: x[0])

    pairs = list({x[0] for x in pair_interval})
    pairs.sort()
    result = {
        'length': len(pairs),
        'pairs': pairs,
        'pair_interval': pair_interval,
    }
    return result


@router.get('/sysinfo', response_model=SysInfo, tags=['info'])
def sysinfo():
    return RPC._rpc_sysinfo()


@router.get('/health', response_model=Health, tags=['info'])
def health(rpc: RPC = Depends(get_rpc)):
    return rpc._health()
