# pragma pylint: disable=missing-docstring, protected-access, C0103

import json
import re
import uuid
from pathlib import Path
from shutil import copyfile
from unittest.mock import MagicMock, PropertyMock

import arrow
import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from freqtrade.configuration import TimeRange
from freqtrade.constants import AVAILABLE_DATAHANDLERS, DATETIME_PRINT_FORMAT
from freqtrade.data.converter import ohlcv_to_dataframe
from freqtrade.data.history.hdf5datahandler import HDF5DataHandler
from freqtrade.data.history.history_utils import (_download_pair_history, _download_trades_history,
                                                  _load_cached_data_for_updating,
                                                  convert_trades_to_ohlcv, get_timerange, load_data,
                                                  load_pair_history, refresh_backtest_ohlcv_data,
                                                  refresh_backtest_trades_data, refresh_data,
                                                  validate_backtest_data)
from freqtrade.data.history.idatahandler import IDataHandler, get_datahandler, get_datahandlerclass
from freqtrade.data.history.jsondatahandler import JsonDataHandler, JsonGzDataHandler
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.misc import file_dump_json
from freqtrade.resolvers import StrategyResolver
from tests.conftest import (CURRENT_TEST_STRATEGY, get_patched_exchange, log_has, log_has_re,
                            patch_exchange)


# Change this if modifying UNITTEST/BTC testdatafile
_BTC_UNITTEST_LENGTH = 13681


def _backup_file(file: Path, copy_file: bool = False) -> None:
    """
    Backup existing file to avoid deleting the user file
    :param file: complete path to the file
    :param copy_file: keep file in place too.
    :return: None
    """
    file_swp = str(file) + '.swp'
    if file.is_file():
        file.rename(file_swp)

        if copy_file:
            copyfile(file_swp, file)


def _clean_test_file(file: Path) -> None:
    """
    Backup existing file to avoid deleting the user file
    :param file: complete path to the file
    :return: None
    """
    file_swp = Path(str(file) + '.swp')
    # 1. Delete file from the test
    if file.is_file():
        file.unlink()

    # 2. Rollback to the initial file
    if file_swp.is_file():
        file_swp.rename(file)


def test_load_data_30min_timeframe(mocker, caplog, default_conf, testdatadir) -> None:
    ld = load_pair_history(pair='UNITTEST/BTC', timeframe='30m', datadir=testdatadir)
    assert isinstance(ld, DataFrame)
    assert not log_has(
        'Download history data for pair: "UNITTEST/BTC", timeframe: 30m '
        'and store in None.', caplog
    )


def test_load_data_7min_timeframe(mocker, caplog, default_conf, testdatadir) -> None:
    ld = load_pair_history(pair='UNITTEST/BTC', timeframe='7m', datadir=testdatadir)
    assert isinstance(ld, DataFrame)
    assert ld.empty
    assert log_has(
        'No history for UNITTEST/BTC, spot, 7m found. '
        'Use `freqtrade download-data` to download the data', caplog
    )


def test_load_data_1min_timeframe(ohlcv_history, mocker, caplog, testdatadir) -> None:
    mocker.patch('freqtrade.exchange.Exchange.get_historic_ohlcv', return_value=ohlcv_history)
    file = testdatadir / 'UNITTEST_BTC-1m.json'
    load_data(datadir=testdatadir, timeframe='1m', pairs=['UNITTEST/BTC'])
    assert file.is_file()
    assert not log_has(
        'Download history data for pair: "UNITTEST/BTC", interval: 1m '
        'and store in None.', caplog
    )


def test_load_data_mark(ohlcv_history, mocker, caplog, testdatadir) -> None:
    mocker.patch('freqtrade.exchange.Exchange.get_historic_ohlcv', return_value=ohlcv_history)
    file = testdatadir / 'futures/UNITTEST_USDT-1h-mark.json'
    load_data(datadir=testdatadir, timeframe='1h', pairs=['UNITTEST/BTC'], candle_type='mark')
    assert file.is_file()
    assert not log_has(
        'Download history data for pair: "UNITTEST/USDT", interval: 1m '
        'and store in None.', caplog
    )


def test_load_data_startup_candles(mocker, caplog, default_conf, testdatadir) -> None:
    ltfmock = mocker.patch(
        'freqtrade.data.history.jsondatahandler.JsonDataHandler._ohlcv_load',
        MagicMock(return_value=DataFrame()))
    timerange = TimeRange('date', None, 1510639620, 0)
    load_pair_history(pair='UNITTEST/BTC', timeframe='1m',
                      datadir=testdatadir, timerange=timerange,
                      startup_candles=20,)

    assert ltfmock.call_count == 1
    assert ltfmock.call_args_list[0][1]['timerange'] != timerange
    # startts is 20 minutes earlier
    assert ltfmock.call_args_list[0][1]['timerange'].startts == timerange.startts - 20 * 60


@pytest.mark.parametrize('candle_type', ['mark', ''])
def test_load_data_with_new_pair_1min(ohlcv_history_list, mocker, caplog,
                                      default_conf, tmpdir, candle_type) -> None:
    """
    Test load_pair_history() with 1 min timeframe
    """
    tmpdir1 = Path(tmpdir)
    mocker.patch('freqtrade.exchange.Exchange.get_historic_ohlcv', return_value=ohlcv_history_list)
    exchange = get_patched_exchange(mocker, default_conf)
    file = tmpdir1 / 'MEME_BTC-1m.json'

    # do not download a new pair if refresh_pairs isn't set
    load_pair_history(datadir=tmpdir1, timeframe='1m', pair='MEME/BTC', candle_type=candle_type)
    assert not file.is_file()
    assert log_has(
        f"No history for MEME/BTC, {candle_type}, 1m found. "
        "Use `freqtrade download-data` to download the data", caplog
    )

    # download a new pair if refresh_pairs is set
    refresh_data(datadir=tmpdir1, timeframe='1m', pairs=['MEME/BTC'],
                 exchange=exchange, candle_type=CandleType.SPOT
                 )
    load_pair_history(datadir=tmpdir1, timeframe='1m', pair='MEME/BTC', candle_type=candle_type)
    assert file.is_file()
    assert log_has_re(
        r'\(0/1\) - Download history data for "MEME/BTC", 1m, '
        r'spot and store in .*', caplog
    )


def test_testdata_path(testdatadir) -> None:
    assert str(Path('tests') / 'testdata') in str(testdatadir)


@pytest.mark.parametrize("pair,timeframe,expected_result,candle_type", [
    ("ETH/BTC", "5m", "freqtrade/hello/world/ETH_BTC-5m.json", ""),
    ("ETH/USDT", "1M", "freqtrade/hello/world/ETH_USDT-1Mo.json", ""),
    ("Fabric Token/ETH", "5m", "freqtrade/hello/world/Fabric_Token_ETH-5m.json", ""),
    ("ETHH20", "5m", "freqtrade/hello/world/ETHH20-5m.json", ""),
    (".XBTBON2H", "5m", "freqtrade/hello/world/_XBTBON2H-5m.json", ""),
    ("ETHUSD.d", "5m", "freqtrade/hello/world/ETHUSD_d-5m.json", ""),
    ("ACC_OLD/BTC", "5m", "freqtrade/hello/world/ACC_OLD_BTC-5m.json", ""),
    ("ETH/BTC", "5m", "freqtrade/hello/world/futures/ETH_BTC-5m-mark.json", "mark"),
    ("ACC_OLD/BTC", "5m", "freqtrade/hello/world/futures/ACC_OLD_BTC-5m-index.json", "index"),
])
def test_json_pair_data_filename(pair, timeframe, expected_result, candle_type):
    fn = JsonDataHandler._pair_data_filename(
        Path('freqtrade/hello/world'),
        pair,
        timeframe,
        CandleType.from_string(candle_type)
    )
    assert isinstance(fn, Path)
    assert fn == Path(expected_result)
    fn = JsonGzDataHandler._pair_data_filename(
        Path('freqtrade/hello/world'),
        pair,
        timeframe,
        candle_type=CandleType.from_string(candle_type)
    )
    assert isinstance(fn, Path)
    assert fn == Path(expected_result + '.gz')


@pytest.mark.parametrize("pair,expected_result", [
    ("ETH/BTC", 'freqtrade/hello/world/ETH_BTC-trades.json'),
    ("Fabric Token/ETH", 'freqtrade/hello/world/Fabric_Token_ETH-trades.json'),
    ("ETHH20", 'freqtrade/hello/world/ETHH20-trades.json'),
    (".XBTBON2H", 'freqtrade/hello/world/_XBTBON2H-trades.json'),
    ("ETHUSD.d", 'freqtrade/hello/world/ETHUSD_d-trades.json'),
    ("ACC_OLD_BTC", 'freqtrade/hello/world/ACC_OLD_BTC-trades.json'),
])
def test_json_pair_trades_filename(pair, expected_result):
    fn = JsonDataHandler._pair_trades_filename(Path('freqtrade/hello/world'), pair)
    assert isinstance(fn, Path)
    assert fn == Path(expected_result)

    fn = JsonGzDataHandler._pair_trades_filename(Path('freqtrade/hello/world'), pair)
    assert isinstance(fn, Path)
    assert fn == Path(expected_result + '.gz')


def test_load_cached_data_for_updating(mocker, testdatadir) -> None:

    data_handler = get_datahandler(testdatadir, 'json')

    test_data = None
    test_filename = testdatadir.joinpath('UNITTEST_BTC-1m.json')
    with open(test_filename, "rt") as file:
        test_data = json.load(file)

    test_data_df = ohlcv_to_dataframe(test_data, '1m', 'UNITTEST/BTC',
                                      fill_missing=False, drop_incomplete=False)
    # now = last cached item + 1 hour
    now_ts = test_data[-1][0] / 1000 + 60 * 60
    mocker.patch('arrow.utcnow', return_value=arrow.get(now_ts))

    # timeframe starts earlier than the cached data
    # should fully update data
    timerange = TimeRange('date', None, test_data[0][0] / 1000 - 1, 0)
    data, start_ts, end_ts = _load_cached_data_for_updating(
        'UNITTEST/BTC', '1m', timerange, data_handler, CandleType.SPOT)
    assert data.empty
    assert start_ts == test_data[0][0] - 1000
    assert end_ts is None

    # timeframe starts earlier than the cached data - prepending

    timerange = TimeRange('date', None, test_data[0][0] / 1000 - 1, 0)
    data, start_ts, end_ts = _load_cached_data_for_updating(
        'UNITTEST/BTC', '1m', timerange, data_handler, CandleType.SPOT, True)
    assert_frame_equal(data, test_data_df.iloc[:-1])
    assert start_ts == test_data[0][0] - 1000
    assert end_ts == test_data[0][0]

    # timeframe starts in the center of the cached data
    # should return the cached data w/o the last item
    timerange = TimeRange('date', None, test_data[0][0] / 1000 + 1, 0)
    data, start_ts, end_ts = _load_cached_data_for_updating(
        'UNITTEST/BTC', '1m', timerange, data_handler, CandleType.SPOT)

    assert_frame_equal(data, test_data_df.iloc[:-1])
    assert test_data[-2][0] <= start_ts < test_data[-1][0]
    assert end_ts is None

    # timeframe starts after the cached data
    # should return the cached data w/o the last item
    timerange = TimeRange('date', None, test_data[-1][0] / 1000 + 100, 0)
    data, start_ts, end_ts = _load_cached_data_for_updating(
        'UNITTEST/BTC', '1m', timerange, data_handler, CandleType.SPOT)
    assert_frame_equal(data, test_data_df.iloc[:-1])
    assert test_data[-2][0] <= start_ts < test_data[-1][0]
    assert end_ts is None

    # no datafile exist
    # should return timestamp start time
    timerange = TimeRange('date', None, now_ts - 10000, 0)
    data, start_ts, end_ts = _load_cached_data_for_updating(
        'NONEXIST/BTC', '1m', timerange, data_handler, CandleType.SPOT)
    assert data.empty
    assert start_ts == (now_ts - 10000) * 1000
    assert end_ts is None

    # no datafile exist
    # should return timestamp start and end time time
    timerange = TimeRange('date', 'date', now_ts - 1000000, now_ts - 100000)
    data, start_ts, end_ts = _load_cached_data_for_updating(
        'NONEXIST/BTC', '1m', timerange, data_handler, CandleType.SPOT)
    assert data.empty
    assert start_ts == (now_ts - 1000000) * 1000
    assert end_ts == (now_ts - 100000) * 1000

    # no datafile exist, no timeframe is set
    # should return an empty array and None
    data, start_ts, end_ts = _load_cached_data_for_updating(
        'NONEXIST/BTC', '1m', None, data_handler, CandleType.SPOT)
    assert data.empty
    assert start_ts is None
    assert end_ts is None


@pytest.mark.parametrize('candle_type,subdir,file_tail', [
    ('mark', 'futures/', '-mark'),
    ('spot', '', ''),
])
def test_download_pair_history(
    ohlcv_history_list,
    mocker,
    default_conf,
    tmpdir,
    candle_type,
    subdir,
    file_tail
) -> None:
    mocker.patch('freqtrade.exchange.Exchange.get_historic_ohlcv', return_value=ohlcv_history_list)
    exchange = get_patched_exchange(mocker, default_conf)
    tmpdir1 = Path(tmpdir)
    file1_1 = tmpdir1 / f'{subdir}MEME_BTC-1m{file_tail}.json'
    file1_5 = tmpdir1 / f'{subdir}MEME_BTC-5m{file_tail}.json'
    file2_1 = tmpdir1 / f'{subdir}CFI_BTC-1m{file_tail}.json'
    file2_5 = tmpdir1 / f'{subdir}CFI_BTC-5m{file_tail}.json'

    assert not file1_1.is_file()
    assert not file2_1.is_file()

    assert _download_pair_history(datadir=tmpdir1, exchange=exchange,
                                  pair='MEME/BTC',
                                  timeframe='1m',
                                  candle_type=candle_type)
    assert _download_pair_history(datadir=tmpdir1, exchange=exchange,
                                  pair='CFI/BTC',
                                  timeframe='1m',
                                  candle_type=candle_type)
    assert not exchange._pairs_last_refresh_time
    assert file1_1.is_file()
    assert file2_1.is_file()

    # clean files freshly downloaded
    _clean_test_file(file1_1)
    _clean_test_file(file2_1)

    assert not file1_5.is_file()
    assert not file2_5.is_file()

    assert _download_pair_history(datadir=tmpdir1, exchange=exchange,
                                  pair='MEME/BTC',
                                  timeframe='5m',
                                  candle_type=candle_type)
    assert _download_pair_history(datadir=tmpdir1, exchange=exchange,
                                  pair='CFI/BTC',
                                  timeframe='5m',
                                  candle_type=candle_type)
    assert not exchange._pairs_last_refresh_time
    assert file1_5.is_file()
    assert file2_5.is_file()


def test_download_pair_history2(mocker, default_conf, testdatadir) -> None:
    tick = [
        [1509836520000, 0.00162008, 0.00162008, 0.00162008, 0.00162008, 108.14853839],
        [1509836580000, 0.00161, 0.00161, 0.00161, 0.00161, 82.390199]
    ]
    json_dump_mock = mocker.patch(
        'freqtrade.data.history.jsondatahandler.JsonDataHandler.ohlcv_store',
        return_value=None)
    mocker.patch('freqtrade.exchange.Exchange.get_historic_ohlcv', return_value=tick)
    exchange = get_patched_exchange(mocker, default_conf)
    _download_pair_history(datadir=testdatadir, exchange=exchange, pair="UNITTEST/BTC",
                           timeframe='1m', candle_type='spot')
    _download_pair_history(datadir=testdatadir, exchange=exchange, pair="UNITTEST/BTC",
                           timeframe='3m', candle_type='spot')
    _download_pair_history(datadir=testdatadir, exchange=exchange, pair="UNITTEST/USDT",
                           timeframe='1h', candle_type='mark')
    assert json_dump_mock.call_count == 3


def test_download_backtesting_data_exception(mocker, caplog, default_conf, tmpdir) -> None:
    mocker.patch('freqtrade.exchange.Exchange.get_historic_ohlcv',
                 side_effect=Exception('File Error'))
    tmpdir1 = Path(tmpdir)
    exchange = get_patched_exchange(mocker, default_conf)

    assert not _download_pair_history(datadir=tmpdir1, exchange=exchange,
                                      pair='MEME/BTC',
                                      timeframe='1m', candle_type='spot')
    assert log_has('Failed to download history data for pair: "MEME/BTC", timeframe: 1m.', caplog)


def test_load_partial_missing(testdatadir, caplog) -> None:
    # Make sure we start fresh - test missing data at start
    start = arrow.get('2018-01-01T00:00:00')
    end = arrow.get('2018-01-11T00:00:00')
    data = load_data(testdatadir, '5m', ['UNITTEST/BTC'], startup_candles=20,
                     timerange=TimeRange('date', 'date', start.int_timestamp, end.int_timestamp))
    assert log_has(
        'Using indicator startup period: 20 ...', caplog
    )
    # timedifference in 5 minutes
    td = ((end - start).total_seconds() // 60 // 5) + 1
    assert td != len(data['UNITTEST/BTC'])
    start_real = data['UNITTEST/BTC'].iloc[0, 0]
    assert log_has(f'UNITTEST/BTC, spot, 5m, '
                   f'data starts at {start_real.strftime(DATETIME_PRINT_FORMAT)}',
                   caplog)
    # Make sure we start fresh - test missing data at end
    caplog.clear()
    start = arrow.get('2018-01-10T00:00:00')
    end = arrow.get('2018-02-20T00:00:00')
    data = load_data(datadir=testdatadir, timeframe='5m', pairs=['UNITTEST/BTC'],
                     timerange=TimeRange('date', 'date', start.int_timestamp, end.int_timestamp))
    # timedifference in 5 minutes
    td = ((end - start).total_seconds() // 60 // 5) + 1
    assert td != len(data['UNITTEST/BTC'])

    # Shift endtime with +5 - as last candle is dropped (partial candle)
    end_real = arrow.get(data['UNITTEST/BTC'].iloc[-1, 0]).shift(minutes=5)
    assert log_has(f'UNITTEST/BTC, spot, 5m, '
                   f'data ends at {end_real.strftime(DATETIME_PRINT_FORMAT)}',
                   caplog)


def test_init(default_conf, mocker) -> None:
    assert {} == load_data(
        datadir=Path(''),
        pairs=[],
        timeframe=default_conf['timeframe']
    )


def test_init_with_refresh(default_conf, mocker) -> None:
    exchange = get_patched_exchange(mocker, default_conf)
    refresh_data(
        datadir=Path(''),
        pairs=[],
        timeframe=default_conf['timeframe'],
        exchange=exchange,
        candle_type=CandleType.SPOT
    )
    assert {} == load_data(
        datadir=Path(''),
        pairs=[],
        timeframe=default_conf['timeframe']
    )


def test_file_dump_json_tofile(testdatadir) -> None:
    file = testdatadir / 'test_{id}.json'.format(id=str(uuid.uuid4()))
    data = {'bar': 'foo'}

    # check the file we will create does not exist
    assert not file.is_file()

    # Create the Json file
    file_dump_json(file, data)

    # Check the file was create
    assert file.is_file()

    # Open the Json file created and test the data is in it
    with file.open() as data_file:
        json_from_file = json.load(data_file)

    assert 'bar' in json_from_file
    assert json_from_file['bar'] == 'foo'

    # Remove the file
    _clean_test_file(file)


def test_get_timerange(default_conf, mocker, testdatadir) -> None:
    patch_exchange(mocker)

    default_conf.update({'strategy': CURRENT_TEST_STRATEGY})
    strategy = StrategyResolver.load_strategy(default_conf)

    data = strategy.advise_all_indicators(
        load_data(
            datadir=testdatadir,
            timeframe='1m',
            pairs=['UNITTEST/BTC']
        )
    )
    min_date, max_date = get_timerange(data)
    assert min_date.isoformat() == '2017-11-04T23:02:00+00:00'
    assert max_date.isoformat() == '2017-11-14T22:58:00+00:00'


def test_validate_backtest_data_warn(default_conf, mocker, caplog, testdatadir) -> None:
    patch_exchange(mocker)

    default_conf.update({'strategy': CURRENT_TEST_STRATEGY})
    strategy = StrategyResolver.load_strategy(default_conf)

    data = strategy.advise_all_indicators(
        load_data(
            datadir=testdatadir,
            timeframe='1m',
            pairs=['UNITTEST/BTC'],
            fill_up_missing=False
        )
    )
    min_date, max_date = get_timerange(data)
    caplog.clear()
    assert validate_backtest_data(data['UNITTEST/BTC'], 'UNITTEST/BTC',
                                  min_date, max_date, timeframe_to_minutes('1m'))
    assert len(caplog.record_tuples) == 1
    assert log_has(
        "UNITTEST/BTC has missing frames: expected 14396, got 13680, that's 716 missing values",
        caplog)


def test_validate_backtest_data(default_conf, mocker, caplog, testdatadir) -> None:
    patch_exchange(mocker)

    default_conf.update({'strategy': CURRENT_TEST_STRATEGY})
    strategy = StrategyResolver.load_strategy(default_conf)

    timerange = TimeRange('index', 'index', 200, 250)
    data = strategy.advise_all_indicators(
        load_data(
            datadir=testdatadir,
            timeframe='5m',
            pairs=['UNITTEST/BTC'],
            timerange=timerange
        )
    )

    min_date, max_date = get_timerange(data)
    caplog.clear()
    assert not validate_backtest_data(data['UNITTEST/BTC'], 'UNITTEST/BTC',
                                      min_date, max_date, timeframe_to_minutes('5m'))
    assert len(caplog.record_tuples) == 0


@pytest.mark.parametrize('trademode,callcount', [
    ('spot', 4),
    ('margin', 4),
    ('futures', 8),  # Called 8 times - 4 normal, 2 funding and 2 mark/index calls
])
def test_refresh_backtest_ohlcv_data(
        mocker, default_conf, markets, caplog, testdatadir, trademode, callcount):
    dl_mock = mocker.patch('freqtrade.data.history.history_utils._download_pair_history',
                           MagicMock())
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets)
    )
    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    mocker.patch.object(Path, "unlink", MagicMock())

    ex = get_patched_exchange(mocker, default_conf)
    timerange = TimeRange.parse_timerange("20190101-20190102")
    refresh_backtest_ohlcv_data(exchange=ex, pairs=["ETH/BTC", "XRP/BTC"],
                                timeframes=["1m", "5m"], datadir=testdatadir,
                                timerange=timerange, erase=True,
                                trading_mode=trademode
                                )

    assert dl_mock.call_count == callcount
    assert dl_mock.call_args[1]['timerange'].starttype == 'date'

    assert log_has("Downloading pair ETH/BTC, interval 1m.", caplog)


def test_download_data_no_markets(mocker, default_conf, caplog, testdatadir):
    dl_mock = mocker.patch('freqtrade.data.history.history_utils._download_pair_history',
                           MagicMock())

    ex = get_patched_exchange(mocker, default_conf)
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value={})
    )
    timerange = TimeRange.parse_timerange("20190101-20190102")
    unav_pairs = refresh_backtest_ohlcv_data(exchange=ex, pairs=["BTT/BTC", "LTC/USDT"],
                                             timeframes=["1m", "5m"],
                                             datadir=testdatadir,
                                             timerange=timerange, erase=False,
                                             trading_mode='spot'
                                             )

    assert dl_mock.call_count == 0
    assert "BTT/BTC" in unav_pairs
    assert "LTC/USDT" in unav_pairs
    assert log_has("Skipping pair BTT/BTC...", caplog)


def test_refresh_backtest_trades_data(mocker, default_conf, markets, caplog, testdatadir):
    dl_mock = mocker.patch('freqtrade.data.history.history_utils._download_trades_history',
                           MagicMock())
    mocker.patch(
        'freqtrade.exchange.Exchange.markets', PropertyMock(return_value=markets)
    )
    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    mocker.patch.object(Path, "unlink", MagicMock())

    ex = get_patched_exchange(mocker, default_conf)
    timerange = TimeRange.parse_timerange("20190101-20190102")
    unavailable_pairs = refresh_backtest_trades_data(exchange=ex,
                                                     pairs=["ETH/BTC", "XRP/BTC", "XRP/ETH"],
                                                     datadir=testdatadir,
                                                     timerange=timerange, erase=True
                                                     )

    assert dl_mock.call_count == 2
    assert dl_mock.call_args[1]['timerange'].starttype == 'date'

    assert log_has("Downloading trades for pair ETH/BTC.", caplog)
    assert unavailable_pairs == ["XRP/ETH"]
    assert log_has("Skipping pair XRP/ETH...", caplog)


def test_download_trades_history(trades_history, mocker, default_conf, testdatadir, caplog,
                                 tmpdir) -> None:
    tmpdir1 = Path(tmpdir)
    ght_mock = MagicMock(side_effect=lambda pair, *args, **kwargs: (pair, trades_history))
    mocker.patch('freqtrade.exchange.Exchange.get_historic_trades',
                 ght_mock)
    exchange = get_patched_exchange(mocker, default_conf)
    file1 = tmpdir1 / 'ETH_BTC-trades.json.gz'
    data_handler = get_datahandler(tmpdir1, data_format='jsongz')

    assert not file1.is_file()

    assert _download_trades_history(data_handler=data_handler, exchange=exchange,
                                    pair='ETH/BTC')
    assert log_has("New Amount of trades: 5", caplog)
    assert file1.is_file()

    ght_mock.reset_mock()
    since_time = int(trades_history[-3][0] // 1000)
    since_time2 = int(trades_history[-1][0] // 1000)
    timerange = TimeRange('date', None, since_time, 0)
    assert _download_trades_history(data_handler=data_handler, exchange=exchange,
                                    pair='ETH/BTC', timerange=timerange)

    assert ght_mock.call_count == 1
    # Check this in seconds - since we had to convert to seconds above too.
    assert int(ght_mock.call_args_list[0][1]['since'] // 1000) == since_time2 - 5
    assert ght_mock.call_args_list[0][1]['from_id'] is not None

    file1.unlink()

    mocker.patch('freqtrade.exchange.Exchange.get_historic_trades',
                 MagicMock(side_effect=ValueError))

    assert not _download_trades_history(data_handler=data_handler, exchange=exchange,
                                        pair='ETH/BTC')
    assert log_has_re('Failed to download historic trades for pair: "ETH/BTC".*', caplog)

    file2 = tmpdir1 / 'XRP_ETH-trades.json.gz'
    copyfile(testdatadir / file2.name, file2)

    ght_mock.reset_mock()
    mocker.patch('freqtrade.exchange.Exchange.get_historic_trades',
                 ght_mock)
    # Since before first start date
    since_time = int(trades_history[0][0] // 1000) - 500
    timerange = TimeRange('date', None, since_time, 0)

    assert _download_trades_history(data_handler=data_handler, exchange=exchange,
                                    pair='XRP/ETH', timerange=timerange)

    assert ght_mock.call_count == 1

    assert int(ght_mock.call_args_list[0][1]['since'] // 1000) == since_time
    assert ght_mock.call_args_list[0][1]['from_id'] is None
    assert log_has_re(r'Start earlier than available data. Redownloading trades for.*', caplog)
    _clean_test_file(file2)


def test_convert_trades_to_ohlcv(testdatadir, tmpdir, caplog):
    tmpdir1 = Path(tmpdir)
    pair = 'XRP/ETH'
    file1 = tmpdir1 / 'XRP_ETH-1m.json'
    file5 = tmpdir1 / 'XRP_ETH-5m.json'
    filetrades = tmpdir1 / 'XRP_ETH-trades.json.gz'
    copyfile(testdatadir / file1.name, file1)
    copyfile(testdatadir / file5.name, file5)
    copyfile(testdatadir / filetrades.name, filetrades)

    # Compare downloaded dataset with converted dataset
    dfbak_1m = load_pair_history(datadir=tmpdir1, timeframe="1m", pair=pair)
    dfbak_5m = load_pair_history(datadir=tmpdir1, timeframe="5m", pair=pair)

    tr = TimeRange.parse_timerange('20191011-20191012')

    convert_trades_to_ohlcv([pair], timeframes=['1m', '5m'],
                            datadir=tmpdir1, timerange=tr, erase=True)

    assert log_has("Deleting existing data for pair XRP/ETH, interval 1m.", caplog)
    # Load new data
    df_1m = load_pair_history(datadir=tmpdir1, timeframe="1m", pair=pair)
    df_5m = load_pair_history(datadir=tmpdir1, timeframe="5m", pair=pair)

    assert df_1m.equals(dfbak_1m)
    assert df_5m.equals(dfbak_5m)

    assert not log_has('Could not convert NoDatapair to OHLCV.', caplog)

    convert_trades_to_ohlcv(['NoDatapair'], timeframes=['1m', '5m'],
                            datadir=tmpdir1, timerange=tr, erase=True)
    assert log_has('Could not convert NoDatapair to OHLCV.', caplog)


def test_datahandler_ohlcv_get_pairs(testdatadir):
    pairs = JsonDataHandler.ohlcv_get_pairs(testdatadir, '5m', candle_type=CandleType.SPOT)
    # Convert to set to avoid failures due to sorting
    assert set(pairs) == {'UNITTEST/BTC', 'XLM/BTC', 'ETH/BTC', 'TRX/BTC', 'LTC/BTC',
                          'XMR/BTC', 'ZEC/BTC', 'ADA/BTC', 'ETC/BTC', 'NXT/BTC',
                          'DASH/BTC', 'XRP/ETH'}

    pairs = JsonGzDataHandler.ohlcv_get_pairs(testdatadir, '8m', candle_type=CandleType.SPOT)
    assert set(pairs) == {'UNITTEST/BTC'}

    pairs = HDF5DataHandler.ohlcv_get_pairs(testdatadir, '5m', candle_type=CandleType.SPOT)
    assert set(pairs) == {'UNITTEST/BTC'}

    pairs = JsonDataHandler.ohlcv_get_pairs(testdatadir, '1h', candle_type=CandleType.MARK)
    assert set(pairs) == {'UNITTEST/USDT', 'XRP/USDT'}

    pairs = JsonGzDataHandler.ohlcv_get_pairs(testdatadir, '1h', candle_type=CandleType.FUTURES)
    assert set(pairs) == {'XRP/USDT'}

    pairs = HDF5DataHandler.ohlcv_get_pairs(testdatadir, '1h', candle_type=CandleType.MARK)
    assert set(pairs) == {'UNITTEST/USDT:USDT'}


@pytest.mark.parametrize('filename,pair,timeframe,candletype', [
    ('XMR_BTC-5m.json', 'XMR_BTC', '5m', ''),
    ('XMR_USDT-1h.h5', 'XMR_USDT', '1h', ''),
    ('BTC-PERP-1h.h5', 'BTC-PERP', '1h', ''),
    ('BTC_USDT-2h.jsongz', 'BTC_USDT', '2h', ''),
    ('BTC_USDT-2h-mark.jsongz', 'BTC_USDT', '2h', 'mark'),
    ('XMR_USDT-1h-mark.h5', 'XMR_USDT', '1h', 'mark'),
    ('XMR_USDT-1h-random.h5', 'XMR_USDT', '1h', 'random'),
    ('BTC-PERP-1h-index.h5', 'BTC-PERP', '1h', 'index'),
    ('XMR_USDT_USDT-1h-mark.h5', 'XMR_USDT_USDT', '1h', 'mark'),
])
def test_datahandler_ohlcv_regex(filename, pair, timeframe, candletype):
    regex = JsonDataHandler._OHLCV_REGEX

    match = re.search(regex, filename)
    assert len(match.groups()) > 1
    assert match[1] == pair
    assert match[2] == timeframe
    assert match[3] == candletype


@pytest.mark.parametrize('input,expected', [
    ('XMR_USDT', 'XMR/USDT'),
    ('BTC_USDT', 'BTC/USDT'),
    ('USDT_BUSD', 'USDT/BUSD'),
    ('BTC_USDT_USDT', 'BTC/USDT:USDT'),  # Futures
    ('XRP_USDT_USDT', 'XRP/USDT:USDT'),  # futures
    ('BTC-PERP', 'BTC-PERP'),
    ('BTC-PERP_USDT', 'BTC-PERP:USDT'),  # potential FTX case
    ('UNITTEST_USDT', 'UNITTEST/USDT'),
])
def test_rebuild_pair_from_filename(input, expected):

    assert IDataHandler.rebuild_pair_from_filename(input) == expected


def test_datahandler_ohlcv_get_available_data(testdatadir):
    paircombs = JsonDataHandler.ohlcv_get_available_data(testdatadir, TradingMode.SPOT)
    # Convert to set to avoid failures due to sorting
    assert set(paircombs) == {
        ('UNITTEST/BTC', '5m', CandleType.SPOT),
        ('ETH/BTC', '5m', CandleType.SPOT),
        ('XLM/BTC', '5m', CandleType.SPOT),
        ('TRX/BTC', '5m', CandleType.SPOT),
        ('LTC/BTC', '5m', CandleType.SPOT),
        ('XMR/BTC', '5m', CandleType.SPOT),
        ('ZEC/BTC', '5m', CandleType.SPOT),
        ('UNITTEST/BTC', '1m', CandleType.SPOT),
        ('ADA/BTC', '5m', CandleType.SPOT),
        ('ETC/BTC', '5m', CandleType.SPOT),
        ('NXT/BTC', '5m', CandleType.SPOT),
        ('DASH/BTC', '5m', CandleType.SPOT),
        ('XRP/ETH', '1m', CandleType.SPOT),
        ('XRP/ETH', '5m', CandleType.SPOT),
        ('UNITTEST/BTC', '30m', CandleType.SPOT),
        ('UNITTEST/BTC', '8m', CandleType.SPOT),
        ('NOPAIR/XXX', '4m', CandleType.SPOT),
    }

    paircombs = JsonDataHandler.ohlcv_get_available_data(testdatadir, TradingMode.FUTURES)
    # Convert to set to avoid failures due to sorting
    assert set(paircombs) == {
        ('UNITTEST/USDT', '1h', 'mark'),
        ('XRP/USDT', '1h', 'futures'),
        ('XRP/USDT', '1h', 'mark'),
        ('XRP/USDT', '8h', 'mark'),
        ('XRP/USDT', '8h', 'funding_rate'),
    }

    paircombs = JsonGzDataHandler.ohlcv_get_available_data(testdatadir, TradingMode.SPOT)
    assert set(paircombs) == {('UNITTEST/BTC', '8m', CandleType.SPOT)}
    paircombs = HDF5DataHandler.ohlcv_get_available_data(testdatadir, TradingMode.SPOT)
    assert set(paircombs) == {('UNITTEST/BTC', '5m', CandleType.SPOT)}


def test_jsondatahandler_trades_get_pairs(testdatadir):
    pairs = JsonGzDataHandler.trades_get_pairs(testdatadir)
    # Convert to set to avoid failures due to sorting
    assert set(pairs) == {'XRP/ETH', 'XRP/OLD'}


def test_jsondatahandler_ohlcv_purge(mocker, testdatadir):
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))
    unlinkmock = mocker.patch.object(Path, "unlink", MagicMock())
    dh = JsonGzDataHandler(testdatadir)
    assert not dh.ohlcv_purge('UNITTEST/NONEXIST', '5m', '')
    assert not dh.ohlcv_purge('UNITTEST/NONEXIST', '5m', candle_type='mark')
    assert unlinkmock.call_count == 0

    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    assert dh.ohlcv_purge('UNITTEST/NONEXIST', '5m', '')
    assert dh.ohlcv_purge('UNITTEST/NONEXIST', '5m', candle_type='mark')
    assert unlinkmock.call_count == 2


def test_jsondatahandler_ohlcv_load(testdatadir, caplog):
    dh = JsonDataHandler(testdatadir)
    df = dh.ohlcv_load('XRP/ETH', '5m', 'spot')
    assert len(df) == 711

    df_mark = dh.ohlcv_load('UNITTEST/USDT', '1h', candle_type="mark")
    assert len(df_mark) == 99

    df_no_mark = dh.ohlcv_load('UNITTEST/USDT', '1h', 'spot')
    assert len(df_no_mark) == 0

    # Failure case (empty array)
    df1 = dh.ohlcv_load('NOPAIR/XXX', '4m', 'spot')
    assert len(df1) == 0
    assert log_has("Could not load data for NOPAIR/XXX.", caplog)
    assert df.columns.equals(df1.columns)


def test_jsondatahandler_trades_load(testdatadir, caplog):
    dh = JsonGzDataHandler(testdatadir)
    logmsg = "Old trades format detected - converting"
    dh.trades_load('XRP/ETH')
    assert not log_has(logmsg, caplog)

    # Test conversation is happening
    dh.trades_load('XRP/OLD')
    assert log_has(logmsg, caplog)


def test_jsondatahandler_trades_purge(mocker, testdatadir):
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))
    unlinkmock = mocker.patch.object(Path, "unlink", MagicMock())
    dh = JsonGzDataHandler(testdatadir)
    assert not dh.trades_purge('UNITTEST/NONEXIST')
    assert unlinkmock.call_count == 0

    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    assert dh.trades_purge('UNITTEST/NONEXIST')
    assert unlinkmock.call_count == 1


@pytest.mark.parametrize('datahandler', AVAILABLE_DATAHANDLERS)
def test_datahandler_ohlcv_append(datahandler, testdatadir, ):
    dh = get_datahandler(testdatadir, datahandler)
    with pytest.raises(NotImplementedError):
        dh.ohlcv_append('UNITTEST/ETH', '5m', DataFrame(), CandleType.SPOT)
    with pytest.raises(NotImplementedError):
        dh.ohlcv_append('UNITTEST/ETH', '5m', DataFrame(), CandleType.MARK)


@pytest.mark.parametrize('datahandler', AVAILABLE_DATAHANDLERS)
def test_datahandler_trades_append(datahandler, testdatadir):
    dh = get_datahandler(testdatadir, datahandler)
    with pytest.raises(NotImplementedError):
        dh.trades_append('UNITTEST/ETH', [])


def test_hdf5datahandler_trades_get_pairs(testdatadir):
    pairs = HDF5DataHandler.trades_get_pairs(testdatadir)
    # Convert to set to avoid failures due to sorting
    assert set(pairs) == {'XRP/ETH'}


def test_hdf5datahandler_trades_load(testdatadir):
    dh = HDF5DataHandler(testdatadir)
    trades = dh.trades_load('XRP/ETH')
    assert isinstance(trades, list)

    trades1 = dh.trades_load('UNITTEST/NONEXIST')
    assert trades1 == []
    # data goes from 2019-10-11 - 2019-10-13
    timerange = TimeRange.parse_timerange('20191011-20191012')

    trades2 = dh._trades_load('XRP/ETH', timerange)
    assert len(trades) > len(trades2)
    # Check that ID is None (If it's nan, it's wrong)
    assert trades2[0][2] is None

    # unfiltered load has trades before starttime
    assert len([t for t in trades if t[0] < timerange.startts * 1000]) >= 0
    # filtered list does not have trades before starttime
    assert len([t for t in trades2 if t[0] < timerange.startts * 1000]) == 0
    # unfiltered load has trades after endtime
    assert len([t for t in trades if t[0] > timerange.stopts * 1000]) > 0
    # filtered list does not have trades after endtime
    assert len([t for t in trades2 if t[0] > timerange.stopts * 1000]) == 0


def test_hdf5datahandler_trades_store(testdatadir, tmpdir):
    tmpdir1 = Path(tmpdir)
    dh = HDF5DataHandler(testdatadir)
    trades = dh.trades_load('XRP/ETH')

    dh1 = HDF5DataHandler(tmpdir1)
    dh1.trades_store('XRP/NEW', trades)
    file = tmpdir1 / 'XRP_NEW-trades.h5'
    assert file.is_file()
    # Load trades back
    trades_new = dh1.trades_load('XRP/NEW')

    assert len(trades_new) == len(trades)
    assert trades[0][0] == trades_new[0][0]
    assert trades[0][1] == trades_new[0][1]
    # assert trades[0][2] == trades_new[0][2]  # This is nan - so comparison does not make sense
    assert trades[0][3] == trades_new[0][3]
    assert trades[0][4] == trades_new[0][4]
    assert trades[0][5] == trades_new[0][5]
    assert trades[0][6] == trades_new[0][6]
    assert trades[-1][0] == trades_new[-1][0]
    assert trades[-1][1] == trades_new[-1][1]
    # assert trades[-1][2] == trades_new[-1][2]  # This is nan - so comparison does not make sense
    assert trades[-1][3] == trades_new[-1][3]
    assert trades[-1][4] == trades_new[-1][4]
    assert trades[-1][5] == trades_new[-1][5]
    assert trades[-1][6] == trades_new[-1][6]


def test_hdf5datahandler_trades_purge(mocker, testdatadir):
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))
    unlinkmock = mocker.patch.object(Path, "unlink", MagicMock())
    dh = HDF5DataHandler(testdatadir)
    assert not dh.trades_purge('UNITTEST/NONEXIST')
    assert unlinkmock.call_count == 0

    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    assert dh.trades_purge('UNITTEST/NONEXIST')
    assert unlinkmock.call_count == 1


@pytest.mark.parametrize('pair,timeframe,candle_type,candle_append,startdt,enddt', [
    # Data goes from 2018-01-10 - 2018-01-30
    ('UNITTEST/BTC', '5m', 'spot',  '', '2018-01-15', '2018-01-19'),
    # Mark data goes from to 2021-11-15 2021-11-19
    ('UNITTEST/USDT:USDT', '1h', 'mark', '-mark', '2021-11-16', '2021-11-18'),
])
def test_hdf5datahandler_ohlcv_load_and_resave(
    testdatadir,
    tmpdir,
    pair,
    timeframe,
    candle_type,
    candle_append,
    startdt, enddt
):
    tmpdir1 = Path(tmpdir)
    tmpdir2 = tmpdir1
    if candle_type not in ('', 'spot'):
        tmpdir2 = tmpdir1 / 'futures'
        tmpdir2.mkdir()
    dh = HDF5DataHandler(testdatadir)
    ohlcv = dh._ohlcv_load(pair, timeframe, None, candle_type=candle_type)
    assert isinstance(ohlcv, DataFrame)
    assert len(ohlcv) > 0

    file = tmpdir2 / f"UNITTEST_NEW-{timeframe}{candle_append}.h5"
    assert not file.is_file()

    dh1 = HDF5DataHandler(tmpdir1)
    dh1.ohlcv_store('UNITTEST/NEW', timeframe, ohlcv, candle_type=candle_type)
    assert file.is_file()

    assert not ohlcv[ohlcv['date'] < startdt].empty

    timerange = TimeRange.parse_timerange(f"{startdt.replace('-', '')}-{enddt.replace('-', '')}")

    # Call private function to ensure timerange is filtered in hdf5
    ohlcv = dh._ohlcv_load(pair, timeframe, timerange, candle_type=candle_type)
    ohlcv1 = dh1._ohlcv_load('UNITTEST/NEW', timeframe, timerange, candle_type=candle_type)
    assert len(ohlcv) == len(ohlcv1)
    assert ohlcv.equals(ohlcv1)
    assert ohlcv[ohlcv['date'] < startdt].empty
    assert ohlcv[ohlcv['date'] > enddt].empty

    # Try loading inexisting file
    ohlcv = dh.ohlcv_load('UNITTEST/NONEXIST', timeframe, candle_type=candle_type)
    assert ohlcv.empty


def test_hdf5datahandler_ohlcv_purge(mocker, testdatadir):
    mocker.patch.object(Path, "exists", MagicMock(return_value=False))
    unlinkmock = mocker.patch.object(Path, "unlink", MagicMock())
    dh = HDF5DataHandler(testdatadir)
    assert not dh.ohlcv_purge('UNITTEST/NONEXIST', '5m', '')
    assert not dh.ohlcv_purge('UNITTEST/NONEXIST', '5m', candle_type='mark')
    assert unlinkmock.call_count == 0

    mocker.patch.object(Path, "exists", MagicMock(return_value=True))
    assert dh.ohlcv_purge('UNITTEST/NONEXIST', '5m', '')
    assert dh.ohlcv_purge('UNITTEST/NONEXIST', '5m', candle_type='mark')
    assert unlinkmock.call_count == 2


def test_gethandlerclass():
    cl = get_datahandlerclass('json')
    assert cl == JsonDataHandler
    assert issubclass(cl, IDataHandler)
    cl = get_datahandlerclass('jsongz')
    assert cl == JsonGzDataHandler
    assert issubclass(cl, IDataHandler)
    assert issubclass(cl, JsonDataHandler)
    cl = get_datahandlerclass('hdf5')
    assert cl == HDF5DataHandler
    assert issubclass(cl, IDataHandler)
    with pytest.raises(ValueError, match=r"No datahandler for .*"):
        get_datahandlerclass('DeadBeef')


def test_get_datahandler(testdatadir):
    dh = get_datahandler(testdatadir, 'json')
    assert type(dh) == JsonDataHandler
    dh = get_datahandler(testdatadir, 'jsongz')
    assert type(dh) == JsonGzDataHandler
    dh1 = get_datahandler(testdatadir, 'jsongz', dh)
    assert id(dh1) == id(dh)

    dh = get_datahandler(testdatadir, 'hdf5')
    assert type(dh) == HDF5DataHandler
