""" Binance exchange subclass """
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import arrow
import ccxt

from freqtrade.enums import CandleType, MarginMode, TradingMode
from freqtrade.exceptions import DDosProtection, OperationalException, TemporaryError
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier
from freqtrade.misc import deep_merge_dicts, json_load


logger = logging.getLogger(__name__)


class Binance(Exchange):

    _ft_has: Dict = {
        "stoploss_on_exchange": True,
        "stoploss_order_types": {"limit": "stop_loss_limit"},
        "order_time_in_force": ['GTC', 'FOK', 'IOC'],
        "ohlcv_candle_limit": 1000,
        "trades_pagination": "id",
        "trades_pagination_arg": "fromId",
        "l2_limit_range": [5, 10, 20, 50, 100, 500, 1000],
        "ccxt_futures_name": "future"
    }
    _ft_has_futures: Dict = {
        "stoploss_order_types": {"limit": "limit", "market": "market"},
        "tickers_have_price": False,
    }

    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = [
        # TradingMode.SPOT always supported and not required in this list
        # (TradingMode.MARGIN, MarginMode.CROSS),
        # (TradingMode.FUTURES, MarginMode.CROSS),
        (TradingMode.FUTURES, MarginMode.ISOLATED)
    ]

    def stoploss_adjust(self, stop_loss: float, order: Dict, side: str) -> bool:
        """
        Verify stop_loss against stoploss-order value (limit or price)
        Returns True if adjustment is necessary.
        :param side: "buy" or "sell"
        """
        order_types = ('stop_loss_limit', 'stop', 'stop_market')

        return (
            order.get('stopPrice', None) is None
            or (
                order['type'] in order_types
                and (
                    (side == "sell" and stop_loss > float(order['stopPrice'])) or
                    (side == "buy" and stop_loss < float(order['stopPrice']))
                )
            ))

    def get_tickers(self, symbols: Optional[List[str]] = None, cached: bool = False) -> Dict:
        tickers = super().get_tickers(symbols=symbols, cached=cached)
        if self.trading_mode == TradingMode.FUTURES:
            # Binance's future result has no bid/ask values.
            # Therefore we must fetch that from fetch_bids_asks and combine the two results.
            bidsasks = self.fetch_bids_asks(symbols, cached)
            tickers = deep_merge_dicts(bidsasks, tickers, allow_null_overrides=False)
        return tickers

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
        trading_mode = trading_mode or self.trading_mode

        if self._config['dry_run'] or trading_mode != TradingMode.FUTURES:
            return

        try:
            self._api.set_leverage(symbol=pair, leverage=round(leverage))
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not set leverage due to {e.__class__.__name__}. Message: {e}') from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    async def _async_get_historic_ohlcv(self, pair: str, timeframe: str,
                                        since_ms: int, candle_type: CandleType,
                                        is_new_pair: bool = False, raise_: bool = False,
                                        until_ms: Optional[int] = None
                                        ) -> Tuple[str, str, str, List]:
        """
        Overwrite to introduce "fast new pair" functionality by detecting the pair's listing date
        Does not work for other exchanges, which don't return the earliest data when called with "0"
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        """
        if is_new_pair:
            x = await self._async_get_candle_history(pair, timeframe, candle_type, 0)
            if x and x[3] and x[3][0] and x[3][0][0] > since_ms:
                # Set starting date to first available candle.
                since_ms = x[3][0][0]
                logger.info(f"Candle-data for {pair} available starting with "
                            f"{arrow.get(since_ms // 1000).isoformat()}.")

        return await super()._async_get_historic_ohlcv(
            pair=pair,
            timeframe=timeframe,
            since_ms=since_ms,
            is_new_pair=is_new_pair,
            raise_=raise_,
            candle_type=candle_type,
            until_ms=until_ms,
        )

    def funding_fee_cutoff(self, open_date: datetime):
        """
        :param open_date: The open date for a trade
        :return: The cutoff open time for when a funding fee is charged
        """
        return open_date.minute > 0 or (open_date.minute == 0 and open_date.second > 15)

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
        MARGIN: https://www.binance.com/en/support/faq/f6b010588e55413aa58b7d63ee0125ed
        PERPETUAL: https://www.binance.com/en/support/faq/b3c689c1f50a44cabb3a84e663b81d93

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

        # * Only required for Cross
        :param mm_ex_1: (TMM)
            Cross-Margin Mode: Maintenance Margin of all other contracts, excluding Contract 1
            Isolated-Margin Mode: 0
        :param upnl_ex_1: (UPNL)
            Cross-Margin Mode: Unrealized PNL of all other contracts, excluding Contract 1.
            Isolated-Margin Mode: 0
        """

        side_1 = -1 if is_short else 1
        cross_vars = upnl_ex_1 - mm_ex_1 if self.margin_mode == MarginMode.CROSS else 0.0

        # mm_ratio: Binance's formula specifies maintenance margin rate which is mm_ratio * 100%
        # maintenance_amt: (CUM) Maintenance Amount of position
        mm_ratio, maintenance_amt = self.get_maintenance_ratio_and_amt(pair, stake_amount)

        if (maintenance_amt is None):
            raise OperationalException(
                "Parameter maintenance_amt is required by Binance.liquidation_price"
                f"for {self.trading_mode.value}"
            )

        if self.trading_mode == TradingMode.FUTURES:
            return (
                (
                    (wallet_balance + cross_vars + maintenance_amt) -
                    (side_1 * amount * open_rate)
                ) / (
                    (amount * mm_ratio) - (side_1 * amount)
                )
            )
        else:
            raise OperationalException(
                "Freqtrade only supports isolated futures for leverage trading")

    @retrier
    def load_leverage_tiers(self) -> Dict[str, List[Dict]]:
        if self.trading_mode == TradingMode.FUTURES:
            if self._config['dry_run']:
                leverage_tiers_path = (
                    Path(__file__).parent / 'binance_leverage_tiers.json'
                )
                with open(leverage_tiers_path) as json_file:
                    return json_load(json_file)
            else:
                try:
                    return self._api.fetch_leverage_tiers()
                except ccxt.DDoSProtection as e:
                    raise DDosProtection(e) from e
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    raise TemporaryError(f'Could not fetch leverage amounts due to'
                                         f'{e.__class__.__name__}. Message: {e}') from e
                except ccxt.BaseError as e:
                    raise OperationalException(e) from e
        else:
            return {}
