"""
Spread pair list filter
"""
import logging
from typing import Any, Dict

from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.plugins.pairlist.IPairList import IPairList


logger = logging.getLogger(__name__)


class SpreadFilter(IPairList):

    def __init__(self, exchange, pairlistmanager,
                 config: Config, pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        self._max_spread_ratio = pairlistconfig.get('max_spread_ratio', 0.005)
        self._enabled = self._max_spread_ratio != 0

        if not self._exchange.exchange_has('fetchTickers'):
            raise OperationalException(
                'Exchange does not support fetchTickers, therefore SpreadFilter cannot be used.'
                'Please edit your config and restart the bot.'
            )

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requires tickers, an empty Dict is passed
        as tickers argument to filter_pairlist
        """
        return True

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        """
        return (f"{self.name} - Filtering pairs with ask/bid diff above "
                f"{self._max_spread_ratio:.2%}.")

    def _validate_pair(self, pair: str, ticker: Dict[str, Any]) -> bool:
        """
        Validate spread for the ticker
        :param pair: Pair that's currently validated
        :param ticker: ticker dict as returned from ccxt.fetch_tickers()
        :return: True if the pair can stay, false if it should be removed
        """
        if 'bid' in ticker and 'ask' in ticker and ticker['ask'] and ticker['bid']:
            spread = 1 - ticker['bid'] / ticker['ask']
            if spread > self._max_spread_ratio:
                self.log_once(f"Removed {pair} from whitelist, because spread "
                              f"{spread:.3%} > {self._max_spread_ratio:.3%}",
                              logger.info)
                return False
            else:
                return True
        self.log_once(f"Removed {pair} from whitelist due to invalid ticker data: {ticker}",
                      logger.info)
        return False
