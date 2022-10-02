"""
External Pair List provider

Provides pair list from Leader data
"""
import logging
from typing import Any, Dict, List, Optional

from freqtrade.exceptions import OperationalException
from freqtrade.plugins.pairlist.IPairList import IPairList


logger = logging.getLogger(__name__)


class ProducerPairList(IPairList):
    """
    PairList plugin for use with external_message_consumer.
    Will use pairs given from leader data.

    Usage:
        "pairlists": [
            {
                "method": "ProducerPairList",
                "number_assets": 5,
                "producer_name": "default",
            }
        ],
    """

    def __init__(self, exchange, pairlistmanager,
                 config: Dict[str, Any], pairlistconfig: Dict[str, Any],
                 pairlist_pos: int) -> None:
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

        self._num_assets: int = self._pairlistconfig.get('number_assets', 0)
        self._producer_name = self._pairlistconfig.get('producer_name', 'default')
        if not config.get('external_message_consumer', {}).get('enabled'):
            raise OperationalException(
                "ProducerPairList requires external_message_consumer to be enabled.")

    @property
    def needstickers(self) -> bool:
        """
        Boolean property defining if tickers are necessary.
        If no Pairlist requires tickers, an empty Dict is passed
        as tickers argument to filter_pairlist
        """
        return False

    def short_desc(self) -> str:
        """
        Short whitelist method description - used for startup-messages
        -> Please overwrite in subclasses
        """
        return f"{self.name} - {self._producer_name}"

    def _filter_pairlist(self, pairlist: Optional[List[str]]):
        upstream_pairlist = self._pairlistmanager._dataprovider.get_producer_pairs(
            self._producer_name)

        if pairlist is None:
            pairlist = self._pairlistmanager._dataprovider.get_producer_pairs(self._producer_name)

        pairs = list(dict.fromkeys(pairlist + upstream_pairlist))
        if self._num_assets:
            pairs = pairs[:self._num_assets]

        return pairs

    def gen_pairlist(self, tickers: Dict) -> List[str]:
        """
        Generate the pairlist
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: List of pairs
        """
        pairs = self._filter_pairlist(None)
        self.log_once(f"Received pairs: {pairs}", logger.debug)
        pairs = self._whitelist_for_active_markets(self.verify_whitelist(pairs, logger.info))
        return pairs

    def filter_pairlist(self, pairlist: List[str], tickers: Dict) -> List[str]:
        """
        Filters and sorts pairlist and returns the whitelist again.
        Called on each bot iteration - please use internal caching if necessary
        :param pairlist: pairlist to filter or sort
        :param tickers: Tickers (from exchange.get_tickers()). May be cached.
        :return: new whitelist
        """
        return self._filter_pairlist(pairlist)
