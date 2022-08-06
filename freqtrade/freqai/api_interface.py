import logging
import time
from typing import Any, Callable, Dict

# import datetime
import dateutil.parser
import numpy as np
import pandas as pd
import requests
from pandas import DataFrame
from freqtrade.exceptions import OperationalException

from freqtrade.freqai.data_drawer import FreqaiDataDrawer


logger = logging.getLogger(__name__)


class FreqaiAPI:
    """
    Class designed to enable FreqAI "poster" instances to share predictions via API with other
    FreqAI "getter" instances.
    :param: config: dict = user provided config containing api token and url information
    :param: data_drawer: FreqaiDataDrawer = persistent data storage associated with current
    FreqAI instance.
    :param: payload_fun: Callable = User defined schema for the "poster" FreqAI instance.
    Defined in the IFreqaiModel (inherited prediction model class such as CatboostPredictionModel)
    """

    def __init__(self, config: dict, data_drawer: FreqaiDataDrawer,
                 payload_func: Callable, mode: str):

        self.config = config
        self.freqai_config = config.get('freqai', {})
        self.api_token = self.freqai_config.get('freqai_api_token')
        self.api_base_url = self.freqai_config.get('freqai_api_url')
        self.post_url = f"{self.api_base_url}pairs"
        self.dd = data_drawer
        self.create_api_payload = payload_func
        if mode == 'getter':
            self.headers = {
                "X-BLOBR-KEY": self.api_token,
                "Content-Type": "application/json"
            }
        else:
            self.headers = {
                "Authorization": self.api_token,
                "Content-Type": "application/json"
            }
        self.api_dict: Dict[str, Any] = {}
        self.num_posts = 0

    def start_fetching_from_api(self, dataframe: DataFrame, pair: str) -> DataFrame:

        fetch_new = self.check_if_new_fetch_required(dataframe, pair)
        if fetch_new:
            response = self.fetch_all_pairs_from_api(dataframe)
            if not response:
                self.create_null_api_dict(pair)
            else:
                self.parse_response(response)
        self.make_return_dataframe(dataframe, pair)
        return self.dd.attach_return_values_to_return_dataframe(pair, dataframe)

    def post_predictions(self, dataframe: DataFrame, pair: str) -> None:
        """
        FreqAI "poster" instance will call this function to post predictions
        to an API. API schema is user defined in the IFreqaiModel.create_api_payload().
        API schema is flexible but must follow standardized method where
        f"{self.post_url}/{pair}" retrieves the predictions for current candle of
        the specified pair. Additionally, the schema must contain "returns"
        which defines the return strings expected by the getter.
        """
        subpair = pair.split('/')
        pair = f"{subpair[0]}{subpair[1]}"

        get_url = f"{self.post_url}/{pair}"

        payload = self.create_api_payload(dataframe, pair)

        if self.num_posts < len(self.config['exchange']['pair_whitelist']):
            response = requests.request("GET", get_url, headers=self.headers)
            self.num_posts += 1
            if response.json()['data'] is None:
                requests.request("POST", self.post_url, json=payload, headers=self.headers)
            else:
                requests.request("PATCH", get_url, json=payload, headers=self.headers)
        else:
            requests.request("PATCH", get_url, json=payload, headers=self.headers)

    def check_if_new_fetch_required(self, dataframe: DataFrame, pair: str) -> bool:

        if not self.api_dict:
            return True

        subpair = pair.split('/')
        coin = f"{subpair[0]}{subpair[1]}"
        candle_date = dataframe['date'].iloc[-1]
        ts_candle = candle_date.timestamp()
        ts_dict = dateutil.parser.parse(self.api_dict[coin]['updatedAt']).timestamp()

        if ts_dict < ts_candle:
            logger.info('Local dictionary outdated, fetching new predictions from API')
            return True
        else:
            return False

    def fetch_all_pairs_from_api(self, dataframe: DataFrame) -> dict:

        candle_date = dataframe['date'].iloc[-1]
        get_url = f"{self.post_url}"
        n_tries = 0
        ts_candle = candle_date.timestamp()
        ts_pair = ts_candle - 1
        ts_pair_oldest = int(ts_candle)

        while 1:
            response = requests.request("GET", get_url, headers=self.headers).json()['data']
            for pair in response:
                ts_pair = dateutil.parser.parse(pair['updatedAt']).timestamp()
                if ts_pair < ts_pair_oldest:
                    ts_pair_oldest = ts_pair
                    outdated_pair = pair['name']
            if ts_pair_oldest < ts_candle:
                logger.warning(
                    f'{outdated_pair} is not uptodate, waiting on API db to update before'
                    ' retrying.')
                n_tries += 1
                if n_tries > 5:
                    logger.warning(
                        'Tried to fetch updated DB 5 times with no success. Returning null values'
                        ' back to strategy')
                    return {}
                time.sleep(5)
            else:
                logger.info('Successfully fetched updated DB')
                break

        return response

    def parse_response(self, response_dict: dict) -> None:

        for coin_pred in response_dict:
            coin = coin_pred['name']
            self.api_dict[coin] = coin_pred   # {}
            # for return_str in coin_pred['returns']:
            #     coin_dict[coin][return_str] = coin_pred[return_str]

    def make_return_dataframe(self, dataframe: DataFrame, pair: str) -> None:

        subpair = pair.split('/')
        coin = f"{subpair[0]}{subpair[1]}"

        if coin not in self.api_dict:
            raise OperationalException(
                'Getter is looking for a coin that is not available at this API. '
                'Ensure whitelist only contains available coins.')

        if pair not in self.dd.model_return_values:
            self.set_initial_return_values(pair, self.api_dict[coin], len(dataframe.index))
        else:
            self.append_model_predictions_from_api(pair, self.api_dict[coin], len(dataframe.index))

    def set_initial_return_values(self, pair: str, response_dict: dict, len_df: int) -> None:
        """
        Set the initial return values to a persistent dataframe so that the getter only needs
        to retrieve a single data point per candle.
        """
        mrv_df = self.dd.model_return_values[pair] = DataFrame()

        for expected_str in response_dict['returns']:
            return_str = expected_str['name']
            mrv_df[return_str] = np.ones(len_df) * response_dict[return_str]

    def append_model_predictions_from_api(self, pair: str,
                                          response_dict: dict, len_df: int) -> None:
        """
        Function to append the api retrieved predictions to the return dataframe, but
        also detects if return dataframe should change size. This enables historical
        predictions to be viewable in FreqUI.
        """

        length_difference = len(self.dd.model_return_values[pair]) - len_df
        i = 0

        if length_difference == 0:
            i = 1
        elif length_difference > 0:
            i = length_difference + 1

        mrv_df = self.dd.model_return_values[pair] = self.dd.model_return_values[pair].shift(-i)

        for expected_str in response_dict['returns']:
            return_str = expected_str['name']
            mrv_df[return_str].iloc[-1] = response_dict[return_str]

        if length_difference < 0:
            prepend_df = pd.DataFrame(
                np.zeros((abs(length_difference) - 1, len(mrv_df.columns))), columns=mrv_df.columns
            )
            mrv_df = pd.concat([prepend_df, mrv_df], axis=0)

    def create_null_api_dict(self, pair: str) -> None:
        """
        Set values in api_dict to 0 and return to user. This is only used in case the API is
        unresponsive, but  we still want FreqAI to return to the strategy to continue handling 
        open trades.
        """
        subpair = pair.split('/')
        pair = f"{subpair[0]}{subpair[1]}"

        for expected_str in self.api_dict[pair]['returns']:
            return_str = expected_str['name']
            self.api_dict[pair][return_str] = 0
