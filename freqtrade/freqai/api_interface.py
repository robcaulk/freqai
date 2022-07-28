import requests
from pandas import DataFrame
import pandas as pd
from freqtrade.freqai.data_drawer import FreqaiDataDrawer
import numpy as np
from typing import Callable


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
    def __init__(self, config: dict, data_drawer: FreqaiDataDrawer, payload_func: Callable):

        self.config = config
        self.freqai_config = config.get('freqai', {})
        self.api_token = self.freqai_config.get('freqai_api_token')
        self.api_base_url = self.freqai_config.get('freqai_api_url')
        self.post_url = f"{self.api_base_url}pairs"
        self.dd = data_drawer
        self.create_api_payload = payload_func
        self.headers = {
            "Authorization": self.api_token,
            "Content-Type": "application/json"
        }

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

        response = requests.request("GET", get_url, headers=self.headers)

        payload = self.create_api_payload(dataframe, pair)

        if response.json()['data'] is None:
            requests.request("POST", self.post_url, json=payload, headers=self.headers)
        else:
            requests.request("PATCH", self.post_url, json=payload, headers=self.headers)

    def fetch_prediction_from_api(self, pair: str) -> dict:
        subpair = pair.split('/')
        pair = f"{subpair[0]}{subpair[1]}"

        get_url = f"{self.post_url}/{pair}"

        return requests.request("GET", get_url, headers=self.headers).json()['data']

    def start_fetching_from_api(self, dataframe: DataFrame, pair: str) -> None:
        """
        FreqAI "getter" instance will first create a full dataframe of constant
        values based on the first fetched prediction. Afterwards, it will append
        single predictions to the return dataframe.
        """
        response_dict = self.fetch_prediction_from_api(pair)

        if pair not in self.dd.model_return_values:
            self.set_initial_return_values(pair, response_dict, len(dataframe.index))
        else:
            self.append_model_predictions_from_api(pair, response_dict, len(dataframe.index))

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
