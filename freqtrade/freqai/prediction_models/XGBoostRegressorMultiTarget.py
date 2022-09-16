import logging
from typing import Any, Dict

from xgboost import XGBRegressor

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.base_models.FreqaiMultiOutputRegressor import FreqaiMultiOutputRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


logger = logging.getLogger(__name__)


class XGBoostRegressorMultiTarget(BaseRegressionModel):
    """
    User created prediction model. The class needs to override three necessary
    functions, predict(), train(), fit(). The class inherits ModelHandler which
    has its own DataHandler where data is held, saved, loaded, and managed.
    """

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary constructed by DataHandler to hold
                                all the training and test data/labels.
        """

        xgb = XGBRegressor(**self.model_training_parameters)

        X = data_dictionary["train_features"]
        y = data_dictionary["train_labels"]
        sample_weight = data_dictionary["train_weights"]

        eval_weights = None
        eval_sets = [None] * y.shape[1]

        if self.freqai_info.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            eval_weights = [data_dictionary["test_weights"]]
            for i in range(data_dictionary['test_labels'].shape[1]):
                eval_sets[i] = [(  # type: ignore
                    data_dictionary["test_features"],
                    data_dictionary["test_labels"].iloc[:, i]
                )]

        init_model = self.get_init_model(dk.pair)
        if init_model:
            init_models = init_model.estimators_
        else:
            init_models = [None] * y.shape[1]

        fit_params = []
        for i in range(len(eval_sets)):
            fit_params.append(
                {'eval_set': eval_sets[i], 'sample_weight_eval_set': eval_weights,
                 'xgb_model': init_models[i]})

        model = FreqaiMultiOutputRegressor(estimator=xgb)
        thread_training = self.freqai_info.get('multitarget_parallel_training', False)
        if thread_training:
            model.n_jobs = y.shape[1]
        model.fit(X=X, y=y, sample_weight=sample_weight, fit_params=fit_params)

        return model
