import copy
import datetime
import logging
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame
from sklearn import linear_model
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split

from freqtrade.configuration import TimeRange
from freqtrade.data.history.history_utils import refresh_backtest_ohlcv_data
from freqtrade.exceptions import OperationalException
from freqtrade.resolvers import ExchangeResolver
from freqtrade.strategy.interface import IStrategy


SECONDS_IN_DAY = 86400
SECONDS_IN_HOUR = 3600

logger = logging.getLogger(__name__)


class FreqaiDataKitchen:
    """
    Class designed to analyze data for a single pair. Employed by the IFreqaiModel class.
    Functionalities include holding, saving, loading, and analyzing the data.

    Record of contribution:
    FreqAI was developed by a group of individuals who all contributed specific skillsets to the
    project.

    Conception and software development:
    Robert Caulk @robcaulk

    Theoretical brainstorming:
    Elin Törnquist @th0rntwig

    Code review, software architecture brainstorming:
    @xmatthias

    Beta testing and bug reporting:
    @bloodhunter4rc, Salah Lamkadem @ikonx, @ken11o2, @longyu, @paranoidandy, @smidelis, @smarm
    Juha Nykänen @suikula, Wagner Costa @wagnercosta
    """

    def __init__(
        self,
        config: Dict[str, Any],
        live: bool = False,
        pair: str = "",
    ):
        self.data: Dict[Any, Any] = {}
        self.data_dictionary: Dict[Any, Any] = {}
        self.config = config
        self.freqai_config = config["freqai"]
        self.full_df: DataFrame = DataFrame()
        self.append_df: DataFrame = DataFrame()
        self.data_path = Path()
        self.label_list: List = []
        self.training_features_list: List = []
        self.model_filename: str = ""
        self.live = live
        self.pair = pair
        self.svm_model: linear_model.SGDOneClassSVM = None
        self.keras = self.freqai_config.get("keras", False)
        self.set_all_pairs()
        if not self.live:
            if not self.config["timerange"]:
                raise OperationalException(
                    'Please pass --timerange if you intend to use FreqAI for backtesting.')
            self.full_timerange = self.create_fulltimerange(
                self.config["timerange"], self.freqai_config.get("train_period_days")
            )

            (self.training_timeranges, self.backtesting_timeranges) = self.split_timerange(
                self.full_timerange,
                config["freqai"]["train_period_days"],
                config["freqai"]["backtest_period_days"],
            )

        if self.live:
            db_url = self.config.get('db_url', 'sqlite://')
            self.database_path = '' if db_url == 'sqlite://' else str(db_url).split('///')[1]
            self.trade_database_df: DataFrame = pd.DataFrame()

        self.data['extra_returns_per_train'] = self.freqai_config.get('extra_returns_per_train', {})

    def set_paths(
        self,
        pair: str,
        trained_timestamp: int = None,
    ) -> None:
        """
        Set the paths to the data for the present coin/botloop
        :params:
        metadata: dict = strategy furnished pair metadata
        trained_timestamp: int = timestamp of most recent training
        """
        self.full_path = Path(
            self.config["user_data_dir"] / "models" / str(self.freqai_config.get("identifier"))
        )

        self.data_path = Path(
            self.full_path
            / f"sub-train-{pair.split('/')[0]}_{trained_timestamp}"
        )

        return

    def make_train_test_datasets(
        self, filtered_dataframe: DataFrame, labels: DataFrame
    ) -> Dict[Any, Any]:
        """
        Given the dataframe for the full history for training, split the data into
        training and test data according to user specified parameters in configuration
        file.
        :filtered_dataframe: cleaned dataframe ready to be split.
        :labels: cleaned labels ready to be split.
        """
        feat_dict = self.freqai_config["feature_parameters"]

        weights: npt.ArrayLike
        if feat_dict.get("weight_factor", 0) > 0:
            weights = self.set_weights_higher_recent(len(filtered_dataframe))
        else:
            weights = np.ones(len(filtered_dataframe))

        if feat_dict.get("stratify_training_data", 0) > 0:
            stratification = np.zeros(len(filtered_dataframe))
            for i in range(1, len(stratification)):
                if i % feat_dict.get("stratify_training_data", 0) == 0:
                    stratification[i] = 1
        else:
            stratification = None

        if self.freqai_config.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            (
                train_features,
                test_features,
                train_labels,
                test_labels,
                train_weights,
                test_weights,
            ) = train_test_split(
                filtered_dataframe[: filtered_dataframe.shape[0]],
                labels,
                weights,
                stratify=stratification,
                **self.config["freqai"]["data_split_parameters"],
            )
        else:
            test_labels = np.zeros(2)
            test_features = pd.DataFrame()
            test_weights = np.zeros(2)
            train_features = filtered_dataframe
            train_labels = labels
            train_weights = weights

        return self.build_data_dictionary(
            train_features, test_features, train_labels, test_labels, train_weights, test_weights
        )

    def filter_features(
        self,
        unfiltered_dataframe: DataFrame,
        training_feature_list: List,
        label_list: List = list(),
        training_filter: bool = True,
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Filter the unfiltered dataframe to extract the user requested features/labels and properly
        remove all NaNs. Any row with a NaN is removed from training dataset or replaced with
        0s in the prediction dataset. However, prediction dataset do_predict will reflect any
        row that had a NaN and will shield user from that prediction.
        :params:
        :unfiltered_dataframe: the full dataframe for the present training period
        :training_feature_list: list, the training feature list constructed by
        self.build_feature_list() according to user specified parameters in the configuration file.
        :labels: the labels for the dataset
        :training_filter: boolean which lets the function know if it is training data or
        prediction data to be filtered.
        :returns:
        :filtered_dataframe: dataframe cleaned of NaNs and only containing the user
        requested feature set.
        :labels: labels cleaned of NaNs.
        """
        filtered_dataframe = unfiltered_dataframe.filter(training_feature_list, axis=1)
        filtered_dataframe = filtered_dataframe.replace([np.inf, -np.inf], np.nan)

        drop_index = pd.isnull(filtered_dataframe).any(1)  # get the rows that have NaNs,
        drop_index = drop_index.replace(True, 1).replace(False, 0)  # pep8 requirement.
        if (
            training_filter
        ):  # we don't care about total row number (total no. datapoints) in training, we only care
            # about removing any row with NaNs
            # if labels has multiple columns (user wants to train multiple models), we detect here
            labels = unfiltered_dataframe.filter(label_list, axis=1)
            drop_index_labels = pd.isnull(labels).any(1)
            drop_index_labels = drop_index_labels.replace(True, 1).replace(False, 0)
            filtered_dataframe = filtered_dataframe[
                (drop_index == 0) & (drop_index_labels == 0)
            ]  # dropping values
            labels = labels[
                (drop_index == 0) & (drop_index_labels == 0)
            ]  # assuming the labels depend entirely on the dataframe here.
            logger.info(
                f"dropped {len(unfiltered_dataframe) - len(filtered_dataframe)} training points"
                f" due to NaNs in populated dataset {len(unfiltered_dataframe)}."
            )
            if (1 - len(filtered_dataframe) / len(unfiltered_dataframe)) > 0.1 and self.live:
                worst_indicator = str(unfiltered_dataframe.count().idxmin())
                logger.warning(
                    f" {(1 - len(filtered_dataframe)/len(unfiltered_dataframe)) * 100:.0f} percent "
                    " of training data dropped due to NaNs, model may perform inconsistent "
                    f"with expectations. Verify {worst_indicator}"
                )
            self.data["filter_drop_index_training"] = drop_index

        else:
            # we are backtesting so we need to preserve row number to send back to strategy,
            # so now we use do_predict to avoid any prediction based on a NaN
            drop_index = pd.isnull(filtered_dataframe).any(1)
            self.data["filter_drop_index_prediction"] = drop_index
            filtered_dataframe.fillna(0, inplace=True)
            # replacing all NaNs with zeros to avoid issues in 'prediction', but any prediction
            # that was based on a single NaN is ultimately protected from buys with do_predict
            drop_index = ~drop_index
            self.do_predict = np.array(drop_index.replace(True, 1).replace(False, 0))
            if (len(self.do_predict) - self.do_predict.sum()) > 0:
                logger.info(
                    "dropped %s of %s prediction data points due to NaNs.",
                    len(self.do_predict) - self.do_predict.sum(),
                    len(filtered_dataframe),
                )
            labels = []

        return filtered_dataframe, labels

    def build_data_dictionary(
        self,
        train_df: DataFrame,
        test_df: DataFrame,
        train_labels: DataFrame,
        test_labels: DataFrame,
        train_weights: Any,
        test_weights: Any,
    ) -> Dict:

        self.data_dictionary = {
            "train_features": train_df,
            "test_features": test_df,
            "train_labels": train_labels,
            "test_labels": test_labels,
            "train_weights": train_weights,
            "test_weights": test_weights,
        }

        return self.data_dictionary

    def normalize_data(self, data_dictionary: Dict) -> Dict[Any, Any]:
        """
        Normalize all data in the data_dictionary according to the training dataset
        :params:
        :data_dictionary: dictionary containing the cleaned and split training/test data/labels
        :returns:
        :data_dictionary: updated dictionary with standardized values.
        """
        # standardize the data by training stats
        train_max = data_dictionary["train_features"].max()
        train_min = data_dictionary["train_features"].min()
        data_dictionary["train_features"] = (
            2 * (data_dictionary["train_features"] - train_min) / (train_max - train_min) - 1
        )
        data_dictionary["test_features"] = (
            2 * (data_dictionary["test_features"] - train_min) / (train_max - train_min) - 1
        )

        for item in train_max.keys():
            self.data[item + "_max"] = train_max[item]
            self.data[item + "_min"] = train_min[item]

        for item in data_dictionary["train_labels"].keys():
            if data_dictionary["train_labels"][item].dtype == str:
                continue
            train_labels_max = data_dictionary["train_labels"][item].max()
            train_labels_min = data_dictionary["train_labels"][item].min()
            data_dictionary["train_labels"][item] = (
                2
                * (data_dictionary["train_labels"][item] - train_labels_min)
                / (train_labels_max - train_labels_min)
                - 1
            )
            if self.freqai_config.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
                data_dictionary["test_labels"][item] = (
                    2
                    * (data_dictionary["test_labels"][item] - train_labels_min)
                    / (train_labels_max - train_labels_min)
                    - 1
                )

            self.data[f"{item}_max"] = train_labels_max  # .to_dict()
            self.data[f"{item}_min"] = train_labels_min  # .to_dict()
        return data_dictionary

    def normalize_data_from_metadata(self, df: DataFrame) -> DataFrame:
        """
        Normalize a set of data using the mean and standard deviation from
        the associated training data.
        :params:
        :df: Dataframe to be standardized
        """

        for item in df.keys():
            df[item] = (
                2
                * (df[item] - self.data[f"{item}_min"])
                / (self.data[f"{item}_max"] - self.data[f"{item}_min"])
                - 1
            )

        return df

    def denormalize_labels_from_metadata(self, df: DataFrame) -> DataFrame:
        """
        Normalize a set of data using the mean and standard deviation from
        the associated training data.
        :params:
        :df: Dataframe of predictions to be denormalized
        """

        for label in self.label_list:
            if df[label].dtype == object:
                continue
            df[label] = (
                (df[label] + 1)
                * (self.data[f"{label}_max"] - self.data[f"{label}_min"])
                / 2
            ) + self.data[f"{label}_min"]

        return df

    def split_timerange(
        self, tr: str, train_split: int = 28, bt_split: int = 7
    ) -> Tuple[list, list]:
        """
        Function which takes a single time range (tr) and splits it
        into sub timeranges to train and backtest on based on user input
        tr: str, full timerange to train on
        train_split: the period length for the each training (days). Specified in user
        configuration file
        bt_split: the backtesting length (dats). Specified in user configuration file
        """

        if not isinstance(train_split, int) or train_split < 1:
            raise OperationalException(
                "train_period_days must be an integer greater than 0. " f"Got {train_split}."
            )
        train_period_days = train_split * SECONDS_IN_DAY
        bt_period = bt_split * SECONDS_IN_DAY

        full_timerange = TimeRange.parse_timerange(tr)
        config_timerange = TimeRange.parse_timerange(self.config["timerange"])
        if config_timerange.stopts == 0:
            config_timerange.stopts = int(
                datetime.datetime.now(tz=datetime.timezone.utc).timestamp()
            )
        timerange_train = copy.deepcopy(full_timerange)
        timerange_backtest = copy.deepcopy(full_timerange)

        tr_training_list = []
        tr_backtesting_list = []
        tr_training_list_timerange = []
        tr_backtesting_list_timerange = []
        first = True

        while True:
            if not first:
                timerange_train.startts = timerange_train.startts + bt_period
            timerange_train.stopts = timerange_train.startts + train_period_days

            first = False
            start = datetime.datetime.utcfromtimestamp(timerange_train.startts)
            stop = datetime.datetime.utcfromtimestamp(timerange_train.stopts)
            tr_training_list.append(start.strftime("%Y%m%d") + "-" + stop.strftime("%Y%m%d"))
            tr_training_list_timerange.append(copy.deepcopy(timerange_train))

            # associated backtest period

            timerange_backtest.startts = timerange_train.stopts

            timerange_backtest.stopts = timerange_backtest.startts + bt_period

            if timerange_backtest.stopts > config_timerange.stopts:
                timerange_backtest.stopts = config_timerange.stopts

            start = datetime.datetime.utcfromtimestamp(timerange_backtest.startts)
            stop = datetime.datetime.utcfromtimestamp(timerange_backtest.stopts)
            tr_backtesting_list.append(start.strftime("%Y%m%d") + "-" + stop.strftime("%Y%m%d"))
            tr_backtesting_list_timerange.append(copy.deepcopy(timerange_backtest))

            # ensure we are predicting on exactly same amount of data as requested by user defined
            #  --timerange
            if timerange_backtest.stopts == config_timerange.stopts:
                break

        # print(tr_training_list, tr_backtesting_list)
        return tr_training_list_timerange, tr_backtesting_list_timerange

    def slice_dataframe(self, timerange: TimeRange, df: DataFrame) -> DataFrame:
        """
        Given a full dataframe, extract the user desired window
        :params:
        :tr: timerange string that we wish to extract from df
        :df: Dataframe containing all candles to run the entire backtest. Here
        it is sliced down to just the present training period.
        """

        start = datetime.datetime.fromtimestamp(timerange.startts, tz=datetime.timezone.utc)
        stop = datetime.datetime.fromtimestamp(timerange.stopts, tz=datetime.timezone.utc)
        df = df.loc[df["date"] >= start, :]
        df = df.loc[df["date"] <= stop, :]

        return df

    def principal_component_analysis(self) -> None:
        """
        Performs Principal Component Analysis on the data for dimensionality reduction
        and outlier detection (see self.remove_outliers())
        No parameters or returns, it acts on the data_dictionary held by the DataHandler.
        """

        from sklearn.decomposition import PCA  # avoid importing if we dont need it

        n_components = self.data_dictionary["train_features"].shape[1]
        pca = PCA(n_components=n_components)
        pca = pca.fit(self.data_dictionary["train_features"])
        n_keep_components = np.argmin(pca.explained_variance_ratio_.cumsum() < 0.999)
        pca2 = PCA(n_components=n_keep_components)
        self.data["n_kept_components"] = n_keep_components
        pca2 = pca2.fit(self.data_dictionary["train_features"])
        logger.info("reduced feature dimension by %s", n_components - n_keep_components)
        logger.info("explained variance %f", np.sum(pca2.explained_variance_ratio_))
        train_components = pca2.transform(self.data_dictionary["train_features"])
        test_components = pca2.transform(self.data_dictionary["test_features"])

        self.data_dictionary["train_features"] = pd.DataFrame(
            data=train_components,
            columns=["PC" + str(i) for i in range(0, n_keep_components)],
            index=self.data_dictionary["train_features"].index,
        )

        # keeping a copy of the non-transformed features so we can check for errors during
        # model load from disk
        self.data["training_features_list_raw"] = copy.deepcopy(self.training_features_list)
        self.training_features_list = self.data_dictionary["train_features"].columns

        if self.freqai_config.get('data_split_parameters', {}).get('test_size', 0.1) != 0:
            self.data_dictionary["test_features"] = pd.DataFrame(
                data=test_components,
                columns=["PC" + str(i) for i in range(0, n_keep_components)],
                index=self.data_dictionary["test_features"].index,
            )

        self.data["n_kept_components"] = n_keep_components
        self.pca = pca2

        logger.info(f"PCA reduced total features from  {n_components} to {n_keep_components}")

        if not self.data_path.is_dir():
            self.data_path.mkdir(parents=True, exist_ok=True)

        return None

    def pca_transform(self, filtered_dataframe: DataFrame) -> None:
        """
        Use an existing pca transform to transform data into components
        :params:
        filtered_dataframe: DataFrame = the cleaned dataframe
        """
        pca_components = self.pca.transform(filtered_dataframe)
        self.data_dictionary["prediction_features"] = pd.DataFrame(
            data=pca_components,
            columns=["PC" + str(i) for i in range(0, self.data["n_kept_components"])],
            index=filtered_dataframe.index,
        )

    def compute_distances(self) -> float:
        """
        Compute distances between each training point and every other training
        point. This metric defines the neighborhood of trained data and is used
        for prediction confidence in the Dissimilarity Index
        """
        logger.info("computing average mean distance for all training points")
        tc = self.freqai_config.get("model_training_parameters", {}).get("thread_count", -1)
        pairwise = pairwise_distances(self.data_dictionary["train_features"], n_jobs=tc)
        avg_mean_dist = pairwise.mean(axis=1).mean()

        return avg_mean_dist

    def use_SVM_to_remove_outliers(self, predict: bool) -> None:
        """
        Build/inference a Support Vector Machine to detect outliers
        in training data and prediction
        :params:
        predict: bool = If true, inference an existing SVM model, else construct one
        """

        if self.keras:
            logger.warning(
                "SVM outlier removal not currently supported for Keras based models. "
                "Skipping user requested function."
            )
            if predict:
                self.do_predict = np.ones(len(self.data_dictionary["prediction_features"]))
            return

        if predict:
            if not self.svm_model:
                logger.warning("No svm model available for outlier removal")
                return
            y_pred = self.svm_model.predict(self.data_dictionary["prediction_features"])
            do_predict = np.where(y_pred == -1, 0, y_pred)

            if (len(do_predict) - do_predict.sum()) > 0:
                logger.info(
                    f"svm_remove_outliers() tossed {len(do_predict) - do_predict.sum()} predictions"
                )
            self.do_predict += do_predict
            self.do_predict -= 1

        else:
            # use SGDOneClassSVM to increase speed?
            svm_params = self.freqai_config["feature_parameters"].get(
                "svm_params", {"shuffle": False, "nu": 0.1})
            self.svm_model = linear_model.SGDOneClassSVM(**svm_params).fit(
                self.data_dictionary["train_features"]
            )
            y_pred = self.svm_model.predict(self.data_dictionary["train_features"])
            dropped_points = np.where(y_pred == -1, 0, y_pred)
            # keep_index = np.where(y_pred == 1)
            self.data_dictionary["train_features"] = self.data_dictionary["train_features"][
                (y_pred == 1)
            ]
            self.data_dictionary["train_labels"] = self.data_dictionary["train_labels"][
                (y_pred == 1)
            ]
            self.data_dictionary["train_weights"] = self.data_dictionary["train_weights"][
                (y_pred == 1)
            ]

            logger.info(
                f"svm_remove_outliers() tossed {len(y_pred) - dropped_points.sum()}"
                f" train points from {len(y_pred)}"
            )

            # same for test data
            if self.freqai_config['data_split_parameters'].get('test_size', 0.1) != 0:
                y_pred = self.svm_model.predict(self.data_dictionary["test_features"])
                dropped_points = np.where(y_pred == -1, 0, y_pred)
                self.data_dictionary["test_features"] = self.data_dictionary["test_features"][
                    (y_pred == 1)
                ]
                self.data_dictionary["test_labels"] = self.data_dictionary["test_labels"][(
                    y_pred == 1)]
                self.data_dictionary["test_weights"] = self.data_dictionary["test_weights"][
                    (y_pred == 1)
                ]

            logger.info(
                f"svm_remove_outliers() tossed {len(y_pred) - dropped_points.sum()}"
                f" test points from {len(y_pred)}"
            )

        return

    def find_features(self, dataframe: DataFrame) -> None:
        """
        Find features in the strategy provided dataframe
        :params:
        dataframe: DataFrame = strategy provided dataframe
        :returns:
        features: list = the features to be used for training/prediction
        """
        column_names = dataframe.columns
        features = [c for c in column_names if "%" in c]
        labels = [c for c in column_names if "&" in c]
        if not features:
            raise OperationalException("Could not find any features!")

        self.training_features_list = features
        self.label_list = labels
        # return features, labels

    def check_if_pred_in_training_spaces(self) -> None:
        """
        Compares the distance from each prediction point to each training data
        point. It uses this information to estimate a Dissimilarity Index (DI)
        and avoid making predictions on any points that are too far away
        from the training data set.
        """

        tc = self.freqai_config.get("model_training_parameters", {}).get("thread_count", -1)
        distance = pairwise_distances(
            self.data_dictionary["train_features"],
            self.data_dictionary["prediction_features"],
            n_jobs=tc,
        )

        self.DI_values = distance.min(axis=0) / self.data["avg_mean_dist"]

        do_predict = np.where(
            self.DI_values < self.freqai_config["feature_parameters"]["DI_threshold"],
            1,
            0,
        )

        if (len(do_predict) - do_predict.sum()) > 0:
            logger.info(
                f"DI tossed {len(do_predict) - do_predict.sum():.2f} predictions for "
                "being too far from training data"
            )

        self.do_predict += do_predict
        self.do_predict -= 1

    def set_weights_higher_recent(self, num_weights: int) -> npt.ArrayLike:
        """
        Set weights so that recent data is more heavily weighted during
        training than older data.
        """
        wfactor = self.config["freqai"]["feature_parameters"]["weight_factor"]
        weights = np.exp(-np.arange(num_weights) / (wfactor * num_weights))[::-1]
        return weights

    def append_predictions(self, predictions, do_predict, len_dataframe):
        """
        Append backtest prediction from current backtest period to all previous periods
        """

        append_df = DataFrame()
        for label in self.label_list:
            append_df[label] = predictions[label]
            append_df[f"{label}_mean"] = self.data["labels_mean"][label]
            append_df[f"{label}_std"] = self.data["labels_std"][label]

        append_df["do_predict"] = do_predict
        if self.freqai_config["feature_parameters"].get("DI_threshold", 0) > 0:
            append_df["DI_values"] = self.DI_values

        if self.full_df.empty:
            self.full_df = append_df
        else:
            self.full_df = pd.concat([self.full_df, append_df], axis=0)

        return

    def fill_predictions(self, dataframe):
        """
        Back fill values to before the backtesting range so that the dataframe matches size
        when it goes back to the strategy. These rows are not included in the backtest.
        """

        len_filler = len(dataframe) - len(self.full_df.index)  # startup_candle_count
        filler_df = pd.DataFrame(
            np.zeros((len_filler, len(self.full_df.columns))), columns=self.full_df.columns
        )

        self.full_df = pd.concat([filler_df, self.full_df], axis=0, ignore_index=True)

        to_keep = [col for col in dataframe.columns if not col.startswith("&")]
        self.return_dataframe = pd.concat([dataframe[to_keep], self.full_df], axis=1)

        # self.append_df = DataFrame()
        self.full_df = DataFrame()

        return

    def create_fulltimerange(self, backtest_tr: str, backtest_period_days: int) -> str:

        if not isinstance(backtest_period_days, int):
            raise OperationalException("backtest_period_days must be an integer")

        if backtest_period_days < 0:
            raise OperationalException("backtest_period_days must be positive")

        backtest_timerange = TimeRange.parse_timerange(backtest_tr)

        if backtest_timerange.stopts == 0:
            # typically open ended time ranges do work, however, there are some edge cases where
            # it does not. accomodating these kinds of edge cases just to allow open-ended
            # timerange is not high enough priority to warrant the effort. It is safer for now
            # to simply ask user to add their end date
            raise OperationalException("FreqAI backtesting does not allow open ended timeranges. "
                                       "Please indicate the end date of your desired backtesting. "
                                       "timerange.")
            # backtest_timerange.stopts = int(
            #     datetime.datetime.now(tz=datetime.timezone.utc).timestamp()
            # )

        backtest_timerange.startts = (
            backtest_timerange.startts - backtest_period_days * SECONDS_IN_DAY
        )
        start = datetime.datetime.utcfromtimestamp(backtest_timerange.startts)
        stop = datetime.datetime.utcfromtimestamp(backtest_timerange.stopts)
        full_timerange = start.strftime("%Y%m%d") + "-" + stop.strftime("%Y%m%d")

        self.full_path = Path(
            self.config["user_data_dir"] / "models" / f"{self.freqai_config['identifier']}"
        )

        config_path = Path(self.config["config_files"][0])

        if not self.full_path.is_dir():
            self.full_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(
                config_path.resolve(),
                Path(self.full_path / config_path.parts[-1]),
            )

        return full_timerange

    def check_if_model_expired(self, trained_timestamp: int) -> bool:
        """
        A model age checker to determine if the model is trustworthy based on user defined
        `expiration_hours` in the configuration file.
        :params:
        trained_timestamp: int = The time of training for the most recent model.
        :returns:
        bool = If the model is expired or not.
        """
        time = datetime.datetime.now(tz=datetime.timezone.utc).timestamp()
        elapsed_time = (time - trained_timestamp) / 3600  # hours
        max_time = self.freqai_config.get("expiration_hours", 0)
        if max_time > 0:
            return elapsed_time > max_time
        else:
            return False

    def check_if_new_training_required(
        self, trained_timestamp: int
    ) -> Tuple[bool, TimeRange, TimeRange]:

        time = datetime.datetime.now(tz=datetime.timezone.utc).timestamp()
        trained_timerange = TimeRange()
        data_load_timerange = TimeRange()

        # find the max indicator length required
        max_timeframe_chars = self.freqai_config["feature_parameters"].get(
            "include_timeframes"
        )[-1]
        max_period = self.freqai_config["feature_parameters"].get(
            "indicator_max_period_candles", 50
        )
        additional_seconds = 0
        if max_timeframe_chars[-1] == "d":
            additional_seconds = max_period * SECONDS_IN_DAY * int(max_timeframe_chars[-2])
        elif max_timeframe_chars[-1] == "h":
            additional_seconds = max_period * 3600 * int(max_timeframe_chars[-2])
        elif max_timeframe_chars[-1] == "m":
            if len(max_timeframe_chars) == 2:
                additional_seconds = max_period * 60 * int(max_timeframe_chars[-2])
            elif len(max_timeframe_chars) == 3:
                additional_seconds = max_period * 60 * int(float(max_timeframe_chars[0:2]))
            else:
                logger.warning(
                    "FreqAI could not detect max timeframe and therefore may not "
                    "download the proper amount of data for training"
                )

        # logger.info(f'Extending data download by {additional_seconds/SECONDS_IN_DAY:.2f} days')

        if trained_timestamp != 0:
            elapsed_time = (time - trained_timestamp) / SECONDS_IN_HOUR
            retrain = elapsed_time > self.freqai_config.get("live_retrain_hours", 0)
            if retrain:
                trained_timerange.startts = int(
                    time - self.freqai_config.get("train_period_days", 0) * SECONDS_IN_DAY
                )
                trained_timerange.stopts = int(time)
                # we want to load/populate indicators on more data than we plan to train on so
                # because most of the indicators have a rolling timeperiod, and are thus NaNs
                # unless they have data further back in time before the start of the train period
                data_load_timerange.startts = int(
                    time
                    - self.freqai_config.get("train_period_days", 0) * SECONDS_IN_DAY
                    - additional_seconds
                )
                data_load_timerange.stopts = int(time)
        else:  # user passed no live_trained_timerange in config
            trained_timerange.startts = int(
                time - self.freqai_config.get("train_period_days") * SECONDS_IN_DAY
            )
            trained_timerange.stopts = int(time)

            data_load_timerange.startts = int(
                time
                - self.freqai_config.get("train_period_days", 0) * SECONDS_IN_DAY
                - additional_seconds
            )
            data_load_timerange.stopts = int(time)
            retrain = True

        # logger.info(
        #     f"downloading data for "
        #     f"{(data_load_timerange.stopts-data_load_timerange.startts)/SECONDS_IN_DAY:.2f} "
        #     " days. "
        #     f"Extension of {additional_seconds/SECONDS_IN_DAY:.2f} days"
        # )

        return retrain, trained_timerange, data_load_timerange

    def set_new_model_names(self, pair: str, trained_timerange: TimeRange):

        coin, _ = pair.split("/")
        self.data_path = Path(
            self.full_path
            / f"sub-train-{pair.split('/')[0]}_{int(trained_timerange.stopts)}"
        )

        self.model_filename = f"cb_{coin.lower()}_{int(trained_timerange.stopts)}"

    def download_all_data_for_training(self, timerange: TimeRange) -> None:
        """
        Called only once upon start of bot to download the necessary data for
        populating indicators and training the model.
        :params:
        timerange: TimeRange = The full data timerange for populating the indicators
        and training the model.
        """
        exchange = ExchangeResolver.load_exchange(
            self.config["exchange"]["name"], self.config, validate=False, load_leverage_tiers=False
        )

        new_pairs_days = int((timerange.stopts - timerange.startts) / SECONDS_IN_DAY)

        refresh_backtest_ohlcv_data(
            exchange,
            pairs=self.all_pairs,
            timeframes=self.freqai_config["feature_parameters"].get("include_timeframes"),
            datadir=self.config["datadir"],
            timerange=timerange,
            new_pairs_days=new_pairs_days,
            erase=False,
            data_format=self.config.get("dataformat_ohlcv", "json"),
            trading_mode=self.config.get("trading_mode", "spot"),
            prepend=self.config.get("prepend_data", False),
        )

    def set_all_pairs(self) -> None:

        self.all_pairs = copy.deepcopy(
            self.freqai_config["feature_parameters"].get("include_corr_pairlist", [])
        )
        for pair in self.config.get("exchange", "").get("pair_whitelist"):
            if pair not in self.all_pairs:
                self.all_pairs.append(pair)

    def use_strategy_to_populate_indicators(
        self,
        strategy: IStrategy,
        corr_dataframes: dict = {},
        base_dataframes: dict = {},
        pair: str = "",
        prediction_dataframe: DataFrame = pd.DataFrame(),
    ) -> DataFrame:
        """
        Use the user defined strategy for populating indicators during
        retrain
        :params:
        strategy: IStrategy = user defined strategy object
        corr_dataframes: dict = dict containing the informative pair dataframes
        (for user defined timeframes)
        base_dataframes: dict = dict containing the current pair dataframes
        (for user defined timeframes)
        metadata: dict = strategy furnished pair metadata
        :returns:
        dataframe: DataFrame = dataframe containing populated indicators
        """

        # for prediction dataframe creation, we let dataprovider handle everything in the strategy
        # so we create empty dictionaries, which allows us to pass None to
        # `populate_any_indicators()`. Signaling we want the dp to give us the live dataframe.
        tfs = self.freqai_config["feature_parameters"].get("include_timeframes")
        pairs = self.freqai_config["feature_parameters"].get("include_corr_pairlist", [])
        if not prediction_dataframe.empty:
            dataframe = prediction_dataframe.copy()
            for tf in tfs:
                base_dataframes[tf] = None
                for p in pairs:
                    if p not in corr_dataframes:
                        corr_dataframes[p] = {}
                    corr_dataframes[p][tf] = None
        else:
            dataframe = base_dataframes[self.config["timeframe"]].copy()

        sgi = False
        for tf in tfs:
            if tf == tfs[-1]:
                sgi = True  # doing this last allows user to use all tf raw prices in labels
            dataframe = strategy.populate_any_indicators(
                pair,
                dataframe.copy(),
                tf,
                informative=base_dataframes[tf],
                set_generalized_indicators=sgi
            )
            if pairs:
                for i in pairs:
                    if pair in i:
                        continue  # dont repeat anything from whitelist
                    dataframe = strategy.populate_any_indicators(
                        i,
                        dataframe.copy(),
                        tf,
                        informative=corr_dataframes[i][tf]
                    )

        return dataframe

    def fit_labels(self) -> None:
        """
        Fit the labels with a gaussian distribution
        """
        import scipy as spy

        self.data["labels_mean"], self.data["labels_std"] = {}, {}
        for label in self.label_list:
            f = spy.stats.norm.fit(self.data_dictionary["train_labels"][label])
            self.data["labels_mean"][label], self.data["labels_std"][label] = f[0], f[1]

        # KEEPME incase we want to let user start to grab quantiles.
        # upper_q = spy.stats.norm.ppf(self.freqai_config['feature_parameters'][
        #                                                   'target_quantile'], *f)
        # lower_q = spy.stats.norm.ppf(1 - self.freqai_config['feature_parameters'][
        #                                                       'target_quantile'], *f)
        # self.data["upper_quantile"] = upper_q
        # self.data["lower_quantile"] = lower_q
        return

    def remove_features_from_df(self, dataframe: DataFrame) -> DataFrame:
        """
        Remove the features from the dataframe before returning it to strategy. This keeps it
        compact for Frequi purposes.
        """
        to_keep = [
            col for col in dataframe.columns if not col.startswith("%") or col.startswith("%%")
        ]
        return dataframe[to_keep]

    def get_current_trade_database(self) -> None:

        if self.database_path == '':
            logger.warning('No trade databse found. Skipping analysis.')
            return

        data = sqlite3.connect(self.database_path)
        query = data.execute("SELECT * From trades")
        cols = [column[0] for column in query.description]
        df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
        self.trade_database_df = df.dropna(subset='close_date')
        data.close()

    def np_encoder(self, object):
        if isinstance(object, np.generic):
            return object.item()

    # Functions containing useful data manpulation examples. but not actively in use.

    # Possibly phasing these outlier removal methods below out in favor of
    # use_SVM_to_remove_outliers (computationally more efficient and apparently higher performance).
    # But these have good data manipulation examples, so keep them commented here for now.

    # def determine_statistical_distributions(self) -> None:
    #     from fitter import Fitter

    #     logger.info('Determining best model for all features, may take some time')

    #     def compute_quantiles(ft):
    #         f = Fitter(self.data_dictionary["train_features"][ft],
    #                    distributions=['gamma', 'cauchy', 'laplace',
    #                                   'beta', 'uniform', 'lognorm'])
    #         f.fit()
    #         # f.summary()
    #         dist = list(f.get_best().items())[0][0]
    #         params = f.get_best()[dist]
    #         upper_q = getattr(spy.stats, list(f.get_best().items())[0][0]).ppf(0.999, **params)
    #         lower_q = getattr(spy.stats, list(f.get_best().items())[0][0]).ppf(0.001, **params)

    #         return ft, upper_q, lower_q, dist

    #     quantiles_tuple = Parallel(n_jobs=-1)(
    #         delayed(compute_quantiles)(ft) for ft in self.data_dictionary[
    #                                                       'train_features'].columns)

    #     df = pd.DataFrame(quantiles_tuple, columns=['features', 'upper_quantiles',
    #                                                 'lower_quantiles', 'dist'])
    #     self.data_dictionary['upper_quantiles'] = df['upper_quantiles']
    #     self.data_dictionary['lower_quantiles'] = df['lower_quantiles']

    #     return

    # def remove_outliers(self, predict: bool) -> None:
    #     """
    #     Remove data that looks like an outlier based on the distribution of each
    #     variable.
    #     :params:
    #     :predict: boolean which tells the function if this is prediction data or
    #     training data coming in.
    #     """

    #     lower_quantile = self.data_dictionary["lower_quantiles"].to_numpy()
    #     upper_quantile = self.data_dictionary["upper_quantiles"].to_numpy()

    #     if predict:

    #         df = self.data_dictionary["prediction_features"][
    #             (self.data_dictionary["prediction_features"] < upper_quantile)
    #             & (self.data_dictionary["prediction_features"] > lower_quantile)
    #         ]
    #         drop_index = pd.isnull(df).any(1)
    #         self.data_dictionary["prediction_features"].fillna(0, inplace=True)
    #         drop_index = ~drop_index
    #         do_predict = np.array(drop_index.replace(True, 1).replace(False, 0))

    #         logger.info(
    #             "remove_outliers() tossed %s predictions",
    #             len(do_predict) - do_predict.sum(),
    #         )
    #         self.do_predict += do_predict
    #         self.do_predict -= 1

    #     else:

    #         filter_train_df = self.data_dictionary["train_features"][
    #             (self.data_dictionary["train_features"] < upper_quantile)
    #             & (self.data_dictionary["train_features"] > lower_quantile)
    #         ]
    #         drop_index = pd.isnull(filter_train_df).any(1)
    #         drop_index = drop_index.replace(True, 1).replace(False, 0)
    #         self.data_dictionary["train_features"] = self.data_dictionary["train_features"][
    #             (drop_index == 0)
    #         ]
    #         self.data_dictionary["train_labels"] = self.data_dictionary["train_labels"][
    #             (drop_index == 0)
    #         ]
    #         self.data_dictionary["train_weights"] = self.data_dictionary["train_weights"][
    #             (drop_index == 0)
    #         ]

    #         logger.info(
    #             f'remove_outliers() tossed {drop_index.sum()}'
    #             f' training points from {len(filter_train_df)}'
    #         )

    #         # do the same for the test data
    #         filter_test_df = self.data_dictionary["test_features"][
    #             (self.data_dictionary["test_features"] < upper_quantile)
    #             & (self.data_dictionary["test_features"] > lower_quantile)
    #         ]
    #         drop_index = pd.isnull(filter_test_df).any(1)
    #         drop_index = drop_index.replace(True, 1).replace(False, 0)
    #         self.data_dictionary["test_labels"] = self.data_dictionary["test_labels"][
    #             (drop_index == 0)
    #         ]
    #         self.data_dictionary["test_features"] = self.data_dictionary["test_features"][
    #             (drop_index == 0)
    #         ]
    #         self.data_dictionary["test_weights"] = self.data_dictionary["test_weights"][
    #             (drop_index == 0)
    #         ]

    #         logger.info(
    #             f'remove_outliers() tossed {drop_index.sum()}'
    #             f' test points from {len(filter_test_df)}'
    #         )

    #     return

    # def standardize_data(self, data_dictionary: Dict) -> Dict[Any, Any]:
    #     """
    #     standardize all data in the data_dictionary according to the training dataset
    #     :params:
    #     :data_dictionary: dictionary containing the cleaned and split training/test data/labels
    #     :returns:
    #     :data_dictionary: updated dictionary with standardized values.
    #     """
    #     # standardize the data by training stats
    #     train_mean = data_dictionary["train_features"].mean()
    #     train_std = data_dictionary["train_features"].std()
    #     data_dictionary["train_features"] = (
    #         data_dictionary["train_features"] - train_mean
    #     ) / train_std
    #     data_dictionary["test_features"] = (
    #         data_dictionary["test_features"] - train_mean
    #     ) / train_std

    #     train_labels_std = data_dictionary["train_labels"].std()
    #     train_labels_mean = data_dictionary["train_labels"].mean()
    #     data_dictionary["train_labels"] = (
    #         data_dictionary["train_labels"] - train_labels_mean
    #     ) / train_labels_std
    #     data_dictionary["test_labels"] = (
    #         data_dictionary["test_labels"] - train_labels_mean
    #     ) / train_labels_std

    #     for item in train_std.keys():
    #         self.data[item + "_std"] = train_std[item]
    #         self.data[item + "_mean"] = train_mean[item]

    #     self.data["labels_std"] = train_labels_std
    #     self.data["labels_mean"] = train_labels_mean

    #     return data_dictionary

    # def standardize_data_from_metadata(self, df: DataFrame) -> DataFrame:
    # """
    # Normalizes a set of data using the mean and standard deviation from
    # the associated training data.
    # :params:
    # :df: Dataframe to be standardized
    # """

    # for item in df.keys():
    #     df[item] = (df[item] - self.data[item + "_mean"]) / self.data[item + "_std"]

    # return df
