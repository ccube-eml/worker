from collections import Counter

from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn import metrics

from worker import utils
from worker.fuser_policies.fuser_policy import FuserPolicy


class VotingFuserPolicy(FuserPolicy):
    """
    Applies the voting ensemble.
    """

    def __init__(
            self,
            models_executors,
            training_data,
            training_actual_values,
            test_data,
            test_actual_values,
            true_class_value,
            class_attribute_type,
            configuration,
    ):
        super().__init__(
            models_executors,
            training_data,
            training_actual_values,
            test_data,
            test_actual_values,
            true_class_value,
            class_attribute_type,
            configuration,
        )

    class ConstantPredictionsClassifier(BaseEstimator):
        def __init__(self, predictions):
            self.predictions = predictions

        def fit(self, X, y):
            self._X = None
            self._y = None

            return self

        def predict(self, X):
            check_is_fitted(self, ['_X', '_y'])

            return self.predictions

    def fuse(self):
        # Sets the true value and counts.
        class_values_counter = Counter(self._test_actual_values)
        true_class_value = self._true_class_value
        false_class_value = list(set(class_values_counter.keys()) - set([true_class_value]))[0]

        # Predicts the test data.
        predictors_values = []
        for model_executor in self._models_executors:
            # Execute the model.
            predictors_values.append(model_executor.execute(self._test_data))

        # Builds the ensemble.
        predictors = [
            (str(i), VotingFuserPolicy.ConstantPredictionsClassifier(predictor_values))
            for i, predictor_values in enumerate(predictors_values)
        ]
        ensemble = VotingClassifier(predictors, voting='hard')

        # Dummy fit.
        ensemble = ensemble.fit(self._test_actual_values, self._test_actual_values)

        # Predicts on the actual values.
        ensemble_predictions = ensemble.predict(self._test_actual_values)

        # Computes the metrics.
        confusion_matrix = utils.compute_confusion_matrix(
            self._test_actual_values,
            ensemble_predictions,
            [false_class_value, true_class_value],
        )

        ensemble_metrics = {
            'accuracy': metrics.accuracy_score(self._test_actual_values, ensemble_predictions),
            'f-measure': metrics.f1_score(self._test_actual_values, ensemble_predictions),
            'mcc': metrics.matthews_corrcoef(self._test_actual_values, ensemble_predictions),
            'precision': metrics.precision_score(self._test_actual_values, ensemble_predictions),
            'recall': metrics.recall_score(self._test_actual_values, ensemble_predictions),
            'true_positives': confusion_matrix['true_positives'],
            'false_positives': confusion_matrix['false_positives'],
            'true_negatives': confusion_matrix['true_negatives'],
            'false_negatives': confusion_matrix['false_negatives'],
        }

        return ensemble_metrics
