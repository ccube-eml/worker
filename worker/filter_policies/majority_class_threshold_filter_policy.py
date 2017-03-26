from collections import Counter

import numpy

from worker import utils
from worker.filter_policies.filter_policy import FilterPolicy


class MajorityCostThresholdFilterPolicy(FilterPolicy):
    """
    Applies the majority cost thresold filtering, only working with binary class values.
    """

    def __init__(
            self,
            models_executors,
            training_data,
            actual_values,
            true_class_value,
            class_attribute_type,
            configuration,
    ):
        super().__init__(
            models_executors,
            training_data,
            actual_values,
            true_class_value,
            class_attribute_type,
            configuration,
        )

        self.__false_negative_weight = configuration.get('threshold')
        self.__false_positive_weight = 1 - self.__false_negative_weight

    def filter(self):
        # Computes the majority cost of the naive classifier (always answering as the majority class value).
        result = []

        # Counts occurrences.
        class_values_counter = Counter(self._actual_values)

        # Sets the true value and counts.
        true_class_value = self._true_class_value
        true_class_value_count = class_values_counter.get(true_class_value)
        false_class_value = list(set(class_values_counter.keys()) - set([true_class_value]))[0]
        false_class_value_count = class_values_counter.get(false_class_value)

        # Finds the majority class value.
        majority_class_value = class_values_counter.most_common()[0][0]

        # Fills the naive classifier.
        naive_predicted_values = numpy.repeat(majority_class_value, len(self._actual_values))

        # Computes the false positive and false negative rates.
        naive_confusion_matrix = utils.compute_confusion_matrix(
            self._actual_values,
            naive_predicted_values,
            [false_class_value, true_class_value],
        )
        naive_false_negatives_rate = naive_confusion_matrix['false_negatives'] / true_class_value_count
        naive_false_positives_rate = naive_confusion_matrix['false_positives'] / false_class_value_count

        # Computes the majority cost.
        majority_cost =\
            self.__false_negative_weight * naive_false_negatives_rate\
            + self.__false_positive_weight * naive_false_positives_rate

        for model_executor in self._models_executors:
            # Execute the model.
            predicted_values = model_executor.execute(self._training_data)

            # Computes the false positive and false negative rates.
            model_confusion_matrix = utils.compute_confusion_matrix(
                self._actual_values,
                predicted_values,
                [false_class_value, true_class_value],
            )
            model_false_negatives_rate = model_confusion_matrix['false_negatives'] / true_class_value_count
            model_false_positives_rate = model_confusion_matrix['false_positives'] / false_class_value_count

            # Computes the cost.
            model_cost = \
                self.__false_negative_weight * model_false_negatives_rate \
                + self.__false_positive_weight * model_false_positives_rate

            # Filters the model.
            if model_cost < majority_cost:
                result.append(True)
            else:
                result.append(False)

        return result
