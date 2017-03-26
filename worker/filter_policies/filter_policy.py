from abc import ABCMeta, abstractmethod

from worker import utils


class FilterPolicy(object):
    """
    Defines a filter policy.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(
            self,
            models_executors,
            training_data,
            actual_values,
            true_class_value,
            class_attribute_type,
            configuration,
    ):
        """
        Initializes a policy for filtering.

        :param models_executors: the model executors
        :type models_executors: object

        :param training_data: the training data file
        :type training_data: str

        :param actual_values: the actual values
        :type actual_values: list

        :param true_class_value: the class value to consider as true
        :type true_class_value: str

        :param class_attribute_type: the type of the class attribute
        :type class_attribute_type: str

        :param configuration: the configuration for the policy
        :type configuration: dict
        """
        self._models_executors = models_executors
        self._training_data = training_data
        self._actual_values = actual_values
        self._true_class_value = utils.convert_string(true_class_value, class_attribute_type)
        self._configuration = configuration

    @abstractmethod
    def filter(
            self,
    ):
        """
        Applies the filter policy.

        :return: a list of boolean values indicating which predictor has been selected
        :rtype: list[bool]
        """
        pass
