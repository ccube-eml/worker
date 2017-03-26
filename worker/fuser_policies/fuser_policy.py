from abc import ABCMeta, abstractmethod

from worker import utils


class FuserPolicy(object):
    """
    Defines a fusion policy.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
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
        """
        Initializes a policy for fuse and test with actual values.

        :param models_executors: the model executors
        :type models_executors: object

        :param training_data: the training data file
        :type training_data: str

        :param training_actual_values: the actual values for training
        :type training_actual_values: list

        :param test_data: the test data file
        :type test_data: str

        :param test_actual_values: the actual values for test
        :type test_actual_values: list

        :param true_class_value: the class value to consider as true
        :type true_class_value: str

        :param class_attribute_type: the type of the class attribute
        :type class_attribute_type: str

        :param configuration: the configuration for the policy
        :type configuration: dict
        """
        self._models_executors = models_executors
        self._training_data = training_data
        self._training_actual_values = training_actual_values
        self._test_data = test_data
        self._test_actual_values = test_actual_values
        self._true_class_value = utils.convert_string(true_class_value, class_attribute_type)
        self._configuration = configuration

    @abstractmethod
    def fuse(
            self,
    ):
        """
        Applies the fusion policy using training data, testing on a dataset.

        :return: a dict containing the computed metrics
        :rtype: dict
        """
        pass
