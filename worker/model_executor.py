import tempfile

from worker.process_manager import ProcessManager
from worker import utils


PREDICT_DATASET_FILE_VARIABLE_NAME = 'CCUBE_PREDICT_DATASET_FILE'
PREDICT_INPUT_FILES_VARIABLE_NAME = 'CCUBE_PREDICT_INPUT_FILES'
PREDICT_PARAMETERS_PROPERTIES_FILE_VARIABLE_NAME = 'CCUBE_PREDICT_PARAMETERS_PROPERTIES_FILE'


class ModelExecutor(object):
    """
    It executes a model.
    """

    def __init__(
            self,
            model_files_stream,
            model_parameters,
            predictions_file_path,
            execution_command,
            working_directory,
            environment_variables,
    ):
        # Prepares the process manager.
        """

        :param model_files_stream: the the base64 zip string
        :type model_files_stream: str

        :param model_parameters: the parameters to execute the model
        :type model_parameters: dict


        :param predictions_file_path: the file containing the predictions
        :type predictions_file_path: str

        :param execution_command: the command to execute
        :type execution_command: str

        :param working_directory: the working directory for the command
        :type working_directory: str

        :param environment_variables: the environment variables to set
        :type environment_variables: dict
        """
        self.__process_manager = ProcessManager(
            command=execution_command,
            working_directory=working_directory,
            environment_variables=environment_variables,
        )

        self.__model_files_stream = model_files_stream
        self.__model_parameters = model_parameters
        self.__predictions_file_path = predictions_file_path

    def execute(
            self,
            data_file_path,
    ):
        """
        Executes the command and returns the predicted values.

        :param data_file_path: the data file to test the model
        :type data_file_path: str

        :return: the predicted values
        :rtype: list
        """
        # Extracts the output in a temporary directory.
        model_files_directory = tempfile.TemporaryDirectory()
        utils.extract_zip_base64_string(self.__model_files_stream, model_files_directory.name)

        # Adds the environment variables.
        self.__process_manager.add_environment_variables(
            {
                PREDICT_DATASET_FILE_VARIABLE_NAME: data_file_path,
                PREDICT_INPUT_FILES_VARIABLE_NAME: model_files_directory.name,
            }
        )

        # Prepares the properties.
        string_properties = utils.convert_values_to_string(self.__model_parameters)
        self.__process_manager.add_environment_variables(string_properties)
        properties_file = utils.create_temporary_properties_file(string_properties)
        self.__process_manager.add_environment_variables(
            {
                PREDICT_PARAMETERS_PROPERTIES_FILE_VARIABLE_NAME: properties_file.name
            }
        )

        # Echoes the command.
        stdout, return_code = self.__process_manager.echo()

        # Runs the process.
        stdout, return_code = self.__process_manager.run()

        # Reads the predicted values.
        predicted_values = utils.read_values_from_file(self.__predictions_file_path)

        # Closes the temporary files.
        properties_file.close()

        # Cleanup the directory.
        model_files_directory.cleanup()

        # Returns the predicted values.
        return predicted_values
