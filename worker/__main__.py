import os
import tempfile
import io
import datetime
import socket
import json

import click
import requests
import pymongo

from worker.process_manager import ProcessManager
from worker.amqp_manager import AMQPManager
from worker.filter_policies.majority_class_threshold_filter_policy import MajorityCostThresholdFilterPolicy
from worker.fuser_policies.voting_fuser_policy import VotingFuserPolicy
from worker import utils
from worker.model_executor import ModelExecutor


LEARN_COMMAND_VARIABLE_NAME = 'CCUBE_LEARN_COMMAND'
LEARN_WORKING_DIRECTORY_VARIABLE_NAME = 'CCUBE_LEARN_WORKING_DIRECTORY'
LEARN_OUTPUT_FILES_VARIABLE_NAME = 'CCUBE_LEARN_OUTPUT_FILES'

LEARN_DATASET_FILE_VARIABLE_NAME = 'CCUBE_LEARN_DATASET_FILE'
LEARN_PARAMETERS_PROPERTIES_FILE_VARIABLE_NAME = 'CCUBE_LEARN_PARAMETERS_PROPERTIES_FILE'
LEARN_DURATION_SECONDS_VARIABLE_NAME = 'CCUBE_LEARN_DURATION_SECONDS'
LEARN_DURATION_MINUTES_VARIABLE_NAME = 'CCUBE_LEARN_DURATION_MINUTES'

GET_DATASET_TRAINING_SAMPLE_REQUEST_URL = 'http://{hostname_}:{port_}/dataset/{dataset_name_}/split/training/sample'

PREDICT_COMMAND_VARIABLE_NAME = 'CCUBE_PREDICT_COMMAND'
PREDICT_WORKING_DIRECTORY_VARIABLE_NAME = 'CCUBE_PREDICT_WORKING_DIRECTORY'
PREDICT_INPUT_FILES_VARIABLE_NAME = 'CCUBE_PREDICT_INPUT_FILES'
PREDICT_PREDICTIONS_FILE_VARIABLE_NAME = 'CCUBE_PREDICT_PREDICTIONS_FILE'

PREDICT_DATASET_FILE_VARIABLE_NAME = 'CCUBE_PREDICT_DATASET_FILE'
PREDICT_PARAMETERS_PROPERTIES_FILE_VARIABLE_NAME = 'CCUBE_PREDICT_PARAMETERS_PROPERTIES_FILE'

GET_DATASET_FUSION_SPLIT_REQUEST_URL = 'http://{hostname_}:{port_}/dataset/{dataset_name_}/split/fusion'
GET_DATASET_FUSION_SPLIT_CLASS_REQUEST_URL = 'http://{hostname_}:{port_}/dataset/{dataset_name_}/split/fusion/class'
GET_DATASET_TEST_SPLIT_REQUEST_URL = 'http://{hostname_}:{port_}/dataset/{dataset_name_}/split/test'
GET_DATASET_TEST_SPLIT_CLASS_REQUEST_URL = 'http://{hostname_}:{port_}/dataset/{dataset_name_}/split/test/class'

DATA_CHUNK_SIZE = 4096

LEARN_TASKS_QUEUE_NAME = '{job_name_}@learner.tasks'
LEARN_OUTPUTS_QUEUE_NAME = '{job_name_}@learner.outputs'
FILTER_TASKS_QUEUE_NAME = '{job_name_}@filter.tasks'
FILTER_OUTPUTS_QUEUE_NAME = '{job_name_}@filter.outputs'
FUSER_TASKS_QUEUE_NAME = '{job_name_}@fuser.tasks'


@click.group()
def cli():
    pass


@cli.command()
@click.option('--job', '-j', 'job_name', type=str, required=True)
def learn(job_name):
    """
    Start the learning phase.

    Manages the learner using information from the environment variables and the tasks queue.

    It requires the following environment variables:
    - CCUBE_LEARN_COMMAND: the shell string used to launch the learner
    - CCUBE_LEARN_WORKING_DIRECTORY: the working the directory in which executing the command
    - CCUBE_LEARN_OUTPUT_FILES: the learner output files to include in the final output

    It fills some environment variables to let the user using them in the CCUBE_LEARN_COMMAND:
    - CCUBE_LEARN_TRAINING_FILE: the file path of the dataset training sample
    - CCUBE_LEARN_PARAMETERS_PROPERTIES_FILE: the .properties file filled with the learner parameters for the task
    - CCUBE_LEARN_DURATION_SECONDS: the duration in seconds, required for the task
    - CCUBE_LEARN_DURATION_MINUTES: the duration in minutes, required for the task

    Moreover, it automatically generates the environment variables for the learner parameters included in the task,
    with the same provided names.

    :param job_name: the name of the job
    :type job_name: str
    """
    learner_start_time = datetime.datetime.utcnow()

    # Get ip and hostname
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)

    # MongoDB connection.
    mongodb_client = pymongo.MongoClient('mongodb', 27017)
    database = mongodb_client.ccube
    times = database.times

    # Reads the environment variables.
    environment_variables = os.environ.copy()

    learn_command = environment_variables.get(LEARN_COMMAND_VARIABLE_NAME)

    learn_working_directory = environment_variables.get(LEARN_WORKING_DIRECTORY_VARIABLE_NAME)

    learn_output_files = environment_variables.get(LEARN_OUTPUT_FILES_VARIABLE_NAME, "")
    learn_output_files = os.path.expandvars(learn_output_files)
    learn_output_files = learn_output_files.split(os.pathsep)

    # Retrieves the queues names.
    learner_tasks_queue_name = LEARN_TASKS_QUEUE_NAME.format(job_name_=job_name)
    learner_outputs_queue_name = LEARN_OUTPUTS_QUEUE_NAME.format(job_name_=job_name)

    # Creates the queues, if they do not exist.
    amqp_manager = AMQPManager(environment_variables.get('AMQP_HOSTNAME', 'rabbitmq'))
    amqp_manager.create_queue(learner_tasks_queue_name)
    amqp_manager.create_queue(learner_outputs_queue_name)

    # Consumes a task.
    tasks, delivery_tags = amqp_manager.consume_messages(learner_tasks_queue_name, 1)
    task = tasks[0]

    # Stores the start time.
    times.insert_one({
        'ip': ip,
        'hostname': hostname,
        'random_seed': task.get('random_seed'),
        'type': 'learner_start',
        'time': learner_start_time,
    })

    # Gets the sample.
    request_params = {
        'training_rate': task.get('training_rate'),
        'sample_rate': task.get('sample_rate'),
        'sample_number': task.get('task_number'),
        'class_attribute': task.get('class_attribute'),
        'include_attributes': task.get('include_attributes'),
        'exclude_attributes': task.get('exclude_attributes'),
        'attributes_rate': task.get('attributes_rate'),
        'random_seed': task.get('random_seed'),
        'include_header': task.get('include_header'),
    }
    response = requests.get(
        GET_DATASET_TRAINING_SAMPLE_REQUEST_URL.format(
            hostname_=environment_variables.get('FACTORIZER_HOSTNAME', 'factorizer'),
            port_=environment_variables.get('FACTORIZER_PORT', '5000'),
            dataset_name_=task.get('dataset_name'),
        ),
        params=request_params,
        stream=True,
    )

    # Saves the response into a temporary file.
    training_file = tempfile.NamedTemporaryFile()
    for chunk in response.iter_content(chunk_size=DATA_CHUNK_SIZE):
        if chunk:
            training_file.write(chunk)
    training_file.flush()
    training_file.seek(0)

    # Extracts the learn parameters.
    learn_parameters = task.get('learn_parameters')

    # Prepares the process manager.
    process_manager = ProcessManager(
        command=learn_command,
        working_directory=learn_working_directory,
        environment_variables=environment_variables,
    )

    # Adds the environment variables.
    process_manager.add_environment_variables(get_duration_variables(task.get('duration')))
    process_manager.add_environment_variables({LEARN_DATASET_FILE_VARIABLE_NAME: training_file.name})

    # Prepares the learn properties.
    string_learn_properties = utils.convert_values_to_string(learn_parameters)

    process_manager.add_environment_variables(string_learn_properties)

    properties_file = utils.create_temporary_properties_file(string_learn_properties)
    process_manager.add_environment_variables({LEARN_PARAMETERS_PROPERTIES_FILE_VARIABLE_NAME: properties_file.name})

    # Echoes the command.
    process_manager.echo()

    # Runs the process.
    stdout, return_code = process_manager.run()

    # Prepares the output message.
    learner_success = (return_code == 0)
    learner_output_zip_stream = None

    if learner_success:
        learner_output_zip_stream = utils.create_zip_base64_string(learn_output_files)
    learner_output_message = {
        'success': learner_success,
        'files': learner_output_zip_stream,
    }

    # Writes the output message in the outputs queue.
    amqp_manager.publish_messages(learner_outputs_queue_name, [learner_output_message])

    # Closes the temporary files.
    properties_file.close()
    training_file.close()

    # Acknowledges the task.
    amqp_manager.acknowledge_messages(learner_tasks_queue_name, delivery_tags)

    # Closes the connection with AMQP.
    amqp_manager.close()

    # Stores the finish time.
    times.insert_one({
        'ip': ip,
        'hostname': hostname,
        'random_seed': task.get('random_seed'),
        'type': 'learner_finish',
        'time': datetime.datetime.utcnow(),
    })


@cli.command()
@click.option('--job', '-j', 'job_name', type=str, required=True)
def filter(job_name):
    """
    Filters the models and sends the selected to the Fuser queue.

    Manages the exeucutor using information from the environment variables and the tasks queue.

    It requires the following environment variables:
    - CCUBE_PREDICT_COMMAND: the shell string used to execute the model
    - CCUBE_PREDICT_WORKING_DIRECTORY: the working the directory in which executing the command
    - CCUBE_PREDICT_PREDICTIONS_FILE: the file that will contain the predictions after the execution

    It fills some environment variables to let the user using them in the CCUBE_LEARN_COMMAND:
    - CCUBE_PREDICT_INPUT_FILES: the directory in which CCUBE will extract the models
    - CCUBE_PREDICT_DATASET_FILE: the file path of the dataset split
    - CCUBE_PREDICT_PARAMETERS_PROPERTIES_FILE: the .properties file filled with the executor parameters for the task

    Moreover, it automatically generates the environment variables for the filter parameters included in the task,
    with the same provided names.

    :param job_name: the name of the job
    :type job_name: str
    """
    filter_start_time = datetime.datetime.utcnow()

    # Get ip and hostname
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)

    # MongoDB connection.
    mongodb_client = pymongo.MongoClient('mongodb', 27017)
    database = mongodb_client.ccube
    times = database.times

    # Reads the environment variables.
    environment_variables = os.environ.copy()

    predict_command = environment_variables.get(PREDICT_COMMAND_VARIABLE_NAME)
    predict_working_directory = environment_variables.get(PREDICT_WORKING_DIRECTORY_VARIABLE_NAME)

    predict_predictions_file = environment_variables.get(PREDICT_PREDICTIONS_FILE_VARIABLE_NAME, "")
    predict_predictions_file = os.path.expandvars(predict_predictions_file)

    # Retrieves the queues names.
    learner_outputs_queue_name = LEARN_OUTPUTS_QUEUE_NAME.format(job_name_=job_name)
    filter_tasks_queue_name = FILTER_TASKS_QUEUE_NAME.format(job_name_=job_name)
    filter_outputs_queue_name = FILTER_OUTPUTS_QUEUE_NAME.format(job_name_=job_name)

    # Creates the queues, if they do not exist.
    amqp_manager = AMQPManager(environment_variables.get('AMQP_HOSTNAME', 'rabbitmq'))
    amqp_manager.create_queue(filter_tasks_queue_name)
    amqp_manager.create_queue(filter_outputs_queue_name)

    # Consumes a task.
    tasks, filter_tasks_delivery_tags = amqp_manager.consume_messages(filter_tasks_queue_name, 1)
    task = tasks[0]

    # Stores the start time.
    times.insert_one({
        'ip': ip,
        'hostname': hostname,
        'random_seed': task.get('random_seed'),
        'type': 'filter_start',
        'time': filter_start_time,
    })

    # Gets the fusion split.
    request_params = {
        'training_rate': task.get('training_rate'),
        'fusion_rate': task.get('fusion_rate'),
        'class_attribute': task.get('class_attribute'),
        'include_attributes': task.get('include_attributes'),
        'exclude_attributes': task.get('exclude_attributes'),
        'attributes_rate': task.get('attributes_rate'),
        'random_seed': task.get('random_seed'),
        'include_header': task.get('include_header'),
    }

    response = requests.get(
        GET_DATASET_FUSION_SPLIT_REQUEST_URL.format(
            hostname_=environment_variables.get('FACTORIZER_HOSTNAME', 'factorizer'),
            port_=environment_variables.get('FACTORIZER_PORT', '5000'),
            dataset_name_=task.get('dataset_name'),
        ),
        params=request_params,
        stream=True,
    )

    # Saves the response into a temporary file.
    fusion_file = tempfile.NamedTemporaryFile()
    for chunk in response.iter_content(chunk_size=DATA_CHUNK_SIZE):
        if chunk:
            fusion_file.write(chunk)
    fusion_file.flush()
    fusion_file.seek(0)

    # Gets the fusion split class column, the actual values.
    request_params = {
        'training_rate': task.get('training_rate'),
        'fusion_rate': task.get('fusion_rate'),
        'class_attribute': task.get('class_attribute'),
        'random_seed': task.get('random_seed'),
        'include_header': task.get('include_header'),
    }

    response = requests.get(
        GET_DATASET_FUSION_SPLIT_CLASS_REQUEST_URL.format(
            hostname_=environment_variables.get('FACTORIZER_HOSTNAME', 'factorizer'),
            port_=environment_variables.get('FACTORIZER_PORT', '5000'),
            dataset_name_=task.get('dataset_name'),
        ),
        params=request_params,
        stream=True,
    )

    # Retrieves the actual values.
    actual_values = utils.read_values_from_file(io.BytesIO(response.content))

    # Extracts the predict parameters.
    predict_parameters = task.get('predict_parameters')

    # Consumes the learner outputs.
    learner_outputs_number = task.get('learner_outputs_number')
    learner_outputs, learner_outputs_delivery_tags = amqp_manager.consume_messages(learner_outputs_queue_name, learner_outputs_number)

    # Iterates all the learner outputs.
    learner_output_zip_streams = []
    models_executors = []

    for learner_output in learner_outputs:
        if not learner_output.get('success'):
            continue

        # Creates the model executor.
        models_executors.append(
            ModelExecutor(
                model_files_stream=learner_output.get('files'),
                model_parameters=predict_parameters,
                predictions_file_path=predict_predictions_file,
                execution_command=predict_command,
                working_directory=predict_working_directory,
                environment_variables=environment_variables,
            )
        )

        # Saves the learner output zip stream.
        learner_output_zip_streams.append(learner_output.get('files'))

    # Applies the filter policy.
    filter_policy = MajorityCostThresholdFilterPolicy(
        models_executors=models_executors,
        training_data=fusion_file.name,
        actual_values=actual_values,
        true_class_value=task.get('true_class_value'),
        class_attribute_type=task.get('class_attribute_type'),
        configuration={'threshold': task.get('threshold')},
    )
    filtered_models = filter_policy.filter()

    # Sends the filtered model to the fuser.
    filtered_learner_output_zip_streams = []
    for i in range(len(filtered_models)):
        if filtered_models[i]:
            filtered_learner_output_zip_streams.append(learner_output_zip_streams[i])

    filter_output_message = [{'files': stream} for stream in filtered_learner_output_zip_streams]

    amqp_manager.publish_messages(filter_outputs_queue_name, [filter_output_message])

    # Closes the temporary files.
    fusion_file.close()

    # Acknowledges the task.
    amqp_manager.acknowledge_messages(filter_tasks_queue_name, filter_tasks_delivery_tags)
    amqp_manager.acknowledge_messages(learner_outputs_queue_name, learner_outputs_delivery_tags)

    # Closes the connection with AMQP.
    amqp_manager.close()

    # Stores the finish time.
    times.insert_one({
        'ip': ip,
        'hostname': hostname,
        'random_seed': task.get('random_seed'),
        'type': 'filter_finish',
        'time': datetime.datetime.utcnow(),
    })


@cli.command()
@click.option('--job', '-j', 'job_name', type=str, required=True)
def fuser(job_name):
    """
    Fuses the models and sends the ensemble.

    Manages the exeucutor using information from the environment variables and the tasks queue.

    It requires the following environment variables:
    - CCUBE_PREDICT_COMMAND: the shell string used to execute the model
    - CCUBE_PREDICT_WORKING_DIRECTORY: the working the directory in which executing the command
    - CCUBE_PREDICT_PREDICTIONS_FILE: the file that will contain the predictions after the execution

    It fills some environment variables to let the user using them in the CCUBE_LEARN_COMMAND:
    - CCUBE_PREDICT_INPUT_FILES: the directory in which CCUBE will extract the models
    - CCUBE_PREDICT_DATASET_FILE: the file path of the dataset split
    - CCUBE_PREDICT_PARAMETERS_PROPERTIES_FILE: the .properties file filled with the executor parameters for the task

    Moreover, it automatically generates the environment variables for the filter parameters included in the task,
    with the same provided names.

    :param job_name: the name of the job
    :type job_name: str
    """
    fuser_start_time = datetime.datetime.utcnow()

    # Get ip and hostname
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)

    # MongoDB connection.
    mongodb_client = pymongo.MongoClient('mongodb', 27017)
    database = mongodb_client.ccube
    times = database.times
    models = database.models

    # Reads the environment variables.
    environment_variables = os.environ.copy()

    predict_command = environment_variables.get(PREDICT_COMMAND_VARIABLE_NAME)
    predict_working_directory = environment_variables.get(PREDICT_WORKING_DIRECTORY_VARIABLE_NAME)

    predict_predictions_file = environment_variables.get(PREDICT_PREDICTIONS_FILE_VARIABLE_NAME, "")
    predict_predictions_file = os.path.expandvars(predict_predictions_file)

    # Retrieves the queues names.
    filter_outputs_queue_name = FILTER_OUTPUTS_QUEUE_NAME.format(job_name_=job_name)
    fuser_tasks_queue_name = FUSER_TASKS_QUEUE_NAME.format(job_name_=job_name)

    # Creates the queues, if they do not exist.
    amqp_manager = AMQPManager(environment_variables.get('AMQP_HOSTNAME', 'rabbitmq'))
    amqp_manager.create_queue(filter_outputs_queue_name)
    amqp_manager.create_queue(fuser_tasks_queue_name)

    # Consumes a task.
    tasks, fuser_tasks_delivery_tags = amqp_manager.consume_messages(fuser_tasks_queue_name, 1)
    task = tasks[0]

    # Stores the start time.
    times.insert_one({
        'ip': ip,
        'hostname': hostname,
        'random_seed': task.get('random_seed'),
        'type': 'fuser_start',
        'time': fuser_start_time,
    })

    # Gets the fusion split.
    fusion_response = requests.get(
        GET_DATASET_FUSION_SPLIT_REQUEST_URL.format(
            hostname_=environment_variables.get('FACTORIZER_HOSTNAME', 'factorizer'),
            port_=environment_variables.get('FACTORIZER_PORT', '5000'),
            dataset_name_=task.get('dataset_name'),
        ),
        params={
            'training_rate': task.get('training_rate'),
            'fusion_rate': task.get('fusion_rate'),
            'class_attribute': task.get('class_attribute'),
            'include_attributes': task.get('include_attributes'),
            'exclude_attributes': task.get('exclude_attributes'),
            'attributes_rate': task.get('attributes_rate'),
            'random_seed': task.get('random_seed'),
            'include_header': task.get('include_header'),
        },
        stream=True,
    )
    fusion_file = tempfile.NamedTemporaryFile()
    for chunk in fusion_response.iter_content(chunk_size=DATA_CHUNK_SIZE):
        if chunk:
            fusion_file.write(chunk)
    fusion_file.flush()
    fusion_file.seek(0)

    # Gets the fusion split actual values.
    fusion_response = requests.get(
        GET_DATASET_FUSION_SPLIT_CLASS_REQUEST_URL.format(
            hostname_=environment_variables.get('FACTORIZER_HOSTNAME', 'factorizer'),
            port_=environment_variables.get('FACTORIZER_PORT', '5000'),
            dataset_name_=task.get('dataset_name'),
        ),
        params={
            'training_rate': task.get('training_rate'),
            'fusion_rate': task.get('fusion_rate'),
            'class_attribute': task.get('class_attribute'),
            'random_seed': task.get('random_seed'),
            'include_header': task.get('include_header'),
        },
        stream=True,
    )
    fusion_actual_values = utils.read_values_from_file(io.BytesIO(fusion_response.content))

    # Gets the test split.
    test_response = requests.get(
        GET_DATASET_TEST_SPLIT_REQUEST_URL.format(
            hostname_=environment_variables.get('FACTORIZER_HOSTNAME', 'factorizer'),
            port_=environment_variables.get('FACTORIZER_PORT', '5000'),
            dataset_name_=task.get('dataset_name'),
        ),
        params={
            'training_rate': task.get('training_rate'),
            'fusion_rate': task.get('fusion_rate'),
            'class_attribute': task.get('class_attribute'),
            'include_attributes': task.get('include_attributes'),
            'exclude_attributes': task.get('exclude_attributes'),
            'attributes_rate': task.get('attributes_rate'),
            'random_seed': task.get('random_seed'),
            'include_header': task.get('include_header'),
        },
        stream=True,
    )
    test_file = tempfile.NamedTemporaryFile()
    for chunk in test_response.iter_content(chunk_size=DATA_CHUNK_SIZE):
        if chunk:
            test_file.write(chunk)
    test_file.flush()
    test_file.seek(0)

    # Gets the fusion split actual values.
    test_response = requests.get(
        GET_DATASET_TEST_SPLIT_CLASS_REQUEST_URL.format(
            hostname_=environment_variables.get('FACTORIZER_HOSTNAME', 'factorizer'),
            port_=environment_variables.get('FACTORIZER_PORT', '5000'),
            dataset_name_=task.get('dataset_name'),
        ),
        params={
            'training_rate': task.get('training_rate'),
            'fusion_rate': task.get('fusion_rate'),
            'class_attribute': task.get('class_attribute'),
            'random_seed': task.get('random_seed'),
            'include_header': task.get('include_header'),
        },
        stream=True,
    )
    test_actual_values = utils.read_values_from_file(io.BytesIO(test_response.content))

    # Extracts the predict parameters.
    predict_parameters = task.get('predict_parameters')

    # Consumes one filter output from the filter queue.
    filter_outputs, filter_outputs_delivery_tags = amqp_manager.consume_messages(filter_outputs_queue_name, 1)
    filter_output = filter_outputs[0]

    # Iterates all the filter outputs.
    models_executors = []
    for model_message in filter_output:
        # Creates the model executor.
        models_executors.append(
            ModelExecutor(
                model_files_stream=model_message.get('files'),
                model_parameters=predict_parameters,
                predictions_file_path=predict_predictions_file,
                execution_command=predict_command,
                working_directory=predict_working_directory,
                environment_variables=environment_variables,
            )
        )

    # Applies the fuser policy.
    fuser_policy = VotingFuserPolicy(
        models_executors=models_executors,
        training_data=fusion_file.name,
        training_actual_values=fusion_actual_values,
        test_data=test_file.name,
        test_actual_values=test_actual_values,
        true_class_value=task.get('true_class_value'),
        class_attribute_type=task.get('class_attribute_type'),
        configuration={'threshold': task.get('threshold')},
    )
    metrics = fuser_policy.fuse()

    # Stores the metrics.
    models.insert_one({
        'ip': ip,
        'hostname': hostname,
        'random_seed': task.get('random_seed')
        #'metrics': json.dumps(metrics)
    })

    # Closes the temporary files.
    fusion_file.close()
    test_file.close()

    # Acknowledges the task.
    amqp_manager.acknowledge_messages(fuser_tasks_queue_name, fuser_tasks_delivery_tags)
    amqp_manager.acknowledge_messages(filter_outputs_queue_name, filter_outputs_delivery_tags)

    # Closes the connection with AMQP.
    amqp_manager.close()

    # Stores the finish time.
    times.insert_one({
        'ip': ip,
        'hostname': hostname,
        'random_seed': task.get('random_seed'),
        'type': 'fuser_finish',
        'time': datetime.datetime.utcnow(),
    })


def get_duration_variables(seconds):
    """
    Retrieves the duration environment variables.

    :param seconds: the duration expressed in seconds
    :type seconds: int

    :return: the variables dictionary
    :rtype: dict[str, str]
    """
    return {
        LEARN_DURATION_SECONDS_VARIABLE_NAME: str(seconds),
        LEARN_DURATION_MINUTES_VARIABLE_NAME: str(int(seconds / 60)),
    }


if __name__ == '__main__':
    cli()
