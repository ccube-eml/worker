import unittest
import os
import tempfile
import shutil

from click.testing import CliRunner

from worker import __main__
from worker.amqp_manager import AMQPManager


THIS_DIRECTORY_PATH = os.path.dirname(os.path.abspath(__file__))

JOB_NAME = 'gpfunction'

AMQP_HOSTNAME = 'localhost'

LEARN_TASKS = [
    {
        'job_name': 'gpfunction',
        'task_number': 0,
        'dataset_name': 'higgs',
        'training_rate': 0.5,
        'fusion_rate': 0.3,
        'sample_rate': 0.1,
        'class_attribute': 'label',
        'class_attribute_type': 'integer',
        'true_class_value': '1',
        'include_attributes': [],
        'exclude_attributes': [],
        'attributes_rate': 0.5,
        'random_seed': 0,
        'include_header': False,
        'duration': 60,
        'learn_parameters': {
            'xover_op': 'operator.SinglePointKozaCrossover',
            'external_threads': 4,
            'false_negative_weight': 0.5,
            'pop_size': 500,
        },
    },
]
LEARN_COMMAND = 'java -jar gpfunction.jar ' \
                  '-train ${CCUBE_LEARN_DATASET_FILE} ' \
                  '-minutes ${CCUBE_LEARN_DURATION_MINUTES} ' \
                  '-properties ${CCUBE_LEARN_PARAMETERS_PROPERTIES_FILE}'
LEARN_OUTPUT_FILES = '${CCUBE_LEARN_WORKING_DIRECTORY}/mostAccurate.txt'
LEARN_EXECUTABLE_FILE = 'resources/learners/gpfunction.jar'
LEARN_OUTPUTS = [
    {
        'success': True,
        'files': 'UEsDBBQAAAAIADsDQUpa/W2KrgAAAIABAAAQAAAAbW9zdEFjY3VyYXRlLnR4dI1Oyw6CMBC8+xV7\nbBFJX3S7f9KrAjEkIipi5O8tDxESTZyk3dnHTIZtgRXPC7BQd8CqLi8fZV6AN+Cl4cAi8BpYc233\nt2K4qrpTfZyWTXnuZx4HcdYewklWN2ufT7Ng0Sj2wWV4cvoCQuUjAgv0nS3q26W1HhW/M84R+839\njwvv+AzwxHksErJKo0mdJE1IZL6PFK2hw8waic4YS4KkIMRYi0RblUpKSSE6KXHzAlBLAQIUAxQA\nAAAIADsDQUpa/W2KrgAAAIABAAAQAAAAAAAAAAAAAACkgQAAAABtb3N0QWNjdXJhdGUudHh0UEsF\nBgAAAAABAAEAPgAAANwAAAAAAA==\n',
    },
    {
        'success': False,
        'files': False,
    },
    {
        'success': True,
        'files': 'UEsDBBQAAAAIAEuWQUoxobvZzAAAAAACAAAQAAAAbW9zdEFjY3VyYXRlLnR4dG1QSQ7CMAy884oc\nE7qoztIkP8kVaIUqAaUbgt+TrQW1zcGKPR6PxzhBOEN4joYjfGkHhIeuH+13OtcePHpk6KZTbwv3\nT9W8mqqODAOczL0hLg0WsrTm4cc79ujAW3u1PEKQYSQ+hOv3M9IdFIph8r+cm7POw9pJTJ3kmucs\nGaBB8JcuChvL22Uif7/dX2jtfcevP4U3vRwzbOT05mOE0sqCAQgrp0WulaKaM+BMAZWF3i+VaSZy\nyQQwSUtBFRRapTyXSpa0ECCFUpofvlBLAQIUAxQAAAAIAEuWQUoxobvZzAAAAAACAAAQAAAAAAAA\nAAAAAACkgQAAAABtb3N0QWNjdXJhdGUudHh0UEsFBgAAAAABAAEAPgAAAPoAAAAAAA==\n',
    },
]

FILTER_TASKS = [
    {
        'job_name': 'gpfunction',
        'learner_outputs_number': 3,
        'dataset_name': 'higgs',
        'training_rate': 0.5,
        'fusion_rate': 0.3,
        'sample_rate': 0.1,
        'class_attribute': 'label',
        'class_attribute_type': 'integer',
        'true_class_value': '1',
        'include_attributes': [],
        'exclude_attributes': [],
        'attributes_rate': 0.5,
        'random_seed': 0,
        'include_header': False,
        'threshold': 0.47,
        'predict_parameters': None,
    },
]

FUSER_TASKS = [
    {
        'job_name': 'gpfunction',
        'dataset_name': 'higgs',
        'training_rate': 0.5,
        'fusion_rate': 0.3,
        'sample_rate': 0.1,
        'class_attribute': 'label',
        'class_attribute_type': 'integer',
        'true_class_value': '1',
        'include_attributes': [],
        'exclude_attributes': [],
        'attributes_rate': 0.5,
        'random_seed': 0,
        'include_header': False,
        'predict_parameters': None,
    },
]

PREDICT_COMMAND = 'java -jar gpfunction.jar ' \
                  '-predict ${CCUBE_PREDICT_DATASET_FILE} ' \
                  '-model ${CCUBE_PREDICT_INPUT_FILES}/mostAccurate.txt ' \
                  '-o predictions.csv'
PREDICT_PREDICTIONS_FILE = '${CCUBE_PREDICT_WORKING_DIRECTORY}/predictions.csv'
PREDICT_EXECUTABLE_FILE = 'resources/learners/gpfunction.jar'

FACTORIZER_HOSTNAME = 'localhost'
FACTORIZER_PORT = '5000'

FILTER_OUTPUTS = [
    [
        {
            'files': 'UEsDBBQAAAAIADsDQUpa/W2KrgAAAIABAAAQAAAAbW9zdEFjY3VyYXRlLnR4dI1Oyw6CMBC8+xV7\nbBFJX3S7f9KrAjEkIipi5O8tDxESTZyk3dnHTIZtgRXPC7BQd8CqLi8fZV6AN+Cl4cAi8BpYc233\nt2K4qrpTfZyWTXnuZx4HcdYewklWN2ufT7Ng0Sj2wWV4cvoCQuUjAgv0nS3q26W1HhW/M84R+839\njwvv+AzwxHksErJKo0mdJE1IZL6PFK2hw8waic4YS4KkIMRYi0RblUpKSSE6KXHzAlBLAQIUAxQA\nAAAIADsDQUpa/W2KrgAAAIABAAAQAAAAAAAAAAAAAACkgQAAAABtb3N0QWNjdXJhdGUudHh0UEsF\nBgAAAAABAAEAPgAAANwAAAAAAA==\n'
        },
        {
            'files': 'UEsDBBQAAAAIAEuWQUoxobvZzAAAAAACAAAQAAAAbW9zdEFjY3VyYXRlLnR4dG1QSQ7CMAy884oc\nE7qoztIkP8kVaIUqAaUbgt+TrQW1zcGKPR6PxzhBOEN4joYjfGkHhIeuH+13OtcePHpk6KZTbwv3\nT9W8mqqODAOczL0hLg0WsrTm4cc79ujAW3u1PEKQYSQ+hOv3M9IdFIph8r+cm7POw9pJTJ3kmucs\nGaBB8JcuChvL22Uif7/dX2jtfcevP4U3vRwzbOT05mOE0sqCAQgrp0WulaKaM+BMAZWF3i+VaSZy\nyQQwSUtBFRRapTyXSpa0ECCFUpofvlBLAQIUAxQAAAAIAEuWQUoxobvZzAAAAAACAAAQAAAAAAAA\nAAAAAACkgQAAAABtb3N0QWNjdXJhdGUudHh0UEsFBgAAAAABAAEAPgAAAPoAAAAAAA==\n'
        },
        {
            'files': 'UEsDBBQAAAAIAEuWQUoxobvZzAAAAAACAAAQAAAAbW9zdEFjY3VyYXRlLnR4dG1QSQ7CMAy884oc\nE7qoztIkP8kVaIUqAaUbgt+TrQW1zcGKPR6PxzhBOEN4joYjfGkHhIeuH+13OtcePHpk6KZTbwv3\nT9W8mqqODAOczL0hLg0WsrTm4cc79ujAW3u1PEKQYSQ+hOv3M9IdFIph8r+cm7POw9pJTJ3kmucs\nGaBB8JcuChvL22Uif7/dX2jtfcevP4U3vRwzbOT05mOE0sqCAQgrp0WulaKaM+BMAZWF3i+VaSZy\nyQQwSUtBFRRapTyXSpa0ECCFUpofvlBLAQIUAxQAAAAIAEuWQUoxobvZzAAAAAACAAAQAAAAAAAA\nAAAAAACkgQAAAABtb3N0QWNjdXJhdGUudHh0UEsFBgAAAAABAAEAPgAAAPoAAAAAAA==\n'
        },
    ]
]


class GPFunctionTest(unittest.TestCase):
    def setUp(self):
        self.__amqp_manager = AMQPManager(AMQP_HOSTNAME)

        self.__learner_tasks_queue_name = __main__.LEARN_TASKS_QUEUE_NAME.format(job_name_=JOB_NAME)
        self.__learner_outputs_queue_name = __main__.LEARN_OUTPUTS_QUEUE_NAME.format(job_name_=JOB_NAME)
        self.__amqp_manager.delete_queue(self.__learner_tasks_queue_name)
        self.__amqp_manager.delete_queue(self.__learner_outputs_queue_name)

        self.__filter_tasks_queue_name = __main__.FILTER_TASKS_QUEUE_NAME.format(job_name_=JOB_NAME)
        self.__filter_outputs_queue_name = __main__.FILTER_OUTPUTS_QUEUE_NAME.format(job_name_=JOB_NAME)
        self.__amqp_manager.delete_queue(self.__filter_tasks_queue_name)
        self.__amqp_manager.delete_queue(self.__filter_outputs_queue_name)

        self.__fuser_tasks_queue_name = __main__.FUSER_TASKS_QUEUE_NAME.format(job_name_=JOB_NAME)
        self.__amqp_manager.delete_queue(self.__fuser_tasks_queue_name)

        self.__runner = CliRunner()

        self.__temporary_working_directory = tempfile.TemporaryDirectory()
        shutil.copy(os.path.join(THIS_DIRECTORY_PATH, LEARN_EXECUTABLE_FILE), self.__temporary_working_directory.name)

    def tearDown(self):
        self.__temporary_working_directory.cleanup()

        self.__amqp_manager.delete_queue(self.__learner_tasks_queue_name)
        self.__amqp_manager.delete_queue(self.__learner_outputs_queue_name)
        self.__amqp_manager.delete_queue(self.__filter_tasks_queue_name)
        self.__amqp_manager.delete_queue(self.__filter_outputs_queue_name)
        self.__amqp_manager.delete_queue(self.__fuser_tasks_queue_name)

    def test_learn(self):
        # Publishes the tasks.
        self.__amqp_manager.create_queue(self.__learner_tasks_queue_name)
        self.__amqp_manager.publish_messages(self.__learner_tasks_queue_name, LEARN_TASKS)

        # Adds the environment variables.
        os.environ['AMQP_HOSTNAME'] = AMQP_HOSTNAME
        os.environ['FACTORIZER_HOSTNAME'] = FACTORIZER_HOSTNAME
        os.environ['FACTORIZER_PORT'] = FACTORIZER_PORT
        os.environ['CCUBE_LEARN_COMMAND'] = LEARN_COMMAND
        os.environ['CCUBE_LEARN_WORKING_DIRECTORY'] = self.__temporary_working_directory.name
        os.environ['CCUBE_LEARN_OUTPUT_FILES'] = LEARN_OUTPUT_FILES

        result = self.__runner.invoke(
            __main__.cli,
            [
                'learn',
                '--job', JOB_NAME,
            ],
            catch_exceptions=False,
        )
        print(result.output)

    def test_filter(self):
        # Publishes the tasks.
        self.__amqp_manager.create_queue(self.__learner_outputs_queue_name)
        self.__amqp_manager.publish_messages(self.__learner_outputs_queue_name, LEARN_OUTPUTS)

        self.__amqp_manager.create_queue(self.__filter_tasks_queue_name)
        self.__amqp_manager.publish_messages(self.__filter_tasks_queue_name, FILTER_TASKS)

        # Adds the environment variables.
        os.environ['AMQP_HOSTNAME'] = AMQP_HOSTNAME
        os.environ['FACTORIZER_HOSTNAME'] = FACTORIZER_HOSTNAME
        os.environ['FACTORIZER_PORT'] = FACTORIZER_PORT
        os.environ['CCUBE_PREDICT_COMMAND'] = PREDICT_COMMAND
        os.environ['CCUBE_PREDICT_WORKING_DIRECTORY'] = self.__temporary_working_directory.name
        os.environ['CCUBE_PREDICT_PREDICTIONS_FILE'] = PREDICT_PREDICTIONS_FILE

        result = self.__runner.invoke(
            __main__.cli,
            [
                'filter',
                '--job', JOB_NAME,
            ],
            catch_exceptions=False,
        )
        print(result.output)

    def test_fuser(self):
        # Publishes the tasks.
        self.__amqp_manager.create_queue(self.__filter_outputs_queue_name)
        self.__amqp_manager.publish_messages(self.__filter_outputs_queue_name, FILTER_OUTPUTS)

        self.__amqp_manager.create_queue(self.__fuser_tasks_queue_name)
        self.__amqp_manager.publish_messages(self.__fuser_tasks_queue_name, FUSER_TASKS)

        # Adds the environment variables.
        os.environ['AMQP_HOSTNAME'] = AMQP_HOSTNAME
        os.environ['FACTORIZER_HOSTNAME'] = FACTORIZER_HOSTNAME
        os.environ['FACTORIZER_PORT'] = FACTORIZER_PORT
        os.environ['CCUBE_PREDICT_COMMAND'] = PREDICT_COMMAND
        os.environ['CCUBE_PREDICT_WORKING_DIRECTORY'] = self.__temporary_working_directory.name
        os.environ['CCUBE_PREDICT_PREDICTIONS_FILE'] = PREDICT_PREDICTIONS_FILE

        result = self.__runner.invoke(
            __main__.cli,
            [
                'fuser',
                '--job', JOB_NAME,
            ],
            catch_exceptions=False,
        )
        print(result.output)
