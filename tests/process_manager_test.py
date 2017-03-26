import unittest
import os

from worker.process_manager import ProcessManager


class ProcessManagerTest(unittest.TestCase):
    def setUp(self):
        self.__process_manager = ProcessManager(
            'printenv',
            '.',
            os.environ.copy(),
        )

    def tearDown(self):
        pass

    def test_run(self):
        self.__process_manager.add_environment_variables(
            {
                'CCUBE_PROCESS_MANAGER_VARIABLE_1': '1',
                'CCUBE_PROCESS_MANAGER_VARIABLE_2': '2',
            }
        )

        stdout, return_code = self.__process_manager.run()

        self.assertIn('CCUBE_PROCESS_MANAGER_VARIABLE_1=1', stdout)
        self.assertIn('CCUBE_PROCESS_MANAGER_VARIABLE_1=1', stdout)
