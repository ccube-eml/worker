import subprocess
import sys


class ProcessManager(object):
    """
    Implements a manager for process to be executed in the environment.
    """

    def __init__(
            self,
            command,
            working_directory,
            environment_variables,
    ):
        """
        Initializes the manager.

        :param command: the command to execute
        :type command: str

        :param working_directory: the working directory for the command
        :type working_directory: str

        :param environment_variables: the environment variables starting set
        :type environment_variables: dict[str, str]
        """
        self.__command = command
        self.__working_directory = working_directory
        self.__environment_variables = environment_variables

    @property
    def environment_variables(self):
        """
        Returns the current set of environment variables.

        :return: the environment variables
        :rtype: dict[str, str]
        """
        return self.__environment_variables

    def add_environment_variables(
            self,
            variables,
    ):
        """
        Adds the variables to the environment variables already set.

        :param variables: the variables dictionary to add
        :type variables: dict[str, str]
        """
        self.__environment_variables.update(variables)

    def run(self):
        """
        Executes the command.

        :return: the STDOUT and STDERR, together with the return code of the command
        """
        process = subprocess.Popen(
            self.__command,
            cwd=self.__working_directory,
            env=self.__environment_variables,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
        )

        stdout = ''
        for line in iter(process.stdout.readline, b''):
            line = str(line, 'utf-8')
            stdout += line
            print(line)
            sys.stdout.flush()

        return_code = process.wait()

        return stdout, return_code

    def echo(self):
        process = subprocess.Popen(
            'echo ' + self.__command,
            cwd=self.__working_directory,
            env=self.__environment_variables,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
        )

        stdout = ''
        for line in iter(process.stdout.readline, b''):
            line = str(line, 'utf-8')
            stdout += line
            print(line)
            sys.stdout.flush()

        return_code = process.wait()

        return stdout, return_code
