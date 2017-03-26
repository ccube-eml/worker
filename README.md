# *cCube* worker

The **worker** is a component of *cCube*, the cloud microservices architecture for Evolutionary Machine Learning (EML) classification.
The **worker** provides the code for the **learner**, **filter** and **fuser** microservices as a single container.
An EML researcher intended to use *cCube* has to inject the worker code into its execution environment.
The **worker** acts as daemon inside the container and is in charge of communication with a *cCube* cluster and executing the EML algorithm for learning or prediction by means of environment variables.

An example of use is given in the [GP Function repository](https://github.com/ccube-eml/gpfunction).

## Usage

To inject the *cCube* **worker** code, it is required to define a *Docker* environment into a `.ccube` file, which is actually a `Dockerfile`.
The injection is made by adding the following lines of code:

```docker
RUN curl -sSL https://raw.githubusercontent.com/ccube-eml/worker/master/install.sh | sh
WORKDIR /ccube
ENTRYPOINT ["python3", "-m", "worker"]
```

Moreover, it is required to define the following environment variables:

| Variable | Description |
| --- | --- |
| `CCUBE_LEARN_COMMAND` | The shell string used to launch the learner |
| `CCUBE_LEARN_WORKING_DIRECTORY` | The working the directory in which executing the command |
| `CCUBE_LEARN_OUTPUT_FILES` | The learner output files to include in the final output |
| `CCUBE_PREDICT_COMMAND` | The shell string used to execute the model |
| `CCUBE_PREDICT_WORKING_DIRECTORY` | The working the directory in which executing the command |
| `CCUBE_PREDICT_PREDICTIONS_FILE` | The file that will contain the predictions after the execution |

The EML researcher can also take advantage of the following environment variables that *cCube* automatically fills while running:

| Variable | Description |
| --- | --- |
| `CCUBE_LEARN_TRAINING_FILE` | The file path of the dataset training sample |
| `CCUBE_LEARN_PARAMETERS_PROPERTIES_FILE` | The `.properties` file filled with the learner parameters for the task |
| `CCUBE_LEARN_DURATION_SECONDS` | The duration in seconds, required for the task |
| `CCUBE_LEARN_DURATION_MINUTES` | The duration in minutes, required for the task |
| `CCUBE_PREDICT_INPUT_FILES` | The directory in which CCUBE will extract the models
| `CCUBE_PREDICT_DATASET_FILE` | The file path of the dataset split
| `CCUBE_PREDICT_PARAMETERS_PROPERTIES_FILE` | The `.properties` file filled with the executor parameters for the task

Also, the **worker** automatically generates the environment variables for the learner parameters included in the task, with the same provided names.

A complete template is provided in the [`.ccube.template`](.ccube.template) file.

## License

*cCube* is licensed under the terms of the [MIT License](https://opensource.org/licenses/MIT).
Please see the [LICENSE](LICENSE.md) file for full details.
