[![Build](https://github.com/ORNL/flowcept/actions/workflows/create-release-n-publish.yml/badge.svg)](https://github.com/ORNL/flowcept/actions/workflows/create-release-n-publish.yml)
[![PyPI](https://badge.fury.io/py/flowcept.svg)](https://pypi.org/project/flowcept)
[![Tests](https://github.com/ORNL/flowcept/actions/workflows/run-tests.yml/badge.svg)](https://github.com/ORNL/flowcept/actions/workflows/run-tests.yml)
[![Code Formatting](https://github.com/ORNL/flowcept/actions/workflows/code-formatting.yml/badge.svg)](https://github.com/ORNL/flowcept/actions/workflows/code-formatting.yml)
[![License: MIT](https://img.shields.io/github/license/ORNL/flowcept)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# FlowCept

FlowCept is a runtime data integration system that empowers any data processing system to capture and query workflow 
provenance data using data observability, requiring minimal or no changes in the target system code. It seamlessly integrates data from multiple workflows, enabling users to comprehend complex, heterogeneous, and large-scale data from various sources in federated environments.

FlowCept is intended to address scenarios where multiple workflows in a science campaign or in an enterprise run and generate 
important data to be analyzed in an integrated manner. Since these workflows may use different data manipulation tools (e.g., provenance or lineage capture tools, database systems, performance profiling tools) or can be executed within
different parallel computing systems (e.g., Dask, Spark, Workflow Management Systems), its key differentiator is the 
capability to seamless and automatically integrate data from various workflows using data observability.
It builds an integrated data view at runtime enabling end-to-end exploratory data analysis and monitoring.
It follows [W3C PROV](https://www.w3.org/TR/prov-overview/) recommendations for its data schema.
It does not require changes in user codes or systems (i.e., instrumentation). All users need to do is to create adapters for their systems or tools, if one is not available yet. 
In addition to observability, we provide instrumentation options for convenience. For example, by adding a `@flowcept_task` decorator on functions, FlowCept will observe their executions when they run. Also, we provide special features for PyTorch modules. Adding `@torch_task` to them will enable extra model inspection to be captured and integrated in the database at runtime.    
 

Currently, FlowCept provides adapters for: [Dask](https://www.dask.org/), [MLFlow](https://mlflow.org/), [TensorBoard](https://www.tensorflow.org/tensorboard), and [Zambeze](https://github.com/ORNL/zambeze). 

See the [Jupyter Notebooks](notebooks) for utilization examples.

See the [Contributing](CONTRIBUTING.md) file for guidelines to contribute with new adapters. Note that we may use the
term 'plugin' in the codebase as a synonym to adapter. Future releases should standardize the terminology to use adapter.


## Install and Setup:

1. Install FlowCept: 

`pip install .[full]` in this directory (or `pip install flowcept[full]`).

For convenience, this will install all dependencies for all adapters. But it can install
dependencies for adapters you will not use. For this reason, you may want to install 
like this: `pip install .[adapter_key1,adapter_key2]` for the adapters we have implemented, e.g., `pip install .[dask]`.
See [extra_requirements](extra_requirements) if you want to install the dependencies individually.
 
2. Start MongoDB and Redis:

To enable the full advantages of FlowCept, one needs to start a Redis and MongoDB instances.
FlowCept uses Redis as its message queue system and Mongo for its persistent database.
For convenience, we set up a [docker-compose file](deployment/compose.yml) deployment file for this. Run `docker-compose -f deployment/compose.yml up`. RabbitMQ is only needed if Zambeze messages are observed, otherwise, feel free to comment out RabbitMQ service in the compose file.

3. Define the settings (e.g., routes and ports) accordingly in the [settings.yaml](resources/settings.yaml) file.
You may need to set the environment variable `FLOWCEPT_SETTINGS_PATH` with the absolute path to the settings file. 

4. Start the observation using the Controller API, as shown in the [Jupyter Notebooks](notebooks).

5. To use FlowCept's Query API, see utilization examples in the notebooks.

### Simple Example with Decorators Instrumentation

In addition to existing adapters to Dask, MLFlow, and others (it's extensible for any system that generates data), FlowCept also offers instrumentation via @decorators. 

```python 
from flowcept import Flowcept, flowcept_task

@flowcept_task
def sum_one(n):
    return n + 1


@flowcept_task
def mult_two(n):
    return n * 2


with Flowcept(workflow_name='test_workflow'):
    n = 3
    o1 = sum_one(n)
    o2 = mult_two(o1)
    print(o2)

print(Flowcept.db.query(filter={"workflow_id": Flowcept.current_workflow_id}))
```



## Performance Tuning for Performance Evaluation

In the settings.yaml file, the following variables might impact interception performance:

```yaml
main_redis:
  buffer_size: 50
  insertion_buffer_time_secs: 5

plugin:
  enrich_messages: false
```

And other variables depending on the Plugin. For instance, in Dask, timestamp creation by workers add interception overhead.
As we evolve the software, other variables that impact overhead appear and we might not stated them in this README file yet.
If you are doing extensive performance evaluation experiments using this software, please reach out to us (e.g., create an issue in the repository) for hints on how to reduce the overhead of our software.

## Install AMD GPU Lib

On the machines that have AMD GPUs, we use the official AMD ROCM library to capture GPU runtime data.

Unfortunately, this library is not available as a pypi/conda package, so you must manually install it. See instructions in the link: https://rocm.docs.amd.com/projects/amdsmi/en/latest/

Here is a summary:

1. Install the AMD drivers on the machine (check if they are available already under `/opt/rocm-*`).
2. Suppose it is /opt/rocm-6.2.0. Then, make sure it has a share/amd_smi subdirectory and pyproject.toml or setup.py in it.
3. Copy the amd_smi to your home directory: `cp -r /opt/rocm-6.2.0/share/amd_smi ~`
4. cd ~/amd_smi
5. In your python environment, do a pip install .

Current code is compatible with this version: amdsmi==24.6.2+2b02a07
Which was installed using Frontier's /opt/rocm-6.2.0/share/amd_smi

## See also

- [Zambeze Repository](https://github.com/ORNL/zambeze)

## Cite us

If you used FlowCept for your research, consider citing our paper.

```
Towards Lightweight Data Integration using Multi-workflow Provenance and Data Observability
R. Souza, T. Skluzacek, S. Wilkinson, M. Ziatdinov, and R. da Silva
19th IEEE International Conference on e-Science, 2023.
```

**Bibtex:**

```latex
@inproceedings{souza2023towards,  
  author = {Souza, Renan and Skluzacek, Tyler J and Wilkinson, Sean R and Ziatdinov, Maxim and da Silva, Rafael Ferreira},
  booktitle = {IEEE International Conference on e-Science},
  doi = {10.1109/e-Science58273.2023.10254822},
  link = {https://doi.org/10.1109/e-Science58273.2023.10254822},
  pdf = {https://arxiv.org/pdf/2308.09004.pdf},
  title = {Towards Lightweight Data Integration using Multi-workflow Provenance and Data Observability},
  year = {2023}
}

```

## Disclaimer & Get in Touch

Please note that this a research software. We encourage you to give it a try and use it with your own stack. We
are continuously working on improving documentation and adding more examples and notebooks, but we are still far from
a good documentation covering the whole system. If you are interested in working with FlowCept in your own scientific
project, we can give you a jump start if you reach out to us. Feel free to [create an issue](https://github.com/ORNL/flowcept/issues/new), 
[create a new discussion thread](https://github.com/ORNL/flowcept/discussions/new/choose) or drop us an email (we trust you'll find a way to reach out to us :wink: ).

## Acknowledgement

This research uses resources of the Oak Ridge Leadership Computing Facility 
at the Oak Ridge National Laboratory, which is supported by the Office of 
Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.
