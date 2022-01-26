# Profiling Datasets with WhyLogs/Whylabs

Data logging and profiling is a an important part of any production ML
pipeline. [Whylogs](https://whylabs.ai/whylogs) is an open source library
that analyzes your data and creates statistical summaries called whylogs
profiles. Whylogs profiles can be visualized locally or uploaded to the
WhyLabs platform where more comprehensive analyses can be carried out.

ZenML integrates seamlessly with Whylogs and WhyLabs. This example shows
how easy it is to enhance steps in an existing ML pipeline with Whylogs
profiling features. Changes to the user code are minimal while ZenML takes
care of all aspects related to Whylogs session initialization, profile
serialization, versioning and persistence and even uploading generated
profiles to Whylabs.

The ZenML Whylogs integration comes in two forms, both showcased in this
example:

* an `enable_whylogs` ZenML pipeline step decorator that enhances an
existing ZenML step with Whylogs profiling capabilities.
* a predefined `WhylogsProfilerStep` ZenML step class that can be
instantiated and inserted into any pipeline to generate a Whylogs profile
out of a Pandas Dataframe and return the profile as a step output artifact.
Instantiating this type of step is simplified even further through the
use of the `whylogs_profiler_step` function.


## How the example is implemented

The ZenML pipeline in this example is rather simple, it consists of a couple
of steps involving some processing of datasets. How these datasets are used in
the pipeline is irrelevant for the example - it could be data ingestion, feature
engineering, data processing, model training and validation or inference. What
is important is how ZenML allows you to effortlessly add Whylogs profiling
capabilities to all the points in your ML pipeline where data is involved.

The first step in the pipeline shows how applying the `enable_whylogs`
decorator to an existing step adds the `whylogs` data profiling extension
to the step context. The Whylogs data profiles are returned as step artifacts
which will be versioned and persisted in the Artifact Store just as any other
artifacts.

```python
from zenml.integrations.whylogs.whylogs_step_decorator import enable_whylogs
from zenml.steps import Output, step

@enable_whylogs
@step(enable_cache=True)
def data_loader(
    context: StepContext,
) -> Output(data=pd.DataFrame, profile=DatasetProfile,):
    ...

    # leverage the whylogs sub-context to generate a whylogs profile
    profile = context.whylogs.profile_dataframe(
        df, dataset_name="input_data", tags={"datasetId": "model-14"}
    )

    return df, profile
```

Additional Whylogs profiling steps can also be created using the
`whylogs_profiler_step` shortcut:

```python
from zenml.integrations.whylogs.steps import whylogs_profiler_step

train_data_profiler = whylogs_profiler_step(
    "train_data_profiler", dataset_name="train", tags={"datasetId": "model-15"}
)
test_data_profiler = whylogs_profiler_step(
    "test_data_profiler", dataset_name="test", tags={"datasetId": "model-16"}
)
```

### Post execution analysis

The ZenML `WhylogsVisualizer` can be used to visualize the Whylogs
profiles persisted in the Artifact Store locally:

```python
def visualize_statistics(step_name: str):
    repo = Repository()
    pipe = repo.get_pipelines()[-1]
    whylogs_outputs = pipe.runs[-1].get_step(name=step_name)
    WhylogsVisualizer().visualize(whylogs_outputs)

visualize_statistics("data_loader")
visualize_statistics("train_data_profiler")
visualize_statistics("test_data_profiler")
```
![WhyLogs visualizer](assets/whylogs-visualizer.png)

Furthermore, all the generated profiles are uploaded to Whylabs
automatically if the Whylabs environment variables are set:

```python
import os
os.environ["WHYLABS_API_KEY"] = "YOUR-API-KEY"
os.environ["WHYLABS_DEFAULT_ORG_ID"] = "YOUR-ORG-ID"
```

The `datasetId` tags set for the profiles are used to associate
the datasets models with the models in the Whylogs platform. 

![WhyLabs UI image 1](assets/whylabs-ui-01.png)
![WhyLabs UI image 2](assets/whylabs-ui-02.png)


## Run it locally

### Pre-requisites
In order to run this example, you need to install and initialize ZenML:

```shell
# install CLI
pip install zenml

# install ZenML integrations
zenml integration install whylogs
zenml integration install sklearn

# pull example
zenml example pull whylogs
cd zenml_examples/whylogs

# initialize
zenml init
```

### Run the project
Now we're ready. Execute:

```shell
python run.py
```

### Clean up
In order to clean up, delete the remaining ZenML references.

```shell
rm -rf zenml_examples
```

## SuperQuick `whylogs` run

If you're really in a hurry and you want just to see this example pipeline run,
without wanting to fiddle around with all the individual installation and
configuration steps, just run the following:

```shell
zenml example run whylogs
```