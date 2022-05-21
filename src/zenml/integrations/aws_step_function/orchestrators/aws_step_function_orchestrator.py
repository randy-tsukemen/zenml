#  Copyright (c) ZenML GmbH 2021. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
import os
import sys
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple
from uuid import UUID
import base64, boto3
from docker.client import DockerClient

from tfx.proto.orchestration.pipeline_pb2 import Pipeline as Pb2Pipeline

from zenml.entrypoints import StepEntrypointConfiguration
from zenml.environment import Environment
from zenml.integrations.constants import AWS_STEP_FUNCTION
from zenml.io import fileio
from zenml.logger import get_logger
from zenml.orchestrators import BaseOrchestrator
from zenml.repository import Repository
from zenml.stack import Stack
from zenml.steps import BaseStep
from zenml.utils.docker_utils import get_image_digest, build_docker_image
from zenml.utils.source_utils import get_source_root_path

import logging

import sagemaker
import stepfunctions
from stepfunctions.steps import TrainingStep, Chain
from stepfunctions.workflow import Workflow

stepfunctions.set_stream_logger(level=logging.INFO)

if TYPE_CHECKING:
    from zenml.pipelines.base_pipeline import BasePipeline
    from zenml.runtime_configuration import RuntimeConfiguration

logger = get_logger(__name__)


class AWSStepFunctionOrchestrator(BaseOrchestrator):
    """Orchestrator responsible for running pipelines using AWS Step Functions."""

    sagemaker_role: str = "arn:aws:iam::536079580069:role/service-role/AmazonSageMaker-ExecutionRole-20220521T140815"
    workflow_execution_role: str = "arn:aws:iam::536079580069:role/AmazonSageMaker-StepFunctionsWorkflowExecutionRole"
    instance_type: str = "ml.t3.medium"

    custom_docker_base_image_name: Optional[str] = None
    # Class Configuration
    FLAVOR: ClassVar[str] = AWS_STEP_FUNCTION

    def get_docker_image_name(
        self, pipeline_name: str, step_name: Optional[str]
    ) -> str:
        """Returns the full docker image name including registry and tag."""

        base_image_name = f"zenml-awsstepfunction-{pipeline_name}:"
        if step_name:
            tag = f"{step_name}"
        else:
            tag = "pipeline"
        base_image_name += tag

        container_registry = Repository().active_stack.container_registry

        if container_registry:
            registry_uri = container_registry.uri.rstrip("/")
            return f"{registry_uri}/{base_image_name}"
        else:
            return base_image_name

    def prepare_pipeline_deployment(
        self,
        pipeline: "BasePipeline",
        stack: "Stack",
        runtime_configuration: "RuntimeConfiguration",
    ) -> None:
        """Builds a docker image for the current environment and uploads it to
        a container registry if configured.
        """

        image_name = self.get_docker_image_name(pipeline.name)
        requirements = {*stack.requirements(), *pipeline.requirements}
        logger.debug("Docker container requirements: %s", requirements)

        build_docker_image(
            build_context_path=get_source_root_path(),
            image_name=image_name,
            dockerignore_path=pipeline.dockerignore_file,
            requirements=requirements,
            base_image=self.custom_docker_base_image_name,
            environment_vars=self._get_environment_vars_from_secrets(
                pipeline.secrets
            ),
        )

        assert stack.container_registry  # should never happen due to validation

        docker_client = DockerClient.from_env()
        docker_client.images.push(image_name)

    def prepare_or_run_pipeline(
        self,
        sorted_steps: List[BaseStep],
        pipeline: "BasePipeline",
        pb2_pipeline: Pb2Pipeline,
        stack: "Stack",
        runtime_configuration: "RuntimeConfiguration",
    ) -> Any:
        """
        Creates a kfp yaml file as intermediary representation of the
        pipeline which is then deployed to the kubeflow pipelines instance.

        How it works:
        -------------
        Before this method is called the `prepare_pipeline_deployment()`
        method builds a docker image that contains the code for the
        pipeline, all steps the context around these files.

        Based on this docker image a callable is created which builds
        container_ops for each step (`_construct_kfp_pipeline`).
        To do this the entrypoint of the docker image is configured to
        run the correct step within the docker image. The dependencies
        between these container_ops are then also configured onto each
        container_op by pointing at the downstream steps.

        This callable is then compiled into a kfp yaml file that is used as
        the intermediary representation of the kubeflow pipeline.

        This file, together with some metadata, runtime configurations is
        then uploaded into the kubeflow pipelines cluster for execution.
        """

        # First check whether the code running in a notebook
        if Environment.in_notebook():
            raise RuntimeError(
                "The AWSStepFunction orchestrator cannot run pipelines in a notebook "
                "environment. The reason is that it is non-trivial to create "
                "a Docker image of a notebook. Please consider refactoring "
                "your notebook cells into separate scripts in a Python module "
                "and run the code outside of a notebook when using this "
                "orchestrator."
            )

        # Create a callable for future compilation into a dsl.Pipeline.
        # list of states for aws step function to chain together
        states_dag = []

        # The command will be needed to eventually call the python step
        # within the docker container
        session = sagemaker.Session()
        command = StepEntrypointConfiguration.get_entrypoint_command()
        base_image = self.get_docker_image_name(pipeline.name)
        for step in sorted_steps:
            # The arguments are passed to configure the entrypoint of the
            # docker container when the step is called.
            arguments = StepEntrypointConfiguration.get_entrypoint_arguments(
                step=step,
                pb2_pipeline=pb2_pipeline,
            )
            entrypoint = " ".join(command + arguments)

            image_name = self.get_docker_image_name(pipeline.name, step.name)

            build_docker_image(
                build_context_path=get_source_root_path(),
                image_name=image_name,
                entrypoint=entrypoint,
                base_image=base_image,
                environment_vars=self._get_environment_vars_from_secrets(
                    pipeline.secrets
                ),
            )

            estimator = sagemaker.estimator.Estimator(
                image_name,
                self.sagemaker_role,
                instance_count=1,
                instance_type=self.instance_type,
                sagemaker_session=session,
            )

            states_dag.append(
                TrainingStep(
                    state_id=step.name,
                    estimator=estimator,
                    job_name=step.name + self.uuid,
                )
            )

        # Get a filepath to use to save the finished yaml to
        assert runtime_configuration.run_name

        # write the argo pipeline yaml
        # AWS STEP FUNCTION
        # First we chain the start pass state
        Chain_path = Chain(states_dag)

        Chain_workflow = Workflow(
            name=pipeline.name,
            definition=Chain_path,
            role=self.workflow_execution_role,
        )
        print(Chain_workflow.definition.to_json(pretty=True))

        Chain_workflow.create()
