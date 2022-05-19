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
from zenml.utils.docker_utils import get_image_digest
from zenml.utils.source_utils import get_source_root_path

if TYPE_CHECKING:
    from zenml.pipelines.base_pipeline import BasePipeline
    from zenml.runtime_configuration import RuntimeConfiguration

logger = get_logger(__name__)


class AWSStepFunctionOrchestrator(BaseOrchestrator):
    """Orchestrator responsible for running pipelines using AWS Step Functions."""

    custom_docker_base_image_name: Optional[str] = None

    # Class Configuration
    FLAVOR: ClassVar[str] = AWS_STEP_FUNCTION

    def prepare_pipeline_deployment(
        self,
        pipeline: "BasePipeline",
        stack: "Stack",
        runtime_configuration: "RuntimeConfiguration",
    ) -> None:
        """Builds a docker image for the current environment and uploads it to
        a container registry if configured.
        """
        from zenml.utils.docker_utils import build_docker_image

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

        # Get AWS ECR credentials and upload image to container registry
        # Should have AWS credentials set up first
        # ex:
        # export AWS_ACCESS_KEY_ID=youraccesskey
        # export AWS_SECRET_ACCESS_KEY=yoursecretaccesskey
        token = boto3.client("ecr").get_authorization_token()
        username, password = (
            base64.b64decode(
                token["authorizationData"][0]["authorizationToken"]
            )
            .decode()
            .split(":")
        )
        auth_config = {"username": username, "password": password}

        docker_client = DockerClient.from_env()
        docker_client.images.push(image_name, auth_config=auth_config)

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

        image_name = self.get_docker_image_name(pipeline.name)
        image_name = get_image_digest(image_name) or image_name

        # Create a callable for future compilation into a dsl.Pipeline.
        def _construct_kfp_pipeline() -> None:
            """Create a container_op for each step which contains the name
            of the docker image and configures the entrypoint of the docker
            image to run the step.

            Additionally, this gives each container_op information about its
            direct downstream steps.

            If this callable is passed to the `_create_and_write_workflow()`
            method of a KFPCompiler all dsl.ContainerOp instances will be
            automatically added to a singular dsl.Pipeline instance.
            """

            # Dictionary of container_ops index by the associated step name
            step_name_to_container_op: Dict[str, dsl.ContainerOp] = {}

            for step in sorted_steps:
                # The command will be needed to eventually call the python step
                # within the docker container
                command = StepEntrypointConfiguration.get_entrypoint_command()

                # The arguments are passed to configure the entrypoint of the
                # docker container when the step is called.
                metadata_ui_path = "/outputs/mlpipeline-ui-metadata.json"
                arguments = (
                    StepEntrypointConfiguration.get_entrypoint_arguments(
                        step=step,
                        pb2_pipeline=pb2_pipeline,
                    )
                )

                # Create a container_op - the kubeflow equivalent of a step. It
                # contains the name of the step, the name of the docker image,
                # the command to use to run the step entrypoint
                # (e.g. `python -m zenml.entrypoints.step_entrypoint`)
                # and the arguments to be passed along with the command. Find
                # out more about how these arguments are parsed and used
                # in the base entrypoint `run()` method.
                container_op = dsl.ContainerOp(
                    name=step.name,
                    image=image_name,
                    command=command,
                    arguments=arguments,
                    output_artifact_paths={
                        "mlpipeline-ui-metadata": metadata_ui_path,
                    },
                )

                # Mounts persistent volumes, configmaps and adds labels to the
                # container op
                self._configure_container_op(container_op=container_op)

                # Find the upstream container ops of the current step and
                # configure the current container op to run after them
                upstream_step_names = self.get_upstream_step_names(
                    step=step, pb2_pipeline=pb2_pipeline
                )
                for upstream_step_name in upstream_step_names:
                    upstream_container_op = step_name_to_container_op[
                        upstream_step_name
                    ]
                    container_op.after(upstream_container_op)

                # Update dictionary of container ops with the current one
                step_name_to_container_op[step.name] = container_op

        # Get a filepath to use to save the finished yaml to
        assert runtime_configuration.run_name
        fileio.makedirs(self.pipeline_directory)
        pipeline_file_path = os.path.join(
            self.pipeline_directory, f"{runtime_configuration.run_name}.yaml"
        )

        # write the argo pipeline yaml
        KFPCompiler()._create_and_write_workflow(
            pipeline_func=_construct_kfp_pipeline,
            pipeline_name=pipeline.name,
            package_path=pipeline_file_path,
        )
        connection_config = (
            Repository().active_stack.metadata_store.get_tfx_metadata_config()
        )

        logger.debug(f"Using deployment config:\n {deployment_config}")
        logger.debug(f"Using connection config:\n {connection_config}")

        # AWS STEP FUNCTION
        import stepfunctions
        import logging

        from stepfunctions.steps import *
        from stepfunctions.steps import ModelStep
        from stepfunctions.workflow import Workflow

        stepfunctions.set_stream_logger(level=logging.INFO)

        workflow_execution_role = "<execution-role-arn>"  # paste the AmazonSageMaker-StepFunctionsWorkflowExecutionRole ARN from above

        start_pass_state = Pass(state_id="MyPassState")
        # First we chain the start pass state
        basic_path = Chain([start_pass_state])

        basic_workflow = Workflow(
            name="MyWorkflow_Simple",
            definition=basic_path,
            role=workflow_execution_role,
        )
        print(basic_workflow.definition.to_json(pretty=True))

        basic_workflow.create()

        import json
        import base64

        def lambda_handler(event, context):
            return {
                "statusCode": 200,
                "input": event["input"],
                "output": base64.b64encode(event["input"].encode()).decode(
                    "UTF-8"
                ),
            }

        lambda_state = LambdaStep(
            state_id="Convert HelloWorld to Base64",
            parameters={
                "FunctionName": lambda_handler,  # replace with the name of the function you created
                "Payload": {"input": "HelloWorld"},
            },
        )

        # Run each component. Note that the pipeline.components list is in
        # topological order.
        for node in pb2_pipeline.nodes:
            pipeline_node: PipelineNode = node.pipeline_node

            # fill out that context
            context_utils.add_context_to_node(
                pipeline_node,
                type_=MetadataContextTypes.STACK.value,
                name=str(hash(json.dumps(stack.dict(), sort_keys=True))),
                properties=stack.dict(),
            )

            # Add all pydantic objects from runtime_configuration to the context
            context_utils.add_runtime_configuration_to_node(
                pipeline_node, runtime_configuration
            )

            # Add pipeline requirements as a context
            requirements = " ".join(sorted(pipeline.requirements))
            context_utils.add_context_to_node(
                pipeline_node,
                type_=MetadataContextTypes.PIPELINE_REQUIREMENTS.value,
                name=str(hash(requirements)),
                properties={"pipeline_requirements": requirements},
            )

            node_id = pipeline_node.node_info.id
            executor_spec = runner_utils.extract_executor_spec(
                deployment_config, node_id
            )
            custom_driver_spec = runner_utils.extract_custom_driver_spec(
                deployment_config, node_id
            )

            p_info = pb2_pipeline.pipeline_info
            r_spec = pb2_pipeline.runtime_spec

            # set custom executor operator to allow custom execution logic for
            # each step
            step = get_step_for_node(
                pipeline_node, steps=list(pipeline.steps.values())
            )
            custom_executor_operators = {
                executable_spec_pb2.PythonClassExecutableSpec: step.executor_operator
            }

            component_launcher = launcher.Launcher(
                pipeline_node=pipeline_node,
                mlmd_connection=metadata.Metadata(connection_config),
                pipeline_info=p_info,
                pipeline_runtime_spec=r_spec,
                executor_spec=executor_spec,
                custom_driver_spec=custom_driver_spec,
                custom_executor_operators=custom_executor_operators,
            )
            stack.prepare_step_run()
            execute_step(component_launcher)
            stack.cleanup_step_run()
