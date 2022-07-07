import subprocess
from datetime import datetime
from typing import Optional
from lightning.app.storage.path import Path
from lightning_app import LightningWork
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel


class SagemakerPyTorchPredictor(LightningWork):

    def __init__(
        self,
        *args,
        id: str,
        script_path: Optional[str] = None,
        instance_type: str = "ml.m4.xlarge",
        parallel=True,
        **kwargs,
    ):
        super().__init__(*args, parallel=parallel, **kwargs)
        self.id = id
        self.script_path = script_path or ""
        self.instance_type = instance_type
        self._predictor = None
        self.version = "0.0.1"
        self.endpoint_name = None

    def run(self, best_model_path: Path):
        subprocess.call(["mv", str(best_model_path), "model.pt"])

        tar_filename = f"{self.id}.tar.gz"
        subprocess.call(["tar", "-czvf", tar_filename, "model.pt", "handler.py"])

        endpoint_name = f"{self.id}-{self.instance_type}-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
        self.endpoint_name = endpoint_name.replace(".", "").replace("_", "")

        sess = boto3.Session()
        sagemaker_session = sagemaker.Session(boto_session=sess)

        def resolve_sm_role():
            client = boto3.client("iam", region_name="us-east-1")
            response_roles = client.list_roles(
                PathPrefix="/",
                # Marker='string',
                MaxItems=999,
            )
            for role in response_roles["Roles"]:
                if role["RoleName"].startswith("AmazonSageMaker-ExecutionRole-"):
                    print("Resolved SageMaker IAM Role to: " + str(role))
                    return role["Arn"]
            raise Exception("Could not resolve what should be the SageMaker role to be used")

        model_data = sagemaker_session.upload_data(path=tar_filename, bucket="pl-flash-data", key_prefix="artefacts")

        pytorch = PyTorchModel(
            model_data=model_data,
            role=resolve_sm_role(),
            entry_point="handler.py",
            image="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.11.0-gpu-py38-cu113-ubuntu20.04-sagemaker",
        )

        pytorch.deploy(
            initial_instance_count=1,
            instance_type=self.instance_type,
            endpoint_name=self.endpoint_name,
            wait=True,
        )