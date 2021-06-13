from sagemaker.tensorflow import TensorFlow
from sagemaker.session import Session
from .exceptions import *
from .pre_processor import Preprocessor

AI_JOB_BUCKET = "empacloud-ai-jobs"


class Sagemaker:
    def __init__(self, environment: str, job_id: str, process_id: str, network_type: str, custom_layer=None,
                 file_key=None, epochs=50, batch_size=64,
                 learning_rate=0.01):

        self.environment = environment
        self.job_id = job_id
        self.network_type = network_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.process_id = process_id
        self.learning_rate = learning_rate
        self.custom_layer = custom_layer
        self.hyperparameters = None
        self.file_key = file_key

        if not process_id and not file_key:
            raise InvalidSagemakerParameterException()

        if not process_id and file_key:
            self.process_id = Preprocessor(
                file_key=file_key,
                environment=self.environment,
                job_id=self.job_id
            ).save()

        if self.custom_layer is None:
            self.custom_layer = []

        self._initialize_hyperparameters()
        self._create_estimator(
            base_job_name=self.process_id,
            hyperparameters=self.hyperparameters
        )

        print(self.estimator)

    def _initialize_hyperparameters(self):
        self.hyperparameters = {
            "epochs": self.epochs,
            "batch-size": self.batch_size,
            "learning-rate": self.learning_rate,
            "network_type": self.network_type,
            "custom_layer": self.custom_layer
        }

        if self.network_type == "CUSTOM" and not self.custom_layer:
            raise LayerInfoNotFoundException()

    def _create_estimator(self, base_job_name, hyperparameters, instance_count=1, instance_type='ml.m5.large'):
        self.estimator = TensorFlow(base_job_name=base_job_name,
                                    entry_point="chalicelib/modules/ai/main.py",
                                    instance_count=instance_count,
                                    instance_type=instance_type,
                                    framework_version='1.12',
                                    sagemaker_session=Session(default_bucket="empacloud-sagemaker-builds"),
                                    role="arn:aws:iam::742020528822:role/Empacloud-AI-Sagemaker-Role",
                                    py_version='py3',
                                    script_mode=True,
                                    output_data_path='s3://{}/{}/{}/MODELS/checkpoints/'.format(AI_JOB_BUCKET,
                                                                                                self.environment,
                                                                                                self.job_id),
                                    hyperparameters=hyperparameters
                                    )

    def start_modelling(self):
        return self.estimator.fit(
            inputs={"input_path": f"s3://{AI_JOB_BUCKET}/{self.environment}/{self.job_id}/PROCESSED/{self.process_id}"},
            wait=False
        )
