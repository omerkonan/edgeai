
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput


region = boto3.session.Session().region_name
sess = sagemaker.Session()
bucket_name = sess.default_bucket()
role = "arn:aws:iam::463160973496:role/service-role/AmazonSageMaker-ExecutionRole-20210114T192957"
sklearn_processor = SKLearnProcessor(
    framework_version="0.20.0", role=role, instance_type="ml.m5.xlarge", instance_count=1
)

data_file_name = "dataset.txt"
input_data = "s3://{}/{}".format(bucket_name, data_file_name) 
headers = ['label','axis1','axis2','axis3'] #first column has to be label column
label_list = ['Still','Lifting','Falling','Shaking']
feature_method_name = "SMV"
windows_length = 20
normalization =  True
standardization = True

sklearn_processor.run(
    code="feature_extraction.py",
    inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
    outputs=[
        ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test"),
    ],
    arguments=["--train-ratio", "0.2","--validation-ratio", "0.2","--headers", str(headers),"--label-list", str(label_list),"--feature-method", feature_method_name,\
        "--windows-length", str(windows_length),"--normalization", str(normalization),"--standardization", str(standardization),"--data-file-name", data_file_name],
)

preprocessing_job_description = sklearn_processor.jobs[-1].describe()

output_config = preprocessing_job_description["ProcessingOutputConfig"]
for output in output_config["Outputs"]:
    if output["OutputName"] == "train_data":
        preprocessed_training_data = output["S3Output"]["S3Uri"]
    if output["OutputName"] == "test_data":
        preprocessed_test_data = output["S3Output"]["S3Uri"]