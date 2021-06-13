import sagemaker
import pandas as pd
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
import tensorflow as tf
import numpy as np
import os 
from sagemaker.tensorflow import TensorFlow
from data import Data
from time import gmtime, strftime 

def init_session(role = "arn:aws:iam::463160973496:role/service-role/AmazonSageMaker-ExecutionRole-20210114T192957", bucket_name = None ):

  sess = sagemaker.Session()
  role = role #sagemaker.get_execution_role()
  if bucket_name == None:
    bucket_name = sess.default_bucket()
    print("first:", bucket_name)
  else:
    bucket_name = bucket_name
  return sess, role, bucket_name

                         
def create_estimator(base_job_name, role, hyperparameters, bucket_name, entry_point='main.py', \
                     instance_count=1, instance_type='ml.m5.large', source_dir=os.getcwd()):
  tf_estimator = TensorFlow(base_job_name= base_job_name,
                          entry_point=entry_point , 
                          role=role,
                          instance_count = instance_count, 
                          instance_type = instance_type,
                          framework_version='1.12', 
                          py_version='py3',
                          script_mode=True,
                          output_data_path = 's3://{}/{}'.format(bucket_name, "model_training_output"),
                          source_dir = source_dir, 
                          hyperparameters = hyperparameters
                         )
  return tf_estimator

def create_tuner(tf_estimator, objective_metric_name, metric_definitions,hyperparameter_ranges, objective_type='Minimize', max_jobs=1, \
                 max_parallel_jobs=1, early_stopping_type='Auto'):
  tuner = HyperparameterTuner(
      tf_estimator,
      objective_metric_name,
      hyperparameter_ranges,
      metric_definitions,
      max_jobs=1,
      max_parallel_jobs=1,
      strategy='Random',
      base_tuning_job_name="TuningJob",
      early_stopping_type='Auto',
      objective_type=objective_type
  )
  return tuner

def get_hyperparameters(bucket_name, layers_list, epochs = 1, batch_size = 2, learning_rate = 0.01):
  hyperparameters = {"epochs":epochs, "batch-size": batch_size, "learning-rate": learning_rate, "bucket-name": bucket_name, "layers_list": layers_list}
  return hyperparameters

def get_inputs(training_input_path, validation_input_path, test_input_path):
  estimator_inputs = {'training': training_input_path, 'validation': validation_input_path, 'test':test_input_path}
  return estimator_inputs

def get_tuning_params(lr = (0.001, 0.2), er = (50, 100), br = (64, 256)):
  
  """
  lr: learning_range
  er: epoch range
  br: batch size range
  """

  objective_metric_name = 'val_loss'
  objective_type = "Minimize"
  hyperparameter_ranges = {
    'learning_rate': ContinuousParameter(lr[0], lr[1], scaling_type="Logarithmic"),
    'epochs': IntegerParameter(er[0], er[1]),
    'batch_size': IntegerParameter(br[0], br[1]),
  }
  metric_definitions = [{'Name': 'loss',
                        'Regex': ' loss: ([0-9\\.]+)'},
                      {'Name': 'val_loss',
                        'Regex': ' val_loss: ([0-9\\.]+)'}]

  return objective_metric_name, objective_type, hyperparameter_ranges, metric_definitions


def run_instance(base_job_name,tuning_job_name, role, hyperparameters, bucket_name, estimator_inputs, objective_metric_name, \
                 metric_definitions, hyperparameter_ranges, objective_type):
  
  tf_estimator = create_estimator(base_job_name, role, hyperparameters, bucket_name) 
  tf_estimator.fit(estimator_inputs)


  print("Tuning Job Starting")
  
  tuner = create_tuner(tf_estimator,  objective_metric_name, metric_definitions, hyperparameter_ranges, objective_type=objective_type)
  tuner.fit(estimator_inputs, job_name=tuning_job_name)
  #tuner.wait()

  tuner_metrics = sagemaker.HyperparameterTuningJobAnalytics(tuning_job_name)
  print("tuner_metrics", tuner_metrics)
  print("--------------")
  tuner_metrics.dataframe().sort_values(['FinalObjectiveValue'], ascending=True).head(5)
  print(tuner_metrics.dataframe().sort_values(['FinalObjectiveValue'], ascending=True).head(5))

  total_time = tuner_metrics.dataframe()['TrainingElapsedTimeSeconds'].sum() / 3600
  print("The total training time is {:.2f} hours".format(total_time))
  print (tuner_metrics.dataframe()['TrainingJobStatus'].value_counts())

  job_name = tuner.latest_tuning_job.name
  print("job_name",job_name)


############TEST FUNCTION###################

def main():

  sess, role, bucket_name = init_session()   
  headers = ['label','axis1','axis2','axis3'] #first column have to be label column
  label_list = ['Still','Lifting','Falling','Shaking']
  datapath = "../data/dataset.txt"

  data = Data(datapath, headers, label_list) # deafult settings --> test_rate=0.2, val_rate=0.2, seperator=','  

  layers_list = {
  "default_mode":
  [
  ("conv1d", {"filters":64, "kernel_size":2, "activation":"relu", "input_shape":(3,1)}), ("dropout",{"rate":0.5}),
  ("maxpooling1d",{"pool_size":1}), ("dense",{"units":64, "activation":"relu"}), 
  ("flatten",None), ("dense",{"units":100, "activation":"relu"}), ("dense",{"units":4, "activation":"softmax"}) 
  ]
  }

  if layers_list["default_mode"][0][0] == "conv1d" or layers_list["deafult_mode"] == "cnn":
    x_train, y_train, x_test, y_test, x_val, y_val = data.convertConv1D()
  else:
    x_train, y_train, x_test, y_test, x_val, y_val = data.x_train, data.y_train, data.x_test, data.y_test, data.x_val, data.y_val

  np.savez('../data/training', image=x_train, label=y_train)
  np.savez('../data/test', image=x_test, label=y_test)
  np.savez('../data/validation', image=x_val, label=y_val)



  training_input_path   = sess.upload_data('../data/training.npz', key_prefix='data/training')
  test_input_path = sess.upload_data('../data/test.npz', key_prefix='data/test')
  validation_input_path = sess.upload_data('../data/validation.npz', key_prefix='data/validation')

  hyperparameters = get_hyperparameters(bucket_name, layers_list)
  estimator_inputs =get_inputs(training_input_path, validation_input_path, test_input_path) 

  learning_range = (0.001, 0.2)
  epoch_range = (20, 100)
  batch_size_range = (64, 256)
  objective_metric_name, objective_type, hyperparameter_ranges, metric_definitions = get_tuning_params(learning_range, epoch_range, batch_size_range)

  base_job_name = "AI-PlatformNew"
  tuning_job_name = "tunerjoblibraryNew4" #"tuner-{}".format(strftime("%d-%H-%M-%S", gmtime()))

  run_instance(base_job_name, tuning_job_name, role, hyperparameters, bucket_name, estimator_inputs, objective_metric_name, \
                 metric_definitions, hyperparameter_ranges, objective_type)


if __name__ == "__main__":
  main()