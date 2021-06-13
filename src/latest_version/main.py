import os
from typing import List, type_check_only
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import boto3
import pandas as pd
import json

from tensorflow.python.ops.gen_math_ops import LinSpace





class Training:
    def __init__(self):
        self.log_table = []
        self.s3 = boto3.client("s3", region_name='eu-west-1')
        self.process_id = os.getenv("SM_HP_EMC_PROCESS_ID")
        self.environment = os.getenv("SM_HP_EMC_ENVIRONMENT")
        self.job_id = os.getenv("SM_HP_EMC_JOB_ID")
        self.bucket = os.getenv("SM_HP_EMC_AI_BUCKET")
        self.custom_layer = os.getenv("SM_HP_CUSTOM_LAYER")

        if self.custom_layer:
            self.custom_layer = json.loads(self.custom_layer)

        self.input_path = os.getenv("SM_CHANNEL_INPUT_PATH")
        self.output_data_dir = os.getenv("SM_OUTPUT_DATA_DIR")
        self.model_dir = os.getenv("SM_MODEL_DIR")
        self.network_type = os.getenv("SM_HP_NETWORK_TYPE")
        self.epochs = int(os.getenv("SM_HP_EPOCHS"))
        self.batch_size = int(os.getenv("SM_HP_BATCH_SIZE"))
        self.lr = float(os.getenv("SM_HP_LEARNING_RATE"))
        self.test_rate = float(os.getenv("SM_HP_TEST_RATE"))
        self.val_rate = float(os.getenv("SM_HP_VAL_RATE"))
        
        self.label_list = None
        self.df_label, self.df_feature = self._get_processed_data()
        self.x_train, self.y_train, self.x_test, self.y_test, self.x_val, self.y_val = self._adjust_dataset()
        self.model = self._create_keras_model()

        
        self.accuracy = None
        self.solutions = None
        self.c_matrix = None
        self.loss = None

    def _get_processed_data(self):
        df = pd.read_csv(os.path.join(self.input_path, "processed.csv"), engine="python")

        for column in df.columns:
            if str(column).lower() == 'label':
                label_header = column
                break

        self.label_list = pd.unique(df[label_header])


        for i in range(len(self.label_list)):
            df = df.replace({label_header: self.label_list[i]}, i)


        df_label = df[self.label_header]
        df_feature = df.drop(self.label_header, axis=1)
        return df_label, df_feature



    def _adjust_dataset(self):
        if self.network_type == "CNN" or (self.custom_layer and self.custom_layer[0][0] == "conv1d"):

            x_train, y_train, x_test, y_test, x_val, y_val = self._split_data()
            x_train, y_train, x_test, y_test, x_val, y_val = self._convert_conv1d(x_train, y_train, x_test, y_test,
                                                                                  x_val, y_val)
        else:
            x_train, y_train, x_test, y_test, x_val, y_val = self._split_data()

        return x_train, y_train, x_test, y_test, x_val, y_val

    def get_modal_summary(self):
        return self.model.summary()

    def start_training(self):
        self.model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train, epochs=self.epochs)

    def _create_keras_model(self):
        model_sequential = []

        if self.network_type == "CNN":
            return self._create_cnn_model()
        elif self.network_type == "ANN":
            return self._create_ann_model()
        else:
            for layer in self.custom_layer:
                if layer[0] == "conv1d":
                    model_sequential.append(
                        keras.layers.Conv1D(
                            filters=layer[1]["filters"],
                            kernel_size=layer[1]["kernel_size"],
                            activation=layer[1]["activation"],
                            input_shape=layer[1]["input_shape"]
                        )
                    )
                if layer[0] == "dropout":
                    model_sequential.append(keras.layers.Dropout(layer[1]["rate"]))
                if layer[0] == "dense":
                    model_sequential.append(keras.layers.Dense(layer[1]["units"], activation=layer[1]["activation"]))
                if layer[0] == "maxpooling1d":
                    model_sequential.append(keras.layers.MaxPooling1D(pool_size=layer[1]["pool_size"]))
                if layer[0] == "flatten":
                    model_sequential.append(keras.layers.Flatten())

            return keras.Sequential(model_sequential)






    def _create_cnn_model(self):
        return keras.Sequential([
            keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1)),
            keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.MaxPooling1D(pool_size=1),
            keras.layers.Flatten(),
            keras.layers.Dense(100, activation='relu'),
            keras.layers.Dense(4, activation='softmax')]
        )

    def _create_ann_model(self):
        return keras.Sequential((
            [keras.layers.Dense(20, activation=tf.nn.relu, input_shape=(3,)),
             keras.layers.Dense(100, activation=tf.nn.relu),
             keras.layers.Dense(100, activation=tf.nn.relu),
             keras.layers.Dense(50, activation=tf.nn.relu),
             keras.layers.Dense(20, activation=tf.nn.relu),
             keras.layers.Flatten(),
             keras.layers.Dense(4, activation=tf.nn.softmax)
             ]
        ))

    def _split_data(self):
        x, x_test, y, y_test = train_test_split(self.df_feature, self.df_label, test_size=self.test_rate,
                                                train_size=(1 - self.test_rate))
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=self.val_rate, train_size=(1 - self.val_rate))

        return x_train, y_train, x_test, y_test, x_val, y_val

    def _convert_conv1d(self, x_train, y_train, x_test, y_test, x_val, y_val):
        x_train = np.expand_dims(x_train, axis=2)
        y_train = np.expand_dims(y_train, axis=1)
        x_test = np.expand_dims(x_test, axis=2)
        y_test = np.expand_dims(y_test, axis=1)
        x_val = np.expand_dims(x_val, axis=2)
        y_val = np.expand_dims(y_val, axis=1)
        return x_train, y_train, x_test, y_test, x_val, y_val

    def generate_model_output(self):
        score = self.model.evaluate(self.x_val, self.y_val)
        self.accuracy, self.loss = score[1], score[0]
        self.solutions = self.model.predict(self.x_test)  # Label List
        self.solutions = self.solutions.argmax(axis=-1)  # y_test ile karsılaştır
        self.c_matrix = confusion_matrix(self.y_test, self.solutions)

    def _send_attached_logs(self):
        client = boto3.client('logs')

        log_args = dict(
            logGroupName='Empacloud-AI-Process-Logs',
            logStreamName=f"{self.process_id}",
            logEvents=self.log_table
        )

        if self.log_sequence_token:
            log_args['sequenceToken'] = self.log_sequence_token

        response = client.put_log_events(
            **log_args
        )

        if response['sequenceToken']:
            self.log_sequence_token = response['sequenceToken']

    def _clear_log_table(self):
        self.log_table.clear()

    def _log(self, log):
        self.log_table.append(log)

    def save_model(self):
        file_path = self.model_dir + "/model.h5"
        self.model.save(file_path)
        self.s3.upload_file(Filename=file_path, Bucket=self.bucket,
                            Key=f'{self.environment}/{self.job_id}/MODELS/{self.process_id}/model.h5')
        # TODO: Set Training Process To DONE
        # TODO: Hyperparameter Denerken üst üste kaydetmeyi kontrol et


    def decode_with_label(self, list):

        for index,item in enumerate(list):
            if type(item) == 'List':
                list[index] = self.label_list[item[0]]
            else:
                list[index] = self.label_list[item]

if __name__ == "__main__":
    training = Training()
    training.get_modal_summary()
    training.start_training()
    training.save_model()
    training.generate_model_output()



    print(f"Accuracy : {training.accuracy}")
    print(f"LOSS : {training.loss}")
    print(f"matrix : {training.c_matrix}")
    print(f"Solutions : {training.solutions}")
    print(f"Labels : {training.y_test}")
