import os
import argparse
from re import split
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import boto3
import pandas as pd

def createModel(layers_list):
    """
    Example of input data:
        layers_list = {
        "default_mode":
        [
        ("conv1d", {"filters":64, "kernel_size":2, "activation":"relu", "input_shape":(3,1)}), ("dropout",{"rate":0.5}),
        ("maxPooling1d",{"pool_size":1}), ("dense",{"units":64, "activation":"relu"}),
        ("flatten",None), ("dense",{"units":4, "activation":"softmax"}), ("dense",{"units":4, "activation":"softmax"})
        ]
        }

    """

    layers_list = {
        "default_mode": "cnn"
    }
    model_sequential = []
    if layers_list["default_mode"] == "cnn":
        model = convNetwork()
        print("Default mode: cnn")
        return model

    elif layers_list["default_mode"] == "dnn":
        model = deepNNetwork()
        print("Deafult mode: dnn")
        return model

    else:
        print("Custom Mode")
        for key in layers_list["default_mode"]:
            if key[0] == "conv1d":
                item = key[1]
                model_sequential.append(keras.layers.Conv1D(filters=item["filters"], kernel_size=item["kernel_size"], \
                                                            activation=item["activation"],
                                                            input_shape=item["input_shape"]))

            if key[0] == "dropout":
                item = key[1]
                model_sequential.append(keras.layers.Dropout(item["rate"]))

            if key[0] == "dense":
                item = key[1]
                model_sequential.append(keras.layers.Dense(item["units"], activation=item["activation"]))
            if key[0] == "maxpooling1d":
                item = key[1]
                model_sequential.append(keras.layers.MaxPooling1D(pool_size=item["pool_size"]))
            if key[0] == "flatten":
                model_sequential.append(keras.layers.Flatten())

        print("model_seq:", model_sequential)
        print("----------------------------------------------------------------")
        print("layer_list:", layers_list)

        model = keras.Sequential(model_sequential)
        return model


def convNetwork():
    model = keras.Sequential([
        keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1)), \
        keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'), \
        keras.layers.Dropout(0.5), \
        keras.layers.MaxPooling1D(pool_size=1), \
        keras.layers.Flatten(), \
        keras.layers.Dense(100, activation='relu'), \
        keras.layers.Dense(4, activation='softmax')])
    return model


def deepNNetwork():
    model = keras.Sequential(([keras.layers.Dense(20, activation=tf.nn.relu, input_shape=(3,)), \
                               keras.layers.Dense(100, activation=tf.nn.relu), \
                               keras.layers.Dense(100, activation=tf.nn.relu), \
                               keras.layers.Dense(50, activation=tf.nn.relu), \
                               keras.layers.Dense(20, activation=tf.nn.relu), \
                               keras.layers.Flatten(),
                               keras.layers.Dense(4, activation=tf.nn.softmax)]))
    return model

def get_label_header(df):
    for column in df.columns:
            if str(column).lower() == 'label':
                return column
def getData(input_path):

    df = pd.read_csv(input_path, engine='c', error_bad_lines=False, warn_bad_lines=False)
    label_header = get_label_header(df)
    df_label = df[label_header]
    df_feature = df.drop(label_header, axis=1)
    #download from s3
    #return self.df_feature, self.df_label
    return df_label,df_feature




def split_data(df_label, df_feature):
    """
    TODO: test_rate ve val_rate
    """
    x, x_test, y, y_test = train_test_split(df_feature, df_label, test_size=self.test_rate,
                                            train_size=(1 - self.test_rate))
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=self.val_rate, train_size=(1 - self.val_rate))
    
    return x_train, y_train, x_test, y_test, x_val, y_val

def convert_conv1d():
    """
    Keras conv1d layer input shape should be 3d. This function adjust data dimension for con1d
    """
    x_train = np.expand_dims(self.x_train, axis=2)
    y_train = np.expand_dims(self.y_train, axis=1)
    x_test = np.expand_dims(self.x_test, axis=2)
    y_test = np.expand_dims(self.y_test, axis=1)
    x_val = np.expand_dims(self.x_val, axis=2)
    y_val = np.expand_dims(self.y_val, axis=1)
    return x_train, y_train, x_test, y_test, x_val, y_val


def main(args):
    input_path = args.input_path
    output_data_dir = args.output_data_dir
    model_dir = args.model_dir
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    network_type = args.network_type
    custom_layer = args.custom_layer



    print("epochs:", epochs)
    print("lr:", lr)
    print("batch_size:", batch_size)
    print("network_type:", network_type)
    print("custom_layer:", custom_layer)
    print("model-dir:", model_dir)
    print("output_data_dir", output_data_dir)

    print("--------------------------------------------------------------------------------")

    # TODO adjust saving path, check on s3
    checkpoints = keras.callbacks.ModelCheckpoint((output_data_dir + '/checkpoint-{epoch}.h5'),
                                                  monitor='loss', save_best_only=False, period=5)

    
    df_label, df_feature = getData(input_path)
        

    if custom_layer["default_mode"][0][0] == "conv1d" or network_type == "cnn":
        x_train, y_train, x_test, y_test, x_val, y_val = convert_conv1d(split_data(df_label, df_feature))
    else:
        x_train, y_train, x_test, y_test, x_val, y_val = split(df_label, df_feature)



    model = createModel(network_type, custom_layer)

    print("---------------------TRAINING-START------------------------")
    model.summary()
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, callbacks=[checkpoints])
    
    print("Validation starting")
    score = model.evaluate(x_val, y_val)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    solutions = model.predict(x_test)
    print("Test Labels:")
    print(y_test)
    print("Model Solutions:")
    print(solutions.argmax(axis=-1))
    
    cm = confusion_matrix(y_test, solutions.argmax(axis=-1))
    print("Confusion Matrix:")
    print(cm)

    model_saved_path = model_dir + "/model_test.h5"
    model.save(model_saved_path)

    s3_client = boto3.client('s3')
    s3_client.upload_file(Filename=model_saved_path, Bucket=bucket_name, Key='model_Test.h5')

    
    
    



    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--network-type', type=str, default=1)
    parser.add_argument('--custom-layer', type=list)
    parser.add_argument('--input-path', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    args, _ = parser.parse_known_args()

    main(args)