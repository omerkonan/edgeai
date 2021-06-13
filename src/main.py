import os
import argparse
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import boto3
import json

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
        "default_mode":"cnn"
    }
    model_sequential= []
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
                                            activation=item["activation"], input_shape=item["input_shape"]))
            
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
                        keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3,1)),\
                        keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'),\
                        keras.layers.Dropout(0.5),\
                        keras.layers.MaxPooling1D(pool_size=1),\
                        keras.layers.Flatten(),\
                        keras.layers.Dense(100, activation='relu'),\
                        keras.layers.Dense(4, activation='softmax')])
    return model

def deepNNetwork():
    model = keras.Sequential(([keras.layers.Dense(20,activation=tf.nn.relu, input_shape=(3,)),\
                        keras.layers.Dense(100, activation=tf.nn.relu),\
                        keras.layers.Dense(100, activation=tf.nn.relu),\
                        keras.layers.Dense(50, activation=tf.nn.relu),\
                        keras.layers.Dense(20, activation=tf.nn.relu),\
                        keras.layers.Flatten(),
                        keras.layers.Dense(4, activation=tf.nn.softmax)]))
    return model

def getparams():
    json_path = os.getcwd() + "/parameters.json"
    with open(json_path) as f:
            json_parameters = json.loads(f.read())

    print("json_path", json_parameters)
    for key, param in json_parameters.items():
        if key == "epochs":
            epochs = param
        if key == "batch_size":
            batch_size = param
        if  key == "learning_rate":
            lr = param
        if key == "bucket_name":
            bucket_name = param
    
    return epochs, batch_size, lr, bucket_name
        
def main(args):
    
    training_dir   = args.training
    validation_dir = args.validation
    test_dir = args.test
    output_data_dir = args.output_data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    bucket_name = args.bucket_name
    layers_list = args.layers_list


    
    print("training", training_dir)
    print("test", test_dir)
    print("validation", validation_dir)
    print("epochs:", epochs)
    print("lr:", lr)
    print("batch_size:", batch_size)
    print("model-dir:",model_dir)
    print("bucket_name:", bucket_name)
    print("output_data_dir", output_data_dir)
    print("output_dir", output_dir)
    
    print("--------------------------------------------------------------------------------")
 
    checkpoints = keras.callbacks.ModelCheckpoint((output_data_dir + '/checkpoint-{epoch}.h5'),
                                                    monitor='loss', save_best_only=False, period=1)

    cnn_x_train = np.load(os.path.join(training_dir, 'training.npz'))['image']
    cnn_y_train = np.load(os.path.join(training_dir, 'training.npz'))['label']
    cnn_x_val   = np.load(os.path.join(validation_dir, 'validation.npz'))['image']
    cnn_y_val   = np.load(os.path.join(validation_dir, 'validation.npz'))['label']
    cnn_x_test  = np.load(os.path.join(test_dir, 'test.npz'))['image']
    cnn_y_test  = np.load(os.path.join(test_dir, 'test.npz'))['label']
   

    #model = convNetwork()
    #model = deepNNetwork(label_list)

    model = createModel(layers_list)


    
    print("---------------------TRAINING-START------------------------")
    model.summary()
    model.compile(optimizer ='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.fit(cnn_x_train, cnn_y_train, epochs=epochs, callbacks=[checkpoints])
    print("validation")
    score = model.evaluate(cnn_x_val, cnn_y_val)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    model_saved_path = model_dir + "/model_test.h5"
    model.save(model_saved_path)
    tf.contrib.saved_model.save_keras_model(model, model_dir)
    
    
    print("saved")
    
    #model = tf.keras.models.load_model(model_saved_path)
    print("model loaded")
    print("Test solutions:")
    solutions = model.predict(cnn_x_test)
    print(solutions.argmax(axis=-1))
    print("cnn_y_test:")
    print(cnn_y_test)
    
    cm = confusion_matrix(cnn_y_test,solutions.argmax(axis=-1)) 
    print(cm)
    
    
    s3_client = boto3.client('s3')
    s3_client.upload_file(Filename=model_saved_path, Bucket=bucket_name, Key='model_Test.h5')



if __name__ =="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--bucket-name', type=str, default=1)
    parser.add_argument('--layers-list', type=dict)
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_data_dir',type=str,default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--output-dir',type=str,default=os.environ.get('SM_OUTPUT_DIR'))

    args, _ = parser.parse_known_args()


    main(args)