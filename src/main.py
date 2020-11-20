from data import Data
from numpy import loadtxt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix



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

def deepNNetwork(label_list):
	model = keras.Sequential(([keras.layers.Dense(20,activation=tf.nn.relu, input_shape=(3,)),\
						keras.layers.Dense(100, activation=tf.nn.relu),\
						keras.layers.Dense(100, activation=tf.nn.relu),\
						keras.layers.Dense(50, activation=tf.nn.relu),\
						keras.layers.Dense(20, activation=tf.nn.relu),\
						keras.layers.Flatten(),
						keras.layers.Dense(len(label_list), activation=tf.nn.softmax)]))
	return model

def main():

	print("--------------------------------------------------------------------------------")
	headers = ['label','axis1','axis2','axis3',] #first column have to be label column
	label_list = ['Still','Lifting','Falling','Shaking']
	seperator = ','
	datapath = "/home/konan/Desktop/edgeai/data/dataset.txt"
	test_rate = 0.2
	val_rate = 0.2
	data = Data(datapath, headers, seperator, label_list, test_rate, val_rate)
	
	cnn_x_train, cnn_y_train, cnn_x_test, cnn_y_test,\
		cnn_x_val, cnn_y_val, = data.convertConv1D()
	
	checkpoints = keras.callbacks.ModelCheckpoint("../models/weights.{epoch:02d}-{loss:.2f}.hdf5", \
													monitor='loss', save_best_only=False, period=10)

	#model = convNetwork()
	#model = deepNNetwork()
	"""
	model.summary()
	model.compile(optimizer ='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
	model.fit(cnn_x_train,cnn_y_train,epochs=100, callbacks=[checkpoints])
	print("validation")
	score = model.evaluate(cnn_x_val, cnn_y_val)
	print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
	model.save("../models./model.h5")
	"""
	model = tf.keras.models.load_model("../models/model.h5")
	print("Test solutions:")
	solutions = model.predict(cnn_x_test)
	print(solutions.argmax(axis=-1))
	print("cnn_y_test:")
	print(cnn_y_test)

	cm = confusion_matrix(cnn_y_test,solutions.argmax(axis=-1)) 
	print(cm)
	#model.save("../models/model")



if __name__ =="__main__":
	main()