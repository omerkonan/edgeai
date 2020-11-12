from data import Data
from numpy import loadtxt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn import preprocessing



def main():

	print("--------------------------------------------------------------------------------")
	headers = ['label','axis1','axis2','axis3',] #first column have to be label column
	label_list = ['Still','Lifting','Falling','Shaking']
	seperator = ','
	datapath = "/home/konan/Desktop/edgeai/data/dataset.txt"
	test_rate = 0.2
	val_rate = 0.2
	data = Data(datapath, headers, seperator, label_list, test_rate, val_rate)
	print(data.x_train)
	model = keras.Sequential(([keras.layers.Dense(20,activation=tf.nn.relu, input_shape=(3,)),\
						keras.layers.Dense(100, activation=tf.nn.relu),\
						keras.layers.Dense(100, activation=tf.nn.relu),\
						keras.layers.Dense(50, activation=tf.nn.relu),\
						keras.layers.Dense(20, activation=tf.nn.relu),\
						keras.layers.Flatten(),
						keras.layers.Dense(len(label_list), activation=tf.nn.softmax)]))
	
	model.compile(optimizer ='sgd', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
	model.summary()
	print(data.y_val)
	print(data.x_val)

	model.fit(data.x_train,data.y_train,epochs=100)
	print("validation")
	score = model.evaluate(data.x_val, data.y_val)
	print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
	model.predict()
	






if __name__ =="__main__":
	main()