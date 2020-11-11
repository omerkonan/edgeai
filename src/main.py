from data import Data

def main():

	print("--------------------------------------------------------------------------------")
	headers = ['label','axis1','axis2','axis3',]
	seperator = ','
	datapath = "/home/konan/Desktop/edgeai/data/dataset.txt"
	test_rate = 0.2
	val_rate = 0.2
	data = Data(datapath, headers, seperator,test_rate, val_rate)
	







if __name__ =="__main__":
	main()