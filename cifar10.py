#coding=utf-8
import glob
import numpy as np
from matplotlib import pylab as plt
import os
import cPickle



def unpickle(file):
	fo = open(file, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	return dict

def get_images_labels(file):
	content  = unpickle(file)
	data = content['data']
	# print content["labels"]
	channels = np.split(data,3,1)
	reshape_channels = [c.reshape([10000,32,32,1]) for c in channels]
	image_arr = np.concatenate(reshape_channels, axis=3)
	return image_arr,np.array(content['labels'])


def load_cifar10(cifar_dir):
	images,labels = [],[]
	for f in glob.glob(os.path.join(cifar_dir,"data_batch_[1-5]")):
		imgs,lbs = get_images_labels(f)
		images.append(imgs)
		labels.append(lbs)
	dataX =  np.concatenate(np.array(images),axis=0)
	dataY =  np.concatenate(np.array(labels),axis=0)
	# print "-----------"
	# print dataX.shape
	# print dataY.shape
	#test_data
	testX,testY = get_images_labels(os.path.join(cifar_dir,"test_batch"))
	# print "-----------"
	# print testX.shape
	# print testY.shape
	return dataX,dataY,testX,testY


# dataX,dataY,testX,testY = load_cifar10('./cifar10_data/cifar-10-batches-py')
# plt.imshow(testX[3])
# plt.show()