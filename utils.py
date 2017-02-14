#coding=utf-8
import numpy as np
from cifar10 import load_cifar10


IMG_HEIGHT =32
IMG_WIDTH =32 
IMG_DEPTH = 3



def horizontal_flip(image, axis):
    '''
    Flip an image at 50% possibility
    :param image: a 3 dimensional numpy array representing an image
    :param axis: 0 for vertical flip and 1 for horizontal flip
    :return: 3D image after flip
    '''
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        # image = cv2.flip(image, axis)
        # print "image horizontal flip!"
        # print "before"
        # misc.imshow(image)
        image = image[:,::-1,:]
        # print "after"
        # misc.imshow(image)

    return image


def whitening_image(image_np):
    '''
    Performs per_image_whitening
    :param image_np: a 4D numpy array representing a batch of images
    :return: the image numpy array after whitened
    '''
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        # Use adjusted standard deviation here, in case the std == 0.
        std = np.max(np.std(image_np[i, ...]), 1.0/np.sqrt(IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH))
        image_np[i,...] = (image_np[i, ...] - mean) / std
    return image_np


def random_crop_and_flip(batch_data, padding_size):
    '''
    Helper to random crop and random flip a batch of images
    :param padding_size: int. how many layers of 0 padding was added to each side
    :param batch_data: a 4D batch array
    :return: randomly cropped and flipped image
    '''
    cropped_batch = np.zeros(len(batch_data) * IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH).reshape(
        len(batch_data), IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+IMG_HEIGHT,
                      y_offset:y_offset+IMG_WIDTH, :]

        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)

    return cropped_batch


def padding_image_data(data, padding_size=2):
	"""padding images, usually used as preprocess for random crop
	Do this once for all training data for effiency
	
	Arguments:
		data 4D-array -- should be shape of (batch, height, width, channels)
		padding_size integer -- padding num along each axis 
	
	Returns:
		4D-array  -- same shape for input data
	"""
	pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
	data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)
	return data


def generate_random_augment_train_batch(all_train_data, all_train_labels, batch_size=64,padding_size=2):
    '''
    This function helps generate a batch of train data, and random crop, horizontally flip
    and whiten them at the same time
    :param train_data: 4D numpy array
    :param train_labels: 1D numpy array
    :param train_batch_size: int
    :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
    '''
    EPOCH_SIZE = len(all_train_data)
    offset = np.random.choice(EPOCH_SIZE - batch_size, 1)[0]
    batch_data = all_train_data[offset:offset+batch_size, ...]
    batch_data = random_crop_and_flip(batch_data, padding_size=padding_size)

    batch_data = whitening_image(batch_data)
    batch_label = all_train_labels[offset:offset+batch_size]

    return batch_data, batch_label




def gen_batches_sequecial(dataX,dataY,batch_size=50,shuffle=True):
	dataX,dataY = np.array(dataX),np.array(dataY)
	assert dataX.shape[0]==dataY.shape[0]
	data_size = dataX.shape[0]
	if shuffle:
		idx = np.arange(data_size)
		np.random.shuffle(idx) 
		dataX = dataX[idx]
		dataY = dataY[idx]
	batch_length = (data_size-1)//batch_size
	for i in range(batch_length):
		yield whitening_image(dataX[i*batch_size:i*batch_size+batch_size]),dataY[i*batch_size:i*batch_size+batch_size]



def generate_vali_batch(self, vali_data, vali_label, vali_batch_size):
    '''
    If you want to use a random batch of validation data to validate instead of using the
    whole validation data, this function helps you generate that batch
    :param vali_data: 4D numpy array
    :param vali_label: 1D numpy array
    :param vali_batch_size: int
    :return: 4D numpy array and 1D numpy array
    '''
    offset = np.random.choice(10000 - vali_batch_size, 1)[0]
    vali_data_batch = vali_data[offset:offset+vali_batch_size, ...]
    vali_label_batch = vali_label[offset:offset+vali_batch_size]
    return vali_data_batch, vali_label_batch



def preprocess_train_test_data():
	trainX, trainY, testX, testY = get_origin_train_test_data()
	#padding all train data
	padded_trainX = padding_image_data(trainX,padding_size=2)
	return padded_trainX, trainY, testX, testY


def get_origin_train_test_data():
    trainX, trainY, testX, testY= load_cifar10('./cifar10_data/cifar-10-batches-py')
    return trainX, trainY, testX, testY
