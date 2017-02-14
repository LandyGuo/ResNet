#coding=utf-8
import tensorflow as tf
from tensorflow.python.training import moving_averages
import numpy as np
from cifar10 import load_cifar10
import os,logging
# from resnet import inference
from utils import get_origin_train_test_data,gen_batches_sequecial,preprocess_train_test_data,generate_random_augment_train_batch
import shutil


BN_DECAY= 0.9997
LOGDIR = "log/augment_test"
Batch_Size = 128
L2_Weight_Decay =0.0002
Initial_Learning_Rate = 0.1
Learning_Rate_Decat_Steps = 7000
Model_Save_Dir = "model"
Epochs = 3000
Reload = False
DataDir = "cifar10_data/cifar-10-batches-py"
epsilon = 1e-4
Learning_Steps_Decay = [40000,60000]#how many steps decay to 0.1


logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='augment_test.log',
                filemode='w+')



###################################utils#####################################
def bn(inputs, is_training=True, perfect=True):
	# 	#BN params
	params_shape = inputs.get_shape()[-1:]
	scale = tf.get_variable('scale',
		shape=params_shape,
		initializer=tf.constant_initializer(1),
		trainable=True)

	beta = tf.get_variable('offset',
		shape=params_shape,
		initializer=tf.constant_initializer(0),
		trainable=True)

	pop_mean = tf.get_variable(name="moving_mean",
								shape = params_shape,
								initializer = tf.constant_initializer(0),
								trainable = False)
	pop_var = tf.get_variable(name="moving_variance",
								shape = params_shape,
								initializer = tf.constant_initializer(1),
								trainable = False
								)
	if is_training:
		tf.histogram_summary(pop_mean.op.name,pop_mean)
		tf.histogram_summary(pop_var.op.name,pop_var)

	#perfect BN ,to comment
	axes = range(len(inputs.get_shape())-1)
	batch_mean, batch_var = tf.nn.moments(inputs,axes)

	if perfect:
		return tf.nn.batch_normalization(inputs,
			        batch_mean, batch_var, beta, scale, epsilon)

	if is_training:
		axes = range(len(inputs.get_shape())-1)
		batch_mean, batch_var = tf.nn.moments(inputs,axes)
		# print "BN:inputs shape:%s" % inputs.get_shape()
		# print "BN:batch_mean shape:%s" % batch_mean.get_shape()
		train_mean = tf.assign(pop_mean,
		                       pop_mean * BN_DECAY + batch_mean * (1 - BN_DECAY))
		train_var = tf.assign(pop_var,
		                      pop_var * BN_DECAY + batch_var * (1 - BN_DECAY))
		with tf.control_dependencies([train_mean, train_var]):
		    return tf.nn.batch_normalization(inputs,
		        batch_mean, batch_var, beta, scale, epsilon)
	else:
		return tf.nn.batch_normalization(inputs,
		    pop_mean, pop_var, beta, scale, epsilon)


def conv(input,out_channels,k_size=3,stride=1):
	w = tf.get_variable("weight",[k_size,k_size,input.get_shape().as_list()[-1],out_channels],dtype=tf.float32,
		initializer = tf.contrib.layers.xavier_initializer(),
		regularizer = tf.contrib.layers.l2_regularizer(L2_Weight_Decay))
	out = tf.nn.conv2d(input,w,strides=[1,stride,stride,1],padding="SAME")
	return out



def fc(input,out_nums):
	w = tf.get_variable("weight",shape=[input.get_shape().as_list()[-1],out_nums],dtype=tf.float32,
		initializer = tf.contrib.layers.xavier_initializer(),
		regularizer = tf.contrib.layers.l2_regularizer(L2_Weight_Decay))
	b = tf.get_variable("bias",shape=[out_nums],dtype=tf.float32,
		initializer = tf.constant_initializer(0.0))
	return tf.matmul(input,w)+b


################################ResNet Conv###################################
#BN,relu,conv
def block(input,out_channels,isTrain,first=False,channel_Increment=True,mode="padding"):
	"""basic block for ResNet
	
	Arguments:
		input {4D-tensor} -- [input 4D-tensor,should be format 'NHWC']
		out_channels {int} -- [channels for this block's ouput]
	
	Keyword Arguments:
		channel_Increment {bool} -- [whether this block increment channel ] (default: {True})
		mode {str} -- [shortcut mathching method,"padding" or "projection"] (default: {"padding"})
	
	Returns:
		[4D-tensor] -- [if channel_Increment=True, the shape of output tensor will be (None,w,h,out_channels)
		 			otherwise the shape will be (None,w/2,h/2,out_channels)
	"""
	stride = 2 if channel_Increment else 1
	#path1
	x = input
	with tf.variable_scope("conv1_in_block"):
		if not first:
			x = bn(x,isTrain)
			x = tf.nn.relu(x)
		x = conv(x,out_channels,stride=stride) 
	with tf.variable_scope("conv2_in_block"):
		x = bn(x,isTrain)
		x = tf.nn.relu(x)
		x = conv(x,out_channels)
	#path2 
	input_channels = input.get_shape().as_list()[-1]
	with tf.variable_scope("shortcut_in_block"):
		if channel_Increment:
			if mode=="projection":
				#use 1*1 convolution for dimension matching
				shortcut = conv(input,input_channels*2,k_size=1,stride=2)
			else:#padding
				#increase channel, reduce input dim
				s = tf.nn.avg_pool(input,[1,2,2,1],[1,2,2,1],padding="VALID")
				shortcut = tf.pad(s,[[0,0],[0,0],[0,0],[input_channels//2,input_channels//2]],mode="CONSTANT")
		else:
			shortcut = input
	logging.debug(tf.get_variable_scope().name)
	logging.debug("shortcut.get_shape:%s"%shortcut.get_shape())
	logging.debug("x shape:%s"%x.get_shape())
	assert shortcut.get_shape().as_list()[1:]==x.get_shape().as_list()[1:],"shape incorrect for shortcut"
	return tf.add(x,shortcut,name="block_out")




def top_k_error(logits,labels,k):

	return 1-tf.reduce_mean(tf.cast(tf.nn.in_top_k(tf.nn.softmax(logits),labels,k),tf.float32))



#########################placeholders######################################

def create_placeholders():
	image_placeholder = tf.placeholder(tf.float32,shape=[None,32,32,3],name="image_placeholder")
	label_placeholder = tf.placeholder(tf.int64,shape=[None],name="label_placeholder")
	return image_placeholder, label_placeholder


########################################################################
def inference(input, reuse=False,isTrain=True,resLayers=110):

	n = (resLayers-2) /6
	assert (6*n+2)==resLayers, "reslayers invalid"

	reslayers = []
	with tf.variable_scope("conv1",reuse=reuse):
		x = conv(input,16)
		x = bn(x,isTrain)
		x = tf.nn.relu(x)
		reslayers.append([tf.get_variable_scope().name,x])

	assert reslayers[-1][1].get_shape().as_list()[1:]==[32,32,16]
	#n*block
	for i in range(n):
		with tf.variable_scope("conv2_%s"%i,reuse=reuse):
			if i==0:
				x = block(reslayers[-1][1],16,isTrain,first=True,channel_Increment=False)
			else:
				x = block(reslayers[-1][1],16,isTrain,channel_Increment=False)
			reslayers.append([tf.get_variable_scope().name,x])

	assert reslayers[-1][1].get_shape().as_list()[1:]==[32,32,16]

	for i in range(n):
		with tf.variable_scope("conv3_%s"%i,reuse=reuse):
			if i==0:
				x = block(reslayers[-1][1],32,isTrain,channel_Increment=True)
			else:
				x = block(reslayers[-1][1],32,isTrain,channel_Increment=False)
			reslayers.append([tf.get_variable_scope().name,x])

	assert reslayers[-1][1].get_shape().as_list()[1:]==[16,16,32]

	for i in range(n):
		with tf.variable_scope("conv4_%s"%i,reuse=reuse):
			if i==0:
				x = block(reslayers[-1][1],64,isTrain,channel_Increment=True)
			else:
				x = block(reslayers[-1][1],64,isTrain,channel_Increment=False)
			reslayers.append([tf.get_variable_scope().name,x])

	assert reslayers[-1][1].get_shape().as_list()[1:]==[8,8,64]

	with tf.variable_scope("fc",reuse=reuse):
		in_layer = reslayers[-1][1]
		x = bn(in_layer,isTrain)
		x = tf.nn.relu(x)
		global_pool = tf.reduce_mean(x, [1, 2])

		assert global_pool.get_shape().as_list()[-1]==64
		logits = fc(global_pool,10)
		reslayers.append([tf.get_variable_scope().name,logits])

	return logits


def loss_function(logits, labels):
	"""Calculates the loss from the logits and the labels."""
	#Attention: logits must not be softmax regularized!!!!!!
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits, labels, name='xentropy')
	loss = tf.reduce_mean(cross_entropy, name='xentropy_sum')
	#l2 loss
	regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	total_loss = tf.add_n([loss]+regu_losses,name="total_loss")
	# Add a scalar summary for the snapshot loss.
	tf.scalar_summary('loss', loss)
	tf.scalar_summary('regularized_loss', tf.add_n(regu_losses))
	tf.scalar_summary('total_loss', total_loss)
	return total_loss

def training_function(loss):
	# Create a variable to track the global step.
	global_step = tf.Variable(0, name='global_step', trainable=False)
	# Create the gradient descent optimizer with the given learning rate.
	learning_rate = tf.Variable(Initial_Learning_Rate,dtype=tf.float32,name="learning_rate")
	# learning_rate = tf.train.exponential_decay(Initial_Learning_Rate,global_step,Learning_Rate_Decat_Steps,0.1,staircase=True)
	#summary for learning rate
	tf.scalar_summary('learning_rate', learning_rate)
	optimizer = tf.train.MomentumOptimizer(learning_rate,momentum=0.9)
	# Use the optimizer to apply the gradients that minimize the loss
	# (and also increment the global step counter) as a single training step.
	train_op = optimizer.minimize(loss, global_step=global_step)
	return train_op


def evaluate(sess, data,labels,images_placeholder,labels_placeholder,
	eval_correct_op ,batch_size=50):
	#pass whole data,every 100 images one time
	NUM_IMAGES = len(data)
	BATCH_NUM = NUM_IMAGES//batch_size
	correct_count = 0
	for i in range(BATCH_NUM):
		eval_data = data[i*batch_size:(i+1)*batch_size]
		eval_label = labels[i*batch_size:(i+1)*batch_size]
		correct_count+=sess.run(eval_correct_op,feed_dict={images_placeholder:eval_data,
			labels_placeholder:eval_label})
	logging.debug("wrong:%d / total:%d" % (BATCH_NUM*batch_size-correct_count,BATCH_NUM*batch_size))
	err_rate = 1.0-correct_count/(BATCH_NUM*batch_size)
	return err_rate



def run_training():
	g = tf.get_default_graph()
	with g.as_default():
		images_placeholder, label_placeholder = create_placeholders()

		#session config
		config = tf.ConfigProto()  
		config.gpu_options.allow_growth=True  
		sess = tf.Session(config=config)

		#output logits
		logits = inference(images_placeholder,False,True,resLayers=110)
		valid_logits = inference(images_placeholder,True,False, resLayers=110)
		# logits = inference(images_placeholder,18,False)
		# valid_logits = inference(images_placeholder,18,True)
		
		#get loss
		los = loss_function(logits, label_placeholder)

		#use loss value update model
		train_op = training_function(los)

		#calculate precision
		eval_correct_op_top1 = tf.reduce_sum(tf.cast(tf.nn.in_top_k(tf.nn.softmax(logits),label_placeholder,1),tf.float32))
		valid_eval_correct_op_top1 = tf.reduce_sum(tf.cast(tf.nn.in_top_k(tf.nn.softmax(valid_logits),label_placeholder,1),tf.float32))

		tf.scalar_summary("train_top1rate",100.0*(1.0-valid_eval_correct_op_top1/Batch_Size))

		test_top1_rate = tf.Variable(0.0,trainable=False)
		tf.scalar_summary("test_top1_rate",test_top1_rate)

		#collect summary
		summary = tf.merge_all_summaries()
		
		# Instantiate a SummaryWriter to output summaries and the Graph.
		summary_writer = tf.train.SummaryWriter(LOGDIR, sess.graph)
		
		# Create a saver for writing training checkpoints.
		saver = tf.train.Saver()
		
		#steps to continue
		step = 0
		
		sess.run(tf.initialize_all_variables())

		#reload model
		if Reload and tf.gfile.Exists(Model_Save_Dir):
			model_checkpoint_path = tf.train.latest_checkpoint(Model_Save_Dir)
			if model_checkpoint_path:
				saver.restore(sess,model_checkpoint_path)
				reader = tf.train.NewCheckpointReader(model_checkpoint_path)
				step =  reader.get_tensor("global_step")
				#get tensors' value in check point
				#print reader.get_tensor("conv1/bias")
				logging.debug("Model reload, Continue training...")
		else:
			tf.gfile.MakeDirs(Model_Save_Dir)

		#training data ready
		# dataX,dataY,testX,testY = get_origin_train_test_data()
		dataX,dataY,testX,testY= preprocess_train_test_data()
		logging.debug("step:%s"%step)

		epoch_contain_batches = len(dataX)//Batch_Size-1
		for epoch in range(Epochs):
			for i in range(epoch_contain_batches):
				imgs,ls = generate_random_augment_train_batch(dataX, dataY, batch_size=Batch_Size,padding_size=2)
			# for imgs,ls in gen_batches_sequecial(dataX,dataY,batch_size=Batch_Size,shuffle=True):
				feed_dict={images_placeholder:imgs,label_placeholder:ls}
				_, losVal,top1_right = sess.run([train_op,los,eval_correct_op_top1],feed_dict=feed_dict)
				logging.debug('Epoch:%s step:%s lossVal:%s top1_err:%3f%% ' % (epoch,step,losVal,100.0*(1.0-top1_right/Batch_Size)))
				
				#learning rate decay
				if step in Learning_Steps_Decay:
					learning_rate = tf.Graph.get_tensor_by_name("learning_rate")
					Initial_Learning_Rate *= 0.1
					sess.run(tf.assign(learning_rate,Initial_Learning_Rate))

				
				if (step + 1)% 300 == 0:
					#save model
					checkpoint_file = os.path.join(Model_Save_Dir, 'model.ckpt')
					saver.save(sess, checkpoint_file, global_step=step)
					logging.debug('Model Saved')
					logging.debug('Test on whole Test dataset')
					#evaluate model on whole training and testing data
					top1_err = evaluate(sess,testX,testY ,images_placeholder,
						label_placeholder,valid_eval_correct_op_top1,Batch_Size)
					# precison = sess.run(precision,feed_dict={labels:dataset.labels})
					sess.run(tf.assign(test_top1_rate,100*top1_err))
					logging.debug('Epoch:%s step:%s test top1_err:%3f%%' % (epoch,step,100.0*top1_err))
				
				#events update
				if (step+1)%300==0:
					# Update the events file.
					feed_dict={images_placeholder:imgs,label_placeholder:ls}
					summary_str = sess.run(summary, feed_dict=feed_dict)
					summary_writer.add_summary(summary_str, step)
					summary_writer.flush()
					logging.debug('Collect Summary')

				step+=1


def test():
	images_placeholder, label_placeholder = create_placeholders()
	sess = tf.Session()
	saver = tf.train.Saver()
	#output logits
	valid_logits = inference(images_placeholder,True,False, resLayers=110)







def main(_):
	#clear log_dir
	if tf.gfile.Exists(LOGDIR):
		shutil.rmtree(LOGDIR)
	tf.gfile.MakeDirs(LOGDIR)
	run_training()


if __name__=="__main__":
	tf.app.run(main=main)

