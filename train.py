'''
	Use early stopping with this condition , if the testset accuracy doesnt increase for a certain number of epochs = stop(store max accuracy and compare)

	References Used : 
		https://www.coursera.org/learn/machine-learning/supplement/pjdBA/backpropagation-algorithm
		http://neuralnetworksanddeeplearning.com/chap1.html

	IMPORTANT : for speedup its necessary that  the input data is in the required format, 1 column vector otherwise it has to be converted into that form again and again in each epoch which is a redundant operation and takes a lot of time
				(this was a huge speedup from 250s to 50 s for each epoch)
				converted formats for input of train_data,test_data and valid_data and output of train_data

				Second speed up(in setup part) : use np.reshape(x,(784,1)) when you know the size of x instead of np.vstack, reshape is lot faster: np.vstack has a completely different purpose, it is used to add a row to an array

	The only difference in cross entropy and mean square error is that in mse in (INITIALLY for last layer weights)delta = np.subtract( self.activations[-1],output)*der_act(here derivative is multiplied so learning(i.e weight update) slows down as derivative becomes 0) whereas in ce #delta = np.subtract( self.activations[-1],output) --- > refer : http://neuralnetworksanddeeplearning.com/chap3.html#the_cross-entropy_cost_function
	Ref : https://github.com/cazala/synaptic/issues/33


	Caution : At some points ,code gives :  RuntimeWarning: overflow encountered in exp
  return 1.0 / (1.0 + np.exp(-z) ) ---> This is probably because the value of -z becomes too negative and hence it creates problem for exp function, try to scale down hyperparameters to avoid this
	MORE INSIGHT ON THIS ERROR : https://stats.stackexchange.com/questions/210726/neural-network-can-only-follow-increasing-function

	Step Size = Learning Rate , EXTRA Addition : L2 Regularisation(Proabably it helps in correcting the Runtime error due to over shooting of weights)(BUT RESULTS WERE BETTER WITHOUT ANY REGULARISATION)
	DEFN : An epoch means one iteration over all of the training data. For instance if you have 20,000 images and a batch size of 100 then the epoch should contain 20,000 / 100 = 200 steps. 

'''

import numpy as np
#Python 3 , in Python 2 the library is named cPickle
import pickle 
import gzip
import time
import math
import os

import operator
import random
random.seed(1234)
np.random.seed(4567)

###################################	Code For Argument Parse	#######################################################

import argparse

loss_function = {'ce': 0, 'sq': 1}
optimization_algorithm = {'gd': 0,'momentum': 1,'nag': 2, 'adam': 3}
activation_function  = {'sigmoid': 0, 'tanh': 1}

parser = argparse.ArgumentParser()
parser.add_argument('--lr', help='initial learning rate for gradient descent algorithms',default = 0.25, type = float)
parser.add_argument("--momentum", help="momentum to be used by momentum based algorithms",default = 0, type = float)
parser.add_argument('--num_hidden', help='number of hidden layers',default = 1, type = int)
parser.add_argument('--sizes', help='list of neurons in each hidden layer',default = [30], type = int, nargs = '*' )
parser.add_argument('--activation', help='choice of activation function',default = 'sigmoid', choices = activation_function)
parser.add_argument('--loss', help=' loss function to be used by optimization algorithm ',default = 'ce',choices = loss_function)
parser.add_argument('--opt', help='optimization algorithm',default = 'gd', choices=['gd','momentum','nag', 'adam'])
parser.add_argument('--batch_size', help='batch size to be used',default = 100, choices = [1] + [i for i in range(5,50005,5)], type = int)
parser.add_argument('--anneal', help='to half the learning rate when validation loss decreases in an epoch',default = False , type = bool)
parser.add_argument('--save_dir', help='dir in which pickled model(weights and biases of network) will be saved',default = os.path.abspath(os.curdir))
parser.add_argument('--expt_dir', help='dir in which log files will be stores',default = os.path.abspath(os.curdir))
parser.add_argument('--mnist', help='path to mnist data in pcikled format',default = os.path.abspath(os.curdir))
parser.add_argument('--epoch', help='maximum number of epochs',default = 5 ,type = int)


args = parser.parse_args()

args.loss = loss_function[args.loss]
args.opt = optimization_algorithm[args.opt]
args.activation = activation_function[args.activation]


###################################################################################################################
file_names = ['/log_loss_train.txt','/log_err_train.txt' , '/log_loss_valid.txt', '/log_err_valid.txt', '/log_loss_test.txt' , '/log_err_test.txt']

openfiles = []
for i in range(6):
	file_path = args.save_dir + file_names[i]
	file = open(file_path, 'a')
	openfiles.append(file)



############################################## Class ##################################################################33

class NN(object):

	'''
		INIT fuction, nodes in layers is an aray containing the number of neurons in each layer 
		nodesInLayers = [784,15,10]
	'''
	def __init__(self, nodesInLayers, momentum, activation_function_value, loss_function_value ):  # WHY USING MOMENTUM OF VALUE 1 GAVE SATURATION AROUND 900 correct detections
			# the value of momentum makes things a bit uncertain, keep it below 0.9
		
		self.layers = len(nodesInLayers)
		self.sizes = nodesInLayers 		#total number of units(neurons) in each layer, excluding the biases
		self.bias = [np.random.randn(y,1) for y in nodesInLayers[1:] ]	 # bias node of current level will have weights for each neuron(node/unit) of the next layer
		self.weights = [np.random.randn(y,x) for x,y in zip(nodesInLayers,nodesInLayers[1:])] # weights matrix for all nodes between successive layers(except bias weights)
		self.activations = []
		self.activation_function_value = activation_function_value #( 0 mean sigmoid, 1 mean tanh)
		self.loss_function_value = loss_function_value #( 0 means cross entropy , 1 means mean square error)
		self.momentum_coefficient = momentum

		#Load Data
		f = gzip.open(args.mnist + '/mnist.pkl.gz', 'rb')
		self.train_set, self.valid_set, self.test_set = pickle.load(f, encoding='latin1')
		# self.train_set = self.valid_set
		f.close()

		print('Data Values for arguments')
		print('nodesInLayers = ',nodesInLayers)
		print('mometum = ',momentum)
		print('activation function = ',activation_function_value)
		print('loss function = ',loss_function_value)
		print('Anneal = ',args.anneal)

		

	def setup(self):
		self.test_set = self.format_data(self.test_set)
		self.valid_set = self.format_data(self.valid_set)
		self.train_set = self.format_data_train(self.train_set)
		# self.train_set = self.train_set[:10000]


	def format_data(self,data_set):		
		n = len(data_set[0])
		x,y = data_set			#data_set is a tuple, x contains 2d image pixel representaion , y contains integer for that representation
		data_set = [(np.reshape(a,(784,1)),b) for a,b in zip(x,y) ]
		return data_set

	def format_data_train(self,data_set):
		n = len(data_set[0])
		x,y = data_set			#data_set is a tuple, x contains 2d image pixel representaion , y contains integer for that representation
		data_set = [(np.reshape(a,(784,1)), np.reshape(self.vectorized_result(b) , (10,1) ) ) for a,b in zip(x,y) ]
		return data_set

	def vectorized_result(self,j):
		"""Return a 10-dimensional unit vector with a 1.0 in the jth
		position and zeroes elsewhere.  This is used to convert a digit
		(0...9) into a corresponding desired output from the neural
		network(one hot encoding)."""
		e = np.zeros((10, 1))
		e[j] = 1.0
		return e

	def activation_function(self,z, der = False):
		if der == True :
			if self.activation_function_value == 0 :
				return self.sigmoid(z,True)
			else: 
				return self.tanh(z,True)

		elif self.activation_function_value == 0 :
			return self.sigmoid(z)
		else:
			return self.tanh(z)
		

	def sigmoid(self,z,der = False):
		if(der == True) :
			return z * (1 - z)
		else :
			return 1.0 / (1.0 + np.exp(-z) )

	def tanh(self,z,der = False):
		if(der == True) :
			return (1 - (z*z) )
		else :
			pos_z = np.exp(z)
			neg_z = np.exp(-z)
			return (pos_z - neg_z)/ ( pos_z + neg_z)


	''' Gradient Descent
		It's actually Momentum, but converts to SGD when momentum coefficient = 0
	'''
	def SGD(self,epoch,eta,mini_batch_size,lmbda = 0):		
		# eta is learning rate

		#print("bias\n",self.bias[-1])
		self.learning_rate = eta
			
		t = 0
		prev_min = 1000000000 			  # infinite value, should use infinity designed inside python, this is just a workaround


		while t < epoch:
			trainining_loss = 0
			# number of examples in training data
			n = len(self.train_set)
			velocity_weight = [np.zeros(w.shape) for w in self.weights] 
			velocity_bias = [np.zeros(b.shape) for b in self.bias]
			
			start_time = time.time()
			
			# random.shuffle(self.train_set)

			mini_batches = [self.train_set[k:k+mini_batch_size]
				for k in range(0, n, mini_batch_size)]

			min_validation_loss_in_epoch = 1000000000 

			for step,mini_batch in enumerate(mini_batches):
				delta_cost = [np.zeros(w.shape) for w in self.weights]  # Matrix to store Patrial Derivatices of cost function w.r.t weights
				delta_bias_cost = [np.zeros(b.shape) for b in self.bias]
			
				for i in range( len(mini_batch) ):

					x,y = mini_batch[i][0], mini_batch[i][1]

					#Forward Propagation
					self.forwardprop(x)

					#Back Propagation
					delta_cost_temp,delta_bias_cost_temp = self.backprop(x,y)
					delta_cost = [dc + dct for dc,dct in zip(delta_cost,delta_cost_temp)]
					delta_bias_cost = [dbc + dbct for dbc,dbct in zip(delta_bias_cost,delta_bias_cost_temp)]
					
					if self.activation_function_value == 0 : 
						trainining_loss = trainining_loss - np.argmax(y)*math.log(self.activations[-1][np.argmax(y)])
					else:
						trainining_loss += 0.5*np.sum(np.square(y - self.activations[-1]))	#mean square error loss


				
				velocity_weight = np.multiply(self.momentum_coefficient , velocity_weight )  # here old value of velocity is used
				velocity_bias = np.multiply(self.momentum_coefficient , velocity_bias ) 
				
				velocity_weight = velocity_weight  +  (eta/mini_batch_size) * np.array(delta_cost)		# here old value is used to calculate new value
				velocity_bias = velocity_bias +  (eta/(mini_batch_size)) * np.array( delta_bias_cost)

				# ADDING L2 REGULARISATION
				self.weights = np.subtract(np.multiply( 1 - (eta * lmbda/n), self.weights), velocity_weight)

				self.bias = np.subtract(self.bias,  velocity_bias)

				# TEST OF ANNEALING



				#STORING VALUES
				if(step + 1) % 100 == 0:
					
					#Store error and loss values for each data set
					self.print_log(t,step+1)
						
					# # Store Weights and biases	
					self.store_weights()
			

			if(args.anneal):
				_,loss = self.evaluate(self.valid_set)
				if min_validation_loss_in_epoch > loss :
					min_validation_loss_in_epoch = loss
					
			if min_validation_loss_in_epoch - prev_min > 1:
				print('Minimum validation loss in last epoch is greater than minimum validation loss in this epoch --- Annealing Learning Rate')
				self.learning_rate = self.learning_rate/2
				print('New LR = ',self.learning_rate)
				continue


			print("On Test Set : accuracy = %d/10000 & loss = %f" %(self.evaluate() ))
			
			# USE THIS TO FIND OUT ETA(LEARNING RATE / STEP SIZE)
			print("Training Cost/Loss = ",(1/n)*trainining_loss + (lmbda/(2*n)) * (sum(np.sum(np.multiply(self.weights[i],self.weights[i]) ) for i in range(self.layers-1)  ) ) )


			# print("On Train Set : accuracy = %d/10000 & loss = %f" %(self.evaluate(self.train_set) ))
			print("time for epoch no.",(t+1),"=",time.time()-start_time )

			prev_min = min_validation_loss_in_epoch
			t = t + 1



	def NAG(self,epoch,eta,mini_batch_size,lmbda=0):	#Nesterov Accelerated Gradient	
		# eta is learning rate

		#print("bias\n",self.bias[-1])
		
		self.learning_rate = eta
		t=0
		prev_min = 1000000000
		while t < epoch:
			trainining_loss = 0
			n = len(self.train_set)
			start_time = time.time()
			
			random.shuffle(self.train_set)

			mini_batches = [self.train_set[k:k+mini_batch_size]
				for k in range(0, n, mini_batch_size)]

			velocity_weight = [np.zeros(w.shape) for w in self.weights] 
			velocity_bias = [np.zeros(b.shape) for b in self.bias]

			trainining_loss = 0
			min_validation_loss_in_epoch = 1000000000 

			for step,mini_batch in enumerate(mini_batches):
				delta_cost = [np.zeros(w.shape) for w in self.weights]  # Matrix to store Patrial Derivatices of cost function w.r.t weights
				delta_bias_cost = [np.zeros(b.shape) for b in self.bias]

				velocity_weight = np.multiply(self.momentum_coefficient , velocity_weight )  # here old value of velocity is used
				velocity_bias = np.multiply(self.momentum_coefficient , velocity_bias ) 
				
				#Update weights before finding gradient(Weight_Look_ahead)
				self.weights = np.subtract(self.weights, velocity_weight)
				self.bias = np.subtract(self.bias,  velocity_bias)

			
				for i in range( len(mini_batch) ):
					
				#	print(i)
					x,y = mini_batch[i][0], mini_batch[i][1]

					#Forward Propagation
					self.forwardprop(x)
				#	print("after forward prop")

					#Back Propagation
					delta_cost_temp,delta_bias_cost_temp = self.backprop(x,y)
					delta_cost = [dc + dct for dc,dct in zip(delta_cost,delta_cost_temp)]
					delta_bias_cost = [dbc + dbct for dbc,dbct in zip(delta_bias_cost,delta_bias_cost_temp)]
					if self.activation_function_value == 0 : 
						trainining_loss = trainining_loss - np.argmax(y)*math.log(self.activations[-1][np.argmax(y)])
					else:
						trainining_loss += 0.5*np.sum(np.square(y - self.activations[-1]))	#mean square error loss

				# REVERT BACK TO ORIGINAL Weights after FINDING GRADIENT
				self.weights = np.add(self.weights, velocity_weight)
				self.bias = np.add(self.bias,  velocity_bias)
				
				velocity_weight = velocity_weight  + (eta/(mini_batch_size)) * np.array(delta_cost)		# here old value is used to calculate new value
				velocity_bias = velocity_bias + (eta/(mini_batch_size)) * np.array( delta_bias_cost)

				self.weights = np.subtract(np.multiply( 1 - (eta * lmbda/n), self.weights),  velocity_weight)

				self.bias = np.subtract(self.bias,velocity_bias)



				#STORING VALUES
				if(step + 1) % 100 == 0:

					#Store error and loss values for each data set
					# self.print_log(t,step+1)
						
					# Store Weights and biases	
					self.store_weights()

			if(args.anneal):
				_,loss = self.evaluate(self.valid_set)
				if min_validation_loss_in_epoch > loss :
					min_validation_loss_in_epoch = loss
					

			if min_validation_loss_in_epoch - prev_min > 1:
				print('Minimum validation loss in last epoch is greater than minimum validation loss in this epoch --- Annealing Learning Rate')
				self.learning_rate = self.learning_rate/2
				continue


			print("On Test Set : accuracy = %d/10000 & loss = %f" %(self.evaluate() ))
			
			# USE THIS TO FIND OUT ETA(LEARNING RATE / STEP SIZE)
			print("Traingin Cost/Loss = ",(1/n)*trainining_loss + (lmbda/(2*n)) * (sum(np.sum(np.multiply(self.weights[i],self.weights[i]) ) for i in range(self.layers-1)  ) ) )


			# print("On Train Set : accuracy = %d/10000 & loss = %f" %(self.evaluate(self.train_set) ))
			print("time for epoch no.",(t+1),"=",time.time()-start_time )

			prev_min = min_validation_loss_in_epoch
			t = t + 1

	def ADAM(self,epoch,eta,mini_batch_size,lmbda = 0, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):		
		# eta is learning rate
		self.learning_rate = eta
		#print("bias\n",self.bias[-1])
		t=0
		prev_min = 1000000000

		while t < epoch:
			trainining_loss = 0
			min_validation_loss_in_epoch = 1000000000
			# number of examples in training data
			n = len(self.train_set)

			
			start_time = time.time()
			
			random.shuffle(self.train_set)

			mini_batches = [self.train_set[k:k+mini_batch_size]
				for k in range(0, n, mini_batch_size)]

			final_grad = [np.zeros(w.shape) for w in self.weights]
			velocity_weight = [np.zeros(w.shape) for w in self.weights] 
			beta1_exp = 1.0   # increasing precision is important as we are getting overflow , but it also slows down computation, it was faster t increase batch size than precision
			beta2_exp = 1.0
			
			for step,mini_batch in enumerate(mini_batches):
				delta_cost = [np.zeros(w.shape) for w in self.weights]  # Matrix to store Patrial Derivatices of cost function w.r.t weights
				delta_bias_cost = [np.zeros(b.shape) for b in self.bias]

				
				for i in range( len(mini_batch) ):
					
				#	print(i)
					x,y = mini_batch[i][0], mini_batch[i][1]

					#Forward Propagation
					self.forwardprop(x)
				#	print("after forward prop")

					#Back Propagation
					delta_cost_temp,delta_bias_cost_temp = self.backprop(x,y)
					delta_cost = [dc + dct for dc,dct in zip(delta_cost,delta_cost_temp)]
					delta_bias_cost = [dbc + dbct for dbc,dbct in zip(delta_bias_cost,delta_bias_cost_temp)]
				
					if self.activation_function_value == 0 : 
						trainining_loss = trainining_loss - np.argmax(y)*math.log(self.activations[-1][np.argmax(y)])
					else:
						trainining_loss += 0.5*np.sum(np.square(y - self.activations[-1]))	#mean square error loss


				

				beta1_exp *= beta1
				beta2_exp *= beta2
				delta_bias_cost = (1/mini_batch_size) * np.array(delta_bias_cost)
				delta_cost = np.array(delta_cost)
				velocity_weight = np.array(velocity_weight)
				# velocity_weight = np.array(velocity_weight)
				delta_cost = (1/(mini_batch_size) ) * np.array(delta_cost)

				final_grad = beta1 * np.array(final_grad )+ (1 - beta1)*np.array(delta_cost)
				velocity_weight = beta2 * velocity_weight + (1 - beta2)*(np.multiply(delta_cost,delta_cost))

				final_grad = np.array(final_grad)/(1-beta1_exp)
				velocity_weight = np.array(velocity_weight )/(1-beta2_exp)
				
				# print(np.sqrt(velocity_weight[i] for i in range(len(velocity_weight[0]))) )
				# ADDING L2 REGULARISATION
				for i in range(len(self.weights)) :
					self.weights[i] = np.subtract(np.multiply( 1 - (eta * lmbda/n), self.weights[i]), (eta/(np.sqrt(velocity_weight[i]) + epsilon))*final_grad[i] )

				self.bias = np.subtract(self.bias, (eta) *np.array( delta_bias_cost) )
				

				#STORING VALUES
				if(step + 1) % 100 == 0:

					#Store error and loss values for each data set
					# self.print_log(t,step+1)
					pass	
					# # Store Weights and biases	
					# self.store_weights()


			if(args.anneal):
				_,loss = self.evaluate(self.valid_set)
				if min_validation_loss_in_epoch > loss :
					min_validation_loss_in_epoch = loss
					

			print("accuracy = %d/10000 & loss = %f" %(self.evaluate() ))
			print("accuracy = %d/10000 & loss = %f" %(self.evaluate(self.test_set) ))
			print("Traingin Cost/Loss = ",(1/n)*trainining_loss + (lmbda/(2*n)) * (sum(np.sum(np.multiply(self.weights[i],self.weights[i]) ) for i in range(self.layers-1)  ) ) )
			print("time for epoch no.",(t+1),"=",time.time()-start_time )

			if min_validation_loss_in_epoch - prev_min > 1:
				print('Minimum validation loss in last epoch is greater than minimum validation loss in this epoch --- Annealing Learning Rate')
				self.learning_rate = self.learning_rate/2
				print('lr = ',self.learning_rate)
				prev_min = min_validation_loss_in_epoch
				continue

			prev_min = min_validation_loss_in_epoch
			t = t + 1

	def store_weights(self):
		path_weights = args.save_dir + '/weights.npy'
		file = open(path_weights,'a')   # Trying to create a new file or open one
		file.close()
		path_bias = args.save_dir + '/bias.npy'
		file = open(path_bias,'a')   # Trying to create a new file or open one
		file.close()
		np.save(path_weights,self.weights)
		np.save(path_bias,self.bias)

	def print_log(self,current_epoch,current_step):
		
		error, loss = self.evaluate(self.train_set)
		openfiles[0].write('Epoch %d, Step %d, Loss: %f, lr: %f\n'%(current_epoch,current_step,loss, self.learning_rate))
		openfiles[1].write('Epoch %d, Step %d, Error: %.2f, lr: %f\n'%(current_epoch,current_step,round((50000 - error)/500,2), self.learning_rate))


		error, loss  = self.evaluate(self.valid_set)
		openfiles[2].write('Epoch %d, Step %d, Loss: %f, lr: %f\n'%(current_epoch,current_step,loss, self.learning_rate))
		openfiles[3].write('Epoch %d, Step %d, Error: %.2f, lr: %f\n'%(current_epoch,current_step,round((10000 - error)/100,2), self.learning_rate))
	

		error, loss = self.evaluate(self.test_set)				
		openfiles[4].write('Epoch %d, Step %d, Loss: %f, lr: %f\n'%(current_epoch,current_step,loss, self.learning_rate))
		openfiles[5].write('Epoch %d, Step %d, Error: %.2f, lr: %f\n'%(current_epoch,current_step,round((10000 - error)/100,2), self.learning_rate))
		

		# file_path = args.save_dir + file_names[0]
		# with open(file_path, 'a') as f:
		# 	print('Epoch %d, Step %d, Loss: %f, lr: %f'%(current_epoch,current_step,loss, self.learning_rate) ,file=f )
		# file_path = args.save_dir + file_names[1]
		# with open(file_path, 'a') as f:
		# 	print('Epoch %d, Step %d, Error: %.2f, lr: %f'%(current_epoch,current_step,round((50000 - error)/500,2), self.learning_rate),file=f )	
		

		# file_path = args.save_dir + file_names[2]
		# with open(file_path, 'a') as f:
		# 	print('Epoch %d, Step %d, Loss: %f, lr: %f'%(current_epoch,current_step,loss, self.learning_rate) ,file=f )
		# file_path = args.save_dir + file_names[3]
		# with open(file_path, 'a') as f:
		# 	print('Epoch %d, Step %d, Error: %.2f, lr: %f'%(current_epoch,current_step,round((10000 - error)/100,2), self.learning_rate),file=f )	
		

		# file_path = args.save_dir + file_names[4]
		# with open(file_path, 'a') as f:
		# 	print('Epoch %d, Step %d, Loss: %f, lr: %f'%(current_epoch,current_step,loss, self.learning_rate) ,file=f )
		# file_path = args.save_dir + file_names[5]
		# with open(file_path, 'a') as f:
		# 	print('Epoch %d, Step %d, Error: %.2f, lr: %f'%(current_epoch,current_step,round((10000 - error)/100,2), self.learning_rate),file=f )	
		
# def SGD(self,epoch,eta,mini_batch_size):		
# 		# eta is learning rate

# 		#print("bias\n",self.bias[-1])
# 		self.epoch = epoch
# 		index = 0

# 		for t in range(self.epoch):
			
# 			n = len(self.train_set)

			
# 			start_time = time.time()
			
# 			random.shuffle(self.train_set)

# 			mini_batches = [self.train_set[k:k+mini_batch_size]
# 				for k in range(0, n, mini_batch_size)]
			
# 			for mini_batch in mini_batches:
# 				delta_cost = [np.zeros(w.shape) for w in self.weights]  # Matrix to store Patrial Derivatices of cost function w.r.t weights
# 				delta_bias_cost = [np.zeros(b.shape) for b in self.bias]
# 			# for num_iter in range(len(self.train_set[0]) // batch_size):
# 			# 	listL=[]
# 			# 	listR=[]
# 			# 	for k in range(batch_size):
# 			# #		index = randint(0,len(self.train_set[0] - 1) )  // This method was not good as some training examples might always get skipped, this will make the hypotheses fuction inacurate
					
# 			# 		listL.append(self.train_set[0][index])
# 			# 		listR.append(self.train_set[1][index])
# 			# 		index = (index + 1) % len(self.train_set[0])
# 			# 		# mini_batch[0][k] =   self.train_set[0][index]
# 			# 		# mini_batch[1][k] =   self.train_set[1][index]
# 			# 	mini_batch = (listL,listR)
			
# 				for i in range( len(mini_batch) ):
					
# 				#	print(i)
# 					x,y = mini_batch[i][0], mini_batch[i][1]

# 					#Forward Propagation
# 					self.forwardprop(x)
# 				#	print("after forward prop")

# 					#Back Propagation
# 					delta_cost_temp,delta_bias_cost_temp = self.backprop(x,y)
# 					delta_cost = [dc + dct for dc,dct in zip(delta_cost,delta_cost_temp)]
# 					delta_bias_cost = [dbc + dbct for dbc,dbct in zip(delta_bias_cost,delta_bias_cost_temp)]

# 				self.weights = np.subtract(self.weights, (eta/(mini_batch_size)) *np.array(delta_cost) )
# 				self.bias = np.subtract(self.bias, (eta/(mini_batch_size)) *np.array( delta_bias_cost) )

# 				#print("bias\n",delta_bias_cost)
# 			print(self.evaluate())
# 			print("time for epoch no.",(t+1),"=",time.time()-start_time )

	

	def forwardprop(self,input):
		self.activations = []
	
		self.activations.append(input)
		for i in range(self.layers - 1):
			activation = self.activation_function(np.add( np.dot(self.weights[i],self.activations[i]) , self.bias[i]) )	
			self.activations.append(activation)	

	def backprop(self,input,output):
		
		''' Finding Partial Derivatives of Cost Function(der_cost or CAPITAL DELTA) '''
		der_cost = [np.zeros(w.shape) for w in self.weights]  # Matrix to store Patrial Derivatices of cost function w.r.t weights
		der_cost_bias = [np.zeros(b.shape) for b in self.bias]
		der_act = self.activation_function(self.activations[-1],True)	

		if self.loss_function_value == 1 :
			delta = np.multiply(np.subtract( self.activations[-1],output),der_act)	  # activations[-1] gives last element of activation that is output layer values	
			# print('derivative = ',der_act)
		else :
			delta = np.subtract( self.activations[-1],output)
		

		for t in range(self.layers-1,0,-1):	
			der_cost_bias[t-1] = delta		
			der_cost[t-1] =   np.dot(delta, np.transpose(self.activations[t-1])) 	# this delta is of layer t+1, that for the first case of the output layer , there is no delta for input layer there there is not error in activation value of input layer which is the input values itself
			

			#check this change
			der_act = self.activation_function(self.activations[t-1],True)			# derivative of activation vector of layer
			delta = np.multiply ( np.dot( np.transpose(self.weights[t-1]), delta) , der_act )		 # (small)delta of previous layer 

		return (der_cost,der_cost_bias)
		

	def evaluate(self,evaluation_set = None):
		"""Return the number of test inputs for which the neural
		network outputs the correct result. Note that the neural
		network's output is assumed to be the index of whichever
		neuron in the final layer has the highest activation."""
		
		
		if evaluation_set == None:
			evaluation_set = self.valid_set

		# random.shuffle(evaluation_set)
		evaluation_set_loss = 0
		test_results = []
		for i in range(len(evaluation_set)):
			if(evaluation_set == self.train_set):
				x,y = evaluation_set[i][0],np.argmax(evaluation_set[i][1])
			else:
				x,y = evaluation_set[i][0],evaluation_set[i][1]
			self.forwardprop(x)

			test_results.append( (np.argmax(self.activations[-1]), y )	)	#argmax returns indices

			if self.activation_function_value == 0 : 
				evaluation_set_loss = evaluation_set_loss - y*math.log(self.activations[-1][y])
			else:
				evaluation_set_loss += 0.5*(np.square(y - self.activations[-1][y]))	#mean square error loss

	   
		evaluation_set_loss = (1/len(evaluation_set) ) * evaluation_set_loss

		return (sum(int(x == y) for (x, y) in test_results), evaluation_set_loss)




def start():
	nodesInLayers =[784] + args.sizes + [10]

	net = NN(nodesInLayers,args.momentum, args.activation, args.loss)

	net.setup()
	opt_algo = args.opt

	epoch = args.epoch


	if opt_algo == 0:		# JUST FOR safe check
		args.momentum = 0

	if opt_algo == 3:
		net.ADAM(epoch, args.lr, args.batch_size)
	elif opt_algo == 2:
		net.NAG(epoch, args.lr, args.batch_size)
	else:
		net.SGD(epoch, args.lr, args.batch_size)		# for 0 and 1 , 0 = SGD , 1 = MOMENTUM


	for i in range(6):
		openfiles[i].close()

	file = open(args.expt_dir + '/valid_predictions.txt','w')
	for x,_ in net.valid_set:
		net.forwardprop(x)
		file.write( '%d \n' %(np.argmax(net.activations[-1])) )
	file.close()


	file = open(args.expt_dir + '/test_predictions.txt','w')
	for x,_ in net.test_set:
		net.forwardprop(x)
		file.write( '%d \n' %(np.argmax(net.activations[-1])) )
	file.close()

if __name__ == "__main__":

	start()

  

