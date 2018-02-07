'''
	References Used : 
		https://www.coursera.org/learn/machine-learning/supplement/pjdBA/backpropagation-algorithm
		http://neuralnetworksanddeeplearning.com/chap1.html

	IMPORTANT : for speedup its necessary that  the input data is in the required format, 1 column vector otherwise it has to be converted into that form again and again in each epoch which is a redundant operation and takes a lot of time
				(this was a huge speedup from 250s to 50 s for each epoch)
				converted formats for input of train_data,test_data and valid_data and output of train_data

				Second speed up(in setup part) : use np.reshape(x,(784,1)) when you know the size of x instead of np.vstack, reshape is lot faster: np.vstack has a completely different purpose, it is used to add a row to an array

	The only difference in cross entropy and mean square error is that in mse in (INITIALLY for last layer weights)delta = np.subtract( self.activations[-1],output)*der_act(here derivative is multiplied so learning(i.e weight update) slows down as derivative becomes 0) whereas in ce #delta = np.subtract( self.activations[-1],output) --- > refer : http://neuralnetworksanddeeplearning.com/chap3.html#the_cross-entropy_cost_function

	Caution : At some points ,code gives :  RuntimeWarning: overflow encountered in exp
  return 1.0 / (1.0 + np.exp(-z) ) ---> This is probably because the value of -z becomes too negative and hence it creates problem for exp function, try to scale down hyperparameters to avoid this
	MORE INSIGHT ON THIS ERROR : https://stats.stackexchange.com/questions/210726/neural-network-can-only-follow-increasing-function

	Step Size = Learning Rate , EXTRA Addition : L2 Regularisation(Proabably it helps in correcting the Runtime error due to over shooting of weights)(BUT RESULTS WERE BETTER WITHOUT ANY REGULARISATION)
'''

import numpy as np
#Python 3 , in Python 2 the library is named cPickle
import pickle 
import gzip
import time

import operator
import random

class NN(object):

	'''
		INIT fuction, nodes in layers is an aray containing the number of neurons in each layer 
		nodesInLayers = [784,15,10]
	'''
	def __init__(self, nodesInLayers,momentum = 0.5):  # WHY USING MOMENTUM OF VALUE 1 GAVE SATURATION AROUND 900 correct detections
			# the value of momentum makes things a bit uncertain, keep it below 0.9
		self.layers = len(nodesInLayers)
		self.sizes = nodesInLayers 		#total number of units(neurons) in each layer, excluding the biases
		self.bias = [np.random.randn(y,1) for y in nodesInLayers[1:] ]	 # bias node of current level will have weights for each neuron(node/unit) of the next layer
		self.weights = [np.random.randn(y,x) for x,y in zip(nodesInLayers,nodesInLayers[1:])] # weights matrix for all nodes between successive layers(except bias weights)
		self.activations = []
		self.activation_function_value = 0
		self.loss_function_value = 0 #( 0 means cross entropy , 1 means mean square error)
		self.momentum_coefficient = momentum

		#Load Data
		f = gzip.open('mnist.pkl.gz', 'rb')
		self.train_set, self.valid_set, self.test_set = pickle.load(f, encoding='latin1')
		f.close()
		

	def setup(self):
		self.test_set = self.format_data(self.test_set)
		self.valid_set = self.format_data(self.valid_set)
		self.train_set = self.format_data_train(self.train_set)


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

		if self.activation_function_value == 0 :
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
	def SGD(self,numIterations,eta,mini_batch_size,lmbda = 0.1):		
		# eta is learning rate

		#print("bias\n",self.bias[-1])

			

		for t in range(numIterations):
			
			# number of examples in training data
			n = len(self.train_set)
			velocity_weight = [np.zeros(w.shape) for w in self.weights] 
			velocity_bias = [np.zeros(b.shape) for b in self.bias]
			
			start_time = time.time()
			
			random.shuffle(self.train_set)

			mini_batches = [self.train_set[k:k+mini_batch_size]
				for k in range(0, n, mini_batch_size)]

			

			for mini_batch in mini_batches:
				delta_cost = [np.zeros(w.shape) for w in self.weights]  # Matrix to store Patrial Derivatices of cost function w.r.t weights
				delta_bias_cost = [np.zeros(b.shape) for b in self.bias]

				


			# for num_iter in range(len(self.train_set[0]) // batch_size):
			# 	listL=[]
			# 	listR=[]
			# 	for k in range(batch_size):
			# #		index = randint(0,len(self.train_set[0] - 1) )  // This method was not good as some training examples might always get skipped, this will make the hypotheses fuction inacurate
					
			# 		listL.append(self.train_set[0][index])
			# 		listR.append(self.train_set[1][index])
			# 		index = (index + 1) % len(self.train_set[0])
			# 		# mini_batch[0][k] =   self.train_set[0][index]
			# 		# mini_batch[1][k] =   self.train_set[1][index]
			# 	mini_batch = (listL,listR)
			
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

				
				velocity_weight = np.multiply(self.momentum_coefficient , velocity_weight )  # here old value of velocity is used
				velocity_bias = np.multiply(self.momentum_coefficient , velocity_bias ) 
				
				velocity_weight = velocity_weight  +  (eta/mini_batch_size) * np.array(delta_cost)		# here old value is used to calculate new value
				velocity_bias = velocity_bias +  (eta/(mini_batch_size)) * np.array( delta_bias_cost)

				# ADDING L2 REGULARISATION
				self.weights = np.subtract(np.multiply( 1 - (eta * lmbda/n), self.weights), velocity_weight)

				self.bias = np.subtract(self.bias,  velocity_bias)

				#print("bias\n",delta_bias_cost)
			print(self.evaluate())
			print("time for epoch no.",(t+1),"=",time.time()-start_time )




	def NAG(self,numIterations,eta,mini_batch_size,lmbda=0):	#Nesterov Accelerated Gradient	
		# eta is learning rate

		#print("bias\n",self.bias[-1])
		

		for t in range(numIterations):
			
			n = len(self.train_set)
			start_time = time.time()
			
			random.shuffle(self.train_set)

			mini_batches = [self.train_set[k:k+mini_batch_size]
				for k in range(0, n, mini_batch_size)]

			velocity_weight = [np.zeros(w.shape) for w in self.weights] 
			velocity_bias = [np.zeros(b.shape) for b in self.bias]

			for mini_batch in mini_batches:
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

				# REVERT BACK TO ORIGINAL Weights after FINDING GRADIENT
				self.weights = np.add(self.weights, velocity_weight)
				self.bias = np.add(self.bias,  velocity_bias)
				
				velocity_weight = velocity_weight  + (eta/(mini_batch_size)) * np.array(delta_cost)		# here old value is used to calculate new value
				velocity_bias = velocity_bias + (eta/(mini_batch_size)) * np.array( delta_bias_cost)

				self.weights = np.subtract(np.multiply( 1 - (eta * lmbda/n), self.weights),  velocity_weight)

				self.bias = np.subtract(self.bias,velocity_bias)

				#print("bias\n",delta_bias_cost)
			print(self.evaluate())
			print("time for epoch no.",(t+1),"=",time.time()-start_time )


	def ADAM(self,numIterations,eta,mini_batch_size,lmbda = 0.1, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):		
		# eta is learning rate

		#print("bias\n",self.bias[-1])

		for t in range(numIterations):
			
			# number of examples in training data
			n = len(self.train_set)

			
			start_time = time.time()
			
			random.shuffle(self.train_set)

			mini_batches = [self.train_set[k:k+mini_batch_size]
				for k in range(0, n, mini_batch_size)]

			final_grad = [np.zeros(w.shape) for w in self.weights]
			velocity_weight = [np.zeros(w.shape) for w in self.weights] 
			beta1_exp = 1.0
			beta2_exp = 1.0

			for mini_batch in mini_batches:
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

				beta1_exp *= beta1
				beta2_exp *= beta2
				delta_bias_cost = (1/mini_batch_size) * np.array(delta_bias_cost)
				delta_cost = np.array(delta_cost)
				velocity_weight = np.array(velocity_weight)
				# velocity_weight = np.array(velocity_weight)
				delta_cost = (1/(mini_batch_size) ) * np.array(delta_cost)

				final_grad = beta1 * np.array(final_grad )+ (1 - beta1)*np.array(delta_cost)
				velocity_weight = beta2 * velocity_weight + (1 - beta2)*(np.multiply(delta_cost,delta_cost))

				final_grad = (1/(1-beta1_exp))*np.array(final_grad)
				velocity_weight = (1/(1-beta2_exp))*np.array(velocity_weight)
				
				# print(np.sqrt(velocity_weight[i] for i in range(len(velocity_weight[0]))) )
				# ADDING L2 REGULARISATION
				for i in range(len(self.weights)) :
					self.weights[i] = np.subtract(np.multiply( 1 - (eta * lmbda/n), self.weights[i]), (eta/(np.sqrt(velocity_weight[i]) + epsilon))*final_grad[i] )

				self.bias = np.subtract(self.bias, (eta) *np.array( delta_bias_cost) )
				#print("bias\n",delta_bias_cost)
			print(self.evaluate())
			print("time for epoch no.",(t+1),"=",time.time()-start_time )



# def SGD(self,numIterations,eta,mini_batch_size):		
# 		# eta is learning rate

# 		#print("bias\n",self.bias[-1])
# 		self.numIterations = numIterations
# 		index = 0

# 		for t in range(self.numIterations):
			
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
		# TRY REPLACING VALUES HERE INSTEAD OF APPENDING EACH TIME
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
			delta = np.subtract( self.activations[-1],output)*der_act	  # activations[-1] gives last element of activation that is output layer values	
		else :
			delta = np.subtract( self.activations[-1],output)
		

		for t in range(self.layers-1,0,-1):	
			der_cost_bias[t-1] = delta		
			der_cost[t-1] =   np.dot(delta, np.transpose(self.activations[t-1])) 	# this delta is of layer t+1, that for the first case of the output layer , there is no delta for input layer there there is not error in activation value of input layer which is the input values itself
			

			#check this change
			der_act = self.activation_function(self.activations[t-1],True)			# derivative of activation vector of layer
			delta = np.multiply ( np.dot( np.transpose(self.weights[t-1]), delta) , der_act )		 # (small)delta of previous layer 

		return (der_cost,der_cost_bias)
		

	def evaluate(self):
		"""Return the number of test inputs for which the neural
		network outputs the correct result. Note that the neural
		network's output is assumed to be the index of whichever
		neuron in the final layer has the highest activation."""
		#random.shuffle(self.test_set)
		evaluation_set = self.valid_set	# or test_set
		test_results = []
		for i in range(len(evaluation_set)):
			x,y = evaluation_set[i][0],evaluation_set[i][1]
			self.forwardprop(x)

			test_results.append( (np.argmax(self.activations[-1]), y)	)	#argmax returns indices
		# print(test_results[0])		   
		return sum(int(x == y) for (x, y) in test_results)

