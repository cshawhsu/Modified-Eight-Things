import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import FunctionsforEightThings as f

#reproducibility
tf.reset_default_graph()
numpy.random.seed(1234)
tf.set_random_seed(1234)

#set properties of neural net such as size, input, output
#determined by dataset used, or if we create own dataset
nDomains = 4
nItemsPerDomain = 8
nSingleOutputSize = 36 #output size per domain, depends on dataset

nSizeofItemInput = nDomains*nItemsPerDomain #size of item input vector
#nSizeofContextInput = nDomains*nContextPerDomain#size of context input vector

nSizeofInput = nItemsPerDomain*nDomains #length of a single input vector (context + item)
nSizeofOutput = nSingleOutputSize*nDomains # length of a single output vector
nNumberOfPossibleInputPatterns = nItemsPerDomain*nDomains #number of training examples in dataset

sizeItemRepLayer = 6 #ize of first item hidden layer
sizeSecondLayer = 8  #size of second hidden layer

#training parameter
nEpochs = 1000
eta = 0.05
batch_size = nNumberOfPossibleInputPatterns

#construct the input matrix, same as eight things dataset, but made our own
InputMatrix= f.input_matrix(nDomains, 1, nItemsPerDomain)
print InputMatrix.shape
numpy.savetxt('Input.txt', InputMatrix, fmt = '%u')

#construct the output matrix, from eight things dataset format
a,b,c = f.read_data_file('EightThingsshaw2.txt')
tempb = f.context_collapse(b,4)
OutputMatrix = f.make_multi_domain(tempb,nDomains)
print OutputMatrix.shape
numpy.savetxt('Output.txt', OutputMatrix, fmt = '%u')

#set up network below
inputvec_item = tf.placeholder(tf.float32, shape = [nSizeofItemInput,None])#input item vector
outputvec = tf.placeholder(tf.float32, shape = [nSizeofOutput,None])#output vector

#multiplied weights
W1_item = tf.Variable(tf.random_uniform([sizeItemRepLayer,nSizeofItemInput],-1,1))*0.1 #weights from item input to item hidden layer1
B1_item = tf.Variable(tf.random_uniform([sizeItemRepLayer,1],-1,1))*0.1 #bias from item input to item hidden layer1

W2_item = tf.Variable(tf.random_uniform([sizeSecondLayer,sizeItemRepLayer],-1,1))*0.1#weights from item hidden layer1 to layer2
B2_item = tf.Variable(tf.random_uniform([sizeSecondLayer,1],-1,1))*0.1 #bias from item hidden layer1 to layer2

W3 = tf.Variable(tf.random_uniform([nSizeofOutput,sizeSecondLayer],-1,1))*0.1 #weights from context hidden layer1 to layer2
B3 = tf.Variable(tf.random_uniform([nSizeofOutput,1],-1,1))*0.1 #bias from context hidden layer1 to layer2

#regular weights
#W1_item = tf.Variable(tf.random_uniform([sizeItemRepLayer,nSizeofItemInput],-1,1))#*0.05 #weights from item input to item hidden layer1
#B1_item = tf.Variable(tf.random_uniform([sizeItemRepLayer,1],-1,1))#*0.05 #bias from item input to item hidden layer1
#
#W2_item = tf.Variable(tf.random_uniform([sizeSecondLayer,sizeItemRepLayer],-1,1))#*0.05#weights from item hidden layer1 to layer2
#B2_item = tf.Variable(tf.random_uniform([sizeSecondLayer,1],-1,1))#*0.05 #bias from item hidden layer1 to layer2
#
#W3 = tf.Variable(tf.random_uniform([nSizeofOutput,sizeSecondLayer],-1,1))#*0.05 #weights from context hidden layer1 to layer2
#B3 = tf.Variable(tf.random_uniform([nSizeofOutput,1],-1,1))#*0.05 #bias from context hidden layer1 to layer2

#construct network
itemhidden = tf.nn.sigmoid(tf.matmul(W1_item,inputvec_item) + B1_item)
secondhidden = tf.nn.sigmoid((tf.matmul(W2_item,itemhidden) + B2_item))
output = tf.nn.sigmoid(tf.matmul(W3, secondhidden) + B3)


#batch learning error (this is MSE)
error = tf.reduce_mean(tf.reduce_sum(tf.square(output - outputvec),0))

#non-batch learning error for single input
#error2 = tf.reduce_sum(tf.square(output - outputvec))

train = tf.train.GradientDescentOptimizer(eta).minimize(error)
#train = tf.train.AdamOptimizer(eta).minimize(error)

#some functions
def train_error():
    train_error = 0.0
    #batch training
    train_error = session.run(error, feed_dict = {inputvec_item: numpy.transpose(InputMatrix[0:nNumberOfPossibleInputPatterns,0:nSizeofItemInput]), 
    									outputvec: numpy.transpose(OutputMatrix[0:nNumberOfPossibleInputPatterns])})
    return train_error
    
    
    #seperate training
#    for i in range(nNumberOfPossibleInputPatterns):
#	    train_error += session.run(error, feed_dict = {inputvec_item: InputMatrix[i][0:(nSizeofItemInput)].reshape([nSizeofItemInput,1]),	
#				inputvec_context: InputMatrix[i][nSizeofItemInput:].reshape([nSizeofContextInput,1]), 
#				outputvec: OutputMatrix[i].reshape([nSizeofOutput,1])})
#    return train_error/nNumberOfPossibleInputPatterns 

def get_network_outputs():
    network_outputs = numpy.zeros([nNumberOfPossibleInputPatterns,nSizeofOutput])
    for i in range(nNumberOfPossibleInputPatterns):
	    network_outputs[i,:] = session.run(output, feed_dict = {inputvec_item: InputMatrix[i][0:(nSizeofItemInput)].reshape([nSizeofItemInput,1]),	 
				outputvec: OutputMatrix[i].reshape([nSizeofOutput,1])}).flatten()
    return network_outputs


def get_item_reps():
    item_reps = numpy.zeros([nNumberOfPossibleInputPatterns,sizeItemRepLayer])
    for i in range(nNumberOfPossibleInputPatterns):
	    item_reps[i,:] = session.run(itemhidden, feed_dict = {inputvec_item: InputMatrix[i][0:(nSizeofItemInput)].reshape([nSizeofItemInput,1]),	 
				outputvec: OutputMatrix[i].reshape([nSizeofOutput,1])}).flatten()
    return item_reps

def get_second_reps():
    second_reps = numpy.zeros([nNumberOfPossibleInputPatterns,sizeSecondLayer])
    for i in range(nNumberOfPossibleInputPatterns):
	    second_reps[i,:] = session.run(secondhidden, feed_dict = {inputvec_item: InputMatrix[i][0:(nSizeofItemInput)].reshape([nSizeofItemInput,1]),	 
            				outputvec: OutputMatrix[i].reshape([nSizeofOutput,1])}).flatten()
    return second_reps

def log_images(epoch=0):
	network_outputs = get_network_outputs()
     #plt.imshow(numpy.dot(network_outputs,OutputMatrix.transpose()),cmap='Greys',interpolation='none')
	plt.imshow(f.euclid_dis_matrix(network_outputs),cmap='Greys_r',interpolation='none')
	plt.savefig('output/epoch_%i_networks_output_vs_output.png' %epoch)
	plt.close()

 
	item_reps = get_item_reps()
	itemrep_sim = f.euclid_dis_matrix(item_reps)
	#numpy.savetxt('itemrepsim.txt', testing_sim)
	plt.imshow(itemrep_sim, cmap='Greys_r',interpolation='none') #cosine distance
	plt.savefig('item/epoch_%i_item_rep_similarity.png' %epoch)
	plt.close()



	second_reps = get_second_reps() 
	second_rep_sim = f.euclid_dis_matrix(second_reps)
	plt.imshow(second_rep_sim,cmap='Greys_r',interpolation='none') #cosine distance
	plt.savefig('secondlayer/epoch_%i_second_rep_similarity.png' %epoch)
	plt.close()


#listerror = []
#run the network
init = tf.initialize_all_variables()
with tf.Session() as session:
	session.run(init)
	print "Initial training MSE: ",train_error()
	#listerror.append(train_error())
	log_images(0)
	for epoch in xrange(1, nEpochs+1):
         batch_start = 0
         batch_end = batch_size
         while batch_start < nNumberOfPossibleInputPatterns:
             if batch_end > nNumberOfPossibleInputPatterns:
                 batch_end = nNumberOfPossibleInputPatterns          
                 
             _, trainerror = session.run([train,error], feed_dict = {inputvec_item: numpy.transpose(InputMatrix[batch_start:batch_end,0:nSizeofItemInput]), 
                                                 outputvec: numpy.transpose(OutputMatrix[batch_start:batch_end])})
             batch_start = batch_end
             batch_end = batch_end + batch_size

             
		#train one training example at a time, random order
#		this_order = numpy.random.permutation(nNumberOfPossibleInputPatterns) #train example in random order each time
#		for i in this_order:
#			session.run(train, feed_dict = {inputvec_item: InputMatrix[i%nNumberOfPossibleInputPatterns][0:(nSizeofItemInput)].reshape([nSizeofItemInput,1]), 
#											inputvec_context: InputMatrix[i%nNumberOfPossibleInputPatterns][nSizeofItemInput:].reshape([nSizeofContextInput,1]),
#											outputvec: OutputMatrix[i%nNumberOfPossibleInputPatterns].reshape([nSizeofOutput,1])})
#		
            
		#train with domain A first
		# if epoch < 1500:
		# 	for i in xrange(nNumberOfPossibleInputPatterns//nDomains):
		# 		session.run(train, feed_dict = {inputvec_item: InputMatrix[i%nNumberOfPossibleInputPatterns][0:(nSizeofItemInput)].reshape([nSizeofItemInput,1]), 
		# 										inputvec_context: InputMatrix[i%nNumberOfPossibleInputPatterns][nSizeofItemInput:].reshape([nSizeofContextInput,1]),
		# 										outputvec: OutputMatrix[i%nNumberOfPossibleInputPatterns].reshape([nSizeofOutput,1])})
		# else:
		# 	for i in xrange(nNumberOfPossibleInputPatterns):
		# 		session.run(train, feed_dict = {inputvec_item: InputMatrix[i%nNumberOfPossibleInputPatterns][0:(nSizeofItemInput)].reshape([nSizeofItemInput,1]), 
		# 									inputvec_context: InputMatrix[i%nNumberOfPossibleInputPatterns][nSizeofItemInput:].reshape([nSizeofContextInput,1]),
		# 									outputvec: OutputMatrix[i%nNumberOfPossibleInputPatterns].reshape([nSizeofOutput,1])})


  
		#listerror.append(train_error())
         if epoch%1000 == 0:
              #print "On epoch %i, training MSE: %f" %(epoch, listerror[-1])
              print "On epoch %i, training MSE: %f" %(epoch, trainerror)
              
              
         if epoch%1000 == 0:
              log_images(epoch)   
 


# plt.plot(range(nEpochs+1), listerror)
# plt.savefig('error plot.png')
#plt.show()

#plt.plot([numpy.mean(listerror[k - nNumberOfPossibleInputPatterns:k]) for k in range(0,len(listerror),10)])
#plt.show()




