import numpy as np

def process_data(data,mean=None,std=None):
    # normalize the data to have zero mean and unit variance (add 1e-15 to std to avoid numerical issue)
    if mean is not None:
        # directly use the mean and std precomputed from the training data
        for i in range(len(data)):
            for j in range(len(data[0])):
                data[i][j] = (data[i][j] - mean[j])/std[j]
                
        return data
    else:
        # compute the mean and std based on the training data
        mean = std = 0 # placeholder
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-15
        
        for i in range(len(data)): # normalizing by x-u/std
            for j in range(len(data[0])):
                data[i][j] = (data[i][j] - mean[j])/std[j]
                
        return data, mean, std

def process_label(label):
    # convert the labels into one-hot vector for training
    one_hot = np.zeros([len(label),10])
    
    for i in range(len(label)):
        one_hot[i][label[i]] = 1

    return one_hot

def tanh(x):
    # implement the hyperbolic tangent activation function for hidden layer
    # You may receive some warning messages from Numpy. No worries, they should not affect your final results
    
    #e = np.exp(x)
    #en = np.exp(-(x))
    #f_x = (e - en)/ (e + en)
    
    f_x = np.tanh(x)
    return f_x

def softmax(x):
    # implement the softmax activation function for output layer

    f_x = []

    for i in range(len(x)):
        e = np.exp(x[i])
        #soft = e/e.sum()
        soft = e/np.sum(e,axis=0)
        f_x.append(soft)


    return f_x

class MLP:
    def __init__(self,num_hid):
        # initialize the weights
        self.weight_1 = np.random.random([64,num_hid])
        self.bias_1 = np.random.random([1,num_hid])
        self.weight_2 = np.random.random([num_hid,10])
        self.bias_2 = np.random.random([1,10])
        self.num_hid = num_hid
    def fit(self,train_x,train_y, valid_x, valid_y):
        # learning rate
        lr = 5e-3
        # counter for recording the number of epochs without improvement
        count = 0
        best_valid_acc = 0
        
        
        """
        Stop the training if there is no improvment over the best validation accuracy for more than 50 iterations
        """
        
        while count<=50:
            # training with all samples (full-batch gradient descents)
            # implement the forward pass (from inputs to predictions)
            
            # z = train_x dot self.weight_1 + bias_1
            # z tanh(z) 
            # y = z dot weight_2 + bias_2
            # probs = softmax(y) 1000 rows

            z = np.add(np.dot(train_x, self.weight_1) , self.bias_1) # 1000 x 4
            z = tanh(z) # 1000 x 4

            y = np.add(np.dot(z, self.weight_2) , self.bias_2) # 1000 x 10
            probs = softmax(y)
            
            # implement the backward pass (backpropagation)
            # compute the gradients w.r.t. different parameters
            
            # delta_V = lr *  (r_i - y_i)*Z_h     1000x10
            # delta_v = lr * zh.T dot (ri - yi)   hidx10
            sub = np.subtract(train_y , probs)
            
            delta_v  = lr * (np.dot(z.T,sub))
            delta_b2 = lr * sub
            delta_b2 = np.sum(delta_b2,axis=0)
            delta_b2 = np.reshape(delta_b2,[1,10])
            
            # delta_w = lr * signa_t (sigma_i (r_i - y_i)* self.weight_2.T) * (1-Z_h^2) * Xj(train_x)
                                            #1000 x 10  10 x hid > 1000xhid     1000xhid  1000x64
            # delta w = lr * train_x.T dot [dot product r-y and weight2 ] * (1-z^2)
            # 1 r-y dot weiht 2
            # 2 1 * 1-z2
            # 3 dot product train_x.T and step 2
            # 4 * lr. 64 x hid
            
            # delta_b1 = lr * sigma_t (sigma_i (r_i - y_i)* weight_2) * (1-Z_h^2) 
            # 1000 x hid > 1 hid
            w_temp1  = np.dot(sub, self.weight_2.T)
            w_temp2  = w_temp1 * (1 - np.power(z,2))
            w_temp3  = np.dot(train_x.T,w_temp2)
            delta_w  = lr * w_temp3

            delta_b1 = lr * w_temp2
            delta_b1 = np.sum(delta_b1,axis=0)
            delta_b1 = np.reshape(delta_b1,[1,self.num_hid])

            #update the parameters based on sum of gradients for all training samples
            self.weight_2 += delta_v
            self.bias_2   += delta_b2
            self.weight_1 += delta_w
            self.bias_1   += delta_b1


            # evaluate on validation data
            predictions = self.predict(valid_x)
            valid_acc = np.count_nonzero(predictions.reshape(-1)==valid_y.reshape(-1))/len(valid_x)

            # compare the current validation accuracy with the best one
            if valid_acc>best_valid_acc:
                best_valid_acc = valid_acc
                count = 0
            else:
                count += 1

        return best_valid_acc

    def predict(self,x):
        # generate the predicted probability of different classes
        y = np.zeros([len(x),]).astype('int') # placeholder
        z = np.add(np.dot(x, self.weight_1) , self.bias_1) # 1000 x 4
        z = tanh(z) # 1000 x 4
        y_soft = np.add(np.dot(z, self.weight_2) , self.bias_2) # 1000 x 10
        probs = softmax(y_soft)
        
        for t in range(len(x)):

            try:
                # value is a 1x10 vector that converts preditions to numbers 0-9
                # the highest prediction moves to the front of the value[0]
                value = np.hstack(np.where(probs[t] == probs[t].max()))
                y[t] = value[0]
            except:
                value = np.array([0])
                y[t] = value[0]

        return y

    def get_hidden(self,x):
        # extract the intermediate features computed at the hidden layers (after applying activation function)

        zh = np.dot(x, self.weight_1) # 1000 x 4
        z = np.tanh(zh) # 1000 x 4

        return z

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2