import numpy as np

class GaussianDiscriminant:
    def __init__(self,k=2,d=8,priors=None,shared_cov=False):
        self.mean = np.zeros((k,d)) # mean
        self.shared_cov = shared_cov # using class-independent covariance or not
        if self.shared_cov:
            self.S = np.zeros((d,d)) # class-independent covariance
        else:
            self.S = np.zeros((k,d,d)) # class-dependent covariance
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d

    def fit(self, Xtrain, ytrain):
        # compute the mean for each class
        c1,c2 = [],[]
        for d in range(0,100):
            if ytrain[d] == 1:
                c1.append(Xtrain[d,:])
            if ytrain[d] == 2:
                c2.append(Xtrain[d,:])

        C1 = np.array(c1)
        C2 = np.array(c2)
        mu1 = np.mean(C1,axis=0)
        mu2 = np.mean(C2,axis=0)
        self.mean = [mu1,mu2]

        if self.shared_cov:
            # compute the class-independent covariance
            s1 = np.cov(C1.T, ddof=0)
            # class 2
            s2 = np.cov(C2.T, ddof=0)
            
            # Common Covariance
            common_S = (self.p[0] * s1) + (self.p[1] * s2)
            
            self.S = np.array(common_S)
            #pass # placeholder
        else:
            # compute the class-dependent covariance
            # class 1
            s1 = np.cov(C1.T, ddof=0)
            # class 2
            s2 = np.cov(C2.T, ddof=0)
            
            self.S = np.array((s1,s2))
            #pass

    def predict(self, Xtest):
        # predict function to get predictions on test set
        predicted_class = np.ones(Xtest.shape[0]) # placeholder

        for i in np.arange(Xtest.shape[0]): # for each test set example
            gi = []
            # calculate the value of discriminant function for each class
            for c in np.arange(self.k):
                if self.shared_cov:
                    
                    inv_si = np.linalg.inv(self.S)
                    
                    wi = np.dot(inv_si, self.mean[c])
                    wi0 = -(np.dot(self.mean[c].T,np.dot(inv_si,self.mean[c]))/2) + np.log(self.p[c])

                    gi.append(np.dot(wi.T,Xtest[i]) + wi0)
                    pass # placeholder
                else:

                    inv_si = np.linalg.inv(self.S[c])
                    det_si = np.linalg.det(self.S[c])

                    Wi = -(inv_si)/2
                    wi = np.dot(inv_si, self.mean[c])
                    wi0 = -((np.dot(self.mean[c].T, np.dot(inv_si,self.mean[c])))/2) - ((np.log(det_si))/2)\
                    + np.log(self.p[c])

                    gi.append(np.dot(Xtest[i], np.dot(Wi,Xtest[i].T)) + np.dot(wi.T,Xtest[i]) + wi0)
                    pass
            # determine the predicted class based on the values of discriminant function
            # compare their posteriors 
            if gi[0]>gi[1]:
                predicted_class[i] = 1
            else:
                predicted_class[i] = 2

        return predicted_class

    def params(self):
        if self.shared_cov:
            return self.mean[0], self.mean[1], self.S
        else:
            return self.mean[0],self.mean[1],self.S[0,:,:],self.S[1,:,:]


class GaussianDiscriminant_Diagonal:
    def __init__(self,k=2,d=8,priors=None):
        self.mean = np.zeros((k,d)) # mean
        self.S = np.zeros((d,)) # variance
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d
    def fit(self, Xtrain, ytrain):
        # compute the mean for each class
        c1,c2 = [],[]
        for d in range(0,100):
            if ytrain[d] == 1:
                c1.append(Xtrain[d,:])
            if ytrain[d] == 2:
                c2.append(Xtrain[d,:])

        C1 = np.array(c1)
        C2 = np.array(c2)
        mu1 = np.mean(C1,axis=0)
        mu2 = np.mean(C2,axis=0)
        self.mean = [mu1,mu2]
        # compute the variance of different features
        var = np.var(Xtrain,axis=0)
        self.S = np.array(var)

        pass # placeholder

    def predict(self, Xtest):
        # predict function to get prediction for test set
        predicted_class = np.ones(Xtest.shape[0]) # placeholder

        for i in np.arange(Xtest.shape[0]): # for each test set example
            gi = []
            # calculate the value of discriminant function for each class
            for c in np.arange(self.k):
                sigma = 0
                for j in range(0,8):
                    sigma += ((Xtest[i][j] - self.mean[c][j])/self.S[j])**2
                
                gi.append(-(sigma/2) + np.log(self.p[c]) )
                
                pass

            # determine the predicted class based on the values of discriminant function
            if gi[0]>gi[1]:
                predicted_class[i] = 1
            else:
                predicted_class[i] = 2

        return predicted_class

    def params(self):
        return self.mean[0], self.mean[1], self.S
