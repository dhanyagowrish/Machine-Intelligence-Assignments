# import packages

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


# The neural network has 1 hidden layer
# input layer has 12 neurons ( 12 dimensional input feature vector )
# hidden layer has 14 neurons and the activation function is ReLu
# output layer has 1 neuron and the activation function is Sigmoid
# Loss function used: Cross Entropy loss

# creating Neural Network class
class NN:
    
    def __init__(self):
        self.x=None
        self.y=None
        self.parameters={}     # dictionary with weights of the layers
        self.layers=[12,14,1]
        self.learning_rate=0.001
        self.iterations=800
        self.loss=[]
    
    def initialize_weights(self):
        
        np.random.seed(1)   # seed the random function
        
        # initializing the weight matrices for the layers of appropriate dimensions
        # the values in the matrices are initialized with random values sampled from normal Gaussian distribution
        
        # w1 is weight matrix with dimensions: 12x14
        # b1 is bias matrix with dimensions: 1x14
        # w2 is weight matrix with dimensions: 14x1
        # b2 is bias matric with dimensions: 1x1

        self.parameters["w1"] = np.random.randn(self.layers[0], self.layers[1]) 
        self.parameters["b1"]  =np.random.randn(self.layers[1],)
        self.parameters["w2"] = np.random.randn(self.layers[1],self.layers[2]) 
        self.parameters["b2"] = np.random.randn(self.layers[2],)
        
    
    # function to return sigmoid(z)
    # takes real numbers and outputs a real valued output between 0 and 1
    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))
    
    # function to return relu(z)
    # input values less than 0: outputs 0
    # input values greater than 0: outputs the same value
    def reLu(self,z):       
        return np.maximum(0,z)
    
 
    # function to calculate cross entropy loss given the true y and predicted y values
    def cross_entropy_loss(self,y,yhat):
        n = len(y)
        loss = -1/n * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((1 - y), np.log(1 - yhat))))
        
        return loss
    
    
    # function to perform forward propogation
    def forward_propagation(self):
        
        # compute the weighted sum between input and first layer's weights and add the bias
        # z1= x.w1 + b1
        z1 = self.x.dot(self.parameters['w1']) + self.parameters['b1']  

        # pass the result through ReLu activation function
        # a1=relu(z1)
        a1 = self.reLu(z1)

        # compute the weighted sum between a1 and second layer's weights and add bias term
        # z2=a1.w2 + b2 
        z2 = a1.dot(self.parameters['w2']) + self.parameters['b2']

        # computing the predicted result
        # yhat = sigmoid(z2)
        yhat = self.sigmoid(z2)

        # computing the loss between predicted output and the true labels
        loss = self.cross_entropy_loss(self.y,yhat)

        # save the parameters     
        self.parameters['z1'] = z1
        self.parameters['z2'] = z2
        self.parameters['a1'] = a1

        return yhat,loss
 
 

    # function that returns derivative of n=ReLu(x)
    # derivative of ReLu is 1 if n>0 and 0 if n<=0   
    def reLu_derivative(self,n):
            n[n<=0]=0
            n[n>0]=1
            return n

    

    # function to perform back propogation
    def back_propagation(self,yhat):
        
        # compute the gradients
        deriv_yhat = -(np.divide(self.y,yhat) - np.divide((1 - self.y),(1-yhat)))
        
        deriv_sigmoid = yhat * (1-yhat)
        
        deriv_z2 = deriv_yhat * deriv_sigmoid

        deriv_a1 = deriv_z2.dot(self.parameters['w2'].T)
        deriv_w2 = self.parameters['a1'].T.dot(deriv_z2)
        deriv_b2 = np.sum(deriv_z2, axis=0)

        deriv_z1 = deriv_a1 * self.reLu_derivative(self.parameters['z1'])
        deriv_w1 = self.x.T.dot(deriv_z1)
        deriv_b1 = np.sum(deriv_z1, axis=0)

        # updating the weights and bias
        self.parameters['w1'] = self.parameters['w1'] - self.learning_rate * deriv_w1
        self.parameters['w2'] = self.parameters['w2'] - self.learning_rate * deriv_w2
        self.parameters['b1'] = self.parameters['b1'] - self.learning_rate * deriv_b1
        self.parameters['b2'] = self.parameters['b2'] - self.learning_rate * deriv_b2



     # function to train the neural network   
    def fit(self, X, Y):
        self.x = X
        self.y = Y
        self.initialize_weights() 
        
        # for every iteration, perform forward and backward propogation
        for i in range(self.iterations):
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.loss.append(loss)


    # function to test the model
    # to predict on test data
    # performs a forward pass on test data ( use saved weights and biases from training phase )
    def predict(self, X):
        z1 = X.dot(self.parameters['w1']) + self.parameters['b1']
        a1 = self.reLu(z1)
        z2 = a1.dot(self.parameters['w2']) + self.parameters['b2']
        prediction = self.sigmoid(z2)
        
        return np.round(prediction)  
       
    
    # function to find the Confusion Matrix
    # Prints Confusion Matrix, Precision, Recall, F1 Score, Accuracy
    def CM(self,y_test,y_test_obs):
        for i in range(len(y_test_obs)):
            if(y_test_obs[i]>0.6):
                y_test_obs[i]=1
            else:
                y_test_obs[i]=0

        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0

        for i in range(len(y_test)):
            if(y_test[i]==1 and y_test_obs[i]==1):
                tp=tp+1
            if(y_test[i]==0 and y_test_obs[i]==0):
                tn=tn+1
            if(y_test[i]==1 and y_test_obs[i]==0):
                fp=fp+1
            if(y_test[i]==0 and y_test_obs[i]==1):
                fn=fn+1
        cm[0][0]=tn
        cm[0][1]=fp
        cm[1][0]=fn
        cm[1][1]=tp

        p= tp/(tp+fp)
        r=tp/(tp+fn)
        f1=(2*p*r)/(p+r)
        a=(tp+tn)/(tp+tn+fp+fn)

        p=round(p,3)
        r=round(r,3)
        f1=round(f1,3)
        a=round(a,5)*100
        a="{:.3f}".format(a)

        print("Confusion Matrix : ")
        print(cm,"\n")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")  
        print("Accuracy: ",a,"%") 



# main function


if __name__=='__main__':
    
    # load the cleaned dataset as a dataframe
    dataset_path=r'LBW_Dataset_cleaned.csv'
    df = pd.read_csv(dataset_path)

    # x_df is the dataframe with only features which will be input to the model
    # drop the 'Result' column
    x_df = df.drop(columns=['Result'])

    # obtaining the true y labels
    y_labels = df['Result'].values.reshape(x_df.shape[0], 1)


    #create train-test split ... 70:30 ratio
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_labels, test_size=0.3, random_state=1)

    
    # convert the x_train and x_test dataframes to n-dimensional arrays
    x_train=x_train.to_numpy()
    x_test=x_test.to_numpy()

    # create a NN object
    nn = NN()

    # initialize the model's weights
    nn.initialize_weights()

    # train the model
    nn.fit(x_train, y_train) 


    train_pred = nn.predict(x_train)
    test_pred = nn.predict(x_test)

    
    # printing train results:
    print("\nTRAIN RESULTS\n")
    nn.CM(y_train,train_pred)
    

    print("\n--------------------------------------------\n")

    # printing test accuracy
    print("TEST RESULTS\n")
    nn.CM(y_test,test_pred)
    





