import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = 'G:\\College\\Fourth year\\First term\\Genetic algorithms\\Assignments\\Assignment 4\\data.txt'
data = pd.read_csv(path,delim_whitespace=(True),nrows=(2),header=None)

#read the parameters of the problem
M = int(data[0][0])
L = int(data[1][0])
N = int(data[2][0])
K = int(data[0][1])

learningRate = 0.01

# read the training examples and separate them to input (X) and output (Y)
examples = pd.read_csv(path,delim_whitespace=(True),skiprows=[0,1],header=None)
X = examples.iloc[:,:M]
Y = examples.iloc[:,M:M+N+1]


# initialize the parameters.
Wh = np.random.randn(L,M)   # weights between input  and hidden layer.
Wo = np.random.randn(N,L)   # weights between hidden and output layer.

Ah = np.random.randn(1,L)   # activation at hidden layer.
Ao = np.random.randn(1,N)   # activation at output layer.

Do = np.random.randn(1,N)   # Delta-output
Dh = np.random.randn(1,L)   # Delta-hidden

ErE = np.zeros(K)           # Errors for training examples.
ErI = []                    # Errors for every iteration (epochs)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def Normalization(): #Utility function to normalize the training data
    global X , Y
    Xmodi = pd.DataFrame(X)
    Ymodi = pd.DataFrame(Y)
    
    df = pd.DataFrame(Xmodi.mean())
    df.to_csv('meansforIn.csv', index=False)
    
    df = pd.DataFrame(Xmodi.std())
    df.to_csv('stdforIn.csv', index=False)
    
    df = pd.DataFrame(Ymodi.mean())
    df.to_csv('meansforOut.csv', index=False)
    
    df = pd.DataFrame(Ymodi.std())
    df.to_csv('stdforOut.csv', index=False)
    

    Xmodi = ((Xmodi - Xmodi.mean())/ Xmodi.std())
    Ymodi = ((Ymodi - Ymodi.mean())/ Ymodi.std())
    Y = Ymodi
    X = Xmodi
    
    
def train():
    global X, Y
    
    Normalization()
    
    
    for itr in range(500):     #of epochs
    
        ErN = np.zeros(N)   # Errors for every output neuron. 
    
        for l in range(K):      #Loop over training examples:
        
            #begin feedforward.
            for j in range(L):  #For each hidden layer neuron j:
                sum = 0
                for i in range(M):
                    sum += X[i][l] * Wh[j][i]
                Ah[0][j] = sigmoid(sum)
            
            for k in range(N):  #For each output layer neuron k:
                sum = 0
                for i in range(L):
                    sum += Ah[0][i] * Wo[k][i]
                Ao[0][k] = sigmoid(sum)
                ErN[k] = pow(Y.values[l][k] - Ao[0][k], 2)
                
            #begin backpropagation.
            for k in range(N):  #For each output layer neuron k:
                Do[0][k] = (Ao[0][k] - Y.values[l][k]) * Ao[0][k] * (1 - Ao[0][k])
        
            for j in range(L):  #For each hidden layer neuron j:
                sum = 0
                for k in range(N):
                    sum += Do[0][k] * Wo[k][j]
                Dh[0][j] = sum * Ah[0][j] * (1 - Ah[0][j])
        
            for j in range(L):  #For each weight 𝒘𝒌𝒋𝒐 going to the output layer:
                for k in range(N):
                    Wo[k][j] = Wo[k][j] - learningRate * Do[0][k] * Ah[0][j]
        
            for i in range(M):  #For each weight 𝒘𝒋𝒊𝒉 going to the hidden layer:
                for j in range(L):
                    Wh[j][i] = Wh[j][i] - learningRate * Dh[0][j] * X[i][l]
                
            Sum = np.sum(ErN)
            ErE[l] = Sum / 2
        ErI.append(ErE.mean())
        
def prog1():
    
    train()
    #Write the weights to a CSV file.
    
    df = pd.DataFrame(Wh)
    df.to_csv('hiddenWeight.csv', index=False)
    
    df = pd.DataFrame(Wo)
    df.to_csv('outputWeight.csv', index=False)
            
def prog2():
    
    #Read the resultant weights from the training process.
    Wh = np.array(pd.read_csv("hiddenWeight.csv").values.tolist())
    Wo = np.array(pd.read_csv("outputWeight.csv").values.tolist())
    
    #Read the statistics info to use it in the normalization step.
    meansIn = np.array(pd.read_csv("meansforIn.csv").stack().tolist())
    stdIn   = np.array(pd.read_csv("stdforIn.csv").stack().tolist())
    meansOut = np.array(pd.read_csv("meansforOut.csv").stack().tolist())
    stdOut  = np.array(pd.read_csv("stdforOut.csv").stack().tolist())
    
    #Read the file that contains the testing data.
    data = pd.read_csv('test.txt',delim_whitespace=(True),nrows=(2),header=None)

    #Read the parameters of the problem
    M = int(data[0][0])
    L = int(data[1][0])
    N = int(data[2][0])
    K = int(data[0][1])

    #Read the input (X) and output (Y)
    examples = pd.read_csv(path,delim_whitespace=(True),skiprows=[0,1],header=None)
    Xtest = examples.iloc[:,:M]
    Ytest = examples.iloc[:,M:M+N+1]
    
    #Initialize the parameters.
    Ahtest = np.random.randn(1,L)   # activation at hidden layer.
    Aotest = np.random.randn(1,N)   # activation at output layer.
    ErNtest = np.zeros(N)           # Error for each neuron.
    ErEtest = np.zeros(K)           # Error for each training example.

    
    #Normalization
    Xmodi = pd.DataFrame(Xtest)
    Ymodi = pd.DataFrame(Ytest)
    Xmodi = ((Xmodi - meansIn)/ stdIn)
    Ymodi = ((Ymodi - meansOut)/ stdOut)
    Ytest = Ymodi
    Xtest = Xmodi
    
    #Start to test the data.
    for l in range(K):          #Loop over training examples:
        
            #begin feedforward.
            for j in range(L):  #For each hidden layer neuron j:
                sum = 0
                for i in range(M):
                    sum += Xtest[i][l] * Wh[j][i]
                Ahtest[0][j] = sigmoid(sum)
            
            for k in range(N):  #For each output layer neuron k:
                sum = 0
                for i in range(L):
                    sum += Ahtest[0][i] * Wo[k][i]
                Aotest[0][k] = sigmoid(sum)
                ErNtest[k] = pow(Ytest.values[l][k] - Aotest[0][k], 2)
            
            Sum = np.sum(ErNtest)
            ErEtest[l] = Sum / 2
            
    print("Cost for testing process: " , ErEtest.mean())
    
                
prog1()
prog2()

plt.title("Cost at training")   
plt.plot(range(0, 500), ErI)
plt.xlabel('Iteration #') 
plt.ylabel('MSE') 
plt.grid(True)   
plt.show()




        
            
        

            

            
        
            
            
    

        
