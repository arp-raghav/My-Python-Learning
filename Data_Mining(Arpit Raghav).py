#Assignment 1 - Data Mining and Visualization
#Name - Arpit Raghav
#Student ID - 201584085

import numpy as np

np.random.seed(1)


	
class Perceptron(object):
    """
    This class contains utilities for training and testing the perceptron model.
1. The function trn(slf,data,tag) is used to train the perceptron on the provided dataset.
2. activation count(slf,X) provides the activation count value by doing a dot product of the weight and the feature vector X.

3. predict(slf,X) gives the predicted value of the data for a feature vector in X, which is either 1 or -1.

4. Reg Train(slf, data, tag, lambd) is used to train and evaluate the L2 regularised perceptron model.

5.Activ count L2(slf, X) returns the activation count value by doing a dot product of the weight and the feature vector X.

6.predictWithRegularisation(slf, X) gives the predicted value of the data for a feature vector in X, which is either 1 or -1. This class contains utility functions for training and testing the perceptron model.
    """
#Initialiser Function
    def __init__(slf, learning, epoch):
     
        slf.learning = learning
        slf.epoch = epoch
        slf.tag = ""
        slf.lambd = 0

   
#Activation Function
    def activation_count(slf, X):
      
        X = X.astype(float)
        return np.dot(X, slf.wt) + slf.b  # calculating activation count value
#Prediction function
    def predict(slf, X):
      
        return np.where(slf.activation_count(X) >= 0.0, 1, -1)  # Predicting based on threshold
#Predict function with regularisation    
    def predictWithRegularisation(slf, X):
      
        return np.where(slf.Activ_count_L2(X) >= 0.0, 1, -1)  # Predicting based on threshold
#Training function    
    def trn(slf, data, tag):
      
        slf.tag = tag
        data[:, 4] = np.where(data[:, 4] == slf.tag, 1, -1)
        np.random.shuffle(data)
        X = data[:, 0:4].astype(float)
        y = data[:, 4].astype(float)
        # Initialising wt vector to be zero
        slf.wt = np.zeros(X.shape[1])
        # Initialising bias to be zero
        slf.b = 0  
        for i in range(slf.epoch):
            for xi, target in zip(X, y):
                d = slf.learning * (target - slf.predict(xi))  
                slf.wt = slf.wt + (d * xi)  
                slf.b = slf.b + d  
        return slf
#Training function with regularisation
    def Reg_Train(slf, data, tag, lambd):
        
        slf.tag = tag
        slf.lambd = lambd
        data[:, 4] = np.where(data[:, 4] == slf.tag, 1, -1)
        np.random.shuffle(data)
        X = data[:, 0:4].astype(np.float64)
        y = data[:, 4].astype(np.float64)
        slf.wt = np.zeros(X.shape[1], dtype=np.float64)  
        slf.b = 0  
        for i in range(slf.epoch):
            for xi, target in zip(X, y):
                d = slf.learning * (target - slf.predictWithRegularisation(xi))  
                slf.wt = (1 - (2 * slf.lambd)) * slf.wt + (d * xi)  
                slf.b = slf.b + d   
        return slf
#Activation function with regularization
    def Activ_count_L2(slf, X):
       
        X = X.astype(np.longdouble)
        return np.dot(X, slf.wt) + slf.b   

#Accuracy function
def accuracy(actual, predicted):
 
    correct = 0
    for i in range(len(actual)):
        if int(actual[i]) == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
#Function to read and load the dataset
def readfile(filename):
 
    arr = []
    with open(filename) as file:
        for line in file:
            arr.append(line.rstrip().split(","))
        arr = np.array(arr)
        l1 = arr[arr[:, 4] == "class-1"]
        l2 = arr[arr[:, 4] == "class-2"]
        l3 = arr[arr[:, 4] == "class-3"]
    return l1, l2, l3


#Reading the train and test dataset

Train_1, Train_2, Train_3 = readfile("C:\\Users\\arpit\\Documents\\Data Mining\\CA1data\\train.data")
Test_1, Test_2, Test_3= readfile("C:\\Users\\arpit\\Documents\\Data Mining\\CA1data\\test.data")



# a) Class 1 vs Class 2
Class1vClass2 = Perceptron(0.01, 20)
Train_1v2 = np.concatenate((Train_1, Train_2), axis=0)
Class1vClass2.trn(Train_1v2, "class-1")

# b) Class 2 vs Class 3
Class2vClass3 = Perceptron(0.01, 20)
Train_2v3 = np.concatenate((Train_2, Train_3), axis=0)
Class2vClass3.trn(Train_2v3, "class-2")

# c) Class 1 vs Class 3
Class1vClass3 = Perceptron(0.01, 20)
Train_1v3 = np.concatenate((Train_1, Train_3), axis=0)
Class1vClass3.trn(Train_1v3, "class-1")

# Training Accuracy (1 v 2)
accTrain_1v2 = []
Train_1v2PredictedY = Class1vClass2.predict(Train_1v2[:, 0:4])
accTrain_1v2.append(accuracy(Train_1v2[:, 4], Train_1v2PredictedY))

# Calculating Training Accuracy(2 v 3)
accTrain_2v3 = []
Train_2v3PredictedY = Class2vClass3.predict(Train_2v3[:, 0:4])
accTrain_2v3.append(accuracy(Train_2v3[:, 4], Train_2v3PredictedY))

# Calculating Training Accuracy(1 v 3)
accTrain_1v3 = []
Train_1v3PredictedY = Class1vClass3.predict(Train_1v3[:, 0:4])
accTrain_1v3.append(accuracy(Train_1v3[:, 4], Train_1v3PredictedY))

# Calculating Testing Accuracy (1 v 2)
accTest_1v2 = []
Test_1v2 = np.concatenate((Test_1, Test_2), axis=0)
Test_1v2[:, 4] = np.where(Test_1v2[:, 4] == "class-1", 1, -1)
Test_1v2PredictedY = Class1vClass2.predict(Test_1v2[:, 0:4])
accTest_1v2.append(accuracy(Test_1v2[:, 4], Test_1v2PredictedY))

# Calculating Testing Accuracy (2 v 3)
accTest_2v3 = []
Test_2v3 = np.concatenate((Test_2, Test_3), axis=0)
Test_2v3[:, 4] = np.where(Test_2v3[:, 4] == "class-2", 1, -1)
Test_2v3PredictedY = Class2vClass3.predict(Test_2v3[:, 0:4])
accTest_2v3.append(accuracy(Test_2v3[:, 4], Test_2v3PredictedY))

# Calculating Testing Accuracy (1 v 3)
accTest_1v3 = []
Test_1v3 = np.concatenate((Test_1, Test_3), axis=0)
Test_1v3[:, 4] = np.where(Test_1v3[:, 4] == "class-1", 1, -1)
Test_1v3PredictedY = Class1vClass3.predict(Test_1v3[:, 0:4])
accTest_1v3.append(accuracy(Test_1v3[:, 4], Test_1v3PredictedY))




"""
Q4. Extend the binary perceptron that you implemented in part (2) above to perform multi-class classification using the 
1-vs-rest approach. 
"""


# Class 1 vs (Class 2, Class 3)
pc1 = Perceptron(0.01, 20)
Train_1v23 = np.concatenate((Train_1, Train_2,Train_3), axis=0)

pc1.trn(Train_1v23, "class-1")

# Class 2 vs (Class 1, Class 3)
pc2 = Perceptron(0.01, 20)
Train_2v13 = np.concatenate((Train_1, Train_2,Train_3), axis=0)

pc2.trn(Train_1v23, "class-2")

# Class 3 vs (Class 1, Class 2)
pc3 = Perceptron(0.01, 20)
Train_2v12 = np.concatenate((Train_1, Train_2,Train_3), axis=0)

pc3.trn(Train_2v12, "class-3")

# Calculating Testing accuracy
Test_1[:, 4] = np.where(Test_1[:, 4] == "class-1", 1, -1)
Test_2[:, 4] = np.where(Test_2[:, 4] == "class-2", 2, -1)
Test_3[:, 4] = np.where(Test_3[:, 4] == "class-3", 3, -1)
class1v2v3Testing = np.concatenate((Test_1, Test_2,Test_3), axis=0)

TP = 0
classAccuracyTesting = np.zeros(3)
for data in class1v2v3Testing:
    count = []
    count.append(pc1.activation_count(data[:4]))
    count.append(pc2.activation_count(data[:4]))
    count.append(pc3.activation_count(data[:4]))
    prediction = np.argmax(count) + 1
    if prediction == int(data[4]):
        classAccuracyTesting[int(data[4]) - 1] += 1
        TP += 1

# Calculating Training accuracy
Train_1[:, 4] = np.where(Train_1[:, 4] == "class-1", 1, -1)
Train_2[:, 4] = np.where(Train_2[:, 4] == "class-2", 2, -1)
Train_3[:, 4] = np.where(Train_3[:, 4] == "class-3", 3, -1)
class1v2v3Training = np.concatenate((Train_1, Train_2,Train_3), axis=0)
TP = 0
Accuracy_Train = np.zeros(3)
for data in class1v2v3Training:
    count = []
    count.append(pc1.activation_count(data[:4]))
    count.append(pc2.activation_count(data[:4]))
    count.append(pc3.activation_count(data[:4]))
    prediction = np.argmax(count) + 1
    if prediction == int(data[4]):
        Accuracy_Train[int(data[4]) - 1] += 1
        TP += 1
train1 = ((Accuracy_Train / (class1v2v3Training.shape[0] / 3))[0]) * 100
train2 = ((Accuracy_Train / (class1v2v3Training.shape[0] / 3))[1]) * 100
train3 = ((Accuracy_Train / (class1v2v3Training.shape[0] / 3))[2]) * 100




test1 = ((classAccuracyTesting / (class1v2v3Testing.shape[0] / 3))[0]) * 100
test2 = ((classAccuracyTesting / (class1v2v3Testing.shape[0] / 3))[1]) * 100
test3 = ((classAccuracyTesting / (class1v2v3Testing.shape[0] / 3))[2]) * 100
print(" Q3. Use the binary perceptron to trn classifiers to discriminate between the following pairs: \n\n")

print("Class 1 vs Class 2 \n")
print("Training  -" , sum(accTrain_1v2) / len(accTrain_1v2))
print("Testing   -", sum(accTest_1v2) / len(accTest_1v2) , "\n")

print("Class 2 vs Class 3 \n")
print("Training  -" , sum(accTrain_2v3) / len(accTrain_2v3))
print("Testing   -", sum(accTest_2v3) / len(accTest_2v3),"\n")

print("Class 1 vs Class 3 \n")
print("Training  -" , sum(accTrain_1v3) / len(accTrain_1v3))
print("Testing   -", sum(accTest_1v3) / len(accTest_1v3),"\n")

print(" Q4. Extend the binary perceptron that you implemented in part (2) above to perform multi-class classification using the 1-vs-rest approach. \n")

print("Class 1 vs Rest \n")
print("Trainging -" , train1)
print("Testing   -", test1 ,"\n")

print("Class 2 vs Rest \n")
print("Trainging -" ,train2)
print("Testing   -", test2 ,"\n")

print("Class 3 vs Rest \n")
print("Trainging -" , train3)
print("Testing   -", test3,"\n\n")



np.seterr(all='ignore')  

"""
Q5. Add an l2 regularisation term to your multi-class classifier implemented in question. Set the regularisation
 coefficient to 0.01, 0.1, 1.0, 10.0, 100.0 and compare the trn and test classification accuracy for each of the
 three classes.
"""
print("\nQ5. Multi- Classification with L2 regularisation \n")
     

"""Lambda 0.01"""
# Class 1 vs (Class 2, Class 3)
pcl1 = Perceptron(0.01, 20)
pcl1.Reg_Train(Train_1v23, "class-1", 0.01)

# Calculating Testing accuracy
TP = 0
Lambd_Test= np.zeros(3)
for data in class1v2v3Testing:
    count = []
    count.append(pcl1.Activ_count_L2(data[:4]))
    
    prediction = np.argmax(count) + 1
    if prediction == int(data[4]):
        Lambd_Test[int(data[4]) - 1] += 1
        TP += 1

# Calculating Training accuracy
Lambd_train = np.zeros(3)
for data in class1v2v3Training:
    count = []
    count.append(pcl1.Activ_count_L2(data[:4]))
    
    prediction = np.argmax(count) + 1
    if prediction == int(data[4]):
        Lambd_train[int(data[4]) - 1] += 1
        TP += 1

trl1 = ((Lambd_train / (class1v2v3Training.shape[0] / 3))[0]) * 100


tsl1 = ((Lambd_Test/ (class1v2v3Testing.shape[0] / 3))[0]) * 100


print("Class 1 vs rest")

print("Lambda = 0.01   ,Training -  " , trl1 , ", Testing - " , tsl1)

"""Lambda 0.1"""
# Class 1 vs (Class 2, Class 3)
pcl4 = Perceptron(0.01, 20)
pcl4.Reg_Train(Train_1v23, "class-1", 0.1)


# Calculating Testing accuracy
TP = 0
Lambd_Test= np.zeros(3)
for data in class1v2v3Testing:
    count = []
    count.append(pcl4.Activ_count_L2(data[:4]))
    
    prediction = np.argmax(count) + 1
    if prediction == int(data[4]):
        Lambd_Test[int(data[4]) - 1] += 1
        TP += 1

# Calculating Training accuracy
Lambd_train = np.zeros(3)

for data in class1v2v3Training:
    count = []
    count.append(pcl4.Activ_count_L2(data[:4]))
    
    prediction = np.argmax(count) + 1
    if prediction == int(data[4]):
        Lambd_train[int(data[4]) - 1] += 1
        TP += 1

trl4 = ((Lambd_train / (class1v2v3Training.shape[0] / 3))[0]) * 100


tsl4 = ((Lambd_Test/ (class1v2v3Testing.shape[0] / 3))[0]) * 100



print("Lambda = 0.1    ,Training -  " , trl4 , ", Testing - " , tsl4)
"""Lambda 1"""
# Class 1 vs (Class 2, Class 3)
pcl7 = Perceptron(0.01, 20)
pcl7.Reg_Train(Train_1v23, "class-1", 1)

# Calculating Testing accuracy
TP = 0
Lambd_Test= np.zeros(3)
for data in class1v2v3Testing:
    count = []
    count.append(pcl7.Activ_count_L2(data[:4]))
   
    prediction = np.argmax(count) + 1
    if prediction == int(data[4]):
        Lambd_Test[int(data[4]) - 1] += 1
        TP += 1

# Calculating Training accuracy
Lambd_train = np.zeros(3)
for data in class1v2v3Training:
    count = []
    count.append(pcl7.Activ_count_L2(data[:4]))
    
    prediction = np.argmax(count) + 1
    if prediction == int(data[4]):
        Lambd_train[int(data[4]) - 1] += 1
        TP += 1

trl7 = ((Lambd_train / (class1v2v3Training.shape[0] / 3))[0]) * 100


tsl7 = ((Lambd_Test/ (class1v2v3Testing.shape[0] / 3))[0]) * 100


print("Lambda = 1      ,Training -  " , trl7 , ", Testing - " , tsl7)
"""Lambda 10"""
# Class 1 vs (Class 2, Class 3)
pcl10 = Perceptron(0.01, 20)
pcl10.Reg_Train(Train_1v23, "class-1", 10)



# Calculating Testing accuracy
TP = 0
Lambd_Test= np.zeros(3)
for data in class1v2v3Testing:
    count = []
    count.append(pcl10.Activ_count_L2(data[:4]))
    
    prediction = np.argmax(count) + 1
    if prediction == int(data[4]):
        Lambd_Test[int(data[4]) - 1] += 1
        TP += 1

# Calculating Training accuracy
Lambd_train = np.zeros(3)
for data in class1v2v3Training:
    count = []
    count.append(pcl10.Activ_count_L2(data[:4]))
    
    prediction = np.argmax(count) + 1
    if prediction == int(data[4]):
        Lambd_train[int(data[4]) - 1] += 1
        TP += 1

trl10 = ((Lambd_train / (class1v2v3Training.shape[0] / 3))[0]) * 100


tsl10 = ((Lambd_Test/ (class1v2v3Testing.shape[0] / 3))[0]) * 100


print("Lambda = 10     ,Training -  " , trl10 , ", Testing - " , tsl10)

"""Lambda 100"""
# Class 1 vs (Class 2, Class 3)
pcl13 = Perceptron(0.01, 20)
pcl13.Reg_Train(Train_1v23, "class-1", 100)


# Calculating Testing accuracy
TP = 0
Lambd_Test= np.zeros(3)
for data in class1v2v3Testing:
    count = []
    count.append(pcl13.Activ_count_L2(data[:4]))
    
    prediction = np.argmax(count) + 1
    if prediction == int(data[4]):
        Lambd_Test[int(data[4]) - 1] += 1
        TP += 1

# Calculating Training accuracy
Lambd_train = np.zeros(3)
for data in class1v2v3Training:
    count = []
    count.append(pcl13.Activ_count_L2(data[:4]))
    
    prediction = np.argmax(count) + 1
    if prediction == int(data[4]):
        Lambd_train[int(data[4]) - 1] += 1
        TP += 1

trl13 = ((Lambd_train / (class1v2v3Training.shape[0] / 3))[0]) * 100


tsl13 = ((Lambd_Test/ (class1v2v3Testing.shape[0] / 3))[0]) * 100


print("Lambda = 100    ,Training -  " , trl13 , ", Testing - " , tsl13)
