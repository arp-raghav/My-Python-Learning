
# Importing the relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

#Redaing the data

path = 'C:\\Users\\arpit\\Documents\\Data Mining\\A2\\'

animals = pd.read_csv(path + "animals", sep = " ", header = None)
countries = pd.read_csv(path + "countries", sep = " ", header = None)
fruits = pd.read_csv(path + "fruits", sep = " ", header = None)
veggies = pd.read_csv(path + "veggies", sep = " ", header = None)

# Adding the cluster category to  a new column

animals['Category'] = 'animals'
countries['Category'] = 'countries'
fruits['Category'] = 'fruits'
veggies['Category'] = 'veggies'


# Joining all the  data together to a sing data frame 

data = pd.concat([animals, countries, fruits, veggies], ignore_index = True)

# Changing all class labels to numbers starting from 0

labels = (pd.factorize(data.Category)[0]+1) - 1 # 0=animals, 1=countries, 2=fruits, 3=veggies
dataset = data.drop([0, 'Category'], axis = 1).values

# Saveing the maximum index for each category for the P/R/F

maxAni = data.index[data['Category'] == 'animals'][-1]
maxCount = data.index[data['Category'] == 'countries'][-1]
maxFruit = data.index[data['Category'] == 'fruits'][-1]
maxVeg = data.index[data['Category'] == 'veggies'][-1]

#Printing the dataset
print("The combined Data\n")
print(data)

#Defining the class

print("Q1 - Implement the k-means clustering algorithm to cluster the instances into k clusters. \n ")
def kMeans(x,k):
    np.random.seed(1)
    """
    Initialisation phase
    Randomly initialise the first centroids
    """
    centroids = []
    temp = np.random.randint(x.shape[0], size = k)
    while (len(temp) > len(set(temp))):
        temp = np.random.randint(x.shape[0], size = k)
    for i in temp:
        centroids.append(x[i])
    centroids_old = np.zeros(np.shape(centroids))
    centroids_new = centroids
    
    
    
    # Creating an error object
    error = np.linalg.norm(centroids_new - centroids_old)
    num_errors = 0

    # Creating a blank distance and cluster assignment object to hold results
    clusters = np.zeros(x.shape[0])
    
    # If there is an error value:
    while error != 0:
        dist = np.zeros([x.shape[0], k])
        # Adding one to the number of errors
        num_errors += 1
        # Calculating the Euclidean distance from each point to each centroid    
        for j in range(len(centroids)):
            dist[:, j] = np.linalg.norm(x - centroids_new[j], axis=1)
       
        # Calculating the cluster assignment
        clusters = np.argmin(dist, axis = 1)
        centroids_old = centroids_new
        
        
        #Optimizing phase in the I am computing new representatives as the means of current clusters

       
        # Calculating the mean to re-adjust the cluster centroids
        for m in range(k):
            centroids_new[m] = np.mean(x[clusters == m], axis = 0)
            
        # Re-calculating the error
        error = np.linalg.norm(np.array(centroids_new) - np.array(centroids_old))

    #Assigning  the final clusters and centroids to new objects
        
    predicted_clusters = clusters
    final_centroids = np.array(centroids_new)
    #print ("Points Belonging to Final Clusters:\n", predicted_clusters)
    #print ("Final Centroid Locations:\n", final_centroids)
    return predicted_clusters


predCl = kMeans(dataset,4)
animal_index = []
country_index = []
veggies_index = []
fruits_index = []
for i in range(len(predCl)):
    if predCl[i] == 0:
        animal_index.append(i)
    elif predCl[i] == 1:
        country_index.append(i)
    elif predCl[i] == 2:
        fruits_index.append(i)
    else:
        veggies_index.append(i)

CountOfAnimals = 0
CountOfCountries = 0
CountOfFruits = 0
CountofVeggies = 0
for ind in animal_index:
    if ind in data[data['Category']=='animals'].index:
        CountOfAnimals += 1
    elif ind in data[data['Category']=='countries'].index:
        CountOfCountries += 1
    elif ind in data[data['Category']=='fruits'].index:
        CountofFruits += 1
    else:
        CountofVeggies += 1

CountTrueanimals = 0
CountTruecountries = 0
CountTruefruits = 0
CountTrueVeggies = 0

p1 = (((CountTrueanimals * CountTrueanimals)/len(animal_index)) + ((CountTruecountries * CountTruecountries)/len(animal_index)) + ((CountTruefruits * CountTruefruits)/len(animal_index)) + ((CountTrueVeggies * CountTrueVeggies)/len(animal_index))) 

r1 = CountTrueanimals * CountTrueanimals/len(animal_index)



for ind in country_index:
    if ind in data[data['Category']=='animals'].index:
        CountTrueanimals += 1
    elif ind in data[data['Category']=='countries'].index:
        CountTruecountries += 1
    elif ind in data[data['Category']=='fruits'].index:
        CountTruefruits += 1
    else:
        CountTrueVeggies += 1
        
p2 = (((CountTrueanimals * CountTrueanimals)/len(country_index)) + 
      ((CountTruecountries * CountTruecountries)/len(country_index)) +
     ((CountTruefruits * CountTruefruits)/len(country_index)) +
       ((CountTrueVeggies * CountTrueVeggies)/len(country_index))) 

        
CountTrueanimals = 0
CountTruecountries = 0
CountTruefruits = 0
CountTrueVeggies = 0

for ind in fruits_index:
    if ind in data[data['Category']=='animals'].index:
        CountTrueanimals += 1
    elif ind in data[data['Category']=='countries'].index:
        CountTruecountries += 1
    elif ind in data[data['Category']=='fruits'].index:
        CountTruefruits += 1
    else:
        CountTrueVegies += 1
        
p3 = (((CountTrueanimals * CountTrueanimals)/len(fruits_index)) + 
      ((CountTruecountries * CountTruecountries)/len(fruits_index)) +
     ((CountTruefruits * CountTruefruits)/len(fruits_index)) +
       ((CountTrueVeggies * CountTrueVeggies)/len(fruits_index))) 
        
CountTrueanimals = 0
CountTruecountries = 0
CountTruefruits = 0
CountTrueVeggies = 0

for ind in veggies_index:
    if ind in data[data['Category']=='animals'].index:
        CountTrueanimals += 1
    elif ind in data[data['Category']=='countries'].index:
        CountTruecountries += 1
    elif ind in data[data['Category']=='fruits'].index:
        CountTruefruits += 1
    else:
        CountTrueVeggies += 1
        
p4 = (((CountTrueanimals * CountTrueanimals)/len(veggies_index)) + 
      ((CountTruecountries * CountTruecountries)/len(veggies_index)) +
     ((CountTruefruits * CountTruefruits)/len(veggies_index)) +
       ((CountTrueVeggies * CountTrueVeggies)/len(veggies_index))) 

print("p1: ", p1)
print("p2: ", p2)
print("p3: ", p3)
print("p4: ", p4)

P = round((p1+p2+p3+p4)/len(predCl),2)
print("Precision: ", P)

#L2 Normalisation
np.random.seed(35)

def kMeansNormalisation(x,k,norm):
   
    if norm==True:
        x = x / np.linalg.norm(x)
        print(x)
    
    
    #This is the Initialisation phase in which I am randomly initialise the first centroids
    
    centroids = []
    temp = np.random.randint(x.shape[0], size = k)
    while (len(temp) > len(set(temp))):
        temp = np.random.randint(x.shape[0], size = k)
    for i in temp:
        centroids.append(x[i])
    centroids_old = np.zeros(np.shape(centroids))
    centroids_new = centroids
    
    # Creating a blank distance and cluster assignment object to hold results
    clusters = np.zeros(x.shape[0])
    
    # Creating an error object
    error = np.linalg.norm(centroids_new - centroids_old)
    num_errors = 0
    
    # If there is an error value:
    while error != 0:
        dist = np.zeros([x.shape[0], k])
        # Adding one to the number of errors
        num_errors += 1
        # Calculating the Euclidean distance from each point to each centroid    
        for j in range(len(centroids)):
            dist[:, j] = np.linalg.norm(x - centroids_new[j], axis=1)
        
        
        #This is the Assignment phase in which I am assigning objects in the dataset to their closest representatives
       
        # Calculating the cluster assignment
        clusters = np.argmin(dist, axis = 1)
        centroids_old = centroids_new
        
        
        #This is the optimization phase in which I am computing new representatives as the means of current clusters
        
        # Calculating the mean to re-adjust the cluster centroids
        for m in range(k):
            centroids_new[m] = np.mean(x[clusters == m], axis = 0)
        # Re-calculate the error
        error = np.linalg.norm(np.array(centroids_new) - np.array(centroids_old))

    #Assigning the final clusters and centroids to new objects
    predicted_clusters = clusters
    final_centroids = np.array(centroids_new)
    
    # Creating objects of the index positioning of the different classes
    animal_pos = predicted_clusters[:maxAni+1]
    countries_pos = predicted_clusters[maxAni+1:maxCount+1]
    fruit_pos = predicted_clusters[maxCount+1:maxFruit+1]
    veggies_pos = predicted_clusters[maxFruit+1:maxVeg+1]

    #Assigning initial values to  True Positives
    TP1 = 0
    TP2 = 0
    TP3 = 0
    TP4 = 0
    #Assigning initial values to  False Negatives
    FN1 = 0
    FN2 = 0
    FN3 = 0
    FN4 = 0
    #Assigning initial values to  True Negatives
    TN1 = 0
    TN2 = 0
    TN3 = 0
    TN4 = 0
    #Assigning initial values to  False Positives
    FP1 = 0
    FP2 = 0
    FP3 = 0
    FP4 = 0
    
   

    for i in range(len(countries_pos)):
    # For every row in countries_pos
        for j in range(len(countries_pos)):
            # If i and j are not the same, and j > i
            if (i != j & j>i):
            # If i is equal to j then add 1 to TP
                if(countries_pos[i] == countries_pos[j]):
                    TP2 += 1
                # Otherwise adding 1 to FN
                else:
                    FN2 += 1
                #For every row in animal_pos                
        for j in range(len(animal_pos)):
                    # If i is equal to j then add 1 to FP
                if(countries_pos[i] == animal_pos[j]):
                    FP2 += 1
                    # Otherwise adding 1 to TN
                else:
                    TN2 += 1
                # For every row in fruit_pos
        for j in range(len(fruit_pos)):
                    # If i is equal to j then add 1 to FP
                if(countries_pos[i]==fruit_pos[j]):
                    FP2 += 1
                    # Otherwise add 1 to TN
                else:
                    TN2 += 1
                # For every row in veggies_pos
        for j in range(len(veggies_pos)):
                    # If i is equal to j then add 1 to FP
                if(countries_pos[i] == veggies_pos[j]):
                    FP2 += 1
                    # Otherwise add 1 to TN
                else:
                    TN2 += 1

        for i in range(len(veggies_pos)):
    # For every row in countries_pos
            for j in range(len(veggies_pos)):
            # If i and j are not the same, and j > i
                if (i != j & j>i):
            # If i is equal to j then add 1 to TP
                    if(veggies_pos[i] == veggies_pos[j]):
                        TP4 += 1
                # Otherwise adding 1 to FN
                    else:
                        FN4 += 1
                #For every row in animal_pos                
            for j in range(len(animal_pos)):
                    # If i is equal to j then add 1 to FP
                    if(veggies_pos[i] == animal_pos[j]):
                        FP4 += 1
                    # Otherwise adding 1 to TN
                    else:
                        TN4 += 1
                # For every row in fruit_pos
            for j in range(len(countries_pos)):
                    # If i is equal to j then add 1 to FP
                    if(veggies_pos[i]==countries_pos[j]):
                        FP4 += 1
                    # Otherwise adding 1 to TN
                    else:
                        TN4 += 1
                # For every row in veggies_pos
            for j in range(len(fruit_pos)):
                    # If i is equal to j then add 1 to FP
                    if(veggies_pos[i] == fruit_pos[j]):
                        FP4 += 1
                    # Otherwise adding 1 to TN
                    else:
                        TN4 += 1



     # For every row in animal_pos
    for i in range(len(animal_pos)):
    # For every row in animal_pos
        for j in range(len(animal_pos)):
            # If i and j are not the same, and j > i
            if (i != j & j>i):
            # If i is equal to j then add 1 to TP
                if(animal_pos[i] == animal_pos[j]):
                    TP1 += 1
                # Otherwise adding 1 to FN
                else:
                    FN1 += 1
                #For every row in countries_pos                
        for j in range(len(countries_pos)):
                    # If i is equal to j then add 1 to FP
                if(animal_pos[i] == countries_pos[j]):
                    FP1 += 1
                    # Otherwise adding 1 to TN
                else:
                    TN1 += 1
                # For every row in fruit_pos
        for j in range(len(fruit_pos)):
                    # If i is equal to j then add 1 to FP
                if(animal_pos[i]==fruit_pos[j]):
                    FP1 += 1
                    # Otherwise adding 1 to TN
                else:
                    TN1 += 1
                # For every row in veggies_pos
        for j in range(len(veggies_pos)):
                    # If i is equal to j then add 1 to FP
                if(animal_pos[i] == veggies_pos[j]):
                    FP1 += 1
                    # Otherwise adding 1 to TN
                else:
                    TN1 += 1

    for i in range(len(fruit_pos)):
    # For every row in countries_pos
        for j in range(len(fruit_pos)):
            # If i and j are not the same, and j > i
            if (i != j & j>i):
            # If i is equal to j then add 1 to TP
                if(fruit_pos[i] == fruit_pos[j]):
                    TP3 += 1
                # Otherwise adding 1 to FN
                else:
                    FN3 += 1
                #For every row in animal_pos                
        for j in range(len(animal_pos)):
                    # If i is equal to j then add 1 to FP
                if(fruit_pos[i] == animal_pos[j]):
                    FP3 += 1
                    # Otherwise adding 1 to TN
                else:
                    TN3 += 1
                # For every row in fruit_pos
        for j in range(len(countries_pos)):
                    # If i is equal to j then add 1 to FP
                if(fruit_pos[i]==countries_pos[j]):
                    FP3 += 1
                    # Otherwise adding 1 to TN
                else:
                    TN3 += 1
                # For every row in veggies_pos
        for j in range(len(veggies_pos)):
                    # If i is equal to j then add 1 to FP
                if(fruit_pos[i] == veggies_pos[j]):
                    FP3 += 1
                    # Otherwise adding 1 to TN
                else:
                    TN3 += 1
    #Calculating the Precision Recall
    r1 = round((TP1 / (TP1 + FN1)), 2)
    r2 = round((TP2 / (TP2 + FN2)), 2)
    r3 = round((TP3 / (TP3 + FN3)), 2)
    r4 = round((TP4 / (TP4 + FN4)), 2)
    #Averaging off the Precission Recall
    R = round(((r1 + r2 + r3 + r4)/4), 2)


    
    
    #Calculating the Precision
    
    p1 = round((TP1 / (TP1 + FP1)), 2)
    p2 = round((TP2 / (TP2 + FP2)), 2)
    p3 = round((TP3 / (TP3 + FP3)), 2)
    p4 = round((TP4 / (TP4 + FP4)), 2)
    #Averaging of the Precision
    P = round(((p1 + p2 + p3 + p4)/4),2)


    #Calculating the F Score
    f1 = round((2 * (p1 * r1) / (p1 + r1)), 2)
    f2 = round((2 * (p2 * r2) / (p2 + r2)), 2)
    f3 = round((2 * (p3 * r3) / (p3 + r3)), 2)
    f4 = round((2 * (p4 * r4) / (p4 + r4)), 2)
    #Averaging of the F score
    F = round(((f1+f2+f3+f4)/4), 2)
    
    return P, R, F

#Defining the class for plotting

def plotting(k, P, R, F,norm):
    # Plotting K against P
    plt.plot(K_list, P_list, label="B-Cubed Precision")
    # Plotting K against R
    plt.plot(K_list, R_list, label="B-Cubed Precision Recall")
    # Plotting K against F
    plt.plot(K_list, F_list, label="B-Cubed Precision F-Score")
    if norm==True:
        # Plotting the title
        plt.title("K-Means Clustering with Normalisation", loc="left")
    else:
        # Plotting the title
        plt.title("K-Means Clustering without Normalisation", loc="left")
    # Plotting the x and y axis labels
    plt.xlabel('Number of Clusters')
    plt.ylabel("Score")
    # Displaying the legend
    plt.legend()
    # Displaying the plot
    plt.show()


#Creating empty lists
P_list = []
R_list = []
F_list = []
K_list = []

for k in range(1,10):
    K_list.append(k)
    P,R,F = kMeansNormalisation(dataset,k,False)
    P_list.append(P)
    R_list.append(R)
    F_list.append(F)
plotting(K_list, P_list, R_list, F_list,False)

P_list = []
R_list = []
F_list = []
K_list = []
for k in range(1,10):
    K_list.append(k)
    P,R,F = kMeansNormalisation(dataset,k,True)
    P_list.append(P)
    R_list.append(R)
    F_list.append(F)
plotting(K_list, P_list, R_list, F_list,True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#formatting the data

'''

Note - Please change the path of different files accordingly to run in your system .

'''
animals = pd.read_csv("C:\\Users\\arpit\\Documents\\Data Mining\\A2\\animals", sep = " ", header = None).to_numpy()
countries = pd.read_csv("C:\\Users\\arpit\\Documents\\Data Mining\\A2\\countries", sep = " ", header = None).to_numpy()
fruits = pd.read_csv("C:\\Users\\arpit\\Documents\\Data Mining\\A2\\fruits", sep = " ", header = None).to_numpy()
veggies = pd.read_csv("C:\\Users\\arpit\\Documents\\Data Mining\\A2\\veggies", sep = " ", header = None).to_numpy()



animals[:,0] = 0
countries[:,0] = 1
fruits[:,0] = 2
veggies[:,0] = 3

data = np.vstack((animals, countries, fruits, veggies))

X = data[:,1:]
y = data[:,:1]



nFeatures = len(X[0]) #number of features


#euclidian distance calculator
def distance(x, y):
    dist = np.linalg.norm(x - y)
    return dist


def mk(X, y, k):
    X = X / np.linalg.norm(X)
    print("K value: ", k)
    iterations = 100
    centroids = X[np.random.choice(X.shape[0], size = k, replace=False), :]
    count = 0

    for _ in range(iterations):
        count += 1
        #initializing empty clusters, empty cluster list for each k value    
        clusters = [[] for _ in range(k)]

        #assigning cluster for each sample
        for idx, i in enumerate(X):
            distList = [distance(i, centroid) for centroid in centroids]
            closestCentroidIndex = np.argmin(distList)
            clusters[closestCentroidIndex].append(idx)
        
        #after updating cluster, storing old centroids
        centroidsOld = centroids
        
        #Initialising new centroids
        centroids = np.zeros((k, nFeatures))

        #for each cluster, calculating the mean and update centroid
        for clusterIdx, cluster in enumerate(clusters):
            clusterMean = np.mean(X[cluster], axis=0)
            #centroids of the current index = clusterMean
            centroids[clusterIdx] = clusterMean

        #checking if no update to centroid
        error = [distance(centroidsOld[i], centroids[i]) for i in range(k)]
        if np.sum(error) == 0:
            break

    #printing(clusters)
    print("Iterations until convergence: ", count)

    #getting cluster labels
    sampleClass = [(np.array(y[cluster])).T for cluster in clusters]

    # creating empty list of class counter dictionaries for each cluster
    clusterCount = [{"0":0,"1":0, "2":0, "3":0} for i in sampleClass]

    
    for idx, i in enumerate(sampleClass):
        for j in i[0]:
            if j == 0:
                clusterCount[idx]["0"] += 1
            elif j ==1:
                clusterCount[idx]["1"] += 1
            elif j ==2:
                clusterCount[idx]["2"] += 1
            else:
                clusterCount[idx]["3"] += 1
    
    
    precision = 0
    recall = 0

    for idx, i in enumerate(clusterCount):
        precision += (i["0"] ** 2) /len(clusters[idx])
        precision += (i["1"] ** 2) /len(clusters[idx])
        precision += (i["2"] ** 2) /len(clusters[idx])
        precision += (i["3"] ** 2) /len(clusters[idx])
        recall += (i["0"] ** 2) / len(animals)
        recall += (i["1"] ** 2) / len(countries)
        recall += (i["2"] ** 2) / len(fruits)
        recall += (i["3"] ** 2) / len(veggies)

    precision = precision/len(X)
    recall = recall/len(X)
    fScore = 2*((precision*recall)/(precision+recall))
    
    return precision,recall,fScore






      





print("Q3 - Run the k-means clustering algorithm you implemented in part (1) to cluster the given instances. Vary the value of k from 1 to 9 and compute the B-CUBED precision, recall, and F-score for each set of clusters. Plot k in the horizontal axis and the B-CUBED precision, recall and F-score in the vertical axis in the same plot. ")


np.random.seed(100)


    #Creating empty lists 
P_list = []
R_list = []
F_list = []
K_list = []
    
for k in range(1,10):
    K_list.append(k)
    P,R,F = mk(X,y,k)
    P_list.append(P)
    R_list.append(R)
    F_list.append(F)
#Passing False for without normalisation
plotting(K_list, P_list, R_list, F_list,False)

P_list = []
R_list = []
F_list = []
K_list = []


print("Q4 - Now re-run the k-means clustering algorithm you implemented in part (1) but normalise each object (vector) to unit `2length before clustering. Vary the value of k from 1 to 9 and compute the B-CUBED precision, recall, and F-score for each set of clusters. Plot k in the horizontal axis and the B-CUBED precision, recall and F-score in the vertical axis in the same plot. ")
for k in range(1,10):
    K_list.append(k)
    P,R,F = mk(X,y,k)
    P_list.append(P)
    R_list.append(R)
    F_list.append(F)
#Passing False for with normalisation
plotting(K_list, P_list, R_list, F_list,True)








#Reading the data from the source files
animals = pd.read_csv(path + "animals", sep = " ", header = None)
countries = pd.read_csv(path + "countries", sep = " ", header = None)
fruits = pd.read_csv(path + "fruits", sep = " ", header = None)
veggies = pd.read_csv(path + "veggies", sep = " ", header = None)

# Adding the cluster category in a new column
animals['Category'] = 'animals'
countries['Category'] = 'countries'
fruits['Category'] = 'fruits'
veggies['Category'] = 'veggies'

# Combining all the data together
data = pd.concat([animals, countries, fruits, veggies], ignore_index = True)

# Changing all class labels to numbers starting from 0
labels = (pd.factorize(data.Category)[0]+1) - 1 # 0=animals, 1=countries, 2=fruits, 3=veggies
dataset = data.drop([0, 'Category'], axis = 1).values

# Saving the maximum index for each category for the P/R/F
maxAni = data.index[data['Category'] == 'animals'][-1]
maxCount = data.index[data['Category'] == 'countries'][-1]
maxFruit = data.index[data['Category'] == 'fruits'][-1]
maxVeg = data.index[data['Category'] == 'veggies'][-1]

#Defining the K meadian class
def kMedians(x,k):
    np.random.seed(1)
    """
    Initialisation phase
    Randomly initialise the first centroids
    """
    centroids = []
    temp = np.random.randint(x.shape[0], size = k)
    while (len(temp) > len(set(temp))):
        temp = np.random.randint(x.shape[0], size = k)
    for i in temp:
        centroids.append(x[i])
    centroids_old = np.zeros(np.shape(centroids))
    centroids_new = centroids
    
    # Creating a blank distance and cluster assignment object to hold results
    clusters = np.zeros(x.shape[0])
    
    # Creating an error object
    error = np.linalg.norm(centroids_new - centroids_old)
    num_errors = 0
    
    # If there is an error value:
    while error != 0:
        dist = np.zeros([x.shape[0], k])
        # Adding one to the number of errors
        num_errors += 1
        # Calculating the Manhattan distance from each point to each centroid    
        for j in range(len(centroids)):
            dist[:, j] = np.sum(np.abs(x - centroids_new[j]), axis=1)
        
        #This is the Asignment phase where I am assigning objects in the dataset to their closest representatives
        
        # Calculating the cluster assignment
        clusters = np.argmin(dist, axis = 1)
        centroids_old = centroids_new
        
        
        #This is the optimization phase where I am computing new representatives as the means of current clusters
        
        # Calculating the mean to re-adjust the cluster centroids
        for m in range(k):
            centroids_new[m] = np.median(x[clusters == m], axis = 0)
        # Re-calculating the error
        error = np.linalg.norm(np.array(centroids_new) - np.array(centroids_old))

    #Assigning the final clusters and centroids to new objects
    predicted_clusters = clusters
    final_centroids = np.array(centroids_new)
    print ("Points Belonging to Final Clusters:\n", predicted_clusters)
    print ("Final Centroid Locations:\n", final_centroids)

print("Q2 - Implement the k-medians clustering algorithm to cluster the instances into k clusters. ")
kMedians(dataset,4)


np.random.seed(35)
#Defining the Normalisation class

def kMediansNormalisation(x,k,norm):
    
    #L2 normalisation
    
    if norm==True:
        x = x / np.linalg.norm(x)
        
    else:
        x = x
        
    
    #This is the initialisation phase where I am randomly initialise the first centroids
    
    centroids = []
    temp = np.random.randint(x.shape[0], size = k)
    while (len(temp) > len(set(temp))):
        temp = np.random.randint(x.shape[0], size = k)
    for i in temp:
        centroids.append(x[i])
    centroids_old = np.zeros(np.shape(centroids))
    centroids_new = centroids
    
    # Creating a blank distance and cluster assignment object to hold results
    clusters = np.zeros(x.shape[0])
    
    # Creating an error object
    error = np.linalg.norm(centroids_new - centroids_old)
    num_errors = 0
    
    # If there is an error value:
    while error != 0:
        dist = np.zeros([x.shape[0], k])
        # Adding one to the number of errors
        num_errors += 1
        # Calculating the Manhattan distance from each point to each centroid    
        for j in range(len(centroids)):
            dist[:, j] = np.sum(np.abs(x - centroids_new[j]), axis=1)
        """
        Assignment phase
        Assigning objects in the dataset to their closest representatives
        """
        # Calculating  the cluster assignment
        clusters = np.argmin(dist, axis = 1)
        centroids_old = centroids_new
        
      
        
#This is optimization phase where I am computing new representatives as the means of current clusters
       
        # Calculating the mean to re-adjust the cluster centroids
        for m in range(k):
            centroids_new[m] = np.median(x[clusters == m], axis = 0)
        # Re-calculateing the error
        error = np.linalg.norm(np.array(centroids_new) - np.array(centroids_old))

    #Assigning the final clusters and centroids to new objects
    predicted_clusters = clusters
    final_centroids = np.array(centroids_new)
    
    # Creating objects of the index positioning of the different classes
    animal_pos = predicted_clusters[:maxAni+1]
    countries_pos = predicted_clusters[maxAni+1:maxCount+1]
    fruit_pos = predicted_clusters[maxCount+1:maxFruit+1]
    veggies_pos = predicted_clusters[maxFruit+1:maxVeg+1]

    # Assigning True Positives
    TP = 0
    # Assigning True Negatives
    TN = 0
    # Assigning False Positives
    FP = 0
    # Assigning False Negatives
    FN = 0
    
    
    
    
  
    #For every row in countries_pos 
    for i in range(len(countries_pos)):
                #For every row in countries_pos 
        for j in range(len(countries_pos)):
                    # If i and j are not the same, and j > i
            if (i != j & j>i):
                        # If i is equal to j then add 1 to TP
                if(countries_pos[i] == countries_pos[j]):
                    TP += 1
                        # Otherwise add 1 to FN
                else:
                    FN += 1
                # For every row in fruit_pos
        for j in range(len(fruit_pos)):
                    # If i is equal to j then add 1 to FP
                if(countries_pos[i] == fruit_pos[j]):
                    FP += 1
                    # Otherwise add 1 to TN
                else:
                    TN += 1
                # For every row in veggies_pos
        for j in range(len(veggies_pos)):
                    # If i is equal to j then add 1 to FP
                if(countries_pos[i] == veggies_pos[j]):
                    FP += 1
                    # Otherwise add 1 to TN
                else:
                    TN += 1
# For every row in animal_pos
    for i in range(len(animal_pos)):
    # For every row in animal_pos
        for j in range(len(animal_pos)):
            # If i and j are not the same, and j > i
            if (i != j & j>i):
                # If i is equal to j then add 1 to TP
                if(animal_pos[i] == animal_pos[j]):
                    TP += 1
                # Otherwise add 1 to FN
                else:
                    FN += 1
                #For every row in countries_pos                
        for j in range(len(countries_pos)):
                    # If i is equal to j then add 1 to FP
                if(animal_pos[i] == countries_pos[j]):
                    FP += 1
                    # Otherwise add 1 to TN
                else:
                    TN += 1
                # For every row in fruit_pos
        for j in range(len(fruit_pos)):
                    # If i is equal to j then add 1 to FP
                if(animal_pos[i]==fruit_pos[j]):
                    FP += 1
                    # Otherwise add 1 to TN
                else:
                    TN += 1
                # For every row in veggies_pos
        for j in range(len(veggies_pos)):
                    # If i is equal to j then add 1 to FP
                if(animal_pos[i] == veggies_pos[j]):
                    FP += 1
                    # Otherwise add 1 to TN
                else:
                    TN += 1

    # For every row in veggies_pos
    for i in range(len(veggies_pos)):       
                # For every row in veggies_pos
        for j in range(len(veggies_pos)):
                    # If i and j are not the same, and j > i
            if (i != j & j>i):
                        # If i is equal to j then add 1 to TP
                if(veggies_pos[i] == veggies_pos[j]):
                    TP += 1
                        # Otherwise add 1 to FN
                else:
                    FN += 1



    # For every row in fruit_pos
    for i in range(len(fruit_pos)):
                # For every row in fruit_pos
        for j in range(len(fruit_pos)):
                    # If i and j are not the same, and j > i
            if (i != j & j>i):
                        # If i is equal to j then add 1 to TP
                if(fruit_pos[i] == fruit_pos[j]):
                    TP += 1
                        # Otherwise add 1 to FN
                else:
                    FN += 1
                # For every row in veggies_pos
        for j in range(len(veggies_pos)):
                    # If i is equal to j then add 1 to FP
            if(fruit_pos[i] == veggies_pos[j]):
                    FP += 1
                    # Otherwise add 1 to TN
            else:
                    TN += 1    
# Calculating the Precision (P), Recall (R), and F-Score (F) and round  to 2 decimal places
    P = round((TP / (TP + FP)), 2)
    R = round((TP / (TP + FN)), 2)
    F = round((2 * (P * R) / (P + R)), 2)
    return P, R, F

def plotting(k, P, R, F,norm):
    # Plotting K against P
    plt.plot(K_list, P_list, label="Precision")
    # Plotting K against R
    plt.plot(K_list, R_list, label="Recall")
    # Plotting K against F
    plt.plot(K_list, F_list, label="F-Score")
    if norm==True:
        # Plotting the title
        plt.title("K-Median Clustering with Normalisation", loc="left")
    else:
        # Plotting the title
        plt.title("K-Median Clustering without Normalisation", loc="left")
    # Plotting the x and y axis labels
    plt.xlabel('Number of Clusters')
    plt.ylabel("Score")
    # Displaying the legend
    plt.legend()
    # Displaying the plot
    plt.show()
print("Q5 - Run the k-medians clustering algorithm you implemented in part (2) over the unnormalised objects. Vary the value of k from 1 to 9 and compute the B-CUBED precision, recall, and F-score for each set of clusters. Plot k in the horizontal axis and the B-CUBED precision, recall and F-score in the vertical axis in the same plot.")
#Creaing empty lists
P_list = []
R_list = []
F_list = []
K_list = []

for k in range(1,10):
    K_list.append(k)
    print("k = ",k)
    P,R,F = kMediansNormalisation(dataset,k,False) 
    P_list.append(P)
    R_list.append(R)
    F_list.append(F)
    print("B cubed Precision = ",P ,"| B cubed  Recall = ",R ,"| B cubed   F Score = ",F)
plotting(K_list, P_list, R_list, F_list,False)

print("Q6 - Now re-run the k-medians clustering algorithm you implemented in part (2)  but normalise each object (vector) to unit l2 length before clustering. Vary the value of  k from 1 to 9 and compute the B-CUBED precision, recall, and F-score for each set of  clusters. Plot k in the horizontal axis and the B-CUBED precision, recall and F-score in  the vertical axis in the same plot.")
#Creating empty list
P_list = []
R_list = []
F_list = []
K_list = []

for k in range(1,10):
    K_list.append(k)
    print("k = ",k)
    P,R,F = kMediansNormalisation(dataset,k,True)
    P_list.append(P)
    R_list.append(R)
    F_list.append(F)
    print("B cubed Precision = ",P ,"| B cubed  Recall = ",R ,"| B cubed F Score = ",F)
plotting(K_list, P_list, R_list, F_list,True)


    
