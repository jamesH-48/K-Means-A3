# James Hooper ~ NETID: jah171230
# Hritik Panchasara ~ NETID: hhp160130
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Data retrieval and pre-processing
def process_data():
    url = "https://raw.githubusercontent.com/jamesH-48/K-Means-A3/master/HTRU_2.csv"
    raw_data = pd.read_csv(url, header=None)
    newdata = raw_data.rename(columns={0:"MIP",1:"SDIP",2:"EKIP",3:"SkIP",4:"MDM",5:"SDDM",6:"EKDM",7:"SkDM",8:"Class"})
    newdata = newdata.drop(['Class'], axis=1)
    # Scale Data
    #newdata = preprocessing.minmax_scale(newdata, axis=0)
    newdata = preprocessing.StandardScaler().fit_transform(newdata)
    return newdata

# Perform KMeans
# Return SSE values from different k-values
def kmeans_elbow(data, state):
    SSE = []
    print("Experiment Number|Value of k|SSE Value")
    for j in range(1, 11):
        kmeans = KMeans(n_clusters=j, random_state = state).fit(data)
        labels = kmeans.labels_
        print("{0:17}|".format(j),end="")
        print("{0:10}|".format(j),end="")
        print("{0:9}".format(kmeans.inertia_,end=""))
        SSE.append(kmeans.inertia_)
    fig1, ax1 = plt.subplots()
    ax1.plot(range(1,11), SSE, marker='*')
    ax1.set_xlabel("k-value")
    ax1.set_ylabel("Sum Squared Error")

# Perform KMeans
# Return Sil values from different k-values
'''
~ Has to start at k = 2
~ Has to use sampl_size of 6000 for the code to run properly with memory
    ~ May be able to use total size with better memory? 
    ~ Values seems similar with 5000 or 7000 sample size.
    ~ 7000+ sample size fails by the 10th iteration
'''
def kmeans_sil(data, state):
    Sil = []
    print("Experiment Number|Value of k|Sil Value")

    for j in range(2, 11):
        kmeans = KMeans(n_clusters=j, random_state = state).fit(data)
        labels = kmeans.labels_
        print("{0:17}|".format(j),end="")
        print("{0:10}|".format(j),end="")
        Sil.append(silhouette_score(data, labels, metric='euclidean', sample_size=6000))
        print("{0:9}".format(silhouette_score(data, labels, metric='euclidean', sample_size=6000), end=""))
    fig1, ax1 = plt.subplots()
    ax1.plot(range(2,11), Sil, marker='*')
    plt.xlabel("k-value")
    plt.ylabel("Silhouette Score")

# Main
if __name__ == "__main__":
    # Set State
    state = 0
    #Retrieve & Pre-process Data
    data = process_data()
    kmeans_elbow(data, state)
    kmeans_sil(data,state)
    # Show any graphs created
    plt.show()
