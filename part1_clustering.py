import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Data retrieval and pre-processing
def process_data():
    url = "https://raw.githubusercontent.com/jamesH-48/K-Means-A3/master/HTRU_2.csv"
    raw_data = pd.read_csv(url, header=None)
    newdata = raw_data.rename(columns={0:"MIP",1:"SDIP",2:"EKIP",3:"SkIP",4:"MDM",5:"SDDM",6:"EKDM",7:"SkDM",8:"Class"})
    newdata = newdata.drop(['Class'], axis=1)
    # newdata = preprocessing.normalize(newdata, axis=0)
    return newdata

# Perform KMeans
# Return SSE values from different k-values
def kmeans(data, state):
    SSE = []
    print("Experiment Number|Value of k|SSE Value")
    for j in range(1, 11):
        kmeans = KMeans(n_clusters=j, random_state = state).fit(data)
        print("{0:17}|".format(j),end="")
        print("{0:10}|".format(j),end="")
        print("{0:9}".format(kmeans.inertia_,end=""))
        SSE.append(kmeans.inertia_)
    plt.plot(range(1,11), SSE, marker='*')
    plt.xlabel("k-value")
    plt.ylabel("Sum Squared Error")
    # Show any graphs created
    plt.show()

# Main
if __name__ == "__main__":
    # Set State
    state = 0
    #Retrieve & Pre-process Data
    data = process_data()
    kmeans(data, state)