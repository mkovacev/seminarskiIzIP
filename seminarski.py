import csv
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score


def loadData():
    data = pd.read_csv('/home/matija/Desktop/2. semestar/IP1/craigslist-carstrucks-database/craigslistVehiclesFull.csv')
    #data = pd.read_csv('testBaza.csv')
    data = data.iloc[::4, :]
    return data
    
def normalize(array):
    maxElem = np.amax(array)
    for i in range(len(array)):
        array[i] = array[i]/maxElem

    return array

def kmeansClustering(x_data, y_data, clusterNumber):
    data = loadData()
    print(2)
    izbaceneNanVrednosti = data[[x_data, y_data]].dropna()

    izbaceneNanVrednosti = izbaceneNanVrednosti.drop(izbaceneNanVrednosti[izbaceneNanVrednosti[x_data]==0].index)
    # brojac = 0
    # for i in range(0,len(izbaceneNanVrednosti)):
    #     if izbaceneNanVrednosti.iat[i,0] == 0:
    #         brojac += 1

    # print(brojac)



    features = izbaceneNanVrednosti.columns

    scaler = MinMaxScaler().fit(izbaceneNanVrednosti[features])
    x = pd.DataFrame(scaler.transform(izbaceneNanVrednosti[features]))
    x.columns = features
        
    fig = plt.figure(figsize=(10, 10))
    plt_ind = 1

    print("1\n")
    est = KMeans(n_clusters=clusterNumber, init='random')
    est.fit(x)
    izbaceneNanVrednosti['labels'] = est.labels_

    centers = pd.DataFrame(scaler.inverse_transform(est.cluster_centers_), columns = features)

    sp = fig.add_subplot(2, 2, plt_ind)

    for j in range(0, clusterNumber):
        cluster = izbaceneNanVrednosti.loc[izbaceneNanVrednosti['labels'] == j]
        k = j+1
        plt.scatter(cluster[x_data], cluster[y_data], cmap='rainbow', label = "klaster %d"%k)

    sp.scatter(centers[x_data], centers[y_data], color = "black", marker='x', label = 'centroidi')
    #plt.title('Senka %0.3f'% silhouette_score(x, est.labels_))
    plt.legend()
    plt_ind += 1

    plt.tight_layout()
    plt.show()

def hierarchyClustering(x_data, y_data, clusterNumber):
    data = loadData()

    izbaceneNanVrednosti = data[[x_data, y_data]].dropna()
    print(izbaceneNanVrednosti)

    print(len(izbaceneNanVrednosti.loc[x_data] == 0))
    
    features = izbaceneNanVrednosti.columns[1:]
    scaler = MinMaxScaler().fit(izbaceneNanVrednosti[features])
    x = pd.DataFrame(scaler.transform(izbaceneNanVrednosti[features]))
    x.columns = features
    
    fig = plt.figure(figsize=(5,5))
    plt_ind = 1

    for link in ['complete', 'average', 'single']:
        est = AgglomerativeClustering(n_clusters=clusterNumber, linkage='complete', affinity='euclidian')
        est.fit(x)
        izbaceneNanVrednosti['lables'] = est.labels_

        fig.add_subplot(2, 2, plt_ind)

        for j in range(0, clusterNumber):
            cluster = izbaceneNanVrednosti.loc[izbaceneNanVrednosti['labels'] == j]
            plt.scatter(cluster[x_data], cluster[y_data], cmap='rainbow', label = "cluster %d"%j)
        
        plt.title("Linkage %s" % link)
        plt.legend()

        plt_ind += 1

    plt.tight_layout()
    plt.show()


def main():
    #hierarchyClustering('price', 'odometer', 3)

    kmeansClustering('price', 'odometer', 3)

if __name__ == "__main__":
    main()