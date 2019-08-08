import csv
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler


def loadData():
    #with open("craigslistVehiclesFull.csv", "r") as f:
    #    dataReader = csv.reader(f)
    #    yield next(dataReader)

    data = pd.read_csv('testBaza.csv')
    
    # for index, row in data.iterrows() :
    #     print(row)
    #     print()

    # indeks = np.array(data['url'])
    # print(indeks)

    return data
    
def normalize(array):
    maxElem = np.amax(array)
    for i in range(len(array)):
        array[i] = array[i]/maxElem

    return array

def kmeansClustering(kmeans_data, brojKlastera):
    kmeans = KMeans(n_clusters=brojKlastera)
    kmeans.fit(kmeans_data)
    #print(kmeans.cluster_centers_)

    

def hierarchyClustering(x_data, y_data, clusterNumber):
    data = loadData()

    izbaceneNanVrednosti = data[[x_data, y_data]].dropna()
    print(izbaceneNanVrednosti)
    
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
            plt.scatter(cluster[x_data], cluster[y_data], color = colors[j], label = "cluster %d"%j)
        
        plt.title("Linkage %s" % link)
        plt.legend()

        plt_ind += 1

    plt.tight_layout()
    plt.show()


def main():
    hierarchyClustering('price', 'odometer', 3)
    # data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.6, random_state=50)
    data = loadData()

    # gradovi = np.array(data['city'])
    # cene = np.array(data['price'])
    # godineProizvodnje = np.array(data['year'])
    # proizvodjaci = np.array(data['manufacturer'])
    # oznakeVozila = np.array(data['make'])
    # stanja = np.array(data['condition'])
    # brCilindara = np.array(data['cylinders'])
    # tipoviGoriva = np.array(data['fuel'])
    # kilometraze = np.array(data['odometer'])
    # statusi = np.array(data['title_status'])
    # tipoviPrenosa = np.array(data['transmission'])
    # tipoviPogona = np.array(data['drive'])
    # velicine = np.array(data['size'])
    # tipoviVozila = np.array(data['type'])
    # boje = np.array(data['paint_color'])
    # geografskeSirine = np.array(data['lat'])
    # geografskeDuzine = np.array(data['long'])

    izbaceneNanVrednosti = data[['price', 'odometer']].dropna()
    print(izbaceneNanVrednosti)

    # create np array for data points
    # points = data[0]
    # print(data)
    
    features = izbaceneNanVrednosti.columns[1:]
    scaler = MinMaxScaler().fit(izbaceneNanVrednosti[features])
    x = pd.DataFrame(scaler.transform(izbaceneNanVrednosti[features]))
    x.columns = features
    
    fig = plt.figure(figsize=(5,5))
    plt_ind = 1
    clusterNumber = 16

    for link in ['complete', 'average', 'single']:
        est = AgglomerativeClustering(n_clusters=clusterNumber, linkage='complete', affinity='euclidian')
        est.fit(x)
        izbaceneNanVrednosti['lables'] = est.labels_

        fig.add_subplot(2, 2, plt_ind)

        for j in range(0, 3):
            cluster = izbaceneNanVrednosti.loc[izbaceneNanVrednosti['labels'] == j]
            plt.scatter(cluster['price'], cluster['odometer'], cmap='rainbow', label = "cluster %d"%j)
        
        plt.title("Linkage %s" % link)
        plt.legend()

        plt_ind += 1

    plt.tight_layout()
    plt.show()

    cene = np.array(izbaceneNanVrednosti['price'], dtype = float)
    kilometraze = np.array(izbaceneNanVrednosti['odometer'], dtype = float)
    godineProizvodnje = np.array(izbaceneNanVrednosti['year'], dtype = float)

    normalizovaneCene = normalize(cene)
    normalizovaneKilometraze = normalize(kilometraze)
    normalizovaneGodineProzivodnje = normalize(godineProizvodnje)
    
    

    kmeans_data = [0]*len(normalizovaneCene)

    for i in range(len(normalizovaneCene)):
        kmeans_data[i] = [normalizovaneCene[i], normalizovaneGodineProzivodnje[i]]

    kmeans_data = np.array(kmeans_data, dtype=float)

    y_km = kmeansClustering(kmeans_data, clusterNumber)
    #print(kmeans_data)
    #print(kmeans_data[y_km ==0,0])
    #colors = iter(cm.rainbow(np.linspace(0, 1, len(y_km))))

    for i in range(clusterNumber):
        plt.scatter(kmeans_data[y_km ==i,0], kmeans_data[y_km == i,1], s=10, cmap='rainbow')
    
    plt.show()
    

if __name__ == "__main__":
    main()