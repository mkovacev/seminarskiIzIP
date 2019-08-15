import csv
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sklearn.metrics as met
import matplotlib
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn import preprocessing



def loadData():
    data = pd.read_csv('/home/matija/Desktop/2. semestar/IP1/craigslist-carstrucks-database/vehicles.csv')
    data = data.drop(['url', 'city', 'manufacturer', 'make', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission', 'vin', 'drive', 'size', 'type', 'paint_color', 'image_url', 'county_fips', 'county_name', 'state_fips', 'state_code', 'state_name', 'weather'], axis=1)
    data = data.dropna()
    data = data.sample(10000)
    #data = data.iloc[::30, :]
    #data = pd.read_csv('testBaza.csv')
    return data
    
def normalize(array):
    maxElem = np.amax(array)
    for i in range(len(array)):
        array[i] = array[i]/maxElem

    return array

def nebalansiranSkup(x_data, y_data, clusterNumber):
    df = loadData()
    df = df[[x_data, y_data]].dropna()


    #prikaz imena kolona + 5 prvih instanci
    # print('Prvih 5 instanci', df.head(), sep='\n')
    # print('\n\n')


    features = df.columns.tolist()

    print(features)
    x_original=df[features]

    #standardizacija atributa
    x=pd.DataFrame(preprocessing.scale(x_original))

    #dodeljivanje imena kolonama
    x.columns = features

    colors = ['darkcyan', 'red', 'green', 'gold', 'blue',  'm', 'plum', 'orange', 'black']

    font = {'family' : 'normal',
            'size'   : 6}

    matplotlib.rc('font', **font)

    fig = plt.figure()
    plt_ind=1

    for i in range(5, 9):
        estimators= { 'K_means': KMeans(n_clusters=i),
                    'hijerarhijsko': AgglomerativeClustering(n_clusters=i, linkage='average'),
                    'DBSCAN': DBSCAN(eps=(i-2)*0.1)
                    }

        for name, est in estimators.items():
            est.fit(x)
            df['labels']= est.labels_

            fig.add_subplot(4, 3, plt_ind)

            if name=='DBSCAN':
                num_clusters = max(est.labels_) + 1
                min=-1
            else:
                num_clusters=i
                min=0
            for j in range(min,num_clusters):
                cluster= df.loc[lambda x: x['labels'] == j, :]
                plt.scatter(cluster[x_data], cluster[y_data], color=colors[j], s=10, marker='o', label="cluster %d"%j)

            plt.title('Algorithm %s, num clasters: %d'%(name, num_clusters), fontsize=8)
            plt_ind += 1

    plt.tight_layout()
    plt.show()
    
def DBscan():
    df = loadData()
    
    features = df.columns

    scaler = MinMaxScaler().fit(df[features])
    x = pd.DataFrame(scaler.transform(df[features]))
    x.columns = features

    
    est = DBSCAN(eps=0.5, min_samples=10)
    est.fit(x)
    df['labels'] = est.labels_
    print('Eps: ', 0.5, 'Min_samples: ', 10)
    print('silhouette_score: ', silhouette_score(x, est.labels_))
    br_klas = max(est.labels_) + 1
    print('broj klastera: ', br_klas)
    print()
        
    
    x.to_csv('DBS.csv')



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

    #print(izbaceneNanVrednosti)

    #print(len(izbaceneNanVrednosti.loc[x_data] == 0))
    
    features = data.columns
    scaler = MinMaxScaler().fit(data[features])
    x = pd.DataFrame(scaler.transform(data[features]))
    x.columns = features

    # for link in ['complete', 'average', 'single']:
    #     est = AgglomerativeClustering(n_clusters=clusterNumber, linkage=link, affinity='euclidean')
    #     est.fit(x)
    #     data['labels'] = est.labels_

    #     print('link: ', link, 'affinity: ', 'eucledian', 'n of clusters: ', clusterNumber, 'silhouette: ', silhouette_score(x, est.labels_))
    #     data.head()
    est = AgglomerativeClustering(n_clusters=clusterNumber, linkage='single', affinity='euclidean')
    est.fit(x)
    data['labels'] = est.labels_

    print('link: ', 'single', 'affinity: ', 'eucledian', 'n of clusters: ', clusterNumber, 'silhouette: ', silhouette_score(x, est.labels_))
        

    data.to_csv("Aggl.csv")


def main():
    #hierarchyClustering('price', 'odometer', 3)
    #kmeansClustering('price', 'odometer', 3)
    DBscan()
    #nebalansiranSkup('price', 'odometer', 3)

if __name__ == "__main__":
    main()