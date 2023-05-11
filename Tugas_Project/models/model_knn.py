import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k):
        self.k = k
    
    def jarakEu(self, data1, data2):
        jarak = np.square(data1-data2)
        jarak = np.sum(jarak)
        return np.sqrt(jarak)
    
    def knn(self, k, datalatih, datauji):
        jarak = np.array([self.jarakEu(datalatih.iloc[x], datauji) for x in range(datalatih.shape[0])])
        indeks_k_minimum = jarak.argsort()[:k]
        k_kelas = datalatih.iloc[indeks_k_minimum][datalatih.columns[-1]].to_numpy()
        counter = Counter(k_kelas)
        kelas_uji = counter.most_common(1)[0][0]
        return kelas_uji

    def predict(self, datalatih, datauji):
        return np.array([self.knn(self.k, datalatih, datauji.iloc[row]) for row in range(len(datauji))])
