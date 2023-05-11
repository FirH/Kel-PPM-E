import numpy as np

def jarakEu(data1, data2) :
  jarak = np.square(data1-data2)
  jarak = np.sum(jarak)
  return np.sqrt(jarak)

from collections import Counter
def predict(k,datalatih,labellatih,datauji) :
  jarak = np.array([jarakEu(datalatih.iloc[x], datauji) for x in range (datalatih.shape[0])])
  indeks_k_minimum = jarak.argsort()[:k]
  k_kelas = labellatih.iloc[indeks_k_minimum].to_numpy()
  counter = Counter(k_kelas)
  kelas_uji = counter.most_common(1)[0][0]
  return kelas_uji

