from collections import Counter
import math
import numpy as np

class NB:
  
  def __init__(self):
    pass
  
  def hitung_prior(self, list_kelas):
    n_data = len(list_kelas)
    prior = Counter(list_kelas)
    for key in prior.keys():
      prior[key] = prior[key]/n_data
    return prior

  def hitung_rata2_std_kelas(self, input_data):
    list_columns = input_data.columns
    class_column_name = input_data.columns[-1]
    list_class = set(input_data[class_column_name])
    rata2 = {}
    std = {}
    for column in list_columns:
      for a_class in list_class :
        rata2 [(a_class, column)] = input_data.loc[input_data[class_column_name]==a_class][column].mean()
        std[(a_class,column)] = input_data.loc[input_data[class_column_name]==a_class][column].std()
    return (rata2, std)

  def hitung_likelihood_gaussian(self, data, rata2, std):
    hasil = (1/math.sqrt(2*math.pi*(std**2)))*math.exp((-1*((data-rata2)**2))/(2*(std**2))) 
    return hasil

  def training(self, data_latih) :
    class_column_name = data_latih.columns[-1]
    prior = self.hitung_prior(data_latih[class_column_name])
    (rata2, std) = self.hitung_rata2_std_kelas(data_latih)
    list_class = set(data_latih[class_column_name])
    list_columns = data_latih.columns[:-1]
    model = {}
    model['prior'] = prior
    model['rata2'] = rata2
    model['std'] = std
    model['list_class'] = list_class
    model['list_columns'] = list_columns
    return model
  
  def predict (self, data_latih, data_uji):
    model = self.training(data_latih)
    prior = model['prior']
    rata2 = model['rata2']
    std = model['std']
    list_class = model['list_class']
    list_columns = model ['list_columns']
    posterior = dict.fromkeys (list_class, 1)
    total_predictions = []
    for index in range(data_uji.shape[0]):    
        for a_class in list_class:
            for column in list_columns:
                posterior [a_class] = posterior[a_class]*self.hitung_likelihood_gaussian(data_uji[column].iloc[index],rata2[(a_class,column)],std[(a_class,column)])
                posterior [a_class] = posterior[a_class] * prior[a_class]
        kelas_uji = max (posterior, key=posterior.get)
        total_predictions.append(kelas_uji)
    return total_predictions
