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

  def hitung_likelihood_general(self, data_latih, cat, num):
    list_columns = data_latih.columns[:-1]
    class_column_name = data_latih.columns[-1]
    list_class = set(data_latih[class_column_name])  
    rata2, std = self.hitung_rata2_std_kelas(data_latih)
    likelihood = {}
    for column in list_columns:
      value = set(data_latih[column])
      for a_class in list_class :
        if column in cat:
          for valtype in value:
            countw = data_latih[column].values == valtype
            countc = data_latih[class_column_name].values == a_class
            countw_c = (countw & countc).sum()
            count_c = countc.sum()
            likelihood[(column, valtype, a_class)] = countw_c / count_c
        elif column in num:
          for i in range (data_latih.shape[0]):
            likelihood[(column, np.NaN, a_class)] = self.hitung_likelihood_gaussian(data_latih[column].iloc[i],rata2[(a_class,column)],std[(a_class,column)])
    return likelihood

  def training(self, data_latih):
    
    class_column_name = data_latih.columns[-1]
    prior = self.hitung_prior(data_latih[class_column_name])
    likelihood = self.hitung_likelihood_general(data_latih, cat_columns, num_columns)
    list_class = set(data_latih[class_column_name])
    list_columns = data_latih.columns[:-1]
    model = {}
    model['prior'] = prior
    model['likelihood'] = likelihood
    model['list_class'] = list_class
    model['list_columns'] = list_columns
    return model
  
  def testing_naive_bayes_gaussian (self, model, data_uji):
    prior = model['prior']
    rata2 = model['rata2']
    std = model['std']
    list_class = model['list_class']
    print(list_class)
    list_columns = model ['list_columns']
    posterior = dict.fromkeys (list_class, 1)
    for a_class in list_class:
      for column in list_columns:
        posterior [a_class] = posterior[a_class]* self.hitung_likelihood_gaussian(data_uji[column],rata2[(a_class,column)],std[(a_class,column)])
        posterior [a_class] = posterior[a_class] * prior[a_class]
    kelas_uji = max (posterior, key=posterior.get)
    print(posterior)
    return kelas_uji

  def predict(self, data_latih, data_uji):
    num_columns = data_uji.select_dtypes(include=['floating']).columns
    cat_columns = data_uji.select_dtypes(include=['integer', 'object']).columns[:-1]
    model = self.training(data_latih)
    prior = model['prior']
    likelihood = model['likelihood']
    # print(any(item in likelihood for item in [0]))
    list_class = model['list_class']
    list_columns = model['list_columns']
    posterior = dict.fromkeys(list_class,1)
    total_predictions = []
    for index in range(data_uji.shape[0]):
      for column in list_columns:
        if column in cat_columns:
          for a_class in list_class:
            posterior [a_class] = posterior[a_class]*likelihood[(column, data_uji[column].iloc[index], a_class)]* prior[a_class]
        elif column in num_columns:
          for a_class in list_class:
            posterior [a_class] = posterior[a_class]*likelihood[(column, np.NaN, a_class)]* prior[a_class]
      # print(posterior)
      kelas_uji = max(posterior, key=posterior.get)
      total_predictions.append(kelas_uji)
    return total_predictions