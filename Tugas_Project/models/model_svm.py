import numpy as np
from sklearn.utils import shuffle

class SVM:
  
    def __init__(self, learning_rate=0.000001, max_epoch=1000, regularization=10000):
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.regularization = regularization
        self.W = None
        
    def hitung_cost_gradient(self, W, X, Y):
        jarak = 1 - (Y * np.dot(X, W))
        dw = np.zeros(len(W))
        if max(0, jarak) == 0:
            di = W
        else:
            di = W - (self.regularization * Y * X)
        dw += di
        return dw

    def sgd(self, data_latih, label_latih):
        data_latih = data_latih.to_numpy()
        label_latih = label_latih.to_numpy()
        self.W = np.zeros(data_latih.shape[1])
        for epoch in range(1, self.max_epoch):
            X, Y = shuffle(data_latih, label_latih, random_state=101)
            for index, x in enumerate(X):
                delta = self.hitung_cost_gradient(self.W, x, Y[index])
                self.W = self.W - (self.learning_rate * delta)

    def predict(self, data_uji):
        prediksi = np.array([])
        for i in range(data_uji.shape[0]):
            y_prediksi = np.sign(np.dot(self.W, data_uji.to_numpy()[i]))
            prediksi = np.append(prediksi, y_prediksi)
        return prediksi

    def fit(self, data_latih, label_latih):
        self.sgd(data_latih, label_latih)
