import numpy as np


class Perceptron:
    def __init__(self, feature_dim, num_classes):
        self.Weight=np.zeros((num_classes,feature_dim+1))


    def train(self, feature_vector, y):
        predicted_label=self.predict(feature_vector)
        if predicted_label!=y:
            self.Weight[predicted_label]=self.Weight[predicted_label]-feature_vector
            self.Weight[y]=self.Weight[y]+feature_vector


    def predict(self, feature_vector):
        return np.argmax(  self.Weight   @ feature_vector.reshape(-1,1)    )

