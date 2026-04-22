from abc import abstractmethod
import numpy as np

class LossFunction:

    @abstractmethod
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true, y_pred):
        raise NotImplementedError


class MeanSquaredError(LossFunction):

    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):
    
    def loss(self, y_true, y_pred):
        # Clip predictions to avoid log(0), then apply the cross-entropy formula.
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # Usar np.mean em vez de np.sum para obter a média do batch
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true, y_pred):
        # Clip predictions to avoid division by zero, then compute the gradient.
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # Calcular o gradiente e dividir pelo tamanho da amostra (número de elementos)
        grad = - (y_true / p) + (1 - y_true) / (1 - p)
        return grad / y_true.size
