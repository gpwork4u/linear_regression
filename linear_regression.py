import numpy as np
import sys
class Linear_model:
    
    def __init__(self, num):
        self.weights = np.random.normal(0, 1, num)
        self.bias = np.random.normal(0, 1, 1) 

    def _output(self, inputs):
        out = np.matmul(self.weights, inputs.transpose()) 
        out += self.bias
        return out

    def _mse_loss(self, inputs, outputs):
        loss = np.mean(np.square(self._output(inputs)-outputs))
        return loss
    
    def _mse_gradient(self, inputs, outputs):
        pred_outputs = self._output(inputs)
        gradient_w = - np.dot(2 * (pred_outputs - outputs), inputs)/inputs.shape[0]
        gradient_b = - np.mean(2 * (pred_outputs - outputs), axis=0)
        return gradient_w, gradient_b

    def _l2_loss(self):
        loss = np.sum(np.square(self.weights))

    def _l2_gradient(self):
        gradient_w = - 2 * self.weights
        gradient_b = - 2 * self.bias
        return gradient_w, gradient_b
    

    def train(self, inputs, outputs, lr=0.1, l2=0):
        gradient_w, gradient_b = self._mse_gradient(inputs,outputs)
        self.weights += lr * gradient_w
        self.bias += lr * gradient_b
        if l2 > 0:
            l2_gradient_w,  l2_gradient_b = self.l2_gradient()
            self.weights += lr * l2 * l2_gradient_w
            self.bias += lr * l2 * l2_gradient_b
        return self._mse_loss(inputs, outputs)

    def save_model(self, path):
        f = open(path, 'w')
        for i in self.weights:
            f.write(str(i)+",")
        f.write(str(self.bias[0]))
        f.close()

    def load_model(self, path):
        f = open(path, 'r')
        weights = f.readline()
        weights = weights.split(',')
        self.weights = np.array(weights[:-1]).astype('float64')
        self.bias = np.array(weights[-1]).astype('float64')
        f.close()

if __name__ == "__main__":
    l = Linear_model(10)
    data_num = 100
    inputs = np.array([np.random.normal(0, 1, 10) for i in range(data_num)])
    outputs = np.zeros([data_num])
    for i in range(10):
        l.train(inputs, outputs)
    test_output = l._output(inputs)[0]
    f = open('test1', 'w')
    f.write(str(test_output))
    f.close()
    l.save_model('test.model')
    l.load_model('test.model')
    test_output = l._output(inputs)[0]
    f = open('test2', 'w')
    f.write(str(test_output))
    f.close()
