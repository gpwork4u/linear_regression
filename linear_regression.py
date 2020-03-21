import numpy as np
import sys
class Linear_model:
    
    def __init__(self, num):
        self.weights = np.random.normal(0, 1, num)
        self.bias = np.random.normal(0, 1, 1) 
    def _output(self, inputs):
        out = np.matmul(self.weights.transpose(), inputs) 
        out += self.bias
        return out

    def _loss(self, inputs, outputs):
        loss = np.mean(np.square(self._output(inputs)-outputs))
        return loss
    
    def _gradient(self, inputs, outputs):
        pred_outputs = self._output(inputs)
        gradient_w = - np.mean(2 * (pred_outputs - outputs) * inputs, axis=0)
        gradient_b = - np.mean(2 * (pred_outputs - outputs), axis=0)
        return gradient_w, gradient_b
    
    def train(self,inputs,outputs, lr):
        print("loss:%4f output:%4f"%(self._loss(inputs,outputs), self._output(inputs)[0]))
        gradient_w, gradient_b = self._gradient(inputs,outputs)
        self.weights += lr * gradient_w
        self.bias += lr * gradient_b
        
if __name__ == "__main__":
    l = Linear_model(10)
    inputs = np.array([np.random.normal(0, 1, 1) for i in range(10)])
    outputs = np.ones([10])
    print(l._gradient(inputs, outputs))
    for i in range(100):
        l.train(inputs, outputs ,float(sys.argv[1]))
    print(l._output(inputs)[0])
