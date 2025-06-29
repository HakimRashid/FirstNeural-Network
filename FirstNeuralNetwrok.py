import sys
import numpy as np
import matplotlib.pyplot as plt

weights = [[2.1, 5.4],
            [3.1, 6.4],
            [7.8, 9.1],
            [4, 0.4]]
output_weights = np.array([4.1, 2.1, 9.2, 7.8])
bias = np.array([1, 2, 3, 4])
inputs = [175, 100]
target = 0
learning_rate = 0.01

#forwardpass begins here
def sigmoid(x: float):
        return 1/(1+np.exp(-x))

z_hidden = np.dot(weights, inputs) + bias
outputs_H = np.array([sigmoid(x) for x in z_hidden])

z_output = np.dot(output_weights, outputs_H) + 5.1
prediction = sigmoid(z_output)

print(prediction)

#backpropagation starts here
loss = -(target*np.log(prediction) + (1-target)*np.log(1-prediction))

blames = np.array([(prediction - target) * prediction * (1 - prediction) * weight * output * (1 - output) for weight, output in zip(output_weights, outputs_H)])
weight_adjustments = np.outer(blames, inputs)

weights = weights - learning_rate * weight_adjustments
