import sys
import numpy as np
import matplotlib.pyplot as plt

weights = np.array([[2.1, 5.4],
            [3.1, 6.4],
            [7.8, 9.1],
            [4, 0.4]])
output_weights = np.array([4.1, 2.1, 9.2, 7.8])
bias = np.array([1, 2, 3, 4])
inputs = np.array([[180.89, 70.83],
        [165.23, 47.93],
        [172.55, 60.19],
        [158.76, 75.24],
        [190.12, 91.56],
        [170.05, 49.33],
        [175.00, 75.87],
        [162.19, 79.54],
        [185.34, 82.01],
        [153.45, 43.08]])
target = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
learning_rate = 0.01
training = False

#forwardpass begins here
def sigmoid(x: float):
        return 1/(1+np.exp(-x))

if inputs.shape[0] != weights.shape[1]:
        z_hidden = np.dot(inputs, weights.T) + bias
else:
        z_hidden = np.dot(weights, inputs) + bias

outputs_H = np.array(sigmoid(z_hidden))

if outputs_H.shape[0] != output_weights.shape[1]:
        z_output = np.dot(outputs_H, output_weights.T) + 5.1
else:
        z_output = np.dot(output_weights, outputs_H) + 5.1

prediction = sigmoid(z_output)

print(prediction)

#backpropagation starts here
loss = -(target*np.log(prediction) + (1-target)*np.log(1-prediction))

error = prediction - target
d_out = error * prediction * (1 - prediction)

d_output_weights = np.dot(outputs_H.T, d_out)

blames_hidden = np.array(d_out[:, np.newaxis] * output_weights * outputs_H * (1 - outputs_H))

blames_output = np.dot(blames_hidden.T, inputs)

weights -= learning_rate * blames_hidden

output_weights -= learning_rate * blames_output