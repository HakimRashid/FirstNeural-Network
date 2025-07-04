import os
import numpy as np
import matplotlib.pyplot as plt

#file loading logic 
if not os.path.exists('weights.npz'):
    weights = np.array([[2.1, 5.4],
                        [3.1, 6.4],
                        [7.8, 9.1],
                        [4, 0.4]])
    output_weights = np.array([4.1, 2.1, 9.2, 7.8])
    bias = np.array([1, 2, 3, 4])
    np.savez('weights.npz', weights=weights, output_weights=output_weights, bias=bias)
data = np.load('weights.npz')
weights = data['weights']
output_weights = data['output_weights']
bias = data['bias']


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

z_hidden = np.dot(inputs, weights.T) + bias

outputs_H = np.array(sigmoid(z_hidden))

z_output = np.dot(outputs_H, output_weights.T) + 5.1

prediction = sigmoid(z_output)

print(prediction)

#backpropagation starts here
loss = np.mean(-(target*np.log(prediction) + (1-target)*np.log(1-prediction)))

error = prediction - target
d_out = error * prediction * (1 - prediction)

d_output_weights = np.dot(outputs_H.T, d_out)

blames_hidden = np.array(d_out[:, np.newaxis] * output_weights * outputs_H * (1 - outputs_H))

d_weights = np.dot(blames_hidden.T, inputs)

weights -= learning_rate * d_weights

output_weights -= learning_rate * d_output_weights

np.savez('weights.npz', weights=weights, output_weights=output_weights, bias=bias)