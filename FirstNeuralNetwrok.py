import os
import numpy as np
import matplotlib.pyplot as plt

#file loading logic 
if not os.path.exists('weights.npz'):
    weights = np.random.randn(4, 2) * 0.001
    output_weights = np.random.randn(4) * 0.001
    bias = np.zeros(4)
    bias_output = 0.0
    np.savez('weights.npz', weights=weights, output_weights=output_weights, bias=bias, bias_output=bias_output)
data = np.load('weights.npz')
weights = data['weights']
output_weights = data['output_weights']
bias = data['bias']
bias_output = data['bias_output']
training_inputs = np.array([[180.89, 70.83],
                [165.23, 47.93],
                [172.55, 60.19],
                [158.76, 75.24],
                [190.12, 91.56],
                [170.05, 49.33],
                [175.00, 75.87],
                [162.19, 79.54],
                [185.34, 82.01],
                [153.45, 43.08]])

train = input("Would you like to train me(Y/N)?")
if train == 'Y':
        training = True
elif train == 'N':
        training = False
 
if training:
        inputs = training_inputs
else:
        inputs = [float(input("\nEnter your height(cm):")), float(input("\nEnter your weight(kg):"))]
target = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
learning_rate = 0.1

#forwardpass begins here
inputs = (inputs - np.min(training_inputs, axis=0)) / (np.max(training_inputs, axis=0) - np.min(training_inputs, axis=0))

def sigmoid(x: float):
        return 1/(1+np.exp(-x))

z_hidden = np.dot(inputs, weights.T) + bias

outputs_H = sigmoid(z_hidden)

z_output = np.dot(outputs_H, output_weights) + bias_output

prediction = sigmoid(z_output)


#backpropagation starts here

if training:
        for epoch in range(100000):
                z_hidden = np.dot(inputs, weights.T) + bias
                outputs_H = np.array(sigmoid(z_hidden))
                z_output = np.dot(outputs_H, output_weights) + bias_output
                prediction = sigmoid(z_output)
                
                prediction = np.clip(prediction, 1e-7, 1 - 1e-7)
                loss = np.mean(-(target*np.log(prediction) + (1-target)*np.log(1-prediction)))

                error = prediction - target
                d_out = error * prediction * (1 - prediction)
                d_output_weights = np.dot(outputs_H.T, d_out)
                blames_hidden = np.array(d_out[:, np.newaxis] * output_weights * outputs_H * (1 - outputs_H))
                d_weights = np.dot(blames_hidden.T, inputs)
                d_bias = np.sum(blames_hidden, 0)
                d_bias_output = np.sum(d_out)

                weights -= learning_rate * d_weights
                output_weights -= learning_rate * d_output_weights
                bias -= learning_rate * d_bias
                bias_output -= learning_rate * d_bias_output
                
                if epoch % 100 == 0:
                        np.savez('weights.npz', weights=weights, output_weights=output_weights, bias=bias, bias_output=bias_output)

#matlib plots
if training:
        plt.plot(prediction, label='Prediction')
        plt.plot(target, label='Target')
        plt.legend()
        plt.show()
else:
        if round(prediction) == 1:
                print("You are healthy!")
        elif round(prediction) == 0:
                print("You are not healthy!")