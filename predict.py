import numpy as np

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return A2

def load_model():
    data = np.load("model/weights.npz")
    return data["W1"], data["b1"], data["W2"], data["b2"]

def predict_digit(img_vector):
    W1, b1, W2, b2 = load_model()
    X = img_vector.reshape(784, 1)
    X = X / 255.
    A2 = forward_prop(W1, b1, W2, b2, X)
    prediction = np.argmax(A2, axis=0)[0]
    return int(prediction)