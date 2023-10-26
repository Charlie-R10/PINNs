import tensorflow as tf
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Want to review the layers and also activation function

def diffusion_PINN():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, activation='tanh', input_shape=(1,))
    ])

    # Number of hidden layers
    for i in range(6):
        model.add(tf.keras.layers.Dense(20, activation='tanh'))

    # Output layer
    model.add(tf.keras.layers.Dense(1, activation='linear', name='output_layer'))

    return model

# Create an instance of model
model = diffusion_PINN()


# Loss function
def loss_function(y_true, y_pred):
    # Coefficients/constants (change if necessary) - maybe should be defined globally
    D = 1.0  # Diffusion coefficient
    Sigma_a = 0.5  # Absorption cross-section
    nu_Sigma_f = 0.2  # Fission-related term
    S = 2.0  # Source term - started with no net neutron source

    x = tf.constant(x_train, dtype=tf.float32)

    # Compute the first and the second derivatives of phi_pred with respect to x
    with tf.GradientTape() as t1:
        t1.watch(x)
        with tf.GradientTape() as t2:
            t2.watch(x)
            phi_pred = model(x)
        dphi_dx = t2.gradient(phi_pred, x)
    d2phi_dx2 = t1.gradient(dphi_dx, x)

    # Residual loss (from eq. - simplify?)
    L_residual = -D * d2phi_dx2 + Sigma_a * phi_pred - nu_Sigma_f * phi_pred - S

    # Boundary condition loss

    # Boundary condition phi(a) = 0
    L_residual += phi_pred

    # J = derivative of phi (current)
    # This is boundary condition J+=1/2*S
    L_residual += dphi_dx - 0.5 * S

    # MSE of residual
    mse_residual = tf.reduce_mean(tf.square(L_residual))

    return mse_residual

model = diffusion_PINN()

# Compile model
model.compile(loss=loss_function, optimizer='adam', run_eagerly=True)


# Define parameters for the problem
D = 1.0           # Diffusion coefficient
Sigma_a = 0.5     # Absorption cross-section
nu_Sigma_f = 0.2  # Fission-related term
S = 2.0           # Source term - set as 2 for now
L = 10.0          # Length of the domain

# x is number of points for data (10 evenly spaced 1m apart for now)
num_points = 10
x = np.linspace(0, L, num_points)

# Generate training data
def generate_training_data(x):
    # Calculate the true neutron flux based on the diffusion equation pg 51 in Stacey
    phi_true = 4*S*((np.sinh(x)/L)/(np.sinh(x/L)+((2*D)/L)*np.cosh(x/L)))

    return x, phi_true

x_train, phi_train_true = generate_training_data(x)



# Train the model
num_epochs = 1000  # YChange if necessary ~ did 1000 took roughly 1.5 hrs set as 10 for now
batch_size = 32

history = model.fit(x_train, phi_train_true, epochs=num_epochs, batch_size=batch_size)


def loss_plotter():
    plt.plot(history.history['loss'])
    plt.title('Model Loss over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.show()

loss_plotter()


