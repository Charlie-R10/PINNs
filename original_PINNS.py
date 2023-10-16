import tensorflow as tf
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


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
    # Coefficients/constants (change if necessary)
    D = 1.0  # Diffusion coefficient
    Sigma_a = 0.5  # Absorption cross-section
    nu_Sigma_f = 0.2  # Fission-related term
    S = 0.0  # Source term - started with no net neutron source

    x = tf.constant(x_train, dtype=tf.float32)

    # Compute the first and the second derivatives of phi_pred with respect to x
    with tf.GradientTape() as t1:
        t1.watch(x)
        with tf.GradientTape() as t2:
            t2.watch(x)
            phi_pred = model(x)
        dphi_dx = t2.gradient(phi_pred, x)
    d2phi_dx2 = t1.gradient(dphi_dx, x)

    if d2phi_dx2 is not None:
        print("d2_phi_dx2:", d2phi_dx2.numpy())
    else:
        print("Second derivative = None")

    # Residual loss (from eq. - simplify?)
    L_residual = -D * d2phi_dx2 + Sigma_a * phi_pred - nu_Sigma_f * phi_pred - S

    # Boundary condition loss

    # Neumann boundary condition at x = 0 (left)
    L_residual += D * dphi_dx

    # Neumann boundary condition at x = L (right)
    L_residual -= D * dphi_dx

    # Reflective boundary condition at x = 0 (left) --- Should this be derivative or simply L_residual += phi_pred
    L_residual += 2.0 * D * dphi_dx

    # Reflective boundary condition at x = L (right)
    L_residual -= 2.0 * D * dphi_dx

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
S = 0.0           # Source term
L = 10.0          # Length of the domain

# Define the number of training points and the domain discretization
num_points = 1000
x = np.linspace(0, L, num_points)  # Spatial domain

# Generate training data - replace this with solution
def generate_training_data(x):
    # Calculate the true neutron flux based on the diffusion equation --- need to insert actual solution for neutron flux just used sin for now
    phi_true = np.exp(x)

    return x, phi_true

x_train, phi_train_true = generate_training_data(x)

# added noise for simulation sake
noise_level = 0.01
noise = noise_level * np.random.randn(num_points)
phi_train_noisy = phi_train_true + noise


# Train the model
num_epochs = 10  # YChange if necessary ~ did 1000 took roughly 1.5 hrs set as 10 for now
batch_size = 32

history = model.fit(x_train, phi_train_noisy, epochs=num_epochs, batch_size=batch_size)


print(history)
