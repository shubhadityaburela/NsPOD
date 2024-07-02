import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import meshgrid
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.autograd import gradcheck
from numpy import exp, mod, meshgrid, cos, sin, exp, pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from math import sqrt
from scipy.sparse import diags
from scipy.linalg import cholesky

import os

impath = "./data/Crossing_waves/mac/"
immpath = "./plots/Crossing_waves/mac/"
os.makedirs(impath, exist_ok=True)
os.makedirs(immpath, exist_ok=True)


seed = 54

if torch.cuda.is_available():
    print("The current device: ", torch.cuda.current_device())
    print("Name of the device: ", torch.cuda.get_device_name(0))
    print("Number of GPUs available: ", torch.cuda.device_count())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


def generate_data(Nx, Nt, coeff1, coeff2, center1, center2, sigma_slope=0.1, sigma_base=4.0):
    sigma = 5.0
    x = np.arange(0, Nx)
    t = np.linspace(-10, 10, Nt)
    t_max = np.max(t)
    [X, T] = np.meshgrid(x, t)
    X = X.T
    T = T.T

    def gaussian(x, mu, sigma=1.0):
        return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sigma, 2.0)))

    q1 = np.zeros_like(X, dtype=np.float64)
    q2 = np.zeros_like(X, dtype=np.float64)
    shift1 = np.polyval(coeff1, t)
    shift2 = np.polyval(coeff2, t)
    for col in range(Nt):
        sigma_t = sigma_base
        for row in range(Nx):
            q1[row, col] = sin(pi * t[col] / t_max) * gaussian(row, center1 + shift1[col], sigma_t)
            q2[row, col] = cos(pi * t[col] / t_max) * gaussian(row, center2 + shift2[col], sigma_t)

    Q = q1 + q2
    Q /= Q.max()

    return Q, x, t


class NuclearNormAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_matrix):
        ctx.save_for_backward(input_matrix)
        return torch.linalg.matrix_norm(input_matrix, ord="nuc")

    @staticmethod
    def backward(ctx, grad_output):
        input_matrix, = ctx.saved_tensors
        u, s, v = torch.svd(input_matrix, some=False)
        rank = torch.sum(s > 0).item()
        dtype = input_matrix.dtype
        eye_approx = torch.diag((s > 0).to(dtype)[:rank])
        grad_input = torch.matmul(torch.matmul(u[:, :rank], eye_approx), v[:, :rank].t())
        return grad_input * grad_output.unsqueeze(-1).unsqueeze(-1)


class ShapeShiftNet(nn.Module):
    def __init__(self):
        super(ShapeShiftNet, self).__init__()
        self.elu = nn.ELU()

        # Subnetwork for f^1 and shift^1
        self.f1_fc1 = nn.Linear(2, 5)
        self.f1_fc2 = nn.Linear(5, 10)
        self.f1_fc3 = nn.Linear(10, 5)
        self.f1_fc4 = nn.Linear(5, 1)

        self.shift1_fc1 = nn.Linear(1, 5)
        self.shift1_fc2 = nn.Linear(5, 5)
        self.shift1_fc3 = nn.Linear(5, 1)

        # Subnetwork for f^2 and shift^2
        self.f2_fc1 = nn.Linear(2, 5)
        self.f2_fc2 = nn.Linear(5, 10)
        self.f2_fc3 = nn.Linear(10, 5)
        self.f2_fc4 = nn.Linear(5, 1)

        self.shift2_fc1 = nn.Linear(1, 5)
        self.shift2_fc2 = nn.Linear(5, 5)
        self.shift2_fc3 = nn.Linear(5, 1)

    def forward(self, x, t):
        # Pathway for f^1 and shift^1
        shift1 = self.elu(self.shift1_fc1(t))
        shift1 = self.elu(self.shift1_fc2(shift1))
        shift1 = self.shift1_fc3(shift1)

        x_shifted1 = x + shift1
        f1 = self.elu(self.f1_fc1(torch.cat((x_shifted1, t), dim=1)))
        f1 = self.elu(self.f1_fc2(f1))
        f1 = self.elu(self.f1_fc3(f1))
        f1 = self.f1_fc4(f1)

        f1_without_shift = self.elu(self.f1_fc1(torch.cat((x, t), dim=1)))
        f1_without_shift = self.elu(self.f1_fc2(f1_without_shift))
        f1_without_shift = self.elu(self.f1_fc3(f1_without_shift))
        f1_without_shift = self.f1_fc4(f1_without_shift)

        # Pathway for f^2 and shift^2
        shift2 = self.elu(self.shift2_fc1(t))
        shift2 = self.elu(self.shift2_fc2(shift2))
        shift2 = self.shift2_fc3(shift2)

        x_shifted2 = x + shift2
        f2 = self.elu(self.f2_fc1(torch.cat((x_shifted2, t), dim=1)))
        f2 = self.elu(self.f2_fc2(f2))
        f2 = self.elu(self.f2_fc3(f2))
        f2 = self.f2_fc4(f2)

        f2_without_shift = self.elu(self.f2_fc1(torch.cat((x, t), dim=1)))
        f2_without_shift = self.elu(self.f2_fc2(f2_without_shift))
        f2_without_shift = self.elu(self.f2_fc3(f2_without_shift))
        f2_without_shift = self.f2_fc4(f2_without_shift)

        return f1, f2, shift1, shift2, f1_without_shift, f2_without_shift, shift1, shift2

############################## Crossing waves example ##################################


# Create data for example model
Nx, Nt, sigma = 400, 200, 4.0
coefficients1 = [0.15, 0, 0.8, 1.5]
coefficients2 = [-18, 2]
center_matrix3 = 200

# Set the seed
np.random.seed(seed)
torch.manual_seed(seed)

# Generate the data
Q, x, t = generate_data(Nx, Nt, coefficients1, coefficients2, center_matrix3, center_matrix3, sigma_base=sigma)

# Define the inputs
inputs = np.stack([x.repeat(Nt), np.tile(t, Nx)], axis=1)
inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(DEVICE)
Q_tensor = torch.tensor(Q).to(DEVICE)

# Instantiate the model
model = ShapeShiftNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move the model to GPU
model.to(DEVICE)

# Training loop starts
num_epochs = 10
lambda_k = 0.5

for epoch in range(num_epochs + 1):
    x_flat, t_flat = inputs_tensor[:, 0:1], inputs_tensor[:, 1:2]
    optimizer.zero_grad()
    f1_full, f2_full, shift1_pred, shift2_pred, f1_full_nos, f2_full_nos, s1, s2 = model(x_flat, t_flat)

    frobenius_loss = torch.norm(Q_tensor - f1_full.view(Nx, Nt) - f2_full.view(Nx, Nt), 'fro') ** 2

    nuclear_loss_q1 = NuclearNormAutograd.apply(f1_full_nos.view(Nx, Nt))
    nuclear_loss_q2 = NuclearNormAutograd.apply(f2_full_nos.view(Nx, Nt))
    nuclear_loss = lambda_k * (nuclear_loss_q1 + nuclear_loss_q2)

    total_loss = frobenius_loss + nuclear_loss

    total_loss.backward(retain_graph=True)
    optimizer.step()

    if frobenius_loss < 1:
        print("Early stopping is triggered")
        break

    if epoch % 100 == 0:
        print(
            f'Epoch {epoch}/{num_epochs}, Frob Loss: {frobenius_loss.item()}, Nuclear Loss: {nuclear_loss.item()}, Total loss: {total_loss.item()},')

combined = f1_full + f2_full
Q_tilde = combined.view(Nx, Nt).detach().numpy()

# Save the weights
torch.save(model.state_dict(), impath + 'Crossing_waves.pth')

# Plot the results
X, T = np.meshgrid(x, t)
fig, axs = plt.subplots(1, 6, figsize=(20, 6))
vmin = np.min(Q)
vmax = np.max(Q)

axs[0].pcolormesh(X.T, T.T, Q, vmin=vmin, vmax=vmax)
axs[0].set_title(r"${\mathbf{Q}}$")
axs[0].set_xlabel("x")
axs[0].set_ylabel("t")
axs[0].set_xticks([])
axs[0].set_yticks([])

#Qtilde
axs[1].pcolormesh(X.T, T.T, Q_tilde, vmin=vmin, vmax=vmax)
axs[1].set_title(r"$\tilde{\mathbf{Q}}$")
axs[1].set_xlabel("x")
axs[1].set_ylabel("t")
axs[1].set_xticks([])
axs[1].set_yticks([])

# f^1
axs[2].pcolormesh(X.T, T.T, f1_full.view(Nx, Nt).detach().numpy(), vmin=vmin)
axs[2].set_title(r"$\mathcal{T}^1\mathbf{Q}^1$")
axs[2].set_xlabel("x")
axs[2].set_ylabel("t")
axs[2].set_xticks([])
axs[2].set_yticks([])

# f^2
axs[3].pcolormesh(X.T, T.T, f2_full.view(Nx,Nt).detach().numpy(), vmin=vmin, vmax=vmax)
axs[3].set_title(r"$\mathcal{T}^2\mathbf{Q}^2$")
axs[3].set_xlabel("x")
axs[3].set_ylabel("t")
axs[3].set_xticks([])
axs[3].set_yticks([])

# f^1
axs[4].pcolormesh(X.T, T.T, f1_full_nos.view(Nx,Nt).detach().numpy(), vmin=vmin, vmax=vmax)
axs[4].set_title(r"$\mathbf{Q}^1$")
axs[4].set_xlabel("x")
axs[4].set_ylabel("t")
axs[4].set_xticks([])
axs[4].set_yticks([])

# f^2
cax4 = axs[5].pcolormesh(X.T, T.T, f2_full_nos.view(Nx,Nt).detach().numpy(), vmin=vmin, vmax=vmax)
axs[5].set_title(r"$\mathbf{Q}^2$")
axs[5].set_xlabel("x")
axs[5].set_ylabel("t")
axs[5].set_xticks([])
axs[5].set_yticks([])

plt.colorbar(cax4, ax=axs.ravel().tolist(), orientation='vertical')

fig.savefig(immpath + "Crossing_waves_NN", dpi=300, transparent=True)

