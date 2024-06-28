import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import meshgrid
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.autograd import gradcheck
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from math import sqrt
from scipy.sparse import diags
from scipy.linalg import cholesky

import os
inpath = "./data/wildlandfire/input/"
impath = "./data/wildlandfire/seed=1/"
immpath = "./plots/wildlandfire/seed=1/"
os.makedirs(impath, exist_ok=True)
os.makedirs(immpath, exist_ok=True)



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

        # Subnetwork for f^3 and shift^3
        self.f3_fc1 = nn.Linear(2, 5)
        self.f3_fc2 = nn.Linear(5, 10)
        self.f3_fc3 = nn.Linear(10, 5)
        self.f3_fc4 = nn.Linear(5, 1)

        self.shift3_fc1 = nn.Linear(1, 5)
        self.shift3_fc2 = nn.Linear(5, 5)
        self.shift3_fc3 = nn.Linear(5, 1)

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

        # Pathway for f^3 and shift^3
        shift3 = self.elu(self.shift3_fc1(t))
        shift3 = self.elu(self.shift3_fc2(shift3))
        shift3 = self.shift3_fc3(shift3)

        x_shifted3 = x + shift3
        f3 = self.elu(self.f3_fc1(torch.cat((x_shifted3, t), dim=1)))
        f3 = self.elu(self.f3_fc2(f3))
        f3 = self.elu(self.f3_fc3(f3))
        f3 = self.f3_fc4(f3)

        f3_without_shift = self.elu(self.f3_fc1(torch.cat((x, t), dim=1)))
        f3_without_shift = self.elu(self.f3_fc2(f3_without_shift))
        f3_without_shift = self.elu(self.f3_fc3(f3_without_shift))
        f3_without_shift = self.f3_fc4(f3_without_shift)

        return f1, f2, f3, f1_without_shift, f2_without_shift, f3_without_shift, shift1, shift2, shift3


# Load the data
Q_wf = np.load(inpath + 'SnapShotMatrix558.49.npy', allow_pickle=True)
t = np.load(inpath + 'Time.npy', allow_pickle=True)
x_grid = np.load(inpath + '1D_Grid.npy', allow_pickle=True)
x = x_grid[0]
T = Q_wf[:len(x), :]
seed = 10
torch.manual_seed(seed)


Q = T/T.max()
Q_tensor = torch.tensor(Q)
Nx = len(x)
Nt = len(t)
xx, tt = np.meshgrid(x, t)


# Inputs to the network
inputs = np.stack([x.repeat(Nt), np.tile(t, Nx)], axis=1)
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)


# Define the model
model = ShapeShiftNet()
pretrained_load = True
if pretrained_load:
    state_dict_original = torch.load("./data/crossing_waves_sine_amplitude/seed=54/model_weights_crossing_sine_waves.pth")

    state_dict_new = model.state_dict()

    for name, param in state_dict_original.items():
        if name in state_dict_new:
            state_dict_new[name].copy_(param)

model.load_state_dict(state_dict_new, strict=False)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 1000
lambda_k = 0.1
for epoch in range(num_epochs + 1):
    x_NN, t_NN = inputs_tensor[:, 0:1], inputs_tensor[:, 1:2]

    optimizer.zero_grad()
    f1_full, f2_full, f3_full, f1_full_nos, f2_full_nos, f3_full_nos, s1, s2, s3 = model(x_NN, t_NN)

    frobenius_loss = torch.norm(Q_tensor - f1_full.view(Nx, Nt) - f2_full.view(Nx, Nt) - f3_full.view(Nx, Nt), 'fro') ** 2

    nuclear_loss_q1 = NuclearNormAutograd.apply(f1_full_nos.view(Nx, Nt))
    nuclear_loss_q2 = NuclearNormAutograd.apply(f2_full_nos.view(Nx, Nt))
    nuclear_loss_q3 = NuclearNormAutograd.apply(f3_full_nos.view(Nx, Nt))
    nuclear_loss = lambda_k * (nuclear_loss_q1 + nuclear_loss_q2 + nuclear_loss_q3)

    total_loss = nuclear_loss + frobenius_loss

    total_loss.backward(retain_graph=True)
    optimizer.step()

    if frobenius_loss < 1.0:
        print("Early stopping is triggered")
        break

    if epoch % 100 == 0:
        print(
            f'Epoch {epoch}/{num_epochs}, Frob Loss: {frobenius_loss.item()}, Nuclear Loss: {nuclear_loss.item()}, Total loss: {total_loss.item()},')


combined = f1_full + f2_full + f3_full
Q_tilde = combined.view(Nx, Nt).detach().numpy()


fig, axs = plt.subplots(1, 8, figsize=(20, 4))
vmin = np.min(Q)
vmax = np.max(Q)

# Q
axs[0].pcolormesh(xx.T, tt.T, Q, vmin=vmin, vmax=vmax)
axs[0].set_title(r"$\mathbf{Q}$")
axs[0].set_xlabel("t")
axs[0].set_ylabel("x")
axs[0].set_xticks([])
axs[0].set_yticks([])

# Qtilde
axs[1].pcolormesh(xx.T, tt.T, Q_tilde, vmin=vmin, vmax=vmax)
axs[1].set_title(r"$\mathbf{\tilde{Q}}$")
axs[1].set_xlabel("t")
axs[1].set_ylabel("x")
axs[1].set_xticks([])
axs[1].set_yticks([])

# f^1
axs[2].pcolormesh(xx.T, tt.T, f1_full.view(Nx, Nt).detach().numpy(), vmin=vmin, vmax=vmax)
axs[2].set_title(r"$\mathcal{T}^1\mathbf{Q}^1$")
axs[2].set_xlabel("t")
axs[2].set_ylabel("x")
axs[2].set_xticks([])
axs[2].set_yticks([])

# f^3
axs[3].pcolormesh(xx.T, tt.T, f3_full.view(Nx,Nt).detach().numpy(), vmin=vmin, vmax=vmax)
axs[3].set_title(r"$\mathcal{T}^2\mathbf{Q}^2$")
axs[3].set_xlabel("t")
axs[3].set_ylabel("x")
axs[3].set_xticks([])
axs[3].set_yticks([])

# f^2
axs[4].pcolormesh(xx.T, tt.T, f2_full.view(Nx, Nt).detach().numpy(), vmin=vmin, vmax=vmax)
axs[4].set_title(r"$\mathcal{T}^3\mathbf{Q}^3$")
axs[4].set_xlabel("t")
axs[4].set_ylabel("x")
axs[4].set_xticks([])
axs[4].set_yticks([])


# f^1
axs[5].pcolormesh(xx.T, tt.T, f1_full_nos.view(Nx,Nt).detach().numpy(), vmin=vmin, vmax=vmax)
axs[5].set_title(r"$\mathbf{Q}^1$")
axs[5].set_xlabel("t")
axs[5].set_ylabel("x")
axs[5].set_xticks([])
axs[5].set_yticks([])

# f^3
axs[6].pcolormesh(xx.T, tt.T, f3_full_nos.view(Nx,Nt).detach().numpy(), vmin=vmin, vmax=vmax)
axs[6].set_title(r"$\mathbf{Q}^2$")
axs[6].set_xlabel("t")
axs[6].set_ylabel("x")
axs[6].set_xticks([])
axs[6].set_yticks([])

# f^2
cax4 = axs[7].pcolormesh(xx.T, tt.T, f2_full_nos.view(Nx,Nt).detach().numpy(), vmin=vmin, vmax=vmax)
axs[7].set_title(r"$\mathbf{Q}^3$")
axs[7].set_xlabel("t")
axs[7].set_ylabel("x")
axs[7].set_xticks([])
axs[7].set_yticks([])

plt.colorbar(cax4, ax=axs.ravel().tolist(), orientation='vertical')
fig.savefig(immpath + "NN-prediction", dpi=300, transparent=True)
torch.save(model.state_dict(), impath + 'model_weights_wildfire_1d.pth')
