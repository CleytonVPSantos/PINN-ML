import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat


# Verifica se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# arquitetura da rede neural: 3 camadas hidden layers com 100 neuronios cada
class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.layer1(x))
        x = self.tanh(self.layer2(x))
        x = self.tanh(self.layer3(x))
        x = self.output_layer(x)
        return x


def loss_function(model, x_init, t_init, u_init, x_boundary, t_boundary, u_boundary, x_collocation, t_collocation, alpha):
    # Condições iniciais
    u_initial_pred = model(torch.hstack((x_init, t_init)))
    mse_initial = torch.mean((u_initial_pred - u_init)**2)

    # Condições de fronteira
    u_boundary_pred = model(torch.hstack((x_boundary, t_boundary)))
    mse_boundary = torch.mean((u_boundary_pred - u_boundary)**2)

    # Rastreia as operações sobre os tensores, permitindo autodiff
    x_collocation.requires_grad = True
    t_collocation.requires_grad = True

    u = model(torch.hstack((x_collocation, t_collocation)))

    u_t = torch.autograd.grad(u, t_collocation, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_collocation, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_collocation, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    f_pred = u_t - alpha * u_xx
    mse_pde = torch.mean(f_pred **2)

    x_collocation.requires_grad = False
    t_collocation.requires_grad = False

    # Loss total
    loss = mse_initial + mse_pde + mse_boundary
    return loss


# Parâmetros
input_size = 2
hidden_size = 100
output_size = 1
model = PINN(input_size, hidden_size, output_size).to(device)
alpha = 0.05 # parametro da eq. do calor

# Quantidade de exemplos de treinamento
n_boundary = 40
n_initial = 40
n_collocation = 1000

# Dados para treinamento
sampler = scipy.stats.qmc.LatinHypercube(d=1) # metodo Latin Hypercube

x_init = torch.Tensor(sampler.random(n=n_initial)).to(device)
t_init = torch.zeros_like(x_init).to(device)
u_init = torch.sin(2 * torch.pi * x_init).to(device)

x_boundary = torch.cat([torch.zeros(n_boundary), torch.ones(n_boundary)]).reshape(-1, 1).to(device)
t_boundary = torch.Tensor(sampler.random(n=n_boundary)).repeat(2, 1).to(device)
u_boundary = torch.zeros_like(t_boundary).to(device)

sampler = scipy.stats.qmc.LatinHypercube(d=2)
collocation = torch.Tensor(sampler.random(n_collocation))
x_collocation = collocation[:,0].reshape(-1,1).to(device)
t_collocation = collocation[:,1].reshape(-1,1).to(device)

# Otimizador
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01, max_iter=50000, max_eval=50000, history_size=50)

# Função de fechamento necessária para L-BFGS
def closure():
    optimizer.zero_grad()
    loss = loss_function(model, x_init, t_init, u_init, x_boundary, t_boundary, u_boundary, x_collocation, t_collocation, alpha)
    loss.backward()
    return loss

# Treinamento
model.train()

for epoch in range(50):
    loss = optimizer.step(closure)
    print(f"Epoca {epoch}, Loss: {loss.item()}")


model.eval()

# Configuração dos plots
fig, axes = plt.subplots(1, 2, figsize=(20, 15))
fig.suptitle('Eq. do calor 1D com PINN')

# PLOT 1
time = 0.0

with torch.no_grad():
    x_plot = torch.linspace(0, 1, 100).reshape(-1, 1).to(device)
    t_plot = torch.tensor([time]).expand_as(x_plot).to(device)  # Solução no tempo t = 1.0
    u_plot = model(torch.cat((x_plot, t_plot), dim=1))

# Plotando a solução
axes[0].plot(x_plot.cpu().numpy(), u_plot.cpu().numpy(), label='Predicted Solution (PINN)', color='blue')
axes[0].set_xlabel('x')
axes[0].set_ylabel(f'u(x, t={time})')
axes[0].set_ylim(-1.2, 1.2)
axes[0].set_title(f'Solução no tempo t = {time}')
axes[0].legend()
axes[0].set_position([0.1, 0.55, 0.35, 0.35])  # Ajustando o tamanho do gráfico

# PLOT 2
time = 0.5

with torch.no_grad():
    x_plot = torch.linspace(0, 1, 100).reshape(-1, 1).to(device)
    t_plot = torch.tensor([time]).expand_as(x_plot).to(device)  # Solução no tempo t = 1.0
    u_plot = model(torch.cat((x_plot, t_plot), dim=1))

# Plotando a solução
axes[1].plot(x_plot.cpu().numpy(), u_plot.cpu().numpy(), label='Predicted Solution (PINN)', color='blue')
axes[1].set_xlabel('x')
axes[1].set_ylabel(f'u(x, t={time})')
axes[1].set_ylim(-1.2, 1.2)
axes[1].set_title(f'Solução no tempo t = {time}')
axes[1].legend()
axes[1].set_position([0.55, 0.55, 0.35, 0.35])  # Ajustando o tamanho do gráfico


# Visualização e plot
fig, axes = plt.subplots(1, 2, figsize=(20, 15), subplot_kw={"projection": "3d"})

model.eval()
with torch.no_grad():
    x_plot = torch.linspace(0, 1, 100).to(device)
    t_plot = torch.linspace(0, 1, 100).to(device)
    x_plot, t_plot = torch.meshgrid(x_plot, t_plot, indexing='ij')
    u_plot = model(torch.cat((x_plot.reshape(-1, 1), t_plot.reshape(-1, 1)), dim=1))
    u_plot = u_plot.reshape(100, 100)

# Plotando a solução do modelo PINN
axes[0].plot_surface(x_plot.cpu().numpy(), t_plot.cpu().numpy(), u_plot.cpu().numpy(), cmap='viridis')
axes[0].set_xlabel('x')
axes[0].set_ylabel('t')
axes[0].set_zlabel('u(x, t)')
axes[0].set_title('Eq. do calor 1D com PINN (alpha = 0.05)')

# Parâmetros da equação do calor para a solução numérica
alpha = 0.05
L = 1.0  # comprimento do domínio
T = 1.0  # tempo total de simulação
nx = 300  # número de pontos no espaço
nt = 10000  # número de passos de tempo
dx = L / (nx - 1)  # espaçamento espacial
dt = T / nt  # passo de tempo

# Coeficiente r
r = alpha * dt / dx**2

# Condição inicial
x = np.linspace(0, L, nx)
u_initial = np.sin(2 * np.pi * x)

# Solução numérica
u = u_initial.copy()
u_next = np.zeros(nx)

# Armazenar solução em todos os tempos
u_all = np.zeros((nt, nx))
u_all[0] = u

for t in range(1, nt):
    u_next[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
    u = u_next.copy()
    # Aplicando as condições de fronteira
    u[0] = 0
    u[-1] = 0
    u_all[t] = u

# Para plotar, pegamos a solução em intervalos de tempo
t_steps = np.linspace(0, T, nt)
x_steps = np.linspace(0, L, nx)
X, T = np.meshgrid(x_steps, t_steps)

# Plotando a solução numérica
axes[1].plot_surface(X, T, u_all, cmap='viridis')
axes[1].set_xlabel('x')
axes[1].set_ylabel('t')
axes[1].set_zlabel('u(x, t)')
axes[1].set_title('Solução numérica da eq. do calor 1D (alpha = 0.05)')

plt.show()

print(f"Final Loss: {loss.item()}")