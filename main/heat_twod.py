import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import scipy


# Verifica se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Arquitetura da rede neural: 5 camadas hidden layers com ativação tanh
class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.layer5 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.layer1(x))
        x = self.tanh(self.layer2(x))
        x = self.tanh(self.layer3(x))
        x = self.tanh(self.layer4(x))
        x = self.tanh(self.layer5(x))
        x = self.output_layer(x)
        return x
    

def loss_function(model, alpha, x_init, y_init, t_init, u_init, x_boundary, y_boundary, t_boundary, u_boundary, x_collocation, y_collocation, t_collocation):
    # Condições iniciais
    u_initial_pred = model(torch.cat((x_init, y_init, t_init), dim=1))
    mse_initial = nn.MSELoss()(u_initial_pred, u_init)

    # Condições de fronteira
    u_boundary_pred = model(torch.cat((x_boundary, y_boundary, t_boundary), dim=1))
    mse_boundary = nn.MSELoss()(u_boundary_pred, u_boundary)

    # Rastreia as operações sobre os tensores, permitindo autodiff
    x_collocation.requires_grad = True
    y_collocation.requires_grad = True
    t_collocation.requires_grad = True

    u = model(torch.cat((x_collocation, y_collocation, t_collocation), dim=1))

    u_t = torch.autograd.grad(u, t_collocation, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_collocation, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_collocation, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_collocation, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_collocation, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    # Desliga o rastreamento dos tensores para poupar memória e processamento
    x_collocation.requires_grad = False
    y_collocation.requires_grad = False
    t_collocation.requires_grad = False

    f_pred = u_t - alpha * (u_xx + u_yy)
    mse_pde = nn.MSELoss()(f_pred, torch.zeros_like(f_pred))

    # Loss total
    loss = mse_initial + mse_boundary + mse_pde
    return loss


# Init
input_size = 3
hidden_size = 100
output_size = 1
model = PINN(input_size, hidden_size, output_size).to(device)

# Quantidade de exemplos de treinamento
n_s_boundary = 10
n_t_boundary = 10
n_initial = 100
n_collocation = 10000

# Dados para treinamento (condições iniciais, fronteira e pontos de collocation)
sampler = scipy.stats.qmc.LatinHypercube(d=2) # metodo Latin Hypercube

xy = torch.Tensor(sampler.random(n=n_initial))
x_init = xy[:,0].reshape(-1,1).to(device)
y_init = xy[:,1].reshape(-1,1).to(device)
t_init = torch.zeros_like(x_init).to(device)
u_init = (torch.sin(torch.pi * x_init) * torch.sin(torch.pi * y_init)).to(device)

sampler = scipy.stats.qmc.LatinHypercube(d=1)
x_boundary = torch.cat([torch.zeros(n_s_boundary),
                        torch.Tensor(sampler.random(n=n_s_boundary)).flatten(),
                        torch.Tensor(sampler.random(n=n_s_boundary)).flatten(),
                        torch.ones(n_s_boundary)]).to(device)

y_boundary = torch.cat([torch.Tensor(sampler.random(n=n_s_boundary)).flatten(),
                        torch.zeros(n_s_boundary),
                        torch.ones(n_s_boundary),
                        torch.Tensor(sampler.random(n=n_s_boundary)).flatten()]).to(device)

t = torch.Tensor(sampler.random(n=n_t_boundary)).flatten().to(device)

x_boundary, t_boundary = torch.meshgrid(x_boundary, t, indexing='ij')
y_boundary, t_boundary = torch.meshgrid(y_boundary, t, indexing='ij')
x_boundary = x_boundary.reshape(-1, 1).to(device)
y_boundary = y_boundary.reshape(-1, 1).to(device)
t_boundary = t_boundary.reshape(-1, 1).to(device)
u_boundary = torch.zeros_like(t_boundary).to(device)

sampler = scipy.stats.qmc.LatinHypercube(d=3)
collocation = torch.Tensor(sampler.random(n=n_collocation))

x_collocation = collocation[:,0].reshape(-1, 1).to(device)
y_collocation = collocation[:,1].reshape(-1, 1).to(device)
t_collocation = collocation[:,2].reshape(-1, 1).to(device)


# Usamos o otimizador L-BFGS
optimizer = optim.LBFGS(model.parameters(), lr=0.01)

alpha = 0.1

# Função de fechamento necessária para L-BFGS
def closure():
    optimizer.zero_grad()  # Zera os gradientes dos parâmetros
    loss = loss_function(model, alpha,
                          x_init, y_init, t_init, u_init,
                          x_boundary, y_boundary, t_boundary, u_boundary,
                          x_collocation, y_collocation, t_collocation)
    loss.backward()  # Backward pass para calcular gradientes
    return loss

# Treinamento
model.train()
for epoch in range(100):
    loss = optimizer.step(closure)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


# Visualização e plot
model.eval()
t_values = [0.0, 0.15, 0.30, 0.45]
fig = plt.figure(figsize=(15, 13))

for i, t_val in enumerate(t_values):
    with torch.no_grad():
        x_plot = torch.linspace(0, 1, 100).to(device)
        y_plot = torch.linspace(0, 1, 100).to(device)
        x_plot, y_plot = torch.meshgrid(x_plot, y_plot, indexing='ij')
        t_plot = torch.tensor([t_val], device=device).expand_as(x_plot).reshape(-1, 1)  # Solução no tempo t = t_val
        x_plot = x_plot.reshape(-1, 1).to(device)
        y_plot = y_plot.reshape(-1, 1).to(device)

        u_plot = model(torch.cat((x_plot, y_plot, t_plot), dim=1)).reshape(100, 100).cpu()

    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    X = x_plot.view(100, 100).cpu().numpy()
    Y = y_plot.view(100, 100).cpu().numpy()
    Z = u_plot.numpy()

    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(f'u(x, y, t={t_val})')
    ax.set_zlim(-0.2, 1.2)
    ax.set_title(f'Solution at t={t_val}')

plt.tight_layout()
plt.show()

print(f"Final Loss: {loss.item()}")