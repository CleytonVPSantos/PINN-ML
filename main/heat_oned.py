import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

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
    u_init_pred = model(torch.hstack((x_init, t_init)))
    mse_initial = nn.MSELoss()(u_init_pred, u_init)

    # Condições de fronteira
    u_boundary_pred = model(torch.hstack((x_boundary, t_boundary)))
    mse_boundary = nn.MSELoss()(u_boundary_pred, u_boundary)

    # Rastreia as operações sobre os tensores, permitindo autodiff
    x_collocation.requires_grad = True
    t_collocation.requires_grad = True

    u = model(torch.hstack((x_collocation, t_collocation)))

    u_t = torch.autograd.grad(u, t_collocation, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_collocation, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_collocation, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    x_collocation.requires_grad = False
    t_collocation.requires_grad = False
    
    f_pred = u_t - alpha * u_xx
    mse_pde = nn.MSELoss()(f_pred, torch.zeros_like(f_pred))

    # Loss total
    loss = mse_initial + mse_pde + mse_boundary
    return loss


def train_model(alpha = 0.05):
    # Parâmetros
    input_size = 2
    hidden_size = 100
    output_size = 1
    model = PINN(input_size, hidden_size, output_size)

    # Quantidade de exemplos de treinamento
    n_boundary = 10
    n_initial = 10
    n_collocation = 100
    
    # Dados para treinamento
    x_init = torch.linspace(0, 1, n_initial).reshape(-1, 1)
    t_init = torch.zeros_like(x_init)
    u_init = torch.sin(2 * torch.pi * x_init)

    x_boundary = torch.cat([torch.zeros(n_boundary), torch.ones(n_boundary)]).reshape(-1, 1)
    t_boundary = torch.linspace(0, 1, n_boundary).reshape(-1, 1).repeat(2, 1)
    u_boundary = torch.zeros_like(t_boundary)

    x_collocation, t_collocation = torch.meshgrid(
        torch.linspace(0, 1, n_collocation),
        torch.linspace(0, 1, n_collocation),
        indexing='ij'
    )
    x_collocation = x_collocation.reshape(-1, 1)
    t_collocation = t_collocation.reshape(-1, 1)

    # Otimizador
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=50000, max_eval=50000, history_size=50)

    # Função de fechamento necessária para L-BFGS
    def closure():
        optimizer.zero_grad()  # Zera os gradientes dos parâmetros
        loss = loss_function(model, alpha, x_init, t_init, u_init, x_boundary, t_boundary, u_boundary, x_collocation, t_collocation)
        loss.backward()  # Backward pass para calcular gradientes
        return loss

    # Treinamento
    model.train()

    for epoch in range(50):
        loss = optimizer.step(closure)
        print(f"Epoch {epoch}, Loss: {loss.item()}")


def visualize_result(model, loss):
    # Visualização e plot
    model.eval()
    with torch.no_grad():
        x_plot = torch.linspace(0, 1, 100)
        t_plot = torch.linspace(0, 1, 100)
        x_plot, t_plot = torch.meshgrid(x_plot, t_plot, indexing='ij')
        u_plot = model(torch.cat((x_plot.reshape(-1, 1), t_plot.reshape(-1, 1)), dim=1))
        u_plot = u_plot.reshape(100, 100)

    # Plotando a solução
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_plot.numpy(), t_plot.numpy(), u_plot.numpy(), cmap='viridis')

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x, t)')
    plt.title('Solution of the Heat Equation using PINN')
    plt.show()

    print(f"Final Loss: {loss.item()}")


def main():
    # parametro da equação do calor
    alpha = 0.05

    model, loss = train_model(alpha)
    visualize_result(model, loss)


