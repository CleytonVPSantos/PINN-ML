import numpy as np
import matplotlib.pyplot as plt

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
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, u_all, cmap='viridis')

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x, t)')
plt.title('Numerical Solution of the Heat Equation')
plt.show()