from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol


def prob(x,e1,nu1,h):
    L = nu1
    H = h
    Nx = 250
    Ny = 10
    mesh = RectangleMesh(Point(0., 0.), Point(L, H), Nx, Ny, "crossed")

    def eps(v):
        return sym(grad(v))

    E = Constant(e1)
    nu = Constant(.3)
    model = "plane_stress"

    mu = E/2/(1+nu)
    lmbda = E*nu/(1+nu)/(1-2*nu)
    if model == "plane_stress":
        lmbda = 2*mu*lmbda/(lmbda+2*mu)

    def sigma(v):
        return lmbda*tr(eps(v))*Identity(2) + 2.0*mu*eps(v)

    rho_g = 1e-3
    f = Constant((0, -rho_g))

    V = VectorFunctionSpace(mesh, 'Lagrange', degree=2)
    du = TrialFunction(V)
    u_ = TestFunction(V)
    a = inner(sigma(du), eps(u_))*dx
    l = inner(f, u_)*dx
    def left(x, on_boundary):
        return near(x[0], 0.)

    bc = DirichletBC(V, Constant((0.,0.)), left)

    u = Function(V, name="Displacement")
    solve(a == l, u, bc)

    #plot(1e3*u, mode="displacement")
    # print("Maximal deflection:", -u(L,H/2.)[1])
    # print("Beam theory deflection:", float(3*rho_g*L**4/2/E/H**3))
    Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
    sig = Function(Vsig, name="Stress")
    sig.assign(project(sigma(u), Vsig))
    #print("Stress at (0,H):", sig(0, H))

    # file_results = XDMFFile("elasticity_results.xdmf")
    # file_results.parameters["flush_output"] = True
    # file_results.parameters["functions_share_mesh"] = True
    # file_results.write(u, 0.)
    # file_results.write(sig, 0.)
    return -u(L,H/2.)[1]+(x-x)
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------


problem = {
    'num_vars': 3,
    'names': ['E', 'L','H'],
    'bounds': [[100000-1000, 100000+1000],[24.9,25.1],[.98,1.02]]
}
param_values = saltelli.sample(problem, 2**4)
x = np.linspace(0,1,2)
y = np.array([prob(x, *params) for params in param_values])
sobol_indices = [sobol.analyze(problem, Y) for Y in y.T]

S1s = np.array([s['S1'] for s in sobol_indices])

fig = plt.figure(figsize=(10, 6), constrained_layout=True)
gs = fig.add_gridspec(2, 2)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1])
ax3 = fig.add_subplot(gs[1, 0])

for i, ax in enumerate([ax1, ax2,ax3]):
    ax.plot(x, S1s[:, i],
            label=r'S1$_\mathregular{{{}}}$'.format(problem["names"][i]),
            color='black')
    ax.set_xlabel("x")
    ax.set_ylabel("First-order Sobol index")

    ax.set_ylim(0, 2)

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    ax.legend(loc='upper right')

ax0.plot(x, np.mean(y, axis=0), label="Mean", color='black')

# in percent
prediction_interval = 95

ax0.fill_between(x,
                 np.percentile(y, 50 - prediction_interval/2., axis=0),
                 np.percentile(y, 50 + prediction_interval/2., axis=0),
                 alpha=0.5, color='black',
                 label=f"{prediction_interval} % prediction interval")

ax0.set_xlabel("x")
ax0.set_ylabel("y")
ax0.legend(title=r"cantilever beam",
           loc='upper center')._legend_box.align = "left"

plt.show()

print(S1s)