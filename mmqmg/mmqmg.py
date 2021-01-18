import functools
import time
import numpy as np
import numpy.linalg as la

from scipy.signal import convolve2d as conv2

# [x,Crit,NGrad]
def mmmg(data, H, H_adj, eta, tau, l, delta, xmin, xmax, phi, x, NbIt):
    """
    MM-MG Algorithm, Version 1.0


    ----------------------------------------------------------------------

    Input:  y: the degraded data
    H: the linear degradation operator (function handle)
    H_adj: the adjoint of the linear degradation operator
    (function handle)
    tau, l, delta: the regularization parameters
    xmin, xmax: the bounds
    phi: the penalty function flag as indicated below
    (1) phi(u) = (1-exp(-u.^2./(2*delta^2)));
    (2) phi(u) = (u.^2)./(2*delta^2 + u.^2);
    (3) phi(u) = log(1 + (u.^2)./(delta^2));
    (4) phi(u) = sqrt(1 + u^2/delta^2)-1;
    (5) phi(u) = 1/2 u^2;
    NbIt: the max number of iterations.

    Output: x: the restored image
    Crit: the values of the criterion along minimization process
    NGrad: the value of gradient norm along minimization process


    Minimization of

    F(x) = 1/2 || H(x)- y ||^2 + l sum(phi(V(x))) + tau ||x||^2 + 1/2 eta ||Proj_[xmin; xmax] - x||^2

    ========================================================================
    Kindly report any suggestions or corrections to
    emilie.chouzenoux@univ-mlv.fr

    """
    shape = x.shape
    Nx, Ny = x.shape

    H = vectorize(H, x.shape)
    H_adj = vectorize(H_adj, data.shape)
    data = data.reshape(-1, 1)
    x = x.reshape(-1, 1)

    stop = min(Nx, Ny) * 1e-4

    if phi == 1:
        print("phi(u) =  (1-exp(-u^2/(2*delta^2)))")
    elif phi == 2:
        print("phi(u) = (u^2)/(2*delta^2 + u^2)")
    elif phi == 3:
        print("phi(u) = log(1 + (u^2)/(delta^2))")
    elif phi == 4:
        print("phi(u) = sqrt(1 + u^2/delta^2)-1")
    elif phi == 5:
        print("phi(u) = 1/2 u^2")
    else:
        print(f"phi={phi} unkown")

    print(f"l={l}, delta={delta}, eta = {eta} and tau = {tau}")
    print(f"xmin = {xmin} and xmax = {xmax}")

    val, grad, Vx = critere(
        x, data, H, H_adj, eta, tau, l, delta, phi, xmin, xmax, Nx, Ny
    )
    print(f"Initial criterion value = {val}")

    timestamp = [0]
    crit = [val]
    norm_grad = [la.norm(grad)]

    for k in range(NbIt):
        tic = time.time()
        # print(f"iteration {k} / {NbIt}")

        # Stopping test
        if norm_grad[-1] < stop:
            break

        Vg = Voperator(grad, Nx, Ny)
        Hg = H(grad)

        if k == 0:
            Dir = -grad
            B = majorantem(Vx, -Vg, -Hg, Dir, eta, tau, l, delta, phi)
            s = np.sum(grad ** 2) / B
            dx = s * Dir
            Vdx = -s * Vg
            Hdx = -s * Hg
        else:
            Dir = np.concatenate((-grad, dx), axis=1)
            VD = np.concatenate((-Vg, Vdx), axis=1)
            HD = np.concatenate((-Hg, Hdx), axis=1)
            B = majorantem(Vx, VD, HD, Dir, eta, tau, l, delta, phi)
            s = -la.pinv(B) @ (Dir.T @ grad)
            dx = Dir @ s
            Vdx = VD @ s
            Hdx = HD @ s

        # update
        x = x + dx

        val, grad, Vx = critere(
            x, data, H, H_adj, eta, tau, l, delta, phi, xmin, xmax, Nx, Ny
        )

        crit.append(val)
        timestamp.append(time.time() - tic)
        norm_grad.append(la.norm(grad))

    print(f"Iteration number = {len(crit)}")
    print(f"Computation time (cpu) = {sum(timestamp)}")
    print(f"Final criterion value = {crit[-1]}")

    return np.reshape(x, shape), crit, norm_grad


def critere(x, data, H, H_adj, eta, tau, l, delta, phi, xmin, xmax, Nx, Ny):

    Vx = Voperator(x, Nx, Ny)
    Hx = H(x)

    if phi == 1:
        phiVx = 1 - np.exp(-(Vx ** 2) / (2 * delta ** 2))
        dphiVx = (Vx / (delta ** 2)) * np.exp(-(Vx ** 2) / (2 * delta ** 2))
    elif phi == 2:
        phiVx = (Vx ** 2) / (2 * delta ** 2 + Vx ** 2)
        dphiVx = (4 * Vx * (delta ** 2)) / (2 * delta ** 2 + (Vx ** 2)) ** 2
    elif phi == 3:
        phiVx = np.log(1 + (Vx ** 2) / (delta ** 2))
        dphiVx = (2 * Vx) / (delta ** 2 + (Vx ** 2))
    elif phi == 4:
        phiVx = np.sqrt(1 + (Vx ** 2) / (delta ** 2)) - 1
        dphiVx = (Vx / (delta ** 2)) * np.sqrt(1 + (Vx ** 2) / delta ** 2)
    elif phi == 5:
        phiVx = Vx ** 2 / 2
        dphiVx = Vx

    F = np.sum((data - Hx) ** 2) / 2 + tau * np.sum(x ** 2) + l * np.sum(phiVx)
    dF = H_adj(Hx - data) + 2 * tau * x + l * VoperatorT(dphiVx, Nx, Ny)

    if not np.isinf(xmin):
        F = F + eta * np.sum((x[x < xmin] - xmin) ** 2 / 2)
        dF = dF + np.where(x > xmin, 0, eta * (x - xmin))

    if not np.isinf(xmax):
        F = F + eta * np.sum((x[x > xmax] - xmax) ** 2 / 2)
        dF = dF + np.where(x < xmax, 0, eta * (x - xmax))

    return F, dF, Vx


def majorantem(Vx, VD, HD, D, eta, tau, l, delta, phi):
    n, m = D.shape

    if phi == 1:
        wVx = np.exp(-(Vx ** 2) / (2 * delta ** 2)) / delta ** 2
    elif phi == 2:
        wVx = (4 * delta ** 2) / (2 * delta ** 2 + Vx ** 2) ** 2
    elif phi == 3:
        wVx = 2 / (delta ** 2 + Vx ** 2)
    elif phi == 4:
        wVx = np.sqrt(1 + (Vx ** 2) / delta ** 2) / (delta ** 2)
    elif phi == 5:
        wVx = np.ones(Vx.shape)

    DtD = D.T @ D
    DtVtWVD = VD.T @ (np.tile(wVx, m) * VD)
    return HD.T @ HD + 2 * tau * DtD + l * DtVtWVD + 2 * eta * DtD


def Voperator(x, Nx, Ny):
    x = np.reshape(x, (Nx, Ny))

    hv = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    hh = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])

    Vv = conv2(x, hv, "same").reshape(-1, 1)
    Vh = conv2(x, hh, "same").reshape(-1, 1)

    return np.concatenate((Vv, Vh), axis=0)


def VoperatorT(x, Nx, Ny):
    hv = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    hh = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])

    Vth = conv2(np.reshape(x[Nx * Ny :], (Nx, Ny)), np.fliplr(np.flipud(hh)), "same")
    Vtv = conv2(np.reshape(x[: Nx * Ny], (Nx, Ny)), np.fliplr(np.flipud(hv)), "same")

    return Vtv.reshape(-1, 1) + Vth.reshape(-1, 1)


def vectorize(func, in_shape):
    @functools.wraps(func)
    def wrapper(arr):
        out = func(np.reshape(arr, in_shape))
        return out.reshape(-1, 1)

    return wrapper


# # Decorator version
# def vectorize(in_shape):
#     def decorator(func):
#         @functools.wraps(func)
#         def wrapper(arr, in_shape):
#             out = func(np.reshape(arr, in_shape)
#             return out.reshape(-1, 1)

#         return wrapper
#     return decorator
