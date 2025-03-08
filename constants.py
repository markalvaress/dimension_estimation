import numpy as np

def S1_dimension(d: int, D: int, eta: float, alpha: float, tau: float, phi_min: float, phi_max: float):
    c1, c2 = 1392, 288 # constants defined in theorem B

    S1 = phi_min / ((d+2)*D*(1 + 1/eta)*(c1*d*phi_max + c2*alpha*tau))
    return S1

def S2_dimension(d: int, D: int, eta: float, w_d: float, delta: float, rho: float, phi_min: float):
    c3, c4 = 41791, 14 # constants defined in theorem B

    S2 = c3*(d + 2)**2 * D**2 * (1 + 1/eta)**2 / (w_d * phi_min) * np.log(c4*D*rho/delta)
    return S2