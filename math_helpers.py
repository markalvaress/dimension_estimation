import numpy as np
import math
import scipy as sp

def noisy_sphere(d: int, R: float, s: float, m: int, seed: int = None) -> np.ndarray:
    """Samples m points uniformly from a d-dml sphere (in R^{d+1}) of radius R with normally distributed noise strictly bounded in magnitude by s."""
    rng = np.random.default_rng(seed = seed)

    if d == 0:
        # the 0 sphere is just {0,1}, so sample these discretely
        pts = R * rng.integers(0,2, (m,1))
    else:
        #basic idea: randomly sample points from a D-dimensional NORMAL distribution (since normal distribution is spherically symmetric, whereas uniform would lead to overconcentration in corners), then push them onto the unit ball, then add some noise
        pts = rng.normal(0, 1, (m, d+1))
        norms = np.linalg.norm(pts, axis = 1)
        for i in range(m):
            pts[i] = R * pts[i] / norms[i]

    # we use an s/3 here as the SD so that virtually all noise (99.7%) will be within 3*sd = s of 0.
    noise = rng.normal(0, s/3, (m, d+1))
    np.clip(noise, -s, s, out = noise) #Bound the noise to strictly within +/- s.
    pts += noise

    return pts

def ball_volume(D: int, r: float = 1) -> float:
    """Calculate the volume of the ball of radius r in D dimensions."""
    w_d = math.pi**(D/2) / math.gamma(D/2 + 1) * r**D
    return w_d

def sphere_surface_area(d: int, R: float = 1) -> float:
    """Calculate the full surface area of the d-dml sphere (sitting in R^{d+1}) of radius R."""
    n = d+1
    sa = 2*math.pi**(n/2) / math.gamma(n/2) * R**d
    return sa

def calculate_measure_concentration_uniform_sphere(r: float, R: float, sphere_surf_area: float, d: int, tau: float, s: float) -> float:
    """Calculate the 'local concentration' of mu_0 on a d-dimensional sphere of radius R, using balls of radius r.

    Args:
        r (float): _description_
        R (float): _description_
        sphere_surf_area (float): _description_
        d (int): _description_
        tau (float): _description_
        s (float): _description_

    Returns:
        float: _description_
    """
    r_minus = r*(1 - r**2/(4*tau**2)) - 2*s
    phi_star = np.arcsin(r_minus/(2*R))
    integral = sp.integrate.quad(lambda phi: np.sin(phi)**(d-1), 0, phi_star)
    Phi = 1/sphere_surf_area * (R/r_minus)**d * integral[0]
    return Phi

# CODE GRAVEYARD ----------------------

## This is just plain wrong for dimension >2. But I'm keeping it in here in case it could have been useful if tweaked a bit. 
# def noisy_sphere(d: int, r: float, s: float, m: int, seed: int = None) -> np.ndarray:
#     """Samples m points uniformly from a d-dml sphere (in R^{d+1}) of radius r with normally distributed noise strictly bounded in magnitude by s."""
#     rng = np.random.default_rng(seed = seed)

#     if d == 0:
#         # the 0 sphere is just {0,1}, so sample these discretely
#         pts = r * rng.integers(0,2, (m,1))
#     elif d == 1:
#         # uniformly sample a theta and then parametrise as (r*cos(theta), r*sin(theta))
#         thetas = rng.uniform(0, 2*np.pi, m)
#         pts = r * np.array([np.cos(thetas), np.sin(thetas)].T)
#     elif d == 2:
#         # uniformly sample theta and phi and parametrise using spherical polars
#         thetas = rng.uniform(0, 2*np.pi, m)
#         phis = rng.uniform(0, np.pi, m)
#         pts = r * np.array([np.sin(phis)*np.cos(thetas), np.sin(phis)*np.sin(thetas), np.cos(phis)]).T
#     else:
#         # use the n-dml spherical coordinate system from https://en.wikipedia.org/wiki/N-sphere
#         # if we have d-dml sphere then n = d+1.
#         phis_secondary = rng.uniform(0, np.pi, (m, d-1)) # this is phi_1, ..., phi_{d-1} 
#         phis_main = rng.uniform(0, 2*np.pi, m) # similar to thetas before. This is phi_{n-1} (phi_d) in the wiki article 

#         pts = np.zeros((m, d+1))

#         # convert the the coordinates in R^d into coordinates on the sphere embedded in R^{d+1}
#         for i in range(d-1):
#             pts[:, i] = r * np.cos(phis_secondary[:, i])
#             for j in range(i):
#                 pts[:, i] *= np.sin(phis_secondary[:, j])

#         # the last two points are slightly different
#         pts[:, d-1] = r * np.cos(phis_main)
#         pts[:, d] = r * np.sin(phis_main)
#         for j in range(d-1):
#                 pts[:, d-1] *= np.sin(phis_secondary[:, j])
#                 pts[:, d] *= np.sin(phis_secondary[:, j])

#     # we use an s/3 here as the SD so that virtually all noise (99.7%) will be within 3*sd = s of 0.
#     noise = rng.normal(0, s/3, (m, d+1)) 
#     np.clip(noise, -s, s, out = noise) # Bound the noise to strictly within +/- s.
#     pts += noise

#     return pts


## This is no longer needed as we found an analytic solution
# def find_geodesic_distance_sphere(p1: np.ndarray, p2: np.ndarray, r: float) -> float:
#     """Finds the geodesic distance between two points on a sphere in arbitrary dimension of radius r.

#     Args:
#         p1 (np.ndarray): 1st point
#         p2 (np.ndarray): 2nd point
#         r (float): Radius of the sphere

#     Raises:
#         ValueError: If for some reason a negative angle between the two points is calculated (this shouldn't happen)

#     Returns:
#         float: Geodesic (arc) distance between the two points.
#     """
#     assert len(p1) == len(p2), "Points must live in the same space"
#     assert r > 0, "Radius must be positive"

#     d_euclidean = np.linalg.norm(p1-p2, 2)
#     theta = 2*np.arcsin(d/(2*r))
#     if theta < 0:
#         raise ValueError(f"Points {p1=} and {p2=} resulted in a negative angle between them.")
#     d_geod = theta*r
# 
#     return d_geod