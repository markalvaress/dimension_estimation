import numpy as np
import math


def noisy_sphere(d: int, r: float, s: float, m: int, seed: int = None) -> np.ndarray:
    """Samples m points uniformly from a d-dml sphere (in R^{d+1}) of radius r with normally distributed noise strictly bounded in magnitude by s."""
    rng = np.random.default_rng(seed = seed)

    if d == 0:
        # the 0 sphere is just {0,1}, so sample these discretely
        pts = r * rng.integers(0,2, (m,1))
    elif d == 1:
        # uniformly sample a theta and then parametrise as (r*cos(theta), r*sin(theta))
        thetas = rng.uniform(0, 2*np.pi, m)
        pts = r * np.array([np.cos(thetas), np.sin(thetas)].T)
    elif d == 2:
        # uniformly sample theta and phi and parametrise using spherical polars
        thetas = rng.uniform(0, 2*np.pi, m)
        phis = rng.uniform(0, np.pi, m)
        pts = r * np.array([np.sin(phis)*np.cos(thetas), np.sin(phis)*np.sin(thetas), np.cos(phis)]).T
    else:
        #basic idea: randomly sample points from a D-dimensional NORMAL distribution (since normal distribution is spherically symmetric, whereas uniform would lead to overconcentration in corners), then push them onto the unit ball, then add some noise
        pts = rng.normal(0, 1, (m, d+1))
        norms = np.linalg.norm(pts, axis = 1)
        for i in range(m):
            pts[i] = r * pts[i] / norms[i]

    # we use an s/3 here as the SD so that virtually all noise (99.7%) will be within 3*sd = s of 0.
    noise = rng.normal(0, s/3, (m, d+1)) 
    np.clip(noise, -s, s, out = noise) #Bound the noise to strictly within +/- s.
    pts += noise

    return pts

def ball_volume(d: int, r: float = 1) -> float:
    """Calculate the volume of the ball of radius r in d dimensions."""
    w_d = math.pi**(d/2) / math.gamma(d/2 + 1)
    return w_d

def find_geodesic_distance_sphere(p1: np.ndarray, p2: np.ndarray, r: float) -> float:
    """Finds the geodesic distance between two points on a sphere in arbitrary dimension of radius r.

    Args:
        p1 (np.ndarray): 1st point
        p2 (np.ndarray): 2nd point
        r (float): Radius of the sphere

    Raises:
        ValueError: If for some reason a negative angle between the two points is calculated (this shouldn't happen)

    Returns:
        float: Geodesic (arc) distance between the two points.
    """
    assert len(p1) == len(p2), "Points must live in the same space"
    assert r > 0, "Radius must be positive"

    d_euclidean = np.linalg.norm(p1-p2, 2)
    theta = 2*np.arcsin(d/(2*r))
    if theta < 0:
        raise ValueError(f"Points {p1=} and {p2=} resulted in a negative angle between them.")
    d_geod = theta*r

    return d_geod