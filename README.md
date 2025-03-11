Here I test the intrinsic dimension estimation algorithm described in https://epubs.siam.org/doi/full/10.1137/22M1522711.

The algorithm is implmented in `estimators.py`. The constants for Theorem B can be calculated with functions from `constants.py`. The notebook `issues_with_constants.ipynb` describes issues with chooses parameter values, and `experiments.ipynb` tests out the algorithm. The `run_estimation_script.py` parallelises the dimension estimation calculation.
