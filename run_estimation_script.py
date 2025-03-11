import multiprocessing as mp
import estimators
import sys
import numpy as np
from functools import partial

if __name__ == "__main__":
    # Computes dimension estimates at every point in a point cloud stored in a matrix.

    nargs = len(sys.argv)

    if nargs < 5:
        print("Usage: ", sys.argv[0], " <matrix_input_filename.npy> <r> <eta> <output_filename.npy> [<verbose>]")
    else:
        global X, r, eta, output_filename, verbose

        matrix_filename = sys.argv[1]
        r = float(sys.argv[2])
        eta = float(sys.argv[3])
        output_filename = sys.argv[4]
        if nargs >= 6:
            # we allow extra arguments - we will just ignore them
            verbose = (sys.argv[5] == 'True') # anything else will be considered False
        else:
            verbose = False

        X = np.load(matrix_filename)
        m = len(X)

        n_processes = mp.cpu_count()
        if verbose:
            print(f"Using {n_processes=}")

        with mp.Pool(n_processes) as p:
            dim_estims = p.map(partial(estimators.tgt_and_dim_estimate_1point, X=X, r=r, eta=eta, dim_only = True, verbose=verbose), range(m))

        np.save(output_filename, np.array(list(dim_estims)))
        print("Dimension estimates computed and saved to ", output_filename)

