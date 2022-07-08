import numpy as np
from scipy.spatial.transform import Rotation

def gen_quat(size):
    # Sonya's code

    quats = np.empty((size, 4))

    count = 0
    while count < size:

        quat = np.random.uniform(
            -1, 1, 4
        )  # note this is a half-open interval, so 1 is not included but -1 is
        norm = np.sqrt(np.sum(quat**2))

        if 0.2 <= norm <= 1.0:
            quats[count] = quat / norm
            count += 1

    return quats

def rot_coordinates(coord, quat=None):

    if quat is None:
        quat = gen_quat(1)[0]
        
    rot_mat = Rotation.from_quat(quat).as_matrix()
    rot_coord = np.matmul(rot_mat, coord)

    return rot_coord

def gen_img(coord, image_params):

    n_atoms = coord.shape[1]
    norm = 1 / (2 * np.pi * image_params["SIGMA"] ** 2 * n_atoms)

    grid_min = -image_params["PIXEL_SIZE"] * (image_params["N_PIXELS"] - 1) * 0.5
    grid_max = (
        image_params["PIXEL_SIZE"] * (image_params["N_PIXELS"] - 1) * 0.5
        + image_params["PIXEL_SIZE"]
    )

    grid = np.arange(grid_min, grid_max, image_params["PIXEL_SIZE"])

    gauss_x = np.exp(
        -0.5 * (((grid[:, np.newaxis] - coord[0, :]) / image_params["SIGMA"]) ** 2)
    )

    gauss_y = np.exp(
        -0.5 * (((grid[:, np.newaxis] - coord[1, :]) / image_params["SIGMA"]) ** 2)
    )

    image = np.matmul(gauss_x, gauss_y.T) * norm

    return image
