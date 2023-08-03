import numpy as np
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from .image_generation import transform_coords, projection, apply_ctf, add_noise
from .param_tool import CryoParams

def gen_quat(n_quats, dtype: float) -> np.ndarray:
    """
    Generate a random quaternion.

    Returns:
        quat (np.ndarray): Random quaternion

    """

    np.random.seed(1234)

    quats = np.empty((n_quats, 4), dtype=dtype)

    count = 0
    while count < n_quats:
        quat = np.random.uniform(
            -1, 1, 4
        )  # note this is a half-open interval, so 1 is not included but -1 is
        norm = np.sqrt(np.sum(quat**2))

        if 0.2 <= norm <= 1.0:
            quat /= norm
            quats[count] = quat
            count += 1

    return jnp.array(quats)

def simulate(
        coords,
        struct_info,
        param_tool: CryoParams,
        seed=1234,
)

    transf_params = np.zeros(5)
    quat = gen_quat(1, dtype=jnp.float64)
    transf_params[:3] = quat[0, :]

    # Transform coordinates
    transform_coords = transform_coords(coords, transf_params, param_tool.pixel_size)

    # Generate projection
    image = projection(transform_coords, struct_info, param_tool.box_size, param_tool.pixel_size, param_tool.res)

    # Apply CTF
    image = apply_ctf(image, param_tool.box_size, param_tool.pixel_size, param_tool.amp, param_tool.defocus, param_tool.bfactor)

    # Add noise
    image = add_noise(image, param_tool.noise_radius_mask, param_tool.snr, seed)

    return image