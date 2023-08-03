import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
from jax.typing import ArrayLike
from functools import partial
from typing import Union


@jax.jit
def transform_coords(coords, transform_params, pixel_size):
    rot_mtx = Rotation.from_euler("xyx", transform_params[:3]).as_matrix()

    coords = jnp.matmul(rot_mtx, coords)
    #coords[2, :] -= transform_params[3] * pixel_size
    #coords[1, :] -= transform_params[4] * pixel_size

    return coords


@partial(jax.jit, static_argnames=["pixel_size", "box_size"])
def projection(
    coords: ArrayLike,
    struct_info: ArrayLike,
    box_size: int,
    pixel_size: float,
    res: float,
) -> ArrayLike:
    # assert pixel_size < 2.0 * res, "Pixel size should be smaller than 2.0 * res due to the Nyquist limit."

    gauss_var = struct_info[0, :] * res**2
    gauss_amp = struct_info[1, :] / jnp.sqrt(gauss_var * 2.0 * jnp.pi)

    # Project
    grid_min = -pixel_size * box_size * 0.5
    grid_max = pixel_size * box_size * 0.5
    grid = jnp.arange(grid_min, grid_max, pixel_size)[0:box_size]

    gauss_z = gauss_amp * jnp.exp(
        -0.5 * (((grid[:, None] - coords[2, :]) / gauss_var) ** 2)
    )
    gauss_y = gauss_amp * jnp.exp(
        -0.5 * (((grid[:, None] - coords[1, :]) / gauss_var) ** 2)
    )
    image = jnp.matmul(gauss_z, gauss_y.T)

    return image


@partial(jax.jit, static_argnames=["pixel_size", "box_size"])
def apply_ctf(image, box_size, pixel_size, amp, defocus, bfactor):
    # # Apply CTF
    freq_pix_1d = jnp.fft.fftfreq(box_size, d=pixel_size)
    freq2_2d = freq_pix_1d[:, None] ** 2 + freq_pix_1d[None, :] ** 2

    elecwavel = 0.019866
    phase = defocus * jnp.pi * 2.0 * 10000 * elecwavel

    env = jnp.exp(-bfactor * freq2_2d * 0.5)
    ctf = (
        (
            amp * jnp.cos(phase * freq2_2d * 0.5)
            - jnp.sqrt(1 - amp**2) * jnp.sin(phase * freq2_2d * 0.5)
            + 0.0j
        )
        * env
        / amp
    )

    # Normalize image
    image = jnp.fft.ifft2(jnp.fft.fft2(image) * ctf).real

    return image


@partial(jax.jit, static_argnames=["pixel_size", "box_size"])
def add_white_noise(image, box_size, noise_radius_mask, snr, seed=1234):
    random_key = jax.random.PRNGKey(seed)

    image /= jnp.linalg.norm(image)

    # add noise
    noise_grid = jnp.linspace(-0.5 * (box_size - 1), 0.5 * (box_size - 1), box_size)
    radii_for_mask = noise_grid[None, :] ** 2 + noise_grid[:, None] ** 2
    mask = radii_for_mask < noise_radius_mask**2

    signal_power = jnp.sqrt(jnp.sum((image * mask) ** 2) / jnp.sum(mask))

    noise_power = signal_power / jnp.sqrt(snr)
    image = image + jax.random.normal(random_key, shape=image.shape) * noise_power

    return image
