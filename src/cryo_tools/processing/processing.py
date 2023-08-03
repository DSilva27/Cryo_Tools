import numpy as np


def pad_image(image, image_params):
    pad_width = int(np.ceil(image_params["N_PIXELS"] * 0.1)) + 1
    padded_image = np.pad(image, pad_width=pad_width)

    return padded_image


def apply_ctf(image, image_params, proc_params):
    def calc_ctf(n_pixels, pixel_size, amp, phase, b_factor):
        ctf = np.zeros((n_pixels, n_pixels), dtype=np.complex128)

        freq_pix_1d = np.fft.fftfreq(n_pixels, d=pixel_size)

        x, y = np.meshgrid(freq_pix_1d, freq_pix_1d)

        freq2_2d = x**2 + y**2
        imag = np.zeros_like(freq2_2d) * 1j

        env = np.exp(-b_factor * freq2_2d * 0.5)
        ctf = (
            amp * np.cos(phase * freq2_2d * 0.5)
            - np.sqrt(1 - amp**2) * np.sin(phase * freq2_2d * 0.5)
            + imag
        )

        return ctf * env / amp

    elecwavel = 0.019866
    phase = proc_params["DEFOCUS"] * np.pi * 2.0 * 10000 * elecwavel

    ctf = calc_ctf(
        image.shape[0],
        image_params["PIXEL_SIZE"],
        proc_params["AMP"],
        phase,
        proc_params["B_FACTOR"],
    )

    conv_image_ctf = np.fft.fft2(image) * ctf

    image_ctf = np.fft.ifft2(conv_image_ctf).real

    return image_ctf


def apply_random_shift(padded_image, image_params):
    shift_x = int(np.ceil(image_params["N_PIXELS"] * 0.1 * (2 * np.random.rand() - 1)))
    shift_y = int(np.ceil(image_params["N_PIXELS"] * 0.1 * (2 * np.random.rand() - 1)))

    pad_width = int(np.ceil(image_params["N_PIXELS"] * 0.1)) + 1

    low_ind_x = pad_width - shift_x
    high_ind_x = padded_image.shape[0] - pad_width - shift_x

    low_ind_y = pad_width - shift_y
    high_ind_y = padded_image.shape[0] - pad_width - shift_y

    shifted_image = padded_image[low_ind_x:high_ind_x, low_ind_y:high_ind_y]

    return shifted_image


def add_noise(img, proc_params):
    # mean_image = np.mean(img)
    std_image = np.std(img)

    mask = np.abs(img) > 0.5 * std_image

    signal_mean = np.mean(img[mask])
    signal_std = np.std(img[mask])

    noise_std = signal_std / np.sqrt(proc_params["SNR"])
    noise = np.random.normal(loc=signal_mean, scale=noise_std, size=img.shape)

    img_noise = img + noise

    return img_noise


def gaussian_normalize_image(image):
    mean_img = np.mean(image)
    std_img = np.std(image)

    return (image - mean_img) / std_img
