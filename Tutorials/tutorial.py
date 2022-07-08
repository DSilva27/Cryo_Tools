import numpy as np
import matplotlib.pyplot as plt

import cryo_tools

config = {}

config["IMAGES"] = {
    "N_PIXELS": 128,
    "PIXEL_SIZE": 1,
    "SIGMA": 1
    }

config["PROCESSING"] = {
    "SNR": 1,
    "DEFOCUS": 1.5
}

# Check that everything works

cryo_tools.utils.check_params(config)

atomic_coordinates = cryo_tools.utils.load_pdb("hsp90.pdb", filter="name CA")
atomic_coordinates = cryo_tools.simulating.rot_coordinates(atomic_coordinates)

image = cryo_tools.simulating.gen_img(atomic_coordinates, config["IMAGES"])

image = cryo_tools.processing.pad_image(image, config["IMAGES"])
image = cryo_tools.processing.apply_ctf(image, config["IMAGES"], config["PROCESSING"])
image = cryo_tools.processing.apply_random_shift(image, config["IMAGES"])
image = cryo_tools.processing.add_noise(image, config["PROCESSING"])

image = cryo_tools.processing.gaussian_normalize_image(image)

plt.imshow(image)
plt.show()
