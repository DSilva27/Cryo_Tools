def check_params(config):

    # Sections
    for section in ["IMAGES", "PROCESSING"]:
        assert (
            section in config.keys()
        ), f"Please provide section {section} in config.ini"

    image_params = config["IMAGES"]
    proc_params = config["PROCESSING"]

    # Images
    for key in ["N_PIXELS", "PIXEL_SIZE", "SIGMA"]:
        assert key in image_params.keys(), f"Please provide a value for {key}"

    # Processing
    for key in ["DEFOCUS", "SNR"]:
        assert key in proc_params.keys(), f"Please provide a value for {key}"

    return
