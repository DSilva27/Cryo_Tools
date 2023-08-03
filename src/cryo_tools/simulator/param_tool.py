import json

class CryoParams:

    def __init__(self, config: dict = None, config_file: str = None):

        if config is None and config_file is None:
            raise ValueError("Either config or config_file must be provided.")
        
        elif config is not None and config_file is not None:
            raise ValueError("Only one of config or config_file must be provided.")
        
        elif config is not None:
            self.init_from_config_(config)

        else:
            self.init_from_config_file_(config_file)

    def validate_config_(self, config: dict):

        required_keys = ["box_size", "pixel_size", "res", "noise_radius_mask", "snr", "defocus", "bfactor", "amp"]

        for key in required_keys:
            if key not in config.keys():
                raise ValueError(f"Config must contain key {key}.")
            
        return

    def init_from_config_(self, config: dict):

        assert isinstance(config, dict), "Config must be a dictionary."
        self.validate_config_(config)

        self.box_size = config["box_size"]
        self.pixel_size = config["pixel_size"]
        self.res = config["res"]
        self.noise_radius_mask = config["noise_radius_mask"]
        self.snr = config["snr"]
        self.defocus = config["defocus"]
        self.bfactor = config["bfactor"]
        self.amp = config["amp"]

        return
    
    def init_from_config_file_(self, config_file: str):

        assert isinstance(config_file, str), "Config file must be a string."
        config = json.load(open(config_file, "r"))

        self.init_from_config_(config)

        return