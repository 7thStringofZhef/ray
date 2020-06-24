# Model config fields
OCNET_DEFAULT_CONFIG = {
    'oc_num_options': 4,
    'oc_option_epsilon': 0.1
}

# Atari filters as in https://arxiv.org/pdf/1609.05140.pdf (Original OC network, Nature CNN)
OCNET_FILTERS = [
    [32, [8, 8], 4],
    [64, [4, 4], 2],
    [64, [3, 3], 1]
]
OCNET_DENSE = 512

# A3C Atari filters as in https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/17421/16660 (A2OC network)
A2OCNET_FILTERS = [
    [16, [8, 8], 4],
    [32, [4, 4], 2]
]
A2OCNET_DENSE = 256