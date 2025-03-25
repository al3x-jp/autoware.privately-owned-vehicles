#! /usr/bin/env python3

#
# Based on/modified version of "prepareData.py" from here: "oneDNN/examples/network/prepareData.py"
#

# Dependancies
import os
from sys import argv

import numpy as np
import torch

#
# Main Program Entry Point
#
if __name__ == "__main__":

    if len(argv) < 2:
        raise "USAGE: ./extract_pretrained_data.py <path-to-*.pth-file>"

    # Location to write data files
    model_file = argv[1]

    # Load saved exported *.pth file
    model = torch.load  (   model_file, 
                            map_location=torch.device("cpu"),
                            weights_only=False
                        )

    # Get a list of layers in the model
    layer_info = list(model.keys())

    # Create a sub-directory for data files
    if not os.path.exists("./pretrained-data"):
        os.mkdir("./pretrained-data")

    # Iterate over layers and extract data
    for name in layer_info:
        dim = model[name].shape
        to_write = model[name].detach().numpy().reshape(dim).squeeze().T
        with open("pretrained-data/" + name.lower() + ".bin", "wb") as f:
            to_write.tofile(f)
        print("INFO - Generating: ./pretrained-data/" + name.lower() + ".bin")
#
# End of File
#