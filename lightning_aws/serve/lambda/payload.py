import base64
import json
import struct

import numpy as np


def create_payload():
    width = 28
    height = 28
    data = np.random.randint(low=0, high=255, size=(1, width, height, 3)).flatten()
    #data = base64.b64encode(
    #    struct.pack("<{}B".format(len(data)), *(data.tolist()))
    #).decode()

    #payload = str({"body": data}).replace("'", '"')
    payload = {"body": data.tolist()}

    with open("payload.json", "w") as file:
        json.dump(payload, file)


if __name__ == "__main__":
    create_payload()