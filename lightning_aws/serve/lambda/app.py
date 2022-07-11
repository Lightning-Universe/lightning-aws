import json
import time
import torch
import io
import base64
from PIL import Image
import traceback
from torchvision.transforms import ToTensor


model = torch.jit.load("model.pt")
for param in model.parameters():
    param.requires_grad = False

transform = ToTensor()

def handler(event, context):
    trace = None
    start_time = time.time()
    try:
        image = event.get("body")
        if isinstance(image, str):
            # if the image is a string of bytesarray.
            image = base64.b64decode(image)

        # If the image is sent as bytesarray
        if isinstance(image, (bytearray, bytes)):
            image = Image.open(io.BytesIO(image))
        else:
            # if the image is a list
            image = torch.FloatTensor(image)

        image = transform(image).unsqueeze(0)
        predictions = model(image).argmax().item()
    except Exception as e:
        trace = traceback.format_exc()
        predictions = None

    return {
        'statusCode': 200,
        'body': json.dumps({'duration': time.time() - start_time, "predictions": predictions, "trace": trace})
    }
