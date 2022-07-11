import base64
import requests

with open("./test_data/0.png", "rb") as f:
    imgstr = base64.b64encode(f.read()).decode("UTF-8")

body = {"body": imgstr}
#resp = handler(body, {})
resp = requests.post("http://localhost:9000/2015-03-31/functions/function/invocations", json=body)
print(resp.json())