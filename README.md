# Lightning AWS

This library provides components ready to be used for creating AWS resources from [Lightning Apps](https://lightning.ai/lightning-docs/).

## Currently available

```python
from lightning_aws.serve.sagemaker import SagemakerPyTorchPredictor

comp = SagemakerPyTorchPredictor("my_name", instance_type="ml.m4.xlarge")
comp.run("model.pt")
```