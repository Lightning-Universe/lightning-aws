from lightning.app.storage.path import Path
from lightning_app import LightningWork
import subprocess

class LambdaPyTorchPredictor(LightningWork):

    def __init__(self, name: str):
        super().__init__()
        self.function_name = name

    def run(self, best_model_path: Path):
        subprocess.run(['mv', str(best_model_path), "model.pt"])
        subprocess.run(["docker", "build", "-t", "lightning-aws", "."])

if __name__ == "__main__":
    comp = LambdaPyTorchPredictor("my_function")
    comp.run("model.pt")