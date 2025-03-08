from PIL import Image
import numpy as np
from typing import Tuple
import torch
import launch, utilities, settings, devices
from paths import paths

class WaifuDiffusionTagger:
    # Adapted from original huggingface implementation
    def __init__(self, model_name, model_filename="model.onnx", label_filename="selected_tags.csv"):
        self.MODEL_FILENAME = model_filename
        self.LABEL_FILENAME = label_filename
        self.MODEL_REPO = model_name
        self.model = None
        self.labels = []

    def load(self):
        import huggingface_hub
        import onnxruntime as ort

        if not self.model:
            path_model = huggingface_hub.hf_hub_download(
                self.MODEL_REPO, self.MODEL_FILENAME, cache_dir=paths.setting_model_path
            )

            available_providers = ort.get_available_providers()
            if settings.current.interrogator_use_cpu:
                providers = ["CPUExecutionProvider"]
            else:
                providers = [p for p in ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
                             if p in available_providers]

            def check_available_device():
                if torch.cuda.is_available():
                    return "cuda"
                elif launch.is_installed("torch-directml"):
                    try:
                        import torch_directml
                        torch_directml.device()
                    except:
                        pass
                    else:
                        return "directml"
                return "cpu"

            if not launch.is_installed("onnxruntime"):
                dev = check_available_device()
                if dev == "cuda":
                    launch.run_pip("install -U onnxruntime-gpu", "onnxruntime-gpu")
                elif dev == "directml":
                    launch.run_pip("install -U onnxruntime-directml", "onnxruntime-directml")
                else:
                    print("No compatible acceleration device found; installing CPU-only onnxruntime.")
                    launch.run_pip("install -U onnxruntime", "onnxruntime for CPU")

            print(f"Running ONNX on {ort.get_device()}")
            self.model = ort.InferenceSession(path_model, providers=providers)

        path_label = huggingface_hub.hf_hub_download(
            self.MODEL_REPO, self.LABEL_FILENAME
        )

        import pandas as pd
        self.labels = pd.read_csv(path_label)["name"].tolist()

    def unload(self):
        if not settings.current.interrogator_keep_in_memory:
            self.model = None
            devices.torch_gc()

    def apply(self, image: Image.Image):
        if not self.model:
            return []

        _, height, width, _ = self.model.get_inputs()[0].shape

        # Resize with edge pixel repetition instead of filling with white
        image = utilities.resize(image, (width, height))
        image_np = np.array(image, dtype=np.float32)

        # Convert PIL RGB to OpenCV BGR
        image_np = image_np[:, :, ::-1]
        image_np = np.expand_dims(image_np, 0)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        probs = self.model.run([label_name], {input_name: image_np})[0]

        labels: list[Tuple[str, float]] = list(zip(self.labels, probs[0].astype(float)))
        return labels
