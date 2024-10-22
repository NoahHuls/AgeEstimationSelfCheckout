import torch
from tensorflow.keras.models import load_model
import os
import cv2

class AgeEsitimationModel:
    def __init__(self, model_path: str = "./models"):
        self.__loadModels__(model_path)
        self

    def __loadModels__(self, model_path: str):
        modelList = os.listdir(model_path)
        self.models = []
        for model in modelList:
            if model.endswith(".pt"):
                self.models.append({"model": f"{model_path}/{model}", "type": "torch"})
            elif model.endswith(".keras"):
                self.models.append({"model": f"{model_path}/{model}", "type": "tensorflow"})
        self.model = torch.load(self.models[0]["model"]) if self.models[0]["type"] == "torch" else load_model(self.models[0]["model"])
        if self.models[0]["type"] == "torch":
            self.model.eval()

        def selectModel(self, idx: int) -> bool:
            if idx >= len(self.models):
                return False
            modelstr = self.models[idx]["model"]
            self.model = torch.load(modelstr) if self.models[idx]["type"] == "torch" else load_model(modelstr)
            if self.models[idx]["type"] == "torch":
                self.model.eval()
            return True

    def getModelList(self) -> list:
        return self.models

    def predict(self, images):
        # todo: added face detection
        for image in images:
            image = cv2.imread(image)
            image = cv2.resize(image, (224, 224))
            image = torch.tensor(image)
            image /= 255
            return self.model(image)

