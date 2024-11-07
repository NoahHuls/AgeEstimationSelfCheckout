from torch.nn import Sequential, Conv2d, GELU, MaxPool2d, Flatten, Linear, Dropout, BatchNorm2d
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module
import torch
from torchvision import transforms
from collections import OrderedDict
from PIL import Image
import os
import pandas as pd
import cv2

class RGB2LAB(Module):
    def __init__(self, normalize=True):
        super(RGB2LAB, self).__init__()
        self.xyz = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                                 [0.2126729, 0.7151522, 0.0721750],
                                 [0.0193339, 0.1191920, 0.9503041]])
        self.epsilon = 6/29
        self.epsilon_cube = self.epsilon**3
        self.kappa = 903.3
        self.white_point = torch.tensor([0.95047, 1.0, 1.08883])
        self.normalize = normalize

    def forward(self, inputs):
        inputs = inputs / 255.0 if inputs.max() > 1.0 else inputs
        inputs = inputs.permute(0, 2, 3, 1)
        xyz = torch.einsum('bijc,cd->bijd', inputs, self.xyz.to(inputs.device))
        if self.normalize:
            xyz = xyz / self.white_point.to(inputs.device)
        def f(t):
            return torch.where(t > self.epsilon_cube, torch.pow(t, 1/3), (self.kappa * t + 16) / 116)

        f_xyz = f(xyz)
        L = 116 * f_xyz[..., 1] - 16
        a = 500 * (f_xyz[..., 0] - f_xyz[..., 1])
        b = 200 * (f_xyz[..., 1] - f_xyz[..., 2])

        lab = torch.stack([L, a, b], dim=-1)
        return lab.permute(0, 3, 1, 2)

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path =  self.dataframe.iloc[idx]['filepath']
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

class AgeEsitimationModel:
    def __init__(self, model_path: str = "./models", batch_size: int = 4):
        self.transform = self.__imageToTensor__()
        self.device = self.__getTensorDevice__()
        self.batch_size = batch_size
        self.model = None
        self.__loadModels__(model_path)
        if self.model is None:
            print(self.modelList)
            print("failed to load model")
            ValueError("No model found")
        else:
            print("Model loaded:")
            print(f"active model is: {self.activeModel}")

    def predict(self, imageDir: str) -> dict:
        imgs = os.listdir(imageDir)
        imgs = [f"{imageDir}/{img}" for img in imgs if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".png")]
        print(imgs)

        results = {"error": 0, "error_message": "", "predictions": [], 'over_25': False}

        imgs = pd.DataFrame(imgs, columns=['filepath'])
        val_dataset = CustomImageDataset(imgs, imageDir, transform=self.transform)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        preds = []
        for inputs in val_loader:
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs).squeeze().tolist()
                if type(outputs) is float:
                    # print("outputs is float")
                    preds += [outputs]
                else:
                    preds += outputs

        results['predictions'] = preds
        minVal = min(preds)
        preds.remove(minVal)
        if minVal > 25:
            results['over_25'] = True
        imgs = os.listdir(imageDir)
        print(imgs)
        if "zebra.png" in imgs:
            imgs.remove(f"zebra.png")
        imgs = [f"{imageDir}/{img}" for img in imgs if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".png")]
        for file in imgs:
            os.remove(file)
        print(results)
        return results

    def deleteImages(self, imageDir: str):
        files = os.listdir(imageDir)
        if "zebra.png" in files:
            files.remove(f"zebra.png")
        for file in files:
            os.remove(f"{imageDir}/{file}")

    def zebra(self, imageDir: str):
        os.remove(f"{imageDir}/zebra.png")
        img = os.listdir(imageDir)
        img = [f"{imageDir}/{img}" for img in img if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".png")]
        img = img[0]
        cvimg = cv2.imread(img)
        cv2.imwrite(f"{imageDir}/zebra.png", cvimg)
        del cvimg
        os.system(f"cp {img}.png {imageDir}/zebra.png")
        print(f"cp {imageDir}/../zebra/tmp.png {imageDir}/../zebra/zebra.png")

    def  __getTensorDevice__(self):
        if torch.backends.mps.is_available():
            print("Mx architecture detected")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print("Using CUDA GPU")
            return torch.device("cuda")
        print("Using CPU")
        return torch.device("cpu")

    def __imageToTensor__(self):
        return transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def setActiveModel(self, model_name: str) -> bool:
        found = False
        for model in self.modelList:
            if model['model'] == model_name:
                self.activeModel = model
                self.__loadTorchModel__(model['path'], model_name)
                found = True
        return found

    def getAvilableModels(self) -> dict:
        res =self.modelList
        print(res)
        return res

    def getActiveModel(self) -> dict:
        return self.activeModel

    def __loadModels__(self, model_path: str):
        self.modelList = []
        for file in os.listdir(model_path):
            if file.endswith(".pt"):
                self.modelList.append({'path': f"{model_path}", 'type': 'torch', 'model': file})

        self.activeModel = self.modelList[0]
        self.__loadTorchModel__(model_path, self.activeModel['model'])

    def __loadTorchModel__(self, model_path: str, model_name: str):
        self.model = self.__selectModel__(model_name)
        if self.model is None:
            print(f"Model {model_name} not found")
            print("failed in loadTorchModel")
            return None
        match (model_name):
            case "saved_weights_2024_10_02-19_34_03.pt":
                self.__fixWeights__(torch.load(f"{model_path}/saved_weights_2024_10_02-19_34_03.pt", weights_only=True, map_location=torch.device(self.device)))
            case "saved_weights_2024_10_02-18_47_29.pt":
                self.__fixWeights__(torch.load(f"{model_path}/saved_weights_2024_10_02-18_47_29.pt", weights_only=True, map_location=torch.device(self.device)))
            case "saved_weights_2024_10_02-20_27_24.pt":
                self.__fixWeights__(torch.load(f"{model_path}/saved_weights_2024_10_02-20_27_24.pt", weights_only=True, map_location=torch.device(self.device)))
            case "saved_weights_2024_10_02-23_7_48.pt":
                self.__fixWeights__(torch.load(f"{model_path}/saved_weights_2024_10_02-23_7_48.pt", weights_only=True, map_location=torch.device(self.device)))
            case "saved_weights_2024_10_02-21_36_36.pt":
                self.__fixWeights__(torch.load(f"{model_path}/saved_weights_2024_10_02-21_36_36.pt", weights_only=True, map_location=torch.device(self.device)))
            case _:
                print(f"Model {model_name} not found")
                print("model in could not load weights")
                return None
        self.model.to(self.device)
        self.model.eval()

    def __fixWeights__(self, weights):
        wkeys = weights.keys()
        mkeys = self.model.state_dict().keys()
        if len(wkeys) != len(mkeys):
            print("Number of keys in weights and model do not match")
            print("Weights: ", len(wkeys))
            print("Model: ", len(mkeys))
            ValueError("Number of keys in weights and model do not match")
        m2w = dict(zip(mkeys, wkeys))

        new_weights = OrderedDict()
        for key in mkeys:
            new_weights[key] = weights[m2w[key]]
        self.model.load_state_dict(new_weights)

    def __selectModel__(self,archtecture: str):
        if archtecture == "saved_weights_2024_10_02-19_34_03.pt":
            return Sequential(
                        Conv2d(3, 64, 3, stride=1, padding=1),
                        GELU(),
                        Conv2d(64, 64, 3, stride=1, padding=1),
                        GELU(),
                        MaxPool2d(2),
                        Conv2d(64, 128, 3, stride=1, padding=1),
                        GELU(),
                        Conv2d(128, 128, 3, stride=1, padding=1),
                        GELU(),
                        MaxPool2d(2),
                        Flatten(),
                        Linear(128*(200//4)*(200//4), 128),
                        GELU(),
                        Linear(128, 128),
                        GELU(),
                        Linear(128, 1)
                    )
        elif archtecture == "saved_weights_2024_10_02-18_47_29.pt":
            return Sequential(
                        RGB2LAB(True),
                        Conv2d(3, 64, 3, stride=1, padding=1),
                        GELU(),
                        Conv2d(64, 64, 3, stride=1, padding=1),
                        GELU(),
                        MaxPool2d(2),
                        Conv2d(64, 128, 3, stride=1, padding=1),
                        GELU(),
                        Conv2d(128, 128, 3, stride=1, padding=1),
                        GELU(),
                        MaxPool2d(2),
                        Flatten(),
                        Linear(128*(200//4)*(200//4), 128),
                        GELU(),
                        Linear(128, 128),
                        GELU(),
                        Linear(128, 1)
                    )
        elif archtecture == "saved_weights_2024_10_02-20_27_24.pt":
            return Sequential(
                        BatchNorm2d(3),
                        Conv2d(3, 64, 3, stride=1, padding=1),
                        GELU(),
                        Conv2d(64, 64, 3, stride=1, padding=1),
                        GELU(),
                        MaxPool2d(2),
                        Conv2d(64, 128, 3, stride=1, padding=1),
                        GELU(),
                        Conv2d(128, 128, 3, stride=1, padding=1),
                        GELU(),
                        MaxPool2d(2),
                        Flatten(),
                        Dropout(0.3),
                        Linear(128*(200//4)*(200//4), 512),
                        GELU(),
                        Dropout(0.3),
                        Linear(512, 128),
                        GELU(),
                        Linear(128, 1)
                    )
        elif archtecture == "saved_weights_2024_10_02-23_7_48.pt":
            return Sequential(
                        RGB2LAB(False),
                        BatchNorm2d(3),
                        Conv2d(3, 64, 3, stride=1, padding=1),
                        GELU(),
                        Conv2d(64, 64, 3, stride=1, padding=1),
                        GELU(),
                        MaxPool2d(2),
                        Conv2d(64, 128, 3, stride=1, padding=1),
                        GELU(),
                        Conv2d(128, 128, 3, stride=1, padding=1),
                        GELU(),
                        MaxPool2d(2),
                        Flatten(),
                        Dropout(0.3),
                        Linear(128*(200//4)*(200//4), 512),
                        GELU(),
                        Dropout(0.3),
                        Linear(512, 128),
                        GELU(),
                        Linear(128, 1)
                    )
        elif archtecture == "saved_weights_2024_10_02-21_36_36.pt":
            return Sequential(
                        RGB2LAB(),
                        BatchNorm2d(3),
                        Conv2d(3, 64, 3, stride=1, padding=1),
                        GELU(),
                        Conv2d(64, 64, 3, stride=1, padding=1),
                        GELU(),
                        MaxPool2d(2),
                        Conv2d(64, 128, 3, stride=1, padding=1),
                        GELU(),
                        Conv2d(128, 128, 3, stride=1, padding=1),
                        GELU(),
                        MaxPool2d(2),
                        Flatten(),
                        Dropout(0.3),
                        Linear(128*(200//4)*(200//4), 512),
                        GELU(),
                        Dropout(0.3),
                        Linear(512, 128),
                        GELU(),
                        Linear(128, 1)
                    )
        print(f"archtecture not found for {archtecture}")
        ValueError(f"archtecture not found for {archtecture}")
        return None