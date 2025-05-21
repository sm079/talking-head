import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

from tqdm import tqdm
from pathlib import Path
from face_parse.model import BiSeNet

class FaceParser():

    def __init__(self):
        self.model = BiSeNet(n_classes=19)
        self.model.load_state_dict(torch.load(r"./src/models/79999_iter.pth"))
        self.model.eval()
        self.model.to("cuda")

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.parsing_dict = {
            "neutral"      : False,
            "skin"         : True ,
            "left eyebrow" : True ,
            "right eyebrow": True ,
            "left eye"     : True ,
            "right eye"    : True ,
            "eyeglasses"   : True ,
            "left ear"     : False,
            "right ear"    : False,
            "earrings"     : False,
            "nose"         : True ,
            "mouth"        : True,
            "upper lip"    : True ,
            "lower lip"    : True ,
            "neck"         : False,
            "necklace"     : False,
            "clothes"      : False,
            "hair"         : False,
            "hat"          : False
        }

        self.mask = torch.tensor([value * 255 for value in self.parsing_dict.values()])

    @torch.no_grad()
    def get_mask(self, image):
        img = torch.unsqueeze(self.to_tensor(image), 0).cuda()
        out = self.model(img)[0]
        out = torch.argmax(out, dim=1)

        for i in range(len(self.mask)):
            out[out == i] = self.mask[i]

        return out.squeeze(0).cpu().numpy()
    
    def save_face_masks_from_folder(self, image_folder: str, output_folder: str):
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        images = list(Path(image_folder).iterdir())

        for image_path in tqdm(images, desc="Face Parsing"):
            image = cv2.imread(str(image_path))
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA) if image.shape[:2] != (512, 512) else image
            mask = self.get_mask(image)
            cv2.imwrite(f"{output_folder}/{image_path.stem}.jpg", mask)


if __name__ == "__main__":
    parser = FaceParser()
    parser.save_face_masks_from_folder(
        image_folder =r"experiments/001/src/aligned",
        output_folder=r"experiments/001/src/masks"
        )