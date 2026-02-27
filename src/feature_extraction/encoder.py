import torch
import numpy as np
from PIL import Image

from .model import EmbeddingModel
from .transforms import get_inference_transform


class ImageEncoder:
    """
    Image -> Embedding pipeline.
    """

    def __init__(self, model_path: str, device: str = None):

        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize architecture
        self.model = EmbeddingModel()

        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.transform = get_inference_transform()

    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Encode image from path.
        Returns: numpy array of shape (1, 512)
        """

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(image)

        return embedding.cpu().numpy()

    def encode_pil_image(self, pil_image: Image.Image) -> np.ndarray:
        """
        Encode already loaded PIL image (useful for API).
        """

        image = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(image)

        return embedding.cpu().numpy()