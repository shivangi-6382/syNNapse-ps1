import os
import numpy as np
from tqdm import tqdm
from typing import List

from feature_extraction.encoder import ImageEncoder


class EmbeddingIndexer:
    """
    Builds embedding database for retrieval.
    """

    def __init__(self, model_path: str):
        self.encoder = ImageEncoder(model_path)

    def build_index(
        self,
        image_paths: List[str],
        labels: List[str] = None,
        save_dir: str = "artifacts"
    ):
        """
        Encode all images and save embeddings.

        Args:
            image_paths: list of image file paths
            labels: corresponding labels (optional)
            save_dir: directory to store .npy files
        """

        os.makedirs(save_dir, exist_ok=True)

        embeddings = []
        valid_paths = []
        valid_labels = []

        for idx, path in enumerate(tqdm(image_paths)):

            try:
                emb = self.encoder.encode_image(path)
                embeddings.append(emb[0])
                valid_paths.append(path)

                if labels is not None:
                    valid_labels.append(labels[idx])

            except Exception as e:
                print(f"Skipping {path} due to error: {e}")

        embeddings = np.array(embeddings)

        # Save artifacts
        np.save(os.path.join(save_dir, "embeddings.npy"), embeddings)
        np.save(os.path.join(save_dir, "image_paths.npy"), np.array(valid_paths))

        if labels is not None:
            np.save(os.path.join(save_dir, "labels.npy"), np.array(valid_labels))

        print("Index built successfully ✅")
        print("Total embeddings:", embeddings.shape)

        return embeddings