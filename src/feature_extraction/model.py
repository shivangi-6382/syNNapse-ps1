import torch
import torch.nn as nn
import torchvision.models as models

class EmbeddingModel(nn.Module):
    """
    ResNet50 backbone + 512-d embedding head.
    Backbone frozen.
    Output embeddings are L2 normalized.
    """
    def __init__(self, emb_dim: int = 512):
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        for param in self.features.parameters():
            param.requires_grad = False

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, emb_dim),
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.embedding(x)
        return torch.nn.functional.normalize(x, dim=1)  # <- NO trailing comma