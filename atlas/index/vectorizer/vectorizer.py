from typing import Any, Dict, List

from transformers import AutoProcessor, AutoModel
from tqdm.auto import tqdm
import torch


DEFAULT_MODEL_MAP = {
    "text": "sentence-transformers/all-mpnet-base-v2",
    "image": "openai/clip-vit-large-patch14",
}


class Vectorizer:
    """A class to manage vectorization of data."""

    def __init__(self, model_name: str = None, modality: str = "text"):
        """
        Initializes the Vectorizer.

        Args:
            model_name (str, optional): The name of the Hugging Face model to use.
                                      If not provided, a default model is used
                                      based on the modality.
            modality (str): The data modality ('text' or 'image').
        """
        if model_name is None:
            model_name = DEFAULT_MODEL_MAP.get(modality)
            if model_name is None:
                raise ValueError(f"No default model found for modality '{modality}'")

        self.model_name = model_name
        self.modality = modality
        print(f"Initializing vectorizer with model: {self.model_name}")
        
        if self.modality == "image":
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
        else:
            # For text, the feature-extraction pipeline is still the most straightforward
            from transformers import pipeline
            self.pipeline = pipeline("feature-extraction", model=self.model_name)


    def vectorize(self, data: List[Any], batch_size: int = 32) -> List[List[float]]:
        """
        Vectorizes the given data.

        Args:
            data (List[Any]): A list of data to vectorize.
            batch_size (int): The batch size to use for vectorization.

        Returns:
            A list of embeddings.
        """
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(data), batch_size), desc="Vectorizing data"):
                batch = data[i : i + batch_size]
                if self.modality == "image":
                    inputs = self.processor(images=batch, return_tensors="pt")
                    image_features = self.model.get_image_features(**inputs)
                    embeddings.extend(image_features.cpu().numpy().tolist())
                else:
                    batch_embeddings = self.pipeline(batch)
                    if isinstance(batch_embeddings[0], list):
                        embeddings.extend([emb[0] for emb in batch_embeddings])
                    else:
                        embeddings.extend(batch_embeddings)
        return embeddings
