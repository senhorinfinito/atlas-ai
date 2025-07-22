from typing import Any, Dict, List, Union

from transformers import AutoProcessor, AutoModel
from tqdm.auto import tqdm
import torch
import pyarrow as pa
from PIL import Image
import io


DEFAULT_MODEL_MAP = {
    "text": "sentence-transformers/all-mpnet-base-v2",
    "image": "openai/clip-vit-large-patch14",
}


class Vectorizer:
    """A class to manage vectorization of data."""

    def __init__(self, model_name: str = None, modality: str = "text"):
        """
        Initializes the Vectorizer.
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
            from transformers import pipeline
            self.pipeline = pipeline("feature-extraction", model=self.model_name)

    def vectorize_batch(
        self, data: List[Any], batch_size: int = 32
    ) -> List[List[float]]:
        """
        Vectorizes a batch of data.
        """
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(data), batch_size), desc="Vectorizing data"):
                batch = data[i : i + batch_size]
                if self.modality == "image":
                    batch = [Image.open(io.BytesIO(d)) for d in batch]
                    inputs = self.processor(images=batch, return_tensors="pt")
                    image_features = self.model.get_image_features(**inputs)
                    embeddings.extend(image_features.cpu().numpy().tolist())
                else:
                    batch_embeddings = self.pipeline(batch)
                    for emb in batch_embeddings:
                        # The pipeline returns a list of token embeddings. We average them.
                        # But first, we need to handle the nesting which can be [[...]]
                        flat_emb = emb
                        while isinstance(flat_emb, list) and len(flat_emb) == 1:
                            flat_emb = flat_emb[0]
                        
                        # Now, flat_emb is a list of token embeddings. Average them.
                        if isinstance(flat_emb[0], list):
                            # It's a list of lists (tokens), so we average
                            num_tokens = len(flat_emb)
                            if num_tokens > 0:
                                sum_vec = [sum(dim) for dim in zip(*flat_emb)]
                                avg_vec = [s / num_tokens for s in sum_vec]
                                embeddings.append(avg_vec)
                        else:
                            # It's already a sentence embedding
                            embeddings.append(flat_emb)

        return embeddings

    def vectorize(
        self, data: Union[List[Any], pa.Array, pa.ChunkedArray], batch_size: int = 32
    ) -> pa.FixedSizeListArray:
        """
        Vectorizes the given data and returns it as a pyarrow FixedSizeListArray.
        """
        if isinstance(data, (pa.Array, pa.ChunkedArray)):
            data = data.to_pylist()

        embeddings = self.vectorize_batch(data, batch_size)
        
        if not embeddings:
            return pa.array([], type=pa.list_(pa.float32()))

        dimension = len(embeddings[0])
        flat_embeddings = [item for sublist in embeddings for item in sublist]
        
        return pa.FixedSizeListArray.from_arrays(
            pa.array(flat_embeddings, type=pa.float32()), dimension
        )