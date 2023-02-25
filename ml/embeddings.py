from typing import List

from InstructorEmbedding import INSTRUCTOR


class EmbeddingModel:
    def __init__(self, model_name: str, device: str):
        self.model = INSTRUCTOR(model_name_or_path=model_name, device=device)

    def get_embeddings(self, input: List[str], instruction: str) -> List[List[float]]:
        embeddings = self.model.encode(
            sentences=[[instruction, text] for text in input]
        )
        return embeddings.tolist()
