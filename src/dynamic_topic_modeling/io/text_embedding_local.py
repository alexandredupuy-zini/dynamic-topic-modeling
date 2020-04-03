from typing import Any, Dict
from kedro.io import AbstractDataSet
import numpy as np 
import torch

class TextEmbeddingDataset(AbstractDataSet):

    def __init__(
        self,
        filepath: str,
    ) -> None:
        """Creates a new instance of ``TextEmbeddingDataset`` pointing to a concrete
        filepath.

        Args:
            filepath: path to a glove embedding file.
        """
        self._filepath = filepath

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath)



    def _load(self) -> dict :

        vectors = {}
        with open(self._filepath, 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                vect = np.array(line[1:]).astype(np.float)
                vectors[word] = vect
        return vectors

    def _save(self, data: str,) -> None:

        with open(self._filepath,'wb') as fs_file:
            fs_file.write(data)


    def _exists(self) -> bool:
        return isfile(self._filepath)