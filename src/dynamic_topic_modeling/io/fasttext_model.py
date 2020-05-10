
import fasttext 
from typing import Any, Dict
from kedro.io import AbstractDataSet

class FastTextModel(AbstractDataSet):

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

        model=fasttext.load_model(self._filepath)
        return model

    def _save(self, model) -> None:

        model.save_model(self._filepath)


    def _exists(self) -> bool:
        return isfile(self._filepath)