from typing import Any, Union, Dict
from gensim.corpora import Dictionary
from os.path import isfile

from kedro.io import AbstractDataSet

class DictionaryDataSet(AbstractDataSet):

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath)

    def __init__(
        self,
        filepath: str,
    ) -> None:
        """Creates a new instance of ``MmCorpusDataSet`` pointing to a concrete
        filepath.

        Args:
            filepath: path to a MmCorpus file.
        """
        self._filepath = filepath

    def _load(self) -> Dictionary:
        return Dictionary.load(self._filepath)

    def _save(self, dictionary: Dictionary) -> None:
        dictionary.save(self._filepath)

    def _exists(self) -> bool:
        return isfile(self._filepath)