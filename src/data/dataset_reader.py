import pandas as pd

from typing import Dict, Iterable, Any

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.fields import Field, TextField, SpanField, LabelField, MetadataField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


class CWISharedTaskDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(True)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # pylint: disable=arguments-differ
        with open(file_path, encoding="utf-8") as file:
            fieldnames = ['hit_id', 'sentence', 'start_offset', 'end_offset', 'target_word', 'native_annots',
                          'nonnative_annots', 'native_complex', 'nonnative_complex', 'gold_label', 'gold_prob']

            for sentence_info in pd.read_csv(file, names=fieldnames, sep="\t", iterator=True):
                yield self.text_to_instance(sentence_info['sentence'], sentence_info['start_offset'],
                                            sentence_info['end_offset'], sentence_info['gold_label'])

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentence: str,
                         start_offset: int,
                         end_offset: int,
                         gold_label: int) -> Instance:
        # pylint: disable=arguments-differ
        tokens_field = TextField([Token(word) for word in sentence.split()], self._token_indexers)
        target_word_field = SpanField(start_offset, end_offset, tokens_field)
        gold_label_field = LabelField(gold_label)
        metadata_field = MetadataField({"original_text": sentence})

        fields: Dict[str, Field] = {'tokens': tokens_field,
                                    'target_word': target_word_field,
                                    'gold_label': gold_label_field,
                                    'metadata': metadata_field}

        return Instance(fields)
