from typing import Any, Dict, List, Optional

import torch

from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.training.metrics.f1_measure import F1Measure

@Model.register("cwi_multilingual")
class NeuralMutilingualCWI(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 context_layer: Seq2SeqEncoder,
                 complex_word_feedforward: FeedForward,
                 lexical_dropout: float = 0.2,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super(NeuralMutilingualCWI, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer

        self._complex_word_scorer = torch.nn.Sequential(complex_word_feedforward,
                                                        torch.nn.Linear(complex_word_feedforward.get_output_dim(), 1))

        self._target_word_extractor = EndpointSpanExtractor(context_layer.get_output_dim(), combination="x,y")

        self._loss = torch.nn.BCELoss()
        self._metric = F1Measure(1)

        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                target_word: torch.IntTensor,
                gold_label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        tokens:

        target_word:
            (batch_size, 2)
        gold_label:
            (batch_size)
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containing the original words in the sentence to be tagged under a 'words' key.

        Returns
        -------
        An output dictionary consisting of:

        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        # Shape: (batch_size, sentence_length, embedding_size)
        tokens_embeddings = self._lexical_dropout(self._text_field_embedder(tokens))

        # Shape: (batch_size, sentence_length)
        tokens_mask = util.get_text_field_mask(tokens).float()

        # Shape: (batch_size, sentence_length, encoding_dim)
        contextualized_embeddings = self._context_layer(tokens_embeddings, tokens_mask)

        # Shape: (batch_size, 2 * encoding_dim)
        target_word_embeddings = self._target_word_extractor(contextualized_embeddings, target_word)

        # Shape: (batch_size)
        complex_word_scores = self._complex_word_scorer(target_word_embeddings).squeeze(-1)
        complex_word_predictions = complex_word_scores.sigmoid()

        output_dict = {"predictions": complex_word_predictions}

        if gold_label is not None:
            output_dict["loss"] = self._loss(complex_word_predictions, gold_label.float())
            self._metric(complex_word_predictions, gold_label)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1 = self._metric.get_metric(reset)
        return {"precision": precision, "recall": recall, "f1": f1}
