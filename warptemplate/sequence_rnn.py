"""Small recurrent neural network used by the Warp SuperNNova backend.

The network maps an irregular light curve to seven unnormalized class scores
(``logits``).  The backend first groups observations into epochs and represents every
epoch by a fixed feature vector.  A batch is then passed in PyTorch's packed-sequence
format so the LSTM can process light curves of different lengths without interpreting
zero padding as observations.

The tensor flow is:

``packed epochs -> LSTM -> one vector per object -> dropout -> class logits``.

The step that produces one vector per object is configurable: the model can use all
final hidden states (``standard``), the mean over valid epochs (``mean``), or a learned
weighted mean (``attention``).  Optional object-level context such as truth redshift
is concatenated only after this temporal encoding; it is not repeated at every epoch.

Data preparation, normalization, training, calibration, and checkpoint handling live
in :mod:`warptemplate.supernnova_backend`.  Keeping this module focused makes the
mathematical model easier to inspect independently of the experiment infrastructure.
"""

from __future__ import annotations

from typing import Any

import torch


class WarpSequenceRNN(torch.nn.Module):
    """Encode variable-length light curves and predict one class per object.

    Parameters
    ----------
    input_size:
        Number of per-epoch features passed through the recurrent encoder.  An
        object-level auxiliary feature is deliberately not included in this count.
    settings:
        Namespace-like configuration supplied by the backend.  It defines the LSTM
        width and depth, dropout, directionality, pooling method, class count, and
        optional ``auxiliary_dim``.

    Notes
    -----
    The output contains logits rather than probabilities.  During training these go
    directly into cross-entropy loss; during prediction the backend applies softmax
    (and, when enabled, a fitted calibration temperature).
    """

    def __init__(self, input_size: int, settings: Any):
        """Construct the recurrent encoder, sequence pooling, and output head."""
        super().__init__()
        self.layer_type = settings.layer_type
        self.output_size = settings.nb_classes
        self.hidden_size = settings.hidden_dim
        self.num_layers = settings.num_layers
        self.dropout = settings.dropout
        self.bidirectional = settings.bidirectional
        self.rnn_output_option = settings.rnn_output_option
        self.auxiliary_dim = int(getattr(settings, "auxiliary_dim", 0))

        # A bidirectional recurrent layer emits a forward and backward vector.  The
        # standard representation retains the final state of every stacked layer;
        # temporal pooling operates only on the output of the final layer.
        direction_count = 2 if self.bidirectional else 1
        if self.rnn_output_option == "standard":
            representation_size = self.hidden_size * direction_count * self.num_layers
        elif self.rnn_output_option in {"mean", "attention"}:
            representation_size = self.hidden_size * direction_count
        else:
            raise ValueError(f"Unsupported RNN pooling: {self.rnn_output_option!r}")

        self.rnn_layer = getattr(torch.nn, self.layer_type.upper())(
            input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        # Attention learns one scalar importance score per valid epoch.  It is needed
        # only for attention pooling; mean and final-state pooling add no parameters.
        self.attention_layer = (
            torch.nn.Linear(representation_size, 1, bias=False)
            if self.rnn_output_option == "attention"
            else None
        )
        self.output_dropout_layer = torch.nn.Dropout(self.dropout)
        self.output_layer = torch.nn.Linear(
            representation_size + self.auxiliary_dim,
            self.output_size,
        )

    def _pool_sequence(self, packed_output: Any) -> torch.Tensor:
        """Reduce valid recurrent outputs to one representation per object.

        Parameters
        ----------
        packed_output:
            Packed LSTM output for a length-sorted batch.  After unpacking its shape
            is ``(max_epochs, batch, hidden_size * directions)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, hidden_size * directions)``.  Padded epochs do
            not contribute to either mean or attention pooling.
        """
        padded, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
        if self.rnn_output_option == "mean":
            # pad_packed_sequence fills invalid positions with zero, so division by
            # each object's true length produces an exact mean over observed epochs.
            return padded.sum(0) / lengths.unsqueeze(-1).to(padded.device)

        # Padding receives exactly zero attention, so short sequences are not diluted.
        scores = self.attention_layer(padded).squeeze(-1)
        positions = torch.arange(padded.shape[0], device=padded.device).unsqueeze(1)
        mask = positions < lengths.to(padded.device).unsqueeze(0)
        weights = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=0)
        return (padded * weights.unsqueeze(-1)).sum(0)

    def forward(self, sequence: Any, auxiliary: torch.Tensor | None = None) -> torch.Tensor:
        """Return class logits for a length-sorted packed light-curve batch.

        ``sequence`` is a PyTorch ``PackedSequence`` created by the backend collator.
        ``auxiliary``, when configured, has shape ``(batch, auxiliary_dim)`` and is in
        the same length-sorted order.  The returned tensor has shape
        ``(batch, number_of_classes)``; it remains in sorted order so the caller can
        align it with targets or undo the sort for prediction output.
        """
        packed_output, hidden = self.rnn_layer(sequence)
        if self.rnn_output_option == "standard":
            # LSTM returns (hidden_state, cell_state); only the hidden state is used.
            # Flattening layers and directions retains their separate final summaries.
            hidden_state = hidden[0] if self.layer_type == "lstm" else hidden
            representation = hidden_state.permute(1, 2, 0).contiguous()
            representation = representation.view(representation.shape[0], -1)
        else:
            representation = self._pool_sequence(packed_output)

        if self.auxiliary_dim:
            if auxiliary is None or auxiliary.shape[1] != self.auxiliary_dim:
                raise ValueError("Model requires the configured late-fusion features")
            representation = torch.cat((representation, auxiliary), dim=1)
        elif auxiliary is not None:
            raise ValueError("Auxiliary features were passed to a photometry-only model")

        return self.output_layer(self.output_dropout_layer(representation))
