from __future__ import annotations

import torch
from torch import nn
from typing import Any

__all__ = [
    "TextEncoder",
]


class TextEncoder(nn.Module):
    """A simple text encoder that embeds tokens and feeds them to an LSTM.

    The final hidden state of the last LSTM layer is returned as the sentence
    representation.  This implementation intentionally keeps *nn.LSTM* outside
    of *nn.Sequential* to avoid the tuple output incompatibility that arises
    when LSTM is used within a Sequential container.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # If bidirectional, final hidden size doubles (fwd + bwd states).
        self._out_dim = hidden_dim * (2 if bidirectional else 1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def output_dim(self) -> int:
        """Dimensionality of the representation returned by *forward*."""
        return self._out_dim

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Parameters
        ----------
        tokens : torch.Tensor
            LongTensor of shape *(batch, seq_len)* containing token indices.
        Returns
        -------
        torch.Tensor
            Tensor of shape *(batch, hidden_dim \* num_directions)* representing
            the final hidden state of the LSTM.
        """
        x = self.embedding(tokens)  # (B, T, E)
        outputs, (h_n, _c_n) = self.lstm(x)
        # h_n: (num_layers * num_directions, B, H)
        last_layer_h = h_n[-1]  # (B, H) for uni-dir; (B, H) for last dir
        if self.lstm.bidirectional:
            # If bidirectional, concatenate forward & backward hidden states
            fwd = h_n[-2]  # forward final
            bwd = h_n[-1]  # backward final
            last_layer_h = torch.cat([fwd, bwd], dim=-1)
        return last_layer_h


# ---------------------------------------------------------------------------
# Factory helper (optional convenience)
# ---------------------------------------------------------------------------

def build_text_encoder(cfg: Any) -> TextEncoder:  # noqa: ANN401
    """Utility helper mirroring the old *_build_text_encoder* API.

    The *cfg* object is expected to expose the attributes referenced below.
    """
    return TextEncoder(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=getattr(cfg, "lstm_layers", 1),
        dropout=getattr(cfg, "dropout", 0.1),
        bidirectional=getattr(cfg, "bidirectional", False),
    )