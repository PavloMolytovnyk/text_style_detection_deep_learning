"""Every predictor in this project is an end-to-end trainable neural network."""

import torch
from torch import nn
from torch.nn import functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, channels=96, kernels=(3, 4, 5), dropout=0.4, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.convolutions = nn.ModuleList([nn.Conv1d(embedding_dim, channels, k) for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.style_head = nn.Linear(channels * len(kernels), 5)
        self.tone_head = nn.Linear(channels * len(kernels), 5)

    def forward(self, input_ids, lengths):
        embeddings = self.embedding(input_ids).transpose(1, 2)
        pooled = []
        for conv in self.convolutions:
            values = F.relu(conv(embeddings))
            valid = (lengths - conv.kernel_size[0] + 1).clamp_min(1)
            mask = torch.arange(values.shape[-1], device=values.device)[None, :] < valid[:, None]
            values = values.masked_fill(~mask[:, None, :], float("-inf"))
            pooled.append(values.amax(dim=-1))
        representation = self.dropout(torch.cat(pooled, dim=-1))
        return self.style_head(representation), self.tone_head(representation)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=96, dropout=0.4, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(2 * hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.style_head = nn.Linear(4 * hidden_size, 5)
        self.tone_head = nn.Linear(4 * hidden_size, 5)

    def forward(self, input_ids, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            self.embedding(input_ids),
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, (hidden, _) = self.lstm(packed)
        values, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        mask = torch.arange(values.shape[1], device=values.device)[None, :] < lengths[:, None]
        attention = self.attention(values).squeeze(-1).masked_fill(~mask, float("-inf")).softmax(dim=-1)
        context = (values * attention.unsqueeze(-1)).sum(dim=1)
        representation = self.dropout(torch.cat([context, hidden[-2], hidden[-1]], dim=-1))
        return self.style_head(representation), self.tone_head(representation)


class TransformerClassifier(nn.Module):
    def __init__(self, encoder_config, dropout=0.3, pretrained=None):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        cfg = dict(encoder_config)
        model_type = cfg.pop("model_type")
        config = AutoConfig.for_model(model_type, **cfg)
        self.encoder = (
            AutoModel.from_pretrained(pretrained, config=config, trust_remote_code=False)
            if pretrained
            else AutoModel.from_config(config, trust_remote_code=False)
        )
        self.dropout = nn.Dropout(dropout)
        self.style_head = nn.Linear(config.hidden_size, 5)
        self.tone_head = nn.Linear(config.hidden_size, 5)

    def forward(self, input_ids, lengths):
        mask = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :] < lengths[:, None]
        outputs = self.encoder(input_ids=input_ids, attention_mask=mask.long())
        representation = self.dropout(outputs.last_hidden_state[:, 0])
        return self.style_head(representation), self.tone_head(representation)


def build_model(architecture: str, config: dict, pretrained=None) -> nn.Module:
    if architecture == "cnn":
        return TextCNN(**config)
    if architecture == "bilstm":
        return BiLSTM(**config)
    if architecture == "transformer":
        return TransformerClassifier(**config, pretrained=pretrained)
    raise ValueError(f"Unknown neural architecture: {architecture}")
