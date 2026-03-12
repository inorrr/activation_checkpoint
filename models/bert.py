import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification

class BertSequenceClassifierWrapper(nn.Module):
    """
    Wrap Hugging Face Bert so it works with the existing profiler API.
    The wrapped model takes a singal dict as input and reterns logits.
    """
    def __init__(self, model: BertForSequenceClassification):
        super().__init__()
        self.model = model
    
    def forward(self, batch: dict) -> torch.Tensor:
        outputs = self.model(**batch)
        return outputs.logits
    

def load_bert_classifier(
    num_labels: int = 2,
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
) -> nn.Module:
    """
    Load a BERT-base sequence classification model.

    We use a randomly initialized config instead of downloading pretrained weights,
    since Phase 1 only needs profiling, not model quality.
    """
    config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        num_labels=num_labels,
    )

    bert_model = BertForSequenceClassification(config)
    wrapped_model = BertSequenceClassifierWrapper(bert_model)
    return wrapped_model
