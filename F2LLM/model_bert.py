
import torch
from transformers import AutoModel, AutoTokenizer

class BertEmbedder:
    def __init__(self,
                 model_path,
                 max_seq_length=512,
                 args=None,
                 pool_strategy="cls"
                 ):
        self.args = args
        self.dtype = torch.bfloat16
        self.device = None
        self.encoder = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype
        )
        if hasattr(self.encoder.config, "use_cache"):
            self.encoder.config.use_cache = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.sep_token is not None:
                self.tokenizer.pad_token = self.tokenizer.sep_token
            elif self.tokenizer.cls_token is not None:
                self.tokenizer.pad_token = self.tokenizer.cls_token

        self.max_seq_length = max_seq_length
        self.pool_strategy = pool_strategy

    def set_device(self):
        self.device = self.encoder.device

    def _pool(self, last_hidden_state, attention_mask):
        # last\_hidden\_state: [bs, seq, d], attention\_mask: [bs, seq]
        if self.pool_strategy == "mean":
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [bs, seq, 1]
            summed = (last_hidden_state * mask).sum(dim=1)                   # [bs, d]
            denom = mask.sum(dim=1).clamp_min(1e-6)                          # [bs, 1]
            return summed / denom
        return last_hidden_state[:, 0, :]                                    # [CLS]

    def forward(self, batch):
        bs = batch['bs']
        num_hard_neg = int((len(batch['input_ids']) - 2 * bs) / bs)

        outputs = self.encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        last_hidden_state = outputs.last_hidden_state
        attn = batch['attention_mask']

        query_emb = self._pool(last_hidden_state[:bs], attn[:bs]).unsqueeze(1)            # [bs, 1, d]
        passage_emb = self._pool(last_hidden_state[bs:2*bs], attn[bs:2*bs]).unsqueeze(1)  # [bs, 1, d]

        if num_hard_neg == 0:
            neg_emb = None
        else:
            neg_all = self._pool(last_hidden_state[2*bs:], attn[2*bs:])                   # [bs*num\_hard\_neg, d]
            neg_emb = neg_all.view(bs, num_hard_neg, -1)                                   # [bs, num\_hard\_neg, d]

        return {
            'query_passage_features': query_emb,
            'passage_passage_features': passage_emb,
            'negative_passage_features': neg_emb
        }