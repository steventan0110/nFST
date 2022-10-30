from transformers import GPT2LMHeadModel, GPT2Config
import torch
from torch.functional import F

config = GPT2Config(
    4,
    n_head=4,
    n_layer=4,
    bos_token_id=0,
    eos_token_id=1,
    n_positions=4,
    n_embd=1024,
)
model = GPT2LMHeadModel(config)
vocab = [0, 1, 2, 3]
s = []
# create all possible strings of length 4
for i in vocab:
    for j in vocab:
        for h in vocab:
            for m in vocab:
                if set([i, j, h, m]) == set(vocab):
                    s.append([i, j, h, m])

s = torch.Tensor(s).long()
logits = model(s).logits
logits = F.log_softmax(logits, dim=2)
log_prob = torch.gather(logits, 2, s.unsqueeze(2)).squeeze(2)
log_prob = torch.sum(log_prob, dim=1)
temp = torch.exp(log_prob)
out = torch.sum(temp)
print(log_prob)
print(temp)
print(out)
