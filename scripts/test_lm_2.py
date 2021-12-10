import torch
import torch.nn.functional as F
from transformers import BartTokenizer, BartForConditionalGeneration

bart = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")


def make_causal_mask_prog(input_dec, encod_out):
    sz = len(encod_out[0])
    list = [1] * sz
    input_deco = torch.Tensor([list] * len(input_dec))
    tensor = fill_with_neg_inf(input_deco)
    for i in range(len(tensor)):
        for j in range(i + 1):
            tensor[i][j] = 1.
    return tensor


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(0.).type_as(t)


transcript = ["My friends are nice but they eat too many carbs", "I like cats more than dogs"]

transcript_tokens = bart_tokenizer(transcript, return_tensors='pt', padding=True)['input_ids']
print(transcript_tokens)

text_latent = bart.model.encoder(transcript_tokens)[0]

# text_dec = torch.LongTensor([[2] * (len(text_latent[0]))] * len(text_latent))
text_dec = torch.full_like(text_latent, 2).long()
output = torch.LongTensor([[0] * (len(text_latent[0]) + 1)] * len(text_latent))

for j in range(len(text_latent)):
    text_dec[j][1] = 0

for i in range(len(text_latent[0]) - 1):

    decod_out = bart.model.decoder(input_ids=text_dec,
                                   encoder_hidden_states=text_latent,
                                   attention_mask=make_causal_mask_prog(text_dec[0], text_latent))[0]

    logits = F.linear(decod_out, bart.model.shared.weight, bias=bart.final_logits_bias)
    for j in range(len(text_latent)):
        output[j][i + 1] = torch.argmax(logits[j][i + 1]).item()

        if (i + 2) < len(text_dec[0]):
            text_dec[j][i + 2] = torch.argmax(logits[j][i + 1]).item()
            output[j][-1] = 2

transcript_recons = output
text_recons = transcript_recons
print(text_recons)
for token in text_recons[-1]:
    print(bart_tokenizer.decode([token]))
print(text_recons)
print(transcript_tokens)
print('ok')
