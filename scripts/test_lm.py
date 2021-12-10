import torch
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartTokenizer

if __name__ == '__main__':
    bart = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    sentence = ["I really like dogs. You can see they love you back.", "Hi! My name is cute."]

    tokens = bart_tokenizer(sentence, return_tensors='pt', padding=True)['input_ids']
    encoding = bart.model.encoder(tokens)

    embedding = encoding[0][:, 0]

    text_dec = torch.full_like(tokens, 2)
    text_dec[:, 0] = 0
    text_latent = embedding.unsqueeze(1)
    for k in range(1, tokens.size(1)):
        decod_out = bart.model.decoder(input_ids=text_dec,
                                       encoder_hidden_states=text_latent)[0]
        logits = F.linear(decod_out, bart.model.shared.weight, bart.final_logits_bias)
        decoded_token = torch.argmax(logits[:, k], dim=1)
        text_dec[:, k] = decoded_token

    decoded_text = [bart_tokenizer.decode(sentence) for sentence in text_dec]

    print('ok')
