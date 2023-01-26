import random


class MixModifier:
    def __init__(self, p):
        self.p = p

    def __call__(self, sentence):
        if random.random() <= self.p:
            sentence = sentence[8:-1]
            tokens = sentence.split("{link}")
            random.shuffle(tokens)
            sentence = "{link}".join(tokens)
            sentence = "{start} " + sentence + "."
        return sentence


class DeleteModifier:
    def __init__(self, p, max_tries=1):
        self.p = p
        self.max_tries = max_tries

    def __call__(self, sentence):
        sentence = sentence[8:-1]
        tokens = sentence.split("{link}")
        for k in range(self.max_tries):
            if random.random() <= self.p:
                remove = random.randint(0, len(tokens) - 1)
                tokens.pop(remove)
            else:
                break
        sentence = "{link}".join(tokens)
        sentence = "{start} " + sentence + "."
        return sentence
