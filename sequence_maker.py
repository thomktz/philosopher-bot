# %%
import torch
import torch.nn as nn
import numpy as np

in_text = open("bot_philo_lower.txt", encoding = "UTF-8").read()

max_len = 200

as_list = in_text.split("\n")
text_lengths = ([len(quote) for quote in as_list])

quotes = []
text = ""
for quote in as_list:
    if len(quote) < max_len:
        quotes.append(quote + " " * (max_len - len(quote)))
        text += quote + "\n"


chars = sorted(list(set(text)))
chars_to_int = dict((c,i) for i,c in enumerate(chars))
n_chars = len(text)
n_vocab = len(chars)
print("Total characters :", n_chars)
print("Total vocab :", n_vocab)


def build_sequences(txt, window):
    x = list()
    y = list()
    for quote in txt:
        for i in range(len(quote)):
            try:
                # Get window of chars from text
                # Then, transform it into its idx representation
                sequence = quote[i:i+window]
                sequence = [chars_to_int[char] for char in sequence]
                
                # Get word target
                # Then, transfrom it into its idx representation
                target = quote[i+window]
                target = chars_to_int[target]
                
                # Save sequences and targets
                x.append(sequence)
                y.append(target)
                
            except:
                pass
        
    x = np.array(x)
    y = np.array(y)
    
    return x, y
window = 15
X,Y = build_sequences(quotes, window)

# %%
