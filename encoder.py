import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_missing_words(text, n, next_word, model_name='distilgpt2', top_m=10):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    best_sequence = None
    best_next_word_prob= 0.0

    def dfs(current_sequence, starting_i):
        nonlocal best_sequence, best_next_word_prob

        # If we have generated n words, check the next word
        current_length = len([w for w in ''.join(current_sequence).split(' ') if any(c.isalpha() for c in w)])
        if current_length == n:
            next_word_tokens = tokenizer.encode(next_word)
            prob_product = 1.0
            for i in range(len(next_word_tokens)):
                next_input_ids = tokenizer.encode(text + ''.join(current_sequence) + ' ' + tokenizer.decode(next_word_tokens[:i]), return_tensors='pt')
                with torch.no_grad():
                    next_outputs = model(next_input_ids)
                    next_logits = next_outputs.logits[:, -1, :]
                    prob_product *= torch.softmax(next_logits, dim=-1)[0, next_word_tokens[i]].item()

            # Update the best sequence if this one is better
            if prob_product > best_next_word_prob:
                best_next_word_prob = prob_product
                best_sequence = current_sequence[:]

            return

        # Continue DFS for each of the top m tokens
        for i in range(starting_i, top_m):
            input_ids = tokenizer.encode(text + ''.join(current_sequence), return_tensors='pt')
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits[:, -1, :]  # Get the logits for the last token
                top_logits, top_idxs = torch.topk(logits, i + 1)
            token = tokenizer.decode(top_idxs[0][-1])
            current_sequence.append(token)
            dfs(current_sequence, starting_i=i+1)
            current_sequence.pop()  # Backtrack

    # Start DFS with an empty sequence
    dfs([],  0)

    return ' '.join(best_sequence) if best_sequence else None


# Example usage
# text = "It was the best of times,"
text = "Dear Jane, It was a pleasure to read your last letter and I'm sorry that I didn't write back sooner. Thank"
n = 5
# next_word = "worst of times"
next_word = "you"
result = generate_missing_words(text, n, next_word)
if result:
    print(text + result + ' ' + next_word)
