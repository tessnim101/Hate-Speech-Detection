import random
import string
import re
import pandas as pd

def character_perturbation(text, noise_level=0.1):
    """
    Randomly insert, delete, or swap characters in a text.
    noise_level: fraction of characters to perturb
    """
    chars = list(text)
    if len(chars) == 0:
        return text  # avoid errors on empty input

    n_perturb = max(1, int(len(chars) * noise_level))
    
    for _ in range(n_perturb):
        if len(chars) == 0:
            break
        op = random.choice(["insert", "delete", "swap"])
        idx = random.randint(0, len(chars) - 1)
        
        if op == "insert":
            chars.insert(idx, random.choice(string.ascii_lowercase))
        elif op == "delete" and len(chars) > 1:
            chars.pop(idx)
        elif op == "swap" and len(chars) > 1:
            idx2 = random.randint(0, len(chars) - 1)
            chars[idx], chars[idx2] = chars[idx2], chars[idx]
    
    return "".join(chars)


# Load top attended tokens for stereotype class from your analysis
top_stereo_tokens_df = pd.read_csv("figures/cross_attn_top_tokens.csv")
top_stereo_tokens = top_stereo_tokens_df[
    top_stereo_tokens_df["class"] == 1
]["token"].tolist()

# Build regex pattern from these tokens
pattern = re.compile(
    r'\b(' + '|'.join(re.escape(t.replace("▁", "")) for t in top_stereo_tokens) + r')\b',
    re.IGNORECASE
)

def targeted_perturbation(text, pattern, noise_level=0.1):
    """Perturb only tokens that match the high-attention pattern."""
    def perturb_match(match):
        word = match.group()
        if random.random() < noise_level:
            return character_perturbation(word, noise_level=0.3)
        return word
    return pattern.sub(perturb_match, text)


if __name__ == "__main__":
    example = 
    print(targeted_perturbation(example, pattern, noise_level=0.5))