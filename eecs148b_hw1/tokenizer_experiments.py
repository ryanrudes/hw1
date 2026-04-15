from eecs148b_hw1.tokenizer import Tokenizer, split_documents, DEFAULT_SPECIAL_TOKENS

import numpy as np

TRAIN_DATA_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VALID_DATA_PATH = "data/TinyStoriesV2-GPT4-valid.txt"

VOCAB_PATH = "tokenizer/vocab.pkl"
MERGES_PATH = "tokenizer/merges.pkl"

tokenizer = Tokenizer.from_files(
    vocab_filepath=VOCAB_PATH,
    merges_filepath=MERGES_PATH,
)

with open(TRAIN_DATA_PATH, "r") as file:
    train_data = file.read()

with open(VALID_DATA_PATH, "r") as file:
    valid_data = file.read()

train_documents = list(split_documents(train_data, DEFAULT_SPECIAL_TOKENS))

mean_compression_ratio = 0
num_samples = 0

for document in np.random.choice(train_documents, 10):
    encoded = tokenizer.encode(document)
    decoded = tokenizer.decode(encoded)
    assert decoded == document

    num_tokens = len(encoded)
    num_bytes = len(document.encode("utf-8"))

    compression_ratio = num_bytes / num_tokens
    num_samples += 1
    mean_compression_ratio += (compression_ratio - mean_compression_ratio) / num_samples

print(f"Mean compression ratio: {mean_compression_ratio}")

encoded_train = tokenizer.encode(train_data)
decoded_train = tokenizer.decode(encoded_train)
assert decoded_train == train_data

encoded_valid = tokenizer.encode(valid_data)
decoded_valid = tokenizer.decode(encoded_valid)
assert decoded_valid == valid_data

# Serialize the encodings as a uint16 NumPy array
train_encodings = np.array(encoded_train, dtype=np.uint16)
valid_encodings = np.array(encoded_valid, dtype=np.uint16)

np.save("tokenizer/train_encodings.npy", train_encodings)
np.save("tokenizer/valid_encodings.npy", valid_encodings)