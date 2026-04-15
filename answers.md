# 2.4 BPE Tokenizer Training

Naively merging bytes next to each other could result in the creation of a vocabulary like {"g!", "g.", "do"}. However, a better vocabulary would be {"dog", "!", "."}, so that semantic meaning doesn't get annhilated by how BPE arbitrarily separates bytestrings into separate tokens.

# 2.7 Experiments
## (a)
The compression ratio is 4.04 bytes per token, taken as an average over 10 documents sampled uniformly at andom from TinyStoriesV2-GPT4-train.
## (b)
The token IDs are non-negative integers, so by using an unsigned integer we get the full range enabled by 16 bits. We use uint16,
instead of uint8, uint32, etc. because uint16 is the smallest unsigned integer size that can still represent the integers including the length of our vocabulary.

# 3.2.4 Insight (softmax): Handling numerical instability
You can subtract the max of v from each v_i. This is because
of the invariance that softmax(v)_i = softmax(v + c)_i for any
constant c, and now the input to the exp function is always
non-positive, so you never get exponential explosion.

# 3.2.4 Insight: Masking
We can set the pre-softmax values corresponding to masked entries to
a very large (in magnitude) negative number (like -inf), thereby making
the exponential in the softmax turn them into roughly zero probability
entries.

# 4.1 Insight: Perplexity
Although perplexity is just a monotonic transformation of cross entropy, found by taking the exponentiation, perplexity is much easier to interpret. For instance, a perplexity score of N means the model is behaving as if it is choosing among about N equally likely next tokens on average, whereas cross entropy is measured in either nats or bits.