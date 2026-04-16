from collections import Counter, defaultdict
from typing import Iterator, Iterable
from pathlib import Path
from tqdm import tqdm

import regex as re
import pickle
import heapq

type bytepair = tuple[bytes, bytes]
type intpair = tuple[int, int]
type intlist = tuple[int, ...]

# The pre-tokenizer expression is an OR of the following patterns:
# 1. "'(?:[sdmt]|ll|ve|re)" – common English contractions, (don't, it's, we'll, etc.)
# 2. " ?\p{L}+" – word-like letter runs, optionally preceded by a space so we can split on spaces
# 3. " ?\p{N}+" – digit runs, optionally preceded by a space so we can split on spaces
# 4. " ?[^\s\p{L}\p{N}]+" – non-whitespace, non-word, non-digit characters, optionally preceded by a space (punctuation, symbols, underscores, etc.)
# 5. "\s+(?!\S)" – trailing whitespace
# 6. "\s+" – non-trailing whitespace (e.g. between words)
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# The default maximum vocabulary size
DEFAULT_VOCAB_SIZE = 10000

# The list of special tokens, which are not to be split into sub-tokens
EOS_TOKEN = "<|endoftext|>"

DEFAULT_SPECIAL_TOKENS = [
    EOS_TOKEN,
]

def iter_pairs(word: intlist) -> Iterator[intpair]:
    """
    Iterate over all pairs of bytes in the word.

    Args:
        word (tuple[int, ...]): The word to iterate over.

    Returns:
        An iterator over the pairs of bytes in the word.
    """
    for i in range(len(word) - 1):
        yield (word[i], word[i + 1])

def merge_word(word: intlist, pair: intpair, new_id: int) -> intlist:
    """
    Merge a word at the given pair.

    Args:
        word (intlist): The word to merge.
        pair (intpair): The pair to merge.
        new_id (int): The new ID of the merged pair.

    Returns:
        The merged word.
    """
    left, right = pair
    merged = []
    i = 0

    while i < len(word):
        if i + 1 < len(word) and word[i] == left and word[i + 1] == right:
            merged.append(new_id)
            i += 2
        else:
            merged.append(word[i])
            i += 1

    return tuple(merged)

def add_word(
    word: intlist,
    freq: int,
    pair_counts: Counter[intpair],
    pair_to_words: dict[intpair, set[intlist]],
    vocab: dict[int, bytes],
    heap: list[tuple[int, intpair]],
):
    """
    Add a word to the pair counts and heap.

    Args:
        word (intlist): The word to add.
        freq (int): The frequency of the word.
        pair_counts (Counter[intpair]): The counter of pair counts.
        pair_to_words (dict[intpair, set[intlist]]): The dictionary of pair to words.
        vocab (dict[int, bytes]): The vocabulary.
        heap (list[tuple[int, intpair]]): The heap of pairs.
    """
    for pair in iter_pairs(word):
        pair_counts[pair] += freq
        pair_to_words[pair].add(word)

        bytes_pair = (vocab[pair[0]], vocab[pair[1]])
        heapq.heappush_max(heap, (pair_counts[pair], bytes_pair, pair))

def remove_word(
    word: intlist,
    word_freq: int,
    pair_counts: Counter[intpair],
    pair_to_words: dict[intpair, set[intlist]],
    vocab: list[bytes],
    heap: list[tuple[int, tuple[bytes, bytes], intpair]],
):
    """
    Remove a word from the pair counts and pair to words.

    Args:
        word (intlist): The word to remove.
        freq (int): The frequency of the word.
        pair_counts (Counter[intpair]): The counter of pair counts.
        pair_to_words (dict[intpair, set[intlist]]): The dictionary of pair to words.
        vocab (list[bytes]): Current id -> bytes mapping (same as train_bpe).
        heap: Lazy max-heap of (count, bytes_pair, pair); must push after each decrement
            so a heap entry exists for the current count (add_word already pushes on add).
    """
    for pair in iter_pairs(word):
        pair_counts[pair] -= word_freq

        assert pair_counts[pair] >= 0

        if pair_counts[pair] == 0:
            pair_counts.pop(pair)
        else:
            bytes_pair = (vocab[pair[0]], vocab[pair[1]])
            heapq.heappush_max(heap, (pair_counts[pair], bytes_pair, pair))

        pair_to_words[pair].discard(word)

        if not pair_to_words[pair]:
            pair_to_words.pop(pair)

def pop_best_pair(
    heap: list[tuple[int, intpair]],
    pair_counts: Counter[intpair],
) -> intpair | None:
    """
    Pop the best pair from the heap.

    Args:
        heap (list[tuple[int, intpair]]): The heap of pairs.
        pair_counts (Counter[intpair]): The counter of pair counts.
    """
    while heap:
        count, bytes_pair, pair = heapq.heappop_max(heap)
        # Heap is lazy: old entries may remain until skipped; add_word and remove_word
        # both push the current count so a matching entry exists for each live pair.
        if count > 0 and pair_counts.get(pair, 0) == count:
            return pair
    return None

def _special_tokens_regex_pattern(special_tokens: list[str]) -> str:
    """Alternation of escaped specials, longest first so overlaps match the longest token."""
    ordered = sorted(special_tokens, key=len, reverse=True)
    return "|".join(map(re.escape, ordered))

def iter_text_and_special_segments(
    text: str, special_tokens: list[str]
) -> Iterator[tuple[bool, str]]:
    """
    Walk `text` left-to-right, yielding (is_special, segment) pairs.

    Plain text runs use is_special=False; exact matches of a special token use is_special=True.
    This is the shared primitive for training (split on specials) and inference (emit one id each).
    """
    if not special_tokens:
        yield (False, text)
        return

    pattern = _special_tokens_regex_pattern(special_tokens)
    # Capturing group keeps delimiters in the result, interleaved with text runs.
    parts = re.split(f"({pattern})", text)
    for i, part in enumerate(parts):
        if part == "":
            continue
        if i % 2 == 0:
            yield (False, part)
        else:
            yield (True, part)

def split_documents(text: str, special_tokens: list[str]) -> Iterator[str]:
    """
    Split the text into documents, which are separated by special tokens.
    This prevents merging across the document boundary.

    Args:
        text (str): The text to split.
        special_tokens (list[str]): The special tokens to split on.

    Returns:
        An iterator over the documents.
    """
    for is_special, segment in iter_text_and_special_segments(text, special_tokens):
        if not is_special:
            yield segment

def collect_pretoken_counts(documents: Iterator[str]) -> Counter[intlist]:
    """
    Collect the counts of each pretoken in the documents.

    Args:
        documents (list[str]): The documents to collect the pretoken counts from.

    Returns:
        A counter of the pretoken counts.
    """
    pretoken_counts = Counter()

    for document in tqdm(documents, desc="Pretokenizing", unit=" documents"):
        for match in re.finditer(PAT, document):
            pretoken = tuple(match.group().encode("utf-8"))
            pretoken_counts[pretoken] += 1

    return pretoken_counts

def train_bpe(
    input_path: Path | str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    special_tokens: list[str] = DEFAULT_SPECIAL_TOKENS,
) -> tuple[dict[int, bytes], list[bytepair]]:
    """
    Train a BPE tokenizer on the text corpus at the given filepath.

    Args:
        input_path (Path | str): Path to a text file with BPE tokenizer training data.
        vocab_size (int): A positive integer that defines the maximum final vocabulary size (in-
            cluding the initial byte vocabulary, vocabulary items produced from merging, and any
            special tokens).
        special_tokens (list[str]): A list of strings to add to the vocabulary. These special
            tokens do not otherwise affect BPE training.

    Returns:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the
            vocabulary) to bytes (token bytes).
        merges (list[tuple[bytes, bytes]]): A list of BPE merges produced from training.
            Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1>
            was merged with <token2>. The merges should be ordered by order of creation.
    """
    if vocab_size < len(special_tokens) + 256:
        raise ValueError(
            f"Vocab size must be greater than the number of special tokens ({len(special_tokens)})" + \
            f"plus the number of single-byte tokens (256)")

    # Read the text corpus from the given filepath
    with open(input_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Initialize the vocabulary with the single-byte tokens
    vocab = {i: bytes([i]) for i in range(256)}

    # Initialize the list of merges
    merges: list[bytepair] = []

    documents = split_documents(text, special_tokens)

    word_counts: Counter[intlist] = collect_pretoken_counts(documents)
    pair_counts: Counter[intpair] = Counter()

    pair_to_words: dict[intpair, set[intlist]] = defaultdict(set)
    heap: list[tuple[int, intpair]] = [] # (count, pair)

    for word, freq in word_counts.items():
        #add_word(word, freq, pair_counts, pair_to_words, vocab, heap)
        for pair in iter_pairs(word):
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)

    for pair, count in pair_counts.items():
        bytes_pair = (vocab[pair[0]], vocab[pair[1]])
        heapq.heappush_max(heap, (count, bytes_pair, pair))

    # Merge byte pairs until the vocabulary size is reached
    pbar = tqdm(total=vocab_size, desc="Merging byte pairs", unit=" merges")
    pbar.update(len(vocab) + len(special_tokens))

    while len(vocab) + len(special_tokens) < vocab_size:
        best_pair = pop_best_pair(heap, pair_counts)

        if best_pair is None:
            break

        left_id, right_id = best_pair
        new_id = len(vocab)
        vocab[new_id] = vocab[left_id] + vocab[right_id]
        merges.append((vocab[left_id], vocab[right_id]))

        affected_words = pair_to_words[best_pair]
        for word in list(affected_words):
            freq = word_counts.pop(word)
            assert freq > 0
            #if freq == 0:
            #    continue

            remove_word(word, freq, pair_counts, pair_to_words, vocab, heap)

            new_word = merge_word(word, best_pair, new_id)

            assert new_word not in word_counts

            word_counts[new_word] = freq

            add_word(new_word, freq, pair_counts, pair_to_words, vocab, heap)

        pbar.update()

    pbar.close()

    # Add the special tokens to the vocabulary
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    return vocab, merges

def merge_word_bytes(
    word: list[bytes],
    pair: tuple[bytes, bytes],
) -> list[bytes]:
    left, right = pair
    merged_symbol = left + right
    out = []
    i = 0

    while i < len(word):
        if i + 1 < len(word) and word[i] == left and word[i + 1] == right:
            out.append(merged_symbol)
            i += 2
        else:
            out.append(word[i])
            i += 1

    return out

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[bytepair],
        special_tokens: list[str] = DEFAULT_SPECIAL_TOKENS,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        
        for token_id, token_bytes in self.vocab.items():
            if token_bytes == EOS_TOKEN.encode("utf-8"):
                self.eos_token_id = token_id
                break
        else:
            raise ValueError(f"EOS token not found in vocabulary")

    @classmethod
    def from_files(
        cls,
        vocab_filepath: Path | str,
        merges_filepath: Path | str,
        special_tokens: list[str] | None = DEFAULT_SPECIAL_TOKENS,
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        bytes_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        specials = self.special_tokens or []

        encoded: list[int] = []

        for is_special, segment in iter_text_and_special_segments(text, specials):
            if is_special:
                encoded.append(bytes_to_id[segment.encode("utf-8")])
                continue

            for match in re.finditer(PAT, segment):
                word = [bytes([b]) for b in match.group().encode("utf-8")]

                while True:
                    best_pair = None
                    best_rank = None

                    # Among all pairs, find the pair with the highest precedence (lowest merge rank)
                    for i in range(len(word) - 1):
                        pair = (word[i], word[i + 1])
                        rank = merge_ranks.get(pair)
                        if rank is not None and (best_rank is None or rank < best_rank):
                            best_pair = pair
                            best_rank = rank

                    if best_pair is None:
                        break

                    word = merge_word_bytes(word, best_pair)

                encoded.extend(bytes_to_id[symbol] for symbol in word)

        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[tok] for tok in ids).decode("utf-8", errors="replace")