from __future__ import annotations

import os
import regex as re
from collections import defaultdict, Counter
from typing import Any, Iterable, BinaryIO, Iterator, IO
from multiprocessing import Pool, cpu_count
import functools

# Import find_chunk_boundaries from pretokenization_utils
from .pretokenization_utils import find_chunk_boundaries

# Pre-tokenization pattern (GPT-2 style)
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class DoublyLinkedNode:
    """Node for doubly linked list representation of pre-token bytes."""
    
    def __init__(self, data: bytes):
        self.data = data
        self.prev = None
        self.next = None


class DoublyLinkedList:
    """Doubly linked list for efficient merging during BPE training."""
    
    def __init__(self, tokens: list[bytes]):
        """Initialize from a list of byte tokens."""
        self.head = None
        self.tail = None
        self.length = len(tokens)
        
        if not tokens:
            return
        
        # Create first node
        self.head = DoublyLinkedNode(tokens[0])
        current = self.head
        
        # Create remaining nodes
        for token in tokens[1:]:
            new_node = DoublyLinkedNode(token)
            new_node.prev = current
            current.next = new_node
            current = new_node
        
        self.tail = current
    
    def get_pairs(self) -> list[tuple[DoublyLinkedNode, bytes, bytes]]:
        """Get all adjacent pairs with their starting nodes."""
        pairs = []
        current = self.head
        while current and current.next:
            pairs.append((current, current.data, current.next.data))
            current = current.next
        return pairs
    
    def merge_at_node(self, node: DoublyLinkedNode, new_token: bytes) -> DoublyLinkedNode:
        """
        Merge node with its next node, return the new merged node.
        Assumes node.next exists.
        """
        if not node.next:
            return node
        
        next_node = node.next
        
        # Create new merged node
        merged_node = DoublyLinkedNode(new_token)
        merged_node.prev = node.prev
        merged_node.next = next_node.next
        
        # Update previous node
        if node.prev:
            node.prev.next = merged_node
        else:
            self.head = merged_node
        
        # Update next node
        if next_node.next:
            next_node.next.prev = merged_node
        else:
            self.tail = merged_node
        
        self.length -= 1
        return merged_node


def process_chunk_for_pretokens(args):
    """
    Process a single chunk to extract pre-tokens.
    This function is used in multiprocessing.
    """
    file_path, start, end, special_tokens = args
    
    # Compile patterns
    pat = re.compile(PAT)
    

    # Create special token pattern if needed
    if special_tokens:
        sorted_special = sorted(special_tokens, key=len, reverse=True)
        escaped_tokens = [re.escape(token) for token in sorted_special]
        combined_pattern = re.compile('|'.join(escaped_tokens + [PAT]))
    else:
        combined_pattern = pat
    
    # Read chunk
    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk_text = chunk_bytes.decode('utf-8', errors='ignore')

    # Count pre-tokens in this chunk
    chunk_counts = Counter()
    
    
    chunks = re.split("|".join([re.escape(special_token) for special_token in special_tokens]), chunk_text)
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            pre_token = match.group()
            if pre_token and pre_token not in special_tokens:
                chunk_counts[pre_token] += 1

            

    
    return chunk_counts


class BPETrainer:
    """Efficient BPE trainer using multiprocessing and doubly linked lists."""
    
    def __init__(self, special_tokens: list[str] = None):
        """
        Initialize BPE trainer.
        
        Args:
            special_tokens: List of special tokens to exclude from merging
        """
        self.special_tokens = special_tokens or []
        
        # Initialize vocabulary with all 256 bytes
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.next_id = 256
        
        # Add special tokens to vocabulary
        for special_token in self.special_tokens:
            special_bytes = special_token.encode('utf-8')
            if special_bytes not in self.vocab.values():
                self.vocab[self.next_id] = special_bytes
                self.next_id += 1
        
        self.merges = []
        
        # Pre-token representations as doubly linked lists
        self.pretoken_lists = {}  # pre_token -> DoublyLinkedList
        self.pretoken_counts = {}  # pre_token -> frequency count
        
        # Pair tracking for efficiency
        self.pair_counts = defaultdict(int)
        self.pair_to_pretokens = defaultdict(set)  # (left, right) -> set of pre_tokens containing this pair
    
    def train(self, input_path: str | os.PathLike, vocab_size: int, num_processes: int = None, chunk_size_mb: int = 50) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Train BPE tokenizer.
        
        Args:
            input_path: Path to training corpus
            vocab_size: Target vocabulary size
            num_processes: Number of processes for multiprocessing (default: auto-calculated based on file size)
            chunk_size_mb: Target chunk size per process in MB for memory management
            
        Returns:
            Tuple of (vocab dict, merge list)
        """
        if num_processes is None:
            # Dynamic num_processes based on file size for consistent memory usage
            file_size = os.path.getsize(input_path)
            max_processes_by_memory = file_size // (chunk_size_mb * 1024 * 1024)
            num_processes = max(1, max_processes_by_memory)
            print(f"Auto-calculated num_processes: {num_processes} (file size: {file_size / (1024*1024):.1f}MB, target chunk size: {chunk_size_mb}MB)")


        # Step 1: Get chunk boundaries
        print("Step 1: Getting chunk boundaries...")
        with open(input_path, 'rb') as f:
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        
        # Step 2: Pre-tokenize chunks with multiprocessing
        print("Step 2: Pre-tokenizing chunks...")
        chunk_args = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            chunk_args.append((input_path, start, end, self.special_tokens))
        
        # Process chunks in parallel
        num_workers = min(20, cpu_count())
        with Pool(processes=num_workers) as pool:
            chunk_results = pool.map(process_chunk_for_pretokens, chunk_args)
        
        # Merge results from all chunks
        print("Merging pre-token counts...")
        total_counts = Counter()
        for chunk_counts in chunk_results:
            total_counts.update(chunk_counts)
        
        self.pretoken_counts = total_counts
        
        # Initialize doubly linked lists for each pre-token
        print("Initializing doubly linked lists...")
        for pre_token, count in self.pretoken_counts.items():
            pre_token_bytes = pre_token.encode('utf-8')
            tokens = [bytes([b]) for b in pre_token_bytes]
            self.pretoken_lists[pre_token] = DoublyLinkedList(tokens)
        
        # Step 3: Initial pair counting
        print("Initial pair counting...")
        self._count_all_pairs()
        
        # Step 3: Iterative merging
        print("Starting iterative merging...")
        current_vocab_size = len(self.vocab)
        iteration = 0
        
        while current_vocab_size < vocab_size:
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, vocab size: {current_vocab_size}")
            
            # Find most frequent pair with deterministic tie-breaking
            if not self.pair_counts:
                break
            
            # Use deterministic tie-breaking: (frequency, lexicographically largest pair)
            best_item = max(self.pair_counts.items(), key=lambda x: (x[1], x[0]))
            best_pair = best_item[0]
            best_count = best_item[1]
            
            if best_count == 0:
                break
            
            # Add merge
            self.merges.append(best_pair)

            # Create new token
            new_token = best_pair[0] + best_pair[1]
            self.vocab[self.next_id] = new_token
            self.next_id += 1
            current_vocab_size += 1
            
            # Apply merge and update statistics
            self._apply_merge(best_pair, new_token)
            
            iteration += 1
        print(f"Training completed. Final vocab size: {current_vocab_size}")
        return self.vocab, self.merges
    
    def _count_all_pairs(self):
        """Count all pairs across all pre-tokens."""
        self.pair_counts.clear()
        self.pair_to_pretokens.clear()
        
        for pre_token, dll in self.pretoken_lists.items():
            count = self.pretoken_counts[pre_token]
            pairs = dll.get_pairs()
            
            for node, left, right in pairs:
                pair = (left, right)
                self.pair_counts[pair] += count
                self.pair_to_pretokens[pair].add(pre_token)
    
    def _apply_merge(self, best_pair: tuple[bytes, bytes], new_token: bytes):
        """Apply merge and update all statistics."""
        left, right = best_pair
        
        # Get all pre-tokens that contain this pair
        affected_pretokens = list(self.pair_to_pretokens[best_pair])
        
        # Remove the merged pair from global statistics
        del self.pair_counts[best_pair]
        del self.pair_to_pretokens[best_pair]
        
        # Update each affected pre-token
        for pre_token in affected_pretokens:
            dll = self.pretoken_lists[pre_token]
            count = self.pretoken_counts[pre_token]
            
            # Step 1: Remove ALL pairs from this pre-token from global counts
            self._remove_all_pairs_for_pretoken(pre_token, dll, count)
            
            # Step 2: Apply merges left-to-right, skipping nodes as we merge them
            current_node = dll.head
            while current_node and current_node.next:
                if current_node.data == left and current_node.next.data == right:
                    # Merge this pair
                    merged_node = dll.merge_at_node(current_node, new_token)
                    # Continue from the merged node (skip checking the consumed second node)
                    current_node = merged_node.next
                else:
                    # Move to next node
                    current_node = current_node.next
            
            # Step 3: Add back ALL pairs from the updated linked list
            self._add_all_pairs_for_pretoken(pre_token, dll, count)
    
    def _remove_all_pairs_for_pretoken(self, pre_token: str, dll: DoublyLinkedList, count: int):
        """Remove all pairs from a pre-token from global statistics."""
        pairs = dll.get_pairs()
        for node, left, right in pairs:
            pair = (left, right)
            if pair in self.pair_counts:
                self.pair_counts[pair] -= count
                if self.pair_counts[pair] <= 0:
                    del self.pair_counts[pair]
            
            if pair in self.pair_to_pretokens:
                self.pair_to_pretokens[pair].discard(pre_token)
                if not self.pair_to_pretokens[pair]:
                    del self.pair_to_pretokens[pair]
    
    def _add_all_pairs_for_pretoken(self, pre_token: str, dll: DoublyLinkedList, count: int):
        """Add all pairs from a pre-token to global statistics."""
        pairs = dll.get_pairs()
        for node, left, right in pairs:
            pair = (left, right)
            
            # Add to global count
            if pair not in self.pair_counts:
                self.pair_counts[pair] = 0
            self.pair_counts[pair] += count
            
            # Track which pre-token contains this pair
            if pair not in self.pair_to_pretokens:
                self.pair_to_pretokens[pair] = set()
            self.pair_to_pretokens[pair].add(pre_token)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer using the efficient BPETrainer.
    
    Args:
        input_path: Path to training corpus
        vocab_size: Target vocabulary size (including special tokens)
        special_tokens: List of special tokens to add to vocabulary
        **kwargs: Additional arguments including:
            - num_processes: Override automatic process calculation
            - chunk_size_mb: Target MB per process (default: 50)
        
    Returns:
        Tuple of (vocab dict, merge list)
    """
    num_processes = kwargs.get("num_processes")
    chunk_size_mb = kwargs.get("chunk_size_mb", 50)
    trainer = BPETrainer(special_tokens=special_tokens)
    return trainer.train(input_path, vocab_size, num_processes=num_processes, chunk_size_mb=chunk_size_mb)


class BPETokenizer:
    """BPE Tokenizer for encoding and decoding text."""
    
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab: Mapping from token ID to token bytes
            merges: List of merge rules
            special_tokens: List of special tokens
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        
        # Create reverse vocab for encoding
        self.byte_to_id = {v: k for k, v in vocab.items()}
        
        # Create merge lookup for efficient encoding
        self.merge_ranks = {merge: i for i, merge in enumerate(merges)}
        
        # Compile regex patterns
        self.pat = re.compile(PAT)
    
    def _pre_tokenize(self, text: str) -> Iterator[str]:
        """
        Split text into pre-tokens using regex pattern.
        
        Special tokens are preserved as single tokens, while regular text
        is tokenized using the PAT pattern.
        
        Args:
            text: Input text to tokenize
            
        Yields:
            Pre-tokens (either special tokens or PAT-matched tokens)
        """
        if self.special_tokens:
            # Use capturing groups to preserve special tokens during split
            escaped_specials = [re.escape(token) for token in self.special_tokens]
            special_pattern = f"({'|'.join(escaped_specials)})"
            chunks = re.split(special_pattern, text)
        else:
            chunks = [text]
        
        for chunk in chunks:
            if not chunk:  # Skip empty chunks from split
                continue
            elif chunk in self.special_tokens:
                # Yield special tokens as-is
                yield chunk
            else:
                # Apply PAT tokenization to regular text
                for match in re.finditer(PAT, chunk):
                    pre_token = match.group()
                    if pre_token:  # Skip empty matches
                        yield pre_token

    def encode(self, text: str) -> list[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of token IDs
        """
        if not text:
            return []
        
        all_ids = []
        
        # Pre-tokenize the text
        for pre_token in self._pre_tokenize(text):
            # Check if this is a special token
            if pre_token in self.special_tokens:
                special_bytes = pre_token.encode('utf-8')
                if special_bytes in self.byte_to_id:
                    all_ids.append(self.byte_to_id[special_bytes])
            else:
                # Regular BPE encoding
                all_ids.extend(self._encode_bytes(pre_token.encode('utf-8')))
        
        return all_ids
    
    def _encode_bytes(self, text_bytes: bytes) -> list[int]:
        """
        Encode bytes without considering special tokens.
        
        Args:
            text_bytes: Bytes to encode
            
        Returns:
            List of token IDs
        """
        if not text_bytes:
            return []
        
        # Start with individual bytes as tokens
        tokens = [bytes([b]) for b in text_bytes]
        original_tokens = tokens[:]
        # Apply merges iteratively
        while len(tokens) > 1:
            # Find the best merge
            best_merge = None
            best_rank = float('inf')
            best_idx = -1
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_merge = pair
                        best_idx = i
            
            # If no merge found, we're done
            if best_merge is None:
                break

            # Apply the merge
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_merge:
                    # Merge the pair
                    merged = tokens[i] + tokens[i + 1]
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            tokens = new_tokens
        
        # Convert to IDs
        return [self.byte_to_id[token] for token in tokens]
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        if not ids:
            return ""
        
        # Convert IDs to bytes
        byte_tokens = [self.vocab[id] for id in ids]
        
        # Concatenate all bytes
        all_bytes = b''.join(byte_tokens)
        
        # Decode to string
        return all_bytes.decode('utf-8', errors='replace')
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Memory-efficient encoding of an iterable of strings.
        
        Args:
            iterable: Iterable of strings to encode
            
        Yields:
            Token IDs one at a time
        """
        for line in iterable:
            for token_id in self.encode(line):
                yield token_id


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> BPETokenizer:
    """
    Create a BPE tokenizer from vocabulary and merges.
    
    Args:
        vocab: Mapping from token ID to token bytes
        merges: List of merge rules
        special_tokens: List of special tokens
        
    Returns:
        BPETokenizer instance
    """
    return BPETokenizer(vocab, merges, special_tokens)


# Adapter function for tests
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Adapter function for test compatibility."""

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens, **kwargs)
    save = kwargs.get("save", False)
    output_filename = kwargs.get("output_filename", None)
    save_format = kwargs.get("save_format", "json_txt")  # Options: "json_txt", "pkl", "both"
    
    if save and output_filename:
        import json
        import pickle
        from tests.common import gpt2_bytes_to_unicode
        
        # Save in JSON/TXT format (default, GPT-2 compatible)
        if save_format in ["json_txt", "both"]:
            # Get GPT-2 byte-to-unicode mapping
            bytes_to_unicode = gpt2_bytes_to_unicode()
            
            # Convert vocab: {id: bytes} -> {"gpt2_string": id}
            vocab_json = {}
            for token_id, token_bytes in vocab.items():
                # Convert bytes to GPT-2 unicode representation
                gpt2_string = ''.join(bytes_to_unicode[b] for b in token_bytes)
                vocab_json[gpt2_string] = token_id
            
            # Save vocab as JSON
            vocab_path = output_filename.replace('TYPE', 'vocab').replace('.pkl', '.json')
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(vocab_json, f, indent=4, ensure_ascii=False)
            
            # Convert merges: [(bytes, bytes)] -> ["gpt2_string1 gpt2_string2"]
            merge_lines = []
            for left_bytes, right_bytes in merges:
                left_gpt2 = ''.join(bytes_to_unicode[b] for b in left_bytes)
                right_gpt2 = ''.join(bytes_to_unicode[b] for b in right_bytes)
                merge_lines.append(f"{left_gpt2} {right_gpt2}")
            
            # Save merges as TXT
            merges_path = output_filename.replace('TYPE', 'merges').replace('.pkl', '.txt')
            with open(merges_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(merge_lines))
            
            print(f"Saved vocab to: {vocab_path}")
            print(f"Saved merges to: {merges_path}")
        
        # Save in PKL format (raw Python objects)
        if save_format in ["pkl", "both"]:
            # Save vocab as PKL
            vocab_pkl_path = output_filename.replace('TYPE', 'vocab')
            if not vocab_pkl_path.endswith('.pkl'):
                vocab_pkl_path += '.pkl'
            with open(vocab_pkl_path, 'wb') as f:
                pickle.dump(vocab, f)
            
            # Save merges as PKL
            merges_pkl_path = output_filename.replace('TYPE', 'merges')
            if not merges_pkl_path.endswith('.pkl'):
                merges_pkl_path += '.pkl'
            with open(merges_pkl_path, 'wb') as f:
                pickle.dump(merges, f)
            
            print(f"Saved vocab (pkl) to: {vocab_pkl_path}")
            print(f"Saved merges (pkl) to: {merges_pkl_path}")
    
    return vocab, merges

if __name__ == "__main__":
    import argparse, time, pickle
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument("--input-path", type=str, default = '/Users/chenli/Project/stanford_cs336/assignment/data/owt_train.txt', help="Path to the input text file or directory.")
    parser.add_argument("--vocab-size", type=int, required=True, help="Vocabulary size to train the BPE tokenizer.")
    parser.add_argument("--special-tokens", type=lambda s: s.split(","), default=['<|endoftext|>'], help="Comma-separated list of special tokens, e.g. <pad>,<eos>.")
    parser.add_argument("--output-filename", type=str, default='/Users/chenli/Project/stanford_cs336/assignment/assignment1-basics/result/owt.TYPE.pkl', help="Folder to save the trained tokenizer.")
    parser.add_argument("--save-format", type=str, choices=["json_txt", "pkl", "both"], default="json_txt", help="Save format: json_txt (default), pkl, or both.")
    parser.add_argument("--num-processes", type=int, help="Number of processes to use (default: auto-calculated based on file size)")
    parser.add_argument("--chunk-size-mb", type=int, default=50, help="Target chunk size per process in MB for memory management (default: 50)")

    args = parser.parse_args()

    start = time.time()
    run_train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        output_filename=args.output_filename,
        save_format=args.save_format,
        num_processes=args.num_processes,
        chunk_size_mb=args.chunk_size_mb,
        save=True
    )
    print(f'FINISH in', time.time() - start, ' SEC')