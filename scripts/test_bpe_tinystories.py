from pathlib import Path
import numpy as np
from tests.adapters import get_tokenizer

def load_vocab_merges(path):
    """Load vocab and merges from the saved text file (bpe_vocab_merges.txt)."""
    vocab = {}
    merges = []
    section = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "Vocab:":
                section = "vocab"
                continue
            elif line == "Merges:":
                section = "merges"
                continue
            if not line:
                continue
            if section == "vocab":
                idx, token = line.split(": ")
                vocab[int(idx)] = eval(token)  # convert "b'\\x00'" → b'\x00'
            elif section == "merges":
                left, rest = line.split(" + ")
                right, _ = rest.split(" = ")
                merges.append((left.encode("utf-8"), right.encode("utf-8")))
    return vocab, merges


def main():
    # -----------------------------
    # LOAD VOCAB + MERGES
    # -----------------------------
    vocab_path = Path("results") / "bpe_vocab_merges.txt"
    vocab, merges = load_vocab_merges(vocab_path)
    special_tokens = ['<|endoftext|>']

    print("✅ Loaded vocab + merges")
    print("   Vocab size:", len(vocab))
    print("   Number of merges:", len(merges))

    # -----------------------------
    # INIT TOKENIZER
    # -----------------------------
    tokenizer = get_tokenizer(vocab, merges, special_tokens)

    # -----------------------------
    # LOAD DATASET
    # -----------------------------
    input_path = Path("data") / "TinyStoriesV2-GPT4-valid.txt"
    with open(input_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    print(f"✅ Loaded dataset: {input_path}")
    print(f"   Dataset length: {len(text_data)} characters")

    # -----------------------------
    # ENCODE DATASET
    # -----------------------------
    encoded_ids = tokenizer.encode(text_data)

    print("\n🔹 Encoding Done")
    print("   Number of tokens:", len(encoded_ids))
    print("   First 50 token IDs:", encoded_ids[:50])

    # -----------------------------
    # SAVE ENCODED IDS AS UINT16
    # -----------------------------
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_ids_file = results_dir / "tinystories_encoded_ids.npy"

    np_encoded = np.array(encoded_ids, dtype=np.uint16)
    np.save(output_ids_file, np_encoded)

    print(f"✅ Encoded IDs saved to {output_ids_file} (dtype=uint16, shape={np_encoded.shape})")

    # -----------------------------
    # COMPRESSION RATIO
    # -----------------------------
    num_bytes = len(text_data.encode("utf-8"))
    num_tokens = len(encoded_ids)
    ratio = num_bytes / num_tokens if num_tokens > 0 else 0

    print(f"\n📊 Compression Ratio = {ratio:.2f} bytes/token")


if __name__ == "__main__":
    main()
