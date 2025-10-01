from pathlib import Path
from tests.adapters import run_train_bpe

def main():
    input_path = str(Path('data') / 'owt_valid.txt')
    vocab_size = 10000
    special_tokens = ['<|endoftext|>']

    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )


    print("Vocab:", vocab)
    print("Merges:", merges)

    # Save vocab and merges to a text file under results folder
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / 'bpe_vocab_merges_opentext.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('Vocab:\n')
        for k, v in vocab.items():
            f.write(f'{k}: {v}\n')
        f.write('\nMerges:\n')
        for left, right in merges:
            f.write(f'{left} + {right} = {left + right}\n')
    print(f"Saved vocab and merges to {output_file}")

if __name__ == "__main__":
    main()