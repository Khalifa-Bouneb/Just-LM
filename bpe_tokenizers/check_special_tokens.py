from tests.adapters import run_train_bpe
from tests.common import FIXTURES_PATH

input_path = FIXTURES_PATH / 'tinystories_sample_5M.txt'
vocab, merges = run_train_bpe(
    input_path=input_path,
    vocab_size=1000,
    special_tokens=['<|endoftext|>'],
)

# Check that the special token is in the vocab
print('Special token in vocab:', b'<|endoftext|>' in vocab.values())

# Check that no other vocab items contain parts of special tokens
vocabs_without_specials = [word for word in vocab.values() if word != b'<|endoftext|>']
problematic_items = [word_bytes for word_bytes in vocabs_without_specials if b'<|' in word_bytes]

print('Number of problematic vocab items:', len(problematic_items))
if problematic_items:
    print('First few problematic items:', problematic_items[:5])
else:
    print('✅ No problematic vocab items found - special token constraint satisfied!')

print('Total vocab size:', len(vocab))
print('Total merges:', len(merges))