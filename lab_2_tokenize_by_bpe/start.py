"""
BPE Tokenizer starter
"""
from pathlib import Path
from lab_2_tokenize_by_bpe.main import collect_frequencies, train, get_vocabulary, decode


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()
    with open(assets_path / 'secrets/secret_1.txt', 'r', encoding='utf-8') as text_file:
        encoded_secret = text_file.read()
    dict_frequencies = collect_frequencies(text, None, '</s>')
    updated_dict_frequencies = train(dict_frequencies, 100)
    if updated_dict_frequencies:
        vocabulary = get_vocabulary(updated_dict_frequencies, '<unk>')
        secret = [int(num) for num in encoded_secret.split()]
        result = decode(secret, vocabulary, '</s>')
        print(result)
        assert result, "Encoding is not working"


if __name__ == "__main__":
    main()
