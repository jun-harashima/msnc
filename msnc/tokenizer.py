from typing import Optional
from typing import Dict


class Tokenizer:
    """
    Wrapper of MeCab and SentencePiece
    Methods: __init__, tokenizer
    """

    def __init__(
        self,
        tokenizer_name: str,
        tokenizer_options: Optional[str]
    ):
        """Initializer for Tokenizer
        
        Arguments:
            tokenizer_name {str} -- specify the name of tokenizer
                                    (mecab/sentencepiece)
            tokenizer_options {Optional[str]} -- options for tokenizer
        """
        if tokenizer_name not in ['mecab', 'sentencepiece']:
            raise Exception("Invalid tokenizer is specified")
        self._tokenizer_name = tokenizer_name

        tokenizer_options = tokenizer_options \
            if tokenizer_options is not None else ""

        if tokenizer_name == 'mecab':
            from natto import MeCab
            import IPython; IPython.embed()
            self._tokenizer = MeCab(tokenizer_options)

        elif tokenizer_name == 'sentencepiece':
            from sentencepiece import SentencePieceProcessor
            self._tokenizer = SentencePieceProcessor()
            self._tokenizer.load(tokenizer_options)

    def tokenize(self, sentence: str):
        """tokenize a sentence into words or word pieces

        Arguments:
            sentence {str} -- raw sentence
        """
        if self._tokenizer_name == 'mecab':
            return self._tokenizer.parse(sentence).split(' ')
        elif self._tokenizer_name == 'sentencepiece':
            return self._tokenizer.EncodeAsPieces(sentence)
