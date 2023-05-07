class Document:
    """
    A tokenized document is a document represented as a collection of words (also known as tokens) which is used for text analysis.
    """

    def __init__(self, text, method=None, pattern=None, custom=None, regex=None, domins=None, language=None):
        """
        Tokenize String into document

        :param text: String to tokenize
        :param method:
        :param pattern:
        :param custom:
        :param regex:
        :param domins:
        :param language:
        """
        self.__tokens = text.split(" ")
        self.__original_text = text

    def tokens(self):
        return self.__tokens

    def __str__(self):
        return str(self.tokens())

    def original_text(self):
        return self.__original_text

    def contains(self, pattern):
        return any(pattern in token for token in self.tokens())

    def remove(self, pattern):
        raise NotImplementedError()


def tokenize_text(text, method=None):
    return Document(text, method)
