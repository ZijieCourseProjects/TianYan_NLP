from unittest import TestCase


class TestDocument(TestCase):
    def test_show(self):
        self.fail()

    def test_tokenize_text(self):
        from Document import tokenize_text
        self.assertListEqual(tokenize_text("Hello World", "split").tokens(), ["Hello", "World"])
