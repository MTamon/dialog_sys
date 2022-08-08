import re
from pyknp import Juman
from transformers import BertTokenizer
import torch

jumanpp = Juman()

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device : ", device)

# parameters
bert_seq_len = 512

class JumanTokenizer():
    def __init__(self):
        self.juman = Juman()

    def tokenize(self, text):
        result = self.juman.analysis(text)
        return [mrph.midasi for mrph in result.mrph_list()]

class BertWithJumanModel(BertTokenizer):
    def __init__(self, vocab_file_name="vocab.txt"):
        super().__init__(
            vocab_file_name,
            do_lower_case=False,
            do_basic_tokenize=False
        )
        self.juman_tokenizer = JumanTokenizer()

    def _preprocess_text(self, text):
        return text.replace(" ", "")  # for Juman

    def __call__(self, text):
        preprocessed_text = self._preprocess_text(text)
        tokens = self.juman_tokenizer.tokenize(preprocessed_text)
        bert_tokens = self.tokenize(" ".join(tokens))

        return bert_tokens

bert_tokenizer = BertWithJumanModel('bert/vocab.txt')

with open('inputs.txt', 'r') as fi, open('inputs_tknz.txt', 'w') as fo:
    for line in fi:
        for sentence in re.split('[,.、。，．]', line):
            sentence = re.sub('\n', '', sentence)
            fo.write(' '.join(bert_tokenizer(sentence)) + '\n')
