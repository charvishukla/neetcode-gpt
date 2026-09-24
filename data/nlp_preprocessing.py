import torch
import torch.nn as nn
from torchtyping import TensorType

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        vocab = set()

        for sentence in positive:
            for word in sentence.split():
                vocab.add(word)
        
        for sentence in negative:
            for word in sentence.split():
                vocab.add(word)


        sorted_list = sorted(list(vocab))
        word_to_int_map = {}
        for i, c in enumerate(sorted_list):
            word_to_int_map[c] = i +1


        tensors = []
        for sentence in positive:
            curr_list = []
            for word in sentence.split():
                curr_list.append(word_to_int_map[word])

            tensors.append(torch.tensor(curr_list))
        
        for sentence in negative:
            curr_list = []
            for word in sentence.split():
                curr_list.append(word_to_int_map[word])

            tensors.append(torch.tensor(curr_list))

        return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)