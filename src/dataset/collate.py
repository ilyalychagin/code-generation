
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import torch


def collate(batch, trunc_len=None, padding_value=0, end_token=None, train=True): # TODO: Не забыить что-то там с падинговыми токенами сделать перед подачей в модель
    if trunc_len is not None:
        inputs = [input['input_ids'][:trunc_len] for input in batch] 
        attention_masks = [input['attention_mask'][:trunc_len] for input in batch]
        if train:
            labels = [input['label_ids'][:trunc_len] for input in batch] 
        else:
            labels = None
    else:
        inputs = [input['input_ids'] for input in batch] 
        attention_masks = [input['attention_mask'] for input in batch]
        if train:
            labels = [input['label_ids'] for input in batch]
        else:
            labels = None

    if end_token is None:
        raise ValueError("end_token must be specified")

    inputs = [input + [end_token] for input in inputs]
    attention_masks = [attention_mask + [1] for attention_mask in attention_masks]
    if train:
        labels = [label + [end_token] for label in labels]
    
    def pad(batch, padding_value):
        return pad_sequence(
            [torch.tensor(sample) for sample in batch], 
            batch_first=True,
            padding_value=padding_value
        )

    inputs = pad(inputs, padding_value)
    attention_masks = pad(attention_masks, 0)
    if train:
        labels = pad(labels, -100)

    collate_output = dict({
        'input_ids' : inputs,
        'attention_mask' : attention_masks
    })

    if train:
        collate_output['labels'] = labels

    return collate_output