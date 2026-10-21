from transformers import PreTrainedTokenizerFast
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    decoders,
    trainers,
)
from pathlib import Path

def train_tokenizer(raw_dataset, output_dir, vocab_size=16000):
    tokenizer = Tokenizer(
        models.BPE(unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
    )
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2, # TODO: добавить загрузку из конфига 
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[
            "[PAD]",
            "[UNK]",
            "[BOS]",
            "[EOS]",
        ],
        show_progress=True,
    )

    def training_texts(dataset):
        for example in dataset:
            yield example["question"]
            yield example["code"]

    tokenizer.train_from_iterator(
        training_texts(raw_dataset['train']),
        trainer=trainer,
    )

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        clean_up_tokenization_spaces=False,
    )

    output_dir = Path(output_dir)
    fast_tokenizer.save_pretrained(output_dir)

    return fast_tokenizer



def load_tokenizer(tokenizer_dir):
    return PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)


# При смене токенизатора старый токенизированный кеш нужно пересоздать

def get_tokenizer(tokenizer_dir, raw_dataset, vocab_size=16000):
    tokenizer_dir = Path(tokenizer_dir)

    if (tokenizer_dir / "tokenizer.json").is_file():
        return load_tokenizer(tokenizer_dir)

    return train_tokenizer(
        raw_dataset,
        output_dir=tokenizer_dir,
        vocab_size=vocab_size,
    )
