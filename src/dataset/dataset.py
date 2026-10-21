from pathlib import Path
from functools import partial
from datasets import load_dataset, load_from_disk, DatasetDict

def tokenize_batch(batch, tokenizer, train=True):
    questions = tokenizer(batch['question'], padding=False, truncation=False, add_special_tokens=False)
    tokenize_output = dict({
        'input_ids': questions['input_ids'],
        'attention_mask': questions['attention_mask']
    })
    if train:
        solutions = tokenizer(batch['code'], padding=False, truncation=False, add_special_tokens=False)
        tokenize_output['label_ids'] = solutions['input_ids']
    return tokenize_output

def download_dataset(name="SlavaYus/tinysms_qa_12m", save_path="data/tinysms_qa_12m"):
    dataset = load_dataset(name)
    output_dir = Path(save_path)
    dataset.save_to_disk(output_dir)
    return dataset

def get_raw_dataset(raw_data_dir):
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists() or not any(raw_data_dir.iterdir()):
        raw_dataset = download_dataset(save_path=raw_data_dir)
    else:
        raw_dataset = load_from_disk(raw_data_dir)
    
    return raw_dataset

def create_dataset(tokenized_data_dir, tokenizer, raw_data_path=None):
    tokenized_data_dir = Path(tokenized_data_dir)

    if raw_data_path is not None:
        path = Path(raw_data_path)
        dataset = get_raw_dataset(path)

        tokenized_dataset = DatasetDict({
            split_name : split_data.map(
                partial(tokenize_batch, tokenizer=tokenizer, train=(split_name=='train')),
                batched=True,
                batch_size=64, # ?
                remove_columns=split_data.column_names,
                keep_in_memory=False
            )
            for split_name, split_data in dataset.items()
        })

        tokenized_dataset.save_to_disk(tokenized_data_dir)
    else:
        tokenized_dataset = load_from_disk(tokenized_data_dir)

    return tokenized_dataset

def get_tokenized_dataset(tokenized_data_dir, tokenizer, raw_data_path=None):
    tokenized_data_dir = Path(tokenized_data_dir)
    if (tokenized_data_dir / "dataset_dict.json").is_file():
        return load_from_disk(tokenized_data_dir)

    if raw_data_path is None:
        raise ValueError("raw_data_path is required to create a tokenized dataset")

    return create_dataset(
        tokenized_data_dir=tokenized_data_dir,
        tokenizer=tokenizer,
        raw_data_path=raw_data_path,
    )
