from time import time

import torch
import os

from torch.utils.data import Subset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_dataset, load_from_disk

from flowcept.commons.utils import replace_non_serializable

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data


# Define a function to yield tokens from the dataset
def yield_tokens(tokenizer, data_iter):
    for item in data_iter:
        if len(item["text"]):
            yield tokenizer(item["text"])


# Define a function to process the raw text and convert it to tensors
def data_process(tokenizer, vocab, raw_text_iter):
    i = 0
    data = []
    for item in raw_text_iter:
        tokens = tokenizer(item["text"])
        indices = []
        for token in tokens:
            indices.append(vocab[token])
        tensor = torch.tensor(indices, dtype=torch.long)
        data.append(tensor)
        i += 1
    tensor_return = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
    return tensor_return


def get_dataset_ref(campaign_id, train_data, val_data, test_data):
    dataset_ref = f"{campaign_id}_train_shape_{list(train_data.shape)}_val_shape_{list(val_data.shape)}_test_shape_{list(test_data.shape)}"
    return dataset_ref


def get_wiki_text_dataset(data_dir, batch_size, eval_batch_size):
    # Load the WikiText2 dataset
    t0 = time()
    train_data = torch.load(os.path.join(data_dir, "train_data.tensor"))
    val_data = torch.load(os.path.join(data_dir, "val_data.tensor"))
    test_data = torch.load(os.path.join(data_dir, "test_data.tensor"))
    t1 = time()
    t_disk_load = t1 - t0

    try:
        if torch.cuda.is_available():
            device = torch.device("gpu")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        t2 = time()
        t_device_available = t2 - t1
        train_data = train_data.to(device)
        val_data = val_data.to(device)
        test_data = test_data.to(device)


        t_gpu_load = time() - t2
    except:
        raise Exception("Couldn't send data to device")

    print("Train data", train_data.shape)
    print("Validation data", val_data.shape)
    print("Test data", test_data.shape)
    return (
        train_data,
        val_data,
        test_data,
        t_disk_load,
        t_device_available,
        t_gpu_load,
        device
    )

def save_workflow(campaign_id, used, generated):
    from flowcept import WorkflowObject, Flowcept
    dataset_prep_wf = WorkflowObject()
    dataset_prep_wf.used = used
    dataset_prep_wf.campaign_id = campaign_id
    dataset_prep_wf.name = "generate_wikitext_dataset"

    dataset_prep_wf.generated = generated
    Flowcept.db.insert_or_update_workflow(dataset_prep_wf)
    print(dataset_prep_wf)
    return dataset_prep_wf.workflow_id


def dataprep_workflow(data_dir="input_data",
                      tokenizer_type="basic_english",  # spacy, moses, toktok, revtok, subword
                      batch_size=20,
                      eval_batch_size=10,
                      subset_size=None,
                      campaign_id=None,
                      ):

    os.makedirs(data_dir, exist_ok=True)

    dataset_path = os.path.join(data_dir, "wikitext-2-v1.data")
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        print("Downloading dataset")
        dataset = load_dataset("wikitext", "wikitext-2-v1")
        print(f"Ok, now saving it into {dataset_path}")
        dataset.save_to_disk(dataset_path)

    test_dataset = dataset["test"]
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]
    dataset_info = {
        "train": replace_non_serializable(train_dataset.info.__dict__),
        "val": replace_non_serializable(validation_dataset.info.__dict__),
        "test": replace_non_serializable(test_dataset.info.__dict__)
    }
    if subset_size is not None and subset_size > 0:
        test_dataset = Subset(test_dataset, range(subset_size))
        train_dataset = Subset(train_dataset, range(subset_size))
        validation_dataset = Subset(validation_dataset, range(subset_size))

    # Build the vocabulary from the training dataset
    tokenizer = get_tokenizer(tokenizer_type)
    vocab = build_vocab_from_iterator(yield_tokens(tokenizer, train_dataset))
    vocab.set_default_index(vocab["<unk>"])
    ntokens = len(vocab)

    # Process the train, validation, and test datasets
    train_data = data_process(tokenizer, vocab, train_dataset)
    val_data = data_process(tokenizer, vocab, validation_dataset)
    test_data = data_process(tokenizer, vocab, test_dataset)

    train_data = batchify(train_data, batch_size)
    val_data = batchify(val_data, eval_batch_size)
    test_data = batchify(test_data, eval_batch_size)

    train_n_batches = len(list(enumerate(range(0, train_data.size(0) - 1, batch_size))))
    val_n_batches = len(list(enumerate(range(0, val_data.size(0) - 1, eval_batch_size))))
    test_n_batches = val_n_batches

    train_data_path = os.path.join(data_dir, "train_data.tensor")
    val_data_path = os.path.join(data_dir, "val_data.tensor")
    test_data_path = os.path.join(data_dir, "test_data.tensor")

    torch.save(train_data, train_data_path)
    torch.save(val_data, val_data_path)
    torch.save(test_data, test_data_path)

    print(f"Saved files in {data_dir}. Now running some asserts.")

    train_data_loaded = torch.load(train_data_path)
    val_data_loaded = torch.load(val_data_path)
    test_data_loaded = torch.load(test_data_path)

    assert torch.equal(train_data, train_data_loaded), "Train data mismatch"
    assert torch.equal(val_data, val_data_loaded), "Validation data mismatch"
    assert torch.equal(test_data, test_data_loaded), "Test data mismatch"

    dataset_ref = get_dataset_ref(campaign_id, train_data, val_data, test_data)

    used = {
        "train_batch_size": batch_size,
        "val_batch_size": eval_batch_size,
        "test_batch_size": eval_batch_size,
        "subset_size": subset_size,
        "tokenizer_type": tokenizer_type,
        "dataset_info": dataset_info,
    }
    generated = {
        "ntokens": ntokens,
        "dataset_ref": dataset_ref,
        "train_n_batches": train_n_batches,
        "test_n_batches": test_n_batches,
        "val_n_batches": val_n_batches,
        "train_data_shape": list(train_data.shape),
        "val_data_shape": list(val_data.shape),
        "test_data_shape": list(test_data.shape),
        "train_data_path": train_data_path,
        "test_data_path": test_data_path,
        "val_data_path": val_data_path,
    }
    wf_id = save_workflow(campaign_id, used, generated)
    return wf_id, generated

