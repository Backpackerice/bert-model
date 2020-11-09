#!pip install transformers==2.11.0
#!pip install torch==1.5.0

import os
os.environ['PYTHONHASHSEED']=str(228)

import sys

import pickle
from path import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split

from transformers import BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup


from contact_dataset import ReviewDataset
from model import ReviewClassification
from train_valid_test import model_train
from utils import ROOT_DIR, set_logging, set_seed, worker_init_fn
from torch.utils.data import Dataset


from utils import Argument
args = Argument()

out_dir = ROOT_DIR / Path('runs/{:d}/'.format(args.experiment_id))

if not out_dir.exists():
    out_dir.mkdir()
else:
    ans = input("Will delete all the files under {:s}. Please enter Yes or No: \n".format(out_dir))
    if ans.lower() == "yes" or ans.lower() == "y":
        filelist = [f for f in out_dir.files()]
        for f in filelist:
            print("deleting {:s}".format(f))
            Path.remove(f)

# set up logging
logger = set_logging(out_dir / "model.log")
stderr_write_copy = sys.stderr.write
stdout_write_copy = sys.stdout.write
# redirect print to log
sys.stderr.write = logger.error
sys.stdout.write = logger.info
sys.stderr.write = stderr_write_copy
sys.stdout.write = stdout_write_copy

# set up gpus
print("args:")
print(str(args))
logger.info("  args:")
logger.info("\n " + str(args))
with open(out_dir/"args.pickle", "wb") as pickle_out:
    pickle.dump(args, pickle_out)

if torch.cuda.device_count() == 0:
    batch_size = args.batch_size_per_gpu
else:
    batch_size = args.batch_size_per_gpu * torch.cuda.device_count()


# ================================================================================
#               loading dataset
# ================================================================================
print("## load dataset")
# if you want to change another tokenizer, go to the preprocessing.py
#cache_file = ROOT_DIR/"cache/Cleaned_data_for_BERT_train.pickle"
cache_file = ROOT_DIR/"cache/Cleaned_data_for_BERT_train.pickle"
dataset = ReviewDataset(embedding_cache=cache_file,
                        trunc_side=args.truncate_side, max_length=args.seq_len, padding_side=args.padding_side)

cache_file = ROOT_DIR /"cache/Cleaned_data_for_BERT_test.pickle"
test_dataset = ReviewDataset(embedding_cache=cache_file,
                             trunc_side=args.truncate_side, max_length=args.seq_len, padding_side=args.padding_side)

# get index for train, valid
n_samples = len(dataset)
n_validsp = int(n_samples * args.valid_pct)
n_trainsp = n_samples - n_validsp

## Set the seed value all over the place to make this reproducible.
# However, not convinced that setting the seed values at the beginning of the training loop is actually creating reproducible results…
set_seed(args.seed)
train_dataset, valid_dataset = random_split(dataset, [n_trainsp, n_validsp])


# Leaning the data
data_loader = {
    "train": DataLoader(train_dataset, batch_size=batch_size, #sampler = train_sampler,
                        num_workers=args.num_workers, drop_last=False, shuffle=True, worker_init_fn = worker_init_fn),
    "valid": DataLoader(valid_dataset, batch_size=batch_size, #sampler = valid_sampler,
                        num_workers=args.num_workers, drop_last=False, shuffle=True, worker_init_fn = worker_init_fn),
    "test": DataLoader(test_dataset, batch_size=batch_size, #sampler = test_sampler,
                       num_workers=args.num_workers, drop_last=False, shuffle=False, worker_init_fn = worker_init_fn)
}

# # compute the weight of each label, use for calculate the weighted average F score
# train_prop = prop_tags(dataset.review_df)[dataset.binarizer.classes_]
# assert np.abs(sum(train_prop) - 1) < 1e-5

# ================================================================================
#               define model
# ================================================================================
print("## define model")
from utils import set_seed
set_seed(seed=args.seed)
config = BertConfig.from_pretrained('bert-base-uncased', output_attention=False, output_hidden_states=False)
model = ReviewClassification.from_pretrained('bert-base-uncased', config=config,
                                             add_agent_text=args.add_agent_text,
                                             agent_text_heads=args.agent_text_heads
                                             )

# ================================================================================
#               training parameter setting
# ================================================================================
set_seed(seed=args.seed)
if args.full_finetuning:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer]}
    ]

# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(optimizer_grouped_parameters,  # or param_optimizer
                  lr=args.lr,  # args.learning_rate - default is 5e-5, our notebook had 1e-5
                  eps=1e-8)  # args.adam_epsilon  - default is 1e-8.
#                   correct_bias=False) #To reproduce BertAdam specific behavior set correct_bias=False

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(data_loader["train"]) * args.num_epochs
)

import gc
# import torch
gc.collect()
torch.cuda.empty_cache()


# ================================================================================
#               model train and model validation
# ================================================================================
## Set the seed value all over the place to make this reproducible.
# However, not convinced that setting the seed values at the beginning of the training loop is actually creating reproducible results…
set_seed(args.seed)
model_train(model=model, train_data_loader=data_loader["train"], valid_data_loader=data_loader["valid"],
            test_data_loader=data_loader["test"], optimizer=optimizer, scheduler=scheduler,
            num_epochs=args.num_epochs, seed=args.seed, logger=logger, out_dir=out_dir)
