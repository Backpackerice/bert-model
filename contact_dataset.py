import pickle
import bz2
import pandas as pd
import numpy as np
from path import Path
from functools import partial
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

import torch
from torch.utils.data import Dataset

from utils import LABEL_NAME, isnotebook

if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# def prop_num_tags(df):
#     # compute the proportion of the number of the tags
#
#     percentage = df["tags"].apply(lambda tags: len(tags.split(", "))).value_counts(normalize=True)
#     percentage.name = "proportion of #tags"
#     print("=" * 50)
#     print("proportion of multi-tags in all of the review text")
#     print(percentage.to_string())
#     return percentage


def pad_trunc_sequences(embedding, max_length=512, padding_side="right", pad_value=0, trunc_side="random"):
    assert padding_side in ["right", "left"], "wrong padding_side"
    assert trunc_side in ["random", "right", "left"], "wrong trunc_side"
    
    # from ast import literal_eval
    # embedding = literal_eval(embedding)
    # need to truncate the review text if its length exceed the MAX LENGTH
    if len(embedding) > max_length - 2:
        if trunc_side == "random":
            beg = np.random.randint(low=0, high=len(embedding) - (max_length - 2))
        elif trunc_side == "left":
            beg = len(embedding) - (max_length - 2)
        elif trunc_side == "right":
            beg = 0
        embedding = embedding[beg: beg + (max_length - 2)]

    # add special token to the text
    input_ids = [101, ] + embedding + [102]
    # convert token to ids
    # padding the tokenized_text according to the padding side
    if padding_side == "right":
        attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
        input_ids = input_ids + [pad_value] * (max_length - len(input_ids))

    elif padding_side == "left":
        attention_mask = [0] * (max_length - len(input_ids)) + [1] * len(input_ids)
        input_ids = [pad_value] * (max_length - len(input_ids)) + input_ids

    return input_ids, attention_mask, [0] * max_length


class ReviewDataset(Dataset):
    def __init__(self, embedding_cache, max_length=512,
                 padding_side="right", pad_value=0, trunc_side="random"):
        assert Path(embedding_cache).exists(), "embedding cache file doesn't exist, need to run preprocessing.py"
        with open(embedding_cache, "rb") as pickle_in:
            print("loading the cache embedding data from pickle...")
            cache_df = pickle.load(pickle_in)

        # Adjust this part if want to try other emmbedding
        self.review_df = cache_df.loc[cache_df["contact_embed"] != '[]']

        partial_pad_trunc = partial(pad_trunc_sequences, max_length=max_length,
                                    padding_side=padding_side, pad_value=pad_value, trunc_side=trunc_side)

        tqdm.pandas(desc="Padding and truncating hmd and head customer embedding...")
        review_input_df = self.review_df["contact_embed"].progress_apply(lambda x: partial_pad_trunc(x))
        self.review_input_ids = np.array([item[0] for item in review_input_df.values], dtype=np.long)
        self.review_attention_mask = np.array([item[1] for item in review_input_df.values], dtype=np.bool)
        self.review_token_type_ids = np.array([item[2] for item in review_input_df.values], dtype=np.long)

        tqdm.pandas(desc="Padding and truncating agent embedding...")
        # change this part if want to use other embed
#         self.review_df['asic_sic_embed'] = self.review_df['asic_embed'] + self.review_df['sic_embed']
        agent_input_df = self.review_df["agent_embed"].progress_apply(lambda x: partial_pad_trunc(x))
        self.agent_input_ids = np.array([item[0] for item in agent_input_df.values], dtype=np.long)
        self.agent_attention_mask = np.array([item[1] for item in agent_input_df.values], dtype=np.bool)
        self.agent_token_type_ids = np.array([item[2] for item in agent_input_df.values], dtype=np.long)
        

        if "anecdote_lead_final" in self.review_df.columns:
            # will convert the tag list to multi-label classification format
            # use this label and order to encode the labels. It will print the warning for I don't encode ""
            self.binarized_label = self.review_df["anecdote_lead_final"].astype(float).values

        print("finished!")

    def __len__(self):
        return self.review_df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, int):
            samples = self.review_df.iloc[idx, :].to_dict()
            samples.update({"_id": self.review_df.iloc[idx, :].name})
        else:
            samples = self.review_df.iloc[idx, :].to_dict("list")
            samples.update({"_id": self.review_df.iloc[idx, :].index.values.tolist()})

        samples.update(
            {
                "review_input_ids":        self.review_input_ids[idx, :],
                "review_attention_mask":   self.review_attention_mask[idx, :],
                "review_token_type_ids":   self.review_token_type_ids[idx, :],
                "agent_input_ids":          self.agent_input_ids[idx, :],
                "agent_attention_mask":     self.agent_attention_mask[idx, :],
                "agent_token_type_ids":     self.agent_token_type_ids[idx, :],
            }
        )

        if "anecdote_lead_final" in self.review_df.columns:
            samples.update(
                {
                    "binarized_label": self.binarized_label[idx]
                }
            )
        return samples

    def label_prop(self):
        if "anecdote_lead_final" in self.review_df.columns:
            if not hasattr(self, 'label_proportion'):
                # compute the proportion of the labels
                count_dict = dict.fromkeys(LABEL_NAME, 0)
                for label in self.review_df["anecdote_lead_final"]:
                    count_dict[label] = count_dict.get(label, 0) + 1

                count = pd.Series(count_dict).sort_index()
                # count = count[count.index != ""]
                self.label_proportion = count / sum(count)
            return self.label_proportion
        else:
            return None


if __name__ == "__main__":
    from utils import set_seed, ROOT_DIR

    set_seed(228)
    embedding_cache = ROOT_DIR / "cache/Cleaned_data_for_BERT_train.pickle"

    dataset = ReviewDataset(embedding_cache=embedding_cache,
                            trunc_side="random", max_length=10, padding_side="right")
    # Below script will bring you to IPython screen 
    from IPython import embed
    embed()


# Then use dataset[0] to see if we can get the text we want
# Check whether the padding, truncate and encode works properly

