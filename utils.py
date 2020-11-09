import random
import numpy as np
import pandas as pd
import multiprocessing as mp
import torch
from path import Path

# ======================================================================================================================
#              Step 1: Constant Variables
# ======================================================================================================================

ROOT_DIR = Path(__file__).dirname()

LABEL_NAME = [0,1]

# ======================================================================================================================
#              Step 2: Useful Functions
# ======================================================================================================================
def isnotebook():
    from IPython import get_ipython
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def parallel_map_df(func, df, desc=None):
    assert isinstance(df, pd.DataFrame)
    temp = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for index, *items in tqdm(pool.imap_unordered(func, df.itertuples(name=None)), total=len(df), desc=desc):
            temp.append([index, *items])
    return temp


def set_logging(filename):
    # logging reference: https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the file handler to the logger
    logger.addHandler(handler)

    return logger

# Set the seed value all over the place to make this reproducible.
def set_seed(seed):
    """ Set all seeds to make results reproducible (deterministic mode).
        When seed is a false-y value or not supplied, disables deterministic mode. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)
    
def format_time(second):
    hour = int(second // 3600)
    minute = int((second - hour*3600) // 60)
    second = int(second % 60)
    return "{}:{:0>2}:{:0>2}".format(hour, minute, second)


# =====================================================================================================================
#              Step 3: Useful Classes
# ======================================================================================================================
class Argument:
    def __init__(self):
        self.experiment_id = 0
        self.valid_pct = 0.1
        self.seed = 228
        self.seq_len = 512
        self.padding_side = "right"
        self.truncate_side = "random"
        self.use_distil_model = False
        self.batch_size_per_gpu = 16 #bigger batch size need bigger learning rate. Smaller batch size need smaller learning rate
        self.num_epochs = 3
        self.num_workers = 8
        self.lr = 3e-5
        self.full_finetuning = True
        self.decay = 0.01  # this acts as l2 penalty
        self.add_agent_text = None
        self.agent_text_heads = 12 if self.add_agent_text == "attention" else None

        self.check_param()

    def __str__(self):
        return """
        experiment ID                       : {}
        valid percentage                    : {}
        seed                                : {}
        padding side                        : {}
        truncate side                       : {}
        use distil model                    : {}
        batch size/gpu                      : {}
        num of epochs                       : {}
        num of workers                      : {}
        maximun seq len                     : {}
        learning rate                       : {}
        full tunning                        : {}
        decay rate                          : {}
        add agent text                      : {}
        number of heads                     : {}
        """.format(self.experiment_id, self.valid_pct, self.seed, self.padding_side, self.truncate_side,
                   self.use_distil_model, self.batch_size_per_gpu, self.num_epochs, self.num_workers, self.seq_len,
                   self.lr, self.full_finetuning, self.decay, 
                   self.add_agent_text, self.agent_text_heads)

    def check_param(self):
        assert self.padding_side in ["right", "left"]
        assert self.truncate_side in ["right", "left", "random"]
        assert self.add_agent_text in ["attention", "concat", None]
