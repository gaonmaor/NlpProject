from os.path import join as os_join


class Config:
    """
    Holds model hyper-params and data information.
    """

    def __init__(self):
        # dropout = 0.15 # TODO: No drop for now
        self.embed_size = 100
        self.hidden_size = 200
        self.batch_size = 10
        self.n_epochs = 2
        self.lr = 0.005
        self.max_length_q = 60
        self.max_length_p = 760
        self.embed_path = os_join("data", "squad",
                                  "glove.trimmed.{}.npz".format(self.embed_size))
        self.max_train_samples = 100
        self.max_val_samples = 100
        self.data_dir = "data/squad"
        self.vocab_path = "data/squad/vocab.dat"
        self.train_dir = "train"
        # TODO: Change when have a stable good modelץ
        self.load_train_dir = ""
