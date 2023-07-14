import os
import pathlib
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import pandas as pd

from RNAformer.utils.data.rna import IndexDataset, CollatorRNA, TokenBasedRandomSampler

IGNORE_INDEX = -100
PAD_INDEX = 0


class DataModuleRNA(pl.LightningDataModule):

    def __init__(
            self,
            dataframe_path,
            num_cpu_worker,
            num_gpu_worker,
            min_len,
            max_len,
            similarity,
            seed,
            batch_size,
            batch_by_token_size,
            batch_token_size,
            shuffle_pool_size,
            cache_dir,
            predict_canonical,
            oversample_pdb,
            random_ignore_mat,
            partial_training,
            design,
            valid_sets,
            test_sets,
            logger,
    ):
        super().__init__()

        if not os.path.exists(dataframe_path):
            raise UserWarning(f"dataframe does not exist: {dataframe_path}")
        self.dataframe_path = dataframe_path
        self.similarity = similarity

        if isinstance(num_gpu_worker, str):
            num_gpu_worker = len(list(map(int, num_gpu_worker.split(","))))

        self.num_gpu_worker = num_gpu_worker
        if num_cpu_worker is None:
            num_cpu_worker = os.cpu_count()
        self.num_cpu_worker = num_cpu_worker

        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.batch_token_size = batch_token_size
        self.shuffle_pool_size = shuffle_pool_size

        self.batch_size = batch_size
        self.batch_by_token_size = batch_by_token_size

        self.min_len = min_len
        self.max_len = max_len
        self.seed = seed
        self.predict_canonical = predict_canonical
        self.oversample_pdb = oversample_pdb
        self.random_ignore_mat = random_ignore_mat
        self.partial_training = partial_training
        self.design = design

        self.samples_cache_file = f"{dataframe_path.split('/')[-1].split('.')[0]}_len{self.min_len}{self.max_len}_design{design}_canonical{predict_canonical}_oversampling{oversample_pdb}_seed{self.seed}_v3.pth"

        self.logger = logger

        self.ignore_index = IGNORE_INDEX
        self.pad_index = PAD_INDEX

        self.seq_vocab = ['A', 'C', 'G', 'U', 'N']
        if self.partial_training:
            self.seq_vocab = ['O'] + self.seq_vocab
        if design:
            self.seq_vocab = ['EOS'] + self.seq_vocab

        self.canonical_pairs = ['GC', 'CG', 'AU', 'UA', 'GU', 'UG']
        self.seq_stoi = dict(zip(self.seq_vocab, range(len(self.seq_vocab))))
        nucs = {
            'T': 'U',
            'P': 'U',
            'R': 'A',  # or 'G'
            'Y': 'C',  # or 'T'
            'M': 'C',  # or 'A'
            'K': 'U',  # or 'G'
            'S': 'C',  # or 'G'
            'W': 'U',  # or 'A'
            'H': 'C',  # or 'A' or 'U'
            'B': 'U',  # or 'G' or 'C'
            'V': 'C',  # or 'G' or 'A'
            'D': 'A',  # or 'G' or 'U'
        }
        self.seq_itos = dict((y, x) for x, y in self.seq_stoi.items())
        for nuc, mapping in nucs.items():
            self.seq_stoi[nuc] = self.seq_stoi[mapping]

        if self.predict_canonical:
            self.struct_vocab = ['.', '(0c', ')0c', '(1c', ')1c', '(2c', ')2c', '(0nc', ')0nc', '(1nc', ')1nc', '(2nc',
                                 ')2nc']
        else:
            self.struct_vocab = ['.', '(0', ')0', '(1', ')1', '(2', ')2']

        self.struct_itos = dict(zip(range(len(self.struct_vocab)), self.struct_vocab))
        self.struct_stoi = dict((y, x) for x, y in self.struct_itos.items())

        self.seq_vocab_size = len(self.seq_vocab)
        self.struct_vocab_size = len(self.struct_vocab)

        self.rng = np.random.RandomState(self.seed)
        self.collator = CollatorRNA(self.pad_index, self.ignore_index)

        self.valid_sets = valid_sets
        self.test_sets = test_sets

    def prepare_data(self):

        if not os.path.exists(self.cache_dir / self.samples_cache_file):
            self.logger.info(
                f"Checked preprocessed data: {(self.cache_dir / self.samples_cache_file).as_posix()} does not exist.")
            self.logger.info("Start prepare data")

            df = pd.read_pickle(self.dataframe_path)
            df = df[df['pos1id'].apply(len) >= 1]  # remove only '.' samples, should be removed already
            df = df[df['pos1id'].apply(len) == df['pos2id'].apply(
                len)]  # remove only '.' samples, should be removed already
            self.logger.info(f'Finished loading dataframe (shape: {df.shape})')

            train_df = df[df['set'].str.contains("train")]
            train_df = train_df[train_df['sequence'].apply(lambda x: self.min_len <= len(x) <= self.max_len)]
            train_df = train_df.reset_index()
            train_samples = []
            for id, sample in train_df.iterrows():
                sample = self._prepare_RNA_sample(sample)
                train_samples.append(sample)
                if sample['pdb_sample'] == 1 and self.oversample_pdb:
                    for _ in range(self.oversample_pdb - 1):
                        train_samples.append(sample)
            samples_dict = {"train": train_samples}
            self.logger.info(f'Finished preprocessing {len(train_samples)} train samples')

            for valid_name in self.valid_sets:
                valid_df = df[df['set'].str.contains(valid_name)]
                valid_df = valid_df[valid_df['sequence'].apply(lambda x: self.min_len <= len(x) <= self.max_len)]
                valid_df = valid_df.reset_index()
                valid_samples = []
                for id, sample in valid_df.iterrows():
                    sample = self._prepare_RNA_sample(sample)
                    valid_samples.append(sample)
                samples_dict[valid_name] = valid_samples
                self.logger.info(f'Finished preprocessing {len(valid_samples)} {valid_name} samples')

            for test_name in self.test_sets:
                test_df = df[df['set'].str.contains(test_name)]
                test_df = test_df.reset_index()
                test_samples = []
                for id, sample in test_df.iterrows():
                    sample = self._prepare_RNA_sample(sample)
                    test_samples.append(sample)
                samples_dict[f"test_{test_name}"] = test_samples
                self.logger.info(f'Finished preprocessing {len(test_samples)} {test_name} samples')

            torch.save(samples_dict, self.cache_dir / self.samples_cache_file)
            self.logger.info('Dumped samples.')

    @staticmethod
    def sequence2index_vector(sequence, mapping):
        int_sequence = list(map(mapping.get, sequence))
        return torch.LongTensor(int_sequence)

    def _create_dot_bracket(self, sequence, pos1id, pos2id, pk_list):
        structure = ['.'] * len(sequence)

        for id1, id2, pk in zip(pos1id, pos2id, pk_list):
            pk = min(pk, 2)
            if self.predict_canonical:
                pair_type = "c" if sequence[id1] + sequence[id2] in self.canonical_pairs else "nc"
                if id1 < id2:
                    structure[id1] = f"({pk}{pair_type}"
                    structure[id2] = f"){pk}{pair_type}"
                else:
                    structure[id2] = f"({pk}{pair_type}"
                    structure[id1] = f"){pk}{pair_type}"
            else:
                if id1 < id2:
                    structure[id1] = f"({pk}"
                    structure[id2] = f"){pk}"
                else:
                    structure[id2] = f"({pk}"
                    structure[id1] = f"){pk}"

        return structure

    def _prepare_RNA_sample(self, input_sample):

        sequence = input_sample["sequence"]
        pos1id = input_sample["pos1id"]
        pos2id = input_sample["pos2id"]
        pk_list = input_sample["pk"]
        if 'is_pdb' in input_sample:
            pdb_sample = int(input_sample['is_pdb'])
        else:
            pdb_sample = 0

        length = len(sequence)

        target_structure = self._create_dot_bracket(sequence, pos1id, pos2id, pk_list)
        src_seq = self.sequence2index_vector(sequence, self.seq_stoi)
        trg_struct = self.sequence2index_vector(target_structure, self.struct_stoi)

        torch_sample = {}
        torch_sample['src_seq'] = src_seq.clone()
        torch_sample['src_struct'] = trg_struct.clone()

        if self.design:
            shifted_sequence = ['EOS'] + sequence[:-1]
            torch_sample['shifted_seq'] = self.sequence2index_vector(shifted_sequence, self.seq_stoi)

        torch_sample['length'] = torch.LongTensor([length])[0]
        torch_sample['pos1id'] = torch.LongTensor(pos1id)
        torch_sample['pos2id'] = torch.LongTensor(pos2id)
        torch_sample['pdb_sample'] = torch.LongTensor([pdb_sample])[0]

        torch_sample['trg_seq'] = src_seq.clone()
        torch_sample['trg_struct'] = trg_struct.clone()

        if self.partial_training:
            torch_sample['post_seq'] = src_seq.clone()
            torch_sample['post_struct'] = trg_struct.clone()

        return torch_sample

    def setup(self, stage):

        local_rank = torch.distributed.get_rank()

        sample_dict = torch.load(self.cache_dir / self.samples_cache_file)
        self.logger.info(
            f"Load preprocessed data from {(self.cache_dir / self.samples_cache_file).as_posix()} at rank {local_rank}.")

        for set_name, set in sample_dict.items():
            self.logger.info(f'Load preprocessed {set_name} {len(set)} samples')

        self.train_samples = sample_dict['train']
        train_indexes = list(range(len(self.train_samples)))
        train_index_dataset = IndexDataset(train_indexes)
        token_key_fn = lambda s: len(self.train_samples[s]['src_seq'])
        self.minibatch_sampler = TokenBasedRandomSampler(train_index_dataset,
                                                         token_key_fn,
                                                         batch_token_size=self.batch_token_size,
                                                         batching=True,
                                                         repeat=False,
                                                         sort_samples=True,
                                                         shuffle=True,
                                                         shuffle_pool_size=self.shuffle_pool_size,
                                                         drop_last=True,
                                                         seed=self.seed
                                                         )

        self.valid_minibatch_sampler = {}
        self.valid_samples_dict = {}
        for valid_name in self.valid_sets:
            valid_samples = sample_dict[valid_name]
            self.valid_samples_dict[valid_name] = valid_samples
            valid_indexes = list(range(len(valid_samples)))
            valid_index_dataset = IndexDataset(valid_indexes)
            valid_token_key_fn = lambda s: len(valid_samples[s]['src_seq'])
            val_sampler = TokenBasedRandomSampler(valid_index_dataset,
                                                  valid_token_key_fn,
                                                  batch_token_size=self.batch_token_size,
                                                  batching=True,
                                                  repeat=False,
                                                  sort_samples=True,
                                                  shuffle=False,
                                                  shuffle_pool_size=self.shuffle_pool_size,
                                                  drop_last=False,
                                                  seed=self.seed
                                                  )
            self.valid_minibatch_sampler[valid_name] = val_sampler

    def ignore_partial_mat(self, batch):

        trg_mat = batch['trg_mat']

        with torch.no_grad():
            filter = torch.nn.Conv2d(1, 1, 3, 1, 1, bias=False)
            torch.nn.init.constant_(filter.weight, 1.0)
            trg_mat_no_ignore = torch.where(trg_mat == self.ignore_index, torch.tensor(0), trg_mat)
            filter_trg = filter(trg_mat_no_ignore.unsqueeze(1).float()).squeeze(1)
            filter_trg = filter_trg.bool()

        rand_mat = torch.rand_like(trg_mat.float()) < self.random_ignore_mat
        rand_mat = torch.logical_and(rand_mat, torch.logical_not(filter_trg))
        trg_mat.masked_fill_(rand_mat, self.ignore_index)

        batch['trg_mat'] = trg_mat
        return batch

    def train_dataloader(self):
        """This will be run every epoch."""

        if self.batch_by_token_size:
            minibatches = self.minibatch_sampler.precompute_minibatches()

            if self.num_gpu_worker > 1:
                local_rank = torch.distributed.get_rank()
                numb_batches = len(minibatches)
                batches_per_rank = numb_batches // self.num_gpu_worker

                minibatches = minibatches[local_rank * batches_per_rank: (local_rank + 1) * batches_per_rank]

            def train_pl_collate_fn(indices):
                indices = minibatches[indices[0]]
                if self.partial_training:
                    raw_samples = [self.partial_sample(self.train_samples[i]) for i in indices]
                else:
                    raw_samples = [self.train_samples[i] for i in indices]

                batch = self.collator(raw_samples)

                if self.random_ignore_mat:
                    batch = self.ignore_partial_mat(batch)

                return batch

            train_indexes = list(range(len(minibatches)))
            train_index_dataset = IndexDataset(train_indexes)

            loader = DataLoader(
                train_index_dataset,
                batch_size=1,
                collate_fn=train_pl_collate_fn,
                num_workers=self.num_cpu_worker,
                pin_memory=False,
                drop_last=False,
            )

        else:
            def train_pl_collate_fn(indices):

                raw_samples = [self.train_samples[i] for i in indices]

                batch = self.collator(raw_samples)

                if self.random_ignore_mat:
                    batch = self.ignore_partial_mat(batch)

                return batch

            train_indexes = list(range(len(self.train_samples)))
            train_indexes = self.rng.permutation(train_indexes)

            if self.num_gpu_worker > 1:
                local_rank = torch.distributed.get_rank()
                numb_samples = len(train_indexes)
                batches_per_rank = numb_samples // self.num_gpu_worker

                if numb_samples != self.num_gpu_worker * batches_per_rank:
                    train_indexes = train_indexes[: numb_samples - numb_samples % self.num_gpu_worker]

                train_indexes = train_indexes[local_rank * batches_per_rank: (local_rank + 1) * batches_per_rank]

            train_index_dataset = IndexDataset(train_indexes)

            loader = DataLoader(
                train_index_dataset,
                batch_size=self.batch_size,
                collate_fn=train_pl_collate_fn,
                num_workers=self.num_cpu_worker,
                shuffle=True,
                pin_memory=False,
                drop_last=True,
            )

        self.logger.info("Finished loading training data")
        return loader

    def val_dataloader(self):

        dataloader_list = []
        for set_name in self.valid_sets:

            if set_name not in self.valid_samples_dict:
                continue

            def val_pl_collate_fn(raw_samples):
                return self.collator(raw_samples)

            val_loader = DataLoader(
                self.valid_samples_dict[set_name],
                batch_size=self.batch_size,
                collate_fn=val_pl_collate_fn,
                num_workers=self.num_cpu_worker,
                shuffle=False,
                pin_memory=False,
                drop_last=False,
            )

            dataloader_list.append(val_loader)
        self.logger.info(f"Finished loading validation data")
        return dataloader_list
