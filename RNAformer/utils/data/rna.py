from typing import List
from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence


class IndexDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_indices):
        self.dataset_indices = dataset_indices

    def __getitem__(self, index):
        return self.dataset_indices[index]

    def __len__(self):
        return len(self.dataset_indices)


class SortedRandomSampler(torch.utils.data.sampler.BatchSampler):

    def __init__(self, sampler, token_key_fn, batch_size, repeat, sort_samples, shuffle,
                 shuffle_pool_size, drop_last, rng=None, *args, **kwargs):

        super().__init__(sampler, batch_size, drop_last)
        self.sample_length = [token_key_fn(s) for s in sampler]
        self.dataset_size = len(sampler)

        self.batch_size = batch_size

        self.repeat = repeat
        self.sort_samples = sort_samples
        self.drop_last = drop_last

        self.shuffle = shuffle
        self.shuffle_pool_size = shuffle_pool_size

        self.rng = rng

        self.reverse = False

        if not self.repeat:
            self.minibatches = self.precompute_minibatches()

    def __len__(self):
        return len(self.minibatches)

    def get_index_list(self):
        index_list = [i for i in range(self.dataset_size)]
        if self.shuffle:
            self.rng.shuffle(index_list)
        return index_list

    def get_infinit_index_iter(self):
        while True:
            index_list = self.get_index_list()
            for i in index_list:
                yield i

    def get_index_iter(self):
        index_list = self.get_index_list()
        for i in index_list:
            yield i

    def pool_and_sort(self, sample_iter):
        pool = []
        for sample in sample_iter:
            if not self.sort_samples:
                yield sample
            else:
                pool.append(sample)
                if len(pool) >= self.shuffle_pool_size:
                    pool.sort(key=lambda x: self.sample_length[x], reverse=self.reverse)
                    self.reverse = not self.reverse
                    while len(pool) > 0:
                        yield pool.pop()
        if len(pool) > 0:
            pool.sort(key=lambda x: self.sample_length[x], reverse=self.reverse)
            self.reverse = not self.reverse
            while len(pool) > 0:
                yield pool.pop()

    def get_minibatches(self, index_iter):

        minibatch, max_size_in_batch = [], 0

        if self.sort_samples:
            index_iter = self.pool_and_sort(index_iter)

        for sample in index_iter:

            minibatch.append(sample)
            if len(minibatch) >= self.batch_size:
                if self.shuffle:
                    self.rng.shuffle(minibatch)
                yield minibatch
                minibatch = []

        if (not self.drop_last) and len(minibatch) > 0:
            yield minibatch

    def precompute_minibatches(self):
        index_iter = self.get_index_list()

        minibatches = [m for m in self.get_minibatches(index_iter) if len(m) > 0]
        if self.shuffle:
            self.rng.shuffle(minibatches)
        return minibatches

    def __iter__(self):
        if self.repeat:
            index_iter = self.get_infinit_index_iter()
            for minibatch in self.get_minibatches(index_iter):
                yield minibatch
        else:
            for m in self.minibatches:
                yield m


class CollatorRNA:

    def __init__(self, pad_index, ignore_index):
        self.ignore_index = ignore_index
        self.pad_index = pad_index

    def __call__(self, samples, neg_samples=False) -> List[List[int]]:
        # tokenize the input text samples

        batch_dict = defaultdict(list)
        with torch.no_grad():
            for sample in samples:
                for k, v in sample.items():
                    if k in ['src_seq', 'length', 'pdb_sample', 'pos1id', 'pos2id']:
                        batch_dict[k].append(v)

            batch_dict['length'] = torch.stack(batch_dict['length'])
            batch_dict['src_seq'] = pad_sequence(batch_dict['src_seq'], batch_first=True, padding_value=self.pad_index)

            if 'pdb_sample' in batch_dict:
                batch_dict['pdb_sample'] = torch.stack(batch_dict['pdb_sample']).float().unsqueeze(-1)
            else:
                batch_dict['pdb_sample'] = torch.ones_like(batch_dict['length']).float().unsqueeze(-1)

            if 'pos1id' in batch_dict:
                max_len = batch_dict['length'].max()
                batch_size = len(samples)

                max_pos_size = max(pos.shape[0] for pos in batch_dict['pos1id'])
                pos1id = torch.full((batch_size, max_pos_size), self.ignore_index)
                pos2id = torch.full((batch_size, max_pos_size), self.ignore_index)
                trg_mat = torch.LongTensor(batch_size, max_len, max_len).fill_(self.ignore_index)
                mask = torch.BoolTensor(batch_size, max_len, max_len).fill_(False)

                for b_id, sample in enumerate(samples):
                    pos1id[b_id, :sample['pos1id'].size(0)] = sample['pos1id']
                    pos2id[b_id, :sample['pos2id'].size(0)] = sample['pos2id']
                    trg_mat[b_id, :batch_dict['length'][b_id], :batch_dict['length'][b_id]] = 0
                    mask[b_id, :batch_dict['length'][b_id], :batch_dict['length'][b_id]] = True
                    trg_mat[b_id, sample['pos1id'], sample['pos2id']] = 1
                    trg_mat[b_id, sample['pos2id'], sample['pos1id']] = 1

                batch_dict['pos1id'] = pos1id
                batch_dict['pos2id'] = pos2id
                batch_dict['trg_mat'] = trg_mat
                batch_dict['mask'] = mask

        return batch_dict
