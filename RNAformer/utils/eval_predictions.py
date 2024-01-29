import numpy as np
import collections
import argparse
from collections import defaultdict
from tabulate import tabulate


def analyse_pairs(mat, sequence):
    pairs = mat2pairs(mat)
    wc = ['AU', 'UA', 'GC', 'CG']
    wobble = ['GU', 'UG']
    pair_types = collections.defaultdict(list)
    per_position_pairs = collections.defaultdict(list)
    closers = []
    multiplet_pairs = []
    for i, (p1, p2) in enumerate(pairs):
        pair_types['all'].append((p1, p2))
        closers.append(p2)
        per_position_pairs[p1].append((p1, p2))
        per_position_pairs[p2].append((p1, p2))
        p_type = sequence[p1] + sequence[p2]
        pair_types[''.join(sorted(p_type))].append((p1, p2))
        if p_type in wc:
            pair_types['wc'].append((p1, p2))
            pair_types['canonical'].append((p1, p2))
        elif p_type in wobble:
            pair_types['wobble'].append((p1, p2))
            pair_types['canonical'].append((p1, p2))
        else:
            pair_types['nc'].append((p1, p2))
        if i > 0 and closers[i - 1] < p2:
            pair_types['pk'].append((p1, p2))
        if len(per_position_pairs[p1]) > 1:
            multiplet_pairs += per_position_pairs[p1]
        if len(per_position_pairs[p2]) > 1:
            multiplet_pairs += per_position_pairs[p2]
    pair_types['multiplets'] = list(set(multiplet_pairs))
    pair_ratios = {k: (len(v) / len(sequence)) if not k == 'pk' else len(v) > 0 for k, v in pair_types.items()}
    return pair_ratios


def mat2pairs(matrix, symmetric=True):
    """
    Convert matrix representation of structure to list of pairs.
    """
    if symmetric:
        return list(set(tuple(sorted(pair)) for pair in np.argwhere(matrix == 1)))
    else:
        return list(tuple(pair) for pair in np.argwhere(matrix == 1))


################################################################################
# Metrics
################################################################################
def f1(tp, fp, tn, fn):
    f1_score = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return f1_score


def recall(tp, fp, tn, fn):
    recall = tp / (tp + fn + 1e-8)
    return recall


def specificity(tp, fp, tn, fn):
    specificity = tn / (tn + fp + 1e-8)
    return specificity


def precision(tp, fp, tn, fn):
    precision = tp / (tp + fp + 1e-8)
    return precision


def mcc(tp, fp, tn, fn):
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    return mcc


def non_correct(tp, fp, tn, fn):
    non_correct = (tp == 0).astype(int)
    return non_correct


def tp_from_matrices(pred, true):
    tp = np.logical_and(pred, true).sum()
    return tp


def tn_from_matrices(pred, true):
    tn = np.logical_and(np.logical_not(pred), np.logical_not(true)).sum()
    return tn


def get_fp(pred, tp):
    fp = pred.sum() - tp
    return fp


def get_fn(true, tp):
    fn = true.sum() - tp
    return fn


def solved_from_mat(pred, true):
    solved = np.all(np.equal(true, pred)).astype(int)
    return solved


def mat_from_pos(pos1, pos2, length):
    mat = np.zeros((length, length))
    for i, j in zip(pos1, pos2):
        mat[i, j] = 1
        mat[j, i] = 1
    return mat


def evaluate(preds, ratios=False):
    results = defaultdict(lambda: defaultdict(list))
    for p in preds:
        length = len(p['sequence'])
        true_mat = mat_from_pos(p['gt_pos1id'], p['gt_pos2id'], length=length)
        pred_mat = p['struct_mat']
        sequence = p['sequence']

        if ratios:
            results[p['dataset']]['pred_ratios'].append(analyse_pairs(pred_mat, sequence))
            results[p['dataset']]['true_ratios'].append(analyse_pairs(true_mat, sequence))

        results[p['dataset']]['length'] = length
        # results[p['dataset']]['Id'] = p['Id']
        results[p['dataset']]['solved'].append(solved_from_mat(pred_mat, true_mat))

        tp = tp_from_matrices(pred_mat, true_mat)
        fp = get_fp(pred_mat, tp)
        fn = get_fn(true_mat, tp)
        tn = tn_from_matrices(pred_mat, true_mat)

        results[p['dataset']]['f1-score'].append(f1(tp, fp, tn, fn))
        results[p['dataset']]['precision'].append(precision(tp, fp, tn, fn))
        results[p['dataset']]['recall'].append(recall(tp, fp, tn, fn))
        results[p['dataset']]['specificity'].append(specificity(tp, fp, tn, fn))
        results[p['dataset']]['precision'].append(precision(tp, fp, tn, fn))
        results[p['dataset']]['mcc'].append(mcc(tp, fp, tn, fn))

    for k, v in results.items():
        results[k] = {k2: np.mean(v2) for k2, v2 in v.items()}

    return results


def merge_dicts_with_lists(data):
    merged_dict = {}

    for d in data:
        for master_key, inner_dict in d.items():
            if master_key not in merged_dict:
                merged_dict[master_key] = {}

            for key, values in inner_dict.items():
                if key not in merged_dict[master_key]:
                    merged_dict[master_key][key] = []
                merged_dict[master_key][key].append(values)

    for master_key, inner_dict in merged_dict.items():
        for key, values in inner_dict.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            merged_dict[master_key][key] = {"mean": mean_value, "std": std_value}

    return merged_dict


def print_dict_tables(data):
    for table_key, table_dict in data.items():
        print(f"Table: {table_key}")
        headers = list(table_dict.keys())
        table_data = [list(table_dict.values())]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()


def print_nested_dict_tables(data):
    for table_key, table_dict in data.items():
        print(f"Table: {table_key}")
        table_data = []
        headers = list(next(iter(table_dict.values())).keys())

        for row_key, col_dict in table_dict.items():
            row_data = [row_key] + list(col_dict.values())
            table_data.append(row_data)

        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('predictions', metavar='P', type=str, nargs='*')

    args = parser.parse_args()
    predictions = args.predictions

    result_list = []
    for file_dir in predictions:
        print('### Read Pred data')
        preds = np.load(file_dir, allow_pickle=True)
        res = evaluate(preds)
        result_list.append(res)

    result = merge_dicts_with_lists(result_list)

    print_nested_dict_tables(result)
