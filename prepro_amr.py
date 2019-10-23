import glob
import itertools
import re
import subprocess
import sys
import tempfile

import tqdm

from amr_utils import AMRTree


def load_amr(data_path):
    amr_data = []
    for fn in tqdm.tqdm(glob.glob(data_path + '*.txt')):
        with open(fn, 'r') as data_file:
            amr_str = ''
            for line in data_file:
                line = line.strip()  # .lower()
                if line.startswith('#'):
                    if line.startswith('# ::id'):
                        amr_id = line.split()[2]
                    elif line.startswith('# ::snt'):
                        sentence = line.split(' ', 2)[-1]
                elif line != '':
                    amr_str += line + ' '
                elif amr_str != '':
                    if 'url' not in amr_str:
                        amr_data.append({
                            'raw_amr': amr_str,
                            'sent': sentence,
                            'id': amr_id
                        })
                    amr_str = ''
    return amr_data


def simplify_amr(amr_data):
    temp = tempfile.NamedTemporaryFile()
    with open(temp.name, 'w') as temp_file:
        for example in amr_data:
            temp_file.write('{}\n'.format(example['raw_amr']))
        cmd1 = 'cd amr_simplifier'
        cmd2 = './anonDeAnon_java.sh anonymizeAmrFull true {}'.format(temp.name)
        p = subprocess.Popen('{} && {}'.format(cmd1, cmd2), shell=True)
        p.wait()
        amr_strs = []
        with open(temp.name + '.anonymized', 'r') as amr_data:
            for line in amr_data:
                raw_amr = clean_quoted_substr(line.strip())
                amr_strs.append(raw_amr)
        temp_file.close()
    return amr_strs


def clean_quoted_substr(raw_amr):
    sub = {}
    for m in re.finditer(r'"[^"]+"', raw_amr):
        quoted_str = raw_amr[m.start():m.end()]
        unquoted_str = re.sub(r'\s', '_', quoted_str)
        unquoted_str = re.sub(r'[^\w]', '', unquoted_str)
        sub[quoted_str] = unquoted_str
    for k, v in sub.items():
        raw_amr = raw_amr.replace(k, v)
    return raw_amr


def simplify_amr_simple(raw_amr):
    raw_amr = clean_quoted_substr(raw_amr)
    sub = {}
    for m in re.finditer(r'\/\s\w+(-\d+)', raw_amr):
        for x in m.groups():
            sub[x] = ''
    for m in re.finditer(r'(\w+\s\/\s)\w+', raw_amr):
        for x in m.groups():
            sub[x] = ''
    for k, v in sub.items():
        raw_amr = raw_amr.replace(k, v)
    raw_amr = re.sub(r'\(', '( ', raw_amr)
    raw_amr = re.sub(r'\)', ' )', raw_amr)
    raw_amr = raw_amr.strip()[1:-2].strip()
    return raw_amr.lower()


def get_subword_amr(amr):
    amr = re.sub(r'\(@@ ', '(', amr)
    sub = {}
    for m in re.finditer(r'(:(\w)*\-?@@) (\w*\-?@@ )*', amr):
        bpe_str = amr[m.start():m.end()]
        sub[bpe_str] = re.sub(r'@@ ', '', bpe_str)
    for k in sorted(sub, key=lambda k: len(k), reverse=True):
        amr = amr.replace(k, sub[k])
    amr = re.sub(r'(:\w+)@@ ', r'\g<1>', amr)
    amr = re.sub(r'\((\w)@@ ', r'(\g<1>', amr)
    amr = re.sub(r'@@ ', r'@@ :bpe ', amr)
    amr = re.sub(r'@@ :bpe ([\)"])', r'\g<1>', amr)
    sub = {}
    for m in re.finditer(r'"(.*?)"', amr):
        quote_str = amr[m.start():m.end()]
        sub[quote_str] = quote_str.replace(':bpe ', '')
    for k in sorted(sub, key=lambda k: len(k), reverse=True):
        amr = amr.replace(k, sub[k])
    amr = re.sub(r':@@ :bpe', ':bpe', amr)
    return amr


def main(dataset_name, split_name, bpe_threshold=10000, experiment='structural_transformer'):
    """
    Simplify AMR graphs by removing variable tags, sense tangs and quotes.
    Resulting AMR graphs are single line, similar to anonymized AMR but without
    entities linking.
    """
    dataset_path = 'data/{}/data/'.format(dataset_name)
    if experiment == 'baseline':
        # step #1: linearize amr
        raw_split_path = dataset_path + 'amrs/split/{}/'.format(split_name)
        amr_data = load_amr(raw_split_path)

        simplified_amrs = simplify_amr(amr_data)
        for amr_str, example in zip(simplified_amrs, amr_data):
            example['amr'] = amr_str

        with open(dataset_path + '{}_source'.format(split_name), 'w') as out1, \
                open(dataset_path + '{}_target'.format(split_name), 'w') as out2:
            for ex in amr_data:
                amr = ex.get('amr', simplify_amr_simple(ex['raw_amr']))
                out1.write('{}\n'.format(amr))
                out2.write('{}\n'.format(ex['sent']))

        # step #2: bpe
        split_path = dataset_path + split_name
        if 'train' in split_name:
            cat_cmd = 'cat {0}_source {0}_target >> {0}'.format(split_path)
            bpe_cmd = 'subword-nmt learn-bpe -s {} < {} > {}/vocab.bpe' \
                .format(bpe_threshold, split_path, dataset_path)
            p = subprocess.Popen('{} && {}'.format(cat_cmd, bpe_cmd), shell=True)
            p.wait()
        for part in ['source', 'target']:
            bpe_cmd = 'subword-nmt apply-bpe -c {0}/vocab.bpe < {1}_{2} > {1}_{2}_bpe' \
                .format(dataset_path, split_path, part)
            p = subprocess.Popen(bpe_cmd, shell=True)
            p.wait()
    elif experiment == 'structural_transformer':
        # step 3: compute structural transformer input data
        with open(dataset_path + '{}_source_bpe'.format(split_name), 'r') as amr_data, \
                open(dataset_path + '{}_concept_bpe'.format(split_name), 'w') as out1, \
                open(dataset_path + '{}_all_8_path_bpe'.format(split_name), 'w') as out2:

            edges_fp = [open(dataset_path + '{}_{}hop_path_bpe'.format(split_name, i), 'w') for i in range(8)]

            for amr in tqdm.tqdm(amr_data):
                bpe_amr = get_subword_amr(amr)
                amr_tree = AMRTree(bpe_amr)
                concepts = amr_tree.concepts
                all_paths = amr_tree.paths
                concept_str = ' '.join(x.val for x in concepts if x.val != '<eos>')

                paths = []
                paths_str = []
                for p in concepts:
                    for q in concepts:
                        paths.append(all_paths[(p, q)] if p != q else ['None'])
                        paths_str.append(''.join(all_paths[(p, q)]) if p != q else 'None')
                path_str = ' '.join(paths_str)
                padded_paths = list(itertools.zip_longest(*paths, fillvalue='<blank>'))
                if (len(concept_str.split()) + 1) ** 2 != len(path_str.split()): continue
                for i in range(8):
                    if i < len(padded_paths):
                        edges_fp[i].write(' '.join(padded_paths[i]))
                    else:
                        edges_fp[i].write(' '.join(['<blank>'] * (len(concepts) ** 2)))
                    edges_fp[i].write('\n')
                assert (len(concept_str.split()) + 1) ** 2 == len(path_str.split())
                out1.write(concept_str)
                out1.write('\n')
                out2.write(path_str)
                out2.write('\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
