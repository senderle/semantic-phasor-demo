# This script examines the contents of three directories in the current
# working folder, `workset-ids`, `volumes`, and `derived`, automatically
# generates lists of volume ids for volumes that still need to be downloaded,
# compresses uncompressed volume, and generates derived document vectors.

# In detail, it follows these steps:
#
# 1. For every file it sees in `workset-ids` matching a particular regex
# (see below) it infers the existence of a dataset with the corresponding name.
# 2. For a given dataset it looks in the corresponding folder inside `volumes`
# and collects the ids of all the volumes there, whether saved as folders
# full of uncompressed text files or as compressed zip files.
# 3. It also looks in the corresponding folder inside
# `derived`, and collects those ids.
# 3. It then saves a list of the ids not represented in either the `volumes`
# or `derived` folders.
# 4. Next, it looks inside `volumes` and compresses any uncompressed folders.
# 5. Then it generates derived document vecors for any new volumes in
# `volumes`.
# 6. It repeats 2-5 for each dataset found at step 1.

import os
import re
import shutil
import argparse
import multiprocessing

from pathlib import Path

from phasor import save_embedding_ffts, path_to_htid


class Dataset:
    def __init__(self, name, worksets, volumes, derived):
        self.name = name
        self.fft_path = Path(args.derived) / Path(name) / Path('fft')
        self.srp_path = Path(args.derived) / Path(name) / Path('srp_fft')
        self.vol_path = Path(args.volumes) / Path(name)
        self.id_file = Path(args.worksets) / Path(name + '-htids.txt')

        if not self.fft_path.exists():
            self.fft_path.mkdir(parents=True)
        if not self.srp_path.exists():
            self.srp_path.mkdir(parents=True)
        if not self.vol_path.exists():
            self.vol_path.mkdir(parents=True)


def volids_in(path, extension=None, exclude=None):
    if extension is not None:
        return set(path_to_htid(v)
                   for v in path.iterdir() if v.suffix == extension)
    elif exclude is not None:
        return set(path_to_htid(v)
                   for v in path.iterdir() if v.suffix != exclude)
    else:
        return set(path_to_htid(v) for v in path.iterdir())


def find_volids(dataset):
    vec_fft = dataset.fft_path
    vec_srp = dataset.srp_path
    vol = dataset.vol_path

    all_ids = set()
    # Volumes must be in both fft and srp folders, or...
    if vec_fft.is_dir():
        all_ids.update(volids_in(dataset.fft_path, extension='.npz'))
    if vec_srp.is_dir():
        all_ids &= volids_in(dataset.srp_path, extension='.npz')

    # ...must be represented by a zip file or folder in volume folder.
    if vol.is_dir():
        all_ids.update(volids_in(dataset.vol_path, extension='.zip'))
    if vol.is_dir():
        all_ids.update(volids_in(dataset.vol_path, exclude='.zip'))
    return all_ids


def load_volids(id_file):
    with open(id_file) as ip:
        return set(l.strip() for l in ip)


def remaining_volids(dataset):
    return load_volids(dataset.id_file) - find_volids(dataset)


def write_remaining(dataset):
    remaining = remaining_volids(dataset)
    out_file = dataset.name + '-htids-remaining.txt'
    with open(out_file, 'w', encoding='utf-8') as op:
        for r in remaining:
            op.write(r)
            op.write('\n')


def compress(root_vol):
    root, vol = root_vol
    curdir = os.getcwd()
    os.chdir(root)
    zip_vol = Path(vol)
    zip_vol = zip_vol.with_suffix(zip_vol.suffix + '.zip')
    if not zip_vol.exists():
        shutil.make_archive(vol, 'zip', '.', vol)

    if zip_vol.is_file():
        # Boldly delete the original folder.
        shutil.rmtree(vol)
        os.chdir(curdir)
        return True
    else:
        os.chdir(curdir)
        return False


def compress_vols(dataset):
    path = dataset.vol_path
    if not path.is_dir():
        return

    curdir = os.getcwd()
    os.chdir(path)
    vols = [f for f in Path().iterdir()
            if not f.suffix == '.zip' and
            not f.stem.startswith('.')]
    os.chdir(curdir)

    with multiprocessing.Pool(processes=10, maxtasksperchild=20) as pool:
        vol_args = [(path, f) for f in vols]
        assert all(pool.imap_unordered(compress, vol_args))


def save_embeddings(dataset):
    save_embedding_ffts(dataset.vol_path, dataset.fft_path, srp=False)
    save_embedding_ffts(dataset.vol_path, dataset.srp_path, srp=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='A processing workflow for generating vector '
        'representations from HTRC worksets.'
    )

    parser.add_argument(
        '-r',
        'root',
        type=str,
        default='.',
        help='The root working directory. Any paths that are not absolute '
        'will be interpreted as relative to this directory. Defaults to '
        'the current working directory.'
    )
    parser.add_argument(
        '-w',
        'worksets',
        type=str,
        default='worksets',
        help='The location of files contianing worksets. Worksets should '
        'be saved in this folder as plaintext files containing one HTID per '
        'line, using the following name scheme: [dataset-name]-htids.txt. '
        'Defaults to \'worksets\'.'
    )
    parser.add_argument(
        '-v',
        'volumes',
        type=str,
        default='volumes',
        help='The location of individual volume files. Volumes should be '
        'downloaded as folders full of individual pages using the following '
        'name scheme: [dataset-name]/[htid]. This script will automatically '
        'compress these folders, deleting them and replacing them with '
        '[htid].zip files. Defaults to \'volumes\'.'
    )
    parser.add_argument(
        '-d',
        'derived',
        type=str,
        default='derived',
        help='The location of all derived vector data. Data will be saved '
        'to subfolders using the following name scheme: '
        '[vector-type]/[dataset-name]/[htid].npz. Defaults to \'derived\'.'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset_re = re.compile('(?P<name>^.+)-htids.txt$')
    datasets = os.listdir(args.worksets)
    datasets = [f for f in datasets if dataset_re.match(f)]
    datasets = [dataset_re.match(f)['name'] for f in datasets]
    datasets = [Dataset(name, args.worksets, args.volumes, args.derived)
                for name in datasets]
    for ds in datasets:
        write_remaining(ds)
        compress_vols(ds)
        save_embeddings(ds)