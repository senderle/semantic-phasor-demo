"""
This script examines the contents of three directories in the current
working folder, `workset-ids`, `volumes`, and `derived`, automatically
generates lists of volume ids for volumes that still need to be downloaded,
compresses uncompressed volume, and generates derived document vectors.

In detail, it follows these steps:

1. For every file it sees in `workset-ids` matching a particular regex
(see below) it infers the existence of a dataset with the corresponding name.
2. For a given dataset it looks in the corresponding folder inside `volumes`
and collects the ids of all the volumes there, whether saved as folders
full of uncompressed text files or as compressed zip files.
3. It also looks in the corresponding folder inside
`derived`, and collects those ids.
3. It then saves a list of the ids not represented in either the `volumes`
or `derived` folders.
4. Next, it looks inside `volumes` and compresses any uncompressed folders.
5. Then it generates derived document vecors for any new volumes in
`volumes`.
6. It repeats 2-5 for each dataset found at step 1.
"""

import os
import re
import time
import shutil
import zipfile
import argparse
import multiprocessing

from pathlib import Path

from phasor import save_embedding_ffts, path_to_htid


class Dataset:
    def __init__(self, name, root, worksets, volumes, derived):
        self.name = name
        self.fft_path = Path(root) / derived / name / 'fft'
        self.srp_path = Path(root) / derived / name / 'srp_fft'
        self.vol_path = Path(root) / volumes / name / 'dir'
        self.zip_path = Path(root) / volumes / name / 'zip'
        self.id_file = Path(root) / worksets / (name + '-htids.txt')

        if not self.fft_path.exists():
            self.fft_path.mkdir(parents=True)
        if not self.srp_path.exists():
            self.srp_path.mkdir(parents=True)
        if not self.vol_path.exists():
            self.vol_path.mkdir(parents=True)
        if not self.zip_path.exists():
            self.zip_path.mkdir(parents=True)


def volids_in(path, extension=None):
    if extension is not None:
        return set(path_to_htid(v)
                   for v in path.iterdir()
                   if v.suffix == extension)
    else:
        return set(path_to_htid(v)
                   for v in path.iterdir()
                   if v.is_dir())


def find_volids(dataset):
    all_ids = set()

    # Volumes must be in both fft and srp folders, or...
    if dataset.fft_path.is_dir():
        all_ids.update(volids_in(dataset.fft_path, extension='.npz'))
    if dataset.srp_path.is_dir():
        all_ids &= volids_in(dataset.srp_path, extension='.npz')

    # ...must be represented by a zip file or folder in volume folder.
    if dataset.vol_path.is_dir():
        all_ids.update(volids_in(dataset.vol_path))
    if dataset.zip_path.is_dir():
        all_ids.update(volids_in(dataset.vol_path, extension='.zip'))
    return all_ids


def load_volids(id_file):
    with open(id_file) as ip:
        return set(line.strip() for line in ip)


def remaining_volids(dataset):
    return load_volids(dataset.id_file) - find_volids(dataset)


def write_remaining(dataset):
    remaining = remaining_volids(dataset)
    out_file = dataset.name + '-htids-remaining.txt'
    with open(out_file, 'w', encoding='utf-8') as op:
        for r in remaining:
            op.write(r)
            op.write('\n')


def valid_zip_file(f):
    try:
        with zipfile.ZipFile(f):
            pass
    except zipfile.BadZipFile:
        return False
    return True


def compress(root_vol):
    root, vol = root_vol
    curdir = os.getcwd()
    os.chdir(root)
    zip_vol = Path(vol)
    zip_vol = zip_vol.with_suffix(zip_vol.suffix + '.zip')
    if not zip_vol.exists():
        shutil.make_archive(vol, 'zip', '.', vol)
    else:
        print(f'Zip file {zip_vol} already exists.')

    if zip_vol.is_file() and valid_zip_file(zip_vol):
        shutil.rmtree(vol)
        os.chdir(curdir)
        return True
    else:
        if zip_vol.is_file():
            os.remove(zip_vol)
        print(f'Generating {zip_vol} failed.')
        os.chdir(curdir)
        return False


def time_since_modified(f):
    return time.time() - Path(f).stat().st_mtime


def compress_vols(dataset):
    path = dataset.vol_path
    if not path.is_dir():
        return

    curdir = os.getcwd()
    os.chdir(path)

    # Compress all non-hidden folders that have not been modified
    # in the last five minutes. Since the htrc download process
    # downloads multiple volumes in a single zip file, and then
    # unpacks the zip file into the destination folder, this
    # should allow more than enough time to be sure the volume
    # has been fully downloaded.
    vols = [f for f in Path().iterdir()
            if f.is_dir() and
            not f.stem.startswith('.') and
            not (dataset.zip_path / f.name).is_file() and
            time_since_modified(f) > 300]
    os.chdir(curdir)

    if vols:
        with multiprocessing.Pool(processes=10, maxtasksperchild=20) as pool:
            vol_args = [(path, f) for f in vols]
            list(pool.imap_unordered(compress, vol_args))

    for f in path.iterdir():
        if f.suffix == '.zip':
            f.rename(dataset.zip_path / f.name)


def save_embeddings(dataset):
    save_embedding_ffts(dataset.zip_path, dataset.fft_path, srp=False)
    save_embedding_ffts(dataset.zip_path, dataset.srp_path, srp=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='A processing workflow for generating vector '
        'representations from HTRC worksets.'
    )

    parser.add_argument(
        '-r',
        '--root',
        type=str,
        default='.',
        help='The root working directory. All other paths are relative to '
        'this directory. Defaults to the current working directory. '
    )
    parser.add_argument(
        '-w',
        '--worksets',
        type=str,
        default='worksets',
        help='The location of files contianing worksets. Worksets should '
        'be saved in this folder as plaintext files containing one HTID per '
        'line, using the following name scheme: [dataset-name]-htids.txt. '
        'Defaults to \'worksets\'.'
    )
    parser.add_argument(
        '-v',
        '--volumes',
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
        '--derived',
        type=str,
        default='derived',
        help='The location of all derived vector data. Data will be saved '
        'to subfolders using the following name scheme: '
        '[vector-type]/[dataset-name]/[htid].npz. Defaults to \'derived\'.'
    )

    return parser.parse_args()


def update_vectors(args):
    dataset_re = re.compile('(?P<name>^.+)-htids.txt$')
    datasets = os.listdir(Path(args.root) / args.worksets)
    datasets = [f for f in datasets if dataset_re.match(f)]
    datasets = [dataset_re.match(f)['name'] for f in datasets]
    datasets = [
        Dataset(name, args.root, args.worksets, args.volumes, args.derived)
        for name in datasets
    ]

    tasks = [
        (write_remaining, 'Collecting remaining volumes'),
        (compress_vols, 'Compressing volumes'),
        (save_embeddings, 'Saving embeddings'),
    ]

    for task, desc in tasks:
        for ds in datasets:
            print(f' ** {desc} for {ds.name}.')
            task(ds)


if __name__ == '__main__':
    args = parse_args()
    update_vectors(args)
