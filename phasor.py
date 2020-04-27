
import os
import textwrap
import multiprocessing

from difflib import SequenceMatcher
from collections import Counter

import spacy
import numpy
import pandas
import umap

from headless import load_pages
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree
from sklearn.random_projection import GaussianRandomProjection
from pyhash import city_64

from bokeh.plotting import figure, show
from bokeh.models import HoverTool, TapTool, OpenURL
from bokeh.palettes import magma

en_nlp = spacy.load('en_core_web_lg', disable=['tagger', 'parser', 'ner'])

N_BANDS = 10


def htid_url(htid):
    """
    Convert a given HTID to the corresponding volume page
    in HathiTrust.
    """
    htid = htid.replace('+', ':').replace('=', '/')
    return 'https://babel.hathitrust.org/cgi/pt?id={}'.format(htid)


def path_to_htid(path):
    """
    Take a Path object or string with an HTID in its name, and extract
    the HTID, undoing the substitutions performed for path conversion.
    This should work for any filename with a single extension. Filenames
    with multiple extensions ('.tar.gz') will not be handled correctly.
    """
    filename = os.path.split(path)[-1]

    # All HTIDs have a library identifier and a
    # record id, separated by a dot. We split them
    # here to eliminate any possible ambiguity between
    # this dot and additional optional dots signifying
    # extensions.
    lib_code, rec_id = filename.split('.', maxsplit=1)

    # If there is an extension, remove it. This assumes
    # that record ids, after being transformed into filenames,
    # will never contain dots. This appears to be true, but
    # needs more verification. (The alternative, assuming
    # that extensions do not contain multiple dots, will cause
    # errors for extensions like `.tar.gz`.)
    rec_id = rec_id.split('.', 1)[0]

    # Finally, we undo the following substitutions
    # applied to record ids to avoid filename issues.
    rec_id = rec_id.replace('+', ':').replace('=', '/').replace(',', '.')

    return f'{lib_code}.{rec_id}'


# This older version didn't always handle extensions correctly.
def _path_to_htid_old(path):
    htid = os.path.split(path)[-1]
    htid = os.path.splitext(htid)[0]
    return htid.replace('+', ':').replace('=', '/').replace(',', '.')


def htid_to_filename(htid):
    """
    Convert a given HTID into a filename, performing
    substitutions on characters that can cause problems
    in filenames. Note the removal of dots from the
    rec_id but not the dot between the lib_code and
    the rec_id. This makes it easier to distinguish
    between extensions and dots that are part of the
    original HTID.
    """

    path = htid.replace(':', '+').replace('/', '=')
    if '.' in path:
        lib_code, rec_id = path.split('.', maxsplit=1)
        path = '.'.join((lib_code, rec_id.replace('.', ',')))
    return path


def test_htid_conversion(path):
    htid = path_to_htid(path)
    ext = os.path.splitext(path)[-1]
    new_path = os.path.split(path)[0]
    new_path = os.path.join(new_path, htid_to_filename(htid))
    new_path += ext
    if path != new_path:
        print(path, new_path)
        assert path == new_path


def volume_paths(path):
    """List all zip files and subfolders in the given folder."""
    files = (os.path.join(path, f) for f in sorted(os.listdir(path)))
    return [f for f in files if os.path.isdir(f) or f.endswith('.zip')]


def numpy_paths(path):
    """List all numpy files in the given folder."""
    files = (os.path.join(path, f) for f in sorted(os.listdir(path)))
    return [f for f in files if f.endswith('.npz')]


# There are surely faster ways to tokenize, look up word vectors,
# and build text vector arrays. But the quality of spacy tokenization
# is higher, and the extra overhead of slow Python loops only winds
# up adding 30-50% more time. The combination of flexibility and
# high quality tokenization is worth the extra time.
class VectorTable():
    def __init__(self, spacy_model=None, ndims=300):
        self._table = {}
        self._doc_count = Counter()

        # If a spacy model with vectors is not available,
        # just use a random projection. This works surprisingly
        # well, but gives vectors that are harder to interpret.
        # Even when a spacy model is available, we still fall
        # back on random vectors for out-of-vocabulary terms.
        if spacy_model is None:
            self._vec_table = self._srp_vec_table
        else:
            self._spacy_strings = spacy_model.vocab.strings
            self._spacy_vectors = spacy_model.vocab.vectors
            self._vec_table = self._sp_vec_table

        self.ndims = ndims

    def __getitem__(self, keys):
        if isinstance(keys, str):
            keys = [keys]
        elif not isinstance(keys, list):
            keys = list(keys)

        # Create a one-use lookup table for all keys.
        # This represents a decent trade-off between
        # speed, memory usage, and flexibility.
        table = self._vec_table(keys)

        # To save memory and a bit of speed, the table
        # only contains indices underlying vector tables.
        # The data is copied only once, directly into the
        # new array.
        result = numpy.empty((len(keys), self.ndims))
        for i, k in enumerate(keys):
            sub_index, sub_table = table[k]
            result[i] = sub_table[sub_index]
        return result

    def _sp_vec_table(self, keys):
        keys = set(keys)
        keys_ids = [(k, self._spacy_strings[k]) for k in keys
                    if k in self._spacy_strings]
        table = {k: (i, self._spacy_vectors) for k, i in keys_ids
                 if i in self._spacy_vectors}
        missing = keys - table.keys()
        table.update(self._srp_vec_table(missing))
        return table

    def _srp_vec_table(self, keys):
        keys = list(set(keys))
        srp = self.srp_matrix(keys, self.ndims)
        return {k: (i, srp) for i, k in enumerate(keys)}

    # This is a quick-and-dirty implementation of what Ben Schmidt calls
    # "Stable Random Projection." (Errors are mine, not his!)
    @classmethod
    def srp_matrix(cls, words, ndims, _hashfunc=city_64(0)):
        multiplier = (ndims - 1) // 64 + 1
        hashes = [
            list(map(_hashfunc, ['{}_{}'.format(w, i)
                                 for i in range(multiplier)]))
            for w in words
        ]

        # Given a `multipier` value of 5, `hashes` is really a V x 5
        # array of 64-bit integers, where V is the vocabulary size...

        hash_arr = numpy.array(hashes, dtype=numpy.uint64)

        # ...but we could also think of it as a V x 40 array of bytes...

        hash_arr = hash_arr.view(dtype=numpy.uint8)

        # ...or even as an array of bits, where every word is represented
        # by 320 bits...

        hash_arr = numpy.unpackbits(hash_arr.ravel()).reshape(-1,
                                                              64 * multiplier)

        # ...or as an array of floating point values, all equal to either
        # 1.0 or 0.0, and truncated to give a final array of V x ndims.

        return (hash_arr.astype(numpy.float64) * 2 - 1)[:, :ndims]


def load_one_sp_embedding(volume_path, nlp=en_nlp,
                          vec=VectorTable(spacy_model=en_nlp)):
    """Parse the text of one volume and extract word vectors."""
    sp_text = nlp.pipe(load_pages(volume_path))
    words = [tok.lower_ for doc in sp_text for tok in doc if not tok.is_space]
    return vec[words]


def load_one_srp_embedding(volume_path, nlp=en_nlp, vec=VectorTable()):
    """Parse the text of one volume and extract word vectors."""
    sp_text = nlp.pipe(load_pages(volume_path))
    return vec[[tok.lower_ for doc in sp_text for tok in doc
                if not tok.is_space and not tok.is_punct]]


def piecewise_avg(vec, n_groups):
    """Divide a vector into pieces and return the average for each."""
    size = len(vec) / n_groups
    ends = []
    for i in range(1, n_groups + 1):
        ends.append(int(size * i))
    ends[-1] = len(vec)

    sums = []
    start = 0
    for end in ends:
        sums.append(vec[start:end].sum() / (end - start))
        start = end

    return numpy.array(sums)


def embedding_fft(sp_embedding, n_bands=N_BANDS):
    """
    Perform a Fourier transform on all the dimensions of an
    array of word embeddings extracted from a document.
    `sp_embedding` is assumed to be an array with a row
    for each document, and a column for each dimension of
    the underlying word embedding vector model.
    """
    fft_cols = []
    n_groups = 1
    while n_groups < n_bands * 10:
        n_groups *= 2

    for col in range(sp_embedding.shape[1]):
        vec = sp_embedding[:, col]
        vec = piecewise_avg(vec, n_groups)
        fft = numpy.fft.rfft(vec)
        fft_cols.append(fft[:n_bands])

    return numpy.array(fft_cols)


def save_compact_fft(filename, fft):
    r = fft.real.astype(numpy.float16)
    i = fft.imag[:, 1:].astype(numpy.float16)
    assert (fft.imag[:, 0] == 0).all()
    numpy.savez_compressed(filename, real=r, imag=i)


def load_compact_fft(filename, ndims=300):
    with numpy.load(filename) as fft_data:
        r = fft_data['real']
        i = fft_data['imag']
    fft = numpy.empty(r.shape, dtype=numpy.complex64)
    fft[:] = r
    fft[:, 1:] += i
    return fft


def flatten_fft(emb_fft, start=0, end=None, drop_zero_imag=False):
    """Reshape an fft array into a single vector."""
    complex_vec = numpy.array(emb_fft)[:, start:end].reshape(-1)

    # Check to see if all imaginary values are zero,
    # and if so only include real
    if drop_zero_imag and complex_vec.imag.ravel().sum() == 0:
        return complex_vec.real
    else:
        return numpy.array([x
                            for r_i in zip(complex_vec.real, complex_vec.imag)
                            for x in r_i])


def unflatten_vec(doc_vector, ndims=300):
    """Turn a document vector back into an fft array."""

    # This hard-codes values that should be parameters.
    array = doc_vector.reshape(ndims, -1)
    real = array[:, ::2]
    imag = array[:, 1::2]
    return real + imag * 1j


def slice_vec_bands(doc_vectors, start=0, end=None):
    return numpy.array([
        flatten_fft(unflatten_vec(dv), start, end, drop_zero_imag=True)
        for dv in doc_vectors
    ])


def test_fft_reshape(volume_path, srp=False):
    """A test of vector-array conversion routines."""
    assert _test_fft_reshape_one(volume_path, srp)


def _test_fft_reshape_one(folder, srp):
    if srp:
        emb = load_one_srp_embedding(folder)
    else:
        emb = load_one_sp_embedding(folder)

    fft_orig = embedding_fft(emb)

    fft_complex = unflatten_vec(flatten_fft(fft_orig))
    return (fft_orig == fft_complex).all()


def _multiprocessing_save_sp(in_out_path):
    vp, np = in_out_path
    if not os.path.exists(np + '.npz'):
        save_compact_fft(
            np,
            embedding_fft(load_one_sp_embedding(vp))
        )
        return np


def _multiprocessing_save_srp(in_out_path):
    vp, np = in_out_path
    if not os.path.exists(np + '.npz'):
        save_compact_fft(
            np,
            embedding_fft(load_one_srp_embedding(vp))
        )
        return np


def save_embedding_ffts(source_path, dest_path=None, srp=False):
    dest_path = source_path if dest_path is None else dest_path
    vol_paths = volume_paths(source_path)
    new_paths = [os.path.split(vp)[-1] for vp in vol_paths]
    new_paths = [vp if not vp.endswith('.zip') else vp[:-4]
                 for vp in new_paths]
    new_paths = [os.path.join(dest_path, vp) for vp in new_paths]

    # This step, though redundant, speeds up the multiprocessing stage
    # dramatically sometimes, because it skips files that have already
    # been processed without incurring any multiprocessing overhead.
    filter = [(vp, np) for vp, np in zip(vol_paths, new_paths)
              if not os.path.exists(np + '.npz')]
    if not filter:
        return []

    vol_paths, new_paths = zip(*filter)
    with multiprocessing.Pool(processes=8, maxtasksperchild=20) as pool:
        if srp:
            res = pool.imap_unordered(
                _multiprocessing_save_srp,
                zip(vol_paths, new_paths)
            )
        else:
            res = pool.imap_unordered(
                _multiprocessing_save_sp,
                zip(vol_paths, new_paths)
            )
        return list(res)


def load_embedding_fft_array(path, start=0, end=None,
                             reload=False, htid_test=None, _cache={}):
    if (reload or not _cache or _cache['start'] != start or
            _cache['end'] != end or htid_test is not None):
        if htid_test is not None:
            assert ([path_to_htid(p) for p in numpy_paths(path)] ==
                    list(htid_test))
        _cache['start'] = start
        _cache['end'] = end
        _cache['data'] = numpy.array(
                [flatten_fft(load_compact_fft(f), start, end)
                 for f in numpy_paths(path)])
    return _cache['data']


def load_metadata(metadata_path, fft_path, csv_delim='\t', htid_col='htid'):
    ids = [path_to_htid(p)
           for p in numpy_paths(fft_path)]
    metadata = (pandas
                .read_csv(metadata_path, delimiter=csv_delim)
                .drop_duplicates(htid_col)
                .set_index(htid_col))
    return metadata.reindex(ids, fill_value='[metadata missing]')


def load_fft_metadata(fft_path, metadata_path, start=0, end=None, reload=False,
                      csv_delim='\t', htid_col='htid'):
    metadata = load_metadata(metadata_path, fft_path, csv_delim, htid_col)
    fft_arr = load_embedding_fft_array(fft_path, start, end,
                                       reload, metadata.index)
    return fft_arr, metadata


class Deduplicator:
    def __init__(self, data, random=False, **umap_kwargs):
        if isinstance(data, Deduplicator):
            self.n_trees = data.n_trees
            self.n_points = data.n_points
            self.data = list(data.data)
            self.data_umap = list(data.data_umap)
            self.data_kd = list(data.data_kd)
        else:
            if random:
                umap_kwargs = {k: umap_kwargs[k] for k in
                               ['n_components']
                               if k in umap_kwargs}
                data_norm = data - data.mean(axis=0)
                data_norm = data_norm / data_norm.std(axis=0)
                data = data_norm
                model = GaussianRandomProjection
            else:
                model = umap.UMAP

            self.n_trees = 1
            self.n_points = len(data)
            self.data = [data]
            self.data_umap = [model(**umap_kwargs).fit_transform(d)
                              for d in self.data]
            self.data_kd = [cKDTree(d) for d in self.data_umap]

    def merge(self, other):
        if other.n_points != self.n_points:
            raise ValueError(
                'Deduplicator size mismatch: '
                f'{self.n_points} != {other.n_points}'
            )
        self.data.extend(other.data)
        self.data_umap.extend(other.data_umap)
        self.data_kd.extend(other.data_kd)
        self.n_trees += other.n_trees

    def _get_pairs_simple(self, distance):
        pairs = self.data_kd[0].query_pairs(distance)
        pairs = set(frozenset(p) for p in pairs)
        for kd in self.data_kd[1:]:
            newpairs = set(frozenset(p) for p in kd.query_pairs(distance)
                           if frozenset(p) in pairs)
            pairs = newpairs
        return pairs

    def _get_pairs_onebatch(self, distance, batch):
        data_umap = self.data_umap[0]
        data_kd = self.data_kd[0]
        pairs = data_kd.query_ball_point(data_umap[batch], distance)
        pairs = set(frozenset((i, m)) for
                    i, matches in zip(batch, pairs)
                    for m in matches
                    if m != i)
        for t in range(1, self.n_trees):
            data_umap = self.data_umap[t]
            data_kd = self.data_kd[t]
            newpairs = data_kd.query_ball_point(data_umap[batch], distance)
            newpairs = set(frozenset((i, m)) for
                           i, matches in zip(batch, newpairs)
                           for m in matches
                           if m != i)
            pairs = set(p for p in newpairs if p in pairs)

        return pairs

    def _get_pairs_batched(self, distance, batchsize=1000):
        indices = range(0, self.n_points)
        batches = [list(indices[i: i + batchsize])
                   for i in range(0, self.n_points, batchsize)]
        pairs = self._get_pairs_onebatch(distance, batches[0])
        for b in batches[1:]:
            pairs.update(self._get_pairs_onebatch(distance, b))
        return pairs

    def __call__(self, distance):
        # return self._get_pairs_simple(distance)
        return self._get_pairs_batched(distance)


def deduplicator_balltree(data, **umap_kwargs):
    data_umap = umap.UMAP(**umap_kwargs).fit_transform(data)
    data_kd = BallTree(data_umap)

    def deduplicate(distance):
        matches = data_kd.query_radius(data_umap, distance)
        pairs = [(i, m) for i, point in enumerate(data_umap)
                 for m in matches[i]]
        return set(frozenset(p) for p in pairs)
    return deduplicate


def umap_concat(data, **umap_kwargs):
    data_tiles = []
    for i in range(5):
        data_i = slice_vec_bands(data, start=i, end=i + 1)
        data_tiles.append(umap.UMAP(**umap_kwargs).fit_transform(data_i))
    data_concat = numpy.empty((data_tiles[0].shape[0], sum(dt.shape[1]
                              for dt in data_tiles)))
    start_col = 0
    for dt in data_tiles:
        end_col = start_col + dt.shape[1]
        data[:, start_col:end_col] = dt
        start_col = end_col

    return data_concat


def string_similarity(a, b):
    return SequenceMatcher(a=a, b=b).ratio()


def show_dataset(folder, n=10):
    volumes = volume_paths(folder)
    for v in volumes:
        print(load_pages(v)[0][0:500])


# def show_umap(data, n_neighbors=20, min_dist=0.001, metric='euclidean'):
#     um = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
#     vis = um.fit_transform(data)
#     plt.gca().axis('equal')
#     plt.scatter(vis[:, 0],
#                 vis[:, 1],
#                 c=[i / len(vis) for i in range(len(vis))],
#                 cmap='plasma')
#     plt.show()


def umap_color(metadata, color_field, n_colors, dtype=None, palette=magma):
    palette = palette(n_colors)
    if color_field is None:
        metadata_cf = metadata
    else:
        metadata_cf = metadata[color_field]

    if dtype is None:
        dtype = type(metadata_cf[0])
    field = [f if isinstance(f, dtype) else dtype()
             for f in metadata_cf]
    n_colors = n_colors if len(set(field)) >= n_colors else len(set(field))
    field_rank = {f: i / len(field) for i, f in enumerate(sorted(field))}
    return [palette[int(field_rank[f] * n_colors)] for f in field]


def show_umap_bokeh(data, metadata, color_field=None,
                    n_neighbors=10, min_dist=0.001, metric='euclidean'):
    if color_field is None:
        dims = 3
    else:
        dims = 2
    um = umap.UMAP(n_neighbors=n_neighbors, n_components=dims,
                   min_dist=min_dist, metric=metric)
    vis = um.fit_transform(data)

    if color_field is None:
        color = umap_color(vis[:, 2], None, 20)
        color_field = "Third UMAP dimension"
    else:
        color = umap_color(metadata, color_field, 20, dtype=int)
    scatter_data = pandas.DataFrame({
        'umap_1': vis[:, 0],
        'umap_2': vis[:, 1],
        'color': color,
        'htid': list(metadata.index),
        'title': ['<br>'.join(textwrap.wrap(t))
                  for t in metadata['title']],
        'author': list(metadata['author']),
        'pub_date': list(metadata['pub_date'])
    })

    plot_figure = figure(
        title=('UMAP Projection of Phasor vectors for ~1000 random '
               'HathiTrust volumes (colored by {})'.format(color_field)),
        plot_width=800,
        plot_height=800,
        tools=('pan, wheel_zoom, tap, reset')
    )

    plot_figure.add_tools(HoverTool(
        tooltips=(
            "<div><span style='font-size: 10px'>@htid{safe}</span></div>"
            "<div><span style='font-size: 10px'>@author{safe}</span></div>"
            "<div><span style='font-size: 10px'>@title{safe}</span></div>"
            "<div><span style='font-size: 10px'>@pub_date{safe}</span></div>"
        )
    ))

    plot_figure.circle(
        'umap_1',
        'umap_2',
        color='color',
        source=scatter_data,
    )

    tap = plot_figure.select(type=TapTool)
    tap.callback = OpenURL(
        url='https://babel.hathitrust.org/cgi/pt?id=@htid{safe}'
    )
    show(plot_figure)
