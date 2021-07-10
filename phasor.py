
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
# from sklearn.preprocessing import StandardScaler
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


def htid_to_filename(htid):
    """
    Convert a given HTID into a valid filename, performing substitutions
    on characters that can cause problems in various filesystems and URI
    schemes. This is a surprisingly subtle process because it needs to
    be reversible -- that is, we need to generate a filename that can be
    converted back into an HTID. But depending on how the substitutions
    are performed, they may cause collisions by assigning two distinct
    volumes the same filename. (This probably doesn't become a problem
    immediately. But later, when file extensions are added in ways that
    cause ambiguity, it does.)

    Fortunately most substitutions are trivial, and we begin with those.
    Colons are replaced by plus signs, and slashes are replaced by equal
    signs. So for example, the sequence `://` will become `+==`.

    The non-trivial substitutions involve periods ("dots"). These can
    appear in both HTIDs and filenames, and can have different meanings
    depending on where they occur. The upshot is that we need to perform
    our substitutions in a particular order.

    An HTID consists of 1) a library code, identifying the library a
    volume came from, and 2) a record id, identifying the volume itself.
    These are joined by a dot. And since library codes cannot contain
    dots, the first dot we encounter is guaranteed to be a separator
    between the library code and the record id. So we start there,
    splitting at the dot to divide the HTID into two parts.

    We name these `lib_code` and `rec_id`. The `lib_code` needs no
    further processing. The `rec_id` may contain additional dots,
    depending on each library's own record id scheme. These we replace
    with commas.

    Finally, we rejoin the code and id with a dot.

    This process is reversed by `path_to_htid`.
    """

    path = htid.replace(':', '+').replace('/', '=')
    if '.' in path:
        lib_code, rec_id = path.split('.', maxsplit=1)
        path = '.'.join((lib_code, rec_id.replace('.', ',')))
    return path


def path_to_htid(path):
    """
    Take a Path object or string with an HTID in its name, and extract
    the HTID, undoing the substitutions performed for path conversion.
    This should work for any filename with a single extension. Filenames
    with multiple extensions ('.tar.gz') may not be handled correctly.
    More corner cases need testing.

    For a more extensive discussion of the substitutions being reversed
    here, and the rationale for them, see the documentation for
    `htid_to_filename`.

    The first dot is guaranteed to be a separator between a library code
    and a record id. The next dot we encounter is in the record id, and
    is guaranteed to be a separator between the record id itself and the
    first of any number of filename extensions (if present).

    We have these guarantees because of the care taken in
    `htid_to_filename` to create a fully reversible substitution process.
    """

    # Drop the leading path.
    filename = os.path.split(path)[-1]

    # Split the library code and record id.
    lib_code, rec_id = filename.split('.', maxsplit=1)

    # Drop any file extensions.
    rec_id = rec_id.split('.', 1)[0]

    # Undo the remaining character substitutions.
    rec_id = rec_id.replace('+', ':').replace('=', '/').replace(',', '.')

    # Reunite the library code and record id.
    return f'{lib_code}.{rec_id}'


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


def numpy_paths(path, n=None):
    """List all numpy files in the given folder."""
    files = (os.path.join(path, f) for f in sorted(os.listdir(path)))
    files = [f for f in files if f.endswith('.npz')]
    return files if n is None else files[:n]


# There are surely faster ways to tokenize, look up word vectors,
# and build text vector arrays. But the quality of spacy tokenization
# is higher, and the extra overhead of slow Python loops only winds
# up adding 30-50% more time. The combination of flexibility and
# high quality tokenization is worth the extra time.
class VectorTable():
    """
    A lookup table for getting vectors from tokens. Uses either a
    `spacy` model with random vectors for out-of-vocbulary terms,
    or a purely random projection.

    Probably not very efficient, but the lack of fast hash tables
    with c-speed vectorized lookup stymies progress.
    """
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

        # To save memory and speed things up a little, the
        # table only contains indices into the underlying
        # vector tables. The data is copied only once,
        # directly into the new array.
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

    @classmethod
    def srp_matrix(cls, words, ndims, _hashfunc=city_64(0)):
        """
        Geenrate a random matrix using a hash function. It will have
        `ndims` columns, and a row for every word, for a total of
        `len(words)` rows. The values will be determined by a hash
        function. To create a row, we hash the corresponding word
        with a tag appended, change the tag and hash it a second time,
        and continue until we have more than `ndims` random bits
        available. The hash values are then unpacked into a matrix
        of bits, and the 0s are changed to -1s (so that the matrix
        approximately preserves length when used for projection).

        Because we use a hash function pre-seeded with a fixed value,
        a given word will always generate the same row of numbers.

        This is a hasty implementation of Ben Schmidt's Stable Random
        Projection (https://culturalanalytics.org/article/11033).
        Errors are mine alone.
        """
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
    """Parse the text of one volume and extract SRP word vectors."""
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
    """
    Save the output of a series of fast Fourier transforms by
    unpacking the real and imaginary parts into separate arrays, and
    saving them as an `.npz` file. The result is a file with two
    2-d arrays, corresponding to a single HathiTrust volume. The
    rows of the arrays are semantic dimensions, and the columns of
    the arrays are frequecy bands.

    We currently save the data as `float16` values, which saves some
    disk space, but does throw out some information. See the
    documetation for `load_compact_fft` for further discussion of
    this issue.
    """
    r = fft.real.astype(numpy.float16)
    i = fft.imag[:, 1:].astype(numpy.float16)

    # The first imaginary band is always be zero for a real signal.
    # So we don't bother saving it -- but we check to make sure we
    # aren't making some kind of mistake and throwing out data.
    assert (fft.imag[:, 0] == 0).all()
    numpy.savez_compressed(filename, real=r, imag=i)


def load_compact_fft(filename, ndims=300):
    """
    Load the data from a fast Fourier transform, as produced by the
    `save_compact_fft` function. For type consistency, we load these
    as `complex128` values, even though they were saved as two
    `float16` values. This wastes some memory, and in the future,
    it might make sense to save the FFT outputs as `float64` values.

    However, it could be that the extra information doesn't affect
    the behavior of these vectors at all, and provide no benefit.
    This should be tested at some point.
    """
    with numpy.load(filename) as fft_data:
        r = fft_data['real']
        i = fft_data['imag']
    fft = numpy.empty(r.shape, dtype=numpy.complex128)
    fft[:] = r
    fft[:, 1:] += i * 1j
    return fft


def load_compact_fft_as_signature(
        filename, n_components=10, n_anchors=3, _cache={}):
    """
    Load the data from a fast Fourier transform, as produced by the
    `save_compact_fft` function. Initialy, this behaves identically
    to `load_compact_fft`. However, instead of returning the full
    data from the FFT transform, it returns a much smaller signature
    vector for each frequency band, using the PhasorSignature class.

    This could probably be sped up.
    """
    with numpy.load(filename) as fft_data:
        r = fft_data['real']
        i = fft_data['imag']
    fft = numpy.empty(r.shape, dtype=numpy.complex128)
    fft[:] = r
    fft[:, 1:] += i * 1j

    cache_valid = (_cache and
                   _cache['n_components'] == n_components and
                   _cache['n_anchors'] == n_anchors and
                   'psig' in _cache)
    if not cache_valid:
        _cache['psig'] = PhasorSignature(n_components=n_components,
                                         n_anchors=n_anchors)
        _cache['n_components'] = n_components
        _cache['n_anchors'] = n_anchors

    psig = _cache['psig']

    rows, cols = fft.shape
    signature_bands = numpy.empty((n_components // 2, cols),
                                  dtype=numpy.complex128)
    for band in range(cols):
        sig = psig.signature(fft[:, band])
        signature_bands[:, band] = sig[::2]
        signature_bands[:, band] += sig[1::2] * 1j

    return signature_bands


def flatten_fft(emb_fft, start=0, end=None, drop_imag=False):
    """
    Transform a 2-d array of complex numbers into a vector of
    ordinary floats by unpacking the real and imaginary components
    and flattening the array.

    `emb_fft` is a two-dimensional, complex-valued array. Each row
    is an array of complex amplitudes for a given semantic dimension.
    Each column corresponds to a frequency band, starting with
    the frequency 0 band, a scalar offset around which the signal
    oscillates; the frequency 1 band, which completes one cycle
    over the duration of the signal; the frequency 2 band, which
    completes two cycles over the duration of the signal; and so on.

    `start` and `end` determine how many frequency bands to pull
    intp the vector.

    `drop_imag` drops the imaginary values instead of unpacking
    them when pulling only the first frequency band. The first
    band is a scalar offset around which the signal oscillates.
    For real signals, the imaginary offset is always zero, and
    can be ignored.
    """

    complex_vec = numpy.array(emb_fft)[:, start:end].reshape(-1)

    if drop_imag and start == 0 and end == 1:
        return complex_vec.real
    else:
        return numpy.array([x
                            for r_i in zip(complex_vec.real, complex_vec.imag)
                            for x in r_i])


def unflatten_vec(doc_vector, ndims=300):
    """
    Turn a document vector back into an fft array. Roughly speaking,
    this is the inverse of `flatten_fft`. Note that we need to know
    the number of dimensions to get the right shape for the array.

    The resulting array is a 2-d array of complex numbers, as expected
    by `flatten_fft`.
    """

    array = doc_vector.reshape(ndims, -1)
    real = array[:, ::2]
    imag = array[:, 1::2]
    return real + imag * 1j


def slice_vec_bands(doc_vectors, start=0, end=None,
                    ndims=300, drop_imag=False):
    """
    This takes a whole batch of document vectors and slices out a
    subset of their frequency bands. It does so by turning them back
    into FFT arrays, pulling out the specific bands we want, and
    re-flattening them. When start is `0`, end is `None`, and
    drop_imag=False, this amounts to the identity function on doc
    vectors.
    """
    return numpy.array([
        flatten_fft(unflatten_vec(dv, ndims=ndims), start, end, drop_imag)
        for dv in doc_vectors
    ])


def test_fft_reshape(volume_path, srp=False):
    """A test of vector-array conversion routines."""
    assert _test_fft_reshape_one(volume_path, srp)


def _test_fft_reshape_one(folder, srp):
    """A helper function for testing vector-array conversion"""
    if srp:
        emb = load_one_srp_embedding(folder)
    else:
        emb = load_one_sp_embedding(folder)

    fft_orig = embedding_fft(emb)

    fft_complex = unflatten_vec(flatten_fft(fft_orig))
    return (fft_orig == fft_complex).all()


def _multiprocessing_save_sp(in_out_path):
    """
    A function that saves spacy embeddings and is suitable for use
    with the multiprocessing library because it is globally namespaced
    and accepts just one argument as input.
    """
    vp, np = in_out_path
    if not os.path.exists(np + '.npz'):
        save_compact_fft(
            np,
            embedding_fft(load_one_sp_embedding(vp))
        )
        return np


def _multiprocessing_save_srp(in_out_path):
    """
    A function that loads SRP embeddings and is suitable for use
    with the multiprocessing library because it is globally namespaced
    and accepts just one argument as input.
    """
    vp, np = in_out_path
    if not os.path.exists(np + '.npz'):
        save_compact_fft(
            np,
            embedding_fft(load_one_srp_embedding(vp))
        )
        return np


def save_embedding_ffts(source_path, dest_path=None, srp=False):
    """
    A function that saves embedding FFT arrays for each volume, using
    multiprocessing to speed things up.
    """
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


def load_embedding_fft_array(path, start=0, end=None, reload=False,
                             htid_test=False, n_docs=None, _cache={}):
    """
    Quickly load embedding fft data, using a cache if nothing about
    the input arguments has changed.
    """

    reload = (reload or not _cache or _cache['start'] != start or
              _cache['end'] != end or htid_test or
              _cache['n_docs'] != n_docs)
    if reload:
        print('Reloading FFT data. This could take a while.')
        if htid_test:
            assert ([path_to_htid(p) for p in numpy_paths(path, n=n_docs)] ==
                    list(htid_test))
        _cache['start'] = start
        _cache['end'] = end
        _cache['n_docs'] = n_docs
        _cache['data'] = numpy.array(
                [flatten_fft(load_compact_fft(f), start, end)
                 for f in numpy_paths(path, n=n_docs)])
    return _cache['data']


def load_embedding_fft_signature_array(
        path, start=0, end=None, reload=False,
        htid_test=False, n_docs=None, _cache={}):
    """
    Quickly load embedding fft signatures, using a cache if nothing about
    the input arguments has changed.
    """
    reload = (reload or not _cache or _cache['start'] != start or
              _cache['end'] != end or htid_test or
              _cache['n_docs'] != n_docs)
    if reload:
        print('Reloading FFT signature data. This could take a while.')
        if htid_test:
            assert ([path_to_htid(p) for p in numpy_paths(path, n=n_docs)] ==
                    list(htid_test))
        _cache['start'] = start
        _cache['end'] = end
        _cache['n_docs'] = n_docs
        _cache['data'] = numpy.array(
                [flatten_fft(load_compact_fft_as_signature(f), start, end)
                 for f in numpy_paths(path, n=n_docs)])
    return _cache['data']


def load_metadata(metadata_path, fft_path, csv_delim='\t', htid_col='htid',
                  n_docs=None):
    """
    Load the metadata for the volumes loaded by
    `load_embedding_fft_array`.
    """
    ids = [path_to_htid(p)
           for p in numpy_paths(fft_path, n=n_docs)]
    metadata = (pandas
                .read_csv(metadata_path, delimiter=csv_delim)
                .drop_duplicates(htid_col)
                .set_index(htid_col))
    return metadata.reindex(ids, fill_value='[metadata missing]')


def load_fft_metadata(fft_path, metadata_path, start=0, end=None,
                      reload=False, csv_delim='\t', htid_col='htid',
                      htid_test=False, n_docs=None):
    """
    Load an array of embedding fft data and the corresponding metadata,
    returning both together.
    """
    metadata = load_metadata(metadata_path, fft_path,
                             csv_delim, htid_col, n_docs=n_docs)
    htid_test = metadata.index if htid_test else False
    fft_arr = load_embedding_fft_array(fft_path, start, end, reload,
                                       htid_test=htid_test, n_docs=n_docs)

    return fft_arr, metadata


def load_fft_signature_metadata(
        fft_path, metadata_path, start=0, end=None, reload=False,
        csv_delim='\t', htid_col='htid', htid_test=False, n_docs=None):
    """
    Load an array of embedding fft signature data and the corresponding
    metadata, returning both together.
    """
    metadata = load_metadata(metadata_path, fft_path,
                             csv_delim, htid_col, n_docs=n_docs)
    htid_test = metadata.index if htid_test else False
    fft_arr = load_embedding_fft_signature_array(
            fft_path, start, end, reload, htid_test=htid_test, n_docs=n_docs)

    return fft_arr, metadata


def complex_blob(n, scale=1.0, seed=0):
    """
    Sample `n` values from a 2-d normal distribution and represent the
    result as a vector of complex numbers. When plotted on the imaginary
    plane, this looks like a blob; hence the name.
    """
    rng = numpy.random.default_rng(seed)
    blob = rng.multivariate_normal(
        [0, 0], [[scale ** 2, 0], [0, scale ** 2]], n
    )
    x, y = blob.T
    return x + y * 1j


def stable_random_hermitian_matrix(reduce, full):
    """
    Generate a fixed random projection matrix for the given size. By
    reseeding the rng for each row / transposed column, we guarantee
    that the first (r, c) rows and columns will always be identical.

    This will be a random hermitian matrix, meaning that the diagonals
    are sampled from a standard normal distribution over the reals,
    the off-diagonals are sampled from the standard normal distribution
    distribution over the complex numbers (i.e. a bivariate normal
    distribution such that (x, y) -> (x + y * i)) and the final
    matrix is equal to its own cojugate transpose (so we fill the
    lower triangle with the transpose of the upper triangle).

    This approach turns out not to generate a unitary matrix, which
    in practice means that it "zooms" toward or away from the data.
    We tried a simplistic approach to fixing this problem -- dividing
    by the zoom factor -- and it worked well enough.

    However, it has no particularly strong mathematical justification,
    and although it is easier to explain in simple language what it
    does, it actually makes things more complicated.

    In the end, we chose an approach that is simpler, and has a
    stronger mathematical justification. However, the justification
    is more abstract, and so harder to communicate.
    """

    # A bit wasteful, but we just fill the whole matrix with
    # random complex numbers first. Because we supply a seed
    # every time the result is both (psudo-)random and deterministic.
    rhm = numpy.zeros((full, full), dtype=numpy.complex128)
    for i in range(full):
        rhm[i, :] = complex_blob(full, seed=i)

    # Then we zero out the lower triangle of the matrix, including
    # the diagonal...
    rhm = numpy.triu(rhm, k=1)
    # And fill it with the conjugate transpose of the upper triangle...
    rhm += rhm.conjugate().T
    # And finaly fill the diagonal with real random values...
    rhm[numpy.diag_indices_from(rhm)] = complex_blob(full, seed=full).real

    # Look at how much the matrix lengthens or shortens vectors.
    test_vecs = numpy.array([complex_blob(full) for i in range(10)]).T
    test_transformed = rhm @ test_vecs

    # Calculate the increase in magnitude caused by this
    # transformation on a few differetn vectors -- call it the
    # "inflation ratio" -- and take the mean. (Won't a matrix change
    # the length of any vector by the same amount, proprtionally
    # speaking? So do we really need to try it out on multiple
    # vectors? But we do because while it might not be necessary,
    # it also doesn't hurt much.)
    inflation_ratio = (magnitude(test_transformed, axis=0) /
                       magnitude(test_vecs, axis=0))
    mean_inflation_ratio = inflation_ratio.mean()

    # Now divide by the mean inflation ratio.
    rhm /= mean_inflation_ratio

    # Finally, we drop all but the first `reduce` rows. We've wasted
    # a bit of time generating the full matrix, but for now
    # the added complexity of filling only the first `reduce`
    # rows is not worth taking on yet.
    reduced_rhm = rhm[0:reduce, :]
    return reduced_rhm


def stable_random_haar_measure_matrix(reduce, full):
    """
    Generate a random matrix "distributed with Haar measure."
    Very closely modeled on code from this article:

    http://www.ams.org/notices/200705/fea-mezzadri-web.pdf

    I don't know much about the meaning of "Haar measure." But
    I know that it means the matrix is unitary.

    Unitary matrices preserve lengths and angles between vectors.
    In other words, if you take two vectors and transform them
    with a unitary matrix, their lengths won't change, and the
    angle between them won't change. These are nice properties
    for a dimension reduction matrix to have.

    So this function selects a random matrix from among all
    those that have this property. (It may also constrain them
    to have some other properties -- I'm not sure.)

    Also, for a given pair of `reduce` and `full` arguments,
    this function always selects the same "random" matrix. That
    is, the matrix is random in the sense that it is arbitrarily
    chosen, but it's also stable -- for any given input, this
    function will always return the same output.

    This ensures that the the signature for a given volume is
    always the same, and we can compare signatures generated
    at different times and places.
    """

    z = numpy.zeros((full, full), dtype=numpy.complex128)
    for i in range(full):
        z[i, :] = complex_blob(full, seed=i)

    q, r = numpy.linalg.qr(z)
    d = numpy.diagonal(r)
    ph = d / numpy.absolute(d)
    q = numpy.multiply(q, ph, q)

    reduced_q = q[0:reduce, :]
    return reduced_q


def vector_magnitude(vec, axis=None):
    """
    Calculate the mangitude of a real vector. If axis is provided,
    vec is treated as an array of vectors and reduced along the
    given axis. This will work on arrays of any dimension.
    """
    return (vec * vec).sum(axis=axis) ** 0.5


def vector_normalize(vec, axis=None):
    """
    Normalize a real vector. If axis is provided, vec is treated
    as an array of vectors, and normalized along the given axis.
    This should work for arrays of any dimension. (Test this!)
    """
    mag = vector_magnitude(vec, axis=axis)
    mag = mag if mag > 0 else 1
    if axis is None:
        return vec / mag
    else:
        axis_ix = [None] * len(vec.shape)
        axis_ix[axis] = slice(None, None, None)
        return vec / numpy.array([mag])[axis_ix]


def complex_magnitude(c):
    """
    Calculate the mangitude of a complex number or array of complex
    numbers. In the case of an array, this returns an array of the
    same shape as the input containing the magnitude of each complex
    number.
    """
    return (c * c.conjugate()) ** 0.5


def complex_normalize(c):
    """
    Normalize a complex number or array of complex numbers. In the
    case of an array of complex numbers, this returns an array of
    the same shape as the input containing normalized (unit-magnitude)
    complex numbers.
    """
    mag = complex_magnitude(c)
    mag = mag if mag > 0 else 1
    return c / mag


def magnitude(complex_vec, axis=None):
    """
    This does some extra work, but is fully generalized -- i.e.
    it correctly caclulates the magnitude of any vector, complex
    or real. See `vector_magnitude` for the behavior of the
    `axis` paramter.
    """
    cv_mag_vector = complex_magnitude(complex_vec)
    return vector_magnitude(cv_mag_vector, axis=axis)


def normalize(complex_vec, axis=None):
    """
    Like magnitude, this does extra work, but correctly
    normalizes both complex and real vectors. See `vector_magnitude`
    for the behavior of the `axis` parameter.
    """
    cv_mag_vector = complex_magnitude(complex_vec)
    return vector_normalize(cv_mag_vector, axis=axis)


class PhasorSignature:
    """
    A class that stores state for the purpose of generating signature
    vectors based on longer input vectors. The resulting signatures are
    substantially smaller than the input vectors, but as useful or even
    more useful for the purpose of deduplication.
    """
    def __init__(self, *, n_components=10, n_anchors=3):
        self.n_components = n_components
        self.n_anchors = n_anchors
        self._argshuf = []
        self._reduce = stable_random_haar_measure_matrix(
            self.n_complex_components, 300
            )

    @property
    def n_complex_components(self):
        """
        The number of complex components required to store
        `self.n_components` real values. When `self.n_components` is
        even, this will be exactly `self.n_components / 2`. When it
        is odd, it will be the first integer larger than
        `self.n_components / 2`. The final imaginary part of the
        last component will be superfluous.
        """
        return self.n_components // 2 + (self.n_components % 2)

    def signature(self, phasors):
        """
        Create a signature based on a phasor blob.

        To develop an intuition for how this works, picture a set
        of phasors as a bivariate normal distribution -- in other
        words, a roundish blob of points.

        What effects do front-matter, end-matter, and OCR errors
        have on that blob? Based on the properties of phasors, we
        can describe them in terms of two distinct actions on the
        blob: rotation and perturbation.

        The rotation is caused by the addition of front- and
        end-matter. Those additions shift the bulk of the body text
        forward or backward, and phasors represent that shift as a
        rotation. So our phasor blob rotates around the origin as
        front-matter is added, and then rotates back the other
        direction as end-matter is added. When these additions
        balance each other out, the net rotation is zero. (It's
        not clear how often this happens, empirically speaking.)

        The perturbation is caused by errors of various kinds, and
        by the content of the front- and end-matter. These textual
        variations slightly modify the way the Fourier transform
        breaks down the semantic structure of the documents.
        The result is that the phasors get bumped about a bit.

        To identify duplicates, we want to find a way to undo
        these two actions. Strictly speaking, of course, this is
        impossible. However, we can come close.

        Let's begin with the noise. We don't know exactly how the
        phasors get bumped about, but let's cross our fingers and
        hope that it's in a way that is essentially random and
        independent. If that's the case, then the noise has a
        useful property: if you add it up, the sum will tend
        towards zero. That's because the jitter is as likely to
        push points in one direction as in any other. In the
        aggregate, it all cancels itself out. So even though
        we can't restore any one phasor to its original location,
        we can take the average of many phasors, and the result
        will be very close to what it would have been before
        the noise was added.

        If we use this strategy, and combine it with the strategy
        of picking the most pronounced outliers in the blob --
        the phasors that are furthest from the origin -- we can
        create an anchor phasor. For two phasor blobs that come
        from the same source text, that anchor phasor will have
        nearly the same position relative to the other phasors
        in the blob. So we can measure the position of other
        phasors relative to that anchor to get a stable signature.

        -------------------

        Some strategies to test:

          Maybe a single outlier can be the anchor.
            (update: No. This didn't work as well. Three seems
             to be the magic number, but this is a parameter now.)
          Maybe we can find an alternative to using outliers as our
            anchor points.
            (update: No. At least not yet. We have not come up with
             any good alternatives.)
          Maybe we should be averaging together points at every
            step -- not just at the anchor step but for each signature
            dimension.
            (update: No. We currently do random projection instead,
             which effectively averages all points togeter with
             different weights. However, we did try this, and it
             also worked reasonably well. Note that it is roughly
             equivalent to projection with a random binary matrix.)
          Maybe there are properties of individual semantic
            dimensions that would make them better or worse anchors.
            (update: We still have no idea about this!)
        """
        phasors = phasors.reshape(-1)

        top_mag = sorted(phasors, key=complex_magnitude)[-self.n_anchors:]
        centroid = sum(top_mag) / self.n_anchors
        offset = complex_normalize(centroid)
        offset = 1 if offset == 0 else offset
        sig_offset = phasors / offset

        # sig_centroids = self.n_centroids(sig_offset)
        sig_rand = self.rand_reduce(sig_offset)

        signature = numpy.empty(len(sig_rand) * 2)
        signature[::2] = sig_rand.real
        signature[1::2] = sig_rand.imag
        return signature[:self.n_components]

    def stable_shuffle(self, seq):
        """
        Choose a random permutation based on the input length of the
        seuqence, and apply that permutation to all inputs of that length.
        Should be stable across runs, but not guaranteed to be stable
        across numpy versions. (TODO: Fix that, if possible.)
        """
        seq = numpy.asarray(seq)
        if len(seq) != len(self._argshuf):
            # Reset the rng using seq length as the seed.
            # Why not just use the same seed every time? Dunno.
            rng = numpy.random.default_rng(len(seq))
            # Save the first permutation generated thereby.
            self._argshuf = rng.permutation(len(seq))
        return seq[self._argshuf]

    def n_centroids(self, phasors):
        n = self.n_complex_components
        phasors = self.stable_shuffle(phasors)
        centroids = []
        for start in range(n):
            span = [phasors[i] for i in range(start, len(phasors), n)]
            centroids.append(sum(span) / len(span))
        return numpy.array(centroids)

    def rand_reduce(self, phasors, nan_debug=False):
        full = len(phasors)
        reduce = self.n_complex_components
        if self._reduce.shape != (reduce, full):
            self._reduce = stable_random_haar_measure_matrix(reduce, full)
        result = self._reduce @ phasors

        if nan_debug:
            nan_indices = numpy.isnan(result)
            if nan_indices.any():
                print(numpy.where(nan_indices))
                ix = numpy.where(nan_indices) + (Ellipsis,)
                print(self._reduce[ix] @ phasors)
                print(numpy.isnan(self._reduce[ix]).any())

        return result

    def fit_transform(self, data):
        """
        This model is stateless, so there is nothing to fit.
        Just return the signature transform.
        """
        return self.transform(data)

    def fit(self, data):
        """This model is stateless, so there is nothing to fit."""
        return self

    def transform(self, data):
        """Calculate the signature of the given data."""
        unflattened = [unflatten_vec(d) for d in data]
        return numpy.array([self.signature(uf)
                            for uf in unflattened])


class Deduplicator:
    def __init__(self, data, random=False, signature=False, **umap_kwargs):
        if isinstance(data, Deduplicator):
            # Create a shallow copy of the input Deduplicator. The
            # lists of data (`data_reduced` and `data_kd`) will be
            # unique to the new object, but the underlying numpy
            # arrays will be shared.

            # We store two different representations of the input
            # data here. The first is the dimension-reduced data,
            # as derived from one of the available sklearn or
            # sklearn-like dimension reduction models.

            # The second is the same data, but inserted into a
            # kd_tree for (reasonably) efficient nearest-neighbor
            # queries. (Someday perhaps we will use some other
            # NN query approach, such as that provided by the
            # pynndescent library. But for now kd_trees are good
            # enough.)

            # We used to store the original data here as well, but
            # it wasn't being used, took up a lot of memory, and
            # prevented us from using pre-reduced data to create
            # new Deduplicator objects. So we stopped.

            self.data_reduced = list(data.data_reduced)
            self.data_kd = list(data.data_kd)
        else:
            if random or signature:
                umap_kwargs = {k: umap_kwargs[k] for k in
                               ['n_components']
                               if k in umap_kwargs}

                # It would be more polite to use sklearn's built-in
                # scaling function, `sklearn.preprocessing.scale`.
                # But that function dutifully issues warnings every
                # time our oddball HathiTrust data causes numerical
                # issues, and those warnings clutter up our notebooks.
                if len(data) > 0:
                    data_norm = data - data.mean(axis=0)
                    data_std = data_norm.std(axis=0)
                    data_std[data_std == 0] = 1
                    data_norm = data_norm / data_std

                if random:
                    model = GaussianRandomProjection
                else:
                    model = PhasorSignature
            else:
                model = umap.UMAP

            if len(data) > 0:
                # Merged Deduplicator objects may contain more than one
                # batch of data, so these are lists. But upon initial
                # object creation, there is just one batch of data.
                self.data_reduced = [model(**umap_kwargs).fit_transform(data)]
                self.data_kd = [cKDTree(self.data_reduced[0])]
            else:
                self.data_reduced = []
                self.data_kd = []

    @classmethod
    def from_reduced(cls, data, random=False, signature=False, **umap_kwargs):
        """
        Create a new Deduplicator from data that has already been
        dimension-reduced. The signature is identical to the normal
        constructor for the sake of consistency.
        """
        new = Deduplicator([], random=random, signature=signature,
                           **umap_kwargs)
        new.data_reduced = [data]
        new.data_kd = [cKDTree(new.data_reduced[0])]

        return new

    @property
    def n_trees(self):
        """
        The number of trees in the Deduplicator.
        """
        return len(self.data_kd)

    @property
    def n_points(self):
        """
        Deduplicator objects may store multiple representations of the
        points they contain, but they always contain the same points,
        so `n_points` will always be equal to the number of rows in
        any of the matrices stored in `self.data_reduced`.

        However, it is possible to pass in data of length zero to the
        constructor, in which case the Deduplicator is empty. In that
        case, there are no matrices stored in `self.data_reduced`, so
        we return zero.
        """

        if self.data_reduced:
            return len(self.data_reduced[0])
        else:
            return 0

    def merge(self, other):
        """
        Combine two Deduplicators. The resulting merged Deduplicator
        contains all the KD trees and reduced datasets from the two
        inputs.

        When consulted, the merged Deduplicator returns points that
        appear in *all* the KD trees it contains.
        """
        if other.n_points != self.n_points:
            raise ValueError(
                'Deduplicator size mismatch: '
                f'{self.n_points} != {other.n_points}'
            )
        self.data_reduced.extend(other.data_reduced)
        self.data_kd.extend(other.data_kd)

    def _get_pairs_simple(self, distance):
        """
        A simple private deduplication routine. Slow -- use only for
        testing.
        """
        pairs = self.data_kd[0].query_pairs(distance)
        pairs = set(frozenset(p) for p in pairs)
        for kd in self.data_kd[1:]:
            newpairs = set(frozenset(p) for p in kd.query_pairs(distance)
                           if frozenset(p) in pairs)
            pairs = newpairs
        return pairs

    def _get_pairs_onebatch(self, distance, batch):
        """
        A pair-finding routine that operates batchwise, for speed.
        Each item in each batch is tested against all items in all
        KD trees, and the resulting pairs are aggregated in
        `_get_pairs_batched`.
        """
        data_reduced = self.data_reduced[0]
        data_kd = self.data_kd[0]
        pairs = data_kd.query_ball_point(data_reduced[batch], distance)
        pairs = set(frozenset((i, m)) for
                    i, matches in zip(batch, pairs)
                    for m in matches
                    if m != i)
        for t in range(1, self.n_trees):
            data_reduced = self.data_reduced[t]
            data_kd = self.data_kd[t]
            newpairs = data_kd.query_ball_point(data_reduced[batch], distance)
            newpairs = set(frozenset((i, m)) for
                           i, matches in zip(batch, newpairs)
                           for m in matches
                           if m != i)
            pairs = set(p for p in newpairs if p in pairs)

        return pairs

    def _get_pairs_batched(self, distance, batchsize=1000):
        """
        Aggregate the pairs generated by `_get_pairs_onebatch`.
        """
        indices = range(0, self.n_points)
        batches = [list(indices[i: i + batchsize])
                   for i in range(0, self.n_points, batchsize)]
        pairs = self._get_pairs_onebatch(distance, batches[0])
        for b in batches[1:]:
            pairs.update(self._get_pairs_onebatch(distance, b))
        return pairs

    def __call__(self, distance):
        """
        Run the full pair-finding routine. Currently the only public
        pair-finding interface.
        """
        # return self._get_pairs_simple(distance)
        return self._get_pairs_batched(distance)


def _deduplicator_balltree(data, **umap_kwargs):
    """
    An extremely simple deduplicator for testing the `BallTree`
    data structure provided by `sklearn.neighbors`, as an alternative
    to standard KD tree structures. `BallTree`s may perform better on
    high-dimensional data, but were less effective than KD trees
    in our initial tests.

    No longer public (hence the leading underscore), but preserved
    here as a record of some of our preliminary work.
    """
    data_umap = umap.UMAP(**umap_kwargs).fit_transform(data)
    data_kd = BallTree(data_umap)

    def deduplicate(distance):
        matches = data_kd.query_radius(data_umap, distance)
        pairs = [(i, m) for i, point in enumerate(data_umap)
                 for m in matches[i]]
        return set(frozenset(p) for p in pairs)
    return deduplicate


def _umap_concat(data, **umap_kwargs):
    """
    A simple approach to creating signatures by concatenating the
    outputs of multiple UMAP models. Not very successful.

    No longer public (hence the leading underscore), but preserved
    here as a record of some of our preliminary work.
    """
    data_tiles = []
    for i in range(5):
        data_i = slice_vec_bands(data, start=i, end=i + 1)
        data_tiles.append(umap.UMAP(**umap_kwargs).fit_transform(data_i))

    data_concat = numpy.empty((
        data_tiles[0].shape[0],
        sum(dt.shape[1] for dt in data_tiles)
    ))

    start_col = 0
    for dt in data_tiles:
        end_col = start_col + dt.shape[1]
        data[:, start_col:end_col] = dt
        start_col = end_col

    return data_concat


def string_similarity(a, b):
    """A readable shortcut for the SequenceMatcher ratio method."""
    return SequenceMatcher(a=a, b=b).ratio()


def show_dataset(folder, n=10):
    """
    Show a few hundred characters from the first page of the first
    `n` volumes in a folder.
    """
    volumes = volume_paths(folder)
    for i, v in enumerate(volumes):
        print(load_pages(v)[0][0:500])
        if i >= n:
            break


def umap_color(metadata, color_field, n_colors, dtype=None, palette=magma):
    """
    A helper function used by `show_umap_bokeh` to build an array of
    color data for use by Bokeh.
    """
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
    """
    Generate a Bokeh scatterplot based on a UMAP model. Colors points
    based on the metadata field indicated by `color_field`. If no
    color field is provided, a third UMAP dimension is used instead.
    """
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
