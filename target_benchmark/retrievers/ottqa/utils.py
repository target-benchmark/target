import gzip
import importlib.util
import json
import math
import os
import re
import sqlite3
import urllib.parse
import warnings
from collections import Counter
from functools import partial
from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from typing import Dict, List, Union

import numpy as np
import scipy.sparse as sp
from dateutil.parser import parse
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

from .drqa import drqa_tokenizers, retriever

warnings.filterwarnings("ignore")


def tokenize(string, remove_dot=False):
    def func(string):
        return " ".join(word_tokenize(string))

    string = string.replace("%-", "-")
    if remove_dot:
        string = string.rstrip(".")

    string = func(string)
    string = string.replace(" %", "%")
    return string


def url2dockey(string):
    string = urllib.parse.unquote(string)
    return string


def filter_firstKsents(string, k):
    string = sent_tokenize(string)
    string = string[:k]
    return " ".join(string)


def compressGZip(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)

    json_str = json.dumps(data) + "\n"  # 2. string (i.e. JSON)
    json_bytes = json_str.encode("utf-8")  # 3. bytes (i.e. UTF-8)

    with gzip.GzipFile(file_name + ".gz", "w") as fout:  # 4. gzip
        fout.write(json_bytes)


def readGZip(file_name):
    if file_name.endswith("gz"):
        with gzip.GzipFile(file_name, "r") as fin:  # 4. gzip
            json_bytes = fin.read()  # 3. bytes (i.e. UTF-8)

        json_str = json_bytes.decode("utf-8")  # 2. string (i.e. JSON)
        data = json.loads(json_str)  # 1. data
        return data
    else:
        with open(file_name, "r") as fin:
            data = json.load(fin)
        return data


class CellHelper(object):
    """Cell Helper to detect the cell type."""

    @staticmethod
    def is_unit(string):
        """Is the input a unit."""
        return re.search(r"\b(kg|m|cm|lb|hz|million)\b", string.lower())

    @staticmethod
    def is_score(string):
        """Is the input a score between two things."""
        if re.search(r"[0-9]+ - [0-9]+", string):
            return True
        elif re.search(r"[0-9]+-[0-9]+", string):
            return True
        else:
            return False

    @staticmethod
    def is_date(string, fuzzy=False):
        """Is the input a date."""
        try:
            parse(string, fuzzy=fuzzy)
            return True
        except Exception:  # pylint: disable=broad-except
            return False

    @staticmethod
    def is_bool(string):
        if string.lower() in ["yes", "no"]:
            return True
        else:
            return False

    @staticmethod
    def is_float(string):
        if "." in string:
            try:
                float(string)
                return True
            except Exception:
                return False
        else:
            return False

    @staticmethod
    def is_normal_word(string):
        if " " not in string:
            return string.islower()
        else:
            return False


def whitelist(string):
    """Is the input a whitelist string."""
    string = string.strip()
    if len(string) < 2:
        return False
    elif string.isdigit():
        if len(string) == 4:
            return True
        else:
            return False
    elif string.replace(",", "").isdigit():
        return False
    elif CellHelper.is_float(string):
        return False
    elif "#" in string or "%" in string or "+" in string or "$" in string:
        return False
    elif CellHelper.is_normal_word(string):
        return False
    elif CellHelper.is_bool(string):
        return False
    elif CellHelper.is_score(string):
        return False
    elif CellHelper.is_unit(string):
        return False
    elif CellHelper.is_date(string):
        return False
    return True


def is_year(string):
    if len(string) == 4 and string.isdigit():
        return True
    else:
        return False


def build_corpus(tables: dict, tmp_file: str):
    fw = open(tmp_file, "w")
    for _, table in tables.items():
        title = table["title"]
        section_title = table["section_title"]
        headers = []
        for h in table["header"]:
            headers.append(h)
        headers = " ".join(headers)
        content = "{} | {} | {}".format(title, section_title, headers)
        fw.write(json.dumps({"id": table["uid"], "text": content}) + "\n")

    fw.close()


def convert_table_representation(
    database_id,
    table_id: str,
    table_contents: List[List],
    section_title: str,
    with_title: bool,
) -> Dict[str, object]:
    table_headers = table_contents[0]
    table_data = table_contents[1:]
    # remove title due to high correspondence but keep uid
    return {
        "uid": str((database_id, table_id)),
        "title": table_id if with_title else "",
        "header": table_headers,
        "data": table_data,
        "section_title": section_title,
    }


def get_filename(
    out_dir: str,
    option: str,
    with_title: bool,
    ngram: int,
    hash_size: int,
    tokenizer: str,
    dataset_name: str,
) -> str:
    basename = "index"
    basename += "-%s-%s-ngram=%d-hash=%d-tokenizer=%s-dataset=%s.npz" % (
        option,
        str(with_title),
        ngram,
        hash_size,
        tokenizer,
        dataset_name,
    )
    return os.path.join(out_dir, basename)


default_hash_size = int(math.pow(2, 24))
default_tokenizer = "simple"


class TFIDFBuilder:
    def __init__(self):
        self.PREPROCESS_FN = None
        self.DOC2IDX = None
        self.PROCESS_TOK = None
        self.PROCESS_DB = None

    def build_tfidf(
        self,
        out_dir: str,
        corpus: Dict[str, object],
        dataset_name: Union[str, None] = None,
        tmp_file: str = "/tmp/tf-idf-input.json",
        tmp_db_file: str = "/tmp/db.json",
        preprocess: str = None,
        num_workers: int = None,
        build_option: str = "title_sectitle_schema ",
        option: str = "tfidf",
        ngram: int = 2,
        hash_size: int = default_hash_size,
        tokenizer: str = default_tokenizer,
        with_title: bool = True,
    ):
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        build_corpus(corpus, tmp_file)

        self.store_contents(tmp_file, tmp_db_file, preprocess, num_workers)

        count_matrix, doc_dict = self.get_count_matrix(
            tokenizer, num_workers, ngram, hash_size, "sqlite", {"db_path": tmp_db_file}
        )

        tfidf = self.get_tfidf_matrix(count_matrix, count_matrix, option=option)

        freqs = self.get_doc_freqs(count_matrix)

        filename = get_filename(out_dir, option, with_title, ngram, hash_size, tokenizer, dataset_name)

        metadata = {
            "doc_freqs": freqs,
            "tokenizer": tokenizer,
            "hash_size": hash_size,
            "ngram": ngram,
            "doc_dict": doc_dict,
        }
        retriever.utils.save_sparse_csr(filename, tfidf, metadata)
        return filename

    def store_contents(self, data_path, save_path, preprocess, num_workers=None):
        """Preprocess and store a corpus of documents in sqlite.

        Args:
            data_path: Root path to directory (or directory of directories) of files
            containing json encoded documents (must have `id` and `text` fields).
            save_path: Path to output sqlite db.
            preprocess: Path to file defining a custom `preprocess` function. Takes
            in and outputs a structured doc.
            num_workers: Number of parallel processes to use when reading docs.
        """
        if os.path.isfile(save_path):
            os.remove(save_path)
            # raise RuntimeError('%s already exists! Not overwriting.' % save_path)

        conn = sqlite3.connect(save_path)
        c = conn.cursor()
        c.execute("CREATE TABLE documents (id PRIMARY KEY, text);")

        with ProcessPool(num_workers) as pool:
            get_contents_partial = partial(get_contents, preprocess)
            files = [f for f in self.iter_files(data_path)]
            count = 0
            with tqdm(total=len(files)) as pbar:
                for pairs in tqdm(pool.imap_unordered(get_contents_partial, files)):
                    count += len(pairs)
                    c.executemany("INSERT INTO documents VALUES (?,?)", pairs)
                    pbar.update()
        conn.commit()
        conn.close()

    def iter_files(self, path):
        """Walk through all files located under a root path."""
        if os.path.isfile(path):
            yield path
        elif os.path.isdir(path):
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    yield os.path.join(dirpath, f)
        else:
            raise RuntimeError("Path %s is invalid" % path)

    def get_count_matrix(self, tokenizer, num_workers, ngram, hash_size, db, db_opts):
        """Form a sparse word to document count matrix (inverted index).

        M[i, j] = # times word i appears in document j.
        """
        # Map doc_ids to indexes
        db_class = retriever.get_class(db)

        with db_class(**db_opts) as doc_db:
            doc_ids = doc_db.get_doc_ids()
        DOC2IDX = {doc_id: i for i, doc_id in enumerate(doc_ids)}

        # Setup worker pool
        tok_class = drqa_tokenizers.get_class(tokenizer)
        with ProcessPool(num_workers) as pool:
            # Compute the count matrix in steps (to keep in memory)
            row, col, data = [], [], []
            step = max(int(len(doc_ids) / 10), 1)
            batches = [doc_ids[i : i + step] for i in range(0, len(doc_ids), step)]
            _count = partial(count, ngram, hash_size, tok_class, db_class, db_opts, DOC2IDX)
            for i, batch in enumerate(batches):
                for b_row, b_col, b_data in pool.imap_unordered(_count, batch):
                    row.extend(b_row)
                    col.extend(b_col)
                    data.extend(b_data)
            pool.close()
            pool.join()

        count_matrix = sp.csr_matrix((data, (row, col)), shape=(hash_size, len(doc_ids)))
        count_matrix.sum_duplicates()
        return count_matrix, (DOC2IDX, doc_ids)

    def get_doc_freqs(self, cnts):
        """Return word --> # of docs it appears in."""
        binary = (cnts > 0).astype(int)
        freqs = np.array(binary.sum(1)).squeeze()
        return freqs

    def get_tfidf_matrix(self, cnts, idf_cnts, option="tfidf"):
        """Convert the word count matrix into tfidf one.

        tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
        * tf = term frequency in document
        * N = number of documents
        * Nt = number of occurences of term in all documents
        """
        # Computing the IDF parameters
        Ns = self.get_doc_freqs(idf_cnts)
        idfs = np.log((idf_cnts.shape[1] - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0
        idfs = sp.diags(idfs, 0)
        if option == "tfidf":
            # Computing the TF parameters
            tfs = cnts.log1p()
        elif option == "bm25":
            k1 = 1.5
            b = 0.75
            # Computing the saturation parameters
            doc_length = np.array(cnts.sum(0)).squeeze()
            doc_length_ratio = k1 * (1 - b + b * doc_length / doc_length.mean())
            doc_length_ratio = sp.diags(doc_length_ratio, 0)
            binary = (cnts > 0).astype(int)
            masked_length_ratio = binary.dot(doc_length_ratio)
            denom = cnts.copy()
            denom.data = denom.data + masked_length_ratio.data
            tfs = cnts * (1 + k1)
            tfs.data = tfs.data / denom.data
        else:
            raise NotImplementedError
        tfidfs = idfs.dot(tfs)
        return tfidfs


def init_preprocess(filename):
    if filename:
        return import_module(filename).preprocess
    return None


def import_module(filename):
    """Import a module given a full path to the file."""
    spec = importlib.util.spec_from_file_location("doc_filter", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_contents(preprocess_file_name, filename):
    preprocess_fn = init_preprocess(preprocess_file_name)
    """Parse the contents of a file. Each line is a JSON encoded document."""
    documents = []
    with open(filename) as f:
        for line in f:
            # Parse document
            doc = json.loads(line)
            # Maybe preprocess the document with custom function
            if preprocess_fn:
                doc = preprocess_fn(doc)
            # Skip if it is empty or None
            if not doc:
                continue
            # Add the document
            documents.append((doc["id"], doc["text"]))
    return documents


def init(tokenizer_class, db_class, db_opts):
    PROCESS_TOK = tokenizer_class()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)
    return PROCESS_TOK, PROCESS_DB


def fetch_text(doc_id, PROCESS_DB):
    return PROCESS_DB.get_doc_text(doc_id)


def tokenize_text(text, PROCESS_TOK):
    return PROCESS_TOK.tokenize(text)


def count(ngram, hash_size, tok_class, db_class, db_opts, DOC2IDX, doc_id):
    """Fetch the text of a document and compute hashed ngrams counts."""
    PROCESS_TOK, PROCESS_DB = init(tok_class, db_class, db_opts)
    row, col, data = [], [], []
    # Tokenize
    tokens = tokenize_text(retriever.utils.normalize(fetch_text(doc_id, PROCESS_DB)), PROCESS_TOK)

    # Get ngrams from tokens, with stopword/punctuation filtering.
    ngrams = tokens.ngrams(n=ngram, uncased=True, filter_fn=retriever.utils.filter_ngram)

    # Hash ngrams and count occurences
    counts = Counter([retriever.utils.hash(gram, hash_size) for gram in ngrams])

    # Return in sparse matrix data format.
    row.extend([int(key) for key in counts.keys()])
    col.extend([DOC2IDX[doc_id]] * len(counts))
    data.extend(counts.values())
    return row, col, data
