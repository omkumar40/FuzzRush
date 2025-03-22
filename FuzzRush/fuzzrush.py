import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
import pandas as pd

class FuzzRush():
    def __init__(self, source_names, target_names):
        self.source_names = source_names
        self.target_names = target_names
        self.ct_vect = None
        self.tfidf_vect = None
        self.vocab = None
        self.sprse_mtx = None
    
    def tokenize(self, analyzer='char_wb', n=3):
        '''
        Tokenizes the list of strings, based on the selected analyzer

        :param str analyzer: Type of analyzer ('char_wb', 'word'). Default is trigram
        :param str n: If using n-gram analyzer, the gram length
        '''
        self.ct_vect = CountVectorizer(analyzer=analyzer, ngram_range=(n, n))
        self.vocab = self.ct_vect.fit(self.source_names + self.target_names).vocabulary_
        self.tfidf_vect = TfidfVectorizer(vocabulary=self.vocab, analyzer=analyzer, ngram_range=(n, n))
    
    def match(self, ntop=1, lower_bound=0, output_fmt='df'):
        '''
        Main match function. Default settings return only the top candidate for every source string.
        
        :param int ntop: The number of top-n candidates that should be returned
        :param float lower_bound: The lower-bound threshold for keeping a candidate, between 0-1.
                                   Default set to 0, so consider all canidates
        :param str output_fmt: The output format. Either dataframe ('df') or dict ('dict')
        '''
        self._awesome_cossim_top(ntop, lower_bound)
        return self._make_matchdf() if output_fmt == 'df' else self._make_matchdict()
    
    def _awesome_cossim_top(self, ntop, lower_bound):
        #Converting To CSR Matrix
        A = self.tfidf_vect.fit_transform(self.source_names).tocsr()
        B = self.tfidf_vect.fit_transform(self.target_names).transpose().tocsr()
        M, _ = A.shape
        _, N = B.shape

        idx_dtype = np.int32
        nnz_max = M * ntop

        indptr = np.zeros(M+1, dtype=idx_dtype)
        indices = np.zeros(nnz_max, dtype=idx_dtype)
        data = np.zeros(nnz_max, dtype=A.dtype)

        ct.sparse_dot_topn(
            M, N, np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype), A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype), B.data,
            ntop, lower_bound, indptr, indices, data)
        
        self.sprse_mtx = csr_matrix((data, indices, indptr), shape=(M,N))
    
    def _make_matchdf(self):
        ''' Build dataframe for result return '''
        cx = self.sprse_mtx.tocoo()
        return pd.DataFrame(
            [(row, self.source_names[row], col, self.target_names[col], val)
             for row, col, val in zip(cx.row, cx.col, cx.data)],
            columns=['Sub Idx', 'Sub Name', 'Master Idx', 'Master Name', 'Score']
        )
    
    def _make_matchdict(self):
        ''' Build dictionary for result return '''
        cx = self.sprse_mtx.tocoo()
        match_dict = {}
        for row, col, val in zip(cx.row, cx.col, cx.data):
            match_dict.setdefault(row, []).append((col, val))
        return match_dict