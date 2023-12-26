#! /usr/bin/env python3
# Time-stamp: <2018-11-21 06:59:24 cp983411>


import string
import pandas as pd
import numpy as np
import sublexstats
from random import choices


def generate_pseudoword(charset, nchar):
    """ returns a random string of characters from charset, each of length nchar."""
    return ''.join(choices(charset, k=nchar))


def compute_pseudowords_and_stats(N, charset, nchar, frallstats, frlexfreqs, enallstats, enlexfreqs):
    """ return a DataFrame with one item per line and its associated stats. """
    #a = pd.DataFrame(columns='item,frletters,frminletters,frmaxletters,frbigrams,frminbigrams,frmaxbigrams,frquadrigrams,frminquadrigrams,frmaxquadrigrams,frisword,frwordfreq,enletters,enminletters,enmaxletters,enallbigrams,enminbigrams,enmaxbigrams,enquadrigrams,enminquadrigrams,enmaxquadrigrams,enisword,enwordfreq'.split(','),
    #                index=range(N))
    a = pd.DataFrame(index=range(N))
    idx = 0
    listofpw = []
    while idx < NITEMS_PER_SET:
        w = generate_pseudoword(CHARSET, NCHAR)
        while w in listofpw:  # resample if it already exists
            w = generate_pseudoword(CHARSET, NCHAR)
        listofpw.append(w)

        frstats = frallstats.get_all_stats(w)
        enstats = enallstats.get_all_stats(w)

        if np.sum(np.array(enstats['quadrigrams'] + frstats['quadrigrams'])) == 0.0:
            None  # do nothing
        else:
            if w in frlexfreqs.index:  # check if w is in the French dictionary
                frisword, frlexfreq = 1, frlexfreqs.loc[w].freqf
            else:
                frisword, frlexfreq = 0, 0.0

            if w in enlexfreqs.index:  # check if w is in the English dictionary
                enisword, enlexfreq = 1, enlexfreqs.loc[w].freq
            else:
                enisword, enlexfreq = 0, 0.0

            a.at[idx, 'item'] = w
            a.at[idx, 'frletters'] = sublexstats.meanlogs(frstats['letters'], 0.000001)
            a.at[idx, 'frminletters'] = np.min(frstats['letters'])
            a.at[idx, 'frmaxletters'] = np.max(frstats['letters'])
            a.at[idx, 'frbigrams'] = sublexstats.meanlogs(frstats['allbigrams'], 0.000001)
            a.at[idx, 'frminbigrams'] = np.min(frstats['allbigrams'])
            a.at[idx, 'frmaxbigrams'] = np.max(frstats['allbigrams'])
            a.at[idx, 'frquadrigrams'] = sublexstats.meanlogs(frstats['quadrigrams'], 0.000001)
            a.at[idx, 'frminquadrigrams'] = np.min(frstats['quadrigrams'])
            a.at[idx, 'frmaxquadrigrams'] = np.max(frstats['quadrigrams'])
            a.at[idx, 'frisword'] = frisword
            a.at[idx, 'frfreq'] = frlexfreq
            a.at[idx, 'enletters'] = sublexstats.meanlogs(enstats['letters'], 0.000001)
            a.at[idx, 'enminletters'] = np.min(enstats['letters'])
            a.at[idx, 'enmaxletters'] = np.max(enstats['letters'])
            a.at[idx, 'enbigrams'] = sublexstats.meanlogs(enstats['allbigrams'], 0.000001)
            a.at[idx, 'enminbigrams'] = np.min(enstats['allbigrams'])
            a.at[idx, 'enmaxbigrams'] = np.max(enstats['allbigrams'])
            a.at[idx, 'enquadrigrams'] = sublexstats.meanlogs(enstats['quadrigrams'], 0.000001)
            a.at[idx, 'enminquadrigrams'] = np.min(enstats['quadrigrams'])
            a.at[idx, 'enmaxquadrigrams'] = np.max(enstats['quadrigrams'])
            a.at[idx, 'enisword'] = enisword
            a.at[idx, 'enfreq'] = enlexfreq
            idx += 1
    return a


def load_sublex_stats(fname):
    """ fname must be a csv file with two columns: ortho, and freq """
    dic = sublexstats.sublexstats()
    dic.import_csv('french-lexique-reduced.tsv')
    return dic


def load_freqs6(fname):
    """ fname must be a csv file with two columns: ortho, and freq """
    lex = pd.read_csv(fname, sep='\t')
    lex6 = lex.loc[lex.ortho.str.len() == 6]  # restrict to 6 letter words
    return lex6.groupby('ortho').sum()  # regroup rows with same ortho


if __name__ == '__main__':
    CHARSET = string.ascii_lowercase
    NCHAR = 6
    NSETS = 3
    NITEMS_PER_SET = 10000

    frlex = load_freqs6('french-lexique-reduced.tsv')
    frstats = load_sublex_stats('french-lexique-reduced.tsv')
    enlex = load_freqs6('english-blp-reduced.tsv')
    enstats = load_sublex_stats('english-blp-reduced.tsv')

    for iset in range(1, NSETS + 1):
        print(f"Generating set #{iset}...")
        items = compute_pseudowords_and_stats(NITEMS_PER_SET, CHARSET, NCHAR, frstats, frlex, enstats, enlex)
        items.to_csv(f'set_{iset:04d}.csv.gz', compression='gzip')