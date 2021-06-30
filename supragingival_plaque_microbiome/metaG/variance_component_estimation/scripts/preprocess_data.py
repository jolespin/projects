
import sys, os, re, itertools,argparse
import pandas as pd
import numpy as np
# Optional
try:
    from tqdm import tqdm
    tqdm_available = True
except ModuleNotFoundError:
    tqdm_available = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, help='Input: path/to/metadata.tsv')
    parser.add_argument('--attribute_matrix', type=str, help = 'Input: Path/to/Tab-separated-value.tsv')
    parser.add_argument('--token', type=str, help = 'Random generated hex string for path')

    opts = parser.parse_args()

    # Random Seed
    # token = "".join(np.random.RandomState(opts.seed).choice(list("123456789acgt"), size=28))
    path_hidden = f"./.tmp_twinlm/{opts.token}.encoding"
    # Base
    df_twinlm_base = pd.read_table(opts.metadata, sep="\t", index_col=0)
    # Attributes
    df_twinlm_attributes = pd.read_table(opts.attribute_matrix, sep="\t", index_col=0)
    # Encoding File
    d_encode = dict([*map(lambda x:(x[1], f"ATTR_{x[0]+1}"), enumerate(df_twinlm_attributes.columns))])

    with open(path_hidden, "w") as f:
        pd.Series(d_encode).to_frame("encoding").to_csv(f, sep="\t")
    df_twinlm_attributes.columns = df_twinlm_attributes.columns.map(lambda x:d_encode[x])

    # Overlapping
    idx_overlap = pd.Index(list(set(df_twinlm_base.index) & set(df_twinlm_attributes.index)))
    df_twinlm_base = df_twinlm_base.loc[idx_overlap,:]
    # Remove single twins
    idx_family = df_twinlm_base["FamilyID"].value_counts().compress(lambda x: x == 2).index
    idx_output = df_twinlm_base.index[df_twinlm_base["FamilyID"].map(lambda x: x in idx_family)]
    # Merge dataframe
    df_output= pd.concat([df_twinlm_base.loc[idx_output,:], df_twinlm_attributes.loc[idx_output,:]], axis=1)

    # Stream DataFrame
    df_output.to_csv(sys.stdout, sep="\t")

    # Shape
    print("X.shape= ", df_twinlm_attributes.shape, file=sys.stderr)
    sys.exit()


if __name__ == "__main__":
    main()
