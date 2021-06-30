# ==========
# Primordial
# ==========
import sys, os, time, gzip, zipfile
from collections import *
from io import StringIO
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from Bio import SeqIO

# =========
# Functions
# =========
# FPKM, RPKM, and TPM
def normalize_sequencing(df_counts, length=None, mode="tpm"):
    """
    # FPKM
    Fragments Per Kilobase of transcript per Million mapped reads
        C = Number of reads mapped to a gene
        N = Total mapped reads in the experiment
        L = exon length in base-pairs for a gene
        Equation: RPKM = (10^9 * C)/(N * L)
            or
        numReads / ( geneLength/1000 * totalNumReads/1,000,000 ) # C/((L/1e3).T*(N/1e6)).T

    # TPM
    Transcripts Per Kilobase Million
        (1) Divide the read counts by the length of each gene in kilobases. This gives you reads per kilobase (RPK).
        (2) Count up all the RPK values in a sample and divide this number by 1,000,000. This is your “per million” scaling factor.
        (3) Divide the RPK values by the “per million” scaling factor. This gives you TPM.
    """
    mode = mode.lower()
    # Clone
    df_counts = df_counts.copy()

    # Conditionals
    cond_a = mode in {"fpkm", "rpkm", "tpm"}
    cond_b = length is not None

    # Index
    if cond_b:
        length = length.copy()
        try:
            length = length.astype(int)
        except ValueError:
            print("Converting `length` into integers from sequences", file=sys.stderr)
            length = length.map(len)
        if set(df_counts.columns) <= set(length.index):
            length = length[df_counts.columns]
        else:
            raise Exception("Error: Not all genes in `df_counts` have sequence lengths in `length`")

    # FPKM, RPKM, and TPM normalization
    if all([cond_a, cond_b]):
        if type(length) in {dict, OrderedDict, defaultdict}:
            length = pd.Series(length)

        # Set up variables
        C = df_counts.as_matrix()
        L = np.array([length[df_counts.columns].tolist()]*df_counts.shape[0])
        N = df_counts.sum(axis=1).as_matrix()

        if mode in {"fpkm","rpkm"}:
            # Compute operations
            celestial = 1e9 * C
            terra = (N*L.T).T
            return pd.DataFrame(celestial/terra, index=df_counts.index, columns=df_counts.columns)

        if mode == "tpm":
            rpk = C/L
            per_million_scaling_factor = rpk.sum(axis=1)/1e6
            return pd.DataFrame( (rpk.T/per_million_scaling_factor), index=df_counts.columns, columns=df_counts.index).T
    if cond_a and not cond_b:
        raise Exception("Error: Must provide `length` if using `mode` in {'rpkm', 'fpkm', 'tpm'}")
        
# Parse the files from KEGG Maple
def mcr_from_directory(directory, name=None, end_tag="module.xls", mcr_field="MCR % (ITR)", index_name_bins = "id_bin", index_name_module = "id_module", store_raw_data=False):
    """
    directory: 
    Path to kegg maple results which should have the following structure:
    | directory
    | | bin_1
    | | | xyz1.module.xls
    | | bin_2
    | | | xyz2.module.xls
    
    Returns:
    namedtuple with the following attributes: 
        .ratios = MCR ratio
        .metadata = module metadata
        .name = user-provided name of experiment
        .raw_data = {id_bin:raw_kegg_maple_}
    
    Assumes subdirectories are bin_ids and 
    each directory has a file that has the 
    following format: 8206709269.module.xls
    
    https://www.genome.jp/maple-bin/mapleSubmit.cgi?aType=sDirect
    """
    MCR = namedtuple("MCR", ["ratios", "metadata", "name", "raw_data"])

    directory = Path(directory)
    
    # Check directory
    assert directory.exists(), f"{str(directory.absolute())} does not exist"
    assert directory.is_dir(), f"{str(directory.absolute())} is not a directory"
    
    # Iterate through each KEGG Maple result dataset
    d_bin_mcr = OrderedDict()
    module_metadata = list()
    d_bin_raw = dict()
    for subdir in tqdm(directory.iterdir(), f"Reading subdirectories in {str(directory)}"):
        sheets = list()
        if subdir.is_dir():
            id_bin = subdir.name
            for file in subdir.iterdir():
                if file.name.endswith(end_tag):
                    for sheetname in ["module_pathway", "module_complex", "module_function", "module_signature"]:
                        df = pd.read_excel(file, index_col=0, sheetname=sheetname).set_index("ID", drop=False)
                        sheets.append(df)
            df_bin = pd.concat(sheets, axis=0)
            module_metadata.append(df_bin.loc[:,["Type", "Small category", "Name(abbreviation)"]])
            d_bin_mcr[id_bin] = df_bin[mcr_field]
            # Store raw data
            if store_raw_data:
                d_bin_raw[id_bin] = df_bin.drop("ID", axis=1)
    # Module Completion Ratios
    df_mcr = pd.DataFrame(d_bin_mcr).T.fillna(0.0)
    df_mcr.index.name = index_name_bins
    df_mcr.columns.name = index_name_module
    
    # Metadata
    df_metadata = pd.concat(module_metadata, axis=0).drop_duplicates() # May not be the most efficient but w/e
    
    return MCR(ratios=df_mcr, metadata=df_metadata, name=name, raw_data=d_bin_raw)
            

# Read Fasta File
def read_fasta(path, description=True, case="upper", func_header=None, into=pd.Series, compression="infer"):
    """
    Reads in a single fasta file or a directory of fasta files into a dictionary.
    """
    # Compression
    if compression == "infer":
        if path.endswith(".gz"):
            compression = "gzip"
        elif path.endswith(".zip"):
            compression = "zip"
        else:
            compression = None

    # Open file object
    if compression == "gzip":
        handle = StringIO(gzip.GzipFile(path).read().decode("utf-8"))
    if compression == "zip":
        handle = StringIO(zipfile.ZipFile(path,"r").read(path.split("/")[-1].split(".zip")[0]).decode("utf-8"))
    if compression == None:
        handle = open(path, "r")

    # Read in fasta
    d_id_seq = OrderedDict()

    # Verbose but faster
    if case == "lower":
        for record in SeqIO.FastaIO.SimpleFastaParser(handle):
            seq = record[1].lower()
            if description == True:
                header = record[0]
            if description == False:
                header = record[0].split(" ")[0]
            d_id_seq[header] = seq
    if case == "upper":
        for record in SeqIO.FastaIO.SimpleFastaParser(handle):
            seq = record[1].upper()
            if description == True:
                header = record[0]
            if description == False:
                header = record[0].split(" ")[0]
            d_id_seq[header] = seq
    if case is None:
        for record in SeqIO.FastaIO.SimpleFastaParser(handle):
            seq = record[1]
            if description == True:
                header = record[0]
            if description == False:
                header = record[0].split(" ")[0]
            d_id_seq[header] = seq

    # Close File
    handle.close()

    # Transform header
    if func_header is not None:
        d_id_seq = OrderedDict( [(func_header(id),seq) for id, seq in d_id_seq.items()])
    return into(d_id_seq)

# Phylogeneomically binned functional potential
def phylogenomically_binned_functional_potential(X:pd.DataFrame,MCR:pd.DataFrame):
    """
    X: DataFrame of scaled compositional data that preserves proportionality (e.g. TPM or relative-abundance)
    MCR: DataFrame
    
    X.shape (n,m)
    MCR.shape (m,p)
    Output: (n,p)
    """
    assert X.shape[1] == MCR.shape[0], "Shapes are not compatible"
    def matmul(A,B):
        return pd.DataFrame(np.matmul(A, B.loc[A.columns,:]), index=A.index, columns=B.columns)
    return matmul(X,MCR)

# =============
# Example usage
# =============
path_to_keggmapleresults = "./Data/genomes/iter7/maple_results/"
mcr_data = mcr_from_directory(path_to_keggmapleresults)
df_mcr = mcr_data.ratios
# (34, 433)

## Previews
# print(df_mcr.head().iloc[:,:5])
# id_module  M00165  M00166  M00167  M00168  M00169
# id_bin                                           
# bin_1        54.5    25.0    71.4    50.0    50.0
# bin_10        0.0     0.0     0.0     0.0     0.0
# bin_11       72.7    50.0    85.7    50.0    50.0
# bin_12       63.6    25.0    85.7   100.0    50.0
# bin_13       63.6    50.0    71.4    50.0     0.0

# print(mcr_data.metadata.head())
#            Type   Small category  \
# M00165  Pathway  Carbon fixation   
# M00166  Pathway  Carbon fixation   
# M00167  Pathway  Carbon fixation   
 

#                                                          Name(abbreviation)  
# M00165                     Reductive pentose phosphate cycle (Calvin cycle)  
# M00166  Reductive pentose phosphate cycle, ribulose-5P => glyceraldehyde-3P  


# Annotations
df_annot = pd.read_table("./Data/annotations/caries.metag.iter7.annot.taxonomy.tsv.gz", sep="\t", index_col=0, compression="gzip")

# Contig counts
df_counts = pd.read_table("./Data/counts/caries.metag.iter7.contig.counts.tsv.gz", sep="\t", index_col=0, compression="gzip")
# (88, 26583)

# Sequences
Se_contigs = read_fasta("./Data/sequences/iter7/caries.metag.iter7.contigs.fa.gz")

# TPM
df_tpm = normalize_sequencing(df_counts, length=Se_contigs.map(len), mode="tpm")
# (88, 26583)

# ID Mapping between contigs -> bin
d_contig_bin = dict(zip(df_annot["contig_id"], df_annot["bin_id"]))

# Bins
df_bins = df_tpm.groupby(d_contig_bin, axis=1).sum()
# (88, 34)

# Phylogenomically binned functional potential
df_pbfp = phylogenomically_binned_functional_potential(df_bins, df_mcr)
# (88, 433)
