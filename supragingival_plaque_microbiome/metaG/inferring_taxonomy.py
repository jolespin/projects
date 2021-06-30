# ==============================================================================
# Primordial
# ==============================================================================
import sys,os,time
from collections import *
import pandas as pd
import numpy as np
import ete3
from joblib import Parallel, delayed
from tqdm import tqdm
# ==============================================================================
# Core HMMs from CheckM
# ==============================================================================
# Core HMMs
core_bacteria = {'TIGR00250', 'PF05000', 'PF01016', 'PF01509', 'PF00687', 'PF08529', 'PF01649', 'TIGR00344', 'PF00828', 'PF00829', 'TIGR00019', 'PF01000', 'TIGR02432', 'PF13184', 'PF00889', 'TIGR00392', 'PF00562', 'PF06421', 'PF02912', 'PF04997', 'PF00189', 'PF06071', 'PF00252', 'PF00411', 'PF11987', 'PF02033', 'PF00572', 'PF00312', 'TIGR00084', 'TIGR00460', 'PF01121', 'PF00380', 'TIGR00855', 'PF00276', 'PF00203', 'PF13603', 'PF01196', 'PF01195', 'PF02367', 'PF00162', 'TIGR02075', 'TIGR00329', 'PF00573', 'PF01193', 'PF01795', 'PF05697', 'PF01250', 'PF03719', 'PF00453', 'PF01746', 'PF00410', 'PF00177', 'TIGR00810', 'PF04565', 'PF00318', 'PF00623', 'PF00338', 'PF03948', 'PF12344', 'TIGR00755', 'TIGR03723', 'PF04983', 'PF00298', 'TIGR00922', 'PF00831', 'PF00297', 'PF00237', 'PF02978', 'PF01632', 'PF01409', 'TIGR00459', 'PF00886', 'PF00466', 'TIGR03263', 'PF10385', 'PF02130', 'PF03946', 'PF00281', 'TIGR00967', 'PF04561', 'TIGR00615', 'PF01281', 'PF00366', 'TIGR01079', 'PF03484', 'PF01245', 'PF00164', 'PF05491', 'PF01668', 'PF00416', 'PF00673', 'PF04563', 'PF00238', 'PF04560', 'PF03947', 'PF00181', 'TIGR03594', 'PF00333', 'PF04998', 'PF01765', 'PF00861', 'PF00347', 'PF01018', 'PF08459'}
core_archaea = {'PF01201', 'PF01172', 'TIGR02389', 'PF01280', 'PF05000', 'PF00687', 'PF01157', 'PF09173', 'PF01982', 'TIGR00468', 'PF01287', 'TIGR00425', 'PF01655', 'PF01984', 'TIGR03679', 'PF00398', 'PF09249', 'TIGR00344', 'PF00867', 'TIGR00057', 'PF01000', 'PF01090', 'PF00833', 'TIGR01080', 'PF00736', 'PF01015', 'PF07541', 'PF01780', 'TIGR00392', 'PF00562', 'PF04566', 'PF01868', 'PF09377', 'PF03439', 'PF01246', 'PF04997', 'TIGR02338', 'PF01922', 'PF13685', 'PF00189', 'PF00935', 'PF08068', 'PF00252', 'PF01198', 'PF00832', 'PF01912', 'PF00411', 'PF11987', 'PF00572', 'TIGR00289', 'PF05670', 'PF00312', 'TIGR03724', 'PF03876', 'TIGR00134', 'PF04127', 'PF00380', 'PF01351', 'PF00900', 'PF02005', 'PF00276', 'TIGR00064', 'PF00203', 'TIGR01046', 'TIGR03677', 'TIGR00549', 'TIGR01213', 'TIGR00670', 'PF06418', 'TIGR00329', 'PF00573', 'PF01282', 'PF01193', 'TIGR00389', 'PF04567', 'PF03719', 'TIGR00270', 'PF00410', 'PF01981', 'PF00177', 'TIGR02076', 'PF00752', 'PF08071', 'PF03950', 'PF04565', 'TIGR03665', 'PF00318', 'PF00623', 'PF01864', 'PF04019', 'TIGR03685', 'TIGR03683', 'PF04983', 'TIGR00419', 'PF01200', 'PF00298', 'PF00831', 'PF00297', 'PF00237', 'PF02978', 'TIGR00408', 'PF00327', 'TIGR00422', 'PF00958', 'PF00466', 'PF00679', 'PF01798', 'PF04010', 'PF01269', 'PF01667', 'TIGR00522', 'PF01192', 'PF00827', 'PF13656', 'PF00281', 'TIGR00432', 'PF03764', 'PF04561', 'PF01191', 'PF01849', 'PF00750', 'PF00366', 'PF03484', 'TIGR00398', 'PF00164', 'PF01194', 'PF03874', 'PF01092', 'PF00749', 'PF00416', 'PF00673', 'PF01866', 'PF04919', 'PF05221', 'PF04563', 'PF00238', 'TIGR00336', 'PF04560', 'PF03947', 'PF06026', 'PF00181', 'PF00333', 'TIGR02153', 'TIGR01018', 'PF08069', 'PF00861', 'TIGR00442', 'PF00347', 'PF01725', 'PF05746'}
# ==============================================================================
# Annotation
# ==============================================================================
# Convert orfs 2 contigs
def fraggenescan_orfs_to_contigs(x, mode=0, func_parse = lambda x:"_".join(x.split("_")[:-3]), name="contig_id"):
    """
    mode == 0: single
    mode == 1: non-redundant list of contigs
    mode == 2: mapping of orf 2 contig
    """
    if mode == 0:
        return func_parse(x)
    if mode == 1:
        return np.unique([*map(func_parse, x)])
    if mode == 2:
        return pd.Series(dict_zip(x, [*map(func_parse, x)]), name=name)
        
def search_taxonomy(taxon_id, name=None, translate_ids=True, ncbi=None):
    """
    Input: Taxonomy ID
    Output: pd.Series of rank and taxonomy labels
    """
    if ncbi is None:
        ncbi = ete3.NCBITaxa()
    if pd.isnull(taxon_id):
        return pd.Series([],name=name)

    # JCVI-specific IDs
    if taxon_id < 0:
        return pd.Series([],name=name)

    else:
        taxon_id = int(taxon_id)
        if name is None:
            name = taxon_id
        lineage = ncbi.get_lineage(taxon_id)
        ranks = dict(filter(lambda x: x[1] != "no rank", ncbi.get_rank(lineage).items()))
        Se_taxonomy = pd.Series(ncbi.get_taxid_translator(ranks.keys()), name=name)
        if translate_ids:
            Se_taxonomy.index = Se_taxonomy.index.map(lambda x:ranks[x])
        return Se_taxonomy

def search_taxonomy_from_annotation(df_annotation):
    """
    Input: Rap output
    Output: pd.DataFrame of taxonomy mapping
    """
#     parallel_results = Parallel(n_jobs=n_jobs)(delayed(search_taxonomy)(taxon_id=Se_data["best_hit_taxon_id"], name=orf_id, translate_ids=True) for orf_id, Se_data in tqdm(df_annotation.iterrows()))
#     return pd.DataFrame(parallel_results)
    results = list()
    ncbi = ete3.NCBITaxa()

    d_id_data = dict()

    for orf_id, Se_data in tqdm(df_annotation.iterrows()):
        taxon_id = Se_data["best_hit_taxon_id"]
        if taxon_id not in d_id_data:
            Se_taxonomy = search_taxonomy(taxon_id=taxon_id, name=None, translate_ids=True, ncbi=ncbi)
            d_id_data[taxon_id] = Se_taxonomy
        else:
            Se_taxonomy = d_id_data[taxon_id].copy()
        Se_taxonomy.name = orf_id
        results.append(Se_taxonomy)
    return pd.DataFrame(results)

def add_taxonomy_to_annotation(df_annotation,):
    """
    Input: Rap output
    Output: Concatenated annotation and taxonomy
    """
    df_taxonomy = search_taxonomy_from_annotation(df_annotation)
    return pd.concat([df_annotation, df_taxonomy], axis=1)

def taxonomy_table(species, ncbi=None):
    """
    `species` should be either a pd.Series with index as bin_ids and value as species ID (or dict representation)
    Should depreacate this...
    """
    if ncbi is None:
        ncbi = ete3.NCBITaxa()
    if is_dict(species):
        species = pd.Series(species, name="species")
    Se_taxon_id = pd.Series(ncbi.get_name_translator(species), name="taxon_id").map(lambda x:x[0])
    df_taxonomy = Se_taxon_id.apply(lambda x:search_taxonomy(x, ncbi=ncbi)).loc[:,[ "phylum", "class", "order", "family", "genus", "species"]]
    df_taxonomy.insert(loc=0, column="taxon_id", value=Se_taxon_id)
    return species.apply(lambda x: df_taxonomy.loc[x,:])

# Add contigs to annotations
def add_contigs_to_annotation(df_annotation, contig_label="contig_id", insert_loc=0):
    df_annotation = df_annotation.copy()
    df_annotation.insert(loc=insert_loc, column=contig_label, value = [*map(fraggenescan_orfs_to_contigs, df_annot.index)])
    return df_annotation
    
# Add bins/clusters to annotations
def add_bins_to_annotation(df_annotation, bin_mapping=None, bin_label="bin_id", assembler=None, insert_loc=1):
    """
    bin_mapping ==> dict {contig_id:bin_id}
    """
    df_annotation = df_annotation.copy()
    # Bin mapping
    if bin_mapping is not None:
        bin_mapping = pd.Series(bin_mapping)
        condition_A = type(bin_mapping[0]) == str # Is it a string label?
        condition_B = is_number(bin_mapping[0], np.int) # Is it an int label?
        # {x:a, b:p} ==> {a:[x,y,z], b:[p,q,r]}
        if any([condition_A, condition_B]):
            bin_mapping = pd.Series(series_collapse(bin_mapping))
        # Convert contigs_ids to orfs_ids
        D_orf_bin = dict()
        for bin_id, query_contigs in bin_mapping.iteritems():
            orf_ids = subset_orfs_from_contigs(query_contigs=query_contigs, orf_ids=df_annotation.index, assembler=assembler)
            D_orf_bin.update(dict_zip(orf_ids, len(orf_ids)*[bin_id]))
        # Add bin column
        df_annotation.insert(loc=insert_loc, column=bin_label, value= pd.Series(D_orf_bin))
    return df_annotation

# Infer taxonomy for contigs or bins from annotations
def infer_taxonomy(df_annotation, group_column="best_hit_species", core=None, mode="contigs", bin_label="bin_id", bin_mapping=None, assembler=None, parse_func=lambda orf_id:"_".join(orf_id.split("_")[:-3]), progressbar=False):
    """
    bin_mapping ==> dict {contig_id:bin_id}
    """
    # Initialise
    df_annotation = df_annotation.copy()
    if mode in {0,1}:
        if mode == 0: mode = "contigs"
        if mode == 1: mode = "bins"
        print("DEPRECATED: Use `mode='contigs'`` for `mode=0` and `mode='bins'` for `mode=1`", file=sys.stderr)
    # Bin mapping
    if bin_mapping is not None:
        df_annotation = add_bins_to_annotation(df_annotation, bin_mapping=bin_mapping, bin_label=bin_label, assembler=assembler)

    # Core domains
    if core:
        if core.lower() == "bacteria": core_hmms = core_bacteria
        if core.lower() == "archaea": core_hmms = core_archaea
        if core.lower() == "all": core_hmms = set(core_bacteria) | set(core_archaea)

    D_x_taxa = defaultdict(lambda: defaultdict(float))
    D_x_num = defaultdict(int)

    if progressbar == True:
        iterable = tqdm(df_annotation.iterrows())
    if progressbar == False:
        iterable = df_annotation.iterrows()
    for orf_id, Se_annotation in iterable:
        taxon = Se_annotation[group_column]
        weight = Se_annotation["best_hit_percent_identity"]/100

        if any(pd.isnull(field) for field in [taxon,weight]) == False:
            # May have to change this for every dataset's contig mapping
            if mode == "contigs": x = parse_func(orf_id)
            if mode == "bins": x = Se_annotation[bin_label]

            # Custom HMM detection
            if core is not None:
            # Searchable row
                str_row = "\t".join(Se_annotation.astype(str))
                # Check for HMMs in query ORF
                for query_hmm in core_hmms:
                    if query_hmm in str_row:
                        # Add weights
                        D_x_taxa[x][taxon] += weight
                        D_x_num[x] += 1
                        break
            # Consider all ORFs [Default]
            if core is None:
                D_x_taxa[x][taxon] += weight
                D_x_num[x] += 1

    # Significance scores
    taxa_table = list()
    for x, D_taxon_weight in D_x_taxa.items():
        Se_taxa = pd.Series(D_taxon_weight).dropna()
        taxa_table.append([x, Se_taxa.argmax(), Se_taxa.max()/Se_taxa.sum(), str(list(D_taxon_weight.items()))])

    if mode == "contigs":
        df_taxon = pd.DataFrame(taxa_table, columns=["contig_id", "predicted_taxon", "score", "taxa"]).set_index("contig_id", drop=True)
    if mode == "bins":
        df_taxon =  pd.DataFrame(taxa_table, columns=[bin_label, "predicted_taxon", "score", "taxa"]).set_index(bin_label, drop=True)
    df_taxon["score"] = df_taxon["score"].astype(float)
    df_taxon.insert(loc=2, column="num_orfs", value=pd.Series(D_x_num).astype(int))

    return df_taxon
