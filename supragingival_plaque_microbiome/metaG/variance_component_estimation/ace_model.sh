#!/bin/bash
# =========================================
# v4.0
# =========================================
# Author: Josh L. Espinoza
# June 08, 2017
# Edits: Feburary 26, 2018
# J. Craig Venter Institute
# =========================================
# Usage: bash ace_model.sh path/to/metadata.tsv path/to/attributes.tsv > output.tsv
# ========================================
# Load in the file for heritablility estimates. Can be gzipped but doesn't have to be compressed.
# We recommend the following transformation, where X is the data (row=samples, col=attributes)
# relative_abundance(X + 1) | glog(, a=1, InverseQ=False) | zscore(, axis=0) > transformed_data
# Can also use any other log (add pseudocount if there are entries with 0)

# Metadata File Input Format:
# FamilyID	Center	Age	Sex	Zygosity	Caries
# 3041.2	3041	MCRI	-1.464374	M	DZ	Diseased
# 3041.1	3041	MCRI	-1.464374	M	DZ	Diseased
# 1067.1	1067	MCRI	-1.090803	F	MZ	Non-diseased
# 1067.2	1067	MCRI	-1.090803	F	MZ	Diseased
# 1072.2	1072	MCRI	-1.090803	F	DZ	Non-diseased

# Attribute File Input Format for Data:
#            NODE_1	     NODE_2	     NODE_3
# 1048.2	-1.077315	-1.213378	-0.609659
# 3038.1	0.800843	-0.077068	0.729418
# 3038.2 	0.005978	0.283853	1.090050
# 2031.2 	-1.339211	-0.308153	0.885450
# 3041.1 	-1.130427	-0.854302	-1.227843

# Operations:
# [Python: Relabels attribute matrix to be compatible, removes subjects if they do not have a twin, and merge metadata and attribute matrix] -> [R: run linear model in R] -> [Python: parse output]
# Creates and deletes temporary file ./.G614C48A46358799GT2788926A9A54146

# ========================================
# Future
# =========================================
# [DONE] Make a script that automatically goes through the attributes and relabels them, stores the mapping, then adjusts the labels once more at the end.
# Make the Rscript parse the headers to incude a variable number of extra attributes (e.g. sex, caries, etc.)
# Make sure that no attribute is constant (e.g. all 1 or all 0)
# =========================================
# Dependencies
# =========================================
# R: mets version 1.2.2
# https://cran.r-project.org/web/packages/mets/index.html
# Python 3: pandas, numpy
# =========================================
META_FILE=$1
ATTR_FILE=$2
DIR_SCRIPT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# SEED=$RANDOM
TOKEN=$(openssl rand -hex 12)

# START
NOW=$(date +"%T")
>&2 echo "Start Time: $NOW"
>&2 echo "Directory: $PWD"
>&2 echo "Token: $TOKEN"
mkdir -p "./.tmp_twinlm"
python $DIR_SCRIPT/scripts/preprocess_data.py --metadata $META_FILE --attribute_matrix $ATTR_FILE --token $TOKEN | Rscript $DIR_SCRIPT/scripts/run_twinlm.r $TOKEN
python $DIR_SCRIPT/scripts/parse_twinlm.py --token $TOKEN
# End
NOW=$(date +"%T")
>&2 echo "End Time: $NOW"
