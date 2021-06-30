#!/Users/jespinoz/anaconda/bin/Rscript
# =========================================
# Author: Josh L. Espinoza
# June 08 2017
# Edits: August 17 2018
# J. Craig Venter Institute
# Adapted from code by Andres Gomez
# =========================================
# Usage: Rscript run_twinlm.r input.tsv.gz output.out
# ========================================
# Load in the file for heritablility estimates. Can be gzipped but doesn't have to be compressed.
# We recommend the following transformation, where X is the data (row=samples, col=attributes)
# relative_abundance(X + 1) | glog(, a=1, InverseQ=False) | zscore(, axis=0) > transformed_data
# Can also use any other log (add pseudocount if there are entries with 0)
# NOTE:
# For some reason, mets requires that attributes to be in all uppercase, separated by _, and followed by a digit (I think starting at 1 but maybe not)
# Example below:
# PatientID FamilyID  num Sex Zygosity  Age  Caries     ATTR_1     ATTR_2     ATTR_3     ATTR_4
# 1054.1  1054    1   F  MZ   78    1 -0.371218  0.194067  0.410902  0.715052
# 1054.2  1054    2   F  MZ   78    1  0.702413  0.152505  1.217374  0.965222
# 1057.1  1057    1   M  DZ   76    1  0.175550  0.768296  1.280967  1.179333
# 1057.2  1057    2   M  DZ   76    2 -0.709408 -0.871837  1.028473  1.255074
# 1059.1  1059    1   M  DZ   66    2 -0.122607 -0.568323 -1.507364 -1.483020
# =========================================
# Dependencies
# =========================================
# Load `mets `package
# v2 prints to stdout
#> library(mets)
#Loading required package: timereg
#Loading required package: survival
#Loading required package: lava
#lava version 1.5
#mets version 1.2.2
# https://cran.r-project.org/web/packages/mets/index.html
# =========================================
require = function(x) {
  if (!base::require(x, character.only = TRUE)) {
  install.packages(x, dep = TRUE) ;
  base::require(x, character.only = TRUE)
  }
}
suppressMessages(require("mets"))
suppressMessages(library("mets"))
library(utils)

# Arguments

# Init args
args = commandArgs(trailingOnly=TRUE)

# Files
path_image  = paste("./.tmp_twinlm/", args[1], ".twinlm.RData", sep="")
path_intermediate = paste("./.tmp_twinlm/", args[1], ".twinlm.txt", sep="")
f_intermediate = file(path_intermediate, "w")

# Load data
df_data  = read.table(file("stdin"), sep="\t", row.names=1, header = TRUE, check.names=FALSE)

# Run TwinLM
write("Running linear models. .. ... ..... ....... ..........", stderr()) # Move the function outside of the lapply loop
run_model = function(x) {
    twinlm(as.formula(paste(x,"~", "Sex+Age+Caries+Center",sep=" ")), # <- You will have to adjust this accordingly to your metadata
           data=df_data,
           DZ="DZ",
           zyg="Zygosity",
           id="FamilyID",
           type="ace"
       )
   }

# Iterable of attributes
# twinlm_output = lapply(X=idx_attrs, FUN=run_model)

loc_start_attributes = 1
for (id_column in colnames(df_data)){
  if (startsWith(x = id_column, prefix = "ATTR")){
    break
  }
  loc_start_attributes = loc_start_attributes + 1
}
idx_attrs = as.array(colnames(df_data)[loc_start_attributes:ncol(df_data)])



# Run ACE Model for each attribute
progressbar   = txtProgressBar(1, length(idx_attrs), style=3, file=stderr())
twinlm_data = list()
i = 1
for (x in idx_attrs){
    results = run_model(x)
    twinlm_data[[i]] = results
    i = i + 1
    setTxtProgressBar(progressbar, i)
}

# Save enivornment
save.image(file=path_image)
cat(capture.output(print(twinlm_data), file=f_intermediate))
close(f_intermediate)

# Print results
# print(twinlm_data)

# Exit
q("no")
