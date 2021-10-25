source activate soothsayer_env
SUBMODEL=$1 # The same command was run for all sub-models
N_JOBS=$2
VERSION="v5.0"
NAME="${VERSION}_${SUBMODEL}"
OUT_DIR="${SUBMODEL}/clairvoyance_output"
X="${SUBMODEL}/X.pbz2"
y="${SUBMODEL}/y.pbz2"
cv="${SUBMODEL}/cv.tsv"

python ~/soothsayer/standalone/soothsayer_clairvoyance.py -X ${X} -y ${y} --cv ${cv} --model_type logistic,tree --n_iter 500 --name ${NAME} --min_threshold None,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --percentiles 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99 --method bruteforce --early_stopping 100 --out_dir ${OUT_DIR} --attr_type edge --class_type status --n_jobs ${N_JOBS} --save_model False
