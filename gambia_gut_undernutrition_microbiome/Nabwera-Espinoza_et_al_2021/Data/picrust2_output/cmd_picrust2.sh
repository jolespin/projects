COUNT=../../../Manuscripts/In_progress/GAMBIAGUT__multimodal-ssns_faecal/Notebooks/Data/otus.counts.filtered.prevalence_geq13.tsv
SEQ=../../../Manuscripts/In_progress/GAMBIAGUT__multimodal-ssns_faecal/Notebooks/Data/otus.counts.filtered.prevalence_geq13.fa
#place_seqs.py -s $SEQ -o output.nw -p 16 --intermediate intermediate/place_seqs
#hsp.py -i 16S -t output.nw -o marker_predicted_and_nsti.tsv.gz -n  -p 16 
hsp.py -i EC -t output.nw -o EC_predicted.tsv.gz -p 16
metagenome_pipeline.py -i ${COUNT} -m marker_predicted_and_nsti.tsv.gz -f EC_predicted.tsv.gz -o EC_metagenome_out --strat_out 


