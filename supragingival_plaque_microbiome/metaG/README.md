-------------
#### References
-------------   
These include the scripts used in the following manuscripts:
  * **Josh L. Espinoza**, Derek M. Harkins, Manolito Torralba, Andres Gomez, Sarah K. Highlander, Marcus B. Jones, Pamela Leong, Richard Saffery, Michelle Bockmann, Claire Kuelbs, Jason M. Inman, Toby Hughes, Jeffrey M. Craig, Karen E. Nelson, Chris L. Dupont. mBio Nov 2018, 9 (6) e01631-18; [doi: 10.1128/mBio.01631-18](https://mbio.asm.org/content/9/6/e01631-18)

  * Gomez A*, **Espinoza JL***, Harkins DM, Leong P, Saffery R, Bockmann M, Torralba M, 
Kuelbs C, Kodukula R, Inman J, Hughes T, Craig JM, Highlander SK, Jones MB,
Dupont CL, Nelson KE. *Host Genetic Control of the Oral Microbiome in Health and
Disease*. Cell Host Microbe. 2017 Sep 13;22(3):269-278.e3. [doi:
10.1016/j.chom.2017.08.013](https://doi.org/10.1016/j.chom.2017.08.013)
-------------
#### Software Versions & Dependencies
-------------
```
Python = 3.6.4 
Pandas = 0.22
NumPy = 1.13.3
ete3 = 3.1.1
tqdm = 4.19.5
joblib = 0.11
BioPython = 1.70
```
------------
#### Content
------------
`inferring_taxonomy.py` has functions for inferring taxonomy from orf annotations and converting taxonomy identifiers to various taxonomic levels.


`phylogenomically_binned_functional_potential.py` has functions for reading in fasta files, normalizing counts to TPM, grouping contig counts by bins, organizing results from [MAPLE-2.3.0](https://www.genome.jp/tools/maple/), and calculating phylogenomically binned functional potential.

**PLEASE REFER TO [Soothsayer](https://github.com/jolespin/soothsayer) FOR UPDATED IMPLEMENTATIONS**

------------
#### Creator
------------
Josh L. Espinoza
