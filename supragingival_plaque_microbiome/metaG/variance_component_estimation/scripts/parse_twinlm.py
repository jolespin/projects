
import sys, os, re, itertools, subprocess, argparse, time
import pandas as pd
import numpy as np



def read_twinlm(path, mode=1, sep="|_|"):
    """
    mets (package) the function is twinlm
    """
    def parse_block_1(block_1):
        # Get data into array
        block_1_parsed = [*map(lambda x: [*filter(lambda y: len(y) > 0,
                                      x.replace("Std. Error", "Std_Err").replace("Z value", "Z_value").replace("<","").replace("e+00", "").split(" "))],
                              block_1.split("\n")[1:])]
        # Separate labels
        block_1_array = np.array(block_1_parsed[1:])
        # Labels
        attr_id = block_1_array[0,0]
        latent_variables = [*map(lambda x: x.split("~")[1], block_1_array[4:,0])]
        og_col_vars = ["Estimate",  "Std_Error", "Z_value", "Pr(>|z|)"] #block_1_parsed[0]
        og_row_vars = ["attr", "sd(A)", "sd(C)", "sd(E)"] + latent_variables
        # Flatten
        idx_block_1 = [*map(lambda x: sep.join(x), itertools.product(og_row_vars, og_col_vars))]
        block_1_flat = block_1_array[:,1:].ravel()
        return pd.Series(block_1_flat, index=idx_block_1, name=attr_id)

    def parse_block_2(block_2):
        block_2_flat = np.array([*map(lambda x: [*filter(lambda y: len(y) > 0, x.split(" "))], block_2.split("\n")[2:])])[:,1:].ravel()
        idx_block_2 = [*map(lambda x:sep.join(x), itertools.product(list("ACE"), ["Estimate", "ci(2.5)", "ci(97.5)"]))]
        return pd.Series(block_2_flat, index=idx_block_2)

    def parse_block_3(block_3):
        block_3 = block_3.replace("Correlation within MZ:", "corr(MZ)").replace("Correlation within DZ:", "corr(DZ)")
        block_3_flat = np.array([*map(lambda x: [*filter(lambda y: len(y) > 0, x.split(" "))], block_3.split("\n")[1:])])[:,1:].ravel()
        idx_block_3 = [*map(lambda x: sep.join(x), itertools.product(["corr(MZ)", "corr(DZ)"], ["Estimate", "ci(2.5)", "ci(97.5)"]))]
        return pd.Series(block_3_flat, index=idx_block_3)

    def parse_block_4(block_4):
        block_4_tmp = [*filter(lambda x: len(x) > 0, block_4.split("\n"))]
        llk, df_ = map(lambda x:x.replace(" ",""), block_4_tmp[0].split("' ")[1].split("(df="))
        df = df_[:-1]
        aic, bic = [*map(lambda x: x.split(": ")[1].replace(" ",""), block_4_tmp[1:])]
        return pd.Series([df, llk, aic, bic], index=["df", "llk", "AIC", "BIC"])

    def parse_pipeline(query_blocks, mode=mode):
        try:
            Se_block_1 = parse_block_1(query_blocks[0])
        except ValueError:
            print(query_blocks)
        Se_block_2 = pd.Series(parse_block_2(query_blocks[2]), name=Se_block_1.name)
        Se_block_3 = pd.Series(parse_block_3(query_blocks[4]), name=Se_block_1.name)
        Se_block_4 = pd.Series(parse_block_4(query_blocks[5]), name=Se_block_1.name)
        if mode == 0:
            return pd.concat([Se_block_4, Se_block_2, Se_block_3])
        if mode == 1:
            return pd.concat([Se_block_4, Se_block_2, Se_block_3, Se_block_1])


    compile_obj = re.compile("\[\[\d+\]\]")
    table = list()

    if path == sys.stdin:
        f = path
        file_close = False
    else:
        f = open(path)
        file_close = True
    with f:
        contents = f.read()
        matches = compile_obj.finditer(contents)
        idx_block = np.array([*map(lambda x: x.end(), matches)])
        # Iterate through the blocks
        for i in range(1, idx_block.size + 1):
            if i < idx_block.size:
                query_blocks = contents[idx_block[i-1]:idx_block[i]].split("\n\n")
            # Edgecase of last attribute
            else:
                query_blocks = contents[idx_block[-1]:].split("\n\n")
            try:
                Se_data = parse_pipeline(query_blocks)
                table.append(Se_data)
            except IndexError:
                pass
    if file_close:
        f.close()
    df_twinlm = pd.DataFrame(table)
    df_twinlm[(df_twinlm == "NA")] = np.nan
    return df_twinlm



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, help = 'Random hex string')

    opts = parser.parse_args()

    # token = "".join(np.random.RandomState(opts.seed).choice(list("123456789acgt"), size=28))

    path_hidden = f"./.tmp_twinlm/{opts.token}.encoding"
    time.sleep(3)
    with open(path_hidden, "r") as f:
        d_encode = pd.read_table(f, sep="\t", index_col=0).iloc[:,0].to_dict()
    d_decode = {v:k for k,v in d_encode.items()}
    df_twinlm = read_twinlm(f"./.tmp_twinlm/{opts.token}.twinlm.txt")
    df_twinlm.index = df_twinlm.index.map(lambda x:d_decode[x])

    print("\nCompleted parsing output", file=sys.stderr)
    idx_failed = list(set(list(d_encode.keys())) - set(df_twinlm.index))
    if len(idx_failed) > 0:
        path_failed = f"./.tmp_twinlm/{opts.token}.twinlm.failed.list"
        print(f"Warning: {len(idx_failed)} attributes failed the statistical tests and written in {path_failed}", file=sys.stderr)
        with open(path_failed, "w") as f:
            print(*idx_failed, sep="\n", file=f)
    # Output
    df_twinlm.to_csv(sys.stdout, sep="\t")
    # Exit
    sys.exit()
if __name__ == "__main__":
    main()
