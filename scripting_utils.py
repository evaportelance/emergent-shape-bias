import os

def summarize_hps(hps, output_dir):
    """
    hps: a list of dictionaries containing the relevant hps
    """
    line = "----------------------------------------"
    print(line)
    print("experiment hps")
    print(line)
    with open(os.path.join(output_dir, "hps.txt"), "w") as f :
        for d in hps:
            for k, v in d.items():
                print(k+": "+str(v))
                f.write(k+": "+str(v)+"\n")
    print(line) 