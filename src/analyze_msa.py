
import sys
import os
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import argparse

def get_entropy_scores(line_set):
    aln_results = []
    for seq in line_set:
        if "*" not in seq:
            aln_results.append(seq.split()[1])
    assert all(len(x) == len(aln_results[0]) for x in aln_results), "all alignments should be equal length"

    length = len(aln_results[0])
    results = []
    for i in range(length):
        # Get all the values in a column of MSA
        entries = [x[i] for x in aln_results]

        # Get the probability of each element, and compute entropy
        values, value_counts = np.unique(entries, return_counts=True)
        probs = value_counts/len(entries)

        entropy = 0.0
        for i in probs:
            entropy -= i * math.log(i)

        # Store the entroy of each column
        results.append(entropy)
    return results

def generate_plots(msa_file, final_results):
    x = [x for x in range(1, len(final_results)+1)]
    plt.bar(x, height=final_results, width=1.0)
    plt.xlabel("Base Position in DNA Sequence")
    plt.ylabel("Shannon Entropy")
    plt.savefig(msa_file + ".png", dpi=500, bbox_inches="tight")

    plt.figure()
    rolling_avg = np.convolve(final_results, np.ones(250)/250, mode='valid')
    x = [x for x in range(1, len(rolling_avg)+1)]
    plt.bar(x, height=rolling_avg, width=1.0)
    plt.xlabel("Base Position in DNA Sequence")
    plt.ylabel("Avg Shannon Entropy (over 250 bp windows)")
    plt.savefig(msa_file + ".rolling.png", dpi=500, bbox_inches="tight")

def extract_sequences(roll_entropy_list, msa_file, each_genome_dict, num_to_extract, output_dir):
    """ Extract the different sections of the Enterovirus genomes """

    # Step 1: Identify the x-positions that we want to cut the MSA
    start = next(i for i in range(500, len(roll_entropy_list)) if roll_entropy_list[i] > 0.35)
    middle = next(i for i in range(3000, len(roll_entropy_list)) if roll_entropy_list[i] <= 0.35)
    end = next(i for i in range(5000, len(roll_entropy_list)) if roll_entropy_list[i] > 0.35)

    # Step 2: Make the plot showing the cut positions
    plt.figure()
    x = [x for x in range(1, len(roll_entropy_list)+1)]
    plt.bar(x, height=roll_entropy_list, width=1.0)
    plt.axvline(x=start, color='red', linestyle="dashed")
    plt.axvline(x=middle, color='red', linestyle="dashed")
    plt.axvline(x=end, color='red', linestyle="dashed")
    plt.axhline(y=0.20, color='black', linestyle="solid")
    plt.xlabel("Base Position in DNA Sequence")
    plt.ylabel("Avg Shannon Entropy (over 250 bp windows)")
    plt.savefig(msa_file + ".rolling_with_cuts.png", dpi=500, bbox_inches="tight")

    # Step 3: Extract each sequence in each region for each genome
    extract_str = lambda a, b, key: "".join([ch for ch in each_genome_dict[key][a:b] if ch != "-"])
    num_to_extract = min(max(1, num_to_extract), len(each_genome_dict))

    for i, key in enumerate(each_genome_dict.keys()):
        # Extract sequences
        left = extract_str(start, middle, key)
        right = extract_str(middle, end, key)

        # Make sure the number of gaps plus remaining sequence length equals total length
        assert each_genome_dict[key][middle:end].count("-") + len(right) == (end-middle)
        assert each_genome_dict[key][start:middle].count("-") + len(left) == (middle-start)

        # Print each section of the genome
        with open(output_dir+f"seq_{i}_left.fna", "w") as out_fd:
            out_fd.write(f">seq_{i}_left\n{left}\n")
        with open(output_dir+f"seq_{i}_right.fna", "w") as out_fd:
            out_fd.write(f">seq_{i}_right\n{right}\n")
        
        if i >= num_to_extract-1:
            break

def main(args):
    """ Parse the MSA file and process it """

    msa_file = args.msa_file
    with open(msa_file, "r") as in_fd:
        all_msa_lines = in_fd.readlines()

    curr_line_set = []
    final_results = []
    each_genome_dict = {}
    for i, line in enumerate(all_msa_lines[3:]):
        line_split = line.split()
        # Alignment line in the file
        if len(line) > 1 and len(line.split()) == 2: # alignment line
            curr_line_set.append(line.strip())
            # Keep track of alignment for each genome
            if line_split[0] not in each_genome_dict and "*" not in line:
                each_genome_dict[line_split[0]] = line_split[1]
            elif line_split[0] in each_genome_dict:
                each_genome_dict[line_split[0]] += line_split[1]
        # Process a block of alignment lines after passing it
        elif len(curr_line_set) > 0:
            final_results += get_entropy_scores(curr_line_set)
            curr_line_set = []
    
    # Make sure each alignment is of equal length
    assert len(set([len(each_genome_dict[x]) for x in each_genome_dict.keys()])) == 1
    
    if args.plots:
        generate_plots(msa_file, final_results)
    if args.extract:
        rolling_list = np.convolve(final_results, np.ones(250)/250, mode='valid')
        extract_sequences(rolling_list, msa_file, each_genome_dict, args.num_to_extract, args.output_dir)


def parse_arguments():
    """ Parse the command-line arguments """
    parser = argparse.ArgumentParser(description="Take in a *.aln file and analyze it")
    parser.add_argument("-i", dest="msa_file", help="path to *.aln file", required=True)
    parser.add_argument("--plots", action="store_true", help="make analysis plots for multiple sequence alignment", default=False)
    parser.add_argument("--extract", action="store_true", help="extract sequences from each sequence", default=False)
    parser.add_argument("-n", dest="num_to_extract", help="number of sequences to extract", default=0, type=int)
    parser.add_argument("-o", dest="output_dir", help="path to output directory", default="")
    args = parser.parse_args()
    return args

def check_arguments(args):
    """ Verifies certain aspects about the input parameters """
    if not os.path.isfile(args.msa_file):
        print("Error: the provided *.aln does not exist.")
        exit(-1)
    if args.extract:
        if not os.path.isdir(args.output_dir):
            print("Error: the output directory path is not valid.")
            exit(1)
        else:
            if "/" != args.output_dir[-1]:
                args.output_dir += "/"
    
if __name__ == "__main__":
    args = parse_arguments()
    check_arguments(args)
    main(args)

# ### Main Method Code
# msa_file = sys.argv[1]
# with open(msa_file, "r") as in_fd:
#     all_msa_lines = in_fd.readlines()

# curr_line_set = []
# final_results = []
# for i, line in enumerate(all_msa_lines[3:]):
#     if len(line) > 1 and len(line.split()) == 2: # alignment line
#         curr_line_set.append(line.strip())
#     elif len(curr_line_set) > 0:
#         final_results += get_entropy_scores(curr_line_set)
#         curr_line_set = []

# print(len(all_msa_lines))
# exit(0)

# ### Plotting
# x = [x for x in range(1, len(final_results)+1)]
# plt.bar(x, height=final_results, width=1.0)
# plt.xlabel("Base Position in Protein Sequence")
# plt.ylabel("Shannon Entropy")
# plt.savefig(msa_file + ".png", dpi=500, bbox_inches="tight")

# plt.figure()
# rolling_avg = np.convolve(final_results, np.ones(250)/250, mode='valid')
# x = [x for x in range(1, len(rolling_avg)+1)]
# plt.bar(x, height=rolling_avg, width=1.0)
# plt.xlabel("Base Position in Protein Sequence")
# plt.ylabel("Avg Shannon Entropy (over 250 aa windows)")
# plt.savefig(msa_file + ".rolling.png", dpi=500, bbox_inches="tight")




