#!/usr/bin/env python3

# Name: extract_mems.py
# Description: This script is meant to take a FASTA file for which
#              matching statistics have been computed for and then
#              extract either the half-mems or mems and output
#              them to stdout.
# Date: June 21, 2022

import argparse
import os

def print_out_half_mems(curr_seq, lengths_array, curr_seq_id, threshold, out_file):
    """ Given a pattern and lengths array (matching statistics), print out half mems to stdout """
    for i in range(len(curr_seq)):
        curr_half_mem_length = lengths_array[i]
        if curr_half_mem_length >= threshold:
            random_quality = "#" * curr_half_mem_length
            curr_half_mem = curr_seq[i:i+curr_half_mem_length]

            out_file.write(f">half_mem_{curr_seq_id}\n{curr_half_mem}\n+\n{random_quality}\n")
            curr_seq_id += 1
    return curr_seq_id

def main(args):
    """ Extracts either the half-mems/mems from a set of patterns matched to database """
    pattern_fd = open(args.pattern_file, "r")
    length_fd = open(args.length_file, "r")
    out_fd = open(args.output_file, "w")

    # Verify the number of sequences in each file is the same
    pattern_lines = [x.strip() for x in pattern_fd.readlines()]
    length_lines = length_fd.readlines()

    seq_count = [0, 0]
    for line in pattern_lines:
        if ">" in line:
            seq_count[0] += 1
    for line in length_lines:
        if ">" in line:
            seq_count[1] += 1
    assert seq_count[0] == seq_count[1], "assertion failed: mismatch in the number of sequences in each file."
    
    # Start to extract the requested output (half-mems or mems) and output to stdout
    seq_id = 0
    curr_half_mem_id = 0
    curr_seq = ""
    for line in pattern_lines:
        header_line = ('>' in line)
        if header_line and len(curr_seq) > 0:
            ## IMPORTANT: this is based on the assumption that SPUMONI outputs the lengths all on one line
            ##            so we know the full file will have (2 times the number of sequences) lines
            lengths_array = [int(x) for x in length_lines[(seq_id*2)+1].split()]
            assert len(lengths_array) == len(curr_seq), "assertion failed: mis-match in terms of the lengths of lengths array"

            if args.half_mems: # deal with half-MEMs
                curr_half_mem_id = print_out_half_mems(curr_seq, lengths_array, curr_half_mem_id, args.threshold, out_fd)
            else: # deal with MEMs
                raise NotImplementError()

            seq_id += 1
            curr_seq = ""
        elif not header_line:
            curr_seq += line.strip()
    
    pattern_fd.close()
    length_fd.close()
    out_fd.close()

def parse_arguments():
    """ Parses the command-line arguments, and returns them """
    parser = argparse.ArgumentParser(description="Takes in matching statistic lengths, and extracts either half-mems or mems, prints to stdout.")
    parser.add_argument("-l", dest="length_file", help="path to matching statistic length file", required=True)
    parser.add_argument("-p", dest="pattern_file", help="path to pattern file that correspond to the matching statistics.", required=True)
    parser.add_argument("--half-mems", action="store_true", default=False, dest="half_mems", help="extract half-mems and print to stdout")
    parser.add_argument("--mems", action="store_true", default=False, dest="mems", help="extract mems and print to stdout (it can either be mems or half-mems, not both)")
    parser.add_argument("-t", dest="threshold", default=0, type=int, help="threshold for half-mem length to use")
    parser.add_argument("-o", dest="output_file", help="path to output file", required=True)
    args = parser.parse_args()
    return args

def check_args(args):
    """ Checks the command-line arguments and makes sure they are valid """

    # Verify the files exist
    if not os.path.isfile(args.length_file):
        print("Error: this matching statistic length file does not exist.")
        exit(1)
    if not os.path.isfile(args.pattern_file):
        print("Error: this pattern file does not exist.")
        exit(1)
    
    # Verify the files are the right type
    if not args.length_file.endswith(".lengths"):
        print("Error: the lengths file has the incorrect file extension.")
        exit(1)
    if not args.pattern_file.endswith(".fa") and not args.pattern_file.endswith(".fna"):
        print("Error: the pattern file has the incorrect file extension.")
        exit(1)
    
    # Verify only one of the options are chosen: half-mems or mems
    if (args.half_mems and args.mems) or (not args.half_mems and not args.mems):
        print("Error: exactly one type needs to be chosen (--half-mems or --mems)")
        exit(1)
    
    # Verify if using half-mems, that we specified a threshold
    if args.half_mems and args.threshold == 0:
        print("Error: need to specify a threshold if using half-mems")
        exit(1)
    if args.half_mems and args.threshold < 0:
        print("Error: need to specify a positive value as the threshold")
        exit(1)

if __name__ == "__main__":
    args = parse_arguments()
    check_args(args)
    main(args)