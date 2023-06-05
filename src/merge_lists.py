#!/usr/bin/env python3

# Name: merge_lists.py
# Description: This is a python script is used by experiment 4 in order to merge
#              the kmer lists from kmc3 in order to build a confusion matrix.
#
# Date: June 14th, 2022

import argparse
import os
import numpy as np
import random

def build_dictionary(pivot_path):
    """ Builds dictionary of pivot kmers with kmer_dict[key][0] as count """
    kmer_dict = {}
    with open(pivot_path, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            kmer = line.split()[0]
            count = line.split()[1]
            kmer_dict[kmer] = [int(count)]
    return kmer_dict

def update_dictionary(pivot_dict, intersect_path, intersect_num, num_datasets):
    """ Updates dictionary with intersect kmers """
    with open(intersect_path, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            kmer = line.split()[0]
            if kmer in pivot_dict:
                pivot_dict[kmer].append(intersect_num % num_datasets)
    return pivot_dict

def calculate_accuracy_values(confusion_matrix, num_datasets):
    """ Calculates accuracy values given confusion matrix [TP TN FP FN] """
    accuracies = []
    for pivot in range(num_datasets):
        tp = confusion_matrix[pivot][pivot]
        fp = fn = tn = 0
        for row in range(num_datasets):
            for column in range(num_datasets + 1):
                curr = confusion_matrix[row][column]
                if column == pivot and row != pivot:
                    fp += curr
                elif row == pivot and column != pivot:
                    fn += curr
                elif row != pivot:
                    tn += curr
        accuracies.append([args.k,pivot,tp,tn,fp,fn])
    return accuracies

def process_read_into_kmers(read, k):
    """ Split read into kmers and return it """
    kmer_list = []
    for i in range(0, len(read)-k+1):
        kmer_list.append(read[i:i+k])
    return kmer_list

def get_canonical_kmer(kmer):
    """ Return the canonical form of a given kmer """
    # Generate reverse complement ...
    comp = ""
    rev_comp_dict = {'A':'T', 'C':'G', 'G':'C', 'T':'A'}
    for ch in kmer:
        if ch in rev_comp_dict:
            comp += rev_comp_dict[ch]
        else:
            return ""
    rev_comp = comp[::-1]

    # Return the smaller of the two kmers ...
    if kmer < rev_comp:
        return kmer
    else:
        return rev_comp
    
def main(args):
    """ main method for the script """

    # Read in all the file paths for text dumps that will
    # be needed of either the pivot or intersection databases
    pivot_files = []
    intersect_files = [] 
    with open(args.pivot_filelist, "r") as input_fd:
        pivot_files = [x.strip() for x in input_fd.readlines()]
    with open(args.intersect_list, "r") as input_fd:
        intersect_files = [x.strip() for x in input_fd.readlines()]

    # Verify each of the text dump file in the file_lists is valid
    for file_path in (pivot_files + intersect_files):
        if not os.path.isfile(file_path):
            print(f"Error: At least one of the file paths in the file lists is not valid ({file_path})")
            exit(1)
    print("[log] all file paths are valid, now will start to merge the lists.")

    # Determine how analysis will be done ...
    if args.use_feat_level:
        print("[log] analysis will be done at feature-level")
    else:
        print("[log] analysis will be done at the read-level")
    print()

    # Define the confusion matrix (last column represents kmers only in pivot)
    num_datasets = args.num_datasets
    confusion_matrix = [[0 for i in range(num_datasets + 1)] for j in range(num_datasets)]

    # This version contains an additional column used to store kmers that are not 
    # present in any database
    confusion_matrix_with_ucol = [[0 for i in range(num_datasets + 1)] for j in range(num_datasets)]

    # Fill in the confusion matrix from the upper-left to bottom right ...
    curr_pivot_num = 0
    curr_intersect_num = 0

    # Loop starts at first row meaning first pivot
    for curr_pivot_file in pivot_files:
        print("[log] processing pivot {pivot} with k = {k}".format(pivot = curr_pivot_num + 1, k = args.k))

        # Build dictionary of kmers -> each kmer will point to list with 
        # count and list of databases it occurs in
        curr_pivot_dict = build_dictionary(curr_pivot_file)
        for i in range(num_datasets):
            curr_pivot_dict = update_dictionary(curr_pivot_dict, intersect_files[curr_intersect_num], curr_intersect_num, num_datasets)
            curr_intersect_num += 1 

        # Determine how many kmers only occur in the pivot (none of the databases)   
        unique_pivot_count = 0
        for kmer in curr_pivot_dict:
            if len(curr_pivot_dict[kmer]) == 1:
                unique_pivot_count += curr_pivot_dict[kmer][0]


        # Fill in the current row of the confusion matrix (depends on whether we
        # want to do analysis at read-level or feature-level)

        if args.use_feat_level: # feature-level
            for kmer in curr_pivot_dict:
                count = curr_pivot_dict[kmer][0]
                matches = curr_pivot_dict[kmer][1:]
                for match in matches:
                    confusion_matrix[curr_pivot_num][match] += 1 / len(matches) * count
                    confusion_matrix_with_ucol[curr_pivot_num][match] += 1 / len(matches) * count

            # For "regular" confusion matrix, spread the unlabeled kmers across 
            # all the columns.
            confusion_matrix[curr_pivot_num][num_datasets] = 0
            for i in range(num_datasets):
                confusion_matrix[curr_pivot_num][i] += 1 / num_datasets * unique_pivot_count

            # For the confusion matrix with the unidentified column empty, set it to zero
            confusion_matrix_with_ucol[curr_pivot_num][num_datasets] = 0

        else: # read-level
            file_path = f"{args.read_level[0]}pivot_{(curr_pivot_num+1)}.fa"
            with open(file_path, "r") as in_fd:
                all_lines = in_fd.readlines()
                count = 0
                for line in all_lines:
                    if ">" not in line:
                        # Grab all kmers from current read
                        kmer_list = process_read_into_kmers(line.strip(), int(args.k))
                        votes = [0 for i in range(num_datasets)]
                        votes_with_ucol = [0 for i in range(num_datasets)]

                        # Process the reads kmers to determine the read's classification
                        count = 0
                        for kmer in kmer_list:
                            canon_kmer = get_canonical_kmer(kmer)
                            
                            # If kmer is "", means it contains a non-ACGT kmer so KMC ignores it
                            # so we will ignore it as well
                            if canon_kmer != "":
                                matches = curr_pivot_dict[canon_kmer][1:]
                                if len(matches) > 0: # Occurs in at least 1 database
                                    for match in matches:
                                        votes[match] += 1/len(matches)
                                        votes_with_ucol[match] += 1/len(matches)
                                elif len(matches) == 0: # Doesn't occur, so we smear weight across databases
                                    for i in range(num_datasets):
                                        votes_with_ucol[i] += 1.0/num_datasets
                                    count += 1
                        
                        # Determine which class is the prediction and update confusion matrix...
                        votes_np = np.array(votes)
                        max_indexes = np.where(votes_np == max(votes))[0]
                        class_decision = random.choice(max_indexes)

                        confusion_matrix[curr_pivot_num][class_decision] += 1
                        confusion_matrix_with_ucol[curr_pivot_num][class_decision] += 1
            
        # Increment the counter ...
        curr_pivot_num += 1 
    
    # Print out "regular" confusion matrix to a csv file
    output_matrix = args.output_path + "confusion_matrix/k_"+ args.k +"_confusion_matrix.txt"
    with open(output_matrix,"w+") as csvfile:
        for row in confusion_matrix:
            csvfile.write(",".join([str(x) for x in row]) + "\n")

    # Print out confusion matrix with unidentified to a csv file
    output_matrix = args.output_path + "confusion_matrix/k_"+ args.k +"_confusion_matrix_with_unidentified.txt"
    with open(output_matrix,"w+") as csvfile:
        for row in confusion_matrix_with_ucol:
            csvfile.write(",".join([str(x) for x in row]) + "\n")
    
    # Calculate confusion matrix summary values and print to csv file: 
    # All the metrics ending with -U represent the scenario where we
    # ignore the kmers in the last column.
    # Ex: k, pivot #, TP, TN, FP, FN, TP-U, TN-U, FP-U, FN-U,
    output_acc = args.output_path + "values/k_"+ args.k +"_accuracy_values.csv"
    with open(output_acc, "w+") as csvfile:
        scores = calculate_accuracy_values(confusion_matrix, num_datasets)
        scores_with_last_col = calculate_accuracy_values(confusion_matrix_with_ucol, num_datasets)

        for chunk_1, chunk_2 in zip(scores, scores_with_last_col):
            csvfile.write(",".join([str(x) for x in chunk_1]) + "," + ",".join([str(x) for x in chunk_2[2:]]) + "\n")

def parse_arguments():
    """ Defines the command-line argument parser, and return arguments """
    parser = argparse.ArgumentParser(description="This script helps to analyze the data in experiment 4 in order to"
                                                 "merge the kmer-lists to form a confusion matrix.")
    parser.add_argument("-n", "--num", dest="num_datasets", required=True, help="number of datasets in this experiment", type=int)
    parser.add_argument("-p", "--pivot_list", dest="pivot_filelist", required=True, help="path to text file with a list of pivot kmer lists")
    parser.add_argument("-i", "--intersect_list", dest="intersect_list", required=True, help="path to text file with a list of kmer lists representing the intersections")
    parser.add_argument("-o", "--output_path", dest = "output_path", required=True, help = "path to output directory") # Match snakemake format
    parser.add_argument("-k", "--k_value", dest = "k", required = True, help = "k value for this experiment")
    parser.add_argument("-r", "--read-level", dest="read_level", help="perform analysis at read-level (default: feature-level)", nargs="*", required=True)
    parser.add_argument("--use-feat-level", dest="use_feat_level", action="store_true", default=False, help="turn this on if you want to do classification at feature-level")
    args = parser.parse_args()
    return args

def check_arguments(args):
    """ Checks the arguments to make sure they are valid """

    check_path(args.pivot_filelist)
    check_path(args.intersect_list)

    if args.num_datasets <= 0:
        print("Error: The number of datasets needs to be positive integer.")
        exit(1)
    
    if args.read_level is not None and not os.path.isdir(args.read_level[0]):
        print("Error: the path provided for read-level analysis is not valid.")
        exit(1)

def check_path(file):
    if not os.path.isfile(file):
        print("Error: One of the provided files is not valid: "+file)
        exit(1)

if __name__ == "__main__":
    args = parse_arguments()
    check_arguments(args)
    main(args)
