# Name: merge_lists.py
# Description: This is a python script is used by experiment 5 in order to
#              read and analyze the SAM file generated by ri-index locate
#              to generate a confusion matrix
#
# Date: June 14th, 2022

from distutils.command.config import config
import os
import argparse
import pysam

def main(args):
    sam_file = pysam.AlignmentFile(args.sam_file, "r")
    refs_list = build_refs_list(args.ref_lists, args.num_datasets)

    confusion_matrix = [[0 for i in range(args.num_datasets)] for j in range(args.num_datasets)]
    
    print("\n[log] building a dictionary of the read alignments")
    read_mappings = {}
    for read in sam_file.fetch():
        dataset = find_class_of_reference_name(refs_list, read.reference_name)
        if read.query_name in read_mappings:
            read_mappings[read.query_name].append(dataset)
        else:
            read_mappings[read.query_name] = [dataset]
    #print(read_mappings)
    sam_file.close()
    for key in read_mappings:
        curr_set = set(read_mappings[key])
        for dataset in curr_set:
            confusion_matrix[0][dataset -1] += 1/len(curr_set)
    print(confusion_matrix)
        
        


def parse_arguments():
    """ Defines the command-line argument parser, and return arguments """
    parser = argparse.ArgumentParser(description="This script helps to analyze the SAM file from experiment 5"
                                                 "in order to form a confusion matrix.")
    parser.add_argument("-n", "--num", dest="num_datasets", required=True, help="number of datasets in this experiment", type=int)
    parser.add_argument("-s", "--sam_file", dest="sam_file", required=True, help="path to SAM file to be analyzed")
    parser.add_argument("-r", "--ref_lists", dest="ref_lists", required=True, help="path to directory with dataset reference lists")
    parser.add_argument("-o", "--output_path", dest = "output", required=False, help="path to output csv") # CHANGE TO REQUIRED
    args = parser.parse_args()
    return args

def build_refs_list(ref_lists_dir, num_datasets):
    ref_list = []
    for i in range(1,num_datasets + 1):
        curr_file = ref_lists_dir+"/dataset_{num}_references.txt".format(num = i)
        curr_set = set()
        with open(curr_file, "r") as input_fd:
            all_headers = [curr_set.add(x.strip()) for x in input_fd.readlines()]
        ref_list.append(curr_set)
    return ref_list

def find_class_of_reference_name(ref_list, ref_name):
    """ Finds the class that a particular reference name occurs in """
    datasets = []
    
    for i, name_set in enumerate(ref_list):
        if ref_name in name_set:
            datasets.append(i+1)
    if len(datasets) != 1:
        print(f"Error: this read hits {len(datasets)} which is not expected.")
        exit(1)
    return datasets[0]

if __name__ == "__main__":
    args = parse_arguments()
    main(args)