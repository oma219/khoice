#!/usr/bin/env python3

################################################################################
# Name: download_genomes.py 
# Description: This is a python script that will allow you to request certain
#              groups of genomes from NCBI databases like RefSeq and GenBank.
#
# Date: January 8th, 2022
################################################################################

import argparse
import os
import subprocess
import time

def run_command(command, loading_bar, sleep_time=1):
    """ 
    Takes in a command, and runs it. Presents a loading bar in case a command takes 
    some time to run like queries to databases or downloading genomes.
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    loading_bar = False # Omar - set to False since it was leading to errors

    if loading_bar:
        print("Loading ...", end='', flush="True")
        while process.poll() is None:
            time.sleep(sleep_time)
            print(".", end='', flush="True")
            (output, error) = process.communicate()
        (output, error) = process.communicate()
    else:
        (output, error) = process.communicate()

    output = output.decode("utf-8").strip() if output is not None else output
    error = error.decode("utf-8").strip() if error is not None else error
    return (output, error)

##################################################################
# NOTE: There is another way to find the FTP links for the 
# genomes that is slightly easier, it involves downloading the 
# assembly file, and then parsing it for what you are interested
# in. The approach is described in the following link:
# https://www.biostars.org/p/255212/
##################################################################

def find_refseq_ftp_paths(requested_species, using_file_input):
    """ Finds all the RefSeq FTP links for requested species """
    command = ("esearch -db assembly -query '\"{}\"[Organism] AND (\"latest refseq\"[filter] AND \"complete genome\"[filter] AND all[filter] NOT anomalous[filter])'"
               " | esummary | "
               " xtract -pattern DocumentSummary -block FtpPath -match '@type:RefSeq' -element FtpPath " 
               " | sed 's/$/\//' ")

    # Executes the command to figure out how many assemblies there are ...
    current_command = command.format(requested_species, requested_species) + " | wc -l"
    output, error = run_command(current_command, (not using_file_input))

    # Interprets output, and asks user if they want to download the genomes ...
    if output.isdigit() and int(output) > 0 and not using_file_input:
        response = input(f"\nWe found {output} complete genomes for {requested_species}. Does that\nsound right, would you like to download them (y/n)? ").upper().strip()
        while response != "Y" and response != "N":
            response = input("\nError in response, do you want to download or not (y/n)? ").upper().strip()
        if response == "N":
            return ([], "no")
    elif output.isdigit() and int(output) > 0 and using_file_input:
        response = "Y"
    elif output.isdigit() and int(output) == 0:
        print("\nError: No genomes were found for this input.")
        exit(1)
    else:
        print(f"\nError occurred during database query: {output}")
        exit(1)
    
    current_command = command.format(requested_species, requested_species)
    output, error = run_command(current_command, False)
    output_lines = output.split("\n")

    genome_paths = []
    for ftp_path in output_lines:
        name = ftp_path.split("/")[-2]
        genome_paths.append(ftp_path + name + "_genomic.fna.gz")
    return (genome_paths, "yes")

def find_genbank_ftp_paths(requested_species):
    """ Finds all the GenBank FTP links for requested species """
    raise NotImplementedError("still working on this ...")

def download_genomes(ftp_paths, output_dir, num, num_genomes, ratio_genomes, decompress):
    """ Takes a list of paths, and downloads them to output directory"""
    with open(output_dir + "ftp_paths_for_curr_dataset.txt", "w") as fd:
        # First, figure out if user specified a certain number or ratio ...

        if num_genomes == 0 and ratio_genomes == 0.0: # Case 1 - take all genomes
            num_genomes = len(ftp_paths)
        elif num_genomes == 0 and ratio_genomes > 0.0: # Case 2 - take a certain ratio
            num_genomes = int(len(ftp_paths) * ratio_genomes)
            if num_genomes == 0:
                print("Error: The ratio provided with -p option was too small.")
                exit(1)
        # Case 3 (Default Case) - use the number of genomes directly from command-line

        for paths in ftp_paths[:num_genomes]:
            fd.write(paths + "\n")
    
    filelist = output_dir + "ftp_paths_for_curr_dataset.txt"
    current_dataset_dir = output_dir + "dataset_" + str(num)

    # Executes a command to download genomes, remove filelist and
    # then decompress all the genomes

    print(f"\nStarting to download files to dataset_{num} folder ...")
    download_command = f"wget --input-file={filelist} --directory-prefix={current_dataset_dir} --no-parent --no-host-directories --cut-dirs=5"
    run_command(download_command, True, sleep_time=10)

    remove_command = f"rm {filelist} "
    run_command(remove_command, False)

    if decompress:
        print(f"\nDecompressing those downloaded files now ...")
        decompress_command = f"for i in $(find {current_dataset_dir} -name '*.fna.gz'); do gzip -d $i; done"
        output, error = run_command(decompress_command, True)

    return num_genomes

def print_status(args):
    """ It will just state what parameters the user as specified prior to downloading any genomes """
    database = "RefSeq" if args.refseq else "GenBank"
    input_style = "File Input" if args.input_list != "" else "Console"

    print("\nProvided Parameters:")
    print(f"\tGenome Database: {database}")
    print(f"\tOutput Directory: {args.output_dir}")
    print(f"\tInput Type: {input_style}\n")

def input_generator(input_file, input_list):
    """ Generates a sequence of species to download: either from user input or file input """
    if input_file == "":
        requested_species = input("Enter the name of species you want to download (type \"done\" to stop): ").strip()
    else:
        if len(input_list) > 0:
            requested_species = input_list[0]
            input_list.pop(0) # lists are passed by reference
        else:
            return "done"
    return requested_species

def grab_species_from_file(input_list):
    """ Extracts the species from each line in file """
    with open(input_list, "r") as input_fd:
        lines = [x.strip() for x in input_fd.readlines()]
        if '' in lines:
            lines.remove('') # remove any empty lines
    return lines

def main(args):
    """
    Main method:
        It will prompt the user continuously for a species name that they would like
        to download to the pre-specified directory, and it will continue to ask until
        user is done.
    """
    dataset_num = 1
    species_list = []
    dataset_summaries = []

    using_file_input = False
    downloaded_genomes = False

    if args.input_list != "":
        species_list = grab_species_from_file(args.input_list)
        using_file_input = True

    print_status(args)
    while True:
        requested_species = input_generator(args.input_list, species_list)
        if requested_species.upper() == "DONE":
            break

        # Determine the paths to genomes for specified species
        if args.refseq:
            paths, choice = find_refseq_ftp_paths(requested_species, using_file_input)
        elif args.genbank:
            paths, choice = find_genbank_ftp_paths(requested_species, using_file_input)

        # Decided to not download the genomes ...
        if choice == "no":
            print()
            continue

        # Execute the download of those genomes ...
        num_downloaded = download_genomes(paths, args.output_dir, dataset_num, args.num_genomes, 
                                          args.ratio_genomes, args.decompress)
        downloaded_genomes = True

        dataset_summaries.append([dataset_num, requested_species, num_downloaded])
        dataset_num += 1
        print()

    # Write out a README explaining what is in each folder ...
    if downloaded_genomes:
        with open(args.output_dir + "README_dataset_summary.txt", "w") as fd:
            fd.write("{:<20}{:<30}{:<20}\n".format("Dataset Number:", "Species:", "Number of Genomes:"))
            for summary in dataset_summaries:
                fd.write("{:<20d}{:<30}{:<20d}\n".format(summary[0], summary[1], summary[2]))

def parse_arguments():
    """ Defines the command-line argument parser, and return arguments """
    parser = argparse.ArgumentParser(description="This script allows users to download certain groups of genomes from the NCBI databases. It "
                                                  "will prompt users for species that you would like to download, and then those genomes "
                                                  "will be saved in the provided folder.")
    parser.add_argument("-r", "--refseq", action="store_true", default=False, help="download genomes from RefSeq")
    parser.add_argument("-g", "--genbank", action="store_true", default=False, help="download genomes from GenBank")
    parser.add_argument("-o", dest="output_dir", required=True, help="directory where all the genomes will be stored.")
    parser.add_argument("-f", dest="input_list", help="path to file with species listed on each line to download.", default="")
    parser.add_argument("-n", dest="num_genomes", help="number of genomes to download from each species (default: all available)", type=int, default=0)
    parser.add_argument("-p", dest="ratio_genomes", help="percentage of genomes to download from available (0.0 < x <= 1.0)", type=float, default=0.0)
    parser.add_argument("-d", dest="decompress", help="decompress genomes when downloaded (default: false)", default=False, action="store_true")
    args = parser.parse_args()
    return args

def check_arguments(args):
    """ Makes sure the arguments provided by user are valid, and make sense """
    if not args.refseq and not args.genbank:
        print("Error: Need to specify where to download genomes from either RefSeq (-r) or GenBank (-g)")
        exit(1)
    if args.input_list != "" and not os.path.isfile(args.input_list):
        print("Error: Input file given via -f option is not a valid path.")
        exit(1)
    if args.refseq and args.genbank:
        print("Error: Can only choose one database to download from.")
        exit(1)
    if args.num_genomes < 0:
        print("Error: The number of genomes must be a positive integer.")
        exit(1)
    if args.ratio_genomes < 0.0:
        print("Error: The ratio of genomes to download must be a positive ratio (0.0 < x <= 1.0)")
        exit(1)
    if args.ratio_genomes > 1.0:
        print("Error: The ratio of genomes to download must be between 0.0 < x <= 1.0.")
        exit(1)
    if args.num_genomes > 0 and args.ratio_genomes > 0.0:
        print("Error: The -n and -p options cannot be used at same time, just one of them or neither.")
        exit(1)
    if not os.path.isdir(args.output_dir):
        print("Error: The provided output directory is not valid.")
        exit(1)
    elif args.output_dir[-1] != "/":
        args.output_dir += "/"
    
    # Adjust output directory to saves files to data folder
    # since workflow expects, the script tries to identify how
    # many databases current exist.
    curr_database_num = 1
    while True:
        if not os.path.isdir(f"{args.output_dir}database_{curr_database_num}/"):
            break
        else:
            curr_database_num += 1
  
    if not os.path.isdir(f"{args.output_dir}database_{curr_database_num}/"):
        os.mkdir(f"{args.output_dir}database_{curr_database_num}/")
    args.output_dir += f"database_{curr_database_num}/"

if __name__ == "__main__":
    args = parse_arguments()
    check_arguments(args)
    main(args)