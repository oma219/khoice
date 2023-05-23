"""
Name: khoice.py
Description: Main Python wrapper that can execute the
             snakemake workflows for building databases
             and running khoice experiments.
Author: Omar Ahmed
"""

import argparse
import os
import logging
import re
from dataclasses import dataclass
import subprocess
import time
from alive_progress import alive_bar

__VERSION__ = "1.0.0"

###########################################################
# Declarations of classes and data-classes
###########################################################

@dataclass
class BuildConfig:
    """Class for maintaing the config setup."""
    data_dir: str
    work_dir: str
    exp_num: int
    num_datasets: int
    num_trials: int
    kmers_per_dataset: int
    repo_dir: str
    pbsim_model: str

###########################################################
# Main methods for different sub-commands
###########################################################

"""
Main function for build sub-command

:param args: argument object containing command-line parameters
"""
def build_main(args):    
    # make sure the arguments are valid
    check_build_arguments(args)
    start = time.process_time()

    if args.dry_run:
        logging.info("Dry run is activated. No snakemake workflows will be executed."); print()

    # inspect the data/working directory and make sure it is as expected
    num_datasets = inspect_data_directory_for_build(args.data_dir)
    logging.info(f"found {num_datasets} different sets of genomes in database folder.")

    curr_db_num = inspect_work_directory_for_build(args.work_dir)
    logging.info(f"building the khoice database in this folder: database_{curr_db_num}/")

    # make sure it will be in the expected folder, because the 
    # code will create it for the user
    prompt_user_for_continuation(f"Verify the build directory, " 
                                "you may need to delete an old folder.")
    
    # update the working directory path, and make directory
    args.work_dir = args.work_dir + f"database_{curr_db_num}" if args.work_dir[-1] == "/" else args.work_dir + f"/database_{curr_db_num}"
    if not args.dry_run:
        print("why is it in here")
        execute_command(f"mkdir {args.work_dir}")

    # create a config object with parameters
    configSettings = BuildConfig(args.data_dir, args.work_dir, 0, num_datasets, args.num_trials,
                                args.num_kmers_per_dataset, args.repo_dir, args.pbsim_model)
    
    # get build command, and then store stdout, stderr
    cmd = build_snakemake_command_for_build(configSettings, args.dry_run); print()
    if not args.dry_run:
        logging.info("snakemake build command is starting now ...")
    else:
        logging.info("snakemake dry-run is being generated now ...")
 
    # save the snakemake command to a log file
    with open("khoice.snakemake.log", "w") as log_fd:
        log_fd.write(cmd + "\n")
    
    # write out the stdout, stderr to files
    with open("khoice.stderr.log", "w+") as stderr_fd, open("khoice.stdout.log", "w+") as stdout_fd:
        if not args.dry_run:
            exit_code = execute_command_with_files_and_poll(cmd, stdout_fd, stderr_fd, "build progress")
        else:
            exit_code = execute_command_with_files(cmd, stderr_fd, stdout_fd)
        
        if exit_code:
            print_error(f"the snakemake command exited with an error code ({exit_code})")
    
    logging.info(f"build has completed, total time ({(time.process_time()-start):.3f} sec)\n")


"""
Main function for run sub-command

:param args: argument object containing command-line parameters
"""
def run_main(args):
    raise NotImplementedError("Run sub-command is not implemented yet.")

###########################################################
# Helper methods for both sub-commands
###########################################################

"""
Inspects the database directory for the build sub-command
and makes sure it has the directory structure expected
if it were to be downloaded using the download script.

:param: data_dir - directory path with the raw data
:return: num_datasets - number of different groups in database

:error: when folder is formated correctly.
"""
def inspect_data_directory_for_build(data_dir) -> int:
    dataset_re = re.compile('dataset_[0-9]+$')
    max_num = 0
    for item in os.listdir(data_dir):
        if dataset_re.match(item):
            max_num = max(max_num, int(item.split("_")[1]))
        elif item != "README_dataset_summary.txt":
            print_error(f"{item} was found, but not expected in data directory.")
    return max_num

"""
Inspect the working directory for the build sub-command and
make sure that the requested database is not already built.

:param: work_dir - path to the working directory, where database_* will go
:return: num_database - the database number that will be created

:error: when there are unexpected folders in that directory
"""
def inspect_work_directory_for_build(work_dir) -> int:
    dataset_re = re.compile('database_[0-9]+$')
    max_num = 0
    for item in os.listdir(work_dir):
        if dataset_re.match(item):
            max_num = max(max_num, int(item.split("_")[1]))
        elif item[0] != ".": # avoid hidden files
            print_error(f"{item} was found, but not expected in working directory.")
    return max_num+1

"""
Builds the snakemake command for the building 
the database

:param: config - Config object with all the of the fields needed
:param: dry_run - boolean variable for if build is dry-run or not
"""
def build_snakemake_command_for_build(config, dry_run) -> str:
    exec_status = "-n" if dry_run else "-c1"
    cmd = "snakemake --config DATABASE_ROOT={} \
                              WORK_ROOT={} \
                              EXP_TYPE={} \
                              NUM_DATASETS={} \
                              NUM_TRIALS={} \
                              KMERS_PER_DATASET={} \
                              REPO_DIRECTORY={} \
                              PBSIM_MODEL={} \
                              {} make_all_data_summaries_exp0".format(config.data_dir, config.work_dir, config.exp_num,
                                                                       config.num_datasets, config.num_trials, config.kmers_per_dataset,
                                                                       config.repo_dir, config.pbsim_model, exec_status)
    return cmd

###########################################################
# Command-line parsing and error-checking methods
###########################################################

"""
Parses the command line arguments 
for all the sub-commands.

:return: arguments object
"""
def parse_arguments():
    # top-level parser
    parser = argparse.ArgumentParser(prog='khoice', 
                                     description="Builds khoice databases and run experiments.")
    # create sub-parser
    sub_parsers = parser.add_subparsers(help='sub-command help')

    # create the parser for the "build" sub-command
    build_parser = sub_parsers.add_parser('build', 
                                          help='build a khoice database',
                                          description="Builds khoice databases after downloading genomes")
    
    build_parser.add_argument("-d", "--database", dest="data_dir", help="folder path with raw database files", required=True, type=str)
    build_parser.add_argument("-w", "--workdir", dest="work_dir", help="folder path to working directory", required=True, type=str)
    build_parser.add_argument("-t", "--num-trials", dest="num_trials", help="number of replicate datasets to create (default: 5)", default=5, type=int)
    build_parser.add_argument("-k", "--num-kmers", dest="num_kmers_per_dataset", help="number of kmers to include from each dataset (default: 1 million)", default=1000000, type=int)
    build_parser.add_argument("--repo-dir", dest="repo_dir", help="path to the khoice repo", required=True, type=str)
    build_parser.add_argument("--pbsim-model", dest="pbsim_model", help="path to PBSIM model file", required=True, type=str)
    build_parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=False, help="do not run the snakemake, just run a dry run")
    build_parser.set_defaults(func="build")

    # create the parser for the "run" sub-command
    run_parser = sub_parsers.add_parser('run', 
                                        help='run an experiment with database', 
                                        description="Runs an experiment using a khoice database.")
    run_parser.set_defaults(func="run")

    args = parser.parse_args()
    return args

"""
Verify the arguments provided to the build
sub-command are valid, and make sense.

:param: args - arguments object from argparse
"""
def check_build_arguments(args):
    if not os.path.isdir(args.data_dir):
        print_error("the data directory path (-d) is not valid.\n")
    if not os.path.isdir(args.work_dir):
        print_error("the working directory (-w) is not valid.\n")
    if args.num_trials < 1 or args.num_trials > 10:
        print_error("the number of trials (-t) requested should be between 1 and 10 inclusive.\n")
    if args.num_kmers_per_dataset > 10000000:
        print_error("the number of kmers (-k) should not exceed 10 million.\n")
    if not os.path.isdir(args.repo_dir):
        print_error("the path provided for the khoice repository is not valid")
    if not os.path.isfile(args.pbsim_model):
        print_error("the path to the PBSIM model is not valid")

"""
Verify the arguments provided to the build
sub-command are valid, and make sense.

:param: args - arguments object from argparse
"""
def check_run_arguments(args):
    raise NotImplementedError("")


###########################################################
# General helper methods for the code
###########################################################

"""
Prints out an error message and exits the
code.

:param: msg - the error message
"""
def print_error(msg) -> None:
    print("\n\033[0;31mError:\033[00m " + msg + "\n")
    exit(1)


"""
Prints out a continue message to user and
verifies that the user wants to continue

:param: msg - string that will prompt the user

:exit: if user decides to exit, they can answer 'n'
"""
def prompt_user_for_continuation(msg):
    print()
    question = "\033[0;31m[question]\033[00m "
    decision = input(question + msg + " Continue [Y/n]? ")
    
    while decision.upper() != "Y" and decision.upper() != "N":
        decision = input(question + msg + " Continue [Y/n]?")

    if decision.upper() == "N":
        print("\nexiting now ...\n")
        exit(0)

"""
Execute a command using subprocess module

:param: cmd - a string with the command-line expression
:param: stdout_fd - file handle for stdout from command
:param: stderr_fd - file handle for stderr from command
:return: exit_code - exit code from command
"""
def execute_command_with_files(cmd, stdout_fd, stderr_fd):
    process = subprocess.Popen(cmd.split(), stdout=stdout_fd, stderr=stderr_fd)
    stdout, stderr = process.communicate()
    exit_code = process.returncode
    return exit_code

"""
Execute a command using subprocess module but
this method is for the case you want to pipe 
the output to files directly and you want to 
wrap the command in a progress bar.

:param: cmd - a string with the command-line expression
:param: title - short description of this command
:param: stdout_fd - file handle that stdout will be piped to
:param: stderr_fd - file handle that stderr will be piped to
:return: exit_code - exit code from command
"""
def execute_command_with_files_and_poll(cmd, stdout_fd, stderr_fd, title):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    progress_line = re.compile('[0-9]+ of [0-9]+ steps \([0-9]+%\) done$')

    # Grab the number of jobs to be completed
    total_steps = 0
    for line in process.stderr:
        stderr_fd.write(line)
        if progress_line.match(line):
            total_steps = int(line.split("steps")[0].split("of")[1])
            break
    print()

    # Update progress bar as jobs are being completed
    with alive_bar(total_steps, title=title, bar='filling', spinner="wait4", manual=True) as bar:
        linenum = 0
        for line in process.stderr:
            stderr_fd.write(line)
            if progress_line.match(line):
                bar(int(line.split("steps")[0].split("of")[0])/(total_steps+0.0))
            linenum += 1
    process.wait()
    print()

    # Write out the stdout as well, and grab exit code
    for line in process.stdout:
        stdout_fd.write(line)
    exit_code = process.returncode

    return exit_code

if __name__ == '__main__':
    args = parse_arguments()
    # old format: datefmt="%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=logging.DEBUG, format='\033[0;32m[log][%(asctime)s]\033[00m %(message)s', datefmt="%H:%M:%S")

    print(f"\n\033[0;31mkhoice version: {__VERSION__}\033[00m\n")
    if args.func == "build":    
        build_main(args)
    elif args.func == "run":
        run_main(args)