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
import pandas as pd
from plotnine import *
import contextlib
import io
import sys

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

@dataclass
class RunConfig:
    """Class for maintaing the config setup."""
    data_dir: str
    work_dir: str
    exp_num: int
    num_datasets: int
    trial_num: int
    repo_dir: str
    out_pivot: bool
    feat_level_opt: str

###########################################################
# Main methods for different sub-commands
###########################################################

def build_main(args):  
    """
    Main function for build sub-command

    :param args: argument object containing command-line parameters
    """  
    # make sure the arguments are valid
    check_build_arguments(args)
    start = time.time()

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
    
    # remove folder if doing a dry-run since snakemake
    # seems to create it even during a dry-run
    if args.dry_run:
        execute_command(f"rm -r {args.work_dir}")

    logging.info(f"build has completed, total time ({(time.time()-start):.3f} sec)\n")

def run_main(args):
    """
    Main function for run sub-command

    :param args: argument object containing command-line parameters
    """
    # make sure the arguments are valid
    check_run_arguments(args)
    start = time.time()

    if args.dry_run:
        logging.info("Dry run is activated. No snakemake workflows will be executed.")
    
    # identify how many trials/replicates are in the database to
    # verify that we have a valid number
    num_replicates, num_datasets = inspect_data_directory_for_run(args.data_dir)
    trials_to_run = validate_trials_arg(args.curr_trial, num_replicates)

    # get the valid config option for the FEAT_LEVEL_OPT variable
    feat_level_opt = "--use-feat-level" if args.feat_level else "dummy_var"

    for curr_trial in trials_to_run:
        print(); logging.info(f"experiment #{args.exp_num} is being used.")
        assert curr_trial >= 1 and curr_trial <= num_replicates

        # create the working directory for specific trial
        curr_work_dir = args.work_dir + f"trial_{curr_trial}/"
        execute_command(f"mkdir {curr_work_dir}")

        # create a config object with parameters
        configSettings = RunConfig(args.data_dir, curr_work_dir, args.exp_num, num_datasets, 
                                curr_trial, args.repo_dir, not args.in_pivot, feat_level_opt)
        
        # get the snakemake command, and then store stdout, stderr
        cmd = build_snakemake_command_for_run(configSettings, args.dry_run)

        if not args.dry_run:
            logging.info(f"snakemake run command is starting now (using trial #{curr_trial})...")
        else:
            logging.info(f"snakemake dry-run is being generated now (using trial #{curr_trial})...")
            if len(trials_to_run) > 1:
                print_warning("dry-run will be run with respect to one trial, despite multipe trials being requested.")

        # save the snakemake command to a log file
        with open("khoice.snakemake.log", "w+") as log_fd:
            log_fd.write(cmd + "\n")
        
        # write out the stdout, stderr to files
        with open("khoice.stderr.log", "w+") as stderr_fd, open("khoice.stdout.log", "w+") as stdout_fd:
            if not args.dry_run:
                exit_code = execute_command_with_files_and_poll(cmd, stdout_fd, stderr_fd, "run progress")
            else:
                exit_code = execute_command_with_files(cmd, stderr_fd, stdout_fd)
            
            if exit_code:
                print_error(f"the snakemake command exited with an error code ({exit_code})")

        # state where the final data is if it was executed.
        saved_loc = ""
        if not args.dry_run:
            saved_loc = final_file_name_for_exp(args, curr_trial)
            logging.info(f"output results saved here: {saved_loc}")
        else:
            break

    logging.info(f"run has completed, total time ({(time.time()-start):.3f} sec)\n")

def plot_main(args):
    """
    Main function for the plot sub-command that 
    is meant to take the run output and make plots.
    """
    # make sure the provided options are valid
    check_plot_arguments(args)

    # verify that the expected files are present in all trials
    required_files = get_required_files_for_plot(args.exp_num)
    trial_list = inspect_work_directory_for_plot(args.work_dir, required_files)

    logging.info(f"found {len(trial_list)} replicates in this results directory, and found expected files."); print()

    if args.exp_num == 2:
        logging.info(f"for experiment 2, we will only be using 1 trial, specifically trial {trial_list[0]}.")
        generate_exp2_plot(args.work_dir, trial_list[0], args.out_dir)



    
    # print(args.exp_num)
    # print(args.out_dir)
    # print(args.work_dir)

###########################################################
# Plotting methods
###########################################################

def generate_exp2_plot(work_dir, trial_num, out_dir):
    """
    Generates the plot for experiment 2.

    :param: work_dir - string, path to folder with results in it
    :param: trial_num - integer, trial number that we will extract data from
    :param: out_dir - string, path to folder we will save results in

    :return: ...
    """
    csv_path = f"{work_dir}trial_{trial_num}/across_dataset_analysis_type_2/across_dataset_analysis.csv"
    assert os.path.isfile(csv_path)

    # TODO: generalize this to create the plot for each group
    df = pd.read_csv(csv_path)
    df_filt = df.loc[df['group_num'] == 'group_1']
    df_filt = pd.melt(df_filt, id_vars=['group_num', 'k'], 
                               value_vars=['percent_1_occ', 'percent_2_to_3', 'percent_4_to_8', 'percent_9_more'],
                               var_name='range', value_name='percent')
    
    # TODO: write some to code to generalize the x-axis with the kmer values

    plot = (ggplot(df_filt, aes(fill="range", x="k", y="percent")) + 
            geom_bar(position="fill", stat="identity") +
            theme_classic() +
            theme(axis_title_x=element_text(size =14),
                  axis_title_y=element_text(size=14),
                  legend_position = "bottom", 
                  legend_text=element_text(size=10),
                  legend_box="vertical",
                  legend_title=element_text(size=10),
                  axis_text=element_text(size=14, color="black")) +
            labs(x="Kmer Length (k)",
                 y="Ratio of Unique Kmers") +
            scale_fill_discrete(name = "Number of Groups the Kmers Occur In:",
                                labels = ("Only Pivot", "2 to 3 Groups", "4 Groups", ">=9 Groups")))

    # save the plot
    file_name = f"/home/oahmed6/scr4_blangme2/oahmed/khoice_test/plots/database_0/exp_2/trial_{trial_num}.pdf"
    with nostdout():
        plot.save(file_name, height=6, width=8, verbose=True)
    logging.info(f"plot was saved in: {file_name}")


###########################################################
# Helper methods for the sub-commands
###########################################################

def final_file_name_for_exp(args, curr_trial):
    """
    Return the name of the final data file based
    on which experiment is being run

    :param: args - command line arguments
    :return: path - string with the name of the final data file
    """
    path = ""
    if args.exp_2:
        path = "within_dataset_analysis_type_2, across_dataset_analysis_type_2"
    elif args.exp_4:
        path = "accuracies_type_4/accuracy_values.csv"
    elif args.exp_6:
        path = f"trial_{curr_trial}_short_acc.csv, trial_{curr_trial}_long_acc.csv"
    elif args.exp_7:
        path =f"trial_{curr_trial}_mems_illumina.csv, trial_{curr_trial}_mems_ont.csv"
    return path

def validate_trials_arg(trial_arg, num_trials):
    """
    Looks at the trial argument and makes sure it is in a valid
    format either using a hyphen (-) or comma (,) to separate
    multiple trials.

    :param: trial_arg - string variable with trials requested by user
    :error: argument is not in valid format
    :error: one or more of the trial # is not valid
    """
    trials = []; numrange = False
    if "-" in trial_arg:
        trials = trial_arg.split("-")
        numrange = True
        if len(trials) != 2:
            print_error("when using hyphen, there must be two numbers provided to specify range.")
    elif "," in trial_arg:
        trials = trial_arg.split(",")
    else:
        if not trial_arg.isdigit():
            print_error("trial numbers provided with -t is not valid format.")
        elif int(trial_arg) < 1 or int(trial_arg) > num_trials:
            print_error("the trial number provided is not valid.")
        else:
            return [int(trial_arg)]
    
    # go through a series of checks, and modifications
    if not all([x.isdigit() for x in trials]):
        print_error("not all the trials provided with -t are numbers.")
    
    trials = [int(x) for x in trials]
    if not all([(x >= 1 and x <= num_trials) for x in trials]): 
        print_error("one or more of the trials provided with -t is not valid.")
    
    if numrange and trials[0] >= trials[1]:
        print_error("range specified with -t is not valid.")
    
    # fill in range if specified
    if numrange:
        trials = list(range(trials[0], trials[1]+1))
    return trials

def inspect_data_directory_for_run(database_dir):
    """
    Inspects the database folder to identify how many trials
    there are in this database, and verifies the structure is 
    as expected.

    :param: database_dir - path to database folder
    :return: [num_replicates, num_datasets] - number of trials that were created
    """
    trial_re = re.compile('trial_[0-9]+$')
    max_num = 0; max_dataset_num = 0
    for item in os.listdir(database_dir):
        if trial_re.match(item):
            max_num = max(max_num, int(item.split("_")[1]))
        elif item != "trial_summaries" and item[0] != ".":
            print_error(f"{item} was found, but not expected in database directory.")
    
    dataset_re = re.compile("dataset_[0-9]+")
    database_dir += "/" if database_dir[-1] != "/" else database_dir
    for item in os.listdir(database_dir + f"trial_{max_num}/exp0_pivot_genomes/"):
        if dataset_re.match(item):
            max_dataset_num = max(max_dataset_num, int(item.split("_")[1]))
    
    return [max_num, max_dataset_num]  

def inspect_data_directory_for_build(data_dir) -> int:
    """
    Inspects the database directory for the build sub-command
    and makes sure it has the directory structure expected
    if it were to be downloaded using the download script.

    :param: data_dir - directory path with the raw data
    :return: num_datasets - number of different groups in database

    :error: when folder is formated correctly.
    """
    dataset_re = re.compile('dataset_[0-9]+$')
    max_num = 0
    for item in os.listdir(data_dir):
        if dataset_re.match(item):
            max_num = max(max_num, int(item.split("_")[1]))
        elif item != "README_dataset_summary.txt":
            print_error(f"{item} was found, but not expected in data directory.")
    return max_num

def inspect_work_directory_for_build(work_dir) -> int:
    """
    Inspect the working directory for the build sub-command and
    make sure that the requested database is not already built.

    :param: work_dir - path to the working directory, where database_* will go
    :return: num_database - the database number that will be created

    :error: when there are unexpected folders in that directory
    """
    dataset_re = re.compile('database_[0-9]+$')
    max_num = 0
    for item in os.listdir(work_dir):
        if dataset_re.match(item):
            max_num = max(max_num, int(item.split("_")[1]))
        elif item[0] != ".": # avoid hidden files
            print_error(f"{item} was found, but not expected in working directory.")
    return max_num+1

def inspect_work_directory_for_plot(work_dir, required_files) -> int:
    """
    Inspect the working directory for the plot sub-command to
    see how many trials there are and validates that the requested files
    are present in each trial.

    :param: work_dir - path to the working directory, where trial_* should be
    :return: trial_list - list of integers, the trial numbers in results folder

    :error: when there are missing files
    :error: when there are no results at all
    """
    # get number of trials
    trial_re = re.compile('trial_[0-9]+$')
    max_num = 0
    trial_list = []
    for item in os.listdir(work_dir):
        if trial_re.match(item):
            max_num = max(max_num, int(item.split("_")[1]))
            trial_list.append(int(item.split("_")[1]))
        elif item[0] != ".": # avoid hidden files
            print_error(f"{item} was found, but not expected in working directory.")
    
    if len(trial_list) == 0:
        print_error("there were no results found in the provided folder.")
    
    # check the existence of each file
    verified = all([all([os.path.isfile(work_dir + f"trial_{i}/" + filename) for filename in required_files]) for i in trial_list])
    trial_list.sort()
    return trial_list

def get_required_files_for_plot(exp_num):
    """
    Returns the relative filenames for files that are expected
    to be present in the results folder in order to make the
    plot.

    :param: exp_num - integer, the experiment we want to plot the data from
    :return: file_list - list of strings, the files needed to plot exp_num

    :error: occurs when a unsupported experiment number if provided
    """
    if exp_num == 2:
        return ["across_dataset_analysis_type_2/across_dataset_analysis.csv"]
    else:
        print_error("unexpected experiment number was provided.")

def build_snakemake_command_for_build(config, dry_run) -> str:
    """
    Builds the snakemake command for the building 
    the database

    :param: config - Config object with all the of the fields needed
    :param: dry_run - boolean variable for if build is dry-run or not
    :return: cmd - a string of the snakemake command that will be run
    """
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

def build_snakemake_command_for_run(config, dry_run) -> str:
    """
    Builds the snakemake command for running an
    experiment

    :param: config - Config object with all the of the fields needed
    :param: dry_run - boolean variable for if build is dry-run or not
    :return: cmd - a string of the snakemake command to be run
    """
    exec_status = "-n" if dry_run else "-c1"
    
    target = ""
    if config.exp_num == 2:
        target = "generate_exp2_output"
    elif config.exp_num == 4:
        target = "generate_exp4_output"
    elif config.exp_num == 6:
        target = "generate_exp6_output"
    elif config.exp_num == 7:
        target = "generate_exp7_output"
    
    cmd = "snakemake --config DATABASE_ROOT={} \
                              WORK_ROOT={} \
                              EXP_TYPE={} \
                              NUM_DATASETS={} \
                              TRIAL_NUM={} \
                              REPO_DIRECTORY={} \
                              OUT_PIVOT={} \
                              FEAT_LEVEL_OPT={} \
                              {} {}".format(config.data_dir, config.work_dir, config.exp_num,
                                                                       config.num_datasets, config.trial_num,
                                                                       config.repo_dir, config.out_pivot, config.feat_level_opt,
                                                                       exec_status, target)
    return cmd

###########################################################
# Command-line parsing and error-checking methods
###########################################################

def parse_arguments():
    """
    Parses the command line arguments 
    for all the sub-commands.

    :return: arguments object
    """
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
    run_parser.add_argument("-d", "--database", dest="data_dir", help="folder path with processed data files", required=True, type=str)
    run_parser.add_argument("-w", "--workdir", dest="work_dir", help="folder path to where the working directory is", required=True, type=str)
    run_parser.add_argument("-t", "--curr-trial", dest="curr_trial", help="trial number(s) to use for an experiment (e.g. 1-5 or 1,3,4)", required=True, type=str)
    run_parser.add_argument("--repo-dir", dest="repo_dir", help="path to khoice repo", required=True, type=str)
    run_parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=False, help="do not run the snakemake, just run a dry run")
    run_parser.add_argument("--exp2", dest="exp_2", action="store_true", default=False, help="run experiment 2.")
    run_parser.add_argument("--exp4", dest="exp_4", action="store_true", default=False, help="run experiment 4.")
    run_parser.add_argument("--exp6", dest="exp_6", action="store_true", default=False, help="run experiment 6.")
    run_parser.add_argument("--exp7", dest="exp_7", action="store_true", default=False, help="run experiment 7.")
    run_parser.add_argument("--in-pivot", dest="in_pivot", action="store_true", default=False, help="(only for exp 4) use in-pivot opposed to out-pivot")
    run_parser.add_argument("--feature-level", dest="feat_level", action="store_true", default=False, help="(only for exp 6/7) perform classification at the feature level. (default: read-level)")
    run_parser.set_defaults(func="run")

    # create the parser for the "plot" sub-command
    plot_parser = sub_parsers.add_parser('plot', 
                                          help='plot the results from khoice experiment',
                                          description="Plots results from running a khoice experiment")
    plot_parser.add_argument("-w", "--workdir", dest="work_dir", help="folder path to where the results are", required=True, type=str)
    plot_parser.add_argument("-o", "--outdir", dest="out_dir", help="folder where the plots will be saved", required=True, type=str)
    plot_parser.add_argument("--repo-dir", dest="repo_dir", help="path to khoice repo", required=True, type=str)
    plot_parser.add_argument("--exp2", dest="exp_2", action="store_true", default=False, help="plot the results from experiment 2.")
    plot_parser.set_defaults(func="plot")

    parser.set_defaults(func="top-level")
    args = parser.parse_args()
    return args

def check_build_arguments(args):
    """
    Verify the arguments provided to the build
    sub-command are valid, and make sense.

    :param: args - arguments object from argparse
    """
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

def check_run_arguments(args):
    """
    Verify the arguments provided to the build
    sub-command are valid, and make sense.

    :param: args - arguments object from argparse
    """
    if not os.path.isdir(args.data_dir):
        print_error("the data directory path (-d) is not valid.\n")
    if not os.path.isdir(args.work_dir):
        print_error("the working directory (-w) is not valid.\n")
    if not os.path.isdir(args.repo_dir):
        print_error("the path provided for the khoice repository is not valid")

    count = 0
    for turned_on, val in [[args.exp_2, 2], [args.exp_4, 4], [args.exp_6, 6], [args.exp_7, 7]]:
        if turned_on:
            args.exp_num = val
            count += 1
    if count != 1:
        print_error("must specify exactly one of the experiments to run.")

    if args.work_dir[-1] != "/":
        args.work_dir += "/"

def check_plot_arguments(args):
    """
    Verify the arguments provided to the plot
    sub-command are valid, and make sense.

    :param: args - arguments object from argparse
    """
    if not os.path.isdir(args.work_dir):
        print_error("the working directory (-w) is not valid.\n")
    if args.work_dir[-1] != "/":
        args.work_dir += "/"

    if not os.path.isdir(args.out_dir):
        print_error("the output directory (-o) is not valid.")
    if args.out_dir[-1] != "/":
        args.out_dir += "/"

    count = 0
    for turned_on, val in [[args.exp_2, 2]]:
        if turned_on:
            args.exp_num = val
            count += 1
    if count != 1:
        print_error("must specify exactly one of the experiments to run.")

###########################################################
# General helper methods for the code
###########################################################

def print_error(msg) -> None:
    """
    Prints out an error message and exits the
    code.

    :param: msg - the error message
    """
    print("\n\033[0;31mError:\033[00m " + msg + "\n")
    exit(1)

def print_warning(msg) -> None:
    """
    Prints out a warning message.

    :param: msg - the error message
    """
    print("\n\033[0;33mWarning:\033[00m " + msg + "\n")

def prompt_user_for_continuation(msg):
    """
    Prints out a continue message to user and
    verifies that the user wants to continue

    :param: msg - string that will prompt the user

    :exit: if user decides to exit, they can answer 'n'
    """
    print()
    question = "\033[0;31m[question]\033[00m "
    decision = input(question + msg + " Continue [Y/n]? ")
    
    while decision.upper() != "Y" and decision.upper() != "N":
        decision = input(question + msg + " Continue [Y/n]?")

    if decision.upper() == "N":
        print("\nexiting now ...\n")
        exit(0)

def execute_command(cmd):
    """
    Execute a command using subprocess module, but return
    the stdout, stderr, and code

    :param: cmd - a string with the command-line expression
    :return: exit_code - exit code from command
    """
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    exit_code = process.returncode
    return [stdout, stderr, exit_code]

def execute_command_with_files(cmd, stdout_fd, stderr_fd):
    """
    Execute a command using subprocess module

    :param: cmd - a string with the command-line expression
    :param: stdout_fd - file handle for stdout from command
    :param: stderr_fd - file handle for stderr from command
    :return: exit_code - exit code from command
    """
    process = subprocess.Popen(cmd.split(), stdout=stdout_fd, stderr=stderr_fd)
    stdout, stderr = process.communicate()
    exit_code = process.returncode
    return exit_code

def execute_command_with_files_and_poll(cmd, stdout_fd, stderr_fd, title):
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

class DummyFile(object):
    """
    Dummy object when you do not care about the
    stdout/stderr from a function or set of lines
    of code

    Copied from https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
    """
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    """
    Suppresses the stdout/stderr from a context, this
    was written since the plotnine save method
    verbose option was not working.
    """
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    sys.stdout = DummyFile()
    sys.stderr = DummyFile()
    yield
    sys.stderr = save_stderr
    sys.stdout = save_stdout

if __name__ == '__main__':
    args = parse_arguments()
    # old format: datefmt="%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=logging.INFO, format='\033[0;32m[log][%(asctime)s]\033[00m %(message)s', datefmt="%H:%M:%S")

    logging.getLogger('matplotlib.font_manager').disabled = True


    print(f"\n\033[0;32mkhoice version: {__VERSION__}\033[00m\n")
    if args.func == "build":    
        build_main(args)
    elif args.func == "run":
        run_main(args)
    elif args.func == "plot":
        plot_main(args)