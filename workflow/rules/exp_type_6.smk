####################################################
# Name: exp_type_6.smk
# Description: Contains functions and rules for
#              the type of experiment type 6: 
#
#              Simulates reads from an out-pivot genome from
#              each species. Then, determines which species each 
#              kmer occurs in (across k) and builds a confusion matrix.
#
# Date: 6/30/22
####################################################

####################################################
# Section 1: Code that is always run, when this 
#            experiment is called. It genereates the
#            needed files.
# 
#            It builds the needed directories for 
#            KMC3, complex ops files for KMC3, and it
#            builds the filelists needed for a python
#            script.
####################################################

if exp_type == 6:
    # Make tmp directory for kmc
    if not os.path.isdir("tmp"):
        os.mkdir("tmp")

    # Parse out a pivot from the downloaded dataset
    if not os.path.isdir("exp6_input/"):
        for i in range(1, num_datasets+1):
            dir_prefix = f"{database_root}/trial_{curr_trial}/exp0_nonpivot_genomes/dataset_{i}/"
            list_of_files = [dir_prefix + x for x in os.listdir(dir_prefix)]

            if not os.path.isdir(f"exp6_input/rest_of_set/dataset_{i}"):
                os.makedirs(f"exp6_input/rest_of_set/dataset_{i}")

            # Copy over all the files at first
            for data_file in list_of_files:
                shutil.copy(data_file, f"exp6_input/rest_of_set/dataset_{i}")
            
            # Extract the pivot and move to new directory
            pivot = f"{database_root}/trial_{curr_trial}/exp0_pivot_genomes/dataset_{i}/pivot_{i}.fna.gz"
            if not os.path.isdir(f"exp6_input/pivot/"):
                os.makedirs(f"exp6_input/pivot/")
            shutil.copy(pivot, f"exp6_input/pivot/pivot_{i}.fna.gz")

            # Copy over reads to approriate places
            if not os.path.isdir(f"exp6_input/pivot_reads_subset/illumina"):
                os.makedirs(f"exp6_input/pivot_reads_subset/illumina")
            shutil.copy(f"{database_root}/trial_{curr_trial}/exp0_pivot_reads/dataset_{i}/illumina/pivot_{i}_subset.fa",
            f"exp6_input/pivot_reads_subset/illumina/pivot_{i}.fa")

            if not os.path.isdir(f"exp6_input/pivot_reads_subset/ont"):
                os.makedirs(f"exp6_input/pivot_reads_subset/ont")
            shutil.copy(f"{database_root}/trial_{curr_trial}/exp0_pivot_reads/dataset_{i}/ont/pivot_{i}_subset.fa",
            f"exp6_input/pivot_reads_subset/ont/pivot_{i}.fa")

    
    # Initialize complex ops directory to start writing files
    if not os.path.isdir("exp6_complex_ops"):
        os.makedirs("exp6_complex_ops")

    # Build the complex ops files
    for k in k_values:
        if not os.path.isdir(f"exp6_complex_ops/k_{k}"):
            os.mkdir(f"exp6_complex_ops/k_{k}")
        
        for num in range(1, num_datasets+1):
            kmc_input_files = []

            for data_file in os.listdir(f"exp6_input/rest_of_set/dataset_{num}"):
                if data_file.endswith(".fna.gz"):
                    base_name = data_file.split(".fna.gz")[0]
                    kmc_input_files.append(f"exp6_genome_sets/rest_of_set/k_{k}/dataset_{num}/{base_name}.transformed")
            
            if not os.path.isdir(f"exp6_complex_ops/k_{k}/dataset_{num}/"):
                os.mkdir(f"exp6_complex_ops/k_{k}/dataset_{num}/")

            with open(f"exp6_complex_ops/k_{k}/dataset_{num}/ops_{num}.txt", "w") as fd:
                fd.write("INPUT:\n")
                result_str = "("
                for i, path in enumerate(kmc_input_files):
                    fd.write(f"set{i+1} = {path}\n")
                    result_str += "set{} + ".format(i+1)
                result_str = result_str[:-2] + ")"
                fd.write("OUTPUT:\n")
                fd.write(f"exp6_unions/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined = {result_str}\n")
                fd.write("OUTPUT_PARAMS:\n-cs5000\n")
    
    # Create filelists that will be used for merging text dumps
    read_types = ["illumina","ont"]
    
    for k in k_values:
        if not os.path.isdir(f"exp6_filelists/k_{k}"):
            os.makedirs(f"exp6_filelists/k_{k}")
        for read_type in read_types:
            if not os.path.isdir(f"exp6_filelists/k_{k}/{read_type}"):
                os.makedirs(f"exp6_filelists/k_{k}/{read_type}")
            intersect_files = []
            pivot_files = []
            for pivot in range(1, num_datasets+1):
                pivot_files.append("/exp6_text_dump/k_{k}/{read_type}/pivot/pivot_{pivot}.txt".format(k = k, pivot = pivot, read_type = read_type))
                for num in range(1, num_datasets+1):
                    intersect_files.append("/exp6_text_dump/k_{k}/{read_type}/intersection/pivot_{pivot}/pivot_{pivot}_intersect_dataset_{num}.txt".format(k = k, pivot = pivot, num = num, read_type = read_type))
            with open(f"exp6_filelists/k_{k}/{read_type}/pivots_filelist.txt", "w") as fd:
                for f in pivot_files:
                    fd.write(base_dir+f+"\n")
            with open(f"exp6_filelists/k_{k}/{read_type}/intersections_filelist.txt", "w") as fd:
                for f in intersect_files:
                    fd.write(base_dir+f+"\n")
    

####################################################
# Section 2: Helper functions needed for these
#            experiment rules.
####################################################

def get_input_data_for_exp6(wildcards):
    """ Returns a list of files needed for the initial copying of the data """
    input_files = []
    input_files.append(f"{database_root}/trial_{curr_trial}/exp0_pivot_genomes/dataset_{wildcards.num}/pivot_{wildcards.num}.fna.gz")
    input_files.append(f"{database_root}/trial_{curr_trial}/exp0_pivot_reads/dataset_{wildcards.num}/illumina/pivot_{wildcards.num}_subset.fa")
    input_files.append(f"{database_root}/trial_{curr_trial}/exp0_pivot_reads/dataset_{wildcards.num}/ont/pivot_{wildcards.num}_subset.fa")
    input_files.append(f"{database_root}/trial_{curr_trial}/exp0_nonpivot_genomes/dataset_{wildcards.num}/nonpivot_names.txt")
    return input_files

def get_all_raw_genome_files_in_dataset_exp6(wildcards):
    """ Return path to all the raw genome files in dataset """
    input_files = []
    for data_file in os.listdir(f"exp6_input/rest_of_set/dataset_{wildcards.num}"):
        if data_file.endswith(".fna.gz"):
            input_files.append(f"exp6_input/rest_of_set/dataset_{wildcards.num}/{data_file}")
    return input_files

def get_all_genome_sets_in_dataset_exp6(wildcards):
    """ Returns a list of database in a certain dataset """
    input_files = []
    for data_file in os.listdir(f"exp6_input/rest_of_set/dataset_{wildcards.num}/"):
        if data_file.endswith(".fna.gz"):
            file_name = data_file.split(".fna.gz")[0]
            input_files.append(f"exp6_genome_sets/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/{file_name}.transformed.kmc_pre")
    return input_files

def get_filelists_and_text_dumps_exp6(wildcards):
    """ Generates text dumps needed for filelists, then returns filelists """
    needed_files = []
    needed_files.append("exp6_filelists/k_{k}/{read_type}/intersections_filelist.txt".format(k = wildcards.k, read_type = wildcards.read_type))
    needed_files.append("exp6_filelists/k_{k}/{read_type}/pivots_filelist.txt".format(k = wildcards.k, read_type = wildcards.read_type))
    for pivot in range(1, num_datasets+1):
            needed_files.append("exp6_text_dump/k_{k}/{read_type}/pivot/pivot_{pivot}.txt".format(k = wildcards.k, pivot = pivot, read_type = wildcards.read_type))
            for num in range(1, num_datasets+1):
                needed_files.append("exp6_text_dump/k_{k}/{read_type}/intersection/pivot_{pivot}/pivot_{pivot}_intersect_dataset_{num}.txt".format(k = wildcards.k, pivot = pivot, num = num, read_type = wildcards.read_type))
    return needed_files

####################################################
# Section 3: Rules needed for this experiment type
####################################################
        
#   Section 3.1: Builds KMC databases for all the non-pivot
#   genomes. Also, builds KMC databases over the simulated
#   reads from the pivot genomes from each dataset.

rule build_kmc_database_on_genome_exp6:
    input:
        "exp6_input/rest_of_set/dataset_{num}/{genome}.fna.gz"
    output:
        "exp6_intermediate/step_1/rest_of_set/k_{k}/dataset_{num}/{genome}.kmc_pre",
        "exp6_intermediate/step_1/rest_of_set/k_{k}/dataset_{num}/{genome}.kmc_suf"
    shell:
        "kmc -fm -m64 -k{wildcards.k} -ci1 {input} exp6_intermediate/step_1/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/{wildcards.genome} tmp/"

rule build_kmc_database_on_pivot_exp6:
    input:
        "exp6_input/pivot_reads_subset/{read_type}/pivot_{num}.fa"
    output:
        "exp6_intermediate/step_1/pivot/k_{k}/{read_type}/pivot_{num}.kmc_pre",
        "exp6_intermediate/step_1/pivot/k_{k}/{read_type}/pivot_{num}.kmc_suf"
    shell:
        """
        kmc -fm -m64 -k{wildcards.k} -ci1 {input} exp6_intermediate/step_1/pivot/k_{wildcards.k}/{wildcards.read_type}/pivot_{wildcards.num} tmp/
        """

#   Section 3.2: Converts each kmer-database for a non-pivot
#   genome to make sure all the kmer's have multiplicity = 1. This
#   step is not necessary, however, it makes it easier to interpret
#   histogram files later on in workflow.

rule transform_genome_to_set_exp6:
    input:
        "exp6_intermediate/step_1/rest_of_set/k_{k}/dataset_{num}/{genome}.kmc_pre",
        "exp6_intermediate/step_1/rest_of_set/k_{k}/dataset_{num}/{genome}.kmc_suf"
    output:
        "exp6_genome_sets/rest_of_set/k_{k}/dataset_{num}/{genome}.transformed.kmc_pre",
        "exp6_genome_sets/rest_of_set/k_{k}/dataset_{num}/{genome}.transformed.kmc_suf"
    shell:
        """
        kmc_tools transform exp6_intermediate/step_1/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/{wildcards.genome} \
        set_counts 1 exp6_genome_sets/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/{wildcards.genome}.transformed

        # Remove step_1 rest of set
        rm {input[0]} {input[1]}
        """

#   Section 3.3: Takes all genomes within a dataset
#   OTHER THAN the pivot, and then unions them together.

rule rest_of_set_union_exp6:
    input:
        get_all_genome_sets_in_dataset_exp6
    output:
        "exp6_unions/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.kmc_pre",
        "exp6_unions/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.kmc_suf"
    shell:
        """
        kmc_tools complex exp6_complex_ops/k_{wildcards.k}/dataset_{wildcards.num}/ops_{wildcards.num}.txt
        
        # remove genome sets for this k and dataset (saves a lot of space)
        rm exp6_genome_sets/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/*.transformed.kmc_*
        """

#   Extra rule: just used as a sanity check for the previous rule to make sure the largest
#   possible kmer multiplicity in the histogram is the number of genomes in the dataset

rule union_histogram_exp6:
    input:
        "exp6_unions/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.kmc_pre",
        "exp6_unions/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.kmc_suf"
    output:
        "exp6_unions/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.hist.txt"
    shell:
        """
        kmc_tools transform \
        exp6_unions/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/dataset_{wildcards.num}.transformed.combined \
        histogram exp6_unions/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/dataset_{wildcards.num}.hist.txt \
        """

#   Section 3.4: Convert the union database of each database into a kmer set, 
#   meaning each kmer should have a multplicity. Again, this is step is not 
#   necessary but it is helpful for interpreting intersection and unions.

rule transform_union_to_set_exp6:
    input: 
        "exp6_unions/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.kmc_pre",
        "exp6_unions/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.kmc_suf"
    output:
        "exp6_genome_sets/unions/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.transformed.kmc_pre",
        "exp6_genome_sets/unions/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.transformed.kmc_suf"
    shell:
     """
        kmc_tools transform exp6_unions/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/dataset_{wildcards.num}.transformed.combined \
        set_counts 1 exp6_genome_sets/unions/k_{wildcards.k}/dataset_{wildcards.num}/dataset_{wildcards.num}.transformed.combined.transformed

        # Remove unions that were used as input (don't need them anymore)
        rm {input[0]} {input[1]}
     """

#   Section 3.5: Performs an intersection between the union of a database
#   and a database of the out-pivot's reads. It also includes a rule
#   to list out the intersection histogram which can be used for sanity 
#   checking since the intersection should not have any kmers that occur only
#   once since we are adding the counts of kmers that occur in both.

rule pivot_intersect_exp6:
    input:
        "exp6_intermediate/step_1/pivot/k_{k}/{read_type}/pivot_{pivot_num}.kmc_pre",
        "exp6_genome_sets/unions/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.transformed.kmc_pre"
    output:
        "exp6_intersection_results/k_{k}/{read_type}/pivot_{pivot_num}/pivot_{pivot_num}_intersect_dataset_{num}.kmc_pre",
        "exp6_intersection_results/k_{k}/{read_type}/pivot_{pivot_num}/pivot_{pivot_num}_intersect_dataset_{num}.kmc_suf"
    shell:
        """
        kmc_tools simple exp6_genome_sets/unions/k_{wildcards.k}/dataset_{wildcards.num}/dataset_{wildcards.num}.transformed.combined.transformed \
        exp6_intermediate/step_1/pivot/k_{wildcards.k}/{wildcards.read_type}/pivot_{wildcards.pivot_num} intersect \
        exp6_intersection_results/k_{wildcards.k}/{wildcards.read_type}/pivot_{wildcards.pivot_num}/pivot_{wildcards.pivot_num}_intersect_dataset_{wildcards.num} -ocsum
        """

rule intersection_histogram_exp6:
    input:
        "exp6_intersection_results/k_{k}/{read_type}/pivot_{pivot_num}/pivot_{pivot_num}_intersect_dataset_{num}.kmc_pre",
        "exp6_intersection_results/k_{k}/{read_type}/pivot_{pivot_num}/pivot_{pivot_num}_intersect_dataset_{num}.kmc_suf"
    output:
        "exp6_intersection_results/k_{k}/{read_type}/pivot_{pivot_num}/pivot_{pivot_num}_intersect_dataset_{num}.hist.txt"
    shell:
        """
        kmc_tools transform \
        exp6_intersection_results/k_{wildcards.k}/{wildcards.read_type}/pivot_{wildcards.pivot_num}/pivot_{wildcards.pivot_num}_intersect_dataset_{wildcards.num} \
        histogram exp6_intersection_results/k_{wildcards.k}/{wildcards.read_type}/pivot_{wildcards.pivot_num}/pivot_{wildcards.pivot_num}_intersect_dataset_{wildcards.num}.hist.txt\
        """

#   Section 3.6 Generates text dump files of 
#   pivot and intersection kmc files since those
#   are needed to build the confusion matrix.

rule pivot_text_dump_exp6:
    input:
        "exp6_intermediate/step_1/pivot/k_{k}/{read_type}/pivot_{num}.kmc_pre",
        "exp6_intermediate/step_1/pivot/k_{k}/{read_type}/pivot_{num}.kmc_suf"
    output:
        "exp6_text_dump/k_{k}/{read_type}/pivot/pivot_{num}.txt"
    shell: 
        """
        kmc_tools transform \
        exp6_intermediate/step_1/pivot/k_{wildcards.k}/{wildcards.read_type}/pivot_{wildcards.num} \
        dump -s exp6_text_dump/k_{wildcards.k}/{wildcards.read_type}/pivot/pivot_{wildcards.num}.txt \
        """

rule intersection_text_dump_exp6:
    input:
        "exp6_intersection_results/k_{k}/{read_type}/pivot_{pivot_num}/pivot_{pivot_num}_intersect_dataset_{num}.kmc_pre",
        "exp6_intersection_results/k_{k}/{read_type}/pivot_{pivot_num}/pivot_{pivot_num}_intersect_dataset_{num}.kmc_suf"
    output:
        "exp6_text_dump/k_{k}/{read_type}/intersection/pivot_{pivot_num}/pivot_{pivot_num}_intersect_dataset_{num}.txt"
    shell:
        """
        kmc_tools transform \
        exp6_intersection_results/k_{wildcards.k}/{wildcards.read_type}/pivot_{wildcards.pivot_num}/pivot_{wildcards.pivot_num}_intersect_dataset_{wildcards.num} \
        dump -s exp6_text_dump/k_{wildcards.k}/{wildcards.read_type}/intersection/pivot_{wildcards.pivot_num}/pivot_{wildcards.pivot_num}_intersect_dataset_{wildcards.num}.txt
        """

#   Section 3.6: Runs Python script to generate a confusion matrix and 
#   values for a single read type and kmer length

rule run_merge_list_exp6:
    input:
        get_filelists_and_text_dumps_exp6
    output:
        "exp6_accuracies/{read_type}/values/k_{k}_accuracy_values.csv",
        "exp6_accuracies/{read_type}/confusion_matrix/k_{k}_confusion_matrix.txt",
        "exp6_accuracies/{read_type}/confusion_matrix/k_{k}_confusion_matrix_with_unidentified.txt"
    shell:
        """
        python3 {repo_dir}/src/merge_lists.py \
        -p exp6_filelists/k_{wildcards.k}/{wildcards.read_type}/pivots_filelist.txt \
        -i exp6_filelists/k_{wildcards.k}/{wildcards.read_type}/intersections_filelist.txt \
        -o exp6_accuracies/{wildcards.read_type}/ \
        -n {num_datasets} \
        -k {wildcards.k}  \
        -r exp6_input/pivot_reads_subset/{wildcards.read_type}/ {feat_level_opt} 

        # Remove text dumps for this k and read type
        rm exp6_text_dump/k_{wildcards.k}/{wildcards.read_type}/intersection/pivot_*/pivot_*_intersect_dataset_*.txt
        rm exp6_text_dump/k_{wildcards.k}/{wildcards.read_type}/pivot/pivot_*.txt
        """

#   Section 3.7: Concatenates confusion matrix summary values
#   for all values of k and read types

rule concatenate_accuracies_exp6:
    input:
        expand("exp6_accuracies/{read_type}/values/k_{k}_accuracy_values.csv", k=k_values, read_type = {"illumina","ont"})
    output:
        f"exp6_accuracies/trial_{curr_trial}_short_acc.csv",
        f"exp6_accuracies/trial_{curr_trial}_long_acc.csv"
    shell:
        """
        printf "k,pivotnum,TP,TN,FP,FN,TP-U,TN-U,FP-U,FN-U\n" > {output[0]}
        printf "k,pivotnum,TP,TN,FP,FN,TP-U,TN-U,FP-U,FN-U\n" > {output[1]}

        cat exp6_accuracies/illumina/values/k_*_accuracy_values.csv >> {output[0]}
        cat exp6_accuracies/ont/values/k_*_accuracy_values.csv >> {output[1]}
        """

rule generate_exp6_output:
    input:
        f"exp6_accuracies/trial_{curr_trial}_short_acc.csv",
        f"exp6_accuracies/trial_{curr_trial}_long_acc.csv"

        
