####################################################
# Name: exp_type_2.smk
# Description: Contains functions and rules for
#              the type of experiment type 2:
#
#              Focuses on the kmer set of a particular 
#              out-pivot genome and looks at how unique its
#              k-mer set when compared to other groups of genomes.
#
# Date: 2/3/22
####################################################

####################################################
# Section 1: Code that is always run, when this 
#            experiment is called. It genereates the
#            needed files.
# 
#            Writes the input files for KMC complex 
#            operations ...
#
# NOTE: the complex operations requires an input file,
#       and this code generates those files, and saves
#       them in the the complex_ops folder.
####################################################

if exp_type == 2:
    # Make tmp directory for kmc
    if not os.path.isdir("tmp"):
        os.mkdir("tmp")
    
    # Parse out a pivot from the downloaded dataset
    if not os.path.isdir("input_type_2/"):
        for i in range(1, num_datasets+1):
            dir_prefix = f"{database_root}/trial_{curr_trial}/exp0_nonpivot_genomes/dataset_{i}/"
            list_of_files = [dir_prefix + x for x in os.listdir(dir_prefix)]

            if not os.path.isdir(f"input_type_2/rest_of_set/dataset_{i}"):
                os.makedirs(f"input_type_2/rest_of_set/dataset_{i}")

            # Copy over all the files at first
            for data_file in list_of_files:
                shutil.copy(data_file, f"input_type_2/rest_of_set/dataset_{i}")
            
            # Take the pivot and move to new directory
            pivot = f"{database_root}/trial_{curr_trial}/exp0_pivot_genomes/dataset_{i}/pivot_{i}.fna.gz"
            if not os.path.isdir(f"input_type_2/pivot/dataset_{i}"):
                os.makedirs(f"input_type_2/pivot/dataset_{i}")
            shutil.copy(pivot, f"input_type_2/pivot/dataset_{i}/pivot_{i}.fna.gz")

    # Initialize complex ops directory to start writing files
    if not os.path.isdir("complex_ops_type_2/"):
        os.makedirs("complex_ops_type_2/within_groups")
    
    # Build the complex ops files for within groups
    for k in k_values:
        if not os.path.isdir(f"complex_ops_type_2/within_groups/k_{k}"):
            os.mkdir(f"complex_ops_type_2/within_groups/k_{k}")
        
        for num in range(1, num_datasets+1):
            kmc_input_files = []

            for data_file in os.listdir(f"input_type_2/rest_of_set/dataset_{num}"):
                if data_file.endswith(".fna.gz"):
                    base_name = data_file.split(".fna.gz")[0]
                    kmc_input_files.append(f"genome_sets_type_2/rest_of_set/k_{k}/dataset_{num}/{base_name}.transformed")
            
            if not os.path.isdir(f"complex_ops_type_2/within_groups/k_{k}/dataset_{num}/"):
                os.mkdir(f"complex_ops_type_2/within_groups/k_{k}/dataset_{num}/")
        
            with open(f"complex_ops_type_2/within_groups/k_{k}/dataset_{num}/within_dataset_{num}.txt", "w") as fd:
                fd.write("INPUT:\n")
                result_str = "("
                for i, path in enumerate(kmc_input_files):
                    fd.write(f"set{i+1} = {path}\n")
                    result_str += "set{} + ".format(i+1)
                result_str = result_str[:-2] + ")"
                fd.write("OUTPUT:\n")
                fd.write(f"within_databases_type_2/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined = {result_str}\n")
                fd.write("OUTPUT_PARAMS:\n-cs5000\n")
    
    if not os.path.isdir("complex_ops_type_2/across_groups"):
        os.mkdir("complex_ops_type_2/across_groups")

    # This loop builds the across_groups operation files
    for k in k_values:
        if not os.path.isdir(f"complex_ops_type_2/across_groups/k_{k}"):
            os.mkdir(f"complex_ops_type_2/across_groups/k_{k}")
        
        # Build a file for each pivot 
        for pivot_num in range(1, num_datasets+1):
            if not os.path.isdir(f"complex_ops_type_2/across_groups/k_{k}/pivot_{pivot_num}"):
                os.mkdir(f"complex_ops_type_2/across_groups/k_{k}/pivot_{pivot_num}")

            kmc_input_files = []
            for i in range(1, num_datasets+1):
                if i != pivot_num:
                    kmc_input_files.append(f"within_databases_type_2/rest_of_set/k_{k}/dataset_{i}/dataset_{i}.transformed.combined.transformed")

            with open(f"complex_ops_type_2/across_groups/k_{k}/pivot_{pivot_num}/across_datasets_pivot_{pivot_num}.txt", "w") as fd:
                fd.write("INPUT:\n")
                result_str = "("
                for i, path in enumerate(kmc_input_files):
                    fd.write(f"set{i+1} = {path}\n")
                    result_str += "set{} + ".format(i+1)
                result_str = result_str[:-2] + ")"
                fd.write("OUTPUT:\n")
                fd.write(f"across_databases_type_2/k_{k}/pivot_{pivot_num}/all_datasets_pivot_{pivot_num}.transformed.combined.transformed.combined = {result_str}\n")
                fd.write("OUTPUT_PARAMS:\n-cs5000\n")


####################################################
# Section 2: Helper functions needed for these
#            experiment rules.
####################################################

# def get_input_data_for_exp2(wildcards):
#     """ Returns a list of files needed for the initial copying of the data """
#     input_files = []
#     input_files.append(f"{database_root}/trial_{curr_trial}/exp0_pivot_genomes/dataset_{wildcards.num}/pivot_{wildcards.num}.fna.gz")
#     input_files.append(f"{database_root}/trial_{curr_trial}/exp0_nonpivot_genomes/dataset_{wildcards.num}/nonpivot_names.txt")
#     return input_files

def get_num_of_dataset_members_exp2(num):
    """ Return the number of genomes in the rest_of_set folder for a specific dataset """
    num_genomes = 0
    for data_file in os.listdir(f"input_type_2/rest_of_set/dataset_{num}/"):
        if data_file.endswith(".fna.gz"):
            num_genomes += 1
    return num_genomes

def get_across_group_union_for_pivot(wildcards):
    """ Returns the input files for this union operation - union across all groups except pivot group """
    k = wildcards.k
    num = wildcards.num
    input_files = [f"within_databases_type_2/rest_of_set/k_{k}/dataset_{i}/dataset_{i}.transformed.combined.transformed.kmc_pre" for i in range(1, num_datasets+1) if i != num]
    return input_files

def get_all_genomes_in_dataset(wildcards):
    """ Returns a list of database in a certain dataset """
    input_files = []
    for data_file in os.listdir(f"input_type_2/rest_of_set/dataset_{wildcards.num}/"):
        if data_file.endswith(".fna.gz"):
            file_name = data_file.split(".fna.gz")[0]
            input_files.append(f"genome_sets_type_2/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/{file_name}.transformed.kmc_pre")
    return input_files

def get_dataset_pivot(wildcards):
    """ Return a path to the pivot for certain dataset """
    pivot = os.listdir(f"input_type_2/pivot/dataset_{wildcards.num}/")
    assert len(pivot) == 1
    return [f"genome_sets_type_2/pivot/k_{wildcards.k}/dataset_{wildcards.num}/" + pivot[0].split(".fna.gz")[0] + ".transformed.kmc_pre"]

def get_within_group_histogram_files(wildcards):
    """ Returns all the histogram files for within group experiment in specific order """
    input_files = []
    for num in range(1, num_datasets+1):
        for k in k_values:
            for op in ['subtract', 'intersect']:
                input_files.append(f"within_dataset_results_type_2/k_{k}/dataset_{num}/{op}/dataset_{num}_pivot_{op}_group.hist.txt")
    return input_files

def get_across_group_histogram_files(wildcards):
    """ Returns all the histogram files for across group experiment in specific order """
    input_files = []
    for num in range(1, num_datasets+1):
        for k in k_values:
            for op in ['subtract', 'intersect']:
                input_files.append(f"across_dataset_results_type_2/k_{k}/dataset_{num}/{op}/dataset_{num}_pivot_{op}_group.hist.txt")
    return input_files

def summarize_histogram_type2(sub_counts, inter_counts, num_genomes_in_dataset, across_group_analysis, k):
    """ 
        Takes in histogram of kmer occurrences, and returns the metrics for the 
        bar charts summarized below:

            %_1_occ - percentage of unique kmers that only occur in one genome
            %_25_or_less - percentage of unique kmers that occur in multiple genomes, in 25% of the genomes or less
            %_25_to_75 - percentage of unique kmers that occur in multiple genomes, in 25% to 75% of the genomes
            %_75_or_more - percentage of unique kmers that occur in multiple genomes, in 75% or more of the genomes
            unique_stat - weighted sum of kmer occurrents = SUM([occ * %_unique_occ for occ in range(255)]) 
    """
    # Perform a couple of assertions to start ...
    assert inter_counts[0] == 0, "intersection counts should have 0 unique kmers"
    assert sum(sub_counts[1:]) == 0, "all of kmers in sub_counts should be unique"

    # Start to calculate the metrics ...
    metrics = [0 for i in range(7)]
    total_unique_kmers = sum(sub_counts) + sum(inter_counts)

    boundaries = [0.25, 0.75]
    boundary_indices = [max(int(percent * num_genomes_in_dataset), 1) for percent in boundaries]

    # Special cases where indices are customized ...
    if across_group_analysis:
        boundary_indices = [3, 8]

    metrics[0] = round(sub_counts[0]/total_unique_kmers, 3)
    metrics[1] = round(sum([inter_counts[i] for i in range(1, boundary_indices[0])])/total_unique_kmers, 3)
    metrics[2] = round(sum([inter_counts[i] for i in range(boundary_indices[0], boundary_indices[1])])/total_unique_kmers, 3)
    metrics[3] = round(sum([inter_counts[i] for i in range(boundary_indices[1], len(inter_counts))])/total_unique_kmers, 3)

    rounding_error = abs(sum(metrics[0:4])-1) 
    assert rounding_error < 0.05, "Issue occurred with histogram summarization"

    # Both unnormalized, and normalized uniqueness statistic
    metrics[4] = (1 * sub_counts[0]/total_unique_kmers)
    metrics[4] += sum([((i+1) * (inter_counts[i]/total_unique_kmers)) for i in range(1, len(inter_counts))])
    metrics[4] = round(metrics[4], 4)

    metrics[5] = ((1/num_genomes_in_dataset) * sub_counts[0]/total_unique_kmers)
    metrics[5] += sum([(((i+1)/num_genomes_in_dataset) * (inter_counts[i]/total_unique_kmers)) for i in range(1, len(inter_counts))])
    metrics[5] = round(metrics[5], 4)

    # Calculate the fraction used by the delta measure
    metrics[6] = round(total_unique_kmers/k, 4)
    return metrics


####################################################
# Section 3: Rules needed for this experiment type
####################################################

#   Section 3.0: Copy over the data from the database folder
#   into the expected structure for the workflow. Build the
#   complex ops files used for KMC.

# rule copy_over_raw_genomes_for_exp2:
#     input:
#         get_input_data_for_exp2
#     output:
#         "input_type_2/pivot/dataset_{num}/pivot_{num}.fna.gz",
#         "input_type_2/rest_of_set/dataset_{num}/nonpivot_names.txt"
#     shell:
#         """
#         cp {input[0]} {output[0]}
#         cp {input[1]} {output[1]}
#         cp {database_root}/trial_{curr_trial}/exp0_nonpivot_genomes/dataset_{wildcards.num}/*.fna.gz input_type_2/rest_of_set/dataset_{wildcards.num}/
#         """

# rule build_complex_ops_within_files_for_exp2:
#     input:
#         "input_type_2/rest_of_set/dataset_{num}/nonpivot_names.txt"
#     output:
#         "complex_ops_type_2/within_groups/k_{k}/dataset_{num}/within_dataset_{num}.txt"
#     run:
#         kmc_input_files = []
#         for data_file in os.listdir(f"input_type_2/rest_of_set/dataset_{wildcards.num}"):
#             if data_file.endswith(".fna.gz"):
#                 base_name = data_file.split(".fna.gz")[0]
#                 kmc_input_files.append(f"genome_sets_type_2/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/{base_name}.transformed")

#         with open(output[0], "w") as fd:
#             fd.write("INPUT:\n")
#             result_str = "("
#             for i, path in enumerate(kmc_input_files):
#                 fd.write(f"set{i+1} = {path}\n")
#                 result_str += "set{} + ".format(i+1)
#             result_str = result_str[:-2] + ")"
#             fd.write("OUTPUT:\n")
#             fd.write(f"within_databases_type_2/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/dataset_{wildcards.num}.transformed.combined = {result_str}\n")
#             fd.write("OUTPUT_PARAMS:\n-cs5000\n")

# rule build_complex_ops_across_files_for_exp2:
#     input:
#         expand("input_type_2/rest_of_set/dataset_{num}/nonpivot_names.txt", num=range(1,num_datasets+1))
#     output:
#         "complex_ops_type_2/across_groups/k_{k}/pivot_{pivot_num}/across_datasets_pivot_{pivot_num}.txt"
#     run:
#             kmc_input_files = []
#             for i in range(1, num_datasets+1):
#                 if i != wildcards.pivot_num:
#                     kmc_input_files.append(f"within_databases_type_2/rest_of_set/k_{wildcards.k}/dataset_{i}/dataset_{i}.transformed.combined.transformed")

#             with open(output[0], "w") as fd:
#                 fd.write("INPUT:\n")
#                 result_str = "("
#                 for i, path in enumerate(kmc_input_files):
#                     fd.write(f"set{i+1} = {path}\n")
#                     result_str += "set{} + ".format(i+1)
#                 result_str = result_str[:-2] + ")"
#                 fd.write("OUTPUT:\n")
#                 fd.write(f"across_databases_type_2/k_{wildcards.k}/pivot_{wildcards.pivot_num}/all_datasets_pivot_{wildcards.pivot_num}.transformed.combined.transformed.combined = {result_str}\n")
#                 fd.write("OUTPUT_PARAMS:\n-cs5000\n")


#   Section 3.1: Builds KMC databases, one rule
#   for pivot and one for all other genomes.

rule build_kmc_database_on_genome_exp_type_2:
    input:
        "input_type_2/rest_of_set/dataset_{num}/{genome}.fna.gz"
        #"input_type_2/rest_of_set/dataset_{num}/nonpivot_names.txt"
    output:
        "step_1_type_2/rest_of_set/k_{k}/dataset_{num}/{genome}.kmc_pre",
        "step_1_type_2/rest_of_set/k_{k}/dataset_{num}/{genome}.kmc_suf"
    shell:
        """
        mkdir tmp_{wildcards.genome}_{wildcards.k}/
        kmc -fm -m64 -k{wildcards.k} -ci1 input_type_2/rest_of_set/dataset_{wildcards.num}/{wildcards.genome}.fna.gz step_1_type_2/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/{wildcards.genome} tmp_{wildcards.genome}_{wildcards.k}/
        rmdir tmp_{wildcards.genome}_{wildcards.k}/
        """

rule build_kmc_database_on_pivot_exp_type_2:
    input:
        "input_type_2/pivot/dataset_{num}/pivot_{num}.fna.gz"
    output:
        "step_1_type_2/pivot/k_{k}/dataset_{num}/pivot_{num}.kmc_pre",
        "step_1_type_2/pivot/k_{k}/dataset_{num}/pivot_{num}.kmc_suf"
    shell:
        """
        mkdir tmp_pivot_{wildcards.num}_{wildcards.k}/
        kmc -fm -m64 -k{wildcards.k} -ci1 {input} step_1_type_2/pivot/k_{wildcards.k}/dataset_{wildcards.num}/pivot_{wildcards.num} tmp_pivot_{wildcards.num}_{wildcards.k}/
        rmdir tmp_pivot_{wildcards.num}_{wildcards.k}/
        """


#   Section 3.2: Converts each kmer database into 
#   a set (multiplicity=1) for all databases.

rule transform_genome_to_set_exp_type2:
    input:
        "step_1_type_2/rest_of_set/k_{k}/dataset_{num}/{genome}.kmc_pre",
        "step_1_type_2/rest_of_set/k_{k}/dataset_{num}/{genome}.kmc_suf"
    output:
        "genome_sets_type_2/rest_of_set/k_{k}/dataset_{num}/{genome}.transformed.kmc_pre",
        "genome_sets_type_2/rest_of_set/k_{k}/dataset_{num}/{genome}.transformed.kmc_suf"
    shell:
        """
        kmc_tools transform step_1_type_2/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/{wildcards.genome} \
        set_counts 1 genome_sets_type_2/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/{wildcards.genome}.transformed
        """

rule transform_pivot_to_set_exp_type2:
    input:
        "step_1_type_2/pivot/k_{k}/dataset_{num}/pivot_{num}.kmc_pre",
        "step_1_type_2/pivot/k_{k}/dataset_{num}/pivot_{num}.kmc_suf"
    output:
        "genome_sets_type_2/pivot/k_{k}/dataset_{num}/pivot_{num}.transformed.kmc_pre",
        "genome_sets_type_2/pivot/k_{k}/dataset_{num}/pivot_{num}.transformed.kmc_suf"
    shell:
        """
        kmc_tools transform step_1_type_2/pivot/k_{wildcards.k}/dataset_{wildcards.num}/pivot_{wildcards.num} \
        set_counts 1 genome_sets_type_2/pivot/k_{wildcards.k}/dataset_{wildcards.num}/pivot_{wildcards.num}.transformed
        """

#   Section 3.3: Takes all genomes within a dataset
#   OTHER THAN the pivot, and then unions them together.

rule within_group_union_exp_type2:
    input:
        get_all_genomes_in_dataset
    output:
        "within_databases_type_2/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.kmc_pre",
        "within_databases_type_2/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.kmc_suf"
    shell:
        "kmc_tools complex complex_ops_type_2/within_groups/k_{wildcards.k}/dataset_{wildcards.num}/within_dataset_{wildcards.num}.txt"


#   Section 3.4: Performs the needed operations to generate
#   the plots for the within_group plots.

rule pivot_intersect_within_group_exp_type2:
    input:
        "within_databases_type_2/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.kmc_pre",
        "genome_sets_type_2/pivot/k_{k}/dataset_{num}/pivot_{num}.transformed.kmc_pre"
    output:
        "within_dataset_results_type_2/k_{k}/dataset_{num}/intersect/dataset_{num}_pivot_intersect_group.kmc_pre",
        "within_dataset_results_type_2/k_{k}/dataset_{num}/intersect/dataset_{num}_pivot_intersect_group.kmc_suf"
    shell:
        """ 
        kmc_tools simple genome_sets_type_2/pivot/k_{wildcards.k}/dataset_{wildcards.num}/pivot_{wildcards.num}.transformed \
        within_databases_type_2/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/dataset_{wildcards.num}.transformed.combined intersect  \
        within_dataset_results_type_2/k_{wildcards.k}/dataset_{wildcards.num}/intersect/dataset_{wildcards.num}_pivot_intersect_group -ocsum
        """

rule pivot_subtract_within_group_exp_type2:
    input:
        "within_databases_type_2/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.kmc_pre",
        "genome_sets_type_2/pivot/k_{k}/dataset_{num}/pivot_{num}.transformed.kmc_pre"
    output:
        "within_dataset_results_type_2/k_{k}/dataset_{num}/subtract/dataset_{num}_pivot_subtract_group.kmc_pre",
        "within_dataset_results_type_2/k_{k}/dataset_{num}/subtract/dataset_{num}_pivot_subtract_group.kmc_suf"
    shell:
        """ 
        kmc_tools simple genome_sets_type_2/pivot/k_{wildcards.k}/dataset_{wildcards.num}/pivot_{wildcards.num}.transformed \
        within_databases_type_2/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/dataset_{wildcards.num}.transformed.combined kmers_subtract \
        within_dataset_results_type_2/k_{wildcards.k}/dataset_{wildcards.num}/subtract/dataset_{wildcards.num}_pivot_subtract_group 
        """

rule within_group_histogram_exp_type2:
    input:
        "within_dataset_results_type_2/k_{k}/dataset_{num}/{op}/dataset_{num}_pivot_{op}_group.kmc_pre",
        "within_dataset_results_type_2/k_{k}/dataset_{num}/{op}/dataset_{num}_pivot_{op}_group.kmc_suf"
    output:
        "within_dataset_results_type_2/k_{k}/dataset_{num}/{op}/dataset_{num}_pivot_{op}_group.hist.txt"
    shell:
        """
        kmc_tools transform \
        within_dataset_results_type_2/k_{wildcards.k}/dataset_{wildcards.num}/{wildcards.op}/dataset_{wildcards.num}_pivot_{wildcards.op}_group \
        histogram within_dataset_results_type_2/k_{wildcards.k}/dataset_{wildcards.num}/{wildcards.op}/dataset_{wildcards.num}_pivot_{wildcards.op}_group.hist.txt \
        """

#   Section 3.5: Performs the within-group analysis, analyzes the specificity
#   of the pivot genome's kmer with respect to other genomes in same dataset

rule within_group_analysis_exp_type2:
    input:
        get_within_group_histogram_files
    output:
        "within_dataset_analysis_type_2/within_dataset_analysis.csv"
    run:
        with open(output[0], "w") as output_fd:
            output_fd.write(f"group_num,k,percent_1_occ,percent_25_or_less,percent_25_to_75,percent_75_or_more,"
                             "unique_stat,unique_stat_norm,delta_frac,delta_frac_norm\n")
            all_metrics = []
            for i in range(0, len(input), 2):
                dataset_num = input[i].split("/")[2].split("_")[1]
                k_value = input[i].split("/")[1].split("_")[1]

                num_genomes_in_dataset = get_num_of_dataset_members_exp2(dataset_num)

                with open(input[i], "r") as sub_fd, open(input[i+1], "r") as inter_fd:
                    sub_results = sub_fd.readlines()
                    sub_counts = [int(record.split()[1]) for record in sub_results]

                    inter_results = inter_fd.readlines()
                    inter_counts = [int(record.split()[1]) for record in inter_results]
                
                metrics = [f"group_{dataset_num}", k_value] + summarize_histogram_type2(sub_counts, inter_counts, num_genomes_in_dataset, False, int(k_value))
                all_metrics.append(metrics)

            # Generate the normalized cardinality/k values for each group
            for i in range(1, num_datasets+1):
                id_str = f"group_{i}"
                values = [metrics[8] for metrics in all_metrics if metrics[0] == id_str]
                max_ratio = max(values)

                # Divide all the values in that group with the max value
                for metrics in all_metrics:
                    if metrics[0] == id_str:
                        metrics.append(round(metrics[8]/max_ratio, 4))
            
            # Print all the values to csv file
            for metrics in all_metrics:
                metrics_str = ",".join([str(x) for x in metrics])
                output_fd.write(f"{metrics_str}\n")


#   Section 3.6: Transform each rest_of_set datbase into counts of 1

rule transform_rest_of_set_to_single_counts:
    input:
        "within_databases_type_2/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.kmc_pre",
        "within_databases_type_2/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.kmc_suf"
    output:
        "within_databases_type_2/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.transformed.kmc_pre",
        "within_databases_type_2/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.transformed.kmc_suf"
    shell:
        """
        kmc_tools transform within_databases_type_2/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/dataset_{wildcards.num}.transformed.combined \
        set_counts 1 within_databases_type_2/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/dataset_{wildcards.num}.transformed.combined.transformed
        """

#   Section 3.7: Generate the across_groups databases for each pivot

rule across_group_union_for_pivot_exp_type2:
    input:
        get_across_group_union_for_pivot
    output:
        "across_databases_type_2/k_{k}/pivot_{num}/all_datasets_pivot_{num}.transformed.combined.transformed.combined.kmc_pre",
        "across_databases_type_2/k_{k}/pivot_{num}/all_datasets_pivot_{num}.transformed.combined.transformed.combined.kmc_suf"
    shell:
        "kmc_tools complex complex_ops_type_2/across_groups/k_{wildcards.k}/pivot_{wildcards.num}/across_datasets_pivot_{wildcards.num}.txt"


#   Section 3.8: Generate the databases needed for the across group analysis

rule pivot_intersect_across_group_exp_type2:
    input:
        "across_databases_type_2/k_{k}/pivot_{num}/all_datasets_pivot_{num}.transformed.combined.transformed.combined.kmc_pre",
        "genome_sets_type_2/pivot/k_{k}/dataset_{num}/pivot_{num}.transformed.kmc_pre"
    output:
        "across_dataset_results_type_2/k_{k}/dataset_{num}/intersect/dataset_{num}_pivot_intersect_group.kmc_pre",
        "across_dataset_results_type_2/k_{k}/dataset_{num}/intersect/dataset_{num}_pivot_intersect_group.kmc_suf"
    shell:
        """ 
        kmc_tools simple genome_sets_type_2/pivot/k_{wildcards.k}/dataset_{wildcards.num}/pivot_{wildcards.num}.transformed \
        across_databases_type_2/k_{wildcards.k}/pivot_{wildcards.num}/all_datasets_pivot_{wildcards.num}.transformed.combined.transformed.combined intersect  \
        across_dataset_results_type_2/k_{wildcards.k}/dataset_{wildcards.num}/intersect/dataset_{wildcards.num}_pivot_intersect_group -ocsum
        """

rule pivot_subtract_across_group_exp_type2:
    input:
        "across_databases_type_2/k_{k}/pivot_{num}/all_datasets_pivot_{num}.transformed.combined.transformed.combined.kmc_pre",
        "genome_sets_type_2/pivot/k_{k}/dataset_{num}/pivot_{num}.transformed.kmc_pre"
    output:
        "across_dataset_results_type_2/k_{k}/dataset_{num}/subtract/dataset_{num}_pivot_subtract_group.kmc_pre",
        "across_dataset_results_type_2/k_{k}/dataset_{num}/subtract/dataset_{num}_pivot_subtract_group.kmc_suf"
    shell:
        """ 
        kmc_tools simple genome_sets_type_2/pivot/k_{wildcards.k}/dataset_{wildcards.num}/pivot_{wildcards.num}.transformed \
        across_databases_type_2/k_{wildcards.k}/pivot_{wildcards.num}/all_datasets_pivot_{wildcards.num}.transformed.combined.transformed.combined kmers_subtract \
        across_dataset_results_type_2/k_{wildcards.k}/dataset_{wildcards.num}/subtract/dataset_{wildcards.num}_pivot_subtract_group
        """

rule across_group_histogram_exp_type2:
    input:
        "across_dataset_results_type_2/k_{k}/dataset_{num}/{op}/dataset_{num}_pivot_{op}_group.kmc_pre",
        "across_dataset_results_type_2/k_{k}/dataset_{num}/{op}/dataset_{num}_pivot_{op}_group.kmc_suf"
    output:
        "across_dataset_results_type_2/k_{k}/dataset_{num}/{op}/dataset_{num}_pivot_{op}_group.hist.txt"
    shell:
        """
        kmc_tools transform \
        across_dataset_results_type_2/k_{wildcards.k}/dataset_{wildcards.num}/{wildcards.op}/dataset_{wildcards.num}_pivot_{wildcards.op}_group \
        histogram across_dataset_results_type_2/k_{wildcards.k}/dataset_{wildcards.num}/{wildcards.op}/dataset_{wildcards.num}_pivot_{wildcards.op}_group.hist.txt \
        """


#   Section 3.9: Performs the across-group analysis, looks at each pivot
#   and analyzes how many of the other datasets does it occur in.

rule across_group_analysis_exp_type2:
    input:
        get_across_group_histogram_files
    output:
        "across_dataset_analysis_type_2/across_dataset_analysis.csv"
    run:
        with open(output[0], "w") as output_fd:
            output_fd.write(f"group_num,k,percent_1_occ,percent_2_to_3,percent_4_to_8,percent_9_more," 
                             "unique_stat,unique_stat_norm,delta_frac,delta_frac_norm\n")
            
            all_metrics = []
            for i in range(0, len(input), 2):
                dataset_num = input[i].split("/")[2].split("_")[1]
                k_value = input[i].split("/")[1].split("_")[1]

                with open(input[i], "r") as sub_fd, open(input[i+1], "r") as inter_fd:
                    sub_results = sub_fd.readlines()
                    sub_counts = [int(record.split()[1]) for record in sub_results]

                    inter_results = inter_fd.readlines()
                    inter_counts = [int(record.split()[1]) for record in inter_results]
                
                metrics = [f"group_{dataset_num}", k_value] + summarize_histogram_type2(sub_counts, inter_counts, num_datasets, True, int(k_value))
                all_metrics.append(metrics)

            # Generate the normalized cardinality/k values for each group
            for i in range(1, num_datasets+1):
                id_str = f"group_{i}"
                values = [metrics[8] for metrics in all_metrics if metrics[0] == id_str]
                max_ratio = max(values)

                # Divide all the values in that group with the max value
                for metrics in all_metrics:
                    if metrics[0] == id_str:
                        metrics.append(round(metrics[8]/max_ratio, 4))
            
            # Print all the values to csv file
            for metrics in all_metrics:
                metrics_str = ",".join([str(x) for x in metrics])
                output_fd.write(f"{metrics_str}\n")


#   Section 3.10: Overall rules to first prepare the data in the working
#   directory, and run the actualy experiment.

# rule prepare_data_for_exp2:
#     input:
#         expand("input_type_2/rest_of_set/dataset_{num}/nonpivot_names.txt", num=range(1, num_datasets+1))

rule generate_exp2_output:
    input:
        "within_dataset_analysis_type_2/within_dataset_analysis.csv",
        "across_dataset_analysis_type_2/across_dataset_analysis.csv"
