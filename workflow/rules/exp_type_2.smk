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
#            experiment is called. It generates the
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
    # if not os.path.isdir("tmp"):
    #     os.mkdir("tmp")
    
    # Parse out a pivot from the downloaded dataset
    if not os.path.isdir("exp2_input_data/"):
        for i in range(1, num_datasets+1):
            dir_prefix = f"{database_root}/trial_{curr_trial}/exp0_nonpivot_genomes/dataset_{i}/"
            list_of_files = [dir_prefix + x for x in os.listdir(dir_prefix)]

            if not os.path.isdir(f"exp2_input_data/rest_of_set/dataset_{i}"):
                os.makedirs(f"exp2_input_data/rest_of_set/dataset_{i}")

            # Copy over all the files at first
            for data_file in list_of_files:
                shutil.copy(data_file, f"exp2_input_data/rest_of_set/dataset_{i}")
            
            # Take the pivot and move to new directory
            pivot = f"{database_root}/trial_{curr_trial}/exp0_pivot_genomes/dataset_{i}/pivot_{i}.fna.gz"
            if not os.path.isdir(f"exp2_input_data/pivot/dataset_{i}"):
                os.makedirs(f"exp2_input_data/pivot/dataset_{i}")
            shutil.copy(pivot, f"exp2_input_data/pivot/dataset_{i}/pivot_{i}.fna.gz")

    # Initialize complex ops directory to start writing files
    if not os.path.isdir("exp2_complex_ops/"):
        os.makedirs("exp2_complex_ops/within_groups")
    
    # Build the complex ops files for within groups
    for k in k_values:
        if not os.path.isdir(f"exp2_complex_ops/within_groups/k_{k}"):
            os.mkdir(f"exp2_complex_ops/within_groups/k_{k}")
        
        for num in range(1, num_datasets+1):
            kmc_input_files = []

            for data_file in os.listdir(f"exp2_input_data/rest_of_set/dataset_{num}"):
                if data_file.endswith(".fna.gz"):
                    base_name = data_file.split(".fna.gz")[0]
                    kmc_input_files.append(f"exp2_genome_sets/rest_of_set/k_{k}/dataset_{num}/{base_name}.transformed")
            
            if not os.path.isdir(f"exp2_complex_ops/within_groups/k_{k}/dataset_{num}/"):
                os.mkdir(f"exp2_complex_ops/within_groups/k_{k}/dataset_{num}/")
        
            with open(f"exp2_complex_ops/within_groups/k_{k}/dataset_{num}/within_dataset_{num}.txt", "w") as fd:
                fd.write("INPUT:\n")
                result_str = "("
                for i, path in enumerate(kmc_input_files):
                    fd.write(f"set{i+1} = {path}\n")
                    result_str += "set{} + ".format(i+1)
                result_str = result_str[:-2] + ")"
                fd.write("OUTPUT:\n")
                fd.write(f"exp2_within_databases/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined = {result_str}\n")
                fd.write("OUTPUT_PARAMS:\n-cs5000\n")
    
    if not os.path.isdir("exp2_complex_ops/across_groups"):
        os.mkdir("exp2_complex_ops/across_groups")

    # This loop builds the across_groups operation files
    for k in k_values:
        if not os.path.isdir(f"exp2_complex_ops/across_groups/k_{k}"):
            os.mkdir(f"exp2_complex_ops/across_groups/k_{k}")
        
        # Build a file for each pivot 
        for pivot_num in range(1, num_datasets+1):
            if not os.path.isdir(f"exp2_complex_ops/across_groups/k_{k}/pivot_{pivot_num}"):
                os.mkdir(f"exp2_complex_ops/across_groups/k_{k}/pivot_{pivot_num}")

            kmc_input_files = []
            for i in range(1, num_datasets+1):
                if i != pivot_num:
                    kmc_input_files.append(f"exp2_within_databases/rest_of_set/k_{k}/dataset_{i}/dataset_{i}.transformed.combined.transformed")

            with open(f"exp2_complex_ops/across_groups/k_{k}/pivot_{pivot_num}/across_datasets_pivot_{pivot_num}.txt", "w") as fd:
                fd.write("INPUT:\n")
                result_str = "("
                for i, path in enumerate(kmc_input_files):
                    fd.write(f"set{i+1} = {path}\n")
                    result_str += "set{} + ".format(i+1)
                result_str = result_str[:-2] + ")"
                fd.write("OUTPUT:\n")
                fd.write(f"exp2_across_databases/k_{k}/pivot_{pivot_num}/all_datasets_pivot_{pivot_num}.transformed.combined.transformed.combined = {result_str}\n")
                fd.write("OUTPUT_PARAMS:\n-cs5000\n")


####################################################
# Section 2: Helper functions needed for these
#            experiment rules.
####################################################

def get_num_of_dataset_members_exp2(num):
    """ Return the number of genomes in the rest_of_set folder for a specific dataset """
    num_genomes = 0
    for data_file in os.listdir(f"exp2_input_data/rest_of_set/dataset_{num}/"):
        if data_file.endswith(".fna.gz"):
            num_genomes += 1
    return num_genomes

def get_across_group_union_for_pivot_exp2(wildcards):
    """ Returns the input files for this union operation - union across all groups except pivot group """
    k = wildcards.k
    num = wildcards.num
    input_files = [f"exp2_within_databases/rest_of_set/k_{k}/dataset_{i}/dataset_{i}.transformed.combined.transformed.kmc_pre" for i in range(1, num_datasets+1) if i != num]
    return input_files

def get_all_genomes_in_dataset(wildcards):
    """ Returns a list of database in a certain dataset """
    input_files = []
    for data_file in os.listdir(f"exp2_input_data/rest_of_set/dataset_{wildcards.num}/"):
        if data_file.endswith(".fna.gz"):
            file_name = data_file.split(".fna.gz")[0]
            input_files.append(f"exp2_genome_sets/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/{file_name}.transformed.kmc_pre")
    return input_files



def get_dataset_pivot(wildcards):
    """ Return a path to the pivot for certain dataset """
    pivot = os.listdir(f"exp2_input_data/pivot/dataset_{wildcards.num}/")
    assert len(pivot) == 1
    return [f"exp2_genome_sets/pivot/k_{wildcards.k}/dataset_{wildcards.num}/" + pivot[0].split(".fna.gz")[0] + ".transformed.kmc_pre"]

def get_within_group_histogram_files(wildcards):
    """ Returns all the histogram files for within group experiment in specific order """
    input_files = []
    for num in range(1, num_datasets+1):
        for k in k_values:
            for op in ['subtract', 'intersect']:
                input_files.append(f"exp2_within_dataset_results/k_{k}/dataset_{num}/{op}/dataset_{num}_pivot_{op}_group.hist.txt")
    return input_files

def get_across_group_histogram_files(wildcards):
    """ Returns all the histogram files for across group experiment in specific order """
    input_files = []
    for num in range(1, num_datasets+1):
        for k in k_values:
            for op in ['subtract', 'intersect']:
                input_files.append(f"exp2_across_dataset_results/k_{k}/dataset_{num}/{op}/dataset_{num}_pivot_{op}_group.hist.txt")
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

#   Section 3.1: Builds KMC databases, one rule
#                for pivot and one for all other genomes.

rule build_kmc_database_on_genome_exp2:
    input:
        "exp2_input_data/rest_of_set/dataset_{num}/{genome}.fna.gz"
    output:
        "exp2_build_step/rest_of_set/k_{k}/dataset_{num}/{genome}.kmc_pre",
        "exp2_build_step/rest_of_set/k_{k}/dataset_{num}/{genome}.kmc_suf"
    shell:
        """
        mkdir tmp_{wildcards.genome}_{wildcards.k}/
        kmc -fm -m64 -k{wildcards.k} \
                     -ci1 \
                      exp2_input_data/rest_of_set/dataset_{wildcards.num}/{wildcards.genome}.fna.gz \
                      exp2_build_step/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/{wildcards.genome} \
                      tmp_{wildcards.genome}_{wildcards.k}/
        rmdir tmp_{wildcards.genome}_{wildcards.k}/
        """

rule build_kmc_database_on_pivot_exp2:
    input:
        "exp2_input_data/pivot/dataset_{num}/pivot_{num}.fna.gz"
    output:
        "exp2_build_step/pivot/k_{k}/dataset_{num}/pivot_{num}.kmc_pre",
        "exp2_build_step/pivot/k_{k}/dataset_{num}/pivot_{num}.kmc_suf"
    shell:
        """
        mkdir tmp_pivot_{wildcards.num}_{wildcards.k}/
        kmc -fm -m64 -k{wildcards.k} \
                     -ci1 \
                      {input} \
                      exp2_build_step/pivot/k_{wildcards.k}/dataset_{wildcards.num}/pivot_{wildcards.num} \
                      tmp_pivot_{wildcards.num}_{wildcards.k}/
        rmdir tmp_pivot_{wildcards.num}_{wildcards.k}/
        """


# Section 3.2: Converts each kmer databases into 
#              a set (multiplicity=1) for all databases.

rule transform_genome_to_set_exp2:
    input:
        "exp2_build_step/rest_of_set/k_{k}/dataset_{num}/{genome}.kmc_pre",
        "exp2_build_step/rest_of_set/k_{k}/dataset_{num}/{genome}.kmc_suf"
    output:
        "exp2_genome_sets/rest_of_set/k_{k}/dataset_{num}/{genome}.transformed.kmc_pre",
        "exp2_genome_sets/rest_of_set/k_{k}/dataset_{num}/{genome}.transformed.kmc_suf"
    shell:
        """
        kmc_tools transform \
                  exp2_build_step/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/{wildcards.genome} \
                  set_counts 1 \
                  exp2_genome_sets/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/{wildcards.genome}.transformed
        """

rule transform_pivot_to_set_exp2:
    input:
        "exp2_build_step/pivot/k_{k}/dataset_{num}/pivot_{num}.kmc_pre",
        "exp2_build_step/pivot/k_{k}/dataset_{num}/pivot_{num}.kmc_suf"
    output:
        "exp2_genome_sets/pivot/k_{k}/dataset_{num}/pivot_{num}.transformed.kmc_pre",
        "exp2_genome_sets/pivot/k_{k}/dataset_{num}/pivot_{num}.transformed.kmc_suf"
    shell:
        """
        kmc_tools transform \
                  exp2_build_step/pivot/k_{wildcards.k}/dataset_{wildcards.num}/pivot_{wildcards.num} \
                  set_counts 1 \
                  exp2_genome_sets/pivot/k_{wildcards.k}/dataset_{wildcards.num}/pivot_{wildcards.num}.transformed
        """


# Section 3.3: Takes all genomes within a dataset
#              other than the pivot, and then unions them together.

rule within_group_union_exp2:
    input:
        get_all_genomes_in_dataset
    output:
        "exp2_within_databases/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.kmc_pre",
        "exp2_within_databases/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.kmc_suf"
    shell:
        """
        kmc_tools complex \
                  exp2_complex_ops/within_groups/k_{wildcards.k}/dataset_{wildcards.num}/within_dataset_{wildcards.num}.txt
        """


# Section 3.4: Performs the needed operations to generate
#              the plots for the within_group plots.

rule pivot_intersect_within_group_exp2:
    input:
        "exp2_within_databases/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.kmc_pre",
        "exp2_genome_sets/pivot/k_{k}/dataset_{num}/pivot_{num}.transformed.kmc_pre"
    output:
        "exp2_within_dataset_results/k_{k}/dataset_{num}/intersect/dataset_{num}_pivot_intersect_group.kmc_pre",
        "exp2_within_dataset_results/k_{k}/dataset_{num}/intersect/dataset_{num}_pivot_intersect_group.kmc_suf"
    shell:
        """ 
        kmc_tools simple \
                  exp2_genome_sets/pivot/k_{wildcards.k}/dataset_{wildcards.num}/pivot_{wildcards.num}.transformed \
                  exp2_within_databases/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/dataset_{wildcards.num}.transformed.combined \
                  intersect \
                  exp2_within_dataset_results/k_{wildcards.k}/dataset_{wildcards.num}/intersect/dataset_{wildcards.num}_pivot_intersect_group \
                  -ocsum
        """

rule pivot_subtract_within_group_exp2:
    input:
        "exp2_within_databases/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.kmc_pre",
        "exp2_genome_sets/pivot/k_{k}/dataset_{num}/pivot_{num}.transformed.kmc_pre"
    output:
        "exp2_within_dataset_results/k_{k}/dataset_{num}/subtract/dataset_{num}_pivot_subtract_group.kmc_pre",
        "exp2_within_dataset_results/k_{k}/dataset_{num}/subtract/dataset_{num}_pivot_subtract_group.kmc_suf"
    shell:
        """ 
        kmc_tools simple \
                  exp2_genome_sets/pivot/k_{wildcards.k}/dataset_{wildcards.num}/pivot_{wildcards.num}.transformed \
                  exp2_within_databases/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/dataset_{wildcards.num}.transformed.combined \
                  kmers_subtract \
                  exp2_within_dataset_results/k_{wildcards.k}/dataset_{wildcards.num}/subtract/dataset_{wildcards.num}_pivot_subtract_group 
        """

rule within_group_histogram_exp2:
    input:
        "exp2_within_dataset_results/k_{k}/dataset_{num}/{op}/dataset_{num}_pivot_{op}_group.kmc_pre",
        "exp2_within_dataset_results/k_{k}/dataset_{num}/{op}/dataset_{num}_pivot_{op}_group.kmc_suf"
    output:
        "exp2_within_dataset_results/k_{k}/dataset_{num}/{op}/dataset_{num}_pivot_{op}_group.hist.txt"
    shell:
        """
        kmc_tools transform \
                  exp2_within_dataset_results/k_{wildcards.k}/dataset_{wildcards.num}/{wildcards.op}/dataset_{wildcards.num}_pivot_{wildcards.op}_group \
                  histogram \
                  exp2_within_dataset_results/k_{wildcards.k}/dataset_{wildcards.num}/{wildcards.op}/dataset_{wildcards.num}_pivot_{wildcards.op}_group.hist.txt 
        """


# Section 3.5: Performs the within-group analysis, analyzes the specificity
#              of the pivot genome's kmer with respect to other genomes in 
#              same dataset

rule within_group_analysis_exp2:
    input:
        get_within_group_histogram_files
    output:
        "exp2_within_dataset_analysis/within_dataset_analysis.csv"
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


# Section 3.6: Transform each rest_of_set datbase into counts of 1

rule transform_rest_of_set_to_single_counts_exp2:
    input:
        "exp2_within_databases/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.kmc_pre",
        "exp2_within_databases/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.kmc_suf"
    output:
        "exp2_within_databases/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.transformed.kmc_pre",
        "exp2_within_databases/rest_of_set/k_{k}/dataset_{num}/dataset_{num}.transformed.combined.transformed.kmc_suf"
    shell:
        """
        kmc_tools transform \
                  exp2_within_databases/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/dataset_{wildcards.num}.transformed.combined \
                  set_counts 1 \
                  exp2_within_databases/rest_of_set/k_{wildcards.k}/dataset_{wildcards.num}/dataset_{wildcards.num}.transformed.combined.transformed
        """

# Section 3.7: Generate the across_groups databases for each pivot

rule across_group_union_for_pivot_exp2:
    input:
        get_across_group_union_for_pivot_exp2
    output:
        "exp2_across_databases/k_{k}/pivot_{num}/all_datasets_pivot_{num}.transformed.combined.transformed.combined.kmc_pre",
        "exp2_across_databases/k_{k}/pivot_{num}/all_datasets_pivot_{num}.transformed.combined.transformed.combined.kmc_suf"
    shell:
        """
        kmc_tools complex \
                  exp2_complex_ops/across_groups/k_{wildcards.k}/pivot_{wildcards.num}/across_datasets_pivot_{wildcards.num}.txt
        """


# Section 3.8: Generate the databases needed for the across group analysis

rule pivot_intersect_across_group_exp2:
    input:
        "exp2_across_databases/k_{k}/pivot_{num}/all_datasets_pivot_{num}.transformed.combined.transformed.combined.kmc_pre",
        "exp2_genome_sets/pivot/k_{k}/dataset_{num}/pivot_{num}.transformed.kmc_pre"
    output:
        "exp2_across_dataset_results/k_{k}/dataset_{num}/intersect/dataset_{num}_pivot_intersect_group.kmc_pre",
        "exp2_across_dataset_results/k_{k}/dataset_{num}/intersect/dataset_{num}_pivot_intersect_group.kmc_suf"
    shell:
        """ 
        kmc_tools simple \
                  exp2_genome_sets/pivot/k_{wildcards.k}/dataset_{wildcards.num}/pivot_{wildcards.num}.transformed \
                  exp2_across_databases/k_{wildcards.k}/pivot_{wildcards.num}/all_datasets_pivot_{wildcards.num}.transformed.combined.transformed.combined intersect  \
                  exp2_across_dataset_results/k_{wildcards.k}/dataset_{wildcards.num}/intersect/dataset_{wildcards.num}_pivot_intersect_group \
                  -ocsum
        """

rule pivot_subtract_across_group_exp2:
    input:
        "exp2_across_databases/k_{k}/pivot_{num}/all_datasets_pivot_{num}.transformed.combined.transformed.combined.kmc_pre",
        "exp2_genome_sets/pivot/k_{k}/dataset_{num}/pivot_{num}.transformed.kmc_pre"
    output:
        "exp2_across_dataset_results/k_{k}/dataset_{num}/subtract/dataset_{num}_pivot_subtract_group.kmc_pre",
        "exp2_across_dataset_results/k_{k}/dataset_{num}/subtract/dataset_{num}_pivot_subtract_group.kmc_suf"
    shell:
        """ 
        kmc_tools simple \
                  exp2_genome_sets/pivot/k_{wildcards.k}/dataset_{wildcards.num}/pivot_{wildcards.num}.transformed \
                  exp2_across_databases/k_{wildcards.k}/pivot_{wildcards.num}/all_datasets_pivot_{wildcards.num}.transformed.combined.transformed.combined \
                  kmers_subtract \
                  exp2_across_dataset_results/k_{wildcards.k}/dataset_{wildcards.num}/subtract/dataset_{wildcards.num}_pivot_subtract_group
        """

rule across_group_histogram_exp2:
    input:
        "exp2_across_dataset_results/k_{k}/dataset_{num}/{op}/dataset_{num}_pivot_{op}_group.kmc_pre",
        "exp2_across_dataset_results/k_{k}/dataset_{num}/{op}/dataset_{num}_pivot_{op}_group.kmc_suf"
    output:
        "exp2_across_dataset_results/k_{k}/dataset_{num}/{op}/dataset_{num}_pivot_{op}_group.hist.txt"
    shell:
        """
        kmc_tools transform \
                  exp2_across_dataset_results/k_{wildcards.k}/dataset_{wildcards.num}/{wildcards.op}/dataset_{wildcards.num}_pivot_{wildcards.op}_group \
                  histogram \
                  exp2_across_dataset_results/k_{wildcards.k}/dataset_{wildcards.num}/{wildcards.op}/dataset_{wildcards.num}_pivot_{wildcards.op}_group.hist.txt
        """

# Section 3.9: Performs the across-group analysis, looks at each pivot
#              and analyzes how many of the other datasets does it occur in.

rule across_group_analysis_exp2:
    input:
        get_across_group_histogram_files
    output:
        "exp2_across_dataset_analysis/across_dataset_analysis.csv"
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


# Section 3.10: Overall rule to generate all the results

rule generate_exp2_output:
    input:
        "exp2_within_dataset_analysis/within_dataset_analysis.csv",
        "exp2_across_dataset_analysis/across_dataset_analysis.csv"
    output:
        "exp2_final_analysis/within_dataset_analysis.csv",
        "exp2_final_analysis/across_dataset_analysis.csv"
    shell:
        """
        cp {input[0]} {output[0]}
        cp {input[1]} {output[1]}
        """
