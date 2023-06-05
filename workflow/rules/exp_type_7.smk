####################################################
# Name: exp_type_7.smk
# Description: Contains functions and rules for
#              the type of experiment type 7:
# 
#              Simulates reads from an out-pivot genome from
#              each species. Then, determines which species each 
#              half-mem/mem occurs in and builds a confusion matrix.
#
# Date: 7/5/22
####################################################

####################################################
# Section 1: Code that is always run, when this 
#            experiment is called. It genereates the
#            needed files.
####################################################

if exp_type == 7:
    if not os.path.isdir("exp7_non_pivot_data/"):
        for i in range(1, num_datasets+1):
            dir_prefix = f"{database_root}/trial_{curr_trial}/exp0_nonpivot_genomes/dataset_{i}/"
            list_of_files = [dir_prefix + x for x in os.listdir(dir_prefix)]

            if not os.path.isdir(f"exp7_non_pivot_data/dataset_{i}"):
                os.makedirs(f"exp7_non_pivot_data/dataset_{i}")

            # Copy over all the files at first
            for data_file in list_of_files:
                shutil.copy(data_file, f"exp7_non_pivot_data/dataset_{i}")
            
            # Copy the pivot to approriate place
            pivot = f"{database_root}/trial_{curr_trial}/exp0_pivot_genomes/dataset_{i}/pivot_{i}.fna.gz"
            if not os.path.isdir(f"exp7_pivot_data/pivot_ref/"):
                os.makedirs(f"exp7_pivot_data/pivot_ref/")
            shutil.copy(pivot, f"exp7_pivot_data/pivot_ref/pivot_{i}.fna.gz")

            # Copy over reads to approriate places
            if not os.path.isdir(f"exp7_ms_data/illumina/pivot_{i}/"):
                os.makedirs(f"exp7_ms_data/illumina/pivot_{i}/")
            shutil.copy(f"{database_root}/trial_{curr_trial}/exp0_pivot_reads/dataset_{i}/illumina/pivot_{i}_subset.fa",
            f"exp7_ms_data/illumina/pivot_{i}/pivot_{i}.fna")

            if not os.path.isdir(f"exp7_ms_data/ont/pivot_{i}/"):
                os.makedirs(f"exp7_ms_data/ont/pivot_{i}/")
            shutil.copy(f"{database_root}/trial_{curr_trial}/exp0_pivot_reads/dataset_{i}/ont/pivot_{i}_subset.fa",
            f"exp7_ms_data/ont/pivot_{i}/pivot_{i}.fna")


####################################################
# Section 2: Helper functions needed for these
#            experiment rules.
####################################################

def get_dataset_non_pivot_genomes_exp7(wildcards):
    """ Returns list of non-pivot genomes for a given dataset """
    input_files = []
    for data_file in os.listdir(f"exp7_non_pivot_data/dataset_{wildcards.num}/"):
        if data_file.endswith(".fna.gz"):
            file_name = data_file.split(".fna.gz")[0]
            input_files.append(f"exp7_non_pivot_data/dataset_{wildcards.num}/{file_name}.fna")
    return input_files

# def get_all_non_pivot_genomes_exp7(wildcards):
#     """ Returns list of all non-pivot genomes """
#     input_files = []
#     for i in range(1, num_datasets + 1):
#         for data_file in os.listdir(f"exp7_non_pivot_data/dataset_{i}/"):
#             if data_file.endswith(".fna"):
#                 file_name = data_file.split(".fna")[0]
#                 input_files.append(f"exp7_non_pivot_data/dataset_{i}/{file_name}.fna")
#     return input_files

def get_combined_ref_dataset_exp7(wildcards):
    """ Returns list of forward combined refs for each dataset """
    input_files = []
    for i in range(1, num_datasets + 1):
        input_files.append(f"exp7_combined_refs/dataset_{i}/combined_ref_forward.fna")
    return input_files

def get_python_input_exp7(wildcards):
    """ Returns a list of all pivot SAM files for a read type """
    input_files = []
    for i in range(1,num_datasets+1):
        for j in range(1, num_datasets+1):
            input_files.append(f"exp7_sam_files/{wildcards.mem_type}/{wildcards.read_type}/pivot_{i}_align_dataset_{j}.sam")
    return input_files

def get_input_data_for_exp7(wildcards):
    """ Returns a list of files needed for the initial copying of the data """
    input_files = []
    input_files.append(f"{database_root}/trial_{curr_trial}/exp0_pivot_genomes/dataset_{wildcards.num}/pivot_{wildcards.num}.fna.gz")
    input_files.append(f"{database_root}/trial_{curr_trial}/exp0_pivot_reads/dataset_{wildcards.num}/illumina/pivot_{wildcards.num}_subset.fa")
    input_files.append(f"{database_root}/trial_{curr_trial}/exp0_pivot_reads/dataset_{wildcards.num}/ont/pivot_{wildcards.num}_subset.fa")
    input_files.append(f"{database_root}/trial_{curr_trial}/exp0_nonpivot_genomes/dataset_{wildcards.num}/nonpivot_names.txt")
    return input_files

####################################################
# Section 3: Rules needed for this experiment type
####################################################

#   Section 3.1: Decompress the genomes from the input dataset
#   in order to be able to build the reference for SPUMONI

rule decompress_non_pivot_genomes_exp7:
    input:
        "exp7_non_pivot_data/dataset_{num}/{genome}.fna.gz"
    output:
        "exp7_non_pivot_data/dataset_{num}/{genome}.fna"
    shell:
        """
        gzip -d {input} -k
        """

rule decompress_pivot_genome_exp7:
    input:
        "exp7_ms_data/{read_type}/pivot_{pivot}/pivot_{pivot}.fna.gz"
    output:
        "exp7_ms_data/{read_type}/pivot_{pivot}/pivot_{pivot}.fna"
    shell:
        """
        gzip -d {input} -k
        """

#   Section 3.2: Contains rules for building a concatenated reference of all
#   the genomes in each database. Then, building reverse complements, and 
#   concatenating those as well. SPUMONI just uses the forward sequences, but
#   we want the forward + rev. comp. for the r-index

rule build_forward_ref_dataset_exp7:
    input:
        get_dataset_non_pivot_genomes_exp7
    output:
        "exp7_combined_refs/dataset_{num}/combined_ref_forward.fna"
    shell:
        """
        cat {input} > {output}
        """

rule build_rev_comp_ref_dataset_exp7:
    input:
        "exp7_combined_refs/dataset_{num}/combined_ref_forward.fna"
    output:
        "exp7_combined_refs/dataset_{num}/combined_ref_rcomp.fna"
    shell:
        "seqtk seq -r {input} | sed 's/^>/>revcomp_/' > {output}"

rule concat_forward_and_rev_comp_dataset_exp7:
    input:
        "exp7_combined_refs/dataset_{num}/combined_ref_forward.fna",
        "exp7_combined_refs/dataset_{num}/combined_ref_rcomp.fna"
    output:
        "exp7_combined_refs/dataset_{num}/combined_ref_all.fna"
    shell:
        "cat {input} > {output}"

rule concat_only_forward_refs_for_all_datasets_exp7:
    input:
        get_combined_ref_dataset_exp7
    output:
        "exp7_combined_refs/combined_ref_forward_only.fna"
    shell:
        "cat {input} > {output}"

#   Section 3.3: Generate FASTA index that can be used to determine the 
#   total size of the input databases in order to figure out what the 
#   random length matching statistic will be with respect to database

rule build_fasta_index_for_combined_forward_exp7:
    input:
        "exp7_combined_refs/combined_ref_forward_only.fna"
    output:
        "exp7_combined_ref_fai/combined_ref_forward_only.fna.fai"
    shell:
        """
        cp {input} exp7_combined_ref_fai/combined_ref_forward_only.fna
        samtools faidx exp7_combined_ref_fai/combined_ref_forward_only.fna
        """

#   Section 3.4: Generate matching statistics for pivot reads with 
#   respect to the full database of all the non-pivot genomes.

rule build_spumoni_index_exp7:
    input:
        "exp7_combined_refs/combined_ref_forward_only.fna"
    output:
        "exp7_indexes/spumoni_index/spumoni_full_ref.fa",
        "exp7_indexes/spumoni_index/spumoni_full_ref.fa.thrbv.ms"
    shell:
        # Only use forward references for spumoni
        """
        cp {input} exp7_indexes/spumoni_index/combined_ref_forward_only.fna
        spumoni build -r exp7_indexes/spumoni_index/combined_ref_forward_only.fna -M -n -o exp7_indexes/spumoni_index/spumoni_full_ref
        """

rule run_spumoni_to_generate_ms_exp7:
    input:
        "exp7_indexes/spumoni_index/spumoni_full_ref.fa",
        "exp7_ms_data/{read_type}/pivot_{pivot}/pivot_{pivot}.fna"
    output:
        "exp7_ms_data/{read_type}/pivot_{pivot}/pivot_{pivot}.fna.lengths"
    shell:
        """
        spumoni run -r exp7_indexes/spumoni_index/spumoni_full_ref -p {input[1]} -M -n 
        """

#   Section 3.5: Extract half-MEMs and MEMs using the pivot's MS lengths
#   and the pivot itself.

rule extract_mems_or_halfmems_based_on_ms_exp7:
    input:
        "exp7_ms_data/{read_type}/pivot_{pivot}/pivot_{pivot}.fna.lengths",
        "exp7_ms_data/{read_type}/pivot_{pivot}/pivot_{pivot}.fna"
    output:
        "exp7_{mem_type}_data/{read_type}/pivot_{pivot}.fastq"
    shell:
        """
        python3 {repo_dir}/src/extract_mems.py \
        -l {input[0]} \
        -p {input[1]} \
        --{wildcards.mem_type} \
        -t {thresh} \
        -o {output}
        """

#   Section 3.6: Build r-index for each dataset individually over
#   the reference containing both the forward and reverse complement.

rule build_rindex_over_individual_databases_exp7:
    input:
        "exp7_combined_refs/dataset_{num}/combined_ref_all.fna"
    output:
        "exp7_combined_refs/dataset_{num}/combined_ref_all.fna.ri"
    shell:
        """
        cd {r_dir} 
        ri-buildfasta {base_dir}/{input}
        cd {base_dir}
        """

#   Section 3.7: Create SAM file using r-index to locate all the 
#   half-MEMs or MEMs in each of the individual databases.

rule align_pivot_against_rindex_of_database_exp7:
    input:
        "exp7_combined_refs/dataset_{num}/combined_ref_all.fna",
        "exp7_combined_refs/dataset_{num}/combined_ref_all.fna.ri",
        "exp7_{mem_type}_data/{read_type}/pivot_{pivot}.fastq"
    output:
        "exp7_sam_files/{mem_type}/{read_type}/pivot_{pivot}_align_dataset_{num}.sam"
    shell:
        """
        cd {r_dir} 
        ri-align -m 1 locate {base_dir}/{input[0]} {base_dir}/{input[2]} \
        > {base_dir}/{output}
        """

#   Section 3.8: Analyze the SAM files with respect to each database
#   to determine how we should split each 

rule analyze_sam_for_each_type_exp7:
    input:
        get_python_input_exp7,
        "exp7_combined_ref_fai/combined_ref_forward_only.fna.fai"
    output:
        "exp7_output_data/{mem_type}/{read_type}/accuracy_values.csv",
        "exp7_output_data/{mem_type}/{read_type}/confusion_matrix.csv"
    shell:
        """
        python3 {repo_dir}/src/analyze_sam.py \
        -n {num_datasets} \
        -s {base_dir}/exp7_sam_files/{wildcards.mem_type}/{wildcards.read_type}/ \
        -o {base_dir}/exp7_output_data/{wildcards.mem_type}/{wildcards.read_type}/ \
        --{wildcards.mem_type} \
        -l exp7_combined_ref_fai/combined_ref_forward_only.fna.fai \
        -r exp7_{wildcards.mem_type}_data/{wildcards.read_type}/ {feat_level_opt}
        """

#   Section 3.9: Gather all the output files needed for full
#   analysis of experiment 7

rule gather_all_output_files_exp7:
    input:
        "exp7_output_data/half_mems/illumina/accuracy_values.csv",
        "exp7_output_data/half_mems/ont/accuracy_values.csv",
        "exp7_output_data/mems/illumina/accuracy_values.csv",
        "exp7_output_data/mems/ont/accuracy_values.csv"
    output:
        f"exp7_final_output/trial_{curr_trial}_half_mems_illumina.csv",
        f"exp7_final_output/trial_{curr_trial}_half_mems_ont.csv",
        f"exp7_final_output/trial_{curr_trial}_mems_illumina.csv",
        f"exp7_final_output/trial_{curr_trial}_mems_ont.csv"
    shell:
        """
        cp {input[0]} {output[0]}
        cp {input[1]} {output[1]}
        cp {input[2]} {output[2]}
        cp {input[3]} {output[3]}
        """

rule generate_exp7_output:
    input:
        f"exp7_final_output/trial_{curr_trial}_half_mems_illumina.csv",
        f"exp7_final_output/trial_{curr_trial}_half_mems_ont.csv",
        f"exp7_final_output/trial_{curr_trial}_mems_illumina.csv",
        f"exp7_final_output/trial_{curr_trial}_mems_ont.csv"

