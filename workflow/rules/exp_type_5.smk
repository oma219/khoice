####################################################
# Name: exp_type_5.smk
# Description: Contains functions and rules for
#              the type of experiment type 5: 
# Date: 6/22/22
####################################################

####################################################
# Section 1: Code that is always run, when this 
#            experiment is called. It genereates the
#            needed files.
####################################################

if exp_type == 5:
    if not os.path.isdir("non_pivot_type_5/"):
        for i in range(1, num_datasets+1):
            dir_prefix = f"data/dataset_{i}/"
            list_of_files = [dir_prefix + x for x in os.listdir(dir_prefix)]

            if not os.path.isdir(f"non_pivot_type_5/dataset_{i}"):
                os.makedirs(f"non_pivot_type_5/dataset_{i}")

            # Copy over all the files at first
            for data_file in list_of_files:
                shutil.copy(data_file, f"non_pivot_type_5/dataset_{i}")
            
            # Extract the pivot and move to new directory
            dir_prefix = f"non_pivot_type_5/dataset_{i}/"
            list_of_files = [dir_prefix + x for x in os.listdir(dir_prefix)]

            print(list_of_files)

            pivot = random.choice(list_of_files)
            if not os.path.isdir(f"pivot_ms_type_5/pivot_{i}"):
                os.makedirs(f"pivot_ms_type_5/pivot_{i}")
            
            # Remove pivot from other set of genomes
            shutil.copy(pivot, f"pivot_ms_type_5/pivot_{i}/pivot_{i}.fna")
            os.remove(pivot)
            
####################################################
# Section 2: Helper functions needed for these
#            experiment rules.
####################################################


def get_all_non_pivot_genomes():
    """ Returns list of all non-pivot genomes """
    input_files = []
    for i in range(1, num_datasets + 1):
        for data_file in os.listdir(f"non_pivot_type_5/dataset_{i}/"):
            if data_file.endswith(".fna"):
                file_name = data_file.split(".fna")[0]
                input_files.append(f"non_pivot_type_5/dataset_{i}/{file_name}.fna")
    return input_files

def get_dataset_non_pivot_genomes(wildcards):
    """ Returns list of non-pivot genomes for a given dataset """
    input_files = []
    for data_file in os.listdir(f"non_pivot_type_5/dataset_{wildcards.num}/"):
        if data_file.endswith(".fna"):
            file_name = data_file.split(".fna")[0]
            input_files.append(f"non_pivot_type_5/dataset_{wildcards}/{file_name}.fna")
    return input_files

####################################################
# Section 3: Rules needed for this experiment type
####################################################

# Section 3.1: 

rule concat_ref_exp_5:
    input:
        get_all_non_pivot_genomes()
    output:
        "combined_type_5/combined_ref.fna"
    shell:
        "cat non_pivot_type_5/dataset_*/*_genomic.fna > combined_type_5/combined_ref.fna"

# Section 3.2: Generate matching statistics for pivot

rule spumoni_build_exp_5:
    input:
        "combined_type_5/combined_ref.fna"
    output:
        "combined_type_5/spumoni_full_ref.fa"
    shell:
        "spumoni build -r combined_type_5/combined_ref.fna -M -n"

rule spumoni_run_exp_5:
    input:
        "combined_type_5/spumoni_full_ref.fa",
        "pivot_ms_type_5/pivot_{num}/pivot_{num}.fna"
    output:
        "pivot_ms_type_5/pivot_{num}/pivot_{num}.fna.lengths"
    shell:
        """
        spumoni run -r combined_type_5/spumoni_full_ref.fa \
        -p pivot_ms_type_5/pivot_{wildcards.num}/pivot_{wildcards.num}.fna -M -n
        """

# Section 3.3: Extract Half-Mems from pivot MS

rule run_extract_mems_exp_5:
    input:
        "pivot_ms_type_5/pivot_{num}/pivot_{num}.fna.lengths",
        "pivot_ms_type_5/pivot_{num}/pivot_{num}.fna"
    output:
        "half_mems/pivot_{num}.fastq"
    shell:
        """
        python3 {repo_dir}/src/extract_mems.py \
        -l {base_dir}/pivot_ms_type_5/pivot_{wildcards.num}/pivot_{wildcards.num}.fna.lengths \
        -p {base_dir}/pivot_ms_type_5/pivot_{wildcards.num}/pivot_{wildcards.num}.fna \
        --half-mems \
        -t {thresh} \
        -o {base_dir}/half_mems/pivot_{wildcards.num}.fastq
        """

# Section 3.4: Create SAM file with located Half-Mems

rule ref_ri_build_fasta_exp_5:
    input:
        "combined_type_5/combined_ref.fna"
    output:
        "combined_type_5/combined_ref.fna.ri"
    shell:
        """
        cd {r_dir} 
        ri-buildfasta {base_dir}/combined_type_5/combined_ref.fna 
        cd {base_dir}
        """

rule ref_ri_align_exp_5:
    input:
        "combined_type_5/combined_ref.fna",
        "half_mems/pivot_{num}.fastq"
    output:
        "sam/pivot_{num}.sam"
    shell:
        """
        cd {r_dir} 
        ri-align locate {base_dir}/combined_type_5/combined_ref.fna {base_dir}/half_mems/pivot_{wildcards.num}.fastq \
        > {base_dir}/sam/pivot_{wildcards.num}.sam
        """

# Section 3.5: Generate List of Reference Genomes

rule gen_ref_list_exp_5:
    input:
        get_dataset_non_pivot_genomes
    output:
        "ref_lists_type_5/dataset_{num}_references.txt"
    shell:
        """
        for file in {input}; do
            grep '^>' $file | awk '{{print substr($1,2)}}' >> "ref_lists_type_5/dataset_{wildcards.num}_references.txt"
        done
        """