####################################################
# Name: prepare_data.smk
# Description: Takes in a set of genomes of different
#              classes, generates the out-pivot 
#              dataset, and simulates both short
#              and long reads.
#
# Date: 10/25/22
####################################################

####################################################
# Section 1: Functions used by rules below
####################################################

def get_all_genomes_in_dataset_exp0(wildcards):
    """ Returns a list of database in a certain dataset """
    input_files = []
    for data_file in os.listdir(f"{database_root}/dataset_{wildcards.num}/"):
        if data_file.endswith(".fna.gz"):
            input_files.append(f"{database_root}/dataset_{wildcards.num}/{data_file}")
    return input_files

def get_complete_list_of_input_files_exp0(wildcards):
    """ Returns a list of input files for the final rule """
    input_files = []
    for num in range(1, num_datasets+1):
        input_files.append(f"trial_{wildcards.trial}/exp0_pivot_genomes/dataset_{num}/pivot_name.txt")
    for num in range(1, num_datasets+1):
        input_files.append(f"trial_{wildcards.trial}/exp0_nonpivot_genomes/dataset_{num}/nonpivot_names.txt")
    for num in range(1, num_datasets+1):
        for read_type in ["illumina", "ont"]:
            input_files.append(f"trial_{wildcards.trial}/exp0_pivot_reads/dataset_{num}/{read_type}/pivot_{num}_subset.fa")
    return input_files

####################################################
# Section 2: Rules needed for this workflow
####################################################

#   Section 2.1: Take pivot genomes from dataset, and move
#   them to a separate folder

rule data_prep_choose_pivot_from_each_dataset:
    input:
        get_all_genomes_in_dataset_exp0
    output:
        "trial_{trial}/exp0_pivot_genomes/dataset_{num}/pivot_{num}.fna.gz",
        "trial_{trial}/exp0_pivot_genomes/dataset_{num}/pivot_name.txt",
        "trial_{trial}/exp0_nonpivot_genomes/dataset_{num}/nonpivot_names.txt"
    shell:
        """
        # Copy all genomes to non-pivot folder
        cp {input} trial_{wildcards.trial}/exp0_nonpivot_genomes/dataset_{wildcards.num}/

        # Choose a pivot genome from the non-pivot folder
        pivot_genome_path=$(ls trial_{wildcards.trial}/exp0_nonpivot_genomes/dataset_{wildcards.num}/*.fna.gz | shuf | head -n1)
        pivot_genome_name=$(echo $pivot_genome_path  | awk -F/ '{{print $NF}}')

        # Copy pivot to pivot folder, save the name, and remove it
        cp $pivot_genome_path {output[0]}
        echo $pivot_genome_name > {output[1]}
        rm $pivot_genome_path

        # Save all the names of the non-pivot genomes to file
        for non_pivot in $(ls trial_{wildcards.trial}/exp0_nonpivot_genomes/dataset_{wildcards.num}/*.fna.gz); do
            echo $non_pivot | awk -F/ '{{print $NF}}' >> {output[2]}
        done
        """

#   Section 2.2: Simulate short and long reads from a pivot genome
#   from each dataset. And subset reads to reach a certain threshold
#   based on config parameters

rule generate_raw_positive_short_reads_exp0:
    input:
        "trial_{trial}/exp0_pivot_genomes/dataset_{num}/pivot_{num}.fna.gz"
    output:
        "trial_{trial}/exp0_pivot_reads/dataset_{num}/illumina/pivot_{num}.fa"
    shell:
        """
        gzip -d {input} -k -f
        art_illumina -ss HS25 -i trial_{wildcards.trial}/exp0_pivot_genomes/dataset_{wildcards.num}/pivot_{wildcards.num}.fna -na -l 150 -f 40.0 \
        -o trial_{wildcards.trial}/exp0_pivot_reads/dataset_{wildcards.num}/illumina/pivot_{wildcards.num}
        rm trial_{wildcards.trial}/exp0_pivot_genomes/dataset_{wildcards.num}/pivot_{wildcards.num}.fna

        seqtk seq -a trial_{wildcards.trial}/exp0_pivot_reads/dataset_{wildcards.num}/illumina/pivot_{wildcards.num}.fq > {output}
        rm trial_{wildcards.trial}/exp0_pivot_reads/dataset_{wildcards.num}/illumina/pivot_{wildcards.num}.fq 
        """

rule generate_raw_positive_long_reads_exp0:
    input:
        "trial_{trial}/exp0_pivot_genomes/dataset_{num}/pivot_{num}.fna.gz"
    output:
        "trial_{trial}/exp0_pivot_reads/dataset_{num}/ont/pivot_{num}.fa"
    shell:
        """
        gzip -d {input} -k -f
        pbsim --depth 40.0 --prefix trial_{wildcards.trial}/exp0_pivot_reads/dataset_{wildcards.num}/ont/pivot_{wildcards.num} \
        --hmm_model {pbsim_model} --accuracy-mean 0.95 --length-min 900 --length-max 1100 trial_{wildcards.trial}/exp0_pivot_genomes/dataset_{wildcards.num}/pivot_{wildcards.num}.fna
        
        cat 'trial_{wildcards.trial}/exp0_pivot_reads/dataset_{wildcards.num}/ont/pivot_{wildcards.num}'*.fastq > trial_{wildcards.trial}/exp0_pivot_reads/dataset_{wildcards.num}/ont/pivot_{wildcards.num}.fastq
        seqtk seq -a trial_{wildcards.trial}/exp0_pivot_reads/dataset_{wildcards.num}/ont/pivot_{wildcards.num}.fastq > trial_{wildcards.trial}/exp0_pivot_reads/dataset_{wildcards.num}/ont/pivot_{wildcards.num}.fa
        rm 'trial_{wildcards.trial}/exp0_pivot_reads/dataset_{wildcards.num}/ont/pivot_{wildcards.num}_'*.fastq
        ls  'trial_{wildcards.trial}/exp0_pivot_reads/dataset_{wildcards.num}/ont/pivot_{wildcards.num}'*.ref | xargs rm
        ls  'trial_{wildcards.trial}/exp0_pivot_reads/dataset_{wildcards.num}/ont/pivot_{wildcards.num}'*.maf | xargs rm
        rm trial_{wildcards.trial}/exp0_pivot_reads/dataset_{wildcards.num}/ont/pivot_{wildcards.num}.fastq
        """

rule subset_raw_reads_exp0:
    input:
        "trial_{trial}/exp0_pivot_reads/dataset_{num}/{read_type}/pivot_{num}.fa"
    output:
        "trial_{trial}/exp0_pivot_reads/dataset_{num}/{read_type}/pivot_{num}_subset.fa"
    shell:
        """
        # Just use k=31 since it a middle-of-the-range value ...
        python3 {repo_dir}/src/subset_reads.py -i {input} -o {output} -n {num_kmers} -k 31 --kmers
        """

#   Section 2.2: Generate a summary file for the dataset that includes all the relevant
#   values of interest like pivot, non-pivot files and the number of reads of each.

rule generate_database_summary_file_exp0:
    input:
        get_complete_list_of_input_files_exp0
    output:
        "trial_summaries/trial_{trial}_summary.txt"
    shell:
        """
        # Create the header line
        printf "Dataset #: | " > {output[0]}.raw
        for i in $(seq 1 {num_datasets}); do
            printf "$i | " >> {output[0]}.raw
        done
        printf "\n" >> {output[0]}.raw

        # Grab the pivot genome names ..
        printf "Pivot Genome: | " >> {output[0]}.raw
        for i in $(seq 1 {num_datasets}); do
            pivot_name=$(head -n1 "trial_{wildcards.trial}/exp0_pivot_genomes/dataset_${{i}}/pivot_name.txt")
            printf "$pivot_name | " >> {output[0]}.raw
        done
        printf "\n" >> {output[0]}.raw

        # Print number of Illumina reads ...
        printf "# of Illumina Reads: | " >> {output[0]}.raw
        for i in $(seq 1 {num_datasets}); do
            num_reads=$(grep -c "^>" "trial_{wildcards.trial}/exp0_pivot_reads/dataset_${{i}}/illumina/pivot_${{i}}_subset.fa")
            printf "$num_reads | " >> {output[0]}.raw
        done
        printf "\n" >> {output[0]}.raw

        # Print number of ONT reads ...
        printf "# of ONT Reads: | " >> {output[0]}.raw
        for i in $(seq 1 {num_datasets}); do
            num_reads=$(grep -c "^>" "trial_{wildcards.trial}/exp0_pivot_reads/dataset_${{i}}/ont/pivot_${{i}}_subset.fa")
            printf "$num_reads | " >> {output[0]}.raw
        done
        printf "\n" >> {output[0]}.raw

        # Get the non-pivot file with the largest number of lines ...
        max_lines=0
        for i in $(seq 1 {num_datasets}); do
            num_lines=$(wc -l "trial_{wildcards.trial}/exp0_nonpivot_genomes/dataset_${{i}}/nonpivot_names.txt" | awk '{{print $1}}')
            if (( $num_lines > $max_lines )); then
                max_lines=$((num_lines))
            fi 
        done

        # Print the non-pivot genomes to file ...
        printf "Non-Pivot genomes: \n" >> {output[0]}.raw
        for line in $(seq 1 $max_lines); do
            printf " | " >> {output[0]}.raw
            for i in $(seq 1 {num_datasets}); do
                genome_name=$(cat "trial_{wildcards.trial}/exp0_nonpivot_genomes/dataset_${{i}}/nonpivot_names.txt" | awk -v var=$line 'NR==var')
                printf "$genome_name | " >> {output[0]}.raw
            done
            printf "\n" >> {output[0]}.raw
        done

        column {output[0]}.raw -t -s "|" > {output[0]}
        rm {output[0]}.raw
        """

rule make_all_data_summaries_exp0:
    input:
        expand("trial_summaries/trial_{trial}_summary.txt", trial=range(1, num_trials+1)) 
