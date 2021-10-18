# User config
configfile: "showyourwork.yml"
    

# Import the showyourwork module
module showyourwork:
    snakefile:
        "showyourwork/workflow/Snakefile"
    config:
        config


# Use all default rules
use rule * from showyourwork


# Custom rule to download a dataset
rule fibonacci_data:
    output:
        report("src/figures/fibonacci.dat", category="Dataset")
    shell:
        "curl https://zenodo.org/record/5187276/files/fibonacci.dat --output {output[0]}"
