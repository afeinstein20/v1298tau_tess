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


# Custom rule to generate the simulation results and upload them to Zenodo
rule generate_simulation_results:
    input:
        "src/figures/run_simulation.py"
    output:
        report("src/figures/simulation_results.dat", category="Dataset")
    conda:
        "environment.yml"
    shell:
        "cd src/figures && python run_simulation.py && python zenodo.py --upload"


# Custom rule to download the simulation results from Zenodo
rule download_simulation_results:
    output:
        report("src/figures/simulation_results.dat", category="Dataset")
    conda:
        "environment.yml"
    shell:
        "cd src/figures && python zenodo.py --download"


# If we are on GitHub Actions CI, use the rule where we download the data
# If not, use the rule where we generate the data.
import os
ON_GITHUB_ACTIONS = os.getenv("CI", "false") == "true"
if ON_GITHUB_ACTIONS:
    ruleorder: download_simulation_results > generate_simulation_results
else:
    ruleorder: generate_simulation_results > download_simulation_results