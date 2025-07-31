# Code for "Collective sleep and activity patterns of college students from wearable devices"

## Installing the conda environment

Install miniconda or Anaconda, and use the `environment.yml` file to install packages:

`conda env create -f environment.yml`

Then activate:

`conda activate oura_collective`.

## Downloading the data
The data is available by reasonable request to the corresponding authors.
However, we are including a sample dataset in the data repository so that users can run the script and understand how the results were generated.

Below is the file tree structure assumed by the code:

    oura_collective
    |--data
       |--activity_data.csv
       |--sleep_data.csv
    |--oura_collective.py
    |--utils.py

## Running the code
Run `python oura_collective.py --activity activity_data.csv --sleep sleep_data.csv` creates a folder `figures/oura_collective_behavior_public` that will contain all the figures. It will also output the regression tables.