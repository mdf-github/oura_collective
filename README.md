# Code for "Collective sleep and activity patterns of college students from wearable devices"

## Installing the conda environment

Install miniconda or Anaconda, and use the `environment.yml` file to install packages:

`conda env create -f environment.yml`

Then activate:

`conda activate oura_collective`.

## Downloading the data
The data is available at https://doi.org/10.6084/m9.figshare.29625014. Download the `data/` folder and place the contents in the directory containing `oura_collective.py`.

    oura_collective
    |--data
       |--activity_data.csv
       |--sleep_data.csv
    |--oura_collective.py
    |--utils.py

## Running the code
Run `python oura_collective.py` creates a folder `figures/oura_collective_behavior_public` that will contain all the figures. It will also output the regression tables.