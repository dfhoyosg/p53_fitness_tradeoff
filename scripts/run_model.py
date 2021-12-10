#!/usr/bin/env python3

# This script runs the fitness model.

# import packages
import argparse

import helper_model_parser
helper_model_parser = helper_model_parser.helper_model_parser()
import helper_model_optimizer
helper_model_optimizer = helper_model_optimizer.helper_model_optimizer()

# input arguments
desc = "This script runs the fitness model for p53."
parser = argparse.ArgumentParser(desc)
parser.add_argument("output_dir", help="output dir")
args = parser.parse_args()
output_dir = args.output_dir

# get model run data

null_freqs, obs_muts, obs_freqs, obs_freqs_95_CIs, hotspots, hotspot_indices, haplo_freqs, \
        fitness_types, fitness_matrices = helper_model_parser.get_model_data()

helper_model_optimizer.get_opt_results(fitness_types, fitness_matrices,
                                       null_freqs, obs_muts, obs_freqs, obs_freqs_95_CIs, hotspots,
                                       hotspot_indices, haplo_freqs, output_dir)
