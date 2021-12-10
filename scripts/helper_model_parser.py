#!/usr/bin/env python3

# import packages
import numpy as np

import helper_data
helper_data = helper_data.helper_data()
import helper_DNA_binding
helper_DNA_binding = helper_DNA_binding.helper_DNA_binding()

class helper_model_parser(object):
    '''
    Helper class to start the run the model.
    '''

    def __init__(self):
        '''
        Constructor.
        '''
        return

    def get_functional_component(self, mut_indices):
        '''
        Returns the array for the functional component, the median probability of lack of DNA binding.
        '''

        functional_array = 1 - helper_DNA_binding.median_prob_binding_array(mut_indices)
        return functional_array

    def get_immune_component(self, mut_indices):
        '''
        Returns the array for the immune component, the effective probability of peptides presenting on MHC-I.
        '''
        data_file = "../data/immune_prob_array.npy"
        immune_array = np.load(data_file)
        return immune_array

    def get_model_data(self):
        '''
        Returns the model data.
        '''

        # all mutations
        ordered_muts = np.array(helper_data.ordered_muts())

        # get the observed mutation frequency information
        # observed mutation indices
        obs_muts_indices = helper_data.obs_muts_indices()
        # other mutation information
        obs_muts = ordered_muts[obs_muts_indices]
        obs_freqs = helper_data.obs_freqs(obs_muts_indices)
        obs_95_CIs = helper_data.obs_freqs_95_CIs(obs_muts_indices)

        # hotspots and their indices, for AFTER subsetting the arrays for the selected mutation indices
        hotspots, hotspot_indices = helper_data.calculate_hotspot_mutations_and_indices(obs_muts)

        # get the background mutation frequencies
        null_freqs = helper_data.background_mut_freqs(obs_muts_indices)

        # haplo frequencies
        haplo_freqs = helper_data.sorted_haplo_probs()

        # get the fitness model components
        functional_component = self.get_functional_component(obs_muts_indices)

        # use specific immune matrix if requested
        immune_component = self.get_immune_component(obs_muts_indices)

        # gather all fitness matrices
        fitness_types = ["Function", "Immune"]
        all_fitness_components = np.array([functional_component, immune_component], dtype='object')
        
        # output
        output = [null_freqs, obs_muts, obs_freqs, obs_95_CIs, hotspots, hotspot_indices, haplo_freqs, 
                  fitness_types, all_fitness_components]
        return output
