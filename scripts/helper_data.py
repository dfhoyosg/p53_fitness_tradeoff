#!/usr/bin/env python3

# import packages
import numpy as np

class helper_data(object):
    '''
    Returns data for the fitness model.
    '''

    def __init__(self):
        '''
        Constructor.
        '''
        return

    def ordered_muts(self):
        '''
        Returns all p53 missense mutations in order.
        '''
        data_file = "../data/all_p53_ordered_muts.txt"
        with open(data_file, "r") as f:
            ordered_muts = np.array([l.strip("\n") for l in f.readlines()])
        return ordered_muts

    def background_mut_freqs(self, mut_indices):
        '''
        Background mutation frequencies.
        '''
        data_file = "../data/background_mutation_frequencies.npy"
        data = np.load(data_file)[mut_indices]
        data /= data.sum()
        return data

    def obs_muts_indices(self):
        '''
        Returns an array of indices for mutations observed.
        '''
        data_file = "../data/observed_mutation_indices.npy"
        data = np.load(data_file)
        return data

    def obs_muts(self):
        '''
        Returns an array of the observed mutations.
        '''
        ordered_muts = self.ordered_muts()
        obs_muts_indices = self.obs_muts_indices()
        obs_muts = ordered_muts[obs_muts_indices]
        return obs_muts

    def obs_freqs(self, mut_indices):
        '''
        Returns an array with mutation frequencies.
        '''
        data_file = "../data/observed_mutation_frequencies.npy"
        data = np.load(data_file)[mut_indices]
        return data

    def obs_freqs_95_CIs(self, mut_indices):
        '''
        Returns an array with the 95% CI interval values for the observed mutation frequencies.
        '''
        data_file = "../data/observed_mutation_95_CIs.npy"
        data = np.load(data_file)[mut_indices]
        return data

    def calculate_hotspot_mutations_and_indices(self, subset_muts):
        '''
        Returns the indices for the hotspot mutations. Note 
        these are the indices AFTER selecting for the observed frequencies.
        '''
        hotspots = np.array(['p.R175H', 'p.R248Q', 'p.R273H', 'p.R248W',
                             'p.R273C', 'p.R282W', 'p.G245S', 'p.Y220C'])
        hotspots = hotspots[np.in1d(hotspots, subset_muts)]
        indices = np.array([np.where(subset_muts == h)[0][0] for h in hotspots])
        return (hotspots, indices)

    def sorted_haplo_probs(self):
        '''
        Returns a vector with sorted haplotype probabilities.
        '''
        data_path = "../data/haplotype_sorted_probabilities.npy"
        data = np.load(data_path)
        return data

    def pred_concs(self, mut_indices):
        '''
        Returns predicted concentration.
        '''
        # read in data (concentration of p53 monomer subunits)
        data_file = "../data/concentration_array.npy"
        data = np.load(data_file)[mut_indices]

        # get the concentration of dimers
        dimer_conc = 0.5 * data

        # MT monomer (peptide) concentration
        mt_pep_conc = data

        return (dimer_conc, mt_pep_conc)
