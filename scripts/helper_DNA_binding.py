#!/usr/bin/env python3

# import packages
import numpy as np

import helper_data
helper_data = helper_data.helper_data()

class helper_DNA_binding(object):
    '''
    Helper class for p53 transcription factor DNA binding.
    '''

    def __init__(self):
        '''
        Constructor.
        '''
        return

    def target_genes(self):
        '''
        All of the target genes considered.
        '''
        data_file = "../data/target_genes.txt"
        with open(data_file, "r") as f:
            genes = [l.strip("\n") for l in f.readlines()]
        return genes

    def Kd_array(self, mut_indices):
        '''
        Returns an array of DNA Kds for target genes of interest.
        '''
        data_file = "../data/all_genes_DNA_Kd.npy"
        data = np.load(data_file)[:, mut_indices]
        return data

    def all_genes_prob_binding_array(self, mut_indices):
        '''
        Returns an array of probability of binding across mutations across genes of interest.
        '''

        # predicted concentrations (nM)
        concs, _ = helper_data.pred_concs(mut_indices)

        # stacked Kds (dimensions (num_genes, num_tetra, num_muts)
        Kd_array = self.Kd_array(mut_indices)

        # calculate probabilities
        ratio = (Kd_array/concs)
        prob_of_binding_array = 1/(1 + ratio**2)

        return prob_of_binding_array

    def median_prob_binding_array(self, mut_indices):
        '''
        Returns the median probability of binding across the target genes of interest.
        '''
        all_genes_prob_binding = self.all_genes_prob_binding_array(mut_indices)
        final_array = np.median(all_genes_prob_binding, axis=0)
        return final_array
