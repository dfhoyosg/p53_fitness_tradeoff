#!/usr/bin/env python3

# import packages
from scipy.optimize import minimize
import time
import numpy as np
import os

import helper_funcs
helper_funcs = helper_funcs.helper_funcs()

class helper_model_optimizer(object):
    '''
    Helper class for optimization and outputting the results.
    '''

    def __init__(self):
        '''
        Constructor.
        '''
        return

    def func_to_min_with_immune(self, weights, matrices, null_freqs, obs_freqs, haplo_freqs):
        '''
        Function to minimize.
        '''
        pred_freqs, _, _ = helper_funcs.predicted_arrays_with_immune(weights, matrices, 
                                                                     haplo_freqs, null_freqs)
        value = helper_funcs.cross_entropy(obs_freqs, pred_freqs)
        # avoid nan
        if np.isnan(value):
            value = np.inf
        print("Weights and cross entropy for present iteration:", weights, value)
        return value

    def cross_entropy_gradient(self, matrix_weights, matrices, null_freqs, obs_freqs, haplo_freqs):
        '''
        Returns the gradient for the cross entropy for each weight.
        '''
        fitness_matrices = matrix_weights * matrices
        fitness_matrix = fitness_matrices.sum(axis=0)
        fitness_matrix -= fitness_matrix.max()
        exp_fitness_matrix = np.exp(fitness_matrix)
        
        Z = (null_freqs * exp_fitness_matrix).sum(axis=1)
        par_Z = np.array([null_freqs * m * exp_fitness_matrix for m in matrices]).sum(axis=2)
        par_invZ = - par_Z / ((Z[np.newaxis,:]**2))
        
        immune_mat = matrices[-1]
        exp_immune_mat = np.exp(fitness_matrices[-1])

        a = 1/((haplo_freqs[:,np.newaxis] * exp_immune_mat/Z[:,np.newaxis]).sum(axis=0))
        c = (haplo_freqs[:,np.newaxis] * (exp_immune_mat/Z[:,np.newaxis]) * \
                (immune_mat - (par_Z[-1,:]/Z)[:,np.newaxis])).sum(axis=0)

        grad = np.zeros(shape=(matrices.shape[0]))

        b = (haplo_freqs[np.newaxis,:,np.newaxis] * \
                exp_immune_mat[np.newaxis,:,:] * \
                par_invZ[:-1,:,np.newaxis]).sum(axis=1)

        tmp_matrices = np.array([m for m in matrices[:-1]])
        grad_1 = - (obs_freqs[np.newaxis,:] * (tmp_matrices + a[np.newaxis,:] * b)).sum(axis=1)
        grad_2 = - (obs_freqs * a * c).sum()
        grad[:-1], grad[-1] = grad_1, grad_2
        return grad

    def conduct_optimization(self, x0, fitness_matrices, null_freqs, obs_freqs, haplo_freqs):
        '''
        Function to do the minimization.
        '''
        # setting bounds
        func_bnd = [(0, None)]
        immune_bnd = [(None, 0)]
        bnds = func_bnd + immune_bnd

        # function to minimize and arguments
        func_to_min = self.func_to_min_with_immune
        args = (fitness_matrices, null_freqs, obs_freqs, haplo_freqs)
        grad = self.cross_entropy_gradient

        # do minimization
        t0 = time.time()
        res = minimize(func_to_min, x0, args=args, 
                       method='L-BFGS-B', jac=grad, 
                       bounds = bnds, 
                       options={'maxiter' : np.inf, 'maxfun' : np.inf})
        time_taken = time.time() - t0
        return (res, time_taken)

    def get_opt_results(self, fitness_types, fitness_matrices, null_freqs, obs_muts, 
                        obs_freqs, obs_freqs_95_CIs, hotspots, hotspot_indices, haplo_freqs, output_dir):
        '''
        Gets the optimal results and writes to file.
        '''
        # parse the fitness types
        functional_type, immune_type = fitness_types

        # initial weights
        x0 = np.array([0, 0])

        # make output directory if needed
        os.makedirs(output_dir, exist_ok=True)

        # result of optimization
        res, time_taken = self.conduct_optimization(x0, fitness_matrices, null_freqs, obs_freqs, haplo_freqs)
        opt_weights = res.x
        pred_freqs, pred_matrix, norm_vec = helper_funcs.predicted_arrays_with_immune(opt_weights, fitness_matrices,
                                                                                      haplo_freqs, null_freqs)

        # save the fitness matrices to file
        np.save("{}/fitness_matrices.npy".format(output_dir), fitness_matrices)

        # save total fitness to file
        total_fit_array = np.average((opt_weights[0] * fitness_matrices[0]) + (opt_weights[1] * fitness_matrices[1]), 
                                     weights=haplo_freqs, axis=0)
        np.save("{}/total_fitness.npy".format(output_dir), total_fit_array)

        # save the optimal results
        helper_funcs.opt_results("cross_entropy", time_taken, res,
                                 null_freqs, obs_muts, obs_freqs, obs_freqs_95_CIs, pred_freqs, pred_matrix, norm_vec,
                                 hotspots, hotspot_indices, haplo_freqs, output_dir)

        return
