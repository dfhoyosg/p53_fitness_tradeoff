#!/usr/bin/env python3

# import packages
import numpy as np
import os
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# class
class helper_funcs(object):
    '''
    This class contains certain functions for the fitness model.
    '''

    def __init__(self):
        '''
        Constructor.
        '''
        return

    def predicted_arrays_with_immune(self, matrix_weights, matrices, haplo_freqs, null_freqs):
        '''
        Computes the predicted mutation frequencies, the matrix used for the 
        prediction, and the normalization vector.
        '''
        # multiply weights and probabilities matrices
        fitness_matrix = (matrix_weights * matrices).sum(axis=0)
        # use trick to avoid overflow (prob. dist. is shift-invariant)
        fitness_matrix -= fitness_matrix.max()
        # multiply null frequencies across the columns
        pred_matrix = null_freqs * np.exp(fitness_matrix)
        # normalize across columns for probability dist. per row, then mean across rows
        norm_vec = pred_matrix.sum(axis=1)[:, np.newaxis]
        pred_matrix /= norm_vec
        # no need to normalize since already gets normalized
        pred_vec = np.average(pred_matrix, 
                              weights=haplo_freqs, axis=0)
        return (pred_vec, pred_matrix, norm_vec)

    def kl_div(self, x, y):
        '''
        Returns KL divergence.
        '''
        return (x * np.log(x/y)).sum()

    def cross_entropy(self, x, y):
        '''
        Returns the cross entropy.
        '''
        return -(x * np.log(y)).sum()

    def pearson_corr(self, x, y):
        '''
        Returns Pearson correlation.
        '''
        return pearsonr(x, y)[0]

    def spearman_corr(self, x, y):
        '''
        Returns Spearman correlation.
        '''
        return spearmanr(x, y)[0]

    def L1_dist(self, x, y):
        '''
        Returns the L1 distance.
        '''
        return (np.absolute(x-y)).sum()

    def L2_dist(self, x, y):
        '''
        Returns the L2 distance.
        '''
        return np.sqrt(((x - y)**2).sum())

    def hotspot_stats(self, x_hotspots, y_hotspots, hotspots):
        '''
        Returns statistics on the hotspots. "x" is the observed, "y" is the predicted.
        '''
        # return formatted values
        diffs = np.round(np.absolute(x_hotspots - y_hotspots), 4)
        label = ["L1 {}={}".format(h.split(".")[1], d) for h,d in zip(hotspots, diffs)]
        return label

    def hotspot_pred_freqs_histograms(self, pred_matrix, haplo_freqs, hotspots, 
                                      obs_hotspot_freqs, obs_hotspot_freqs_95_CIs, hotspot_indices, 
                                      output_dir):
        '''
        For each hotspot mutation, returns a histogram of the predicted frequencies 
        based on the optimum weights and haplotype frequencies.
        '''
        # make output directory
        os.makedirs(output_dir, exist_ok=True)

        # go across hotspots
        for h,f,c,i in zip(hotspots, obs_hotspot_freqs, obs_hotspot_freqs_95_CIs, hotspot_indices):
            freqs = pred_matrix[:,i]
            mean_freqs = round(np.average(freqs, weights=haplo_freqs), 4)
            plt.figure()
            hist_data, _, _ = plt.hist(freqs, bins=10, weights=haplo_freqs, alpha=0.5, ec='k')
            plt.axvline(mean_freqs, ls='--', c='r', 
                        label="Fitness Model Frequency : {}".format(mean_freqs))
            plt.axvline(f, ls='--', c='g', label="Observed Frequency : {}".format(round(f, 4)))
            plt.annotate(text='', xy=(f-c, hist_data.max()/2), xytext=(f+c,hist_data.max()/2), 
                         arrowprops=dict(arrowstyle='-', ls='dashed', color='g'))
            plt.xlim(0.8*min(freqs.min(), f-c), 1.2*max(freqs.max(), f+c))
            plt.xlabel(r"Predicted Frequencies Across Haplotypes, $\hat{x}(H)$")
            plt.ylabel("Fraction of Haplotypes")
            plt.title(h.replace("p.", ""))
            plt.legend()
            plt.savefig("{}/{}_opt_hist.png".format(output_dir, h.split(".")[1]), 
                        bbox_inches="tight")
        return

    def plot_compare_freqs(self, obs_freqs, pred_freqs, obs_freqs_95_CIs, label, 
                           savepath):
        '''
        Plots a scatter plot to compare the observed and predicted frequencies.
        '''
        # make figures
        min_val = min([obs_freqs.min(), pred_freqs.min()]) / 10
        max_val = max([obs_freqs.max(), pred_freqs.max()]) * 10

        # compare values
        plt.figure()
        
        plt.scatter(obs_freqs, pred_freqs, label=label, alpha=0.5, c='b')
        # color the hotspots
        plt.scatter(obs_freqs[obs_freqs.argsort()][-8:], pred_freqs[obs_freqs.argsort()][-8:], alpha=0.5, c='r')
        plt.xlabel("Observed Frequency, $x$")
        plt.ylabel("Predicted Frequency, $\hat{x}$")
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(savepath, bbox_inches="tight")
        # save the correlations to file
        with open(savepath.replace(".png", "") + "_stats.txt", "w") as f:
            f.write("\n".join(["KL_Div:{}".format(self.kl_div(obs_freqs, pred_freqs)), 
                               "Pearsonr:{}".format(pearsonr(obs_freqs, pred_freqs)), 
                               "Spearmanr:{}".format(spearmanr(obs_freqs, pred_freqs))]) + "\n")
        return

    def plot_compare_codon_freqs(self, obs_muts, obs_freqs, pred_freqs, savepath):
        '''
        Plots a scatter plot to compare the observed and predicted codon frequencies.
        '''
        # gather data
        obs_freqs_dict = {x:y for x,y in zip(obs_muts, obs_freqs)}
        pred_freqs_dict = {x:y for x,y in zip(obs_muts, pred_freqs)}
        x_data, y_data = [], []
        for pos in np.arange(1, 394):
            obs_allowed_freqs = [obs_freqs_dict[x] for x in obs_muts if int(x[3:-1]) == pos]
            pred_allowed_freqs = [pred_freqs_dict[x] for x in obs_muts if int(x[3:-1]) == pos]
            if obs_allowed_freqs != [] and pred_allowed_freqs != []:
                x_data.append(sum(obs_allowed_freqs))
                y_data.append(sum(pred_allowed_freqs))

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        # min and max values
        min_val = min([x_data.min(), y_data.min()]) / 10
        max_val = max([x_data.max(), y_data.max()]) * 10

        # summary stats
        kl_div = self.kl_div(x_data, y_data)
        pearson_corr = self.pearson_corr(x_data, y_data)
        spearman_corr = self.spearman_corr(x_data, y_data)
        L1_dist = self.L1_dist(x_data, y_data)
        L2_dist = self.L2_dist(x_data, y_data)

        # label (round to two digits for the figures)
        label_stats = np.array([kl_div, pearson_corr, spearman_corr,
                                L1_dist, L2_dist]).round(2)
        label = ("KL_Div={}\nPearson $r$={}\nSpearman $r$={}".format(round(kl_div, 2), 
                                                                     round(pearson_corr, 2), 
                                                                     round(spearman_corr, 2)))

        # compare values
        plt.figure()
        plt.scatter(x_data, y_data, label=label, alpha=0.5, c='b')
        # color the hotspots
        plt.scatter(x_data[x_data.argsort()][-6:], y_data[x_data.argsort()][-6:], alpha=0.5, c='r')
        plt.xlabel("Observed Codon Frequency")
        plt.ylabel("Predicted Codon Frequency")
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(savepath, bbox_inches="tight")
        # save the correlations to file
        with open(savepath.replace(".png", "") + "_stats.txt", "w") as f:
            f.write("\n".join(["KL_Div:{}".format(kl_div), 
                               "Pearsonr:{}".format(pearsonr(x_data, y_data)),
                               "Spearmanr:{}".format(spearmanr(x_data, y_data))]) + "\n")
        return

    def plot_compare_freq_change(self, null_freqs, obs_freqs, pred_freqs, savepath):
        '''
        Plots a scatter plot to compare the direction and magnitude of change of frequencies 
        from the background mutation frequencies.
        '''
        # compare divided values
        pred_increase = (pred_freqs/null_freqs > 1)
        pred_decrease = (pred_freqs/null_freqs < 1)
        obs_increase = (obs_freqs/null_freqs > 1)
        obs_decrease = (obs_freqs/null_freqs < 1)
        accurate_increase = (pred_increase * obs_increase)[obs_increase].sum() / obs_increase.sum()
        accurate_decrease = (pred_decrease * obs_decrease)[obs_decrease].sum() / obs_decrease.sum()
        accurate_increase = round(100*accurate_increase, 2)
        accurate_decrease = round(100*accurate_decrease, 2)

        #sort_order = pred_freqs.argsort()
        sort_order = obs_freqs.argsort()
        x_data = (obs_freqs/null_freqs)[sort_order]
        y_data = (pred_freqs/null_freqs)[sort_order]

        # correlations
        pearson_corr = self.pearson_corr(x_data, y_data)
        spearman_corr = self.spearman_corr(x_data, y_data)

        min_value = min(x_data.min(), y_data.min())
        max_value = max(x_data.max(), y_data.max())

        plt.figure()
        plt.scatter(x_data, y_data, c=obs_freqs[sort_order], cmap='viridis_r', 
                    vmin=obs_freqs.min(), vmax=obs_freqs.max(), 
                    label="{}\n{}".format(pearsonr(x_data, y_data), spearmanr(x_data, y_data)))
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Observed Frequency, $x$')
        plt.axhline(1, ls='--', c='k', zorder=2)
        plt.axvline(1, ls='--', c='k', zorder=3)
        plt.plot([min_value, max_value], [min_value, max_value], ls='--', c='k', zorder=4)
        plt.xlim(0.9*x_data.min(), 1.1*x_data.max())
        plt.ylim(0.9*y_data.min(), 1.1*y_data.max())
        plt.xlabel("Posterior Ratio, $W$")
        plt.ylabel("Predicted Ratio, $\hat{W}$")
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(savepath, bbox_inches="tight")

        # bar plot
        accurate_increase = (pred_increase * obs_increase)[obs_increase]
        accurate_decrease = (pred_decrease * obs_decrease)[obs_decrease]
        data = [obs_freqs[obs_increase][accurate_increase].sum(), obs_freqs[obs_increase][~accurate_increase].sum(), 
                obs_freqs[obs_decrease][accurate_decrease].sum(), obs_freqs[obs_decrease][~accurate_decrease].sum()]
        plt.figure()
        plt.bar(range(4), data)
        plt.xticks(range(4), ["TP", "FP", "TN", "FN"])
        plt.ylabel("Fraction of Observed Frequency")
        plt.savefig(savepath.replace(".png", "") + "_bargraph.png", bbox_inches="tight")

        return

    def plot_codon_dist(self, obs_muts, pred_freqs, savepath):
        '''
        Plots the mutation distribution across codons.
        '''
        pred_freqs_dict = {x:y for x,y in zip(obs_muts, pred_freqs)}
        x_data, y_data = np.arange(1, 394), np.zeros(393)
        for i,pos in enumerate(x_data):
            allowed_freqs = [pred_freqs_dict[x] for x in obs_muts if int(x[3:-1]) == pos]
            if allowed_freqs != []:
                y_data[i] = 100*sum(allowed_freqs)
        plt.figure()
        markerline, stemlines, baseline = plt.stem(x_data, y_data, use_line_collection=True)
        for pos in [175, 220, 245, 248, 273, 282]:
            plt.annotate(str(pos), xy=(x_data[pos-1], y_data[pos-1]), ha='center', color='red')
        plt.setp(baseline, visible=False)
        plt.setp(markerline, visible=False)
        plt.xlabel("Amino Acid Position")
        plt.ylabel("Frequency (%)")
        plt.savefig(savepath, bbox_inches="tight")
        return

    def opt_results(self, optimizing_on, time_taken, opt_results, 
                    null_freqs, obs_muts, obs_freqs, obs_freqs_95_CIs, pred_freqs, pred_matrix, norm_vec, 
                    hotspots, hotspot_indices, haplo_freqs, output_dir):
        '''
        Returns final optimum results.
        '''
        # figure output files
        output_file_1 = "{}/codon_dist.png".format(output_dir)
        output_file_2 = "{}/comp_freqs.png".format(output_dir)
        output_file_3 = "{}/comp_codon_freqs.png".format(output_dir)
        output_file_4 = "{}/comp_div_freqs.png".format(output_dir)
        # hist output dir
        hist_output_dir = "{}/hotspot_hists/".format(output_dir)
        # results file
        results_file = "{}/opt_results.txt".format(output_dir)

        # save the observed mutations to file
        np.save("{}/obs_muts.npy".format(output_dir), obs_muts)

        # save arrays to file
        np.save("{}/obs_freqs.npy".format(output_dir), obs_freqs)
        np.save("{}/pred_freqs.npy".format(output_dir), pred_freqs)
        np.save("{}/null_freqs.npy".format(output_dir), null_freqs)

        # hotspot information (using the subsetted mutations)
        obs_hotspot_freqs = obs_freqs[hotspot_indices]
        obs_hotspot_freqs_95_CIs = obs_freqs_95_CIs[hotspot_indices]
        pred_hotspot_freqs = pred_freqs[hotspot_indices]
        hotspot_stats = self.hotspot_stats(obs_hotspot_freqs, pred_hotspot_freqs, hotspots)

        # save hotspot arrays
        np.save("{}/hotspots.npy".format(output_dir), hotspots)
        np.save("{}/hotspot_indices.npy".format(output_dir), hotspot_indices)
        np.save("{}/hotspot_obs_freqs.npy".format(output_dir), obs_hotspot_freqs)
        np.save("{}/hotspot_pred_freqs.npy".format(output_dir), pred_hotspot_freqs)

        # save haplo freqs
        np.save("{}/haplo_freqs.npy".format(output_dir), haplo_freqs)

        # summary stats
        kl_div = self.kl_div(obs_freqs, pred_freqs)
        cross_entropy = self.cross_entropy(obs_freqs, pred_freqs)
        pearson_corr = self.pearson_corr(obs_freqs, pred_freqs)
        spearman_corr = self.spearman_corr(obs_freqs, pred_freqs)
        L1_dist = self.L1_dist(obs_freqs, pred_freqs)
        L2_dist = self.L2_dist(obs_freqs, pred_freqs)

        # label (round to two digits for the figures)
        label_stats = np.array([kl_div, pearson_corr, spearman_corr, 
                                L1_dist, L2_dist]).round(2)
        label = ("KL_Div={}\nPearson $r$={}\nSpearman $r$={}".format(round(kl_div, 2), 
                                                                     round(pearson_corr, 2), 
                                                                     round(spearman_corr, 2)))

        # make figures
        self.plot_codon_dist(obs_muts, pred_freqs, output_file_1)
        self.plot_compare_freqs(obs_freqs, pred_freqs, obs_freqs_95_CIs, label,
                                output_file_2)
        self.plot_compare_codon_freqs(obs_muts, obs_freqs, pred_freqs,
                                      output_file_3)
        self.hotspot_pred_freqs_histograms(pred_matrix, haplo_freqs, hotspots,
                                           obs_hotspot_freqs, obs_hotspot_freqs_95_CIs,
                                           hotspot_indices, hist_output_dir)
        self.plot_compare_freq_change(null_freqs, obs_freqs, pred_freqs,
                                      output_file_4)

        # interpret optimum results
        opt_weights = opt_results.x
        success = opt_results.success
        message = opt_results.message
        num_iter = opt_results.nit

        # save optimal weights
        np.save("{}/opt_weights.npy".format(output_dir), opt_weights)

        # output results
        to_write = "\n".join(["Optimizing_on={}", "time_taken_(sec)={}", "success={}",
                              "message={}", "num_iter={}", 
                              "opt_sigmas={}", "kl_div={}", "cross_entropy={}", "pearson_r={}",
                              "spearman_r={}", "L1_dist={}", "L2_dist={}", "{}"])
        to_write = to_write.format(optimizing_on, time_taken, success, message, num_iter,
                                   opt_weights, kl_div, cross_entropy, pearson_corr, 
                                   spearman_corr, L1_dist, L2_dist, ",".join(hotspot_stats))

        with open(results_file, "w") as f:
            f.write(to_write + "\n")

        plt.close('all')

        return
