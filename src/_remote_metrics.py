import glob
import matplotlib
import matplotlib.pyplot as plt
# import seaborn as sns
import utils
import metrics_utils
import metrics_utils_mod
#import sys
#sys.path.insert(0, '../src/')

import sparsity_mnist
is_save = True
figsize = metrics_utils.get_figsize(is_save)


def standard_figure():
    # axis
    plt.gca().set_ylim(bottom=0)
    #plt.gca().set_ylim([0,0.005])
    #plt.gca().set_ylim(bottom=0,top=0.02)

    plt.gca().set_xscale("log", nonposx='clip')
    plt.gca().set_xlim([9, 800])

    # labels, ticks, titles
    ticks = [10, 25, 50, 100, 200, 300, 400, 500, 750]
    labels = [10, 25, 50, 100, 200, 300, 400, 500, 750]
    plt.xticks(ticks, labels, rotation=90)
    plt.ylabel('Reconstruction error (per pixel)')
    plt.xlabel('Number of measurements')

criterions = [['measurement', 'mean'],
             ['l2', 'mean']]
retrieve_lists = [[['measurement', 'mean'], ['measurement', 'std']],
                 [['l2', 'mean'], ['l2', 'std']]]
legend_base_regexs = [
    ('Lasso'  , '../estimated/mnist/full-input/gaussian/0.0/', '/lasso20/*'),

    ('PCA uaffine'  , '../estimated/mnist/full-input/gaussian-reproducible/0.0/', '/lassopca/0.0_cvxpy-constrEq-reweight1-synthesis-unnormalized-own-pcafraction1-affine'),
    ('PCA uaffine 0.1'  , '../estimated/mnist/full-input/gaussian-reproducible/0.0/', '/lassopca/0.0_cvxpy-constrEq-reweight1-synthesis-unnormalized-own-pcafraction0.1-affine'),
    ('PCA rawDict'  , '../estimated/mnist/full-input/gaussian-reproducible/0.0/', '/lassopca/0.0_cvxpy-constrEq-reweight1-synthesis-unnormalized-rawDict-pcafraction1-affine'),

    ('PCA sparseDict 100'  , '../estimated/mnist/full-input/gaussian-reproducible/0.0/', '/lassopca/0.0_cvxpy-constrEq-reweight1-synthesis-normalized-sklearnDictionaryLearning100-samples1000-pcafraction1-affine'),


    ('From VAE flex ' , '../estimated/mnist/full-input/gaussian/0.0/' ,    "/vaeflex5-10-15-20-40-60-80-100/0.0_1.0_0.0_adam_0.1_0.9_False_1000_10_0.01_1_previous-and-random_flexible"),
]

save_paths = ['../results/comparison_between_VAEs_meas.pdf',
             '../results/comparison_between_VAEs.pdf']


for i in range(len(criterions)):
    plt.figure(figsize=figsize)
    legends = []
    criterion = criterions[i]
    retrieve_list = retrieve_lists[i]
    for legend, base, regex in legend_base_regexs:
        metrics_utils_mod.plot(base, regex, criterion, retrieve_list,name='vae')
        legends.append(legend)
    standard_figure()
#    plt.legend(legends, fontsize=12.5)
#    utils.save_plot(is_save, save_path)
plt.show()
