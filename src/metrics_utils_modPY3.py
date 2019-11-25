"""Some utils for plotting metrics"""
# pylint: disable = C0111


import glob
import numpy as np
from utils import loss_names,load_if_pickled
import matplotlib.pyplot as plt
import re
# import seaborn as sns


def int_or_float(val):
    try:
        return int(val)
    except ValueError:        
        return float(val)


def get_figsize(is_save):
    if is_save:
        figsize = [6, 4]
    else:
        figsize = None
    return figsize

def get_data(expt_dir):
    data = {}
    fieldnames = loss_names()
    for field in fieldnames:#
        data.update( {field:load_if_pickled(expt_dir + '/' +field+'.pkl').values()})
    return data


def get_metrics(expt_dir,distr='squared',bar_mode='confidence'):
    data = get_data(expt_dir)
    metrics = {}      
    scal=1#0.5
    fieldnames = loss_names()
    for field in fieldnames:
        #print field
        if distr == 'squared':
            d = np.asarray(list(data[field]))
        else:
            d = np.sqrt(np.asarray(list(data[field]))*784)/784 #HARCODING! ONLY FOR 28x28!!!!!!!
        mean = np.mean(d)
        #std = np.std(d) / np.sqrt(len(d)) #? later scaled by 1.96, i.e. two standard deviation. But why normalized?
        if bar_mode == 'confidence':
            std = np.std(d)
            std_mod,v = points_contained(d,thresh=0.5)
            std=std*std_mod
            #print('For {}, the interval contains {} percent of the values'.format(field,v))
        else:
            std = np.std(d)#scaled later
        metrics[field] = {'mean': mean, 'std': std}
        if len(d)>0:  
            if field=='l2_losses':
                pass
                #print('For l2_losses, the errorbars contain {} percent of the datapoints'.format(len(d[np.abs(d-mean)<=std])/float(len(d))))
    return metrics


def get_expt_metrics(expt_dirs,distr='squared'):
    expt_metrics = {}
    for expt_dir in expt_dirs:
        metrics = get_metrics(expt_dir,distr=distr)
        expt_metrics[expt_dir] = metrics
    return expt_metrics


def get_nested_value(dic, field):
    answer = dic
    for key in field:
        answer = answer[key]
    return answer


def find_best(pattern, criterion, retrieve_list,distr='squared'):
    dirs = glob.glob(pattern)
    metrics = get_expt_metrics(dirs,distr=distr)
    best_merit = 1e10
    answer = [None]*len(retrieve_list)
    for _, val in metrics.items():
        merit = get_nested_value(val, criterion)
        if merit < best_merit:
            best_merit = merit
            for i, field in enumerate(retrieve_list):
                answer[i] = get_nested_value(val, field)
    return answer

#def plot_phase_transition_diagram(base, regex, criterion, retrieve_list,indices='', axis_tmp='',name=''):
    

def plot(base, regex, criterion, retrieve_list,indices='', axis_tmp='',name='',distr='squared',colorl='standard',bar_mode='confidence'):
    try_key = [a.split('/')[-1] for a in glob.glob(base + '*')]
    if name != '':
        re1 = [re.search(name+'([0-9]*)',x) for x in try_key]
        dim_list = [x.group(1) for x in re1 if x != None]
        dim_list = [x for x in dim_list if x !='']
        keys = map(int,dim_list)
    else:
        flag = False
        while flag==False:
            try:
                map(int_or_float,try_key)
                flag = True
            except ValueError:
                #print('Internal processing: {}'.format(try_key))
                try_key = [key[1:] for key in try_key]
                if '' in try_key:
                    raise ValueError('A string is empty: {}'.format(try_key))
                else:
                    flag = False
        keys = map(int_or_float, try_key)
    
    if indices!='':
        keys = np.intersect1d(keys,indices)
#    keys = [a.split('/')[-1] for a in glob.glob(base + '*')]
    means, std_devs = {}, {}
    for i, key in enumerate(keys):
        pattern = base + name + str(key) + regex
        answer = find_best(pattern, criterion, retrieve_list,distr=distr)
        if answer[0] is not None:
            means[key], std_devs[key] = answer
    plot_keys = sorted(means.keys())
    dict_means=means
    means = np.asarray([means[key] for key in plot_keys])
    std_devs = np.asarray([std_devs[key] for key in plot_keys])
    scal=1#0.5
    if axis_tmp=='':
        if colorl!='standard':
            (line, caps, barlinecols) = plt.errorbar(plot_keys, means, yerr=scal*std_devs,
                                    marker='o', markersize=5, capsize=5,ecolor=colorl)
#            (line, caps, barlinecols) = plt.errorbar(plot_keys, means, yerr=1.96*std_devs,
#                                    marker='o', markersize=5, capsize=5,ecolor=colorl)
        else:
            (line, caps, barlinecols) = plt.errorbar(plot_keys, means, yerr=scal*std_devs,
                                    marker='o', markersize=5, capsize=5)            
#            (line, caps, barlinecols) = plt.errorbar(plot_keys, means, yerr=1.96*std_devs,
#                                    marker='o', markersize=5, capsize=5)  
    else:
        if colorl!='standard':
            (line, caps, barlinecols) = axis_tmp.errorbar(plot_keys, means, yerr=scal*std_devs,
                                    marker='o', markersize=5, capsize=5,ecolor=colorl)             
#            (line, caps, barlinecols) = axis_tmp.errorbar(plot_keys, means, yerr=1.96*std_devs,
#                                    marker='o', markersize=5, capsize=5,ecolor=colorl)             
        else:
            (line, caps, barlinecols) = axis_tmp.errorbar(plot_keys, means, yerr=scal*std_devs,
                                    marker='o', markersize=5, capsize=5) 
#            (line, caps, barlinecols) = axis_tmp.errorbar(plot_keys, means, yerr=1.96*std_devs,
#                                    marker='o', markersize=5, capsize=5) 
    for cap in caps:
        cap.set_markeredgewidth(1)
    if colorl!='standard':
        line.set_color(colorl)
    return dict_means
#def gather_data
        
        
def get_cheat():
    from tensorflow.examples.tutorials.mnist import input_data    
    mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
    labels = {i: label for (i, label) in enumerate(mnist.test.labels[:64])}
    return labels


def points_contained(cut,thresh=0.5):
    #print cut
    if len(cut)==0:
        return (np.NaN,np.NaN)
    cut = np.asarray(cut)
    m = np.mean(cut)
    n = float(len(cut))
    s=np.std(cut)
    flag = True
    i=0.
    j=2. #if indeed random variable, should be 75% concentrated at least
#    p=float(j/2)
    while flag:
        p=i+(j-i)/2
#        print(p)        
        crit = len(cut[np.abs(cut-m)<=p*s])/n
#        print(crit)
        if crit>=thresh:
            j=p
        else:
            i=p
        if np.abs(i-j) <= 0.01:
            flag=False
            if crit<thresh:
                p=j
                crit = len(cut[np.abs(cut-m)<=p*s])/n
    return (p,crit)