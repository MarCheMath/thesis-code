#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:24:33 2019

@author: cheng
"""

from argparse import ArgumentParser
import os
import itertools
import sys

def general_executing(hparams,file_name):
    if hasattr(hparams,'base'):
        base = hparams.base
    else:
        base=[]
    if hasattr(hparams,'submit_mode'):
        submit_mode = hparams.submit_mode
        del hparams.submit_mode
    else:
        submit_mode = 'tmux'
    if hasattr(hparams,'qsub_time'):
        qsub_time = hparams.qsub_time
        del hparams.qsub_time
    else:
        qsub_time = 30000   
    
    fields = {field:getattr(hparams,field) for field in dir(hparams) if not field.startswith('_')}
    fields = {v:field if type(field)==list else [field] for v,field in fields.iteritems() }
    b = ''.join(base)
    for setting in itertools.product(*fields.values()):
        setting = dict(zip(fields.keys(),setting))
        head = ''.join(['--'+str(v1).replace('_','-')+' ' +str(v2)+' ' for v1,v2 in setting.iteritems()])
        #head=head.replace('_','-')
        if submit_mode == 'qsub':
            head = head.replace(" '",""" "'""")
            head = head.replace("' ","""'" """)
            print(head)
        string = str(file_name)+ b+' '+head
        ex_string = 'python -u '+string
        print(submit_mode)
        if submit_mode == 'tmux':
            print('tmux new-session -d '+ex_string)
            os.system('tmux new-session -d '+ex_string)
        elif submit_mode == 'qsub':
#            print("qsub -cwd -N 'CS_VAE' -j y -l h_rt=7200 "+string)
#            os.system("qsub -cwd -N 'CS_VAE' -j y -l h_rt=7200 "+string)
            print("qsub -cwd -N 'CS_VAE' -j y -l h_rt={} ".format(qsub_time)+string)
            os.system("echo "+string+ " | qsub -cwd -N 'CS_VAE' -j y -l h_rt={}".format(qsub_time))
        elif submit_mode == 'cluster':
            print("Cluster "+ex_string)
            os.system("Cluster "+ex_string)
        elif submit_mode == 'vanilla':
            print(ex_string)
            os.system(ex_string)            
        else:
            raise NotImplementedError  

if __name__ == '__main__':
    PARSER = ArgumentParser()
    var_list = sys.argv
    var_list = [x for x in var_list if x[:2]=='--']
    for var in var_list:
        PARSER.add_argument(var, nargs = '+')  
    HPARAMS = PARSER.parse_args()
    if hasattr(HPARAMS,'file_name'):
        file_name = HPARAMS.file_name
        if len(file_name)>0:
            UserWarning('More than one file to execute given. First one chosen.')
        file_name = file_name[0]
    else:
        raise ValueError('No filename given!')

    for k,v in HPARAMS.__dict__.items():
        if len(v)==1:
            setattr(HPARAMS,k,v[0])
    del HPARAMS.file_name
    general_executing(HPARAMS,file_name)
    
