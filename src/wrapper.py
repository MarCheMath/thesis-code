#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 19:47:22 2019

@author: marche
"""

class model(object):
    def __init__(self,model_name,mdel_type):
        self.model_name = model_name
        self.model_type = model_type
        if model_name == 'BEGAN':
            import BEGAN
            self.model = BEGAN.BEGAN()
    
    def build_flexible(self):
        