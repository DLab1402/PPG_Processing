
"""
Created on Wed Feb 20 19:50:28 2019

This code using VMD library of Vinícius Rezende Carvalho serves a API class to deploy into specific purpose 
"""
#from __future__ import division# if python 2
import os
import math
import json
import numpy as np
import seaborn as sns
from PIL import Image
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# -*- coding: utf-8 -*-


class ssftFea:
    arg = []

    def __init__(self, data, fs, alpha = 2000, tau = 0., K = 5, DC = 0, init = 1, tol = 1e-7):
        self.fs = fs             # sample frequency
        self.data = data
        self.alpha = alpha       # moderate bandwidth constraint
        self.tau = tau           # noise-tolerance (no strict fidelity enforcement)
        self.K = K               # modes
        self.DC = DC             # no DC part imposed
        self.init = init         # initialize omegas uniformly
        self.tol = tol

    def FreRange(self,u)->np.array:
        pass

    def A(self,u)->np.array:
        pass


    def single_process(self, sig, fstart, fend, ReS = 100, t_cut = 0.1, selec_method = "FreRange", normalize = "True" ):
        #VMD
        u, u_hat, omega = VMD(sig, self.alpha, self.tau, self.K, self.DC, self.init, self.tol)
        if selec_method == "FreRange":
            u = self.FreRange(u)
        elif selec_method == "A":
            u = self.A(u)
        else:
            pass
        L = len(sig)
        s = math.ceil(L*t_cut)
        e = math.ceil(L*(1-t_cut))
        start = np.round(s).astype(np.int64)
        end = np.round(e).astype(np.int64)
        seg = range(end-start)
        z = hilbert(u)
        inse = np.abs(z)
        insp = np.unwrap(np.angle(z))
        insf = np.gradient(insp)[1]*self.fs/(2*np.pi)
        inse = inse[:,seg+start].flatten()
        insf = insf[:,seg+start].flatten()
        time = np.tile(seg,self.K)
        
        #Frequency limit
        t_ind = np.where((insf > fstart) & (insf < fend))[0]
        inse = inse[t_ind]
        insf = insf[t_ind]
        time = time[t_ind]
        
        #quantize
        step = (fend-fstart)/ReS
        findex = np.round(((insf - fstart)/step),0).astype(np.int64)
        #Spectrum
        # col = np.round(np.max(findex)-np.min(findex)).astype(np.int64)+1
        # row = np.round(np.max(time)-np.min(time)).astype(np.int64)+1
        # print(col)
        # print(row)
        H = sp.csr_matrix((inse, (findex, time)))
        #Plot
        # sns.heatmap(H.toarray())
        # plt.show()

        return H.toarray()
    
    def whole_pack_process(self, fstart, fend, Re = 100, t_c= 0.1, se_me = "FreRange", no = "True", file_path = None, width = 512, height = 128 ):
        if file_path == None:
            H = []
            for sig in self.data:
                tem = self.single_process(sig, fstart, fend, ReS = Re, t_cut = t_c, selec_method = se_me, normalize = no )
                H.append(tem)
            return H
        else:
            for index,sig in enumerate(self.data):
                tem = self.single_process(sig, fstart, fend, ReS = Re, t_cut = t_c, selec_method = se_me, normalize = no )
                normalized_image = (tem - np.min(tem)) * (255.0 / (np.max(tem) - np.min(tem)))
                image = Image.fromarray(normalized_image.astype('uint8'))
                image = image.resize((width, height))
                image.save(file_path+"/"+str(index)+".png")
                print(file_path+"/"+str(index)+".png")