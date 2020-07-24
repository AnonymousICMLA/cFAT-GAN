#!/usr/bin/env python
# coding: utf-8

import numpy as np
import json
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
mpl.rc('text', usetex=True)


# load plotting dictionary
def load_dict():
    with open('data/for_plot/eDict.json', 'r') as fp:
        eDict = json.load(fp)
    return eDict


def get_vec(beam):
    e_vec = [beam, 0.0, 0.0, beam]
    p_vec = [beam, 0.0, 0.0, -np.sqrt(beam**2.0 - 0.938 * 0.938)]
    return e_vec, p_vec


def vec4dot(v1, v2):
    term0 = v1[0]*v2[0]
    term1 = v1[1]*v2[1]
    term2 = v1[2]*v2[2]
    term3 = v1[3]*v2[3]
    return term0 - term1 - term2 - term3


# load pythia data for untrained energies to be plotted in to dictionary
def ext_dict():
    eDict = load_dict()
    paths = ['data/train_data/tape_10', 'data/for_plot/tape_15',
             'data/train_data/tape_20', 'data/for_plot/tape_25',
             'data/train_data/tape_30', 'data/for_plot/tape_35',
             'data/train_data/tape_40', 'data/for_plot/tape_45',
             'data/train_data/tape_50', 'data/for_plot/tape_60',
             'data/for_plot/tape_70', 'data/for_plot/tape_80',
             'data/for_plot/tape_90']
    beam = list(eDict['Generated'].keys())
    eDict['Pythia'] = {}
    for i in range(len(beam)):
        beamEnergy = float(beam[i])
        beamElectron4vec, beamProton4vec = get_vec(beamEnergy)
        eDict['Pythia'][str(int(beamEnergy))] = [[], [], [], [], [], [], []]
        with open(paths[i], "r") as fp:
            line = fp.readline()
            while line:
                particle = re.split(' +', line)
                px = float(particle[1])
                py = float(particle[2])
                pz = float(particle[3])
                pxy = px*py
                pxz = px*pz
                pyz = py*pz
                pt = np.sqrt(px*px+py*py)
                e = np.sqrt(px*px+py*py+pz*pz)
                pzt = pz/(pt+0.01)

                q = [i-j for i, j in zip(beamElectron4vec, [e, px, py, pz])]
                Q2 = -vec4dot(q, q)
                if Q2 > 1.0 and e < beamEnergy and px > 1.0:
                    xbj = Q2 / (2.0 * vec4dot(q, beamProton4vec))
                    eDict['Pythia'][beam[i]][0].append(e)
                    eDict['Pythia'][beam[i]][1].append(px)
                    eDict['Pythia'][beam[i]][2].append(py)
                    eDict['Pythia'][beam[i]][3].append(pz)
                    eDict['Pythia'][beam[i]][4].append(pt)
                    eDict['Pythia'][beam[i]][5].append(Q2)
                    eDict['Pythia'][beam[i]][6].append(xbj)
                line = fp.readline()
    return eDict


# plot feature distributions from Pythia and cFAT-GAN at:
# 10,15,20,25,30,35,40,45,50,60,70,80,90 GeV
def feat_dist(eDict):
    fig = plt.figure(figsize=(8, 10))
    gs = gridspec.GridSpec(3, 2, top=0.95, bottom=0.06, left=0.075, right=0.95,
                           hspace=0.12, wspace=0.2)
    beam = list(eDict['Generated'].keys())
    points = np.linspace(0, 1, len(beam))
    colors = cm.hsv(points)

    c10 = Line2D([], [], color=colors[0], linestyle=('solid'),
                 lw=2, label=r'$\rm 10~GeV$')
    c15 = Line2D([], [], color=colors[1], linestyle=('solid'),
                 lw=2, label=r'$\rm 15~GeV$')
    c20 = Line2D([], [], color=colors[2], linestyle=('solid'),
                 lw=2, label=r'$\rm 20~GeV$')
    c25 = Line2D([], [], color=colors[3], linestyle=('solid'),
                 lw=2, label=r'$\rm 25~GeV$')
    c30 = Line2D([], [], color=colors[4], linestyle=('solid'),
                 lw=2, label=r'$\rm 30~GeV$')
    c35 = Line2D([], [], color=colors[5], linestyle=('solid'),
                 lw=2, label=r'$\rm 35~GeV$')
    c40 = Line2D([], [], color=colors[6], linestyle=('solid'),
                 lw=2, label=r'$\rm 40~GeV$')
    c45 = Line2D([], [], color=colors[7], linestyle=('solid'),
                 lw=2, label=r'$\rm 45~GeV$')
    c50 = Line2D([], [], color=colors[8], linestyle=('solid'),
                 lw=2, label=r'$\rm 50~GeV$')
    c60 = Line2D([], [], color=colors[9], linestyle=('solid'),
                 lw=2, label=r'$\rm 60~GeV$')
    c70 = Line2D([], [], color=colors[10], linestyle=('solid'),
                 lw=2, label=r'$\rm 70~GeV$')
    c80 = Line2D([], [], color=colors[11], linestyle=('solid'),
                 lw=2, label=r'$\rm 80~GeV$')
    c90 = Line2D([], [], color=colors[12], linestyle=('solid'),
                 lw=2, label=r'$\rm 90~GeV$')
    pythia = Line2D([], [], color='k', linestyle=('solid'),
                    lw=2, label=r'$\rm True$')
    gan = Line2D([], [], color='k', linestyle=('dotted'),
                 lw=2, label=r'$\rm cFAT-GAN$')

    # plot E' distributions in 10 - 50 GeV range
    ax = fig.add_subplot(gs[0, 0])
    ax.set_yscale('log', nonposy='clip')
    ax.set_xscale('linear', nonposx='clip')

    for i in range(9):
        ax.hist(eDict['Pythia'][beam[i]][0], bins=eBins, density=True,
                histtype='step', color=colors[i], linestyle='solid')
        ax.hist(eDict['Generated'][beam[i]][0], bins=eBins, density=True,
                histtype='step', color=colors[i], linestyle='dotted')

    ax.xaxis.set_label_coords(0.07, 0.94)
    ax.set_xlabel(r"\boldmath$E'$", size=20, rotation=0)
    ax.set_title(r"$\rm Trained~Region$", fontsize=18)
    ax.set_yticks([0.01, 0.1, 1.0])
    ax.set_yticklabels([r'$10^-2$', r'$10^-1$', r'$10^0$'], fontsize=13)
    plt.xticks(fontsize=13)
    ax.set_ylim(0.0015, 2.0)

    # plot E' distributions in 50 - 100 GeV range
    ax = fig.add_subplot(gs[0, 1])
    ax.set_yscale('log', nonposy='clip')
    ax.set_xscale('linear', nonposx='clip')

    for i in range(4):
        ax.hist(eDict['Pythia'][beam[i+9]][0], bins=ext_eBins, density=True,
                histtype='step', color=colors[i+9], linestyle='solid')
        ax.hist(eDict['Generated'][beam[i+9]][0], bins=ext_eBins, density=True,
                histtype='step', color=colors[i+9], linestyle='dotted')

    ax.xaxis.set_label_coords(0.07, 0.94)
    ax.set_xlabel(r"\boldmath$E'$", size=20, rotation=0)
    ax.set_title(r"$\rm Extrapolated~Region$", fontsize=18)
    ax.set_yticks([0.01, 0.1, 1.0])
    ax.set_yticklabels([r'$10^-2$', r'$10^-1$', r'$10^0$'], fontsize=13)
    plt.xticks(fontsize=13)
    ax.set_ylim(0.0015, 2.0)
    ax.set_xlim(25, 100)

    # plot p_t distributions at 10 - 50 GeV range
    ax = fig.add_subplot(gs[1, 0])
    ax.set_yscale('log', nonposy='clip')
    ax.set_xscale('linear', nonposx='clip')

    for i in range(9):
        ax.hist(eDict['Pythia'][beam[i]][4], bins=tBins, density=True,
                histtype='step', color=colors[i], linestyle='solid')
        ax.hist(eDict['Generated'][beam[i]][4], bins=tBins, density=True,
                histtype='step', color=colors[i], linestyle='dotted')

    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    ax.set_xlim(0.0, 35)
    ax.set_ylabel(r'$\rm Normalized$ $\rm Yield$',
                  fontsize=18, fontweight='black')
    ax.xaxis.set_label_coords(0.17, 0.95)
    ax.set_xlabel(r"\boldmath$k'_T$", size=20, rotation=0)

    # plot p_t distributions for 50 - 100 GeV range
    ax = fig.add_subplot(gs[1, 1])
    ax.set_yscale('log', nonposy='clip')
    ax.set_xscale('linear', nonposx='clip')

    for i in range(4):
        ax.hist(eDict['Pythia'][beam[i+9]][4], bins=tBins, density=True,
                histtype='step', color=colors[i+9], linestyle='solid')
        ax.hist(eDict['Generated'][beam[i+9]][4], bins=tBins, density=True,
                histtype='step', color=colors[i+9], linestyle='dotted')

    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    ax.set_xlim(0.0, 35)
    ax.xaxis.set_label_coords(0.17, 0.95)
    ax.set_xlabel(r"\boldmath$k'_T$", size=20, rotation=0)

    # plot p_z distributions for 10 - 50 GeV range
    ax = fig.add_subplot(gs[2, 0])
    ax.set_yscale('log', nonposy='clip')
    ax.set_xscale('linear', nonposx='clip')

    for i in range(9):
        ax.hist(eDict['Pythia'][beam[i]][3], bins=zBins, density=True,
                histtype='step', color=colors[i], linestyle='solid')
        ax.hist(eDict['Generated'][beam[i]][3], bins=zBins, density=True,
                histtype='step', color=colors[i], linestyle='dotted')
    ax.set_yticks([0.01, 0.1, 1.0])
    ax.set_yticklabels([r'$10^-2$', r'$10^-1$', r'$10^0$'], fontsize=13)
    plt.xticks(fontsize=13)

    ax.set_ylim(0.0015, 2.0)
    ax.xaxis.set_label_coords(0.07, 0.93)
    ax.set_xlabel(r"\boldmath $k'_z$", fontsize=20, fontweight='black')

    # plot p_z distributions for 50 - 100 GeV range
    ax = fig.add_subplot(gs[2, 1])
    ax.set_yscale('log', nonposy='clip')
    ax.set_xscale('linear', nonposx='clip')

    for i in range(4):
        ax.hist(eDict['Pythia'][beam[i+9]][3], bins=ext_zBins, density=True,
                histtype='step', color=colors[i+9], linestyle='solid')
        ax.hist(eDict['Generated'][beam[i+9]][3], bins=ext_zBins, density=True,
                histtype='step', color=colors[i+9], linestyle='dotted')
    ax.set_yticks([0.01, 0.1, 1.0])
    ax.set_yticklabels([r'$10^-2$', r'$10^-1$', r'$10^0$'], fontsize=13)
    plt.xticks(fontsize=13)

    ax.set_ylim(0.0015, 2.0)
    ax.set_xlim(25, 100)
    ax.xaxis.set_label_coords(0.07, 0.93)
    ax.set_xlabel(r"\boldmath $k'_z$", fontsize=20, fontweight='black')

    leg1 = ax.legend(handles=[pythia, gan], loc=[-0.65, -0.25],
                     ncol=2, fontsize=14)
    leg2 = ax.legend(handles=[c10, c15, c20, c25, c30, c35, c40,
                     c45, c50, c60, c70, c80, c90],
                     loc=[-1.0, -0.73], ncol=4, fontsize=14)
    ax.add_artist(leg1)

    plt.savefig('gallery/fig1.pdf', bbox_inches='tight')


# plot Q2 xbj joint distributions from Pythia and cFAT-GAN
# at 4 trained energies and 4 interpolated energies
def Q2_xbj(eDict):
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(4, 4, top=0.95, bottom=0.06, left=0.075, right=0.95,
                           hspace=0.075, wspace=0.05)
    keys = [10, 20, 30, 40, 15, 25, 35, 45]
    ppos = [[0, 0], [0, 1], [0, 2], [0, 3], [2, 0], [2, 1], [2, 2], [2, 3]]
    gpos = [[1, 0], [1, 1], [1, 2], [1, 3], [3, 0], [3, 1], [3, 2], [3, 3]]
    titles = [r'$\rm 10~GeV$', r'$\rm 20~GeV$', r'$\rm 30~GeV$',
              r'$\rm 40~GeV$', r'$\rm 15~GeV$', r'$\rm 25~GeV$',
              r'$\rm 35~GeV$', r'$\rm 45~GeV$']
    lev = [0.1, 0.2, 0.3, 0.4]

    # plot Pythia Q2 xbj correlation plots
    for i in range(8):
        ax = fig.add_subplot(gs[ppos[i][0], ppos[i][1]])
        beam = keys[i]
        x = eDict['Pythia'][str(keys[i])][6]
        y = eDict['Pythia'][str(keys[i])][5]
        xbins = np.logspace(-3, 0, 50)
        ybins = np.logspace(0, 3, 50)
        counts, ybins_, xbins_, image = ax.hist2d(x, y, bins=[xbins, ybins],
                                                  norm=LogNorm(),
                                                  cmap=cm.Greys)
        counts, xbins_, ybins_ = np.histogram2d(x, y, bins=(xbins, ybins))
        xbins_ = 0.5 * (xbins_[:-1] + xbins_[1:])
        ybins_ = 0.5 * (ybins_[:-1] + ybins_[1:])
        X, Y = np.meshgrid(xbins_, ybins_)
        cmax = np.amax(counts)
        ax.contour(X, Y, counts.transpose(),
                   extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
                   cmap=cm.gist_rainbow,
                   levels=[cmax * _ for _ in lev],
                   linewidths=3)

        ax.set_xscale("log")               # <- Activate log scale on X axis
        ax.set_yscale("log")               # <- Activate log scale on Y axis

        s = 4 * beam**2
        x = np.logspace(-3, 0, 10)
        Q2 = s * x
        ax.plot(x, Q2, 'k-')

        ax.tick_params(axis='x', which='mayor',
                       labelsize=13, direction='in', length=5)
        ax.tick_params(axis='y', which='mayor',
                       labelsize=13, direction='in', length=5)
        ax.set_xticks([0.001, 0.01, 0.1, 1])
        ax.set_xticklabels([])
        ax.set_yticks([10, 100])
        ax.set_yticklabels([])
        ax.set_title(titles[i], fontsize=16)
        if i == 0:
            ax.yaxis.set_label_coords(-0.12, 0.8)
            ax.set_ylabel(r'\boldmath$Q^2$', size=25, rotation=0)
        if i == 3 or i == 7:
            ax.text(1.05, 0.50, r'$\rm True$', size=24, transform=ax.transAxes)

    # plot cFAT-GAN Q2 xbj correlation plots
    for i in range(8):
        ax = fig.add_subplot(gs[gpos[i][0], gpos[i][1]])
        beam = float(keys[i])
        x = eDict['Generated'][str(keys[i])][6]
        y = eDict['Generated'][str(keys[i])][5]
        xbins = np.logspace(-3, 0, 50)
        ybins = np.logspace(0, 3, 50)
        counts, ybins_, xbins_, image = ax.hist2d(x, y, bins=[xbins, ybins],
                                                  norm=LogNorm(),
                                                  cmap=cm.Greys)
        counts, xbins_, ybins_ = np.histogram2d(x, y, bins=(xbins, ybins))

        xbins_ = 0.5*(xbins_[:-1] + xbins_[1:])
        ybins_ = 0.5*(ybins_[:-1] + ybins_[1:])
        X, Y = np.meshgrid(xbins_, ybins_)
        cmax = np.amax(counts)
        ax.contour(X, Y, counts.transpose(),
                   extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
                   cmap=cm.gist_rainbow,
                   levels=[cmax * _ for _ in lev],
                   linewidths=3)

        ax.set_xscale("log")               # <- Activate log scale on X axis
        ax.set_yscale("log")               # <- Activate log scale on Y axis

        s = 4 * beam**2
        x = np.logspace(-3, 0, 10)
        Q2 = s * x
        ax.plot(x, Q2, 'k-')

        ax.tick_params(axis='x', which='mayor',
                       labelsize=13, direction='in', length=5)
        ax.tick_params(axis='y', which='mayor',
                       labelsize=13, direction='in', length=5)
        ax.set_xticks([0.001, 0.01, 0.1, 1])
        ax.set_yticks([10, 100])
        ax.set_yticklabels([])
        ax.yaxis.set_label_coords(-0.12, 0.8)
        ax.text(0.025, 0.85, r'$\rm Interpolated$',
                size=20, transform=ax.transAxes)
        if i == 3 or i == 7:
            ax.text(1.05, 0.50, r'$\rm cFAT{-}GAN$',
                    size=20, transform=ax.transAxes)
        if i < 4:
            ax.set_xticklabels([])
        if i > 3:
            ax.set_xticklabels([r'$0.001$', r'$0.01$', r'$0.1$'], size=15)
        if i == 7:
            ax.xaxis.set_label_coords(0.95, -0.075)
            ax.set_xlabel(r'\boldmath$x_{bj}$', size=25, rotation=0)

    plt.savefig('gallery/fig2.pdf', bbox_inches='tight')


# Bins for each feature in plotting dictionary
zBins = np.linspace(0.0, 55.0, 256)
ext_zBins = np.linspace(0.0, 95.0, 256)
eBins = np.linspace(0.0, 55.0, 256)
ext_eBins = np.linspace(0.0, 95.0, 256)
Q2Bins = np.logspace(0.0, 4.0, 100)
xbjBins = np.logspace(-3.0, 0.0, 100)
xBins = np.linspace(-20.0, 20.0, 256)
yBins = np.linspace(-20.0, 20.0, 256)
tBins = np.linspace(0.0, 50.0, 256)


def main():

    eDict = ext_dict()
    feat_dist(eDict)
    Q2_xbj(eDict)
    print('finished')

main()
