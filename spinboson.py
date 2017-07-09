# -*- coding: utf-8 -*-
"""
(C) Steven Byrnes 2017

Currently trying to reproduce parts of "Models relevant to excess heat
production in Fleischmann-Pons experiments", Hagelstein & Chaudhary

Requires Python 3.5 or higher.
"""
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from numpy.linalg import eigh
from math import sqrt

############# Set up system, construct the Hamiltonian matrix #################

def S_plus_coef(S,m):
    """S+|S,m> = (BLANK) * ħ |S,m+1>. Calculate BLANK."""
    assert (S+m) % 1 == 0
    return sqrt((S-m) * (S+m+1))
def S_minus_coef(S,m):
    """S-|S,m> = (BLANK) * ħ |S,m-1>. Calculate BLANK."""
    assert (S+m) % 1 == 0
    return sqrt((S+m) * (S-m+1))

def H_and_basis(S, min_n, max_n, sector, V, ħω, ΔE):
    """Construct the Hamiltonian matrix, and a table of contents defining the
    basis it is written in.

    INPUTS:

    * S, the total spin describing the collection of 2-level systems.
    * max_n, min_n, the range of possible phonon counts that we are modeling.
      (inclusive.)
    * sector is either 'odd' or 'even' or 'both' depending on parity of S+m+n
      (since the interaction conserves parity we can model each separately)
    * V, ΔE, ħω, the three parameters in the Hamiltonian (units of energy)

    OUTPUTS:

    * H, the Hamiltonian matrix
    * index_from_Smn, a dictionary with the property that
      index_from_Smn[(S,m,n)] gives the index of the row / column in H
      corresponding to (S,m,n).
    * Smn_from_index, a list in which Smn_from_index[i] is the (S,m,n)
      describing the i'th row / column in H.
    """
    ### "Table of contents" for basis for H matrix
    possible_ns = range(min_n, max_n+1)
    assert ((2*S) % 1 == 0) and (S >= 0)
    possible_ms = np.arange(2*S+1) - S
    Smn_list = product([S], possible_ms, possible_ns)
    if sector == 'even':
        Smn_from_index = [(S,m,n) for (S,m,n) in Smn_list if (S+m+n) % 2 == 0]
    elif sector == 'odd':
        Smn_from_index = [(S,m,n) for (S,m,n) in Smn_list if (S+m+n) % 2 == 1]
    else:
        assert sector == 'both'
        Smn_from_index = Smn_list
    index_from_Smn = {Smn:index for (index,Smn) in enumerate(Smn_from_index)}

    ### Fill in entries of H matrix
    H = np.zeros(shape=(len(Smn_from_index), len(Smn_from_index)))

    # H = ΔE·Sz/ħ + ħ·ω0(a†·a + ½) + V(a† + a)(S+ + S-)/ħ
    for i,(S,m,n) in enumerate(Smn_from_index):
        # ΔE·Sz/ħ term ... note that Sz|S,m> = ħm|S,m>
        H[i,i] += ΔE * m
        # ħ·ω0(a†·a + ½) term
        H[i,i] += ħω * (n + 1/2)
        # V·a†·S+/ħ term
        if (S,m+1,n+1) in index_from_Smn:
            H[index_from_Smn[(S,m+1,n+1)],i] = S_plus_coef(S,m) * sqrt(n+1) * V
        # V·a·S+/ħ term
        if (S,m+1,n-1) in index_from_Smn:
            H[index_from_Smn[(S,m+1,n-1)],i] = S_plus_coef(S,m) * sqrt(n) * V
        # V·a†·S-/ħ term
        if (S,m-1,n+1) in index_from_Smn:
            H[index_from_Smn[(S,m-1,n+1)],i] = S_minus_coef(S,m) * sqrt(n+1) * V
        # V·a·S-/ħ term
        if (S,m-1,n-1) in index_from_Smn:
            H[index_from_Smn[(S,m-1,n-1)],i] = S_minus_coef(S,m) * sqrt(n) * V
    return H, index_from_Smn, Smn_from_index


############## PLOT ALL ENERGIES #############################################

def plot_all_energies(S, min_n, max_n, sector, V_max, ΔE, ħω):
    """Plot all the eigenvalues"""
    V_list = np.linspace(0, V_max, num=100)
    energies_array = []
    for V in V_list:
        H,_,_ = H_and_basis(S, min_n, max_n, sector, V, ħω, ΔE)
        energies,_ = eigh(H)
        energies_array.append(energies)
    energies_array = np.array(energies_array)
    plt.figure(figsize=(5,3))
    for i in range(np.shape(H)[1]):
        plt.plot(V_list / ΔE, energies_array[:,i]/ħω)
    plt.xlabel('V / ΔE')
    plt.ylabel('Eigenstate energy / ħω')
    plt.title('All energy levels')

if True:
    # A few illustrative plot_all_energies() plots
    ΔE = 1 # shouldn't matter
    S = 0.5
    min_n = 0
    max_n = 100
    num_quanta = 11
    #### DOWN TO ZERO PHONONS
    plot_all_energies(S=S, min_n=min_n, max_n=max_n, sector='even', V_max=ΔE/5,
                      ΔE=ΔE, ħω=ΔE/num_quanta)
    plt.ylim(-7, 60)
    plt.xlim(0, 1/5)
    plt.tight_layout()
    plt.savefig('AllEnergies_FewPhonons.png', dpi=300)

    #### DOWN TO ZERO PHONONS, but with dashed line
    plot_all_energies(S=S, min_n=min_n, max_n=max_n, sector='even', V_max=ΔE/5,
                      ΔE=ΔE, ħω=ΔE/num_quanta)
    V_list = np.linspace(ΔE/100, ΔE/5, num=100)
    plt.plot(V_list/ΔE, .08/(V_list/ΔE)**2, 'k--')
    plt.ylim(-7, 60)
    plt.xlim(0, 1/5)
    plt.tight_layout()
    plt.savefig('AllEnergies_FewPhonons_dash.png', dpi=300)

    #### MANY PHONONS
    min_n = 1000
    max_n = 1100
    n_avg = 1050
    V_max = ΔE/sqrt(n_avg)
    plot_all_energies(S=S, min_n=min_n, max_n=max_n, sector='even', V_max=V_max,
                      ΔE=ΔE, ħω=ΔE/num_quanta)
    plt.tight_layout()
    plt.ylim(1035, 1065)
    plt.xlim(0, V_max/ΔE)
    plt.savefig('AllEnergies_LotsaPhonons.png', dpi=300)

    #### MANY PHONONS, with dashed line
    min_n = 1000
    max_n = 1100
    n_avg = 1050
    V_max = ΔE/sqrt(n_avg)
    ħω = ΔE/num_quanta
    plot_all_energies(S=S, min_n=min_n, max_n=max_n, sector='even', V_max=V_max,
                      ΔE=ΔE, ħω=ħω)
    V_list = np.linspace(V_max/1000, V_max, num=100)
    n = 1043
    g_list = V_list * sqrt(n) / ΔE
    plt.plot(V_list/ΔE, (ħω*(n+1/2) + ΔE/2 * np.sqrt(1+8*g_list**2))/ħω, 'k--')
    V_list = np.linspace(V_max/1000, V_max, num=100)
    n = 1054
    g_list = V_list * sqrt(n) / ΔE
    plt.plot(V_list/ΔE, (ħω*(n+1/2) - ΔE/2 * np.sqrt(1+8*g_list**2))/ħω, 'k--')
    plt.tight_layout()
    plt.ylim(1035, 1065)
    plt.xlim(0, V_max/ΔE)
    plt.savefig('AllEnergies_LotsaPhonons_dash.png', dpi=300)

################### ONE STATE #################################################

# ...but first, a plotting utility
def fill_between_stack(x, y_list, label_list):
    """The command
            fill_between_stack(x, [y1,y2,...] ['label1', 'label2', ...])
    is shorthand for
            plt.fill_between(x, 0, y1, label='label1')
            plt.fill_between(x, y1, y1+y2, label='label2')
            ...etc...
    e.g. if we have quantities that sum to 100%, this shows their contributions.
    """
    for i in range(len(y_list)):
        plt.fill_between(x, sum(y_list[0:i]), sum(y_list[0:(i+1)]),
                         label=label_list[i])

def plot_one_state(S, min_n, max_n, V_max, ΔE, ħω, m, n):
    """Make some plots related to one smooth curve in the energy diagram (i.e.
    "continuing straight" at each anticrossing) """
    assert min_n <= n <= max_n
    sector = 'even' if (S+m+n)%2==0 else 'odd'
    V_list = np.linspace(0, V_max, num=30)
    # We will be plotting some of the eigenstate's weights in the unperturbed
    # basis. This is the list of states, and how to refer to them in the plot
    # legend
    weights_to_plot = [[(S,m,n),    'm, n'],
                       [(S,m,n+2),  'm, n+2'],
                       [(S,m,n-2),  'm, n-2'],
                       [(S,m,n+4),  'm, n+4'],
                       [(S,m,n-4),  'm, n-4'],
                       [(S,m,n+6),  'm, n+6'],
                       [(S,m,n-6),  'm, n-6'],
                       [(S,-m,n+1), '-m, n+1'],
                       [(S,-m,n-1), '-m, n-1'],
                       [(S,-m,n+3), '-m, n+3'],
                       [(S,-m,n-3), '-m, n-3']]
    weights_data = [[] for x in weights_to_plot]
    energy_list = []
    g_list = []
    for i,V in enumerate(V_list):
        ###### Calculate H and the state in question and its energy
        H,index_from_Smn,Smn_from_index = H_and_basis(S, min_n, max_n, sector, V, ħω, ΔE)
        energies, eigenstates = eigh(H)
        if i == 0:
            # For V=0, we can just look up the state which looks correct in the
            # unperturbed basis
            index_in_H_basis = index_from_Smn[(S,m,n)]
            index = np.argmax(abs(eigenstates[index_in_H_basis,:]))
        else:
            # For V>0, we find the state which most closely matches the
            # previous iteration. Note that this procedure hopes that we never
            # happen to land super-close to an anticrossing point.
            abs_components = [abs(eigenstates[:,j] @ eigenstate) for j in range(len(H))]
            index = np.argmax(abs_components)
        energy_list.append(energies[index])
        eigenstate = eigenstates[:,index]

        ###### Calculate some interesting aspects of this stat, particularly
        # the weights in the unperturbed basis
        for i, ((S_now,m_now,n_now), _) in enumerate(weights_to_plot):
            weights_data[i].append(eigenstate[index_from_Smn[(S_now,m_now,n_now)]]**2)
        g_list.append(V * sqrt(n) / ΔE)
    plt.figure()
    plt.plot(V_list/ΔE, [E/ħω for E in energy_list])
    plt.figure(figsize=(3,2))
    plt.plot(g_list, [(e-ħω*(n+0.5))/m/ΔE for e in energy_list], label='numerical')
    plt.plot(g_list, [sqrt(1+8*g**2) for g in g_list], label='√(1+8g²)')
    plt.legend()
    plt.xlim(min(g_list), max(g_list))
    plt.grid()
    plt.xlabel('g')
    plt.ylabel('ΔE(g) / ΔE')
    plt.tight_layout()
    plt.savefig('ΔE(g).png', dpi=300)
    plt.figure(figsize=(5,4))
    fill_between_stack(g_list, [np.array(x) for x in weights_data], [i[1] for i in weights_to_plot])
    plt.xlabel('g')
    plt.legend()
    plt.title('Contributions to this eigenstate')
    plt.tight_layout()
    plt.savefig('OneState_Contributions.png', dpi=300)


if True:
    # Example plot of the weights for one example eigenstate
    ΔE = 1 # shouldn't matter
    S = 0.5
    m=S
    min_n = 1000
    max_n = 1100
    n_avg = 1050
    n = n_avg
    V_max = ΔE/sqrt(n)
    num_quanta = 11
    ħω = ΔE/num_quanta
    plot_one_state(S=S, min_n=min_n, max_n=max_n, V_max=V_max, ΔE=ΔE, ħω=ħω,
                   m=m,n=n)


########### ANTICROSSING ######################################################

def get_anticrossing_energies(S, min_n, max_n, sector, V_max, ΔE, ħω):
    """..."""
    n_avg = (min_n + max_n) / 2
    H,_,_ = H_and_basis(S, min_n, max_n, sector, 0, ħω, ΔE)
    index = len(H) // 2
    def energy_separation(V, i=index):
        """energy difference btwn eigenstates i & i+1 (ordered by energy)"""
        H,_,_ = H_and_basis(S, min_n, max_n, sector, V, ħω, ΔE)
        # pick an item halfway through
        energies,_ = eigh(H)
        return energies[i+1] - energies[i]
    # sample at a bunch of points to start
    V_list = np.linspace(0, V_max, num=100)
    energy_sep_list_A = [energy_separation(V) for V in V_list]
    energy_sep_list_B = [energy_separation(V,index-1) for V in V_list]
    results_g_list = []
    results_dE_list = []
    for i in range(len(V_list) - 2):
        if (energy_sep_list_A[i] >= energy_sep_list_A[i+1]
            and energy_sep_list_A[i+2] >= energy_sep_list_A[i+1]):
            # local min here
            V_atmin=minimize_scalar(energy_separation, bracket=V_list[i:i+3]).x
            print('g:', V_atmin * sqrt(n_avg)/ΔE)
            print(energy_separation(V_atmin) / ΔE)
            results_g_list.append(V_atmin * sqrt(n_avg)/ΔE)
            results_dE_list.append(energy_separation(V_atmin) / ΔE)

        if (energy_sep_list_B[i] >= energy_sep_list_B[i+1]
            and energy_sep_list_B[i+2] >= energy_sep_list_B[i+1]):
            # local min here
            V_atmin=minimize_scalar(energy_separation, args=index-1, bracket=V_list[i:i+3]).x
            print('g:', V_atmin * sqrt(n_avg)/ΔE)
            print(energy_separation(V_atmin, index-1) / ΔE)
            results_g_list.append(V_atmin * sqrt(n_avg)/ΔE)
            results_dE_list.append(energy_separation(V_atmin, index-1) / ΔE)
    return results_g_list, results_dE_list

if True:
    # log plots for 11, 21, 31, to reproduce Fig. 2.
    min_n = 1000
    max_n = 1300
    num_quanta = 11
    S = 0.5
    ΔE = 1 # shouldn't matter
    V_max = ΔE/sqrt(min_n)
    plt.figure(figsize=(5,3))
    for num_quanta in (11, 21, 31):
        results_g_list, results_dE_list = get_anticrossing_energies(S=S,
                                                min_n=min_n, max_n=max_n, sector='even',
                                                V_max=V_max, ΔE=ΔE, ħω=ΔE/num_quanta)
        plt.semilogy(results_g_list, results_dE_list,marker='.')
    plt.xlim(0,1)
    plt.ylim(1e-7, 1e-2)
    plt.grid()
    plt.xlabel('g')
    plt.ylabel('Anticrossing splitting / ΔE')
    plt.title('Reproducing Fig. 2')
    plt.tight_layout()
    plt.savefig('AnticrossingFig.png', dpi=300)
