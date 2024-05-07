import copy
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from qutip import *
from scipy.integrate import quad
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Dense

# Utilities functions


# Functions to save and load the datas
def save_object(obj, filename):
    try:
        with open(filename, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)


def process_parallel_data(data):
    x = []
    y = []
    eta = []

    if len(data[0]) == 4:
        s = []
        for d in data:
            x += d[0]
            y += d[1]
            eta += d[2]
            s += d[3]
        output = x, y, eta, s

    else:
        for d in data:
            x += d[0]
            y += d[1]
            eta += d[2]
        output = x, y, eta

    return output


def rejection_sampling(pdf, xrange, num_trajectories, maxp=None):
    """
    Generates samples from a probability distribution using the rejection sampling technique.
    Returns a list of sampled values that are drawn according to the probability density function `pdf`.
    """
    samples = []
    xlist = np.linspace(xrange[0], xrange[-1], 9999)
    y_max = np.max(pdf(xlist)) if maxp is None else maxp

    while len(samples) < num_trajectories:
        x = np.random.uniform(
            low=xrange[0], high=xrange[-1], size=1000
        )  # Sample in batches
        y = np.random.uniform(low=0, high=y_max, size=1000)
        accepted = y <= pdf(x)
        samples.extend(x[accepted])
        if len(samples) >= num_trajectories:  # Stop if enough samples are collected
            return samples[:num_trajectories]

    return samples


# Definition of probability distributions
def gaussian(x, σ=17.6068):
    """
    Returns the value of a gaussian with mean 0 and variance σ^2.
    """
    return np.exp(-(x**2) / (2 * σ**2)) / (σ * np.sqrt(2 * np.pi))


def gaussian_2d(x, y, σ=17.6068):
    """
    Returns the value of a 2D gaussian with mean 0 and variance σ^2 on both axis.
    """
    return np.exp(-(x**2 + y**2) / (2 * σ**2)) / (σ * np.sqrt(2 * np.pi)) ** 2


# Definition of states and operators
g = basis(3, 0)
e = basis(3, 1)
r = basis(3, 2)

Basis = [basis(3, i) for i in range(3)]

sig = [[Basis[i] * Basis[j].dag() for j in range(3)] for i in range(3)]

gg = sig[0][0]
ee = sig[1][1]
rr = sig[2][2]
gr = sig[0][2]
ge = sig[0][1]
re = sig[2][1]

# Part of control Hamiltonian assosiated with tunneling rate Ωp(t)
H1 = (g * e.dag() + e * g.dag()) / 2
# Part of control Hamiltonian assosiated with tunneling rate Ωs(t)
H2 = (e * r.dag() + r * e.dag()) / 2


# Hamiltonian for (anti-)correlated noise (non-Markovian)
def get_H_CN(η, X, Omega_p, Omega_s, pars):
    # Diagonal Hamiltonian
    H0 = (X + pars["δp"]) * e * e.dag() + (η * X + pars["δ"]) * r * r.dag()

    H = [H0, [H1, Omega_p], [H2, Omega_s]]

    return H


# Hamiltonia for not correlated (uncorrelated) noise (non-Markovian)
def get_H_UCN(η, X1, X2, Omega_p, Omega_s, pars):
    # Diagonal Hamiltonian
    H0 = (X1 + pars["δp"]) * e * e.dag() + (η * X2 + pars["δ"]) * r * r.dag()

    H = [H0, [H1, Omega_p], [H2, Omega_s]]

    return H


# Returns function given an array and time points
def function_from_array(y, x):
    """
    Creates a lookup function from paired arrays of y and x values.
    """
    if y.shape[0] != x.shape[0]:
        raise ValueError("y and x must have the same first dimension")

    yx = np.column_stack((y, x))
    yx = yx[yx[:, -1].argsort()]

    def func(t, args):
        idx = np.searchsorted(yx[1:, -1], t, side="right")
        return yx[idx, 0]

    return func


# Hamiltonian for nonquasistatic noise (Markovian)
# This Hamiltonian is required if the calculation is done by averaging over many realizations of noise
def get_H_nonQS(t, η, x, Omega_p, Omega_s, pars):

    X_func = function_from_array(x, t)

    # Diagonal Hamiltonian
    H0 = pars["δp"] * e * e.dag() + pars["δ"] * r * r.dag()
    # Diagonal noise
    H_noise = e * e.dag() + η * r * r.dag()

    H = [H0, [H_noise, X_func], [H1, Omega_p], [H2, Omega_s]]

    return H


# Hamiltonian for nonquasistatic noise (Markovian)
# This Hamiltonian is required if the calculation is done using Master equation
def get_H_nonQS_ME(η, Omega_p, Omega_s, pars):

    # Diagonal Hamiltonian
    H0 = pars["δp"] * e * e.dag() + pars["δ"] * r * r.dag()

    H = [H0, [H1, Omega_p], [H2, Omega_s]]

    return H


# If noise = 1(by default) this func calculates the population of state |2> at final time tf (efficiency),
# with perfect projective measurement i.e. considering (anti-)correlated non-Markovian noise
# If noise = any other number, it calculates the efficiency considering uncorrelated non-Markovian noise
def eff_nonMarkovian(
    t, pdf, Omega_p, Omega_s, Ωp_max, Ωs_max, num_trajectories, η, pars, noise=1
):
    """Calculates the efficiency (the population of state |2> at final time 't[-1]') in the quantum system
    influenced by non-Markovian noise, either (anti-)correlated or uncorrelated. The function uses different
    types of non-Markovian noise to determine how it affects the system's final state efficiency.

    Returns:
        float: The efficiency of the population in state |2> at time 't[-1]', calculated
               based on the specified type of non-Markovian noise.

    Note:
        Efficiency calculation for Markovian noise in this particular way is not supported!!!
    """

    weighted_sum = 0
    w = 0

    N = pars["N"]
    xrange = pars["xrange"]
    ψA0 = pars["ψA0"]

    pars_ = copy.deepcopy(pars)
    pars_["Ωp_max"] = Ωp_max
    pars_["Ωs_max"] = Ωs_max
    # weighted average for correlated noise
    if noise == 1:
        Xrange = np.linspace(xrange[0], xrange[-1], num_trajectories)
        for X in Xrange:
            H_ = get_H_CN(η, X, Omega_p, Omega_s, pars_)
            res = mesolve(H_, ψA0, t, args=pars_)
            pops1 = expect(rr, res.states[-1])
            w += pdf(X)
            weighted_sum += pops1 * pdf(X)

        weighted_avg = weighted_sum / w

    # weighted average for not correlated noise.
    # In this case pdf has to be 2D.
    else:
        X1range = np.linspace(xrange[0], xrange[-1], num_trajectories)
        X2range = np.linspace(xrange[0], xrange[-1], num_trajectories)
        for X1 in X1range:
            for X2 in X2range:
                H_ = get_H_UCN(X1, X2, η, Omega_p, Omega_s, pars_)
                res2 = mesolve(H_, ψA0, t, args=pars_)
                pops2 = expect(rr, res2.states[-1])
                w += pdf(X1, X2)
                weighted_sum += pops2 * pdf(X1, X2)

        weighted_avg = weighted_sum / w

    return weighted_avg


# This func calculates the efficiency only considering Markovian noise using master equation
def eff_Markovian(t, Omega_p, Omega_s, Ωp_max, Ωs_max, η, pars):
    """
    Calculates the efficiency (population of state |2> at final time 't[-1]') in the quantum system
    influenced by Markovian noise using the master equation. The function is specifically tailored to
    work with quantum systems where the noise can be modeled as a Markovian process.

     Returns:
        float: Efficiency of the population |2> at the final time 't[-1]', calculated under Markovian assumptions.
    """

    N = pars["N"]
    ψA0 = pars["ψA0"]
    γ = pars["γ"]

    pars_ = copy.deepcopy(pars)
    pars_["Ωp_max"] = Ωp_max
    pars_["Ωs_max"] = Ωs_max
    if γ >= 0.0:
        c_ops = np.sqrt(γ) * (e * e.dag() + η * r * r.dag())
        H_ = get_H_nonQS_ME(η, Omega_p, Omega_s, pars_)
        res3 = mesolve(H_, ψA0, t, c_ops, args=pars_)
        pops3 = expect(rr, res3.states[-1])
    else:
        print("Error! γ must be positive!")

    return pops3


# If noise = 1(by default) this func calculates the population of state |2> at finat time tf (efficiency),
# by averaging over finite number of projective measurements considering correlated noise
# If noise = 2, the same is done considering not correlated (uncorrelated) noise
# If noise = any other number, the same is done considering non quasistatic noise
def eff_finite_measurements(
    t,
    t_noise,
    pdf,
    Omega_p,
    Omega_s,
    Ωp_max,
    Ωs_max,
    num_trajectories,
    η,
    pars,
    noise=1,
):
    """
    Calculates the efficiency (population of state |2> at final time 't[-1]') by averaging over a finite number of
    projective measurements. The function can handle different types of noise: correlated, uncorrelated, and
    non-quasistatic, based on the 'noise' parameter.

    Returns:
        tuple:
            - float: The average efficiency of the population in state |2> at time 't[-1]'.
            - list: List of binary outcomes (1 or 0) chosen with probabilities (efficiency, 1-efficiency),
                    for each finite projective measurement.

    Note:
        The function utilizes three distinct noise handling methods based on the value of 'noise' parameter to model the
        different physical situations. Each method has different implications for the dynamics and measurement of the system.
    """

    N = pars["N"]
    xrange = pars["xrange"]
    ψA0 = pars["ψA0"]

    pars_ = copy.deepcopy(pars)
    pars_["Ωp_max"] = Ωp_max
    pars_["Ωs_max"] = Ωs_max

    pops = []
    Hamiltonians2 = []
    S = []
    # experimenta avg for correlated noise
    if noise == 1:
        X_list = rejection_sampling(pdf, xrange, num_trajectories)
        for X in X_list:
            # Returns a list of Hamiltonians with different realizations of quasistatic noise for all the trajectories
            H_ = get_H_CN(η, X, Omega_p, Omega_s, pars_)
            res1 = mesolve(H_, ψA0, t, args=pars_)
            pops1 = expect(rr, res1.states[-1])
            pops.append(pops1)

        for eff in pops:
            s = np.random.choice((1, 0), p=[eff, 1 - eff])
            S.append(s)

        expt_avg = np.sum(S) / num_trajectories

    # experimental avg for not correlated noise
    elif noise == 2:
        X1_list = rejection_sampling(pdf, xrange, num_trajectories)
        X2_list = rejection_sampling(pdf, xrange, num_trajectories)
        for X1, X2 in zip(X1_list, X2_list):
            H_ = get_H_UCN(η, X1, X2, Omega_p, Omega_s, pars_)
            res2 = mesolve(H_, ψA0, t, args=pars_)
            pops2 = expect(rr, res2.states[-1])
            pops.append(pops2)

        for eff in pops:
            s = np.random.choice((1, 0), p=[eff, 1 - eff])
            S.append(s)

        expt_avg = np.sum(S) / num_trajectories

    # experimental avg for non qasistatic noise,
    else:
        for trajectory in range(num_trajectories):
            x = np.array(rejection_sampling(pdf, xrange, N))
            H_ = get_H_nonQS(t_noise, η, x, Omega_p, Omega_s, pars_)
            res3 = mesolve(H_, ψA0, t, args=pars_)
            pops3 = expect(rr, res3.states[-1])
            pops.append(pops3)

        for eff in pops:
            s = np.random.choice((1, 0), p=[eff, 1 - eff])
            S.append(s)

        expt_avg = np.sum(S) / num_trajectories

    return expt_avg, S


# (Calculating data with perfect measurement/Weigted average method)
# The next 4 functions generates the data for ML for the 4/5 noise respectively and
# return efficiencies calculated under 3 pulse conditions, the corresponding label and correlation parameter.
def get_data_CN(
    t,
    pdf,
    Omega_p,
    Omega_s,
    Ωp_max,
    Ωs_max,
    ηrange,
    num_trajectories,
    num_samples,
    pars,
):

    efficiency_correlated = []
    y_correlated = []
    eta_CN = []
    for i in range(num_samples):

        # efficiencies for correlated quasistatic noise
        eff1_CN = eff_nonMarkovian(
            t,
            pdf,
            Omega_p,
            Omega_s,
            Ωp_max[0],
            Ωs_max[0],
            num_trajectories,
            ηrange[i],
            pars,
            noise=1,
        )  # Ω_p = Ω_s
        eff2_CN = eff_nonMarkovian(
            t,
            pdf,
            Omega_p,
            Omega_s,
            Ωp_max[1],
            Ωs_max[1],
            num_trajectories,
            ηrange[i],
            pars,
            noise=1,
        )  # Ω_p > Ω_s
        eff3_CN = eff_nonMarkovian(
            t,
            pdf,
            Omega_p,
            Omega_s,
            Ωp_max[2],
            Ωs_max[2],
            num_trajectories,
            ηrange[i],
            pars,
            noise=1,
        )  # Ω_p < Ω_s

        eta_CN.append(ηrange[i])
        efficiency_correlated.append([eff1_CN, eff2_CN, eff3_CN])
        y_correlated.append(1)

    return efficiency_correlated, y_correlated, eta_CN


def get_data_ACN(
    t,
    pdf,
    Omega_p,
    Omega_s,
    Ωp_max,
    Ωs_max,
    ηrange,
    num_trajectories,
    num_samples,
    pars,
):

    efficiency_anti_correlated = []
    y_anti_correlated = []
    eta_ACN = []
    for i in range(num_samples):

        # efficiencies for correlated quasistatic noise
        eff1_ACN = eff_nonMarkovian(
            t,
            pdf,
            Omega_p,
            Omega_s,
            Ωp_max[0],
            Ωs_max[0],
            num_trajectories,
            ηrange[i],
            pars,
            noise=1,
        )  # Ω_p = Ω_s
        eff2_ACN = eff_nonMarkovian(
            t,
            pdf,
            Omega_p,
            Omega_s,
            Ωp_max[1],
            Ωs_max[1],
            num_trajectories,
            ηrange[i],
            pars,
            noise=1,
        )  # Ω_p > Ω_s
        eff3_ACN = eff_nonMarkovian(
            t,
            pdf,
            Omega_p,
            Omega_s,
            Ωp_max[2],
            Ωs_max[2],
            num_trajectories,
            ηrange[i],
            pars,
            noise=1,
        )  # Ω_p < Ω_s

        eta_ACN.append(ηrange[i])
        efficiency_anti_correlated.append([eff1_ACN, eff2_ACN, eff3_ACN])
        y_anti_correlated.append(2)

    return efficiency_anti_correlated, y_anti_correlated, eta_ACN


def get_data_UCN(
    t,
    pdf_2d,
    Omega_p,
    Omega_s,
    Ωp_max,
    Ωs_max,
    ηrange,
    num_trajectories,
    num_samples,
    pars,
):

    efficiency_uncorrelated = []
    y_uncorrelated = []
    eta_UCN = []
    for i in range(num_samples):

        # efficiencies for uncorrelated quasistatic noise
        eff1_UCN = eff_nonMarkovian(
            t,
            pdf_2d,
            Omega_p,
            Omega_s,
            Ωp_max[0],
            Ωs_max[0],
            num_trajectories,
            ηrange[i],
            pars,
            noise=2,
        )  # Ω_p = Ω_s
        eff2_UCN = eff_nonMarkovian(
            t,
            pdf_2d,
            Omega_p,
            Omega_s,
            Ωp_max[1],
            Ωs_max[1],
            num_trajectories,
            ηrange[i],
            pars,
            noise=2,
        )  # Ω_p > Ω_s
        eff3_UCN = eff_nonMarkovian(
            t,
            pdf_2d,
            Omega_p,
            Omega_s,
            Ωp_max[2],
            Ωs_max[2],
            num_trajectories,
            ηrange[i],
            pars,
            noise=2,
        )  # Ω_p < Ω_s

        eta_UCN.append(ηrange[i])
        efficiency_uncorrelated.append([eff1_UCN, eff2_UCN, eff3_UCN])
        y_uncorrelated.append(3)

    return efficiency_uncorrelated, y_uncorrelated, eta_UCN


def get_data_NQS(
    t, Omega_p, Omega_s, Ωp_max, Ωs_max, ηrange, num_samples, pars, separated=False
):

    efficiency_non_quasistaic = []
    y_non_quasistaic = []
    eta_NQS = []
    for i in range(num_samples):

        # efficiency for non quasistatic noise
        eff1_NQS = eff_Markovian(
            t, Omega_p, Omega_s, Ωp_max[0], Ωs_max[0], ηrange[i], pars
        )  # Ω_p = Ω_s
        eff2_NQS = eff_Markovian(
            t, Omega_p, Omega_s, Ωp_max[1], Ωs_max[1], ηrange[i], pars
        )  # Ω_p > Ω_s
        eff3_NQS = eff_Markovian(
            t, Omega_p, Omega_s, Ωp_max[2], Ωs_max[2], ηrange[i], pars
        )  # Ω_p < Ω_s

        eta_NQS.append(ηrange[i])
        efficiency_non_quasistaic.append([eff1_NQS, eff2_NQS, eff3_NQS])
        if not separated:
            y_non_quasistaic.append(4)
        else:
            if ηrange[i] > 0:
                y_non_quasistaic.append(4)
            if ηrange[i] < 0:
                y_non_quasistaic.append(5)

    return efficiency_non_quasistaic, y_non_quasistaic, eta_NQS


# (Calculating data with finite number of measurements)
def get_data_CN_finite(
    t,
    t_noise,
    pdf,
    Omega_p,
    Omega_s,
    Ωp_max,
    Ωs_max,
    ηrange,
    num_trajectories,
    num_samples,
    pars,
):

    efficiency_correlated = []
    y_correlated = []
    eta_CN = []
    S = []
    for i in range(num_samples):

        # efficiencies for correlated quasistatic noise
        eff1_CN, S1 = eff_finite_measurements(
            t,
            t_noise,
            pdf,
            Omega_p,
            Omega_s,
            Ωp_max[0],
            Ωs_max[0],
            num_trajectories,
            ηrange[i],
            pars,
            noise=1,
        )  # Ω_p = Ω_s
        eff2_CN, S2 = eff_finite_measurements(
            t,
            t_noise,
            pdf,
            Omega_p,
            Omega_s,
            Ωp_max[1],
            Ωs_max[1],
            num_trajectories,
            ηrange[i],
            pars,
            noise=1,
        )  # Ω_p > Ω_s
        eff3_CN, S3 = eff_finite_measurements(
            t,
            t_noise,
            pdf,
            Omega_p,
            Omega_s,
            Ωp_max[2],
            Ωs_max[2],
            num_trajectories,
            ηrange[i],
            pars,
            noise=1,
        )  # Ω_p < Ω_s

        eta_CN.append(ηrange[i])
        efficiency_correlated.append([eff1_CN, eff2_CN, eff3_CN])
        y_correlated.append(1)
        S.append([S1, S2, S3])

    return efficiency_correlated, y_correlated, eta_CN, S


def get_data_ACN_finite(
    t,
    t_noise,
    pdf,
    Omega_p,
    Omega_s,
    Ωp_max,
    Ωs_max,
    ηrange,
    num_trajectories,
    num_samples,
    pars,
):

    efficiency_anti_correlated = []
    y_anti_correlated = []
    eta_ACN = []
    S = []
    for i in range(num_samples):

        # efficiencies for correlated quasistatic noise
        eff1_ACN, S1 = eff_finite_measurements(
            t,
            t_noise,
            pdf,
            Omega_p,
            Omega_s,
            Ωp_max[0],
            Ωs_max[0],
            num_trajectories,
            ηrange[i],
            pars,
            noise=1,
        )  # Ω_p = Ω_s
        eff2_ACN, S2 = eff_finite_measurements(
            t,
            t_noise,
            pdf,
            Omega_p,
            Omega_s,
            Ωp_max[1],
            Ωs_max[1],
            num_trajectories,
            ηrange[i],
            pars,
            noise=1,
        )  # Ω_p > Ω_s
        eff3_ACN, S3 = eff_finite_measurements(
            t,
            t_noise,
            pdf,
            Omega_p,
            Omega_s,
            Ωp_max[2],
            Ωs_max[2],
            num_trajectories,
            ηrange[i],
            pars,
            noise=1,
        )  # Ω_p < Ω_s

        eta_ACN.append(ηrange[i])
        efficiency_anti_correlated.append([eff1_ACN, eff2_ACN, eff3_ACN])
        y_anti_correlated.append(2)
        S.append([S1, S2, S3])

    return efficiency_anti_correlated, y_anti_correlated, eta_ACN, S


def get_data_UCN_finite(
    t,
    t_noise,
    pdf,
    Omega_p,
    Omega_s,
    Ωp_max,
    Ωs_max,
    ηrange,
    num_trajectories,
    num_samples,
    pars,
):

    efficiency_uncorrelated = []
    y_uncorrelated = []
    eta_UCN = []
    S = []
    for i in range(num_samples):

        # efficiencies for uncorrelated quasistatic noise
        eff1_UCN, S1 = eff_finite_measurements(
            t,
            t_noise,
            pdf,
            Omega_p,
            Omega_s,
            Ωp_max[0],
            Ωs_max[0],
            num_trajectories,
            ηrange[i],
            pars,
            noise=2,
        )  # Ω_p = Ω_s
        eff2_UCN, S2 = eff_finite_measurements(
            t,
            t_noise,
            pdf,
            Omega_p,
            Omega_s,
            Ωp_max[1],
            Ωs_max[1],
            num_trajectories,
            ηrange[i],
            pars,
            noise=2,
        )  # Ω_p > Ω_s
        eff3_UCN, S3 = eff_finite_measurements(
            t,
            t_noise,
            pdf,
            Omega_p,
            Omega_s,
            Ωp_max[2],
            Ωs_max[2],
            num_trajectories,
            ηrange[i],
            pars,
            noise=2,
        )  # Ω_p < Ω_s

        eta_UCN.append(ηrange[i])
        efficiency_uncorrelated.append([eff1_UCN, eff2_UCN, eff3_UCN])
        y_uncorrelated.append(3)
        S.append([S1, S2, S3])

    return efficiency_uncorrelated, y_uncorrelated, eta_UCN, S


def get_data_NQS_finite(
    t,
    t_noise,
    pdf,
    Omega_p,
    Omega_s,
    Ωp_max,
    Ωs_max,
    ηrange,
    num_trajectories,
    num_samples,
    pars,
):

    efficiency_non_quasistaic = []
    y_non_quasistaic = []
    eta_NQS = []
    S = []
    for i in range(num_samples):

        # efficiency for non quasistatic noise
        eff1_NQS, S1 = eff_finite_measurements(
            t,
            t_noise,
            pdf,
            Omega_p,
            Omega_s,
            Ωp_max[0],
            Ωs_max[0],
            num_trajectories,
            ηrange[i],
            pars,
            noise=3,
        )  # Ω_p = Ω_s
        eff2_NQS, S2 = eff_finite_measurements(
            t,
            t_noise,
            pdf,
            Omega_p,
            Omega_s,
            Ωp_max[1],
            Ωs_max[1],
            num_trajectories,
            ηrange[i],
            pars,
            noise=3,
        )  # Ω_p > Ω_s
        eff3_NQS, S3 = eff_finite_measurements(
            t,
            t_noise,
            pdf,
            Omega_p,
            Omega_s,
            Ωp_max[2],
            Ωs_max[2],
            num_trajectories,
            ηrange[i],
            pars,
            noise=3,
        )  # Ω_p < Ω_s

        eta_NQS.append(ηrange[i])
        efficiency_non_quasistaic.append([eff1_NQS, eff2_NQS, eff3_NQS])
        y_non_quasistaic.append(4)
        S.append([S1, S2, S3])

    return efficiency_non_quasistaic, y_non_quasistaic, eta_NQS, S


# Functions for ML


def make_balanced_data(data_list):
    """
    Split the input data into balanced training, validation, and test sets.

    The function first separates each dataset in data_list into features and labels.
    Then, it divides each dataset into a test set (20% of the data) and the remaining
    data into training and validation sets (80% of the data, with 20% of it that is used for
    validation). This process ensures balanced distribution of data across sets.

    Returns:
    - X_test (array): Features of the test set.
    - Y_test (array): Labels of the test set.
    - X_val (array): Features of the validation set.
    - Y_val (array): Labels of the validation set.
    - X_train (array): Features of the training set.
    - Y_train (array): Labels of the training set.
    """

    X_test = []
    Y_test = []
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []

    for data_ in data_list:
        X = np.array(data_[:, :-1])
        Y = np.array(data_[:, -1]) - 1
        # -1 is to change the levels from 0 to 4 instead of 1 to 5

        # Separate the test data
        x, x_test, y, y_test = train_test_split(X, Y, test_size=1 / 5)

        # Split the remaining data to train and validation
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=1 / 4)

        X_test += [x_test]
        Y_test += [y_test]
        X_val += [x_val]
        Y_val += [y_val]
        X_train += [x_train]
        Y_train += [y_train]

    X_test = np.vstack(X_test)
    Y_test = np.hstack(Y_test)
    X_val = np.vstack(X_val)
    Y_val = np.hstack(Y_val)
    X_train = np.vstack(X_train)
    Y_train = np.hstack(Y_train)

    return X_test, Y_test, X_val, Y_val, X_train, Y_train


def create_model(n):
    Net = Sequential()  # creating a neural network!

    Net.add(
        Dense(128, input_shape=(3,), activation=tf.keras.layers.LeakyReLU(alpha=0.01))
    )
    Net.add(
        Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=0.01))
    )  # third hidden layer: 100 neurons
    Net.add(Dense(n, activation="softmax"))  # output layer: 4 or 5 neuron

    # Compile network: (randomly initialize weights, choose advanced optimizer, set up everything!)
    Net.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer="adam",
        metrics=["accuracy"],
    )  # adam is adaptive and works better than normal gradient descent

    print(Net.summary())
    return Net


def create_fit_model(data_noises, verbose=1):
    np.random.seed(1357)
    X_test, Y_test, X_val, Y_val, X_train, Y_train = make_balanced_data(data_noises)

    # Define network:
    Net = create_model(len(data_noises))

    # training of the model
    # accuracy before training
    print("Initial accuracy:")
    train_loss0_MLP, train_acc0_MLP = Net.evaluate(X_train, Y_train)
    val_loss0_MLP, val_acc0_MLP = Net.evaluate(X_val, Y_val)

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=5, restore_best_weights=True
    )

    print("training...")
    history_MLP = Net.fit(
        X_train,
        Y_train,
        epochs=2000,
        validation_data=(X_val, Y_val),
        batch_size=40,
        callbacks=[stop_early],
        verbose=verbose,
    )

    # The model weights (that are considered the best) are loaded into the model.
    Y_predict_MLP = Net.evaluate(X_test, Y_test)
    print(f"loss and accuracy on test set: {Y_predict_MLP}")

    return (
        train_loss0_MLP,
        train_acc0_MLP,
        val_loss0_MLP,
        val_acc0_MLP,
        history_MLP,
        Net,
        X_test,
        Y_test,
    )


def training_finite(Net, S_CN, S_ACN, S_UCN, S_NQS, num_trajectories, verbose=0):
    rng = np.random.default_rng()
    # selecting randomly the number of trajectories
    res_CN_random = rng.choice(
        rng.permuted(S_CN, axis=2), num_trajectories, replace=False, axis=2
    )
    res_ACN_random = rng.choice(
        rng.permuted(S_ACN, axis=2), num_trajectories, replace=False, axis=2
    )
    res_UCN_random = rng.choice(
        rng.permuted(S_UCN, axis=2), num_trajectories, replace=False, axis=2
    )
    res_NQS_random = rng.choice(
        rng.permuted(S_NQS, axis=2), num_trajectories, replace=False, axis=2
    )

    # Now summing up the 0s and 1s in each list and diving by total num of trajs which will give avg efficiency
    eff_correlated = np.sum(res_CN_random, axis=2) / num_trajectories
    eff_anti_correlated = np.sum(res_ACN_random, axis=2) / num_trajectories
    eff_uncorrelated = np.sum(res_UCN_random, axis=2) / num_trajectories
    eff_non_quasistaic = np.sum(res_NQS_random, axis=2) / num_trajectories

    # joining the efficiencies with corresponding labels
    labeled_data1 = np.column_stack(
        (eff_correlated, np.ones(len(eff_correlated), dtype=int))
    )  # labeled data for quasitatic, correlated noise
    labeled_data2 = np.column_stack(
        (eff_anti_correlated, 2 * np.ones(len(eff_correlated), dtype=int))
    )  # labeled data for quasitatic, anti correlated noise
    labeled_data3 = np.column_stack(
        (eff_uncorrelated, 3 * np.ones(len(eff_correlated), dtype=int))
    )  # labeled data for quasitatic, not correlated noise noise
    labeled_data4 = np.column_stack(
        (eff_non_quasistaic, 4 * np.ones(len(eff_correlated), dtype=int))
    )  # labeled data for correlated but non quasitatic noise

    data_4noises = [labeled_data1, labeled_data2, labeled_data3, labeled_data4]

    # Creating balanced dataset
    X_test, Y_test, X_val, Y_val, X_train, Y_train = make_balanced_data(data_4noises)

    # training of the model named 'Net'
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=5, restore_best_weights=True
    )

    history_MLP = Net.fit(
        X_train,
        Y_train,
        epochs=2000,
        validation_data=(X_val, Y_val),
        batch_size=40,
        verbose=verbose,
        callbacks=[stop_early],
    )

    # testing accuracy
    Y_predict_model = Net.evaluate(X_test, Y_test)

    return Y_predict_model[-1]
