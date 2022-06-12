import numpy as np
from pgmpy.factors.discrete import CPD
from scipy import stats, special


def log_values(cpd, approx_cpd):
    values = cpd.values
    approx_values = approx_cpd.values

    log_values = special.rel_entr(values.flatten(), approx_values.flatten())
    return log_values.reshape(values.shape)


def l1_distance(cpd, approx_cpd):
    return abs(cpd.values - approx_cpd.values).sum()


def kl_divergence(cpd, approx_cpd):
    return log_values(cpd, approx_cpd).sum()


def marg_and_max(values, variables, marginalized):
    # Marg
    axis = tuple(
        variables.index(var) for var in marginalized if var in variables)
    logvalues = values.sum(axis=axis)

    # Max
    return values.max()


def M_error(cpd, approx_cpd, *others):
    other_vars = {
        var for other in others for var in other.variables
        if var in cpd.variables
    }
    logvalues = log_values(cpd, approx_cpd)

    result = marg_and_max(logvalues, cpd.variables, other_vars)

    if (normalization_factor := cpd.values.min()) != 0:
        return result / normalization_factor
    else:
        return float('inf')


def poduct_M_bound(cpds, approx_cpds):

    def others(cpds, cpd):
        others = cpds[:]
        others.remove(cpd)

    return sum(
        M_error(cpd, approx_cpd, others(cpds, cpd))
        for cpd, approx_cpd in zip(cpds, approx_cds))


def relative_weight(weight_cpd, object_cpd):
    return marg_and_max(weight_cpd.values, weight_cpd.variables,
                        object_cpd.variables)


def D_error(cpd, approx_cpd, *others):
    l1_error = abs(cpd.values - approx_cpd.values).sum()
    D_KL = special.rel_entr(cpd.values.flatten(),
                            approx_cpd.values.flatten()).sum()

    return np.prod([relative_weight(other, cpd) for other in others
                   ]) * (l1_error + D_KL)


def poduct_D_bound(cpds, approx_cpds, goal_weight):
    result = sum(
        M_error(cpd, approx_cpd, others(cpds, cpd))
        for cpd, approx_cpd in zip(cpds, approx_cds))

    normalization_factor = goal_weight
    return result / normalization_factor
