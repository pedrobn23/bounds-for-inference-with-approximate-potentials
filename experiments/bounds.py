import numpy as np
from pgmpy.factors.discrete import CPD
from scipy import stats, special

import approximations

def log_values(cpd, approx_cpd):
    values = cpd.values
    approx_values = approx_cpd.values

    log_values = special.rel_entr(values.flatten(),approx_values.flatten())
    return log_values.reshape(values.shape)


def marg_and_max(values, variables, marginalized):
    axis = tuple(variables.index(var) for var in marginalized if var in variables)
    logvalues = logvalues.sum(axis=axis)
    return logvalues.max()
    

def M_error(cpd, approx_cpd, *others):
    other_vars = {var for other in others for var in other.variables if var in cpd.variables}
    logvalues = log_values(cpd, approx_cpd)

    result = marg_and_max(logvalues, cpd.variables, other_vars)
    normalization_factor = np.prod(cpd.cardinality[1:])     

    return result/normalization_factor

def poduct_M_bound(cpds, approx_cpds):
    def others(cpds, cpd):
        others = cpds[:]
        others.remove(cpd)
    
    return sum(M_error(cpd,
                       approx_cpd,
                       others(cpds, cpd)) for cpd, approx_cpd in zip(cpds, approx_cds))

def relative_weight(weight_cpd, object_cpd):
    return marg_and_max(weight_cpd.values, weight_cpd.variables, object_cpd.variables)

def D_error(cpd, approx_cpd, *others):
    l1_error = abs(cpd.values-approx_cpd.values).sum()
    D_KL =  special.rel_entr(cpd.values.flatten(),approx_cpd.values.flatten()).sum()

    return np.prod(relative_weight(other, cpd) for other in others) * (l1_error + D_KL)

def poduct_D_bound(cpds, approx_cpds):
    result = sum(M_error(cpd,
                         approx_cpd,
                         others(cpds, cpd)) for cpd, approx_cpd in zip(cpds, approx_cds))

    cardinalities = dict()
    for cpd in cpds:
        cardinalities.update(cpd.get_cardinality(cpd.variables))
    # acabar esto
    normalization_factor = np.prod(np.prod(cpd.cardinality[1:])for      
