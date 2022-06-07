import numpy as np
from pgmpy.factors.discrete import CPD



def noise(cpd: CPD.TabularCPD, l1_error = 0.01):
    """Create a new CPD altered by noise included.

    Naming noisy_cpd to the altered CPD, the following statements are true:
        - noisy_cpd.sum() == cpd.sum()
        - abs(noisy_cpd - cpd).sum() == l1_error
        - (0 < noisy_cpd).all()
        - (noisy_cpd < 1).all()
    """
    def normalize(data, l1_error):
        data = data - data.mean()       
        data = data / abs(data).sum()
        data = data *l1_error
        return data

    # Create noise values
    values = cpd.values
    noise = normalize(np.random.rand(*values.shape), l1_error)
    while (undervalues := 0 > (noise + values)).any() or (overvalues := (noise + values) > 1).any():
        noise = normalize(np.random.rand(*values.shape), l1_error)

    # Create new cpd altered by noise
    result = cpd.copy()
    result.values = values + noise
    return result
