import numpy as np
from pgmpy.factors.discrete import CPD



def stocastich_noise(cpd: CPD.TabularCPD, l1_error = 0.01):
    """Create a new CPD altered by noise included.

    Naming noisy_cpd to the altered CPD, the following statements are true:
        - noisy_cpd.sum() == cpd.sum()
        - abs(noisy_cpd - cpd).sum() == l1_error
        - (0 < noisy_cpd).all()
        - (noisy_cpd < 1).all()
    
    Noise is produced randomly until it fits all the conditions, so this method may
    take infinitely long under specially hard instances.
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


def adjusted_noise(cpd: CPD.TabularCPD, l1_error = 0.01, filler = 0.001):
    """Create a new CPD altered by noise included.

    Naming noisy_cpd to the altered CPD, the following statements are true:
        - noisy_cpd.sum() is approximately cpd.sum()
        - abs(noisy_cpd - cpd).sum() is approximately l1_error
        - (0 < noisy_cpd).all()
        - (noisy_cpd < 1).all()

    This noise is generated randomly. After the generation, it repeats the next 
    steps until all conditions are achieved:
        - Normalize the noise.
        - impose a botton threshold of <filler> and an upper threshold of 
          1- <filler>.  

    """
    if l1_error > 1:
        raise ValueError(f"l1_error should be inferior to 1, got {l1_error}.")

    def centralize(data):
        return (data - 0.5) * 2

    def adjust(noise, data):
        noise[data + noise < 0] =  (-data + filler)[data + noise < 0]     # noise + data = filler 
        noise[data + noise > 1] =  ((1-data) - filler)[data + noise > 1]  # noise + data = 1-filler

        return noise

    def normalize(data, l1_error):
        data = data - data.mean()       
        data = data / abs(data).sum()
        data = data *l1_error
        return data

    values = cpd.values

    # Create noise values
    noise = centralize(np.random.rand(*values.shape)) 
    noise = normalize(noise, l1_error)
    noise = adjust(noise, values)
        
    # Create new cpd altered by noise
    result = cpd.copy()
    result.values = values + noise
    return result
