import numpy as np
from pgmpy.factors.discrete import CPD


def include_noise(cpd: CPD.TabularCPD):
    values = cpd.values
    noise = np.zeros(values.shape)
    return noise


if __name__ == '__main__':
    cpd = CPD.TabularCPD(
        'grade',
        3, [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
        evidence=['diff', 'intel'],
        evidence_card=[2, 3])
    print(cpd.values)
    print(include_noise(cpd))
