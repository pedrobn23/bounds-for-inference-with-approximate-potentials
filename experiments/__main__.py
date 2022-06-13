import dataclasses
import os
import numpy as np
import pandas as pd
import tqdm

from pgmpy.factors.discrete import CPD
from experiments import approximations, bounds, read
from typing import List

PATH = 'networks'
#NETWORKS = os.listdir(PATH)

NETWORKS = ['child.bif', 'alarm.bif',  'win95pts.bif', 'insurance.bif', 'hailfinder.bif', 'hepar2.bif', 'andes.bif', 'pigs.bif', 'link.bif', 'munin.bif', 'pathfinder.bif', 'barley.bif', 'mildew.bif', 'diabetes.bif']


@dataclasses.dataclass
class Result:
    variable: str
    net: str
    original_DKL_error: float
    original_L1_error: float
    M_error: float
    D_error: float

    @classmethod
    def from_dict(cls, dict_: dict):
        result = cls(0, 0, object, '', 0, 0)

        for field_ in dataclasses.fields(cls):
            try:
                setattr(result, field_.name, dict_[field_.name])
            except KeyError:
                pass

        result.__post_init__()
        return result

    def asdict(self):
        return dataclasses.asdict(self)

    def astuple(self):
        return dataclasses.astuple(self)

    def __str__(self):
        return (f'- Error results for variable {self.variable} in net {net}:\n'
                f'    * correct DKL: {self.original_DKL_error}\n'
                f'    * correct L1: {self.original_L1_error}\n'
                f'    * M bound: {self.M_error}\n'
                f'    * D bound: {self.D_error}.')


if __name__ == "__main__":
    results = []
    nets = read.nets(PATH, NETWORKS):
    for l1_error in [0.01, 0.05, 0.1]:
        print(f' ---- Approximating Error: {l1_error} ----')
        for net, cpds in nets:
            results = []
            print(f' ---- Approximating net: {net} ----')
            approx_cpds = [approximations.adjusted_noise(cpd) for cpd in cpds]
            print(' ---> done')
            
            for cpd, approx_cpd in zip(cpds, approx_cpds):
                print(f' -- Working on: {cpd.variable}')
                others = [other for other in cpds if other != cpd]

                res = Result(
                    variable=cpd.variable,
                    net=net[:-4],
                    original_DKL_error=bounds.kl_divergence(cpd, approx_cpd),
                    original_L1_error=bounds.l1_distance(cpd, approx_cpd),
                    M_error=bounds.M_error(cpd, approx_cpd, *others),
                    D_error=bounds.D_error(cpd, approx_cpd, *others),
                )
                results.append(res)
                print(' -> done')

            print(' -- Performing IO operations')
            df = pd.DataFrame.from_dict([result.asdict() for result in results])
            df.to_csv(f'results/{net[:-4]}.csv', index=False)
            print(' -> done')
        
