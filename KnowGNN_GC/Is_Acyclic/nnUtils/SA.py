import numpy as np
import matplotlib.pyplot as plt
import math


def P_t_SA(Pt, PA, _T):
    Tmin = 1e-8  # minimum value of terperature
    updated_Pt = Pt
    p = 0.0
    if _T>=Tmin:
        if Pt - PA < 0:
            updated_Pt = PA
        else:
            # metropolis principle
            p = math.exp(-(Pt - PA) / _T)
            r = np.random.uniform(low=0, high=1)
            if r < p:
                updated_Pt = PA


    # updated_T = 0.98*_T  #降温函数，也可使用T=0.9T
    print(updated_Pt,_T,p)

    return updated_Pt



if __name__ == '__main__':
    T=0.01
    for i in range(20):
        print(i)
        P_t_SA(0.97, 0.96, T)
        T = 0.98*T