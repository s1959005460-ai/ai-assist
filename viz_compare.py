import numpy as np
from ripser import ripser
from persim import PersLandscapeApprox
from differentiable_topo import compute_landscape_approx

def compute_true_landscape(pc, maxdim=1, start=0.0, stop=10.0, num_steps=200):
    dgms = ripser(pc, maxdim=maxdim)['dgms']
    pla = PersLandscapeApprox(dgms, start=start, stop=stop, num_steps=num_steps)
    return pla.landscapes_

def compare_point_cloud(pc):
    approx = compute_landscape_approx(pc)
    true = compute_true_landscape(pc)
    l2s = []
    for i in range(min(len(approx), len(true))):
        a, b = np.array(approx[i]), np.array(true[i])
        L = min(len(a), len(b))
        l2s.append(np.linalg.norm(a[:L]-b[:L]))
    return l2s

if __name__=="__main__":
    pc = np.random.randn(200,2)
    print(compare_point_cloud(pc))
