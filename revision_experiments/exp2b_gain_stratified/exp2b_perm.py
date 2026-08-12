"""Fast permutation runner for exp2b: caches the (fixed-warping) design matrix once as float32,
then each permutation only slices rows/trial_coef columns and refits."""
import os; os.environ["OMP_NUM_THREADS"]="1"
import sys, time, pickle, argparse
sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM"); sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/outputs")
os.chdir("/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, GLM
CACHE="/sessions/gifted-cool-babbage/mnt/outputs/exp2b_design.npz"
NT=500; PEN=5e-1

def load():
    d=np.load(CACHE, allow_pickle=True)
    return (d["X"], d["y"], int(d["ncf"]), int(d["v1s"]), int(d["v1e"]),
            d["v1_basis"], [np.asarray(a) for a in d["nopen"]])

X,y,ncf,v1s,v1e,v1_basis,nopen = load()
def integ(ks):
    rows=np.concatenate([np.arange(k*NT,(k+1)*NT) for k in ks])
    cols=np.concatenate([np.arange(ncf), ncf+np.asarray(ks)])
    res=GLM.poisson_regression(y[rows], np.ascontiguousarray(X[np.ix_(rows,cols)],dtype=np.float64),
                               L2_pen=PEN, no_penalty=nopen)
    return float(np.sum(v1_basis @ res.params[v1s:v1e]))

NH=None   # set in main() to the observed high-gain group size (= number of running trials)
def work(seed):
    rng=np.random.RandomState(seed); n=X.shape[0]//NT
    p=rng.permutation(n)
    return integ(p[:NH])-integ(p[NH:])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--split",default="rate"); ap.add_argument("--nperm",type=int,default=1000)
    ap.add_argument("--budget",type=float,default=36.0); ap.add_argument("--wave",type=int,default=2)
    a=ap.parse_args(); t0=time.time()
    OUT="/sessions/gifted-cool-babbage/mnt/outputs/exp2b_gain81_%s.pkl"%a.split
    st=pickle.load(open(OUT,"rb"))
    global NH; NH=int(st["n_hi"])
    import multiprocessing as mp
    ctx=mp.get_context("fork"); pool=ctx.Pool(a.wave)
    try:
        while len(st["perm"])<a.nperm and time.time()-t0<a.budget:
            seeds=[5000+len(st["perm"])+i for i in range(a.wave*4)]
            for d in pool.map(work, seeds):
                st["perm"].append(d)
            pickle.dump(st,open(OUT,"wb"))
    finally:
        pool.close(); pool.join()
    pickle.dump(st,open(OUT,"wb"))
    null=np.array(st["perm"]); pv=float(np.mean(np.abs(null)>=abs(st["D_obs"])))
    print("[%s] perms=%d  D_obs=%+.4f  null|D|=%.4f  p=%.4f  (%.0fs)"
          %(a.split,len(null),st["D_obs"],np.abs(null).mean(),pv,time.time()-t0),flush=True)
if __name__=="__main__": main()
