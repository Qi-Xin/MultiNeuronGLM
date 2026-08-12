"""Lightweight stand-in for DataLoader.Allen_dataset, built from the NWB via h5py.
Exposes exactly the attributes GLM.PP_GLM / utils.pooling_pop need, so the REAL pop-GLM
(6-area model, condition-specific membership, group_id==0) can be fitted."""
import numpy as np, pandas as pd, pickle

NPZ="/sessions/gifted-cool-babbage/mnt/outputs/popglm_dataset_757216464.npz"
G="/sessions/gifted-cool-babbage/mnt/Ghigorah/MultiNeuronGLM"

class _Session:
    def __init__(self, units): self.units = units

class Dataset:
    def __init__(self, npz=NPZ):
        d=np.load(npz, allow_pickle=True)
        self.spike_train = d["spike_train"]              # (800, nneuron, 210)
        self.unit_ids    = d["unit_ids"]
        self.nt          = int(d["nt"])                  # 500
        self.npadding    = int(d["npadding"])            # 300
        self.fps         = int(d["fps"])                 # 1000
        self.ntrial      = self.spike_train.shape[2]     # 210
        self.time_line   = np.arange(0.0, 0.5, 1.0/self.fps)
        # stimulus_condition_id must be a numpy int dtype so .loc[i] yields np.int64
        self.presentation_table = pd.DataFrame(
            {"stimulus_condition_id": d["cond"].astype(np.int64)}, index=np.arange(self.ntrial))
        self.running_trial_index    = d["running"]
        self.stationary_trial_index = d["stationary"]
        self.all_trial_index        = np.full(self.ntrial, True)
        self.mean_speed = d["mean_speed"]
        # fallback table (not used when membership covers the probe)
        units = pd.DataFrame({"probe_description": d["unit_probe"],
                              "ecephys_structure_acronym": ["VISp"]*len(self.unit_ids)},
                             index=self.unit_ids)
        self._session = _Session(units)

def load_membership():
    memb = pickle.load(open(G+"/group_id_all_a_c/membership.pickle","rb"))
    cids = pickle.load(open(G+"/group_id_all_a_c/condition_ids.pickle","rb"))
    return memb, np.asarray(cids)     # array so pooling_pop's np.where works
