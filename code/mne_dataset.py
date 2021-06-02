
# import scipy.optimize as sciop
import numpy as np
import mne
from mne.datasets import sample
from scipy.io import savemat

data_path = sample.data_path()
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
ave_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-shrunk-cov.fif'
subjects_dir = data_path + '/subjects'
condition = 'Right Auditory'

# Read noise covariance matrix
noise_cov = mne.read_cov(cov_fname)
# Handling average file
evoked = mne.read_evokeds(ave_fname, condition=condition, baseline=(None, 0))
evoked.crop(tmin=0.04, tmax=0.18)

evoked = evoked.pick_types(eeg=False, meg=True)
# Handling forward solution
forward = mne.read_forward_solution(fwd_fname)





#%%


def getGain( evoked, forward, noise_cov, loose=0.2, depth=0.8):
    from mne.inverse_sparse.mxne_inverse import \
        (_prepare_gain, is_fixed_orient,
         _reapply_source_weighting, _make_sparse_stc)

    all_ch_names = evoked.ch_names

    # Handle depth weighting and whitening (here is no weights)
    forward, gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
        forward, evoked.info, noise_cov, pca=False, depth=depth,
        loose=loose, weights=None, weights_min=None, rank=None)

    # Select channels of interest
    sel = [all_ch_names.index(name) for name in gain_info['ch_names']]
    M = evoked.data[sel]

    # Whiten data
    M = np.dot(whitener, M)
    G = gain
    lammax = np.sqrt(np.max(np.sum((G.T@M)**2,1)   ))
    
    return G,M,lammax


# loose, depth = 0.2, 0.8  # corresponds to loose orientation
loose, depth = 1., 0.  # corresponds to free orientation

G,M,lammax = getGain( evoked, forward, noise_cov, loose, depth)
lam = lammax/20


mdic = {"X": G, "Y": M}
savemat("GroupLassoDatasets/MNE_Data.mat", mdic)
