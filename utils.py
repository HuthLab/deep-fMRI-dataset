import numpy as np

from ridge_utils.stimulus_utils import load_textgrids, load_simulated_trfiles
from ridge_utils.dsutils import make_word_ds
from ridge_utils.npp import zscore


def get_story_wordseqs(stories):
    grids = load_textgrids(stories)
    with open('respdict.json', 'r') as f:
        respdict = json.loads(f)
    trfiles = load_simulated_trfiles(respdict)
    wordseqs = make_word_ds(grids, trfiles)
    return wordseqs

def make_delayed(stim, delays, circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays] 
    (in samples).
    
    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt,ndim = stim.shape
    dstims = []
    for di,d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d<0: ## negative delay
            dstim[:d,:] = stim[-d:,:]
            if circpad:
                dstim[d:,:] = stim[:-d,:]
        elif d>0:
            dstim[d:,:] = stim[:-d,:]
            if circpad:
                dstim[:d,:] = stim[-d:,:]
        else: ## d==0
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)

def apply_zscore_and_hrf(stories, downsampled_feat, trim, ndelays):
    """Get (z-scored and delayed) stimulus for train and test stories.

    Args:
        stories: List of stimuli stories.

    Variables:
        downsampled_feat (dict): Downsampled feature vectors for all stories.
        trim: Trim downsampled stimulus matrix.
        delays: List of delays for Finite Impulse Response (FIR) model.

    Returns:
        delstim: <float32>[TRs, features * ndelays]
    """
    stim = [zscore(downsampled_feat[s][5+trim:-trim]) for s in stories]
    stim = np.vstack(stim)
    delays = range(1, ndelays+1)
    delstim = make_delayed(stim, delays)
    return delstim

def get_response(stories, subject):
    """Get the subject's fMRI response for stories."""
    base = 'derivative/preprocessed_data/%s' % subject
    resp = []
    for story in stories:
        resp_path = os.path.join(base, '%s.hf5' % story)
        hf = h5py.File(resp_path, 'r')
        resp.extend(hf['resp'][:])
        hf.close()
    return np.array(resp)