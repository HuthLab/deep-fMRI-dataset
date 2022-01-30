import os
import sys
import numpy as np
import h5py
import argparse
import json

from feature_spaces import _FEATURE_CONFIG, get_feature_space
from ridge_utils.npp import zscore
from ridge_utils.utils import make_delayed
from ridge_utils.ridge import bootstrap_ridge, ridge


def apply_zscore_and_hrf(stories, downsampled_feat, trim, ndelays):
	"""Get (z-scored and delayed) stimulus for train and test stories.
	The stimulus matrix is delayed (typically by 2,4,6,8 secs) to estimate the
	hemodynamic response function with a Finite Impulse Response model.

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


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--subject', type=str, required=True)
	parser.add_argument('--feature', type=str, required=True)
	parser.add_argument('--sessions', type=int, default=5)
	parser.add_argument('--trim', type=int, default=5)
	parser.add_argument('--ndelays', type=int, default=4)
	parser.add_argument('--nboots', type=int, default=50)
	parser.add_argument('--chunklen', type=int, default=40)
	parser.add_argument('--nchunks', type=int, default=125)
	parser.add_argument('--singcutoff', type=float, default=1e-10)
	parser.add_argument('-use_corr', action='store_true')
	parser.add_argument('-single_alpha', action='store_true')
	args = parser.parse_args()
	globals().update(args.__dict__)

	fs = ' '.join(_FEATURE_CONFIG.keys())
	assert feature in _FEATURE_CONFIG.keys(),
			'This feature sapcepace is not implemented! Available feature spaces:' + fs
	assert np.all((sessions <=5) & (sessions >= 1)), '1 <= last session <= 5'

	sessions = list(map(str, sessions))
	with open('sess_to_story.json', 'r') as f:
		sess_to_story = json.load(f) 
	train_stories, test_stories = [], []
	for sess in sessions:
		stories, tstory = sess_to_story[sess][0], sess_to_story[sess][1]
		train_stories.extend(stories)
		if tstory not in test_stories:
			test_stories.append(tstory)
	assert len(set(train_stories) & set(test_stories)) == 0, 'Train - Test overlap!'
	allstories = list(set(train_stories) | set(test_stories))

	save_location = os.path.join('results', feature, subject)
	print('Saving encoding model & results too:', save_location)
	if not os.path.exists(save_location):
		os.makedirs(save_location)

	downsampled_feat = get_feature_space(feature, allstories)
	print('Stimulus & Response parameters:')
	print('trim: %d, ndelays: %d' % (trim, ndelays))

	# Delayed stimulus
	delRstim = apply_zscore_and_hrf(train_stories, downsampled_feat, trim, ndelays)
	print('delRstim: ', delRstim.shape)
	delPstim = apply_zscore_and_hrf(test_stories, downsampled_feat, trim, ndelays)
	print('delPstim: ', delPstim.shape)

	# Response
	zRresp = get_response(train_stories, subject)
	print('zRresp: ', zRresp.shape)
	zPresp = get_response(test_stories, subject)
	print('zPresp: ', zPresp.shape)

	# Ridge
	alphas = np.logspace(1, 3, 10)

	print('Ridge parameters:')
	print('nboots: %d, chunklen: %d, nchunks: %d, single_alpha: %s, use_corr: %s' % (
		nboots, chunklen, nchunks, single_alpha, use_corr))

	wt, corrs, valphas, bscorrs, valinds = bootstrap_ridge(
		delRstim, zRresp, delPstim, zPresp, alphas, nboots, chunklen, 
		nchunks, singcutoff=singcutoff, single_alpha=single_alpha, 
		use_corr=use_corr)

	# Save regression results.
	np.savez('%s/weights' % save_location, wt)
	np.savez('%s/corrs' % save_location, corrs)
	np.savez('%s/valphas' % save_location, valphas)
	np.savez('%s/bscorrs' % save_location, bscorrs)
	np.savez('%s/valinds' % save_location, np.array(valinds))
	print('Total r2: %d' % sum(corrs * np.abs(corrs)))
