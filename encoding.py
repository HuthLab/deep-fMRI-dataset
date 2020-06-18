import os
import sys
import numpy as np
import h5py
import argparse
import json

from feature_spaces import _FEATURE_CONFIG, get_feature_space
from utils import get_story_wordseqs, apply_zscore_and_hrf, get_response
from ridge_utils.ridge import bootstrap_ridge, ridge

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--subject', type=str, required=True)
	parser.add_argument('--feature', type=str, required=True)
	parser.add_argument('--last_session', type=int, default=5)
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
	assert feature in _FEATURE_CONFIG.keys(), 'Available feature spaces:' + fs
	assert last_session <= 5 and last_session >= 1, '1 <= last session <= 5'

	sessions = list(map(str, range(1, last_session+1)))
	with open('sess_to_story.json', 'r') as f:
		sess_to_story = json.loads(f) 
	allstories, Pstories = [], []
	for sess in sessions:
		stories, test_story = sess_to_story[sess][0], sess_to_story[sess][1]
		allstories.extend(stories)
		if test_story not in Pstories:
			Pstories.append(test_story)
	Rstories = list(set(allstories) - set(Pstories))

	save_location = 'results/%s/%s' % (feature, subject)
	if not os.path.exists(save_location):
		os.makedirs(save_location)

	wordseqs = get_story_wordseqs(allstories)

	downsampled_feat = get_feature_space(feature, wordseqs)
	print('Stimulus & Response parameters:')
	print('trim: %d, extra_trim: %d, ndelays: %d' % (trim, extra_trim, ndelays))

	# Delayed stimulus
	delRstim = apply_zscore_and_hrf(Rstories, downsampled_feat, trim, ndelays)
	print('delRstim: ', delRstim.shape)
	delPstim = apply_zscore_and_hrf(Pstories, downsampled_feat, trim, ndelays)
	print('delPstim: ', delPstim.shape)

	# Response
	zRresp = get_response(Rstories, subject, extra_trim)
	print('zRresp: ', zRresp.shape)
	zPresp = get_response(Pstories, subject, extra_trim)
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
