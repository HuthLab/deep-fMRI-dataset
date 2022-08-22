def run_loo_em(loo_story):
	Rstories_loo = np.sort(list(set(Rstories) - set([loo_story])))
	Pstories_loo = np.sort(list(set(Pstories) | set([loo_story])))
	save_location_loo = os.path.join(save_location, 'loo', loo_story)
	args = [subject, save_location_loo, Rstories_loo, Pstories_loo, 
			downsampled_feat, zresp, trim, extra_trim, ndelays, nboots, 
			chunklen, nchunks, single_alpha, use_corr, singcutoff, False]
	pred_loo, _ = run_encoding_model(*args)
	return pred_loo[loo_story]

	if loo:
		# Run leave-one-out cross validation
		print('L-O-O ENCODING MODEL')
		from multiprocessing.pool import ThreadPool
		start_time = time.time()
		pool = ThreadPool(processes=2)
		pred = pool.map(lambda i: run_loo_em(Rstories[i]), range(len(Rstories)))
		pool.close()
		pred = np.vstack(pred)
		zPresp = np.vstack([zresp[loo_story] for loo_story in Rstories])
		end_time = time.time()
		print((end_time - start_time) / 60)

		print('zPresp:', zPresp.shape)
		print('pred:', pred.shape)

		# Run significance test
		print('L-O-O SIGNIFICANCE TEST')
		run_permutation_test(save_location, zPresp, pred, blocklen, nperms)
		good_vox = cci.download_raw_array(os.path.join(save_location, 'sig_tests', 'good_voxels'))
		print('#L-O-O Significant voxels: %d/%d' % (sum(good_vox), good_vox.shape[0]))


##############################################################################
############################ Significance Testing ############################
##############################################################################

def get_permuted_corrs(perm, true, pred, blocklen):
	nblocks = int(true.shape[0] / blocklen)
	true = true[:blocklen*nblocks]
	block_index = np.random.choice(range(nblocks), nblocks)
	index = []
	for i in block_index:
		start, end = i*blocklen, (i+1)*blocklen
		index.extend(range(start, end))
	pred_perm = pred[index]
	nvox = true.shape[1]
	corrs = np.nan_to_num(mcorr(true, pred_perm))
	return perm, corrs

def permutation_test(true, pred, blocklen, nperms):
	import time
	from multiprocessing.pool import ThreadPool
	start_time = time.time()
	pool = ThreadPool(processes=10)
	perm_rsqs = pool.map(lambda perm: get_permuted_corrs(perm, true, pred, blocklen), range(nperms))
	pool.close()
	end_time = time.time()
	print((end_time - start_time) / 60)
	perm_rsqs = np.array(perm_rsqs)
	assert np.all(perm_rsqs[:, 0] == range(nperms)), 'FUCK!'
	perm_rsqs = np.vstack(perm_rsqs[:, 1]).astype(np.float32)
	real_rsqs = np.nan_to_num(mcorr(true, pred))
	pvals = (real_rsqs <= perm_rsqs).mean(0)
	return np.array(pvals), perm_rsqs, real_rsqs

def run_permutation_test(save_location, zPresp, pred, blocklen, nperms,
						 mode='', thres=0.001):
	save_location = os.path.join(save_location, 'sig_tests')
	if cci.exists_object(os.path.join(save_location, '%spN_thres'%mode)):
		print('Significance test done already!')
		return

	assert zPresp.shape == pred.shape, print(zPresp.shape, pred.shape)

	start_time = time.time()
	ntr, nvox = zPresp.shape
	if ntr > 5000:
		partlen = int(nvox/5)
	elif ntr > 1000:
		partlen = int(nvox/3)
	else:
		partlen = nvox
	pvals, perm_rsqs, real_rsqs = [[] for _ in range(3)]

	for start in range(0, nvox, partlen):
		print(start, start+partlen)
		pv, pr, rs = permutation_test(zPresp[:, start:start+partlen], pred[:, start:start+partlen],
									  blocklen, nperms)
		pvals.append(pv)
		perm_rsqs.append(pr)
		real_rsqs.append(rs)
	pvals, perm_rsqs, real_rsqs = np.hstack(pvals), np.hstack(perm_rsqs), np.hstack(real_rsqs)

	assert pvals.shape[0] == nvox, (pvals.shape[0], nvox)
	assert perm_rsqs.shape[0] == nperms, (perm_rsqs.shape[0], nperms)
	assert perm_rsqs.shape[1] == nvox, (perm_rsqs.shape[1], nvox)
	assert real_rsqs.shape[0] == nvox, (real_rsqs.shape[0], nvox)

	cci.upload_raw_array(os.path.join(save_location, '%spvals'%mode), pvals)
	cci.upload_raw_array(os.path.join(save_location, '%sperm_rsqs'%mode), perm_rsqs)
	cci.upload_raw_array(os.path.join(save_location, '%sreal_rsqs'%mode), real_rsqs)
	print((time.time() - start_time)/60)
	
	pID, pN = fdr_correct(pvals, thres)
	cci.upload_raw_array(os.path.join(save_location, '%sgood_voxels'%mode), (pvals <= pN))
	cci.upload_raw_array(os.path.join(save_location, '%spN_thres'%mode), np.array([pN, thres], dtype=np.float32))
	return

