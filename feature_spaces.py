import os
import sys
import numpy as np

from ridge_utils.interpdata import lanczosinterp2D
from ridge_utils.SemanticModel import SemanticModel
from ridge_utils.dsutils import make_semantic_model


_ENG1000_PATH = 'english1000sm.hf5'


def downsample_word_vectors(stories, word_vectors):
	"""Get Lanczos downsampled word_vectors for specified stories.

	Args:
		stories: List of stories to obtain vectors for.
		word_vectors: Dictionary of {story: <float32>[num_story_words, vector_size]}

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(stories)
	downsampled_semanticseqs = dict()
	for story in stories:
		downsampled_semanticseqs[story] = lanczosinterp2D(
			word_vectors[story], wordseqs[story].data_times, 
			wordseqs[story].tr_times, window=3)
	return downsampled_semanticseqs

########################################
########## WORD RATE Features ##########
########################################

def get_wordrate_vectors(wordseqs):
	"""Get wordrate vectors for specified stories.

	Args:
		wordseqs: (words, word_times, tr_times)

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	stories = list(wordseqs.keys())
	vectors = {}
	for story in stories:
		nwords = len(wordseqs[story].data)
		vectors[story] = np.ones([nwords, 1])
	return downsample_word_vectors(stories, vectors)


######################################
########## ENG1000 Features ##########
######################################

def get_eng1000_vectors(stories):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		wordseqs: (words, word_times, tr_times)

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(_ENG1000_PATH)
	stories = list(wordseqs.keys())
	vectors = {}
	for story in stories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		vectors[story] = sm.data
	return downsample_word_vectors(stories, vectors)

############################################
########## Feature Space Creation ##########
############################################

_FEATURE_CONFIG = {
	'wordrate': get_wordrate_vectors,
	'eng1000': get_eng1000_vectors,
}

def get_feature_space(feature, *args):
	return _FEATURE_CONFIG[feature](*args)