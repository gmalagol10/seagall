import warnings
warnings.filterwarnings("ignore")

import numpy as np

def intersection(l):
	return np.array(list(set.intersection(*map(set,list(l)))))

def flat_list(l):
	return np.array(list(set([item for sublist in l for item in sublist])))

def most_common(l):
	l=list(l)
	return max(set(l), key=l.count)
