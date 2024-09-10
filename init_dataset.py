import fire
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
import gzip
from itertools import islice

import logging
import sys

from dp_ad_click_prediction import logistic
from dp_ad_click_prediction.feature_encodings import *

file_handler = logging.FileHandler(filename='init_dataset.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=handlers
)

label = "click"
allfeatures = ['hash_'+str(i) for i in range(0,19)]

logger = logging.getLogger(__file__)

def main(datapath='data'):
    filename = datapath + "/large_train.csv.gz"
    filename_smalltrain =  datapath + "/small_train.csv.gz"

    logger.info("init features")
    df = pd.read_csv(filename, dtype=np.int32, nrows=4_000_000) 
    logbase=2
    nbStd = 0.1
    nbStd = 0.3
    gaussianStd=17
    gaussianStd=1
    mappings = {}
    sigma = 0

    for f in allfeatures:
        mappings[f] = RawFeatureMapping.FromDF(f, df)
        size = mappings[f].Size
        if size > 100:
            df["d"]=1
            df_f = df[[f, "click", "d"]].groupby(f).sum().reset_index()
            df_f["click"] += np.random.normal(0, sigma, len( df_f ))
            df_f["d"] += np.random.normal(0, sigma, len( df_f ))  
            df_f.loc[df_f['d'] <1, 'd'] = 1
            df_f.loc[df_f['click'] <0, 'click'] = 0
            mappings[f] = RawFeatureMapping.BuildCtrBucketsFromAggDf(f, df_f, logbase=logbase, nbStd=nbStd, gaussianStd=gaussianStd)
            print(f, size, '->', mappings[f].Size ) 

    rawFeaturesSet = RawFeaturesSet(allfeatures, mappings )
    maxNbModalities= {f : 998 for f in allfeatures}
    maxNbModalities["default"] = 1_000_000 
    cfset = CrossFeaturesSet(rawFeaturesSet , "*&*",maxNbModalities=maxNbModalities  )

    with open(f'{datapath}/cfset.save', 'wb') as file:
        cfset.dump(file) 

    logger.info("init train data")
    xs = []
    ys = []
    for x,y in logistic.MyLogistic.batchReadFile(1_000_000, filename, cfset):
        xs.append(x)
        ys.append(y)
    with open(f'{datapath}/xys.dat', 'wb') as file:
        np.save(file, np.hstack(xs))
        np.save(file, np.hstack(ys))

    xs_test = []
    ys_test = []
    logger.info("init test data")
    for x,y in logistic.MyLogistic.batchReadFile(10_000_000, filename_smalltrain, cfset):
        xs_test.append(x)
        ys_test.append(y)
    with open(f'{datapath}/xys_test.dat', 'wb') as file:
        np.save(file, np.hstack(xs_test))
        np.save(file, np.hstack(ys_test))

if __name__ == '__main__':
    fire.Fire(main)