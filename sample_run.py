import sys
import os
import pandas as pd
import numpy as np
import fire
import logging

from dp_ad_click_prediction import logistic
from dp_ad_click_prediction.feature_encodings import *

logger = logging.getLogger(__file__)

def init_logging(filename: str):
    file_handler = logging.FileHandler(filename=filename)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]

    logging.basicConfig(
        level=logging.INFO, 
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=handlers
    )

def main(bs: int=1_000_000, lr: float = 0.001, 
         l2: float = 100, noise: float = 0, 
         epochs: int = 15, decay: float = 0, 
         datapath: str ='data', splits: list[int] = [], 
         model_output: str = None, model_seed: str = None):
    filename = (f"sample_run_bs_{bs}_lr_{lr}_l2_{l2}_noise_{noise}"
                f"_epoch_{epochs}_decay_{decay}_path_{datapath}_splits_{''.join([str(s) for s in splits])}.log")
    init_logging(filename)

    with open(f'{datapath}/cfset.save', 'rb') as file:
        cfset = CrossFeaturesSet.load(file)

    if splits:
        xs = []
        ys = []
        for i in splits:
            name = f"{datapath}/xys_{i}.dat"
            with open(name, 'rb') as file:
                logger.info(f"load file {name}")
                x = np.load(file)
                y = np.load(file)
                xs.append(x)
                ys.append(y)
        xs = np.hstack(xs)
        ys = np.hstack(ys)        
        nbSamples = xs.shape[1]
    else:
        with open(f"{datapath}/xys.dat", 'rb') as file:
            xs = np.load(file)
            ys = np.load(file)
            nbSamples = xs.shape[1]
    
    logger.info(f"total number of samples {nbSamples}")

    with open(f"{datapath}/xys_test.dat", 'rb') as file:
        xs_test = np.load(file)
        ys_test = np.load(file)

    logger.info(f"start training")
    l = logistic.MyLogistic(cfset, regulL2 = l2 / nbSamples, clicksCfs = "*&*", verbose = True, model_seed = model_seed)
    adam = logistic.Adam(lr, decay=decay)
    l.trainSGD(nbSamples, bs, [xs,ys], 1, nbPasses = epochs, adam = adam, noise_multiplier = noise,
               filename_test = [xs_test,ys_test], model_path = model_output)
    logger.info(f"training done")

if __name__ == '__main__':
    fire.Fire(main)