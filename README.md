
# Private learning of a click prediction model inside a trusted server

For an overview of differentially private (DP) learning methods, we refer to [this PATCG presentation](https://github.com/patcg/meetings/files/14936682/PATCG_Boston_private_learning.pdf).

Criteo's ML challenge has already shown that learning on aggregated data can be performed with good results applying [global DP constraints](https://techblog.criteo.com/results-from-the-criteo-adkdd-2021-challenge-50abc9fa3a6). Performance mostly relies on those 2 assets:

- a small un-obfuscated dataset of display-level events;
- aggregated reports encompassing label proportions (i.e., average label information) associated to a fixed set of user features.

However, it seems still unclear in which form a small granular display-level training data set that shares the same user distribution can persist in a future world without third-party cookies. 
The best results obtained without a small display-level dataset are still significantly below the results of a logistic trained on granular data (see [table here](https://github.com/criteo-research/ad_click_prediction_from_aggregated_data#criteo-privacy-preserving-ml-competition--adkdd)).

In this repository, global DP learning on display-level data inside a trusted server is explored. Instead of applying local DP noise on raw user data, this method uses the full granular display-level dataset to directly learn the model and publish the DP noised model by using the [DP-SGD method](https://arxiv.org/abs/1607.00133).

An overview on how this method could be embedded inside a trusted server with TEE technology can be found [here]( https://techblog.criteo.com/pets-in-advertising-scenarios-for-trusted-execution-environments-9d0264c57325).

DP parameter `epsilon` is computed based the DP accountant approach, publicly available in [Google's differential privacy library](https://github.com/google/differential-privacy/tree/main/python/dp_accounting).

We use the full display-level data set published in [Criteo's Privacy Preserving ML Competition](https://competitions.codalab.org/competitions/31485)(90 mio lines, 2.5 GB). 

## Install

- Create venv with python3.9
```
python3.9 -m venv venv
. venv/bin/activate
pip install -e .
```
- install dp_accounting
https://github.com/google/differential-privacy/tree/main/python/dp_accounting

## Initialize data 

- get Criteo ML challenge data
```
./download_dataset.sh data
```

- init pandas structures
```
python init_dataset.py --datapath data
```

## Sample Run 

```
python sample_run.py --datapath data
```