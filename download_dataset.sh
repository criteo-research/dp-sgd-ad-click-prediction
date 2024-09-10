#!/bin/bash

mkdir $1
pushd $1

mkdir tmp
pushd tmp

wget "http://go.criteo.net/criteo-ppml-challenge-adkdd21-dataset.zip"
unzip criteo-ppml-challenge-adkdd21-dataset.zip && rm criteo-ppml-challenge-adkdd21-dataset.zip
gunzip *.gz
paste -d , X_train.csv y_train.csv > ../small_train.csv
gzip ../small_train.csv
paste -d , X_test.csv.gz y_test.csv.gz > ../test.csv
gzip ../test.csv

popd

rm -rf tmp

wget "http://go.criteo.net/criteo-ppml-challenge-adkdd21-dataset-raw-granular-data.csv.gz"
mv criteo-ppml-challenge-adkdd21-dataset-raw-granular-data.csv.gz large_train.csv.gz

popd