import pandas as pd
import numpy as np
from typing import Dict, List, Optional

import pickle
from dataclasses import dataclass


@dataclass
class RawFeatureMapping:
    """class representing one feature and its set of modalities."""

    Name: str
    _dicoModalityToId: Dict[int, int]

    def __post_init__(self):
        self._modalities_broadcast = None
        self._default = max(self._dicoModalityToId.values()) + 1
        self.Size = self._default + 1

    @staticmethod
    def FromDF(name: str, df):
        modalities = RawFeatureMapping.getModalities(df, name)
        dicoModalityToId = {m: i for i, m in enumerate(modalities)}
        return RawFeatureMapping(name, dicoModalityToId)

    # list of modalities observed in df
    @staticmethod
    def getModalities(df, name):
        if type(df) is pd.DataFrame:
            return RawFeatureMapping.getModalitiesPandas(df, name)
        else:
            return RawFeatureMapping.getModalitiesSpark(df, name)

    @staticmethod
    def getModalitiesPandas(df, name):
        return [int(x) for x in (sorted(set(df[name].values)))]

    @staticmethod
    def getModalitiesSpark(df, name):
        modalitieRows = df.select(name).drop_duplicates().orderBy(name).collect()
        modalities = list([row[name] for row in modalitieRows])
        return modalities

    def dump(self, handle):
        pickle.dump(self.Name, handle)
        pickle.dump(self._dicoModalityToId, handle)

    @staticmethod
    def load(handle, ss=None):
        name = pickle.load(handle)
        modalities = pickle.load(handle)
        return RawFeatureMapping(name, modalities)

    def spark_col(self):
        return F.col(self.Name).alias(self.Name)

    def setBroadCast(self, sql_ctx):
        if self._modalities_broadcast is not None:
            return
        self._modalities_broadcast = F.broadcast(
            sql_ctx.createDataFrame(
                [[int(newMod), int(oldMod)] for oldMod, newMod in _dicoModalityToId.items()], schema=("id", self.Name)
            )
        ).persist()

    # replace initial modalities of features by modality index
    def Map(self, df):
        if type(df) is pd.DataFrame:
            return self.MapPandas(df)
        return self.MapSpark(df)

    # def MapSpark(self, df: DataFrame) -> DataFrame:
    def MapSpark(self, df):
        self.setBroadCast(df.sql_ctx)
        return (
            df.join(self._modalities_broadcast, on=self.Name, how="left")
            .fillna({"id": self._default})
            .drop(self.Name)
            .withColumnRenamed("id", self.Name)
            .withColumn(self.Name, self.spark_col())
        )

    def MapPandas(self, df) -> pd.DataFrame:
        df[self.Name] = df[self.Name].apply(lambda x: self._dicoModalityToId.get(x, self._default))
        return df

    def Values(self, df: pd.DataFrame):
        return df[self.Name].values

    @staticmethod
    def BuildCtrBuckets(name: str, df, logbase=10, nbStd=1, gaussianStd=10):
        df["d"] = 1
        df = df.groupby(name).sum()
        df = df.reset_index()
        return RawFeatureMapping.BuildCtrBucketsFromAggDf(name, df, logbase, nbStd, gaussianStd)

    @staticmethod
    def BuildCtrBucketsFromAggDf(name: str, df, logbase, nbStd, gaussianStd):
        dicoModalityToId = RawFeatureMapping.ctrBucketsMapping(name, df, logbase, nbStd, gaussianStd)
        return RawFeatureMapping(name, dicoModalityToId)

    @staticmethod
    def ctrBucketsMapping(f, df, logbase, nbStd, gaussianStd):
        def getThreesholds(gaussianStd, maxN=1_000_000, nbStd=1.0):
            ts = []
            c = 1 + gaussianStd
            while c < maxN:
                std = np.sqrt(c) + gaussianStd
                c += nbStd * std
                ts.append(c)
            return np.array(ts)

        allThreeshold = getThreesholds(gaussianStd, df.click.max() * logbase, nbStd)
        prior = df.click.sum() / df.d.sum()
        df["roundedD"] = roundedD = logbase ** (1 + np.floor(np.log10(df.d) / np.log10(logbase)))
        d = df.d.values
        c = df.click.values
        df["ctr"] = ctr = (c + prior) / (d + 1)
        c_at_roundedD = ctr * roundedD
        import bisect

        df["ctrBucketId"] = [bisect.bisect(allThreeshold, x) for x in c_at_roundedD]
        # priorStd = np.sqrt( prior * (1-prior ) *  roundedD )/roundedD
        # priorStd *= nbStd
        # df["ctrBucketId"] = np.floor (ctr/priorStd) * priorStd
        df["key"] = list(zip(df["roundedD"].values, df["ctrBucketId"].values))
        allkeys = sorted(set(df["key"].values))
        len(allkeys)
        keysDico = {k: i for i, k in enumerate(allkeys)}
        df["newid"] = [keysDico[k] for k in df["key"].values]
        dicoOldModalityToNewModality = {old: new for old, new in zip(df[f], df["newid"])}
        return dicoOldModalityToNewModality


class RawFeaturesSet:
    def __init__(self, features, rawmappings):
        self.features = features
        self.rawmappings = rawmappings

    @staticmethod
    def FromDF(features, df):
        rawmappings = {f: RawFeatureMapping.FromDF(f, df) for f in features}
        return RawFeaturesSet(features, rawmappings)

    def dump(self, handle):
        pickle.dump(self.features, handle)
        for f in self.features:
            self.rawmappings[f].dump(handle)

    @staticmethod
    def load(handle, ss=None):
        features = pickle.load(handle)
        mappings = {}
        for f in features:
            mappings[f] = RawFeatureMapping.load(handle)
        return RawFeaturesSet(features, mappings)

    def Map(self, df):
        if isinstance(df, pd.DataFrame):
            df = df.copy()
        for var in self.rawmappings.values():
            if var.Name in df.columns:
                df = var.Map(df)
            else:
                print("warning:: RawFeaturesSet.Map :: feature " + var.Name + " not found in df")
                toto
        return df

    def __repr__(self):
        return ",".join(f.Name for f in self.rawmappings.values())

    
from typing import List
import pandas as pd
import numpy as np
import numba

try:
    from pyspark.sql import functions as F
except:
    pass
from dataclasses import dataclass


@numba.njit
def projectNUMBA(x, y, nbmods):
    mods = np.zeros(nbmods)
    n = len(x)
    for i in np.arange(0, n):
        mods[x[i]] += y[i]
    return mods


def GetCfName(variables):
    return "&".join(sorted(variables))


class IEncoding:
    Size: int

    def Values_(self, x: np.array):
        pass

    def Values(self, df: pd.DataFrame):
        pass

    def SparkCol(self):
        pass

    def ProjectDF(self, df, colname):
        if type(df) is pd.DataFrame:
            return self.ProjectPandasDF(df, colname)
        return self.ProjectSparkDF(df, colname)

    def ProjectPandasDF(self, df, colname):
        y = df[colname].values
        x = self.Values(df)
        return projectNUMBA(x, y, self.Size)

    def ProjectSparkDF(self, df, sum_on):
        col = self.SparkCol()
        dico = df.select(col.alias("toto"), sum_on).groupBy("toto").agg(F.sum(sum_on).alias(sum_on)).rdd.collectAsMap()
        proj = np.zeros(self.Size)
        proj[np.array(list(dico.keys()))] = np.array(list(dico.values()))
        return proj


@dataclass
class SingleFeatureEncoding(IEncoding):
    _fid: int
    Name: str
    Size: int

    def Values_(self, x: np.array):
        return x[self._fid]

    def Values(self, df: pd.DataFrame):
        return df[self.Name].values % self.Size

    def SparkCol(self):
        return F.col(self.Name) % F.lit(self.Size)

    def Project_(self, x: np.ndarray, y: np.array) -> np.array:
        x_values = x[self._fid] % self.Size
        return projectNUMBA(x_values, y, self.Size)

    def __repr__(self):
        return f"{self.Name}({self.Size})"

    @staticmethod
    def FromRawFeatureMapping(fid, rawmapping, maxSize: int = None):
        if maxSize is None:
            maxSize = rawmapping.Size
        return SingleFeatureEncoding(fid, rawmapping.Name, min(maxSize, rawmapping.Size))


@dataclass
class CrossFeatureEncoding(IEncoding):
    _f1: SingleFeatureEncoding
    _f2: SingleFeatureEncoding
    coefV2: int
    Size: int
    hashed: bool = False

    @property
    def _v1(self):
        return self._f1.Name

    @property
    def _v2(self):
        return self._f2.Name

    @property
    def _fid1(self):
        return self._f1._fid

    @property
    def _fid2(self):
        return self._f2._fid

    @property
    def _variables(self):
        return [self._v1, self._v2]

    @property
    def Name(self):
        return GetCfName([f for f in self._variables])

    @property
    def Modulo(self):
        return self.Size

    def Values_(self, x: np.ndarray) -> np.array:
        return (x[self._fid1] + self.coefV2 * x[self._fid2]) % self.Size

    def Values(self, df):
        return (self._f1.Values(df) + self.coefV2 * self._f2.Values(df)) % self.Size

    def SparkCol(self):
        return (self._f1.SparkCol() + self._f2.SparkCol() * F.lit(self.coefV2)) % F.lit(self.Size)

    def Project_(self, x: np.ndarray, y: np.array) -> np.array:
        x_values = self.Values_(x)
        return projectNUMBA(x_values, y, self.Size)

    def __repr__(self):
        if self.hashed:
            return "hash" + self.Name
        return self.Name

    def marginalize(self, y: np.ndarray, fname):
        if len(y) != self.Size:
            raise ValueError(f"{this}::marginalize len(y)={len(y)} != Size={self.Size}")
        values = self.modalitiesOtherFeature(fname)
        return projectNUMBA(values, y, len(set(values)))

    def modalitiesOtherFeature(self, fname):
        if fname == self._f1.Name:
            values = self._f2modalities()
        elif fname == self._f2.Name:
            values = self._f1modalities()
        else:
            raise ValueError(f"{this}::marginalize: unknown name {fname}")
        return values

    def _f1modalities(self):
        return np.arange(0, self.Size) % self._f1.Size

    def _f2modalities(self):
        return np.arange(0, self.Size) // self._f1.Size

    def fromIndepProbas(self, x1: np.ndarray, x2: np.ndarray):
        return np.outer(x1, x2).flatten()

    @staticmethod
    def FromSingleFeatureEncodings(f1, f2, maxSize=None):
        size = f1.Size * f2.Size
        if maxSize is None or size <= maxSize:
            return CrossFeatureEncoding(
                f1,
                f2,
                coefV2=f1.Size,
                Size=f1.Size * f2.Size,
                hashed=False,
            )
        else:
            return CrossFeatureEncoding(f1, f2, coefV2=7907, Size=maxSize, hashed=True)

        
#from aggregated_models.RawFeatureMapping import *
#from aggregated_models.FeatureEncodings import *

import pandas as pd
import pickle


def parseCFNames(features, crossfeaturesStr):
    cfs = parseCF(features, crossfeaturesStr)
    return [GetCfName(cf) for cf in cfs]


def parseCF(features, crossfeaturesStr):
    cfs = []
    crossfeaturesStr = crossfeaturesStr.split("|")
    for cfStr in crossfeaturesStr:
        cfFeatures = cfStr.split("&")
        nbWildcards = len([f for f in cfFeatures if f == "*"])
        cfFeatures = [[f for f in cfFeatures if not f == "*"]]
        for i in range(0, nbWildcards):
            cfFeatures = [cf + [v] for v in features for cf in cfFeatures]
        cfFeatures = [sorted(f) for f in cfFeatures]
        cfs += cfFeatures
    cfs = [list(sorted(set(cf))) for cf in cfs]
    cfs = [cf for cf in cfs if len(cf) == 2]
    # remove duplicates
    dicoCfsStr = {}
    for cf in cfs:
        s = "&".join([str(f) for f in cf])
        dicoCfsStr[s] = cf
    cfs = [cf for cf in dicoCfsStr.values()]
    return cfs


def getMaxNbModalities(var, maxNbModalities):
    if type(maxNbModalities) is dict:
        if var in maxNbModalities:
            return maxNbModalities[var]
        else:
            return maxNbModalities["default"]
    return maxNbModalities


class CrossFeaturesSet:
    def __init__(self, rawFeaturesSet: RawFeaturesSet, crossfeaturesStr, maxNbModalities=None):

        self.rawFeaturesSet = rawFeaturesSet
        self.crossfeaturesStr = crossfeaturesStr
        self.maxNbModalities = maxNbModalities
        self.build()

    @property
    def features(self):
        return self.rawFeaturesSet.features

    @property
    def mappings(self):
        return self.rawFeaturesSet.rawmappings

    def build(self):
        self.crossfeatures = parseCF(self.features, self.crossfeaturesStr)
        allfeatures = [f for cf in self.crossfeatures for f in cf]
        if any([f not in self.features for f in allfeatures]):
            raise Exception("Error: Some cross feature not declared in features list ")
        self.buildEncodings()

    def buildEncodings(self):
        self.encodings = {}
        for i, f in enumerate(self.features):
            maxNbModalities = getMaxNbModalities(f, self.maxNbModalities)
            rawMapping = self.rawFeaturesSet.rawmappings[f]
            self.encodings[f] = SingleFeatureEncoding.FromRawFeatureMapping(i, rawMapping, maxNbModalities)

        for cf in self.crossfeatures:
            if len(cf) != 2:
                raise Exception("cf of len !=2  not supported yet")
            maxNbModalities = getMaxNbModalities(GetCfName(cf), self.maxNbModalities)
            encoding = CrossFeatureEncoding.FromSingleFeatureEncodings(
                self.encodings[cf[0]], self.encodings[cf[1]], maxNbModalities
            )
            self.encodings[encoding.Name] = encoding

    def dump(self, handle):
        self.rawFeaturesSet.dump(handle)
        pickle.dump(self.crossfeaturesStr, handle)
        pickle.dump(self.maxNbModalities, handle)

    @staticmethod
    def load(handle):
        rawFeaturesSet = RawFeaturesSet.load(handle)
        crossfeaturesStr = pickle.load(handle)
        maxNbModalities = pickle.load(handle)
        return CrossFeaturesSet(rawFeaturesSet, crossfeaturesStr, maxNbModalities)

    def transformDf(self, df, alsoCrossfeatures=False):
        if alsoCrossfeatures:
            print("TODO :  reimplement 'alsoCrossfeatures' ")
            error
        return self.rawFeaturesSet.Map(df)

    def __repr__(self):
        return ",".join(f.Name for f in self.encodings.values())

    def fix_fids(self, features_sublist):
        # print("baseAggModel::fix_fids ")
        fid = 0
        encodings = self.encodings
        for f in features_sublist:
            mapping = encodings[f]
            mapping._fid = fid
            fid += 1

    @staticmethod
    def FromDf(df, features, maxNbModalities, crossfeaturesStr="*&*"):
        rawFeaturesSet = RawFeaturesSet.FromDF(features, df)
        return CrossFeaturesSet(rawFeaturesSet, crossfeaturesStr, maxNbModalities)

class WeightsSet:
    """Map a feature to a subarray in an array of parameters. Usefull to build linear models.
    feature: a SingleFeatureMapping or CrossFeaturesMapping
    offset: first index used for this feature. subarray will be [offset:offset+feature.Size]
    """

    def __init__(self, feature: IEncoding, offset: int):
        self.feature = feature
        self.offset = offset
        self.indices = np.arange(self.offset, self.offset + self.feature.Size)

    def GetIndices_(self, x: np.array):
        """x: np.array of shape (nb features, nb samples)
        return for each sample the index in param vector of (wrapped feature modality)"""
        return self.feature.Values_(x) + self.offset

    def GetIndices(self, df: pd.DataFrame):
        """df: dataframe of samples, containing one col for wrapped feature
        return for each line of df the index of (wrapped feature modality)
        """
        return self.feature.Values(df) + self.offset

    def __repr__(self):
        return f"WeightsSet on {self.feature} offset={self.offset}"