import logging
import gzip
import numpy as np
import pandas as pd
import dp_accounting
from dp_accounting.rdp.rdp_privacy_accountant import RdpAccountant
from dp_ad_click_prediction.feature_encodings import *

logger = logging.getLogger(__name__)

def computeEpsilon(nbSamples,   # size of the train set
                   batchsize,   # size of mini batches
                   noise_multiplier,  # at each minibatch, we should add to the gradient a gaussian iid noise of stdev = noise_multiplier * L2Sensitivity  
                   nbEpochs = 1,    # nb passes on the whole dataset
                   target_delta = 1e-8,
                   nb_learnings = 1):
    T = int(nbEpochs * nbSamples /  batchsize) # Number of minibatches
    accountant = RdpAccountant(neighboring_relation= dp_accounting.privacy_accountant.NeighboringRelation.REPLACE_ONE)
    eventgaussian = dp_accounting.GaussianDpEvent(noise_multiplier)
    event = dp_accounting.SampledWithoutReplacementDpEvent(nbSamples, batchsize, eventgaussian)
    accountant.compose( event, T * nb_learnings )
    epsilon = accountant.get_epsilon(target_delta=target_delta)
    return epsilon


class MyLogistic:
    
    def __init__(
        self,
        cfset,
        regulL2=1.0,
        intercept = np.log(0.1),
        clicksCfs="*&*",
        label = "click",
        verbose = False,
        model_seed = None
    ):
        self.featuresSet = cfset
        self.features = cfset.features
        self.regulL2 = regulL2
        self.label = label
        self.clicksCfs = parseCFNames(self.features, clicksCfs)
        self.intercept = intercept

        featuresAndCfs = self.features + self.clicksCfs
        self.clickWeights, offset = self.prepareWeights(featuresAndCfs)

        if model_seed:
           logger.info(f"seed model from {model_seed}")
           with open(model_seed, 'rb') as file:
                self.parameters = np.load(file)
        else:
            self.parameters = np.zeros(offset)
        self.nbCoefs = sum([w.feature.Size for w in self.clickWeights.values()])
        self._accountant = None
        self.objectivePerturbationVector = None
        self.verbose = verbose
        
    def setObjectivePerturbation(self, sigma, nbsamples ):
        self.objectivePerturbationVector = np.random.normal( 0, sigma/nbsamples  , len(self.parameters) )
        
    @staticmethod
    def DfToX(df, cfset): #  X vector one size NB features (not 1-hot encoded)
        df = cfset.transformDf(df, False)
        x = np.zeros((len(cfset.features), len(df)), dtype=np.int32)
        for f in cfset.features:
            encoding = MyLogistic.getEncoding(f, cfset)
            x[encoding._fid] = encoding.Values(df)
        return x

    @staticmethod
    def getEncoding(var, cfset):
        return cfset.encodings[var]

    def prepareWeights(self, featuresAndCfs, offset=0):
        weights = {}
        for var in featuresAndCfs:
            encoding = MyLogistic.getEncoding(var, self.featuresSet)
            weights[var] = WeightsSet(encoding, offset)
            offset += encoding.Size
        return weights, offset

    # X a matrix of size Nb Examples x Nb features
    def dotproducts_(self, weights, x, parameters):
        results = np.zeros(x.shape[1])
        for w in weights.values():
            results += parameters[w.GetIndices_(x)]
        return results

    def dotproducts(self, weights, x):
        return self.dotproducts_(weights, x, self.parameters)

    def predictDF(self, df):
        x = MyLogistic.DfToX(df, self.featuresSet)
        return self.predict(x)

    def predict(self, x):
        dotprods = self.dotproducts(self.clickWeights, x) + self.intercept
        return 1.0 / (1.0 + np.exp(-dotprods))
    
    def project(self, samples, v):
        x = self.parameters * 0
        for w in self.clickWeights.values():
            x[w.indices] = w.feature.Project_(samples, v)
        return x

    # grad of the loss, defined as a *sum* on the samples
    def computeGradient(self, samples, labels):
        preds = self.predict(samples)
        error = preds - labels
        gradient = self.project( samples, error )
        gradient += 2 * self.parameters * self.regulL2 * len(labels)
        
        if self.objectivePerturbationVector is not None:
            gradient +=  len(labels) * self.objectivePerturbationVector 
        return gradient

    # def computeGradientOnDF(self, df):
    #     x = MyLogistic.DfToX(df, self.featuresSet)
    #     y = df[self.label].values
    #     g = self.computeGradient(x,y)
    #     return g

    # def sgdStepOnDF( self, df , stepsize ):
    #     g = self.computeGradientOnDF(df)
    #     g = g / len(df)
    #     self.parameters = self.parameters - stepsize * g
    
    def _batchReadDF(batchsize, filename):
        df0 = pd.read_csv(filename, nrows=1)
        names = df0.columns
        with  gzip. open(filename, "rb") as file:
            header = file.readline()
            i = 0
            while True:
                df = pd.read_csv(file, nrows=batchsize, header=0, names=names)
                if len(df) < 1 :
                    break
                yield df
    
    @staticmethod
    def batchReadFile(batchsize, filename, cfset, label = "click"):
        if type(filename) is str: 
            for df in MyLogistic._batchReadDF(batchsize, filename):
                x = MyLogistic.DfToX(df, cfset)
                y = df[label].values
                yield (x,y)
        else:
            xy = filename
            n = len(xy[1])
            i = 0
            while i < n: 
                x = xy[0][:,i:i+batchsize]
                y = xy[1][i:i+batchsize]
                i += batchsize
                yield (x,y)
    
    def trainSgdAutoStepSize(self, batchsize, filename, nbPasses = 1):
        for x,y in MyLogistic.batchReadFile(batchsize, filename, self.featuresSet, self.label):
            c =  self.project( x , y )
            c += 5 + self.regulL2 * batchsize
            stepsize =  (1.0/len (self.clickWeights)) *  len( y ) *1.0/ c
            break
        self.trainSGD( batchsize, filename, stepsize, nbPasses = nbPasses )
    
    def aggclicks(self, filename):
        c = 0
        for x,y in MyLogistic.batchReadFile(100_000, filename, self.featuresSet, self.label):
            c += self.project( x, y )
        return c

    def fullgradient(self, filename):
        g = 0
        for x,y in MyLogistic.batchReadFile(100_000, filename, self.featuresSet, self.label):
            g += self.computeGradient(x,y)
        return g
    
    def invDiagStepsize(self, filename):
        c = self.aggclicks(filename)
        ## TODO:  inject noise here on c  (for objective perturbation?)            
        stepsize =  (1.0/len (self.clickWeights)) *  1.0 / (c  + 100)
        return stepsize
        
    def trainFullBatch(self, filename, nbIters, stepsize = None):
        if stepsize is None:         
            stepsize =  self.invDiagStepsize(filename)
        for i in range(0,nbIters):
            g = self.fullgradient(filename)
            self.parameters = self.parameters - stepsize * g
            if self.verbose:
                self.printTestLoss(i)
      
    def printTestLoss(self, nbSamples, filename_test, i, epsilon = np.inf):
        if not filename_test:
            logger.info("Cannot print test loss. need filename_test")
            return

        x,y = next(MyLogistic.batchReadFile(1_000_000, filename_test, self.featuresSet, self.label))
        nllh = self.computeNllh(x,y)
        logger.info( f"NbEpochs:{i}; nllh:{nllh}; epsilon:{epsilon}; noise:{self.noise_multiplier}; lambda*n:{self.regulL2*nbSamples}; B:{self.batchsize} " )
        
    def printEpsilon(self, nbSamples, batchsize, nbPasses, noise_multiplier, delta = 1e-8, nb_learnings=1):
        epsilon = computeEpsilon(nbSamples, batchsize, noise_multiplier, nbPasses, delta, nb_learnings)
        logger.info(f"epsilon: {epsilon}")
    
    def trainSGD(self,  
                 nbSamples,
                 batchsize, 
                 filename,                  
                 stepsize, 
                 nbPasses = 1, 
                 adam = None, 
                 noise_multiplier = 0,
                 max_epsilon = None,
                 nb_learnings = 1,                 
                 model_path = None,
                 filename_test = None):

        sigma = noise_multiplier * self.MaxL2Norm()
        self.noise_multiplier = noise_multiplier
        self.sigma = sigma
        self.batchsize = batchsize
        if noise_multiplier > 0:
            self.printEpsilon(nbSamples, batchsize, nbPasses, noise_multiplier, nb_learnings)            
            self._accountant = RdpAccountant(neighboring_relation= dp_accounting.privacy_accountant.NeighboringRelation.REPLACE_ONE)
            eventgaussian = dp_accounting.GaussianDpEvent(noise_multiplier)
            event = dp_accounting.SampledWithoutReplacementDpEvent(nbSamples, batchsize, eventgaussian)
      
        i = 0
        for nbiter in range(1,nbPasses+1):
            if self.verbose:
                logger.info(f"starting epoch {nbiter}")
            for x,y in MyLogistic.batchReadFile(batchsize, filename, self.featuresSet, self.label):
                i +=1
                if i%10 == 0 and self.verbose:
                    nllh = self.computeNllh(x, y)
                    msg = f"Batch {i} nllh = {nllh}    "
                    if sigma > 0:
                        eps = self._accountant.get_epsilon(target_delta =1e-8)
                        msg += f"eps={eps}     "
                    logger.info(msg)
                g = self.computeGradient(x,y)
                g += np.random.normal(0, sigma, len( self.parameters ) )
                g = g / len(y)
                
                if adam is not None:
                    g = adam.Step( g )
                self.parameters = self.parameters - stepsize * g
                if sigma > 0 :
                    self._accountant.compose( event, 1 * nb_learnings)
            eps = None
            if sigma > 0:
                eps = self._accountant.get_epsilon(target_delta =1e-8)
            self.printTestLoss(nbSamples, filename_test, nbiter, eps)
            if sigma > 0 and max_epsilon is not None and eps > max_epsilon:
                logger.info( "stopping: max epsilon reached" )
        
        if model_path:
            with open(model_path, 'wb') as file:
                np.save(file, self.parameters)
                
    def __repr__(self):
        return f"logistic({self.features},λ={self.regulL2:.1E})"

    # def computeNllhOnDF(self, df):
    #     x = MyLogistic.DfToX(df, self.featuresSet)
    #     y = df[self.label].values
    #     return self.computeNllh(x, y)
    
    def computeNllh(self, x, y):
        preds = self.predict(x)
        return NLLH(preds, y )

    def MaxL2Norm( self ):
        return np.sqrt(len (self.clickWeights) )
    

def LLH(prediction, y):
    llh = np.log(prediction) * y + np.log(1 - prediction) * (1 - y)
    return sum(llh) / len(y)

def Entropy(y):
    py = sum(y > 0) / len(y)
    return Entropy_(py)

def Entropy_(py):
    return py * np.log(py) + (1 - py) * np.log(1 - py)

def NLLH(prediction, y):
    h = Entropy(y)
    llh = LLH(prediction, y)
    return (h - llh) / h        
    

class Adam:
    def __init__(self, alpha, beta1 = 0.9 , beta2 = 0.999, epsilon = 1e-8, decay  = 0 ):
        self.alpha = alpha
        self.beta1 = beta1 
        self.beta2 = beta2 
        self.epsilon = epsilon
        self.decay = decay
        self.m = 0
        self.v = 0        
        self.t = 0
        
    def Step(self, gradient):
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * gradient * gradient
        mhat = self.m / (1.0 - self.beta1**(self.t+1))
        vhat = self.v / (1.0 - self.beta2**(self.t+1))
        step = self.alpha * mhat / (np.sqrt(vhat) + self.epsilon) 
        step = step / ( 1+ self.decay * self.t )

        self.t +=1 
        return step

