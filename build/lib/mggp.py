'''
Created on May 2020

@author: Henrique
'''


from deap import gp, tools, base, creator
import numpy as np
from scipy.linalg import inv
import operator
import random
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors
import re
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
import math
import pickle

from copy import deepcopy

class mggpElement(object):
    def __init__(self,weights=(-1.0,)):
        self.setPset()
        self._weights = weights
        self.renameArguments()
        creator.create("Program", gp.PrimitiveTree,fitness=None,
                       pset=self._pset)
        creator.create("FitnessMin", base.Fitness, weights=weights)
        creator.create("Individual", list, fitness=creator.FitnessMin)
#---primitive-set--------------------------------------------------------------
    def setPset(self,maxDelay=3,numberOfVariables=2,MA=False,constant=False):
        self._MA = MA
        self._constant = constant
        def _makePset(delays,MA):
            if MA:
                pset = gp.PrimitiveSet("main", numberOfVariables+1)
            else:
                pset = gp.PrimitiveSet("main", numberOfVariables)
            pset.addPrimitive(operator.mul,2)
            for roll in delays:
                pset.addPrimitive(roll,1,name='q{}'.format(
                    delays.index(roll)+1))
            if self._constant:
                pset.addTerminal(1,'1')
            return pset
    #---set-one-step-ahead-pset---
        def roll(*args):
            try:
                return np.roll(*args)
            except np.AxisError as e:
                return args[0]
        delays = [lambda x,i=i: roll(x,i) for i in range(1,maxDelay+1)]
        self._pset = _makePset(delays,MA)
    #---set-k-steps-ahead-pset----
        delays = [lambda x,i=i: roll(x,i,1) for i in range(1,maxDelay+1)]
        self._k_pset = _makePset(delays,MA)
        
    def renameArguments(self,dictionary={'ARG0':'y1','ARG1':'u1'}):
        if self._MA:
            if 'e1' in dictionary.values():
                raise Exception('Name \'e1\' reserved for Moving Average operations!')
            dictionary.update({self._pset.arguments[-1]:'e1'})
        self._pset.renameArguments(**dictionary)
        self._k_pset.renameArguments(**dictionary)
#------------------------------------------------------------------------------
    def addPrimitive(self,*args):
        self._pset.addPrimitive(*args)
        self._k_pset.addPrimitive(*args)
#---pset-getters---------------------------------------------------------------
    def get_pset(self):
        return self._pset
    def get_k_pset(self):
        return self._k_pset
#---create-model-function------------------------------------------------------
    def createModel(self,listString):
        model = []
        for string in listString:
            model.append(gp.PrimitiveTree.from_string(string,self._pset))
        model = creator.Individual(model)
        return model
#---Individual-object-to-list-of-strings---------------------------------------
    def model2List(self,model):
        modelList = []
        for tree in model:
            modelList.append(str(tree))
        return modelList
#---model-compilers------------------------------------------------------------
    def compile_model(self,model):
        funcs = []
        kfuncs = []
        for tree in model:
            funcs.append(gp.compile(tree,self._pset))
            kfuncs.append(gp.compile(tree,self._k_pset))
        model.funcs=funcs
        model.kfuncs=kfuncs
        self.setModelLagMax(model)
#---model-maximum-lag-setter---------------------------------------------------
    def setModelLagMax(self,model):
        treelags = []
        for tree in model:
            i = 0
            lagMax = 0
            while i<len(tree):
                count = 0
                if re.search("q\d",tree[i].name):
                    count+=int(tree[i].name[1])
                    subtree = tree[tree.searchSubtree(i+1)]
                    for s in subtree:
                        if re.search("q\d",s.name):
                            count+=int(s.name[1])
                        else:
                            i+=len(subtree)
                else:
                    i+=1
                if count>lagMax:
                    lagMax=count
            tree.lagMax = lagMax
            treelags.append(lagMax)
        model.lagMax = max(treelags)
#---regressor-matrix-builders--------------------------------------------------
# moving average builder for white noise Output Error
    def _noiseModel(self,model,*variables):
        noiseModel = []
        for tree in model:
            findY = True if str(tree).find(self._pset.arguments[0])>-1 else False
            if findY:
                noiseModel.append(tree)
        noiseModel = creator.Individual(noiseModel)
        self.compile_model(noiseModel)
        self.setModelLagMax(noiseModel)
        p = self._p_builder(noiseModel,*variables)
        return p[model.lagMax-noiseModel.lagMax:]
# use for NARX and NARMAX with MA extended from white noise Output Error
    def _p_builder(self,model,*variables):
        p = []
        for i in range(len(model)):
            func = model.funcs[i]
            out = func(*variables)
            if type(out) is int:
                out = (np.ones(variables[0].shape)*out)
            p.append(out.reshape(-1))
        p = np.array(p).T[model.lagMax:]
        return p  
# get regressor Matrix
    def makeRegressors(self,model,y,*inputs,**options):
        opts = {'mode':'default',
                'theta':None}
        opts.update(options)
        listV = [y[:-1].reshape(-1,1)]
        for v in inputs:
            listV.append(v[:-1].reshape(-1,1))
        if opts['mode']=='default':
            return self._p_builder(model,*listV)
        elif opts['mode']=='extended':
            if opts['theta'] is None:
                raise Exception('The option \'theta\' is not set!')
            theta = opts['theta'].reshape(-1,1)
            if self._MA:
                listV.append(np.zeros(listV[0].shape))
                p = self._p_builder(model,*listV)
                yd = y[model.lagMax+1:]
                y_pred = p@theta
                e = yd-y_pred
                e = np.concatenate((np.zeros((model.lagMax+1,1)),e))
                del listV[-1]
                listV.append(e[:-1].reshape(-1,1))
                p = self._p_builder(model,*listV)
                return p
            else:
                p = self._p_builder(model,*listV)
                yd = y[model.lagMax+1:]
                y_pred = p@theta[:len(model)]
                e = yd-y_pred
                e = np.concatenate((np.zeros((model.lagMax+1,1)),e))
                try:
                    listV = [e[:-1]]
                    for v in inputs:
                        listV.append(v[:-1].reshape(-1,1))
                    p_n = self._noiseModel(model,*listV)
                    p_e = np.concatenate((p,p_n),axis=1)
                    return p_e
                except:
                    return p
        else:
            raise Exception('Mode not accepted!')
            
#---least-squares-methods------------------------------------------------------
    def ls(self,model,y,*inputs):
        listV = [y[:-1].reshape(-1,1)]
        for v in inputs:
            listV.append(v[:-1].reshape(-1,1))
        p = self._p_builder(model,*listV)
        if np.linalg.cond(p,-2)<1e-10:
                raise np.linalg.LinAlgError(
                    'Ill conditioned regressors matrix!')
        yd = y[model.lagMax+1:]
        return inv(p.T@p)@p.T@yd
    
    def ls_extended(self,model,y,*inputs):
        iterations = 5
        listV = [y[:-1].reshape(-1,1)]
        for v in inputs:
            listV.append(v[:-1].reshape(-1,1))
        if self._MA:
            listV.append(np.zeros(listV[0].shape))
            p = self._p_builder(model,*listV)
            idx = np.argwhere(np.all(p == 0, axis=0))
            p = np.delete(p, idx, axis=1)
            yd = y[model.lagMax+1:]
            if np.linalg.cond(p,-2)<1e-10:
                raise np.linalg.LinAlgError(
                    'Ill conditioned regressors matrix!')
            theta = inv(p.T@p)@p.T@yd
            y_pred = p@theta
            e = yd-y_pred
            for i in range(iterations):
                del listV[-1]
                e = np.concatenate((np.zeros((model.lagMax+1,1)),e))
                listV.append(e[:-1].reshape(-1,1))
                p = self._p_builder(model,*listV) 
                theta = inv(p.T@p)@p.T@yd
                y_pred = p@theta
                e = yd-y_pred
            return theta
        else:
            p= self._p_builder(model,*listV)
            if np.linalg.cond(p,-2)<1e-10:
                    raise np.linalg.LinAlgError(
                        'Ill conditioned regressors matrix!')
            yd = y[model.lagMax+1:]
            theta = inv(p.T@p)@p.T@yd
            y_pred = p@theta
            e = yd-y_pred
            for i in range(1,iterations+1):
                try:
                    e = np.concatenate((np.zeros((model.lagMax+1,1)),e))
                    listV = [e[:-1]]
                    for v in inputs:
                        listV.append(v[:-1].reshape(-1,1))
                    p_n = self._noiseModel(model,*listV)
                    if len(p_n)==0:
                        return theta
                    p_e = np.concatenate((p,p_n),axis=1)
                    theta = inv(p_e.T@p_e)@p_e.T@yd
                    y_pred = p_e@theta
                    e = yd-y_pred
                except ValueError:
                    return theta
            return theta
#---predictors-----------------------------------------------------------------
    def getDesiredY(self,model,y):
        if len(y.shape)==3:
            return y[:,model.lagMax+1:,:]
        return y[model.lagMax+1:]
    
    def predict(self,model,theta,y,*inputs):
        theta = theta.reshape(-1,1)
        listV = [y[:-1].reshape(-1,1)]
        for v in inputs:
            listV.append(v[:-1].reshape(-1,1))
        p = self._p_builder(model,*listV)
        out = p@theta[:len(model)]
        return out
    
    def predict_extended(self,model,theta,y,*inputs):
        theta = theta.reshape(-1,1)
        listV = [y[:-1].reshape(-1,1)]
        for v in inputs:
            listV.append(v[:-1].reshape(-1,1))
        if self._MA:
            listV.append(np.zeros(listV[0].shape))
            p = self._p_builder(model,*listV)
            yd = y[model.lagMax+1:]
            y_pred = p@theta
            e = yd-y_pred
            e = np.concatenate((np.zeros((model.lagMax+1,1)),e))
            del listV[-1]
            listV.append(e[:-1].reshape(-1,1))
            p = self._p_builder(model,*listV)
            return p@theta
        else:
            p = self._p_builder(model,*listV)
            yd = y[model.lagMax+1:]
            y_pred = p@theta[:len(model)]
            e = yd-y_pred
            e = np.concatenate((np.zeros((model.lagMax+1,1)),e))
            try:
                listV = [e[:-1]]
                for v in inputs:
                    listV.append(v[:-1].reshape(-1,1))
                p_n = self._noiseModel(model,*listV)
                p_e = np.concatenate((p,p_n),axis=1)
                y_pred = p_e@theta
                return y_pred
            except:
                return y_pred
    
    def predict_freeRun(self,model,theta,y0,*inputs):
        theta = theta.reshape(-1,1)
        y = y0[:model.lagMax+1].reshape(-1,1)
        if self._MA:
            e = np.zeros((model.lagMax+1,1))
            for i in range(len(inputs[0])-model.lagMax):
                listU = [y[i:i+model.lagMax+1].reshape(-1,1)]
                for u in inputs:
                    listU.append(u[i:i+model.lagMax+1].reshape(-1,1))
                listU.append(e)
                p = self._p_builder(model,*listU)
                ypred = np.dot(p,theta)
                y = np.vstack((y,ypred))
        else:
            for i in range(len(inputs[0])-model.lagMax):
                listU = [y[i:i+model.lagMax+1].reshape(-1,1)]
                for u in inputs:
                    listU.append(u[i:i+model.lagMax+1].reshape(-1,1))
                p = self._p_builder(model,*listU)
                y = np.vstack((y,np.dot(p,theta[:len(model)])))
        return y
    
    def predict_ksa(self,model,theta,k,y,*inputs):
        theta = theta.reshape(-1,1)
        n_batchs = int(np.floor(inputs[0].shape[0]/(model.lagMax+1+k)))
        N = model.lagMax+1+k
        newshape = (n_batchs,model.lagMax+1+k,1)
        listU = []
        for u in inputs:
            listU.append(np.resize(u,newshape))
        yk = np.resize(y,newshape)
        #yd = yk[:,model.lagMax+1:,:]
        y0 = yk[:,:model.lagMax+1,:]
        for i in range(N-model.lagMax-1):
            p = []
            for j in range(len(model)):
                func = model.kfuncs[j]
                listV = [y0[:,i:i+model.lagMax+1,:]]
                for u in listU:
                    listV.append(u[:,i:i+model.lagMax+1,:])
                out = func(*listV)
                if type(out) is int: 
                    out = np.ones((n_batchs,1,1))
                else:
                    out = out[:,model.lagMax:,:]
                p.append(out)
            p = np.concatenate(p,axis=2)
            y0 = np.concatenate((y0,np.dot(p,theta[:len(model)])),axis=1)
        return y0,yk
#---MSE-scores-----------------------------------------------------------------
    def score_osa(self,model,theta,y,*inputs):
        out = self.predict(model,theta,y,*inputs)
        return mean_squared_error(y[model.lagMax+1:],out)
    def score_osa_ext(self,model,theta,y,*inputs):
        out = self.predict_extended(model,theta,y,*inputs)
        return mean_squared_error(y[model.lagMax+1:],out)
    def score_freeRun(self,model,theta,y,*inputs):
        listI = []
        for i in inputs:
            listI.append(i[:-1])
        out = self.predict_freeRun(model,theta,y,*listI)
        try:
            return mean_squared_error(y[model.lagMax+1:],out[model.lagMax+1:])
        except ValueError:
            return np.inf
    def score_ksa(self,model,theta,k,y,*inputs):
        yp,yk = self.predict_ksa(model,theta,k,y,*inputs)
        yp = self.getDesiredY(model, yp).reshape(-1,1)
        yk = self.getDesiredY(model, yk).reshape(-1,1)
        try:
            return mean_squared_error(yk,yp)
        except ValueError:
            return np.inf
#---OLS-pruning-function-------------------------------------------------------
    def ols(self,model,tol,y,*inputs,**kwargs):
        opts = {'theta':None}
        opts.update(kwargs)
        def s(W,A):
            soma = 0
            for j in range(W.shape[1]):
                soma += A[j]*W[:,j]
            return soma
        yd = y[model.lagMax+1:]
        if self._MA:
            warnings.warn('Warning!! Moving Average is active. Probably all '
                          'residual terms will be removed.\n')
            if opts['theta'] is None:
                msg = ('For Moving Average ols it is needed '
                       'to set the parameter \'theta\' \n'
                       'ols(...,theta=theta)')
                raise Exception(msg)
            P = self.makeRegressors(model,y,*inputs,mode='extended',
                                    theta=opts['theta'])
        else:
            P = self.makeRegressors(model,y,*inputs)
        z = yd
        N = P.shape[0]
        M = P.shape[1]
        W = np.zeros((N,1))
        A = np.zeros((M,1))
        IDX = []
        err = np.zeros((M))
        for i in range(M):
            w = P[:,i].reshape(-1,1)
            g = (w.T@z)/(w.T@w)
            err[i]=((g**2)*(w.T@w)/(z.T@z))
        idx = np.argmax(err)
        ERR = err[idx]
        IDX.append(idx)
        W[:,0] = P[:,idx]
        while len(IDX)<M:
            Aux = np.zeros((M,M))
            err = np.zeros((M))
            for i in range(M):
                if i not in IDX:
                    p = P[:,i].reshape(-1,1)
                    for k in range(W.shape[1]):
                        w = W[:,k].reshape(-1,1)
                        Aux[k,i] = (w.T@p)/(w.T@w)
                    w = p-s(W,Aux[:,i]).reshape(-1,1)
                    g = (w.T@z)/(w.T@w)
                    err[i]=((g**2)*(w.T@w)/(z.T@z))
            idx = np.argmax(np.nan_to_num(err))
            if err[idx]<tol:
                break
            ERR+=err[idx]
            IDX.append(idx)
            W = np.column_stack((W,P[:,idx]-s(W,Aux[:,idx])))
            A = np.column_stack((A,Aux[:,idx]))
        A = A[:A.shape[1],:]
        np.fill_diagonal(A,1)
        for i in range(len(model)):
            if i not in IDX:
                model[i] = None
        i = 0
        while i < len(model):
            if model[i] == None:
                model.remove(model[i])
                model.funcs.remove(model.funcs[i])
                model.kfuncs.remove(model.kfuncs[i])
            else:
                i+=1
#------------------------------------------------------------------------------
    def getTerminalsObjects(self):
        return self._pset.terminals[object]
    def constraint_funcs(self,model,funcs,consts,values=None):
        # print('constraintFuncs')
        flag = False
        if values is None:
            flag = True
        elif type(values) is not list:
            values = [values]
        for tree in model:
            # print('tree',str(tree))
            i=0
            while i<len(tree):
                # print('i',i,tree[i].name)
                if tree[i].name in funcs:
                    # print('Found',tree[i].name)
                    j=1
                    while j<len(tree[tree.searchSubtree(i)]):
                        # print('i+j',j,tree[i+j].name)
                        if tree[i+j].name in consts:
                            # print('Found misplaced',tree[i+j].name)
                            if flag:
                                raise IOError('misplaced function inside sgn!')
                            else:
                                del tree[tree.searchSubtree(i+j)]
                                value = random.choice(values)
                                tree.insert(i+j,value)
                                j+=1
                        else:
                            j+=1
                    i+=len(tree[tree.searchSubtree(i)])
                else:
                    i+=1
#---save-load-file-function---------------------------------------------------------
    def save(self,filename,dictionary):
        with open(filename, 'wb') as f:
            pickle.dump(dictionary,f)
            f.close()
    def load(self,filename):
        with open(filename, 'rb') as f:
            o = pickle.load(f)
            f.close()
            return o

class mggpEvolver(object):
    def __init__(self,popSize=100,CXPB=0.9,MTPB=0.1,n_gen=50,maxHeight=3,
                 maxTerms=5,verbose=True,elite=5,element=None):
    #--------------------------------------------------------------------------
        if element==None:
            raise Exception('It needs an element module!')
        self._pset = element.get_pset()
        self._element = element
        self._elite = elite
        self._popSize = popSize
        self._CXPB = CXPB
        self._MTPB = MTPB
        self._n_gen = n_gen
        self._verbose = verbose
        self._maxTerms = maxTerms
    #--------------------------------------------------------------------------
        self._toolbox = base.Toolbox()
        self._toolbox.register("expr", gp.genHalfAndHalf, pset=self._pset, 
                               min_=0, max_=maxHeight)
        self._toolbox.register("program", tools.initIterate, 
                               creator.Program,self._toolbox.expr)
        self._toolbox.decorate("program",self._generationConstraint())
        self._toolbox.register("individual", tools.initRepeat, creator.Individual,
                         self._toolbox.program)
        self._toolbox.register("population", tools.initRepeat, list, 
                       self._toolbox.individual)
    #---Selection--------------------------------------------------------------
        self._toolbox.register("select",tools.selTournament,tournsize=2)
        self._toolbox.register("selBest",tools.selBest)
        self._toolbox.register('selNSGA2',tools.selNSGA2)
    #---CrossOver--------------------------------------------------------------
        self._toolbox.register("highCross", self._highCross)
        self._toolbox.decorate("highCross",self._highConstraint(max_=maxTerms))
        self._toolbox.register('GPcross',gp.cxOnePoint)
        self._toolbox.decorate("GPcross",self._gpMateMutateConstraint(maxHeight))
        self._toolbox.register("lowCross",self._lowCross,
                               gpCrossFunc=self._toolbox.GPcross)
    #---Mutation---------------------------------------------------------------
        self._toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=maxHeight-1)
        self._toolbox.register("mutateGP", gp.mutUniform, 
                               expr=self._toolbox.expr_mut, pset=self._pset)
        self._toolbox.decorate("mutateGP",self._gpMateMutateConstraint(maxHeight))
        self._toolbox.register("mutate", self._mutOneTree, 
                               gpMutFunc = self._toolbox.mutateGP)
        self._mutList = [self._toolbox.mutate, 
                         self._mutReplaceTree,]

    def _createStatistics(self):
        stats_f1 = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_f2  = tools.Statistics(lambda ind: len(ind))
        mstats = tools.MultiStatistics(fitness1=stats_f1,terms=stats_f2)
        mstats.register("avg", np.mean)
#        mstats.register("std", np.std)
        mstats.register("max", np.max)
        mstats.register("min", np.min)
        return mstats
#------------------------------------------------------------------------------
#---Pop-generation-------------------------------------------------------------
#------------------------------------------------------------------------------
    def _initPop(self,popSize,maxTerms,seed):
        pop = []
        if seed==None:
            for i in range(popSize):
                ind = self._toolbox.individual(n=random.randint(1,maxTerms))
                self._element.setModelLagMax(ind)
                fit = self._toolbox.evaluate(ind)
                ind.fitness.values=fit
                pop.append(ind)
            return pop
        else:
            for i in range(popSize-len(seed)):
                ind = self._toolbox.individual(n=random.randint(1,maxTerms))
                self._element.setModelLagMax(ind)
                fit = self._toolbox.evaluate(ind)
                ind.fitness.values = fit
                pop.append(ind)
            fitnesses = self._toolbox.map(self._toolbox.evaluate, seed)
            for ind, fit in zip(seed, fitnesses):
                ind.fitness.values = fit
            return pop + seed
#------------------------------------------------------------------------------
#---Constraints----------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#---GP-generation-constraint---------------------------------------------------
#------------------------------------------------------------------------------
    def _generationConstraint(self):
        def decorator(func):
            def wrapper(*args, **kargs):
                tree = func(*args,**kargs)
                i = 0
                lagMax = 0
                while i<len(tree):
                    count = 0
                    if re.search("q\d",tree[i].name):
                        count+=int(tree[i].name[1])
                        subtree = tree[tree.searchSubtree(i+1)]
                        for s in subtree:
                            if type(s)==gp.Primitive and not re.search("q\d",s.name):
                                del tree[tree.searchSubtree(i+subtree.index(s)+1)]
                                tree.insert(i+subtree.index(s)+1,gp.genFull(self._pset,0,0)[0])
                                i=0
                                break
                            elif s.name == '1':
                                del tree[tree.searchSubtree(i+subtree.index(s)+1)]
                                tree.insert(i+subtree.index(s)+1,gp.genFull(self._pset,0,0)[0])
                                i=0
                                break
                            elif re.search("q\d",s.name):
                                count+=int(s.name[1])
                            else:
                                i+=len(subtree)
                    else:
                        i+=1
                    if count>lagMax:
                        lagMax=count
                tree.__setattr__('lagMax',lagMax)
                return tree
            return wrapper
        return decorator
#------------------------------------------------------------------------------
#---GP-crossover-and-mutation-constraint---------------------------------------
#------------------------------------------------------------------------------
    def _gpMateMutateConstraint(self,max_height):
        def decorator(func):
            def wrapper(*args, **kargs):
                clone = tuple(map(self._toolbox.clone,args))
                offspring = func(*args,**kargs)
                for tree in offspring:
                    if tree.height>max_height:
                        return clone
                map(self._element.setModelLagMax,offspring)
                return offspring
            return wrapper
        return decorator
    
#---high-level-crossover-constraint--------------------------------------------
    def _highConstraint(self,max_):
        def decorator(func):
            def wrapper(*args, **kargs):
                clone = tuple(map(self._toolbox.clone,args))
                offspring = func(*args,**kargs)
                for child in offspring:
                    if len(child)>max_:
                        return clone
                map(self._element.setModelLagMax,offspring)
                return offspring
            return wrapper
        return decorator
#------------------------------------------------------------------------------
#---Crossover------------------------------------------------------------------
#------------------------------------------------------------------------------
#---High-Level-Crossover-------------------------------------------------------
    def _highCross(self,ind1,ind2):
        idx1 = random.randint(0,len(ind1)-1)
        idx2 = random.randint(0,len(ind2)-1)
        aux = ind1[idx1:]
        del ind1[idx1:]
        ind1+=ind2[idx2:]
        del ind2[idx2:]
        ind2+=aux
        return ind1,ind2
#---Low-Level-Crossover--------------------------------------------------------
    def _lowCross(self,ind1,ind2,gpCrossFunc):
        idx1 = random.randint(0,len(ind1)-1)
        idx2 = random.randint(0,len(ind2)-1)
        ind1[idx1],ind2[idx2] = gpCrossFunc(ind1[idx1],ind2[idx2])
        return ind1,ind2
#------------------------------------------------------------------------------
#---Mutation-------------------------------------------------------------------
#------------------------------------------------------------------------------
    def _mutOneTree(self,ind,gpMutFunc):
        idx = random.randint(0,len(ind)-1)
        ind[idx] = gpMutFunc(ind[idx])[0]
        return ind,
    def _mutTreeUniform(self,ind):
        indpb = 0.1
        for i in range(len(ind)):
            if random.random()<indpb:
                ind[i] = self._toolbox.mutateGP(ind[i])[0]
        return ind,
    def _mutReplaceTree(self,ind):
        idx = random.randint(0,len(ind)-1)
        ind[idx] = self._toolbox.program()
        return ind,
    def _mutRemoveTreeUniform(self,ind):
        indpb = 0.1
        for tree in ind:
            if random.random()<indpb:
                del tree
        return ind,
    def _mutAddTree(self, ind):
        n = random.randint(0,self._maxTerms-len(ind))
        ind.extend(self._toolbox.individual(n=n))
        return ind,
                
    def _delAttr(self,ind):
        try:
            del ind.fitness.values
            del ind.funcs
            del ind.kfuncs
            del ind.lagMax
        except AttributeError:
            pass

    def run(self,evaluate=None,seed=None):
        if evaluate==None:
            raise Exception('It needs an evaluation function!')
        self._toolbox.register("evaluate",evaluate)
    #---Setup--Statistics------------------------------------------------------
        mstats = self._createStatistics()
        logbook = tools.Logbook()
    #---Initialize--Population-------------------------------------------------
        pop = self._initPop(self._popSize,self._maxTerms,seed)
    #---Initialize--Hall-of-Fame-----------------------------------------------
        hofSize = int(round(self._popSize*(self._elite/100)))
        hof = tools.HallOfFame(hofSize)
        hof.update(pop)
    #---Record--Statistics-----------------------------------------------------
        record = mstats.compile(pop)
        logbook.record(gen=0, evals=self._popSize, **record)
    #---Print--Statistics------------------------------------------------------
        header = 'gen','fitness1','terms'
        logbook.header = header
        logbook.chapters['fitness1'].header = 'min','max','avg'
        logbook.chapters['terms'].header = 'min','max','avg'
        if self._verbose:
            print(logbook.stream)
    #---Loop-------------------------------------------------------------------
        for g in range(self._n_gen):
        #---Select--and--clone--the--next--generation--individuals-------------
            offspring = list(map(deepcopy, 
                                 self._toolbox.select(pop, self._popSize-hofSize)))
        #---Apply--crossover--and--mutation--on--the--offspring----------------
        #---CrossOver--------------------------------------------------------------
            for i in range(0,len(offspring),2):
                if np.random.random() < self._CXPB:
                    if np.random.random() < 0.5:
                        offspring[i], offspring[i+1] = self._toolbox.highCross(offspring[i], 
                                                                        offspring[i+1])
                        self._delAttr(offspring[i])
                        self._delAttr(offspring[i+1])
                    else:
                        offspring[i], offspring[i+1] = self._toolbox.lowCross(offspring[i], 
                                                                        offspring[i+1])
                        self._delAttr(offspring[i])
                        self._delAttr(offspring[i+1])
        #---Mutation---------------------------------------------------------------
            for i in range(len(offspring)):
                if offspring[i].fitness.valid:
                    if np.random.random() < self._MTPB:
                        func = random.choice(self._mutList)
                        offspring[i], = func(offspring[i])
                        self._delAttr(offspring[i])
                        # if np.random.random() < 0.5:
                        #     offspring[i], = self._toolbox.mutate(offspring[i])
                        #     del offspring[i].fitness.values
                        # else:
                        #     offspring[i], = self._mutReplaceTree(offspring[i])
                        #     del offspring[i].fitness.values
                                    
        #---Evaluate--the--individuals--with--an--invalid--fitness-------------
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self._toolbox.map(self._toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        #---Apply--Elitism-----------------------------------------------------
            # pop = tools.selBest(offspring+hof.items, self._popSize)
            pop = hof.items+offspring
            hof.update(pop)
        #---Record--Statistics-------------------------------------------------
            record = mstats.compile(pop)
            logbook.record(gen=g+1, evals=len(invalid_ind),**record)
        #---Print--Statistics--------------------------------------------------
            if self._verbose:
                print(logbook.stream)
            minvalue = min(pop,key = lambda item: item.fitness.values[0])
            maxvalue = max(pop,key = lambda item: item.fitness.values[0])
            if minvalue == maxvalue:
                break
        return hof,logbook
    
    def _createMOstatistics(self):
        statsList = {}
        for i in range(len(self._element._weights)):
            statsList.update({'Fitness{}'.format(i+1):
                tools.Statistics(lambda ind,j=i: ind.fitness.values[j])})
        mstats = tools.MultiStatistics(statsList)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("max", np.max)
        mstats.register("min", np.min)
        return mstats
    
    def _paretoSimilar(self,a,b):
        if a.fitness.values == b.fitness.values:
            return True
        else:
            return False
        
    def runMO(self,evaluate=None,seed=None,popPercent=0.8):
        if evaluate==None:
            raise Exception('It needs an evaluation function!')
        self._toolbox.register("evaluate",evaluate)
    #---Setup--Statistics------------------------------------------------------
        mstats = self._createMOstatistics()
        logbook = tools.Logbook()
    #---Initialize--Population-------------------------------------------------
        pop = self._initPop(self._popSize,self._maxTerms,seed)
    #---Initialize--Hall-of-Fame-----------------------------------------------
        pop = self._toolbox.selNSGA2(pop, len(pop))
        hof = tools.ParetoFront(self._paretoSimilar)
        hof.update(pop)
    #---Record--Statistics-----------------------------------------------------
        record = mstats.compile(pop)
        logbook.record(gen=0, evals=self._popSize, **record)
        for char in logbook.chapters:
            logbook.chapters[char].header = 'min','max'
    #---Print--Statistics------------------------------------------------------
        if self._verbose:
            print(logbook.stream)
    #---Loop-------------------------------------------------------------------
        for g in range(self._n_gen):
        #---Select--and--clone--the--next--generation--individuals-------------
            offspring = list(map(self._toolbox.clone, 
                                 tools.selTournamentDCD(pop, int(len(pop)*popPercent))))
        #---Apply--crossover--and--mutation--on--the--offspring----------------
        #---CrossOver--------------------------------------------------------------
            for i in range(0,len(offspring),2):
                if np.random.random() < self._CXPB:
                    if np.random.random() < 0.5:
                        offspring[i], offspring[i+1] = self._toolbox.highCross(offspring[i], 
                                                                        offspring[i+1])
                        self._delAttr(offspring[i])
                        self._delAttr(offspring[i+1])
                    else:
                        offspring[i], offspring[i+1] = self._toolbox.lowCross(offspring[i], 
                                                                        offspring[i+1])
                        self._delAttr(offspring[i])
                        self._delAttr(offspring[i+1])
        #---Mutation---------------------------------------------------------------
            for i in range(len(offspring)):
                    if np.random.random() < self._MTPB:
                        func = random.choice(self._mutList)
                        offspring[i], = func(offspring[i])
                        self._delAttr(offspring[i])
                                    
        #---Evaluate--the--individuals--with--an--invalid--fitness-------------
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self._toolbox.map(self._toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        #---Apply--Elitism-----------------------------------------------------
            pop = self._toolbox.selNSGA2(pop + offspring, self._popSize)
            hof.update(pop)
        #---Record--Statistics-------------------------------------------------
            record = mstats.compile(pop)
            logbook.record(gen=g+1, evals=len(invalid_ind),**record)
        #---Print--Statistics--------------------------------------------------
            if self._verbose:
                print(logbook.stream)
        return hof,logbook