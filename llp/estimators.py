# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 16:23:50 2016

@author: finn
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy.special import expit
import numpy as np
from simulator import Linear
import matplotlib.pyplot as plt
from lmm.mypersonality.laplacian_mean_map import LaplacianMeanMapCV



class Discretizer(object):

    def fit(self,x,bins = None):
        if bins is not None:
            _, boundaries = np.histogram(x,bins)

        else:
            _, boundaries = np.histogram(x)
        self.boundaries = boundaries
        self.nbins = len(self.boundaries)+1
        return self

    def transform(self,x):
        return np.digitize(x,self.boundaries)


class StandardEstimator(object):
    label = "standard"
    def __init__(self):
        self.model_t = LogisticRegression()
        self.model_c = LogisticRegression()

    def fit(self,x_treat,x_control,y_treat,y_control):
        self.model_t.fit(x_treat,y_treat)
        self.model_c.fit(x_control,y_control)
        self.w_t = np.hstack((self.model_t.coef_[0],self.model_t.intercept_))
        self.w_c = np.hstack((self.model_c.coef_[0],self.model_c.intercept_))

    def predict(self,x):
        return self.model_t.predict_proba(x)[:,1] - self.model_c.predict_proba(x)[:,1]

class SoftLogisticRegression(object):
    """
    Logistic regression that takes class probabilities rather than labels as inputs.
    Minimizes the cross entropy.
    """

    def fit(self,x,p):
        """ p is a vector of the probability y_i = 1 """
        assert x.ndim == 2, "x should be a 2d numpy array"
        assert p.ndim == 1, "p should be a 1d numpy array, not ndims:{0}, shape:{1}".format(p.ndim,p.shape)
        assert x.shape[0] == p.shape[0], "number of elements in p: {0}, must match number of rows of x: {1}".format(p.shape[0],x.shape[0])
        x2 = np.hstack((x,np.ones((x.shape[0],1))))
        w0 = np.random.random((x2.shape[1],1))
        optimal = minimize(self._loss,w0,args = (x2,p),jac = self._jac,options = {"disp":False})
        self.w = optimal.x
        self.coef_ = optimal.x[0:-1]
        self.intercept_ = optimal.x[-1]

    def predict_proba(self,x):
        x2 = np.hstack((x,np.ones((x.shape[0],1))))
        return expit(x2.dot(self.w))

    def _loss(self,w,x,p):
        q = expit(x.dot(w))
        l_i = -(p*np.log(q) + (1.0-p)*np.log(1-q))
        return l_i.sum()

    def _jac(self,w,x,p):
        """ the gradient of the loss function """
        xw = x.dot(w)
        q = expit(xw)
        return -(p*q*np.exp(-xw)-(1-p)*(1-q)*np.exp(xw)).dot(x)

    @staticmethod
    def soft_equiv(y):
        """ p is 1 if y is 1, 0 otherwise """
        p = y.copy()
        p[y != 1] = 0
        return p


class EMEstimator(object):
    label = "EM"

    def fit(self,x_treat,x_control,y_treat,y_control, tol = 1e-2, max_iter = 500):
        model_t = SoftLogisticRegression()
        model_c = SoftLogisticRegression()

        model_t.fit(x_treat,SoftLogisticRegression.soft_equiv(y_treat))
        model_c.fit(x_control,SoftLogisticRegression.soft_equiv(y_control))


        model_t.predict_proba(x_control)
        model_c.predict_proba(x_treat)




    def fit2(self,x_treat,x_control,y_treat,y_control, tol = 1e-2, max_iter = 500):
        base_estimator = StandardEstimator()
        base_estimator.fit(x_treat,x_control,y_treat,y_control)

        model_t = SoftLogisticRegression()
        model_c = SoftLogisticRegression()


        p_ct = base_estimator.model_t.predict_proba(x_control)[:,1] # probability p(y=1|x,a=1) for x in X_c
        p_tc = base_estimator.model_c.predict_proba(x_treat)[:,1]   # probability p(y=1|x,a=0) for x in X_t


        p_cc = SoftLogisticRegression.soft_equiv(y_control) # note these are fixed and not updated
        p_tt = SoftLogisticRegression.soft_equiv(y_treat)

        x = np.vstack((x_control,x_treat))
        w_c = base_estimator.w_c
        w_t = base_estimator.w_t


        i = 0
        while (i < max_iter):

            p_c = np.concatenate((p_cc,p_tc)) #results for control, treated if a = 0
            p_t = np.concatenate((p_ct,p_tt)) # results for control,treated if a = 1

            model_c.fit(x,p_c)
            model_t.fit(x,p_t)

            delta_c = mean_squared_error(model_c.w, w_c)
            delta_t = mean_squared_error(model_t.w , w_t)

            if  delta_c < tol and delta_t < tol:
                break

            #print w_c,delta_c,delta_t

            w_c = model_c.w
            w_t = model_t.w

            p_ct = model_t.predict_proba(x_control)
            p_tc = model_c.predict_proba(x_treat)

            i += 1

        self.model_t = model_t
        self.model_c = model_c
        if i == max_iter:
            message = "failed to coverge to requested tolerance within max_iter={0} iterations.\n deltas: {5}, {6}\n{1}\n{2}\n\n{3}\n{4}".format(max_iter,w_c,model_c.w,w_t,model_t.w,delta_c,delta_t)
            raise Exception(message)


    def predict(self,x):
        return self.model_t.predict_proba(x) - self.model_c.predict_proba(x)





class CausalLLMEstimator(object):
    def __init__(self,alpha_grid = np.logspace(-2, 2, 5),gamma_grid=np.logspace(-2, 2, 5),sigma_grid = 2. ** np.linspace(-2, 2, 5)):
        self.alpha_grid = alpha_grid
        self.gamma_grid = gamma_grid
        self.sigma_grid = sigma_grid

    def fit(self,x_treat,x_control,y_treat,y_control):
        x = np.vstack((x_treat,x_control))
        y = np.concatenate((y_treat,y_control))
        a = len(x_treat)
        bags = self._bag(x,y,a)
        train,validate = self._train_validate_split(bags)
        X,labels,bag_ids = self._row_per_instance_form(train,x)
        X_val,labels_val,bag_ids_val = self._row_per_instance_form(validate,x)

        lmm = LaplacianMeanMapCV(alpha_grid=self.alpha_grid, gamma_grid=self.gamma_grid,
                         sigma_grid=self.sigma_grid, cv_val='proportions',
                         verbose=0, n_jobs=4)

        # TODO figure out how to modify LMM to handle differences in proportions
        # TODO figure out how validation/cross-validation works inside this llm implementation
        self.lmm = lmm.fit(X, labels, bag_ids, X_val, labels_val, bag_ids_val)


    def _bag(self,x,y,a):
        """ returns a list of bags where each bag is (instances,cte,bag_id) tuple """
        max_bag_id = 0
        all_bags = []
        for feature in range(x.shape[-1]):
            category = Discretizer().fit(x[:,feature],bins=3).transform(x[:,feature]) # each unique value of category is a new bag
            new_bag_ids = max_bag_id + category

            bags = []
            for i,bag_id in enumerate(np.unique(new_bag_ids)):
                instances_in_bag = np.where(new_bag_ids == bag_id)[0] # indexes of the instances that are in this bag
                treated_instances = instances_in_bag[np.where(instances_in_bag < a)[0]]
                control_instances = instances_in_bag[np.where(instances_in_bag >= a)[0]]
                valid = (len(control_instances > 1) and len(treated_instances) > 1)
                bag = (control_instances,treated_instances,valid)
                bags.append(bag)

            # merge any bags that don't have at least 1 treated & 1 control into adjacent bag
            valid = [bag[-1] for bag in bags]
            while not np.alltrue(valid):
                invalid_bag = valid.index(False)
                if invalid_bag + 1 < len(bags):
                    merge_with = invalid_bag + 1
                elif invalid_bag > 0:
                    merge_with = invalid_bag -1
                else:
                    raise ValueError("no candidate bag to merge with")

                c1,t1,v1 = bags[merge_with]
                c2,t2,v2 = bags[invalid_bag]
                control, target = np.concatenate((c1,c2)), np.concatenate((t1,t2))
                v = (len(control) > 1 and len(target) > 1)
                bags[merge_with] = (control,target,v)
                bags.pop(invalid_bag)
                valid = [bag[-1] for bag in bags]

            bags = [(np.concatenate((c,t)),y[t].mean() - y[c].mean(),i +max_bag_id) for i,(c,t,v) in enumerate(bags)]
            max_bag_id += len(bags)
            all_bags.extend(bags)
        return all_bags


    def _train_validate_split(self,bags,prop_train = .8):
        """ split bags into two groups """
        # TODO doesn't take into account that the bags are different sizes
        # TODO note instances are shared accross train and validate as each instance is in many bags. Maybe I should split earlier
        n_bags = len(bags)
        indx = np.arange(n_bags)
        np.random.shuffle(indx)
        split = int(n_bags*.8)
        indx_train = indx[0:split]
        indx_val = indx[split:]
        train = [bags[i] for i in indx_train]
        validate = [bags[i] for i in indx_val]
        return train,validate

    def _row_per_instance_form(self,bags,x):
        """ expand bags into one row per instance with labels and bag_ids repeated. """
        instances = np.empty(0)
        bag_ids = np.empty(0)
        labels = np.empty(0)
        X = np.empty((0,x.shape[-1]))
        for (bag_instances,ce,bag_id) in bags:
            instances = np.concatenate((instances,bag_instances))
            bag_ids = np.concatenate((bag_ids,np.repeat(bag_id,len(bag_instances))))
            labels = np.concatenate((labels,np.repeat(ce,len(bag_instances))))
            X = np.vstack((X,x[bag_instances,:]))
        return X,labels,bag_ids




def plot_single_feature(sample_size):
    simulator = Linear(1,c=0)
    x_treat,x_control,y_treat,y_control = simulator.sample(sample_size)

    estimator = StandardEstimator()
    estimator.fit(x_treat,x_control,y_treat,y_control)

    estimator2 = EMEstimator()
    estimator2.fit(x_treat,x_control,y_treat,y_control,tol=1e-3)

    x_vals = np.linspace(-4,4,num = 50).reshape(50,1)

    plt.plot(x_vals,simulator.causal_effect(x_vals),label = "true")
    plt.plot(x_vals,estimator.predict(x_vals),label = estimator.label)
    plt.plot(x_vals,estimator2.predict(x_vals),label = estimator2.label)
    plt.legend(loc = "lower right")


def compare_errors(n_features, nrows ,simulations, c_vals = [0,.5,1],test_samples = 1000):
    scores = np.zeros((2,len(c_vals),simulations))
    for c_indx,c in enumerate(c_vals):
        for s in range(simulations):
            model = Linear(n_features, c = c)
            print (c,s,model.w_c,model.w_t)
            x_treat,x_control,y_treat,y_control = model.sample(nrows)
            estimator = StandardEstimator()
            estimator.fit(x_treat,x_control,y_treat,y_control)
            estimator2 = EMEstimator()
            estimator2.fit(x_treat,x_control,y_treat,y_control)
            x_test = model.sample_x(test_samples)
            ce1 = estimator.predict(x_test)
            ce2 = estimator2.predict(x_test)
            ce_true = model.causal_effect(x_test)
            scores[0,c_indx,s] = mean_squared_error(ce_true,ce1)
            scores[1,c_indx,s] = mean_squared_error(ce_true,ce2)
    return scores


class EMEstimator2(object):
    """ Implementation assuming no defiers as described by Tiberio """
    def fit(self,x_treat,x_control,y_treat,y_control, tol = 1e-2, max_iter = 500):
         m_treat = LogisticRegression().fit(x_treat,y_treat)
         m_control = LogisticRegression(x_control,y_control)

         xt0,xc1,xt1,xc0 = x_treat[y_treat == 0],x_control[y_control == 1],x_treat[y_treat == 1],x_control[y_control == 0]

         x = np.vstack((
             xt0, # have label N
             xc1, # have label A
             xt1, # have label A with weight w_A
             xt1, # have label C with weight w_C
             xc0, # have label N with weight w_N
             xc0)) # have label C with weight w_C'

         blocks = [xt0.shape[0],xc1.shape[0],xt1.shape[0],xt1.shape[0],xc0.shape[0],xc0.shape[0]]

         w_A = np.true_divide(m_control.predict_proba(xt1),m_treat.predict_proba(xt1))
         w_N = np.true_divide(1-m_treat.predict_proba(xc0),1-m_control.predict_proba(xc0))

         sample_weights = np.concatenate((np.ones(blocks[0]+blocks[1]),w_A,1-w_A,w_N,1-w_N))

         y = np.repeat([0,1,1,2,0,2],blocks) #N=0,A=1,C=2

         epsilon = np.Inf

         model = LogisticRegression()
         while (epsilon > tol):
             model.fit(x,y,sample_weight = sample_weights)
             p_xt1 = model.predict_proba(xt1)
             p_xc0 = model.predict_proba(xc0)

             #check for convergence

             w_A = p_xt1[1]
             w_C = p_xt1[2]
             w_N = p_xc0[0]
             w_C2 = p_xc0[2]
             sample_weights = np.concatenate((np.ones(blocks[0]+blocks[1]),w_A,w_C,w_N,w_C2))







    def predict_causal_effect(self,x):
        return self.model_t.predict_proba(x)[:,1] - self.model_c.predict_proba(x)[:,1]



if __name__ == "__main__":
    print(5)






