import numpy as np
from sympy import Symbol, Matrix

class LinearGaussianBN(object):
    """
    A Bayesian network in which all variables are linear functions of their parents plus additive gaussian noise.
    The joint distribution over the variables in such a network is a multivariate gaussian. 
    """
    def __init__(self):
        self._variables = []      
        self._mean = None
        self._cov = None
        self._parents = {}
        self._weights = {}
        self._variance = {}
        self._weights_p = {}
        self._variance_p = {}
  
    
    def copy(self):
        model = LinearGaussianBN()
        model._variables = list(self._variables)
        model._mean = None if self._mean is None else self._mean[:,:] # this is a Matrix, slice = copy
        model._cov = None if self._cov is None else self._cov[:,:]
        model._parents = self._parents.copy()
        model._weights = self._weights.copy()
        model._variance = self._variance.copy()
        model._weights_p =self._weights_p.copy()
        model._variance_p = self._variance_p.copy()
        return model 

    
    def _variance_symbol(self,variable):
        """returns a symbol for the variance of variable string specified."""
        return Symbol("V_{0}".format(variable))
    
    def _weight_symbol(self,variable,parent):
        if parent is not None:
            return Symbol("w_"+variable+parent)
        return Symbol("w_"+variable+"0")

    def add_var(self,variable,parents = None, weights = None, variance = None): #TODO allow specifying weights here  - eg may want to set some to 0.
        """
        Add a variable to the network.
        - variable: a string representing the variable
        - parents: a list of the names of the parents of this variable (must already be in the network)
        - weights (optional): the weights of this variable to its parents. The first entry should be the offset
        """
        if parents is None:
            parents = []
        for p in parents:
            if p not in self._variables:
                raise ValueError("Parent {p} is not a variable in the network".format(p = p))
        if variable in self._variables:
            raise ValueError("Duplicate variable name, variable {v} already exists in this network".format(v = variable))
        
        if weights is None:
            beta = [self._weight_symbol(variable,v) if v in parents else 0 for v in self._variables]
            mu = self._weight_symbol(variable,None)
            self._weights[variable] = Matrix([mu] + [self._weight_symbol(variable,v) for v in parents]) # order is with respect to parents as specified (not covariance matrix index)

        else:
            if len(weights) != len(parents) + 1:
                raise ValueError("""The vector of weights has length {0} but should be of length {1}, 
                                     (offset, weight_parent_1, weight_parent_2, ...,weight_parent_n)""".format(len(weights),len(parents)+1))
            symbolic_weights = [Symbol(w) if isinstance(w,str) else w for w in weights]
            symbol_dict = dict(zip(parents,symbolic_weights[1:]))
            beta = [symbol_dict[v] if v in parents else 0 for v in self.variables]
            mu = Symbol(weights[0]) if isinstance(weights[0],str) else weights[0]
            self._weights[variable] = Matrix(symbolic_weights)
        
        if variance is None:
            variance = self._variance_symbol(variable)
        
        self._variance[variable] = variance
        v = variance   
            
        self._parents[variable] = parents        
        
        
        if len(beta) > 0:
            beta = Matrix(beta)
            
            mu +=(beta.T*self._mean)[0,0]
            cv = self._cov*beta # covariance of this variable with previous variables
            v += (beta.T*cv)[0,0] # variance of this variable (unconditional)

            new_vals = Matrix([cv,[v]])
            rows,cols = self._cov.shape
            self._cov = self._cov.row_insert(rows,cv.T)
            self._cov = self._cov.col_insert(cols,new_vals)
            self._mean = Matrix([self._mean,[mu]])

        else: # first time round - everything is None
            self._cov = Matrix([v])
            self._mean = Matrix([mu])
            
        self._variables.append(variable)
    
    @property
    def mu(self):
        return self._mean
    
    @property
    def cov(self):
        """the covariance matrix of the multivariate gaussian corresponding to the network"""
        return self._cov
        
    
    @property
    def variables(self):
        """The list of variables in the order coresponding to their position in the mean vector/covariance matrix"""
        return self._variables
    
    @property
    def information_matrix(self):
        """The inverse of the covariance matrix"""
        return self._cov.inv()
    
    def index(self,variables):
        """returns the indexes of the specified variables in the mean vector/covariance matrix"""
        return [self._variables.index(v) for v in variables]
        
    def marginal(self,variables):
        """ returns a new network with all but the specified variables marginalized out """
        indx = self.index(variables)
        mu = self.mu.extract(indx,[-1]) 
        cov = self.cov.extract(indx,indx)
        marginalized = self.copy()
        marginalized._variables = variables
        marginalized._mean = mu
        marginalized._cov = cov
        return marginalized

    def regression_matrix(self,condition_on):
        b_indx = self.index(condition_on)
        a_vars = [v for v in self._variables if v not in condition_on]
        a_indx = self.index(a_vars)
        
       
        cov_ab = self.cov.extract(a_indx,b_indx)
        cov_bb_inv = self.cov.extract(b_indx,b_indx).inv()
        return cov_ab*cov_bb_inv

    def observe(self,variables,values):
        """ network where we condition on variables equaling some values """
        sym_values = [Symbol(val) if isinstance(val,str) else val for val in values]
        values = Matrix(sym_values)
        # partition variables into A (not conditioned on), and B (conditioned on)
        a_vars = [v for v in self._variables if v not in variables]
        a_indx = self.index(a_vars)
        b_indx = self.index(variables)
        cov_aa = self.cov.extract(a_indx,a_indx)
        cov_ab = self.cov.extract(a_indx,b_indx)
        cov_bb = self.cov.extract(b_indx,b_indx)
        cov_bb_inv = cov_bb.inv()
        mu_a = self.mu.extract(a_indx,[-1])
        mu_b = self.mu.extract(b_indx,[-1])
        
        reg_matrix = cov_ab*cov_bb_inv
        
        mu =  (mu_a+ reg_matrix*(values - mu_b)) #.applyfunc(lambda x:x.simplify())
        cov = (cov_aa - reg_matrix*cov_ab.T) #.applyfunc(lambda x:x.simplify())
        
        observed = self.copy()
        observed._variables = a_vars
        observed._mean = mu
        observed._cov = cov
        return observed
    
    def do(self,variables,values):
        """ network after intervening to set specified variables to values """
        intervened = LinearGaussianBN()
        sym_values = [Symbol(val) if isinstance(val,str) else val for val in values]
        var_to_value = dict(zip(variables,sym_values))
        for variable in self.variables:
            if variable not in variables:
                new_parents = []
                new_weights = [self._weights[variable][0]]
                for indx, p in enumerate(self.parents(variable)):
                    w = self._weights[variable][indx+1]
                    if p not in variables:
                        new_parents.append(p)
                        new_weights.append(w)
                    else:
                        new_weights[0]+=w*var_to_value[p]
                        
                intervened.add_var(variable,new_parents,new_weights) 
        
        return intervened
       
    
    def parents(self,variable):
        return self._parents[variable]
        
    def set_var_params(self,variable,weights,variance):
        """ 
        Set numerical values for the weights and variance of the specified variable.
        first weight is assumed to be constant. 
        Other weights are with respect to parents in the same order as they were originally specified
        (or as returned by parents(variable))
        """
        if variable not in self._weights:
            raise ValueError("Variable {0} not in network".format(variable))

        expected_num_weights = len(self._weights[variable])
        if len(weights) != expected_num_weights:
            raise ValueError("Weights should contain {0} values (number of parents + 1) but contained {1} values").format(expected_num_weights,len(weights))
        self._weights_p[variable] = weights
        self._variance_p[variable] = variance
        
    def set_params(self,variable_param_dict):
        """Set numerical values for all variables based on dictionary."""
        for variable,(weights,variance) in variable_param_dict.items():
            self.set_var_params(variable,weights,variance)
        return self
    
    def parameterized_mean_cov(self):
        """Returns the mean and covariance after substituting parameters. All variables must have parameters set."""
        substitutions = []
        for variable,sym_weights in self._weights.items():
            numeric_weights = self._weights_p[variable]
            substitutions.extend(list(zip(sym_weights,numeric_weights)))
            substitutions.append((self._variance[variable],self._variance_p[variable]))
        cov = self._cov.subs(substitutions)
        mu = self._mean.subs(substitutions)
        return mu,cov
    
    def sample(self,n):
        """sample from the underlying multivariate gaussain. All variables must have parameters set."""
        unparameterized = [v for v in self._weights.keys() if v not in self._weights_p.keys()]
        if len(unparameterized) > 0:
            raise ValueError("The following variables must be numerically parameterized before sampling: {0}".format(unparameterized))
        mu,cov = self.parameterized_mean_cov()
        cov = np.asarray(cov).astype(np.float64)
        mu = np.asarray(mu).astype(np.float64)
        return np.random.multivariate_normal(mu.ravel(),cov,size=n)

