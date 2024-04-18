"""
Class DynamicLifetimeModel and supporting functions for comprehensive lifetime modeling in dynamic stock models

@authors: Kamila Krych, Norwegian University of Science and Technology (NTNU), Trondheim, Norway.
Built on previous work done by Stefan Pauliuk et al. in the dynamic_stock_model and Fernando Aguilar Lopez et al. in the product_component_model. 

standard abbreviation: DLM or dlm 

Repository for this class, documentation, and tutorials: 
https://github.com/NTNU-IndEcol/dynamic_lifetime_model

"""

import numpy as np
import scipy.stats
import math


class DynamicLifetimeModel:
    """
    Class containing a dynamic stock model with dynamic lifetime, i.e., where the lifetime can vary by time (t) or cohort (c).

    Attributes
    ----------
    t : Series of years or other time intervals
    i : Discrete time series of inflow to stock (c)
    o : Discrete time series of outflow from stock (t)
    o_c :Discrete time series of outflow from stock, by cohort (t,c)
    s : Discrete time series for stock, total (t)
    s_c : Discrete time series for stock, by cohort (t,c)
    ds : Discrete time series for stock change (t)
    lt : lifetime distribution: dictionary with distribution type and parameters, where each parameter is of shape (t,c)
    hz: hazard function for different product cohorts (t,c)
    """

    def __init__(self, t: np.ndarray, i=None, s=None, s_c=None, o=None, o_c=None, ds=None, lt=None, hz=None):
        """
        Basic initialisation
        """
        if type(t) == list:
            t = np.array(t)
        self.t = t
        self.i = i # optional
        self.o = o # optional
        self.o_c = o_c # optional
        self.s = s # optional
        self.s_c = s_c # optional
        self.ds = ds # optional
        self.lt = lt # optional
        self.hz = hz # optional
        
    def compute_inflow_driven_model(self):
        """ 
        Computes the model given the lifetime distribution and inflows
        """
        if self.i is None:
            raise Exception('No inflows specified')
        elif np.shape(self.i) != np.shape(self.t):
            raise Exception(f'Non-compatible array shapes. Array t has shape {np.shape(self.t)}, but array i has shape {np.shape(self.i)}')
        if self.hz is None:
            self.compute_hz_from_lt_par()
        self.s_c = np.zeros((len(self.t), len(self.t))) # stock composition per year
        self.o_c = np.zeros((len(self.t), len(self.t))) # outflow compositionO
        for m in range(len(self.t)): # for each year m
            if m>0: # the initial stock is assumed to be 0
                self.o_c[m,:m] = self.s_c[m-1,:m] * self.hz[m,:m] 
                # subtract outflows of cohorts <m from the previous stock 
                self.s_c[m,:m] = self.s_c[m-1,:m] - self.o_c[m,:m]
            # Add new cohort to stock, accounting for outflows in the first year
            self.o_c[m,m] = self.i[m] * self.hz[m,m]
            self.s_c[m,m] = self.i[m] - self.o_c[m,m]
        self.o = self.o_c.sum(axis=1)
        self.s = self.s_c.sum(axis=1)
        self.compute_stock_change()
        return

    def compute_stock_change(self):
        """ Determines stock change from time series for stock. Formula: stock_change(t) = stock(t) - stock(t-1)."""
        if self.s is not None:
            self.ds = np.zeros(len(self.s))
            self.ds[0] = self.s[0]
            self.ds[1::] = np.diff(self.s)
            return self.ds
        else:
            return None

    def compute_stock_driven_model(self): 
        """ 
        Computes the model given the lifetime distribution and the total stock. 
        Inspired by case 1 of the product_component_model by Fernando Aguilar Lopez and colleagues
        """
        if self.s is None:
            raise Exception('No stock specified')
        if self.hz is None:
            self.compute_hz_from_lt_par()
        self.s_c = np.zeros((len(self.t), len(self.t))) # stock composition per year
        self.o_c = np.zeros((len(self.t), len(self.t))) # outflow composition
        self.i = np.zeros(len(self.t)) # product inflows
        self.ds = np.concatenate(([self.s[0]], np.diff(self.s))) # stock change
        # Initializing values
        self.s_c[0,0] = self.s[0]               
        for m in range(len(self.t)):  # for each time period m
            if m>0: # the initial stock is assumed to be 0
                # Probability of any failure is calculated using product hazard function 
                self.o_c[m,:m] = self.s_c[m-1,:m] * self.hz[m,:m] 
                # subtract outflows of cohorts <m from the previous stock 
                self.s_c[m,:m] = self.s_c[m-1,:m] - self.o_c[m,:m]
                # Add new cohort to stock
                self.s_c[m,m] = self.s[m] - self.s_c[m,:m].sum()
           
            # Calculate new inflow, accounting for outflows in the first year
            self.o_c[m,m] = self.s_c[m,m] * self.hz[m,m] 
            self.i[m] = self.ds[m] + self.o_c[m,:].sum()
        self.o = self.o_c.sum(axis=1)
        return

    def add_cohort_effect(self, array, value, effect_year, ref='absolute', trans_start=None, trans_type='linear', 
                          period_interact=False, periods=None, age_interact=False, ages=None):
        """
        Adds a cohort effect to a given array. The strength of the effect is indicated by the provided value (absolute or relative). 
        The effect is fully in force at effect_year, but can be preceded by a transition period starting at trans_start. 
        The type of the transition can be 'linear' or 'logistic', indicated by trans_type. 
        If trans_start is None, then the effect takes place between effect_year-1 and effect_year.
        Period-cohort interaction can be implemented by setting period_interact as True and providing the affected periods.
        Age-cohort interaction can be implemented by setting age_interact as True and providing the affected ages.

        :par array: An array of size (t,t)
        :par value: The strength of the effect, defined as an absolute (e.g., 12) or relative value (e.g., 1.2 for an increase from 10 to 12)
        :par effect_year: The year at which the effect is fully in force
        :par ref: Indicates whether the provided value is 'relative' or 'absolute'
        :par trans_start: The start of the transition period. If None then the effect takes place between effect_year-1 and effect_year.
        :par trans_type: The type of the transition, must be set as 'linear' or 'logistic'
        :par period_interact: Boolean indicating whether period interaction should be considered. If True, the affected periods need to be provided
        :par periods: Numpy array indicating periods affected by the cohort effect
        :par age_interact: Boolean indicating whether age interaction should be considered. If True, the affected ages need to be provided
        :par ages: Numpy array indicating ages affected by the cohort effect
        :return array_new: An array of size (t,t) with the implemented cohort effect
        """
        if type(array) != np.ndarray:
            raise TypeError("Parameter 'array' can only be of type 'numpy.ndarray'.")
        if effect_year not in self.t:
            raise ValueError("The parameter effect_year must be within the limits indicated by the time vector t")
        
        # find the transition start and end points
        if trans_start is None:
            start = math.ceil(effect_year)-1
        else:
            if trans_start not in self.t:
                raise ValueError("The parameter trans_start must be within the limits indicated by the time vector t")
            start = math.floor(trans_start)
        stop = math.ceil(effect_year)
        start_idx = np.where(self.t==start)[0][0]
        stop_idx = np.where(self.t==stop)[0][0]
        t_trans = np.arange(start_idx+1, stop_idx)

        # potential interactions with other dimensions
        if period_interact:
            if type(periods) == np.ndarray:
                period_idx = np.where(np.in1d(self.t,list(periods))==True)[0]
            elif periods == None:
                raise TypeError("Period interaction was requested but the parameter 'periods' was not provided.")
            else:
                raise TypeError("Parameter 'periods' can only be of type 'numpy.ndarray'.")
        else:
            period_idx = range(start_idx, len(self.t)) # for each period
        if age_interact:
            if type(ages) == np.ndarray:
                age_matrix = np.array([self.t]).T - np.array([self.t])
                age_matrix[np.triu_indices(age_matrix.shape[0])] = 0
                age_matrix_bool = np.isin(age_matrix,list(ages))
            elif ages == None:
                raise TypeError("Age interaction was requested but the parameter 'ages' was not provided.")
            else:
                raise TypeError("Parameter 'ages' can only be of type 'numpy.ndarray'.")
        else:
            age_matrix_bool = np.ones_like(array,dtype=bool) # for each age
        
        # warn if this cohort effect erases previously applied cohort effects
        diff = np.diff(np.concatenate((np.reshape(np.zeros_like(self.t),(len(self.t),1)), array),axis=1), axis=1)
        diff = np.tril(diff)
        check = np.zeros_like(array)
        check[period_idx,start_idx+1:] = diff[period_idx,start_idx+1:] # select periods and cohorts
        check = np.where(age_matrix_bool, check, np.zeros_like(array)) # select ages
        if np.any(check):
            print(f"Warning! This cohort effect erases previously applied cohort effects starting after cohort effect_year={effect_year}. To avoid this behavior, add effects chronologically.")

        # calculation of the new values
        array_new = np.array(array)
        for n in period_idx: # for each period
            # select the value to start with
            value_1 = array[n,start_idx]
            # select the value to end with
            if ref=='relative':
                value_2=value_1*value
            elif ref=='absolute':
                value_2 = value
            else:
                raise ValueError("Parameter 'how' can only take values 'relative' or 'absolute'.")
            # find transition values
            if trans_type == 'linear':
                v_trans = value_2+(stop_idx-t_trans)/(stop_idx-start_idx)*(value_1-value_2)
            elif trans_type == 'logistic': 
                ti_idx = (stop_idx+start_idx)/2
                a = 8/(stop-start-1)
                v_trans = (value_2-value_1) / (1 + np.exp(-a * (t_trans - ti_idx))) + value_1
            else:
                raise ValueError("Parameter 'how' can only be set as 'linear' or 'logistic'.")
            # set the transition values
            array_new[n, start_idx+1:stop_idx] = v_trans
            array_new[n, stop_idx:] = value_2
        array_new = np.where(age_matrix_bool, array_new, array)
        return array_new

    def add_period_effect(self, array, value, effect_year, ref='absolute', trans_start=None, trans_type='linear', 
                          coh_interact=False, cohorts=None, age_interact=False, ages=None):
        """
        Adds a period effect to a given array. The strength of the effect is indicated by the provided value (absolute or relative). 
        The effect is fully in force at effect_year, but can be preceded by a transition period starting at trans_start. 
        The type of the transition can be 'linear' or 'logistic', indicated by trans_type. 
        If trans_start is None, then the effect takes place between effect_year-1 and effect_year.
        Period-cohort interaction can be implemented by setting coh_interact as True and providing the affected cohorts.
        Age-cohort interaction can be implemented by setting age_interact as True and providing the affected ages.

        :par array: An array of size (t,t)
        :par value: The strength of the effect, defined as an absolute (e.g., 12) or relative value (e.g., 1.2 for an increase from 10 to 12)
        :par effect_year: The year at which the effect is fully in force
        :par ref: Indicates whether the provided value is 'relative' or 'absolute'
        :par trans_start: The start of the transition period. If None then the effect takes place between effect_year-1 and effect_year.
        :par trans_type: The type of the transition, must be set as 'linear' or 'logistic'
        :par coh_interact: Boolean indicating whether cohort interaction should be considered. If True, the affected cohorts need to be provided
        :par cohorts: Numpy array indicating cohorts affected by the period effect
        :par age_interact: Boolean indicating whether age interaction should be considered. If True, the affected ages need to be provided
        :par ages: Numpy array indicating ages affected by the period effect
        :return array_new: An array of size (t,t) with the implemented period effect
        """
        if type(array) != np.ndarray:
            raise TypeError("Parameter 'array' can only be of type 'numpy.ndarray'.")
        if effect_year not in self.t:
            raise ValueError("The parameter effect_year must be within the limits indicated by the time vector t")

        # find the transition start and end points
        if trans_start is None:
            start = math.ceil(effect_year)-1
        else:
            if trans_start not in self.t:
                raise ValueError("The parameter trans_start must be within the limits indicated by the time vector t")
            start = math.floor(trans_start)
        stop = math.ceil(effect_year)
        start_idx = np.where(self.t==start)[0][0]
        stop_idx = np.where(self.t==stop)[0][0]
        t_trans = np.arange(start_idx+1, stop_idx)

        # potential interactions with other dimensions
        if coh_interact:
            if type(cohorts) == np.ndarray:
                cohort_idx = np.where(np.in1d(self.t,list(cohorts))==True)[0]
            elif cohorts == None:
                raise TypeError("Cohort interaction was requested but the parameter'cohorts' was not provided.")
            else:
                raise TypeError("Parameter 'cohorts' can only be of type 'numpy.ndarray'.")
        else:
            cohort_idx = range(len(self.t)) # for each cohort
        if age_interact:
            if type(ages) == np.ndarray:
                age_matrix = np.array([self.t]).T - np.array([self.t])
                age_matrix[np.triu_indices(age_matrix.shape[0])] = 0
                age_matrix_bool = np.isin(age_matrix,list(ages))
            elif ages == None:
                raise TypeError("Age interaction was requested but the parameter 'ages' was not provided.")
            else:
                raise TypeError("Parameter 'ages' can only be of type 'numpy.ndarray'.")
        else:
            age_matrix_bool = np.ones_like(array,dtype=bool) # for each age
        age_matrix_bool = np.tril(age_matrix_bool,0) # to make sure we don't change values for years lower than cohort (m<n)

        # warn if this period effect erases previously applied period effects
        diff = np.diff(np.concatenate((np.reshape(np.zeros_like(self.t),(1,len(self.t))), array),axis=0), axis=0)
        diff = np.tril(diff,-1)
        check = np.zeros_like(array)
        check[start_idx+1:, cohort_idx] = diff[start_idx+1:, cohort_idx]  # select periods and cohorts
        check = np.where(age_matrix_bool, check, np.zeros_like(array)) # select ages
        if np.any(check):
            print(f"Warning! This period effect erases previously applied period effects starting after period effect_year={effect_year}. To avoid this behavior, add effects chronologically.")

        # calculation of the new values
        array_new = np.array(array)
        for n in cohort_idx: # for each cohort
            # select the value to start with
            if start_idx>n:
                value_1 = array[start_idx,n]
            else: # use the last cohort before the effect starts for the cohorts entering after the effect is already in place
                value_1 = array[start_idx,start_idx]
            # select the value to end with
            if ref=='relative':
                value_2=value_1*value
            elif ref=='absolute':
                value_2 = value
            else:
                raise Exception("Parameter 'how' can only take values 'relative' or 'absolute'.")
            # find transition values
            if trans_type == 'linear':
                v_trans = value_2+(stop_idx-t_trans)/(stop_idx-start_idx)*(value_1-value_2)
            elif trans_type == 'logistic': 
                ti_idx = (stop_idx+start_idx)/2
                a = 8/(stop-start-1)
                v_trans = (value_2-value_1) / (1 + np.exp(-a * (t_trans - ti_idx))) + value_1
            else:
                raise ValueError("Parameter 'how' can only be set as 'linear' or 'logistic'.")
            # set the transition values
            array_new[start_idx+1:stop_idx,n] = v_trans
            array_new[stop_idx:,n] = value_2
        array_new = np.where(age_matrix_bool, array_new, array)
        return array_new

    def calculate_age_stock(self, t=None, s_c=None, i=None, scale_by_inflow=True):
        """
        Calculates the mean age of stocks (measured at the end of each year)
        :par t:   An array describing the time vector t
        :par s_c: An array of size (t,c) with stocks by cohort
        :par i:   An array of size (t) with inflows. Used to scale the values by inflows in respective years
        :par scale_by_inflow: A boolean indicating if scaling by inflow should be performed. 
        :return age: An array of size (t) with the mean age in each year
        """
        if i is None:
            i = self.i
        if t is None:
            t = self.t
        if s_c is None:
            s_c = self.s_c
        if not np.shape(s_c) == (len(t), len(t)):
            raise Exception(f"The array s_c has size {np.shape(s_c)}, while it should have the size {(len(t), len(t))}.")
        if not (np.shape(i) == (len(t)) or np.shape(i) == (len(t),)):
            raise Exception(f"The array i has size {np.shape(i)}, while it should have the size ({len(t)}) or ({len(t)},) .")
        age_matrix = np.array([t]).T - np.array([t])+1
        age_matrix[np.triu_indices(age_matrix.shape[0])] = 0
        np.fill_diagonal(age_matrix, 1)
        if scale_by_inflow:
            array = np.einsum('tc,c->tc', s_c, reciprocal(i))
        else:
            array = s_c
        shares = np.einsum('tc,t->tc',array, reciprocal(array.sum(axis=1))) # calculate the distribution of cohorts in each year (shares of the total)
        age = np.einsum('tc,tc->t',shares,age_matrix)
        return age
    
    def calculate_age_outflow(self, t=None, o_c=None, i=None, scale_by_inflow=True):
        """
        Calculates the mean age of outflows (during the entire year)
        :par t:   An array describing the time vector t
        :par o_c: An array of size (t,c) with outflows by cohort
        :par i:   An array of size (t) with inflows. Used to scale the values by inflows in respective years
        :par scale_by_inflow: A boolean indicating if scaling by inflow should be performed. 
        :return age: An array of size (t) with the mean age in each year
        """
        if i is None:
            i = self.i
        if t is None:
            t = self.t
        if o_c is None:
            o_c = self.o_c
        if not np.shape(o_c) == (len(t), len(t)):
            raise Exception(f"The array s_c has size {np.shape(o_c)}, while it should have the size {(len(t), len(t))}.")
        if not (np.shape(i) == (len(t)) or np.shape(i) == (len(t),)):
            raise Exception(f"The array i has size {np.shape(i)}, while it should have the size ({len(t)}) or ({len(t)},) .")
        age_matrix = np.array([t]).T - np.array([t])
        age_matrix[np.triu_indices(age_matrix.shape[0])] = 0
        if scale_by_inflow:
            array = np.einsum('tc,c->tc', o_c, reciprocal(i))
        else:
            array = o_c
        shares = np.einsum('tc,t->tc',array, reciprocal(array.sum(axis=1))) # calculate the distribution of cohorts in each year (shares of the total)
        age = np.einsum('tc,tc->t',shares,age_matrix)
        return age

    def create_2Darray(self, value):
        array = create_2Darray(value, self.t)
        return array
    
    def compute_hz_from_sf(self, sf, set_hz=True):
        hz = compute_hz_from_sf(sf)
        if set_hz:
            self.hz = hz
        return hz
    
    def compute_sf_from_hz(self, hz):
        sf = compute_sf_from_hz(hz)
        return sf
    
    def compute_pdf_from_sf(self, sf):
        pdf = compute_pdf_from_sf(sf)
        return pdf
    
    def compute_pdf_from_hz(self, hz):
        pdf = compute_pdf_from_hz(hz)
        return pdf
    
    def combine_multiple_hz(self, hz_list, shares, set_hz=True):
        hz = combine_multiple_hz(hz_list, shares)
        if set_hz:
            self.hz = hz
        return hz

    def compute_hz_from_lt_par(self, lt=None, set_hz=True):
        if lt is None:
            if self.lt is None:
                raise Exception('No product lifetime specified')
            else:
                lt = self.lt
        hz = compute_hz_from_lt_par(lt, self.t)
        if set_hz:
            self.hz = hz
        return hz


def create_2Darray(value, t):
    """
    Creates an array sized (t,t) such that the lower triangle (incl. the diagonal) is filled with provided values and the upper triangle 
    (except the diagonal) is set to zero. If the value is float or int, the array will be filled with a constant. If the value is an array
    of size (t,) or (1,t), the array will be filled cohort-wise. If the value is an array of size (t,1), the array will be filled period-wise.
    :par value: A value to fill the lower triangle
    :return array: An array of size (t,t)
    """
    if type(value) == float:
        pass
    elif type(value) == int:
        value = float(value)
    elif type(value) == np.ndarray:
        if np.shape(value) == (len(t),) or np.shape(value) == (1,len(t)):
            pass # cohort-wise
        elif np.shape(value) == (len(t),1):
            pass # period-wise
        elif np.shape(value) == (1,):
            pass # constant
        else:
            raise TypeError(f"The given value is a numpy.ndarray of shape {np.shape(value)} but only shapes {(1,)}, {(len(t),)}, {(1,len(t))} and {(len(t),1)} are accepted.")
    else:
        raise TypeError("The given value should be of type int, float or numpy.ndarray.")
    array = np.full((len(t),len(t)), value, dtype=float)
    array = np.tril(array, 0)
    return array


def reciprocal(array):
    """
    Calculates the element-wise reciprocal of an array while ignoring zero values (to avoid Nan values)
    :par array: A numpy array
    :return new_array: A reciprocal array with zero values left unchanged
    """
    mask = array != 0
    with np.errstate(divide='ignore'):
        new_array = 1 / array
    new_array[mask == 0] = 0
    return new_array


def compute_hz_from_sf(sf):
    """
    Calculates the hazard function hz from a survival function sf. 
    sf : 1D or 2D numpy array with the survival function
    """
    if np.any(np.tril(np.diff(sf, axis=0))>0):
        raise Exception('The provided survival function is incorrect, an increase in the values was detected.')
    if sf.ndim == 2:
        hz = np.zeros_like(sf)
        for n in range(sf.shape[0]): # for each cohort n
            hz[n,n] = 1-sf[n,n]
            for m in range(n,sf.shape[0]-1): # for each age m
                if sf[m,n] != 0:
                    hz[m+1,n] = (sf[m,n] - sf[m+1,n]) / sf[m,n]
                else:
                    hz[m+1,n] = 1
        return hz
    elif sf.ndim == 1:
        hz = np.zeros_like(sf)
        hz[0] = 1-sf[0]
        for m in range(sf.shape[0]-1): # for each age m
            if sf[m] != 0:
                hz[m+1] = (sf[m] - sf[m+1]) / sf[m]
            else:
                hz[m+1] = 1
        return hz
    else:
        raise Exception('The survival function should be a 1D or 2D numpy array')


def compute_sf_from_hz(hz):
    """
    Calculates the survival function sf from a hazard function hz. 
    hz : 1D or 2D numpy array with the hazard function
    """
    if hz.ndim == 2:
        sf = np.zeros_like(hz)
        for n in range(hz.shape[0]): # for each cohort n
            sf[n,n] = 1-hz[n,n]
            for m in range(n,hz.shape[0]-1): # for each age m
                sf[m+1,n] = sf[m,n]*(1-hz[m+1,n])
        return sf
    elif hz.ndim == 1:
        sf = np.zeros_like(hz)
        sf[0] = 1-hz[0]
        for m in range(hz.shape[0]-1): # for each age m
            sf[m+1] = sf[m]*(1-hz[m+1])
        return sf
    else:
        raise Exception('The hazard function should be a 1D or 2D numpy array')


def compute_pdf_from_sf(sf):
    """
    Calculates the probability function pdf from a hazard function sf. 
    sf : 1D or 2D numpy array with the survival function
    """
    if np.any(np.tril(np.diff(sf, axis=0))>0):
        raise Exception('The provided survival function is incorrect, an increase in the values was detected.')
    if sf.ndim == 2:
        pdf = np.zeros_like(sf)
        for n in range(sf.shape[0]): # for each cohort n
            pdf[n,n] = 1-sf[n,n]
            pdf[n+1:,n] = -np.diff(sf[n:,n])
        return pdf
    elif sf.ndim == 1:
        pdf = np.zeros_like(sf)
        pdf[0] = 1-sf[0]
        pdf[1:] = -np.diff(sf)
        return pdf
    else:
        raise Exception('The survival function should be a 1D or 2D numpy array')


def compute_pdf_from_hz(hz):
    """
    Calculates the probability function pdf from a hazard function hz. 
    hz : 1D or 2D numpy array with the hazard function
    """
    sf = compute_sf_from_hz(hz)
    pdf = compute_pdf_from_sf(sf)
    return pdf


def combine_multiple_hz(hz_list, shares):
    """
    Combines multiple hazard functions hz into one. Each function has a weight, all weights should add up to 1. 
    :par hz_list : list with hazard functions
    :par shares : list with weights of each hazard function from hz_list 
    :par set_hz : boolean, if True sets the output hz as self.hz
    """
    if len(hz_list)!=len(shares):
        raise Exception('The list hz_list should have the same length as the list shares')
    if not all([hz.shape == hz_list[0].shape for hz in hz_list[1:]]):
        raise Exception('The shapes of the provided hazard functions are not the same')
    if all([share.size==1 for share in shares]): # one share for each hazard function
        if round(sum(shares),10) !=1:
            raise Exception('The shares should add up to one')
        if hz_list[0].ndim == 2:
            shares = [np.tril(np.full_like(hz_list[0], share)) for share in shares]
        elif hz_list[0].ndim == 1:
            shares = [np.full_like(hz_list[0], share) for share in shares]
    elif all([share.shape==hz_list[0].shape for share in shares]):
        if np.any(np.tril(np.round(np.sum(shares,axis=0),10)!=1)):
            raise Exception('The shares should add up to one')
    else:
        raise Exception('The provided shares should either be floats or have the same size as the hazard function arrays.')
    sf_list = []
    for hz in hz_list:
        sf = compute_sf_from_hz(hz)
        sf_list.append(sf)
    if hz_list[0].ndim == 2:
        hz_output = np.zeros_like(hz_list[0])
        for n in range(hz_list[0].shape[0]): # for each cohort n
            temp1 = []
            for hz, share in zip(hz_list, shares):
                temp1.append(hz[n,n]*share[n,n])
            hz_output[n,n] = sum(temp1)
            for m in range(n,hz_list[0].shape[0]-1): # foreach age m
                temp2 = []
                temp3 = []
                for sf, hz, share in zip(sf_list, hz_list, shares):
                    temp2.append(hz[m+1,n]*share[m,n]*sf[m,n])
                    temp3.append(share[m,n]*sf[m,n])
                hz_output[m+1,n] = sum(temp2)/sum(temp3)
    elif hz_list[0].ndim == 1:
        hz_output = np.zeros_like(hz_list[0])
        temp1 = []
        for hz, share in zip(hz_list, shares):
            temp1.append(hz[0]*share[0])
        hz_output[0] = sum(temp1)
        for m in range(hz_list[0].shape[0]-1): # for each age m
            temp2 = []
            temp3 = []
            for sf, hz, share in zip(sf_list, hz_list, shares):
                temp2.append(hz[m+1]*share[m]*sf[m])
                temp3.append(share[m]*sf[m])
            hz_output[m+1] = sum(temp2)/sum(temp3)
    else:
        raise Exception('The hazard functions should each be a 1D or 2D numpy array')
    return hz_output


def compute_hz_from_lt_par(lt, t):
    """
    Calculates the hazard table self.hz(t,c) from lifetime distribution parameters. 
    The hazard table denotes the probability of a product inflow from year n (cohort) 
    failing during time period m, still present at the beginning of time period m (after m-n years).
    lt : lifetime distribution: dictionary with distribution type and parameters, where each parameter is of shape (t,c)
    """
    # find unique sets of lifetime parameters
    unique, inverse, length = find_unique_lt(lt, t)
    # calculate sf for each unique parameter set
    hz_unique = np.zeros((len(t),length))
    if lt['Type'] == 'Normal':
        for i in range(length): # for each unique parameter set
            if unique['StdDev'][i] != 0:
                sf = scipy.stats.norm.sf(np.arange(len(t)), loc=unique['Mean'][i], scale=unique['StdDev'][i])
                hz_unique[:,i] = compute_hz_from_sf(sf) # calculate hz for each unique parameter set
    elif lt['Type'] == 'FoldedNormal':
        for i in range(length): # for each unique parameter set
            if unique['StdDev'][i] != 0:
                sf = scipy.stats.foldnorm.sf(np.arange(len(t)), c=unique['Mean'][i]/unique['StdDev'][i], loc=0, scale=unique['StdDev'][i])
                hz_unique[:,i] = compute_hz_from_sf(sf) # calculate hz for each unique parameter set
    elif lt['Type'] == 'LogNormal': 
        for i in range(length): # for each unique parameter set
            if unique['StdDev'][i] != 0:
                # calculate parameter sigma of underlying normal distribution:
                LT_LN = np.log(unique['Mean'][i] / np.sqrt(1 + unique['Mean'][i] * unique['Mean'][i] / (unique['StdDev'][i] * unique['StdDev'][i]))) 
                SG_LN = np.sqrt(np.log(1 + unique['Mean'][i] * unique['Mean'][i] / (unique['StdDev'][i] * unique['StdDev'][i])))
                sf = scipy.stats.lognorm.sf(np.arange(len(t)), s=SG_LN, loc = 0, scale=np.exp(LT_LN))
                hz_unique[:,i] = compute_hz_from_sf(sf) # calculate hz for each unique parameter set
    elif lt['Type'] == 'Weibull':
        for i in range(length): # for each unique parameter set
            if unique['Scale'][i] != 0:
                sf = scipy.stats.weibull_min.sf(np.arange(len(t)), c=unique['Shape'][i], loc = 0, scale=unique['Scale'][i])
                hz_unique[:,i] = compute_hz_from_sf(sf) # calculate hz for each unique parameter set
    else:
        raise Exception(f"Distribution type {lt['Type']} is not implemented")
    # calculate hazard table hz for the entire time-cohort matrix
    hz = np.zeros((len(t), len(t)))
    for n in range(len(t)): # for each cohort n
        for m in range(n,len(t)): # for each time period m 
            hz[m,n] = hz_unique[m-n,inverse[m,n]]
    return hz


def find_unique_lt(lt, t):
    """
    Finds unique sets of p lifetime parameters (e.g., for Weibull, the set includes scale and shape), each parameter of shape (t,t).
    :return unique: The dictionary of parameters and their values, such that each set (p1[i], p2[i], ..., pn[i]) is unique. 
    :return inverse: The indices to reconstruct the original lt array from the unique array.
    :return length: Number of unique sets
    """
    params = {k:v for k,v in lt.items() if k!= 'Type'}
    sets = np.concatenate([[p] for p in params.values()], axis=0) # stacks all the p parameters
    sets = sets.reshape(len(params),-1) # reshapes from 3D form (p,t,t) into 2D form (p,t*t)
    unique, inverse = np.unique(sets, return_inverse=True, axis=1)
    inverse = inverse.reshape(len(t),len(t)) # reshapes from 1D form (t*t) into 2D form (t,t)
    length = np.shape(unique)[1]
    unique = {k:unique[p,:] for p,k in enumerate(params.keys())}
    return unique, inverse, length

