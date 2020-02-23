"""
Description:
Given unreliable observations of patient classes by multiple observers,
determine the most likely true class for each patient, class marginals,
and  individual error rates for each observer, using Expectation Maximization

References:
( Dawid and Skene (1979). Maximum Likelihood Estimation of Observer
Error-Rates Using the EM Algorithm. Journal of the Royal Statistical Society.
Series C (Applied Statistics), Vol. 28, No. 1, pp. 20-28. 
"""

import numpy as np
import sys
import time
now = 0
def check_time():
    global now
    if now == 0:
        now = time.time()
        return 0
    else:
        delta = time.time() - now
        now = time.time()
        return delta
def aggregated_responses(original_labels):
    all_labels = []
    for original_label in original_labels:
        all_labels.extend([l.split(':')[-1] for l in original_label['Label'] if ':' in l])
        
    assets = np.array(list(set([label['External ID'] for label in original_labels])))
    observers = np.array(list(set([label['Created By'] for label in original_labels])))
    label_ids = np.array(list(set(all_labels)))
    classes = [False, True]
    counts = np.zeros((len(label_ids), len(assets), len(observers), 2))
    #counts[:,:,:,0] = 1
    assets_lookup = {k:i for i,k in enumerate(assets)}
    observers_lookup = {k:i for i,k in enumerate(observers)}
    label_ids_lookup = {k:i for i,k in enumerate(label_ids)}
    for original_label in original_labels:
        asset_index = assets_lookup[original_label['External ID']]
        observer_index = observers_lookup[original_label['Created By']]
        labels = [l.split(':')[-1] for l in original_label['Label'] if ':' in l]
        counts[:, asset_index, observer_index, 0] = 1
        for label in labels:
            label_index = label_ids_lookup[label]
            counts[label_index, asset_index, observer_index, 1] = 1
            counts[label_index, asset_index, observer_index, 0] = 0
    return assets, observers, label_ids, counts, classes
        
def run(responses,assets=None,observers=None,classes=None,counts=None, tol=0.00001, max_iter=100, init='average', verbose=False):
    """
    Function: dawid_skene()
        Run the Dawid-Skene estimator on response data
    Input:
        responses: a dictionary object of responses:
            {assets: {observers: [labels]}}
        tol: tolerance required for convergence of EM
        max_iter: maximum number of iterations of EM
    """ 
    # convert responses to counts
    if verbose:
        print ('Start', check_time())
    if counts is None or assets is None or observers is None or classes is None:
        (assets, observers, classes, counts) = responses_to_counts(responses)
        
    if verbose:
        print ('Finish convert response', check_time())
    # initialize
    iter = 0
    converged = False
    old_class_marginals = None
    old_error_rates = None
    
    asset_classes = initialize(counts)
    if verbose:
        print ('Finish initialize', check_time())
    # while not converged do:
    while not converged:     
        iter += 1
        # M-step
        if verbose:
            print ('Start M step', check_time())
        (class_marginals, error_rates) = m_step(counts, asset_classes)        
        if verbose:
            print ('Finish M step', check_time())
        # E-setp
        asset_classes = e_step(counts, class_marginals, error_rates)  
        if verbose:
            print ('Finish E step', check_time())
        # check likelihood
        log_L = calc_likelihood(counts, class_marginals, error_rates)
        if verbose:
            print ('Finish calc likelihood step', check_time())
        # check for convergence
        if old_class_marginals is not None:
            class_marginals_diff = np.sum(np.abs(class_marginals - old_class_marginals))
            error_rates_diff = np.sum(np.abs(error_rates - old_error_rates))         
            if (class_marginals_diff < tol and error_rates_diff < tol) or iter > max_iter:
                converged = True
        else:
            pass
        # update current values
        if verbose:
            print ('Finish update current', check_time())
        old_class_marginals = class_marginals
        old_error_rates = error_rates
        
    return {asset:asset_class for asset,asset_class in zip(assets,asset_classes)}
    # return {'assets':assets, 
    #         'observers':observers, 
    #         'classes':classes, 
    #         'counts':counts, 
    #         'class_marginals':class_marginals, 
    #         'error_rate':error_rates, 
    #         'asset_classes':asset_classes} 
def responses_to_counts(responses):
    """
    Function: responses_to_counts()
        Convert a matrix of annotations to count data
    Inputs:
        responses: dictionary of responses {patient:{observers:[responses]}}
    Return:
        patients: list of patients
        observers: list of observers
        classes: list of possible patient classes
        counts: 3d array of counts: [patients x observers x classes]
    """ 
    patients = list(responses.keys())
    patients.sort()
    nPatients = len(patients)
        
    # determine the observers and classes
    observers = set()
    classes = set()
    for i in patients:
        i_observers = responses[i].keys()
        for k in i_observers:
            if k not in observers:
                observers.add(k)
            ik_responses = responses[i][k]
            classes.update(ik_responses)
    
    classes = list(classes)
    classes.sort()
    nClasses = len(classes)
        
    observers = list(observers)
    observers.sort()
    nObservers = len(observers)
            
    # create a 3d array to hold counts
    counts = np.zeros([nPatients, nObservers, nClasses])
    
    # convert responses to counts
    for patient in patients:
        i = patients.index(patient)
        for observer in responses[patient].keys():
            k = observers.index(observer)
            for response in responses[patient][observer]:
                j = classes.index(response)
                counts[i,k,j] += 1
        
    
    return (patients, observers, classes, counts)

def initialize(counts):
    """
    Function: initialize()
        Get initial estimates for the true patient classes using counts
        see equation 3.1 in Dawid-Skene (1979)
    Input:
        counts: counts of the number of times each response was received 
            by each observer from each patient: [patients x observers x classes] 
    Returns:
        patient_classes: matrix of estimates of true patient classes:
            [patients x responses]
    """  
    [nPatients, nObservers, nClasses] = np.shape(counts)
    # sum over observers
    response_sums = np.sum(counts,1)
    # create an empty array
    patient_classes = np.zeros([nPatients, nClasses])
    # for each patient, take the average number of observations in each class
    for p in range(nPatients):
        patient_classes[p,:] = response_sums[p,:] / np.sum(response_sums[p,:],dtype=float)
        
    return patient_classes


def m_step(counts, patient_classes):
    """
    Function: m_step()
        Get estimates for the prior class probabilities (p_j) and the error
        rates (pi_jkl) using MLE with current estimates of true patient classes
        See equations 2.3 and 2.4 in Dawid-Skene (1979)
    Input: 
        counts: Array of how many times each response was received
            by each observer from each patient
        patient_classes: Matrix of current assignments of patients to classes
    Returns:
        p_j: class marginals [classes]
        pi_kjl: error rates - the probability of observer k receiving
            response l from a patient in class j [observers, classes, classes]
    """
    [nPatients, nObservers, nClasses] = np.shape(counts)
    
    # compute class marginals
    class_marginals = np.sum(patient_classes,0)/float(nPatients)
    
    # compute error rates 
    error_rates = np.zeros([nObservers, nClasses, nClasses])
    for k in range(nObservers):
        for j in range(nClasses):
            for l in range(nClasses): 
                error_rates[k, j, l] = np.dot(patient_classes[:,j], counts[:,k,l])
            # normalize by summing over all observation classes
            sum_over_responses = np.sum(error_rates[k,j,:])
            if sum_over_responses > 0:
                error_rates[k,j,:] = error_rates[k,j,:]/float(sum_over_responses)  
    return (class_marginals, error_rates)

  
def e_step_old(counts, class_marginals, error_rates):
    """ 
    Function: e_step()
        Determine the probability of each patient belonging to each class,
        given current ML estimates of the parameters from the M-step
        See equation 2.5 in Dawid-Skene (1979)
    Inputs:
        counts: Array of how many times each response was received
            by each observer from each patient
        class_marginals: probability of a random patient belonging to each class
        error_rates: probability of observer k assigning a patient in class j 
            to class l [observers, classes, classes]
    Returns:
        patient_classes: Soft assignments of patients to classes
            [patients x classes]
    """    
    [nPatients, nObservers, nClasses] = np.shape(counts)
    
    patient_classes = np.zeros([nPatients, nClasses])    
    
    for i in range(nPatients):
        for j in range(nClasses):
            estimate = class_marginals[j]
            estimate *= np.prod(np.power(error_rates[:,j,:], counts[i,:,:]))
            
            patient_classes[i,j] = estimate
        # normalize error rates by dividing by the sum over all observation classes
        patient_sum = np.sum(patient_classes[i,:])
        if patient_sum > 0:
            patient_classes[i,:] = patient_classes[i,:]/float(patient_sum)
    
    return patient_classes
  
def e_step(counts, class_marginals, error_rates):
    """ 
    Function: e_step()
        Determine the probability of each patient belonging to each class,
        given current ML estimates of the parameters from the M-step
        See equation 2.5 in Dawid-Skene (1979)
    Inputs:
        counts: Array of how many times each response was received
            by each observer from each patient
        class_marginals: probability of a random patient belonging to each class
        error_rates: probability of observer k assigning a patient in class j 
            to class l [observers, classes, classes]
    Returns:
        patient_classes: Soft assignments of patients to classes
            [patients x classes]
    """    
    [nPatients, nObservers, nClasses] = np.shape(counts)
    patient_classes = np.zeros([nPatients, nClasses])    
    patient_classes += class_marginals.reshape((1,-1))
    for j in range(nClasses):                
        a = np.ones_like(counts) #replace np.power which takes a long time to compute since counts can only be 0 or 1
        a -= (1 - error_rates[np.newaxis,:,j,:]) * counts        
        b = np.prod(a, axis=(1,2))   
        patient_classes[:,j] *= b
    
    patient_classes /= patient_classes.sum(axis=1).reshape((-1, 1))      
    return patient_classes

def calc_likelihood_old(counts, class_marginals, error_rates):
    """
    Function: calc_likelihood()
        Calculate the likelihood given the current parameter estimates
        This should go up monotonically as EM proceeds
        See equation 2.7 in Dawid-Skene (1979)
    Inputs:
        counts: Array of how many times each response was received
            by each observer from each patient [pt, observers, classes]
        class_marginals: probability of a random patient belonging to each class
        error_rates: probability of observer k assigning a patient in class j 
            to class l [observers, classes, classes]
    Returns:
        Likelihood given current parameter estimates
    """  
    [nPatients, nObservers, nClasses] = np.shape(counts)
    assert nClasses == 2
    log_L = 0.0
    
    for i in range(nPatients):
        patient_likelihood = 0.0
        for j in range(nClasses):        
            class_prior = class_marginals[j]
            patient_class_likelihood = np.prod(np.power(error_rates[:,j,:], counts[i,:,:]))  
            patient_class_posterior = class_prior * patient_class_likelihood
            patient_likelihood += patient_class_posterior
        temp = log_L + np.log(patient_likelihood)
        
        if np.isnan(temp) or np.isinf(temp):
            print (i, log_L, np.log(patient_likelihood), temp)
            sys.exit()

        log_L = temp        
        
    return log_L
def calc_likelihood(counts, class_marginals, error_rates):
    """
    Function: calc_likelihood()
        Calculate the likelihood given the current parameter estimates
        This should go up monotonically as EM proceeds
        See equation 2.7 in Dawid-Skene (1979)
    Inputs:
        counts: Array of how many times each response was received
            by each observer from each patient [pt, observers, classes]
        class_marginals: probability of a random patient belonging to each class
        error_rates: probability of observer k assigning a patient in class j 
            to class l [observers, classes, classes]
    Returns:
        Likelihood given current parameter estimates
    """  
    [nPatients, nObservers, nClasses] = np.shape(counts)
    log_L = 0.0
    
    patient_likelihoods = np.zeros((nPatients, nClasses))
    
    
    for j in range(nClasses): 
        class_prior = class_marginals[j]
        a = np.ones_like(counts) #replace np.power which takes a long time to compute
        a -= (1 - error_rates[np.newaxis,:,j,:]) * counts        
        patient_class_likelihood = np.prod(a, axis=(1,2))
        patient_class_posterior = class_prior * patient_class_likelihood 
        patient_likelihoods[:,j] = patient_class_posterior   
        
    return np.log(patient_likelihoods.sum(axis=1)).sum()
        
