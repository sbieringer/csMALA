import numpy as np

def RMSE(true, fitted):
    '''
    returns the RMSE - no errors are accounted for in this score
    '''
    return np.mean((true - fitted)**2)**0.5

def MAD(true, fitted):
    '''
    returns the mean absolute distance - no errors are accounted for in this score
    '''
    return np.mean(np.abs(true - fitted))

def NLPD(true, fitted, estimated_std):
    '''
    returns the negative log likelihood of the true points assuming normal distribution around the fit. We need to do this as the likelihood is untraceable for MBB-method
    '''
    return np.sum(-0.5*((true-fitted)/estimated_std)**2-np.log(estimated_std)-0.5*np.log(2*np.pi))

def chi2(x, data_generator, true, fitted):
    '''
    returns the chi^2-value of the fit accounting for the error on the data
    '''
    return np.sum(((true - fitted)/data_generator.std_dev(x))**2)

def coverage(true, fitted, estimated_std):
    '''
    returns the fraction of true points within the estimated standard deviation. Optimally this is gives 68%
    '''
    out = 0
    for i in range(len(true)):
        if true[i]>= fitted[i]-estimated_std[i] and true[i] <= fitted[i]+estimated_std[i]:
            out +=1
    return out/len(true)