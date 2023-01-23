from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt

def plot_residuals(y, yhat):

    # Calculate Residuals
    residuals = y - yhat

    # P L O T T I N G
    plt.hlines(0, y.min(), y.max(), ls='--')
    plt.scatter(y, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('y value ($y$)')
    plt.title('y vs yhat')
    plt.show()

def regression_errors(y, yhat): 
    
    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = MSE**.5
    
    ESS = ((yhat - y.mean())**2).sum()
    TSS = ESS + SSE
    
    return SSE, ESS, TSS, MSE, RMSE

def baseline_mean_errors(y):

    baseline = np.repeat(y.mean(), len(y))
    
    MSE = mean_squared_error(y, baseline)
    SSE = MSE * len(y)
    RMSE = MSE**.5
    
    return SSE, MSE, RMSE   

def better_than_baseline(y, yhat):
    SSE, ESS, TSS, MSE, RMSE = regression_errors(y, yhat)
    
    SSE_baseline, MSE_baseline, RMSE_baseline = baseline_mean_errors(y)
    
    if SSE < SSE_baseline:
        print('better than baseline')
    else:
        print('not better than baseline')
