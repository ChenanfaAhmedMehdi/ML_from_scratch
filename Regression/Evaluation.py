import numpy as np
def R2_score(prediction, y, print_comp=False):
    if (print_comp): print('prediction:', prediction,sep='\n',end='\n')
    ### Squared Sum Of Errors
    SSE = ((y - prediction) ** 2).sum()
    if (print_comp): print('SSE=', SSE, end='\n')
    ### Total Squared Sum
    TSS = ((y - y.mean()) ** 2).sum()
    if (TSS == 0):
        raise ValueError("R2 is undefined because y is constant")
    if (print_comp): print('TSS=', TSS, end='\n')
    ### Score
    return (1 - SSE/TSS)

def BIC(pred,y,k): 
    n = pred.shape[0]
    RSS = ((y - pred)**2).sum()
    return n * np.log(RSS / n) + (k + 1) * np.log(n)