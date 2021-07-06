import pandas as pd
import numpy as np
dfx=pd.read_csv('Diabetes_XTrain.csv')
dfy=pd.read_csv('Diabetes_YTrain.csv')
dftest=pd.read_csv('Diabetes_XTest.csv')
dfx.head()
x1=dfx.values
y1=dfy.values
print(x1.shape)
print(y1.shape)
def dist(w,z):
    return np.sqrt(sum((w-z)**2))
def knn(x,y,query_point,k=25):
    m=x.shape[0]
    vals=[]
    for i in range(m):
        d=dist(x[i],query_point)
        vals.append((d,y[i]))
    vals=sorted(vals)
    vals=vals[:k]
    vals=np.array(vals)
    #print(vals)
    vals2=vals[:,1]
    #print(vals2)
    d=sum(vals2)
    if d>3:
        return 1
    else :
        return 0
    #newvals=np.unique(vals,return_counts=True)
    #print(newvals)
test=dftest.values
print(test.shape)
k=test.shape[0]
print(k)
k=int(k)
ans=np.full((192,1),0)
for i in range(k):
    a=knn(x1,y1,test[i],25)
    ans[i][0]=a

#print(ans)    
ans=pd.DataFrame(ans,columns=['Outcome'])
ans.head()
ans=ans.to_csv('Diabetes20.csv',index=False)