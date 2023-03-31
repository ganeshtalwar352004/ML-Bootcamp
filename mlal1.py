import numpy as np
import pandas as pd
def initial_model_output(x,w,b):
    m,n=x.shape
    f=[[]]
    for i in range(m):
        f_x=0
        f_x=f_x+np.dot(w,x[i])+b
        f[i,0]=f_x
    return f
def cost_function_linear(x,y,w,b):
    m,n=x.shape
  
    cost=0.0
    # for i in range (m):
    #     k=((np.dot(w,x[i])+b)-y[i])**2
    #     cost=cost+k
    # cost=(1/(2*m))*cost
    # return cost
    cost=cost+np.sum(((((w@x.T)+b)-y.T)**2),axis=1)
    return cost/(2*m)
def z_score_normalisation(x):
    mu     = np.mean(x, axis=0)                 # mu will have shape (n,)
    
    sigma  = np.std(x, axis=0)                  # sigma will have shape (n,)
    
    X_norm = (x - mu) / sigma      

    return X_norm
# def z_score_normalisation(x):
#     m,n=x.shape
#     print (n)
#     X_nom = np.zeros(shape=(m,n))
#     m,n=X_train.shape
#     for i in range(n):
#         for j in range(m):
           
#             X_nom[j,i]=(x[j,i]-x[:,i].min())/(x[:,i].max()-x[:,i].min())
#     return X_nom
def gradient_descent_calculate_linear(x,y,w,b):
    m,n=x.shape
    dw=np.zeros(n)
    db=0.0
    k=[[]]

# #     for i in range (n):
# #         k=0.0
# #         for j in range(m):
# #             k=k+((np.dot(w,x[j])+b)-y[j])*x[j,i]
# #         dw[i]=k
    p=(w@(x.T)+b)-y.T
    for i in range(n):
        k=np.sum(np.dot(p,x[:,i]),axis=0)
        dw[i]=k
 #     for i in range(m):
# #         db=db+((np.dot(w,x[j])+b)-y[j])
    db=np.sum((x@(w.T)+b)-y.T)
    return dw/m,db/m
def gradient_linear(x,y,win,bin,alpha,iter,gradient_descent_calculate_linear):
    for i in range(iter):
        dw_1,db_1=gradient_descent_calculate_linear(x,y,win,bin)
        win=win-(alpha*dw_1)
        bin=bin-(alpha*db_1)
#          if i<100000:      # prevent resource exhaustion 
#             J_history.append( cost_function(x, y, w , b))
#             p_history.append([w,b])
#         # Print cost every at intervals 10 times or as many iterations if < 10
#         if i% math.ceil(num_iters/10) == 0:
#             print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
#                   f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
#                   f"w: {w: 0.3e}, b:{b: 0.5e}")
    return win,bin
def cal_eucleadian_distance(x1,x2):
    return np.sqrt(np.sum(np.square(x1-x2),axis=1))
def cal_pred(x1,x2):
    k=100
    distance=cal_eucleadian_distance(x1[:,1:785],x2)
    indices=np.argsort(distance)
    predicted_value=np.mean(x1[indices[0:k],0])
    return predicted_value
def sigmoid(z):
    v=1+np.exp(-z)
    k=1/v
    return k

def cost_function_logistic(x,y,w,b):
    m,n=y.shape
    total_cost=0
    for i in range(m):
        f=sigmoid((w@x[i,:].T)+b)
        total_cost=total_cost+((-y[i]*(np.log(f)))-((1-y[i])*np.log(1-f)))
    return total_cost/(2*m)
# def gradient_descent_calculate_logistic(x,y,w,b):
#     m,n=x.shape
#     dw=np.zeros(n)
#     db=0
#     for i in range(n):
#         k=0
#         for j in range (m):
#             f=sigmoid((np.dot(w,x[i,:]))+b)
#             k=k+(f-y[j])*x[j,i]
#         dw[i]=k
#     for i in range (m):
#         f=sigmoid((np.dot(w,x[i,:]))+b)
#         db=db+(f-y[j])
#     return dw,db
# def gradient_logistic(x,y,w,b,alpha,iter,gradient_descent_calculate_logistic):
   
#     for i in range(iter):
#         dw1,db1=gradient_descent_calculate_logistic(x,y,w,b)
#         w=w-dw1
#         b=b-db1
#     return w,b 
def gradient_descent_calculate_logistic(x,y,w,b):
    m,n=x.shape
    dw=np.zeros(n)
    db=0.0
    k=[[]]
    p=sigmoid(w@(x.T)+b)-y.T
    for i in range(n):
        k=np.sum(np.dot(p,x[:,i]),axis=0)
        dw[i]=k
 #     for i in range(m):
# #         db=db+((np.dot(w,x[j])+b)-y[j])
    db=np.sum(sigmoid(x@(w.T)+b)-y.T)
    return dw/m,db/m
def gradient_logistic(x,y,win,bin,alpha,iter,gradient_descent_calculate_logistic):
    for i in range(iter):
        dw_1,db_1=gradient_descent_calculate_logistic(x,y,win,bin)
        win=win-(alpha*dw_1)
        bin=bin-(alpha*db_1)
    return win,bin
def initial_model_output_logistic(x,w,b):
    m,n=x.shape
    f_x=[[]]
    
    f_x=sigmoid((w@x.T)+b).T
      
    return f_x
def relu(Z):
    return np.maximum(0,Z)
def softmax(z):
    return exp(z)/np.sum(exp(z))

def calculate_r2(actual_target_values,predicted_target_values):
    sum_squared_residuals=np.sum(np.square(predicted_target_values-actual_target_values))
    sum_squares=np.sum(np.square(np.mean(actual_target_values)-actual_target_values))
    r2=1-(sum_squared_residuals/sum_squares)
    return r2    