# ==================================================================================
# It icludes all the subfunctions
# @Author: Yina Wei, 2021
#==================================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy import signal
from scipy.ndimage.filters import gaussian_filter
from sklearn import linear_model
from scipy.stats import shapiro  # Shapiro-Wilk Test
from scipy.stats import mannwhitneyu,ttest_ind # Mann-Whitney U test
from sklearn import preprocessing
import scikit_posthocs as sp
from scipy.interpolate import make_interp_spline
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import scikit_posthocs as sp
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from scipy.stats import randint
import random
from numpy.random import seed
from numpy.random import randn
from sklearn import metrics


def spline(x,y,x_smooth):
    y_smooth = make_interp_spline(x,y)(x_smooth) 
    return y_smooth


def two_sample_test(data1,data2):
    stat1, p1 = shapiro(data1)
    stat2, p2 = shapiro(data2)
    alpha = 0.05
    if np.logical_and(p1>alpha,p2>alpha):
        print('Sample looks Gaussian,using two-sample t-test')
        stat, p = ttest_ind(data1, data2)  # the same as using: stat, p = stats.f_oneway(data1,data2) 
    else:
        print('Sample does not look Gaussian, using Mann-Whitney U test')
        stat, p = mannwhitneyu(data1,data2)
    return stat,p


def three_sample_test(data1,data2,data3):
    stat1, p1 = shapiro(data1)
    stat2, p2 = shapiro(data2)
    stat3, p3 = shapiro(data3)
    alpha = 0.05
    x = [data1,data2,data3]
    if np.logical_and(p1>alpha,p2>alpha) and p3>alpha:
        print('Sample looks Gaussian,using one-way ANOVA')
        stat, p = stats.f_oneway(data1,data2,data3)   # the same as using: stat, p = stats.f_oneway(data1,data2) 
        post_hoc_p=np.array(sp.posthoc_ttest(x, p_adjust = 'holm'))
    else:
        print('Sample does not look Gaussian, using Kruskal-Wallis H-test')
        stat, p = stats.kruskal(data1,data2,data3)
        post_hoc_p = np.array(sp.posthoc_mannwhitney(x, p_adjust = 'holm'))
    return stat,p,post_hoc_p


def star_p(p):
    if p>0.05:
        str_p='n.s.'
    elif p<=0.05 and p>0.01:
        str_p='*'
    elif p<=0.01 and p>0.001:
        str_p='**'
    else:
        str_p='***'
    return str_p


def simpleaxis(ax):
    #Hide the right and top spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
def simpleaxis_all(ax):
    #Hide the right and top spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx    
    

def find_neg_peak_indx(waveform,t_STA):
    indx = np.where(np.logical_and(t_STA>=-0.5,t_STA<=1.5))
    t_STA= t_STA[indx]
    waveform = waveform[indx]
    return waveform.argmin()+indx[0][0]
 
# wcss and desity function to find optimal K
def selectionK_kmeans(X,numk):  # X has the dimension of n*m, where m is the number of features
    Nd = X.shape[1]
    K= range(1,numk+1)
    wcss=[]  # within-cluster sum of square
    fk=[]    # density function
    Skm1=0
    for k in K:
        km=KMeans(n_clusters=k, init='k-means++', n_init= 1000, random_state= 10)
        km=km.fit(X)
        wcss.append(km.inertia_)
        Sk=km.inertia_
        fk.append(fK(X,k,Sk,Skm1))
        Skm1=Sk
    wcss = np.array(wcss)
    fk = np.array(fk)
    return K,wcss,fk

def fK(X, thisk, Sk, Skm1):
    Nd = X.shape[1]
  #  print Nd
    a = lambda k, Nd: 1 - 3/(4*Nd) if k == 2 else a(k-1, Nd) + (1-a(k-1, Nd))/6
    if thisk == 1:
        fs = 1
    elif Skm1 == 0:
        fs = 1
    else:
        fs = Sk/(a(thisk,Nd)*Skm1)
    return fs

def findElbow(wcss):
    rate = np.diff(wcss)
    change = rate[0:-1]/rate[1:]
    return change.argmax()+1    # return the index of the rate changes sharply

    
def plot_shaded_errorbar(ax,x,data,mycolor='black',lw=1):
    
    # calculate the mean and standard error of mean (SEM)
    y = np.nanmean(data,axis=0)
    error = np.nanstd(data,axis=0)/np.sqrt(data.shape[0])
    
    ax.plot(x, y, '-',color=mycolor,linewidth=lw)
    ax.fill_between(x, y-error, y+error,alpha=0.5,color=mycolor)  
    simpleaxis(ax)
    
def outlier_index(data):
    
    upper = np.mean(data)+3*np.std(data) #
    lower = np.mean(data)-3*np.std(data) #
    
    index = np.where(np.logical_or(data>upper,data<lower))[0]
    return np.array(index)


def pre_process(all_features):
    # impute all NaNs in the array and replace with the mean
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imp.fit(all_features)
    # replace all NaNs
    #print all_features.shape
    all_features_new=imp.transform(all_features)
   # all_features_new=all_features
    # Normalise all columns of the array
    all_features_new_scaled = preprocessing.scale(all_features_new)
    # Bring all elements to the same type
    all_features_new_scaled.astype(float)
    return all_features_new_scaled


def plot_errorbar(ax,x,data,mycolor='black',lw=1):
    
    # calculate the mean and standard error of mean (SEM)
    y = np.nanmean(data,axis=0)
    error = np.nanstd(data,axis=0)#/np.sqrt(data.shape[0])
    
    ax.plot(x, y, '-',color=mycolor,linewidth=lw)
    plt.errorbar(x,y,yerr=error,color=mycolor)
    #ax.fill_between(x, y-error, y+error,alpha=0.5,color=mycolor)  
    
    simpleaxis(ax)
    
def plot_errorbar_vertical(ax,x,data,mycolor='black',lw=1):
    
    # calculate the mean and standard error of mean (SEM)
    y = np.nanmean(data,axis=0)
    error = np.nanstd(data,axis=0)#/np.sqrt(data.shape[0])
    
    ax.plot(y, x, '-',color=mycolor,linewidth=lw)
    plt.errorbar(y,x,xerr=error,color=mycolor)
    #ax.fill_between(x, y-error, y+error,alpha=0.5,color=mycolor)  
    
    simpleaxis(ax) 


def plot_shaded_errorbar(ax,x,data,mycolor='black'):
    
    # calculate the mean and standard error of mean (SEM)
    y = np.nanmean(data,axis=0)
    error = np.nanstd(data,axis=0)/np.sqrt(data.shape[0])
    
    ax.plot(x, y, '-',color=mycolor,linewidth=2)
    ax.fill_between(x, y-error, y+error,alpha=0.5,color=mycolor)  
    simpleaxis(ax)
    

# remove outlier for linear regression 
def lm_outlier(x,y,idx0):
   
    lm0 = sm.OLS(y, sm.add_constant(x)).fit()
    infl = lm0.get_influence()
    sm_fr = infl.summary_frame()

    # Remove outlier that higher than 1 cook's distance
    outlier_idx=np.where(sm_fr['cooks_d']>=1)[0]

    # Selected idx for analysis
    idx=np.unique(list(set(range(len(x))).difference(set(outlier_idx)))) 
    idx=np.unique(list(set(idx0).union(set(idx))))  # including the idx0
     
    # linear regression model without outlier 
    slope, intercept, _, _, _ = stats.linregress(x[idx],y[idx])

    return slope,intercept

# Distance from point(x1,y1) to the line ax+by+c=0
# Distance = (| a*x1 + b*y1 + c |) / (sqrt( a*a + b*b))
def dist_fc(x1,y1,a=1,b=1,c=0):  
    Distance = (abs(a*x1 + b*y1 + c )) / (np.sqrt( a*a + b*b))
    return Distance

def sym_dist_fc(s1,s2):
    res=np.zeros(len(s1))
    for i in range(len(s1)):
        res[i]=dist_fc(s1[i],s2[i],1,1,0)  # ax+by+c=0==> x+y=0
    return res   

def outlier_below(x,y,Th,numpoint=3):
    idx=range(-1*numpoint+1,1,1)+np.array(len(x)-1) 
    # linear regression model without outlier 
    slope, intercept, _, _, _ = stats.linregress(x[idx],y[idx])

    # Preprocess the data of A,W,TPW,t_NegPeak, deal with unphysiological values 
    flag=0
    for i in range(len(y)-1-numpoint,-1,-1):     
        yhat=slope*x[i]+intercept
        if flag==1:
           y[i]=float("nan")
        else:
            if abs(y[i]-yhat)>Th:
                y[i]=float("nan")
    return y


def outlier_above(x,y,Th,numpoint=3):
    idx=range(numpoint)
    # linear regression model without outlier 
    slope, intercept, _, _, _ = stats.linregress(x[idx],y[idx])
    # Preprocess the data of A,W,TPW,t_NegPeak, deal with unphysiological values
    flag=0
    for i in range(numpoint-1,len(y),1):
        yhat=slope*x[i]+intercept
        if flag==1:
           y[i]=float("nan")
        else:
           if abs(y[i]-yhat)>Th:
               y[i]=float("nan")
               flag=1
    return y


def cal_multi_ch_features(dist0,A,W,t_NegPeak,maxCh,dth,Th,Th0=0.005,numpoint=3):

    idx_below = np.where(np.logical_and(np.logical_and(dist0<=0,dist0>=-dth),A>=max(A[maxCh]*Th0,Th)))[0]
    idx_above = np.where(np.logical_and(np.logical_and(dist0>=0,dist0<=dth),A>=max(A[maxCh]*Th0,Th)))[0]
    
    if len(idx_below)<numpoint:
       idx_below=np.array(maxCh)+range(-1*numpoint+1,1,1)
    if len(idx_above)<numpoint:
       idx_above=np.array(maxCh)+range(numpoint)

    slope0, intercept0 = linregress(dist0[idx_below],W[idx_below])
    slope1, intercept1 = linregress(dist0[idx_above],W[idx_above])  
    W_below=(slope0*1000.0)  
    W_above=(slope1*1000.0) #

     
    slope4, intercept4 = linregress(dist0[idx_below],t_NegPeak[idx_below])
    slope5, intercept5 = linregress(dist0[idx_above],t_NegPeak[idx_above])

    PL_below=(slope4*1000.0)  
    PL_above=(slope5*1000.0) 
    [spread_below,spread_above] = cal_spread2(A,dist0,0.12)  

    return  spread_below,spread_above,W_below,W_above,PL_below,PL_above,intercept0,intercept1,intercept4,intercept5


def linregress(x,y,intercept0=1): 
    mask = ~np.isnan(x) & ~np.isnan(y)     
    x=x[mask]
    y=y[mask]
    if intercept0==1:
        slope,intercept, _, _, _ =  stats.linregress(x, y)
    else:
        intercept=y[x==0]
        slope, _ , _ ,_ = np.linalg.lstsq(np.array(x).reshape(-1,1), y-intercept)
    return slope,intercept


# calculate the spread of amplitude 
def cal_spread(A0,dist0,th=0.12):    
    dist = np.linspace(dist0.min(),dist0.max(),10000) #1000 represents number of points to make between T.min and T.max
    A = spline(dist0,A0,dist)

    midamp = max(A)*th
    indx = range(0,A.argmax())
    idx1 = find_nearest(A[indx],midamp)
    indx = range(A.argmax(),len(A))
    idx2 = find_nearest(A[indx],midamp)+A.argmax()
    return dist[idx2]-dist[idx1],abs(dist[idx2])-abs(dist[idx1])

    

# calculate the spread_above and spread_below of amplitude 
def cal_spread2(A0,dist0,th=0.12):    
    dist = np.linspace(dist0.min(),dist0.max(),10000) #1000 represents number of points to make between T.min and T.max
    A = spline(dist0,A0,dist)

    midamp = max(A)*th
    indx = range(0,A.argmax())
    idx1 = find_nearest(A[indx],midamp)
    indx = range(A.argmax(),len(A))
    idx2 = find_nearest(A[indx],midamp)+A.argmax()

    return abs(dist[idx1]),abs(dist[idx2])#dist[A.argmax()]-dist[idx1],dist[idx2]-dist[A.argmax()]

    
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx    


def find_neg_peak_time(waveform0,t_STA0):
    t_STA = np.linspace(t_STA0.min(),t_STA0.max(),10000) #300 represents number of points to make between T.min and T.max
    waveform = np.array(spline(t_STA0,waveform0,t_STA)) 
    return t_STA[find_neg_peak_indx(waveform,t_STA)]

def find_neg_peak_indx(waveform,t_STA):
    indx = np.where(np.logical_and(t_STA>=-1,t_STA<=2.5))
    t_STA= t_STA[indx]
    waveform = waveform[indx]
    return waveform.argmin()+indx[0][0]

# trough to peak width
def calc_tp_width(waveform0,t_STA0):
    waveform0 = waveform0-waveform0[0] 
    t_STA = np.linspace(t_STA0.min(),t_STA0.max(),10000) #10000 represents number of points to make between T.min and T.max
    waveform = spline(t_STA0,waveform0,t_STA)

    idx1 = find_neg_peak_indx(waveform,t_STA) 

    indx = range(idx1,len(waveform))
    idx2 = waveform[indx].argmax()+idx1
    
    return t_STA[idx2]-t_STA[idx1]


#half-way width
def calc_width(waveform0,t_STA0):
    
    waveform0 = waveform0-waveform0[0] 
    t_STA = np.linspace(t_STA0.min(),t_STA0.max(),10000) #10000 represents number of points to make between T.min and T.max
    waveform = spline(t_STA0,waveform0,t_STA)
   
    idx1 = find_neg_peak_indx(waveform,t_STA)   # negative peak 
    indx = range(idx1,len(waveform))   
    idx2 = waveform[indx].argmax()+idx1         # positive peak after the negative peak

    Amp = abs(waveform[idx1])                   # Amplitude
 
    mid_idx1 = find_nearest(waveform[range(0,idx1)],-0.5*Amp)  # the first midpoint before negative peak
    mid_idx2 = find_nearest(waveform[range(idx1,idx2)],-0.5*Amp)+idx1 # the second midpoint between negative peak and positive peak

    Width = t_STA[mid_idx2]-t_STA[mid_idx1]     # Half-way withd
    
    return Amp,Width


    
def cal_one_ch_features(t_STA0,waveform0): 
    
    waveform0 = waveform0-waveform0[0]                 #
    
    t_STA = np.linspace(t_STA0.min(),t_STA0.max(),10000)# 10000 represents number of points to make between T.min and T.max
    waveform = spline(t_STA0,waveform0,t_STA)

    idx1 = find_neg_peak_indx(waveform,t_STA)          # negative peak
    indx = range(idx1,len(waveform))
    idx2 = waveform[indx].argmax()+idx1                # positive peak after the negative peak
    idx3 = waveform[range(idx2,len(waveform))].argmin()+idx2  # negative peak after the positive peak
    
    Amp = abs(waveform[idx1])                          # Amplitude
    TPW = t_STA[idx2]-t_STA[idx1]                      # TP width
    PTratio = abs(waveform[idx2])/abs(waveform[idx1])  # PT ratio
    
    mid_idx1 = find_nearest(waveform[range(0,idx1)],-Amp/2) # the first midpoint before negative peak
    mid_idx2 = find_nearest(waveform[range(idx1,idx2)],-Amp/2)+idx1  # the second midpoint between negative peak and positive peak
    mid_idx3 = find_nearest(waveform[range(idx2,idx3)],(waveform[idx2])/2)+idx2  # the third midpoint after positive peak
    
    REP = t_STA[mid_idx3]-t_STA[idx2]                  # Repolarization time
    Width = t_STA[mid_idx2]-t_STA[mid_idx1]            # Half-way width
    
    idx_rep=np.where(np.logical_and(t_STA>=t_STA[idx1],t_STA<=t_STA[idx1]+0.03))
    Rslope, intercept1, r_value1, p_value1, std_err1 = stats.linregress(t_STA[idx_rep],waveform[idx_rep])
     
    return [Amp,Width,TPW,Rslope,REP]



def cal_EAP_features(t_STA,Ve_STA,Th):
    num_channels = Ve_STA.shape[0]
    A = np.zeros(num_channels)
    W =  np.zeros(num_channels)
    TPW =  np.zeros(num_channels)
    REP = np.zeros(num_channels)
    t_NegPeak =  np.zeros(num_channels)

    for i in range(num_channels): 
        ecp_STA = Ve_STA[i,:]
        ecp_STA = ecp_STA-ecp_STA[0]   
        if abs(min(ecp_STA))>=Th:
            A[i], W[i] = calc_width(ecp_STA,t_STA)
            TPW[i] = calc_tp_width(ecp_STA,t_STA)
            t_NegPeak[i]= find_neg_peak_time(ecp_STA,t_STA)  #negative peak time
        else:
            A[i] = abs(min(ecp_STA))
            TPW[i] = float('NaN')
            W[i] = float('NaN')
            t_NegPeak[i] = float('NaN')
    return A,W,TPW,t_NegPeak


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut,highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype='band', analog=False,output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut,highcut,fs, order=order)
    y = signal.sosfiltfilt(sos, data)
    return y


def clean_axis(ax):
    #Hide the right and top spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()       
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


# Upsample minority class
def upsampling_data(X,X1,X2,X3,rs):
    maxlen=max(len(X1),max(len(X2),len(X3)))
    X1_upsampled = resample(X1, 
                            replace=True,     # sample with replacement
                            n_samples=maxlen-len(X1),    # to match majority class
                            random_state=rs) # reproducible results
    X2_upsampled = resample(X2, 
                            replace=True,     # sample with replacement
                            n_samples=maxlen-len(X2),    # to match majority class
                            random_state=rs) # reproducible results
    X3_upsampled = resample(X3, 
                            replace=True,     # sample with replacement
                            n_samples=maxlen-len(X3),    # to match majority class
                            random_state=rs) # reproducible results
    # Combine majority class with upsampled minority class
    upsampled = []
    upsampled.append(X)
    upsampled.append(X1_upsampled) 
    upsampled.append(X2_upsampled)
    upsampled.append(X3_upsampled)
    upsampled=np.concatenate(upsampled)
    return upsampled



def sign(x):
    return int(x>0)*1+int(x<0)*(-1)

def classifier(X,Y,newX,newY,model_name,random_num):
    
    # Set random seed 
    random.seed(100)
    
    # Initialization    
    accuracy_XY = 0
    conf_matrix_XY = np.zeros((len(np.unique(Y)),len(np.unique(Y))))
    
    accuracy_newXY = 0
    conf_matrix_newXY = np.zeros((len(np.unique(newY)),len(np.unique(newY))))
    
    w=[]  
    Coef_name=[] 
    count=0 
    y_pred=[]

    # Run model training, test and prediction 
    for i in range(random_num):
        # seperate the data into train and test 
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=random.randint(1,random_num),stratify=Y)

        count=count+1
        
        # train model
        if model_name=='svm':
            model = svm.SVC(kernel='linear',C=1, class_weight='balanced')
        else:  # default is random forest
            model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                    max_depth=None, max_features='sqrt', max_leaf_nodes=None,
                    min_impurity_decrease=0.0, min_impurity_split=None,
                    min_samples_leaf=1, min_samples_split=2,
                    min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,
                    oob_score=True, random_state=0, verbose=0, warm_start=False)
        
        model.fit(X_train, y_train)
        
        # model predict on test data set 
        y_pred0 = model.predict(X_test)
        conf_arr = metrics.confusion_matrix(y_test, y_pred0)
        conf_arr = conf_arr*100/ ( 1.0 * conf_arr.sum() )
        conf_matrix_XY = conf_matrix_XY+conf_arr
        accuracy_XY = accuracy_XY + accuracy_score(y_test, y_pred0)

        #model predict on a new data set
        y_pred_new = model.predict(newX)
        conf_arr = metrics.confusion_matrix(newY, y_pred_new)
        conf_arr = conf_arr*100/ ( 1.0 * conf_arr.sum() )
        conf_matrix_newXY = conf_matrix_newXY+conf_arr
        accuracy_newXY = accuracy_newXY + accuracy_score(newY, y_pred_new)

        if model_name=='svm':
            feature_coef = model.coef_[0]
        else:
            feature_coef = model.feature_importances_
        
        if len(w)==0:
            w = feature_coef
            Coef_name = range(X.shape[1])
            y_pred = y_pred_new
        else:
            Coef_name = np.concatenate((Coef_name,range(X.shape[1])),axis=None)
            w = np.concatenate((w,feature_coef),axis=None)
            y_pred = np.vstack((y_pred,y_pred_new))

    accuracy_XY = accuracy_XY/count
    conf_matrix_XY = conf_matrix_XY/count

    accuracy_newXY = accuracy_newXY/count
    conf_matrix_newXY = conf_matrix_newXY/count
    
    dict_y_pred=[]
    for ii in range(y_pred.shape[1]):
        a = list(y_pred[:,ii])
        dict_y_pred.append(dict((j, a.count(j)) for j in np.unique(y_test))) #{1:0.984,2:0.16}

    return accuracy_XY,conf_matrix_XY,accuracy_newXY,conf_matrix_newXY,w,Coef_name,dict_y_pred





