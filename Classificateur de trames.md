# MODELES DE MACHINE LEARNING DE CLASSIFICATION DES TRAMES RESEAU

## IMPORTATION DES LIBRAIRIES ET DU DATASET


```python
# cd "C:\Users\donfa\OneDrive\Desktop\PROJET RESEAU"
```

    C:\Users\donfa\OneDrive\Desktop\PROJET RESEAU
    


```python
import seaborn as sn
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier # la librairie doit etre importee
# from lightgbm import LGBMClassifier # la librairie doit etre importee
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
import pandas as pd
import numpy as np
data=pd.read_csv('FlowStatsfile.csv')
display(data)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp</th>
      <th>datapath_id</th>
      <th>flow_id</th>
      <th>ip_src</th>
      <th>tp_src</th>
      <th>ip_dst</th>
      <th>tp_dst</th>
      <th>ip_proto</th>
      <th>icmp_code</th>
      <th>icmp_type</th>
      <th>...</th>
      <th>idle_timeout</th>
      <th>hard_timeout</th>
      <th>flags</th>
      <th>packet_count</th>
      <th>byte_count</th>
      <th>packet_count_per_second</th>
      <th>packet_count_per_nsecond</th>
      <th>byte_count_per_second</th>
      <th>byte_count_per_nsecond</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.679206e+09</td>
      <td>4</td>
      <td>10.0.0.11010.0.0.601</td>
      <td>10.0.0.11</td>
      <td>0</td>
      <td>10.0.0.6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>20</td>
      <td>100</td>
      <td>0</td>
      <td>4</td>
      <td>392</td>
      <td>0.800000</td>
      <td>1.428571e-07</td>
      <td>7.840000e+01</td>
      <td>0.000014</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.679206e+09</td>
      <td>4</td>
      <td>10.0.0.6010.0.0.1101</td>
      <td>10.0.0.6</td>
      <td>0</td>
      <td>10.0.0.11</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
      <td>...</td>
      <td>20</td>
      <td>100</td>
      <td>0</td>
      <td>4</td>
      <td>392</td>
      <td>0.800000</td>
      <td>1.250000e-07</td>
      <td>7.840000e+01</td>
      <td>0.000012</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.679206e+09</td>
      <td>2</td>
      <td>10.0.0.1505010.0.0.6585906</td>
      <td>10.0.0.1</td>
      <td>5050</td>
      <td>10.0.0.6</td>
      <td>58590</td>
      <td>6</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>20</td>
      <td>100</td>
      <td>0</td>
      <td>123106</td>
      <td>8125380</td>
      <td>24621.200000</td>
      <td>2.159754e-03</td>
      <td>1.625076e+06</td>
      <td>0.142551</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.679206e+09</td>
      <td>2</td>
      <td>10.0.0.11505010.0.0.6585901</td>
      <td>10.0.0.11</td>
      <td>5050</td>
      <td>10.0.0.6</td>
      <td>58590</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>20</td>
      <td>100</td>
      <td>0</td>
      <td>4</td>
      <td>392</td>
      <td>0.800000</td>
      <td>1.739130e-07</td>
      <td>7.840000e+01</td>
      <td>0.000017</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.679206e+09</td>
      <td>2</td>
      <td>10.0.0.65859010.0.0.150506</td>
      <td>10.0.0.6</td>
      <td>58590</td>
      <td>10.0.0.1</td>
      <td>5050</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>20</td>
      <td>100</td>
      <td>0</td>
      <td>557949</td>
      <td>32038437970</td>
      <td>111589.800000</td>
      <td>8.999177e-03</td>
      <td>6.407688e+09</td>
      <td>516.749000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>898438</th>
      <td>1.679208e+09</td>
      <td>1</td>
      <td>94.7.138.1612154710.0.0.1806</td>
      <td>94.7.138.161</td>
      <td>21547</td>
      <td>10.0.0.1</td>
      <td>80</td>
      <td>6</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>20</td>
      <td>100</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>898439</th>
      <td>1.679208e+09</td>
      <td>3</td>
      <td>10.0.0.9010.0.0.901</td>
      <td>10.0.0.9</td>
      <td>0</td>
      <td>10.0.0.9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
      <td>...</td>
      <td>20</td>
      <td>100</td>
      <td>0</td>
      <td>3964622</td>
      <td>642268764</td>
      <td>495577.750000</td>
      <td>2.202568e-01</td>
      <td>8.028360e+07</td>
      <td>35.681598</td>
      <td>1</td>
    </tr>
    <tr>
      <th>898440</th>
      <td>1.679208e+09</td>
      <td>2</td>
      <td>10.0.0.9010.0.0.901</td>
      <td>10.0.0.9</td>
      <td>0</td>
      <td>10.0.0.9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
      <td>...</td>
      <td>20</td>
      <td>100</td>
      <td>0</td>
      <td>3965485</td>
      <td>642408570</td>
      <td>495685.625000</td>
      <td>4.236629e-03</td>
      <td>8.030107e+07</td>
      <td>0.686334</td>
      <td>1</td>
    </tr>
    <tr>
      <th>898441</th>
      <td>1.679208e+09</td>
      <td>2</td>
      <td>10.0.0.9010.0.0.901</td>
      <td>10.0.0.9</td>
      <td>0</td>
      <td>10.0.0.9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
      <td>...</td>
      <td>20</td>
      <td>100</td>
      <td>0</td>
      <td>7917400</td>
      <td>1282618800</td>
      <td>439855.555556</td>
      <td>8.449733e-03</td>
      <td>7.125660e+07</td>
      <td>1.368857</td>
      <td>1</td>
    </tr>
    <tr>
      <th>898442</th>
      <td>1.679208e+09</td>
      <td>3</td>
      <td>10.0.0.9010.0.0.901</td>
      <td>10.0.0.9</td>
      <td>0</td>
      <td>10.0.0.9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
      <td>...</td>
      <td>20</td>
      <td>100</td>
      <td>0</td>
      <td>7916537</td>
      <td>1282478994</td>
      <td>439807.611111</td>
      <td>4.166598e-01</td>
      <td>7.124883e+07</td>
      <td>67.498894</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>898443 rows × 22 columns</p>
</div>


## EXPLORATION DES DONNEES


```python
print(data.columns)
print(data.info())
print(data.isna().sum())
print(data.describe())

```

    Index(['timestamp', 'datapath_id', 'flow_id', 'ip_src', 'tp_src', 'ip_dst',
           'tp_dst', 'ip_proto', 'icmp_code', 'icmp_type', 'flow_duration_sec',
           'flow_duration_nsec', 'idle_timeout', 'hard_timeout', 'flags',
           'packet_count', 'byte_count', 'packet_count_per_second',
           'packet_count_per_nsecond', 'byte_count_per_second',
           'byte_count_per_nsecond', 'label'],
          dtype='object')
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 898443 entries, 0 to 898442
    Data columns (total 22 columns):
     #   Column                    Non-Null Count   Dtype  
    ---  ------                    --------------   -----  
     0   timestamp                 898443 non-null  float64
     1   datapath_id               898443 non-null  int64  
     2   flow_id                   898443 non-null  object 
     3   ip_src                    898443 non-null  object 
     4   tp_src                    898443 non-null  int64  
     5   ip_dst                    898443 non-null  object 
     6   tp_dst                    898443 non-null  int64  
     7   ip_proto                  898443 non-null  int64  
     8   icmp_code                 898443 non-null  int64  
     9   icmp_type                 898443 non-null  int64  
     10  flow_duration_sec         898443 non-null  int64  
     11  flow_duration_nsec        898443 non-null  int64  
     12  idle_timeout              898443 non-null  int64  
     13  hard_timeout              898443 non-null  int64  
     14  flags                     898443 non-null  int64  
     15  packet_count              898443 non-null  int64  
     16  byte_count                898443 non-null  int64  
     17  packet_count_per_second   898443 non-null  float64
     18  packet_count_per_nsecond  898443 non-null  float64
     19  byte_count_per_second     898443 non-null  float64
     20  byte_count_per_nsecond    898443 non-null  float64
     21  label                     898443 non-null  int64  
    dtypes: float64(5), int64(14), object(3)
    memory usage: 150.8+ MB
    None
    timestamp                   0
    datapath_id                 0
    flow_id                     0
    ip_src                      0
    tp_src                      0
    ip_dst                      0
    tp_dst                      0
    ip_proto                    0
    icmp_code                   0
    icmp_type                   0
    flow_duration_sec           0
    flow_duration_nsec          0
    idle_timeout                0
    hard_timeout                0
    flags                       0
    packet_count                0
    byte_count                  0
    packet_count_per_second     0
    packet_count_per_nsecond    0
    byte_count_per_second       0
    byte_count_per_nsecond      0
    label                       0
    dtype: int64
              timestamp    datapath_id         tp_src         tp_dst  \
    count  8.984430e+05  898443.000000  898443.000000  898443.000000   
    mean   1.679208e+09       3.660857   24423.457933     153.388389   
    std    1.456409e+02       1.437624   21877.163522    2210.749271   
    min    1.679206e+09       1.000000       0.000000       0.000000   
    25%    1.679208e+09       3.000000    1488.500000       0.000000   
    50%    1.679208e+09       4.000000   20110.000000       0.000000   
    75%    1.679208e+09       5.000000   43438.000000      80.000000   
    max    1.679208e+09       6.000000   65535.000000   60828.000000   
    
                ip_proto      icmp_code      icmp_type  flow_duration_sec  \
    count  898443.000000  898443.000000  898443.000000      898443.000000   
    mean        8.491539      -0.769574       1.056207           8.672932   
    std         6.287143       0.421106       3.776276           6.315401   
    min         1.000000      -1.000000      -1.000000           0.000000   
    25%         6.000000      -1.000000      -1.000000           3.000000   
    50%         6.000000      -1.000000      -1.000000          10.000000   
    75%        17.000000      -1.000000      -1.000000          14.000000   
    max        17.000000       0.000000       8.000000         100.000000   
    
           flow_duration_nsec  idle_timeout  hard_timeout     flags  packet_count  \
    count        8.984430e+05      898443.0      898443.0  898443.0  8.984430e+05   
    mean         4.918402e+08          20.0         100.0       0.0  6.525228e+02   
    std          2.882243e+08           0.0           0.0       0.0  2.619871e+04   
    min          0.000000e+00          20.0         100.0       0.0  0.000000e+00   
    25%          2.430000e+08          20.0         100.0       0.0  0.000000e+00   
    50%          4.860000e+08          20.0         100.0       0.0  0.000000e+00   
    75%          7.410000e+08          20.0         100.0       0.0  0.000000e+00   
    max          9.990000e+08          20.0         100.0       0.0  7.917400e+06   
    
             byte_count  packet_count_per_second  packet_count_per_nsecond  \
    count  8.984430e+05            898443.000000             898443.000000   
    mean   2.946058e+07                52.588394                  0.000003   
    std    1.269145e+09              2109.692227                  0.000518   
    min    0.000000e+00                 0.000000                  0.000000   
    25%    0.000000e+00                 0.000000                  0.000000   
    50%    0.000000e+00                 0.000000                  0.000000   
    75%    0.000000e+00                 0.000000                  0.000000   
    max    7.112308e+10            495685.625000                  0.416660   
    
           byte_count_per_second  byte_count_per_nsecond          label  
    count           8.984430e+05           898443.000000  898443.000000  
    mean            2.365993e+06                0.120602       0.993030  
    std             1.043453e+08                8.075328       0.083194  
    min             0.000000e+00                0.000000       0.000000  
    25%             0.000000e+00                0.000000       1.000000  
    50%             0.000000e+00                0.000000       1.000000  
    75%             0.000000e+00                0.000000       1.000000  
    max             8.067727e+09             2109.150948       1.000000  
    


```python
num=data.select_dtypes(include='number').columns.values
print(data['label'].value_counts())
x=data[['timestamp','datapath_id','tp_src','tp_dst','ip_proto','icmp_code'
 ,'icmp_type','flow_duration_sec','flow_duration_nsec','idle_timeout'
 ,'hard_timeout','flags','packet_count','byte_count'
 ,'packet_count_per_second','packet_count_per_nsecond'
 ,'byte_count_per_second','byte_count_per_nsecond']]
y=data['label']
smote=SMOTE()
x_sampled,y_sampled=smote.fit_resample(x,y)
print(x_sampled)
print('-----')
print(y_sampled)

'''for i in x_sampled.columns.values:
    sn.boxplot(data[i])
    plt.show()'''


```

    label
    1    892181
    0      6262
    Name: count, dtype: int64
                timestamp  datapath_id  tp_src  tp_dst  ip_proto  icmp_code  \
    0        1.679206e+09            4       0       0         1          0   
    1        1.679206e+09            4       0       0         1          0   
    2        1.679206e+09            2    5050   58590         6         -1   
    3        1.679206e+09            2    5050   58590         1          0   
    4        1.679206e+09            2   58590    5050         6          0   
    ...               ...          ...     ...     ...       ...        ...   
    1784357  1.679207e+09            3      41   18327         3          0   
    1784358  1.679207e+09            3      80   35106         6         -1   
    1784359  1.679207e+09            2   32928    2661         6          0   
    1784360  1.679208e+09            3   33107   14536         6          0   
    1784361  1.679207e+09            1   49489    1470         9          0   
    
             icmp_type  flow_duration_sec  flow_duration_nsec  idle_timeout  \
    0                0                  5            28000000            20   
    1                8                  5            32000000            20   
    2               -1                  5            57000000            20   
    3                0                  5            23000000            20   
    4                0                  5            62000000            20   
    ...            ...                ...                 ...           ...   
    1784357          0                 39           451000000            20   
    1784358         -1                 10           332096248            20   
    1784359          7                  8           959000000            20   
    1784360          4                 39           537000000            20   
    1784361          0                  0           788000000            20   
    
             hard_timeout  flags  packet_count   byte_count  \
    0                 100      0             4          392   
    1                 100      0             4          392   
    2                 100      0        123106      8125380   
    3                 100      0             4          392   
    4                 100      0        557949  32038437970   
    ...               ...    ...           ...          ...   
    1784357           100      0            40         4065   
    1784358           100      0             4          584   
    1784359           100      0             4          427   
    1784360           100      0            36         3612   
    1784361           100      0            52        78312   
    
             packet_count_per_second  packet_count_per_nsecond  \
    0                       0.800000              1.428571e-07   
    1                       0.800000              1.250000e-07   
    2                   24621.200000              2.159754e-03   
    3                       0.800000              1.739130e-07   
    4                  111589.800000              8.999177e-03   
    ...                          ...                       ...   
    1784357                 1.507503              8.975883e-08   
    1784358                 0.713631              1.204471e-08   
    1784359                 0.609181              4.692558e-09   
    1784360                 0.646775              6.865213e-08   
    1784361                25.881368              6.568875e-08   
    
             byte_count_per_second  byte_count_per_nsecond  
    0                 7.840000e+01            1.400000e-05  
    1                 7.840000e+01            1.225000e-05  
    2                 1.625076e+06            1.425505e-01  
    3                 7.840000e+01            1.704348e-05  
    4                 6.407688e+09            5.167490e+02  
    ...                        ...                     ...  
    1784357           1.970393e+02            9.015009e-06  
    1784358           1.041901e+02            1.758528e-06  
    1784359           5.789004e+01            4.458675e-07  
    1784360           6.338399e+01            6.727909e-06  
    1784361           3.913263e+04            9.932139e-05  
    
    [1784362 rows x 18 columns]
    -----
    0          0
    1          0
    2          0
    3          0
    4          0
              ..
    1784357    0
    1784358    0
    1784359    0
    1784360    0
    1784361    0
    Name: label, Length: 1784362, dtype: int64
    




    'for i in x_sampled.columns.values:\n    sn.boxplot(data[i])\n    plt.show()'




```python
plt.figure(figsize=(17,6))
corr = data.corr(method='kendall')
my_m=np.triu(corr)
sn.heatmap(corr, mask=my_m, annot=True, cmap="Set2")
plt.show()

'''for i in data.select_dtypes(include='number').columns.values:
    for j in data.select_dtypes(include='number').columns.values:
        plt.plot(data[i],marker='o',label=f"{i}",color='red')
        plt.plot(data[j],marker='x',label=f"{j}",color="blue")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()'''


```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[9], line 2
          1 plt.figure(figsize=(17,6))
    ----> 2 corr = data.corr(method='kendall')
          3 my_m=np.triu(corr)
          4 sn.heatmap(corr, mask=my_m, annot=True, cmap="Set2")
    

    File C:\ProgramData\anaconda3\Lib\site-packages\pandas\core\frame.py:11049, in DataFrame.corr(self, method, min_periods, numeric_only)
      11047 cols = data.columns
      11048 idx = cols.copy()
    > 11049 mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
      11051 if method == "pearson":
      11052     correl = libalgos.nancorr(mat, minp=min_periods)
    

    File C:\ProgramData\anaconda3\Lib\site-packages\pandas\core\frame.py:1993, in DataFrame.to_numpy(self, dtype, copy, na_value)
       1991 if dtype is not None:
       1992     dtype = np.dtype(dtype)
    -> 1993 result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
       1994 if result.dtype is not dtype:
       1995     result = np.asarray(result, dtype=dtype)
    

    File C:\ProgramData\anaconda3\Lib\site-packages\pandas\core\internals\managers.py:1694, in BlockManager.as_array(self, dtype, copy, na_value)
       1692         arr.flags.writeable = False
       1693 else:
    -> 1694     arr = self._interleave(dtype=dtype, na_value=na_value)
       1695     # The underlying data was copied within _interleave, so no need
       1696     # to further copy if copy=True or setting na_value
       1698 if na_value is lib.no_default:
    

    File C:\ProgramData\anaconda3\Lib\site-packages\pandas\core\internals\managers.py:1753, in BlockManager._interleave(self, dtype, na_value)
       1751     else:
       1752         arr = blk.get_values(dtype)
    -> 1753     result[rl.indexer] = arr
       1754     itemmask[rl.indexer] = 1
       1756 if not itemmask.all():
    

    ValueError: could not convert string to float: '10.0.0.11010.0.0.601'



    <Figure size 1700x600 with 0 Axes>



```python
data['z-scores']=(data.timestamp-data.timestamp.mean())/(data.timestamp.std())
df=data[(data['z-scores']<3)&(data['z-scores']>-3)]
q1=df.timestamp.quantile(0.28)
q3=df.timestamp.quantile(0.72)
iqr=q3-q1
u=q3+1.5*iqr
l=q1-1.5*iqr
df=df[(df.timestamp > l )&(df.timestamp < u)]

```


```python

df['z-scores']=(df.flow_duration_sec-df.flow_duration_sec.mean())/(df.flow_duration_sec.std())
df=df[(df['z-scores']<3)&(df['z-scores']>-3)]
q_1=df.flow_duration_sec.quantile(0.25)
q_3=df.flow_duration_sec.quantile(0.75)
iq_r=q_3-q_1
upp=q_3+1.5*iq_r
low=q_1-1.5*iq_r
df=df[(df.flow_duration_sec > low )&(df.flow_duration_sec < upp)]

print(data.select_dtypes(include='number').columns.values)
'''for i in x_sampled.columns.values:
    sn.boxplot(df[i])
    plt.show()'''


```

    ['timestamp' 'datapath_id' 'tp_src' 'tp_dst' 'ip_proto' 'icmp_code'
     'icmp_type' 'flow_duration_sec' 'flow_duration_nsec' 'idle_timeout'
     'hard_timeout' 'flags' 'packet_count' 'byte_count'
     'packet_count_per_second' 'packet_count_per_nsecond'
     'byte_count_per_second' 'byte_count_per_nsecond' 'label' 'z-scores']
    




    'for i in x_sampled.columns.values:\n    sn.boxplot(df[i])\n    plt.show()'




```python
colones=['timestamp','datapath_id','tp_src','tp_dst','ip_proto','icmp_code'
 ,'icmp_type','flow_duration_sec','flow_duration_nsec','idle_timeout'
 ,'hard_timeout','flags','packet_count','byte_count'
 ,'packet_count_per_second','packet_count_per_nsecond'
 ,'byte_count_per_second','byte_count_per_nsecond']

x=df[['timestamp','datapath_id','tp_src','tp_dst','ip_proto','icmp_code'
 ,'icmp_type','flow_duration_sec','flow_duration_nsec','idle_timeout'
 ,'hard_timeout','flags','packet_count','byte_count'
 ,'packet_count_per_second','packet_count_per_nsecond'
 ,'byte_count_per_second','byte_count_per_nsecond']]
y=df[['label']]

'''x=data[['timestamp','datapath_id','tp_src','tp_dst','ip_proto','icmp_code'
 ,'hard_timeout','flags','packet_count','byte_count'
 ,'packet_count_per_second','packet_count_per_nsecond'
 ,'byte_count_per_second','byte_count_per_nsecond']]
y=data[['label']]'''

x_train,x_test,y_train,y_test=train_test_split(x,y)

```


```python

```

## TESTS DE DIFFERENTS ALGORITHMES ET MESURES DE SCORES

#### REGRESSION LOGISTIQUE


```python
lr=LogisticRegression()
lr.fit(x_train,y_train)
print('The logistic regression: ',lr.score(x_sampled,y_sampled))
```

    C:\ProgramData\anaconda3\Lib\site-packages\sklearn\utils\validation.py:1300: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    C:\ProgramData\anaconda3\Lib\site-packages\sklearn\linear_model\_logistic.py:469: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    

    The logistic regression:  0.5132394659827995
    

#### CLASSIFICATEUR XGBOOST


```python
# xgb=XGBClassifier()
# xgb.fit(x_train,y_train)
# print("the Xgb : ",xgb.score(x_sampled,y_sampled))
# 

```

#### CLASSIFICATEUR LMBG


```python
# lgb=LGBMClassifier()
# lgb.fit(x_train,y_train)
# print('The LGB',lgb.score(x_sampled,y_sampled))


```

#### ARBRE DE DESCISION


```python
import  joblib

tree=DecisionTreeClassifier()
tree.fit(x_train,y_train)
print('Dtree ',tree.score(x_sampled,y_sampled))
# Exportation du modèle
joblib.dump(tree, 'decision_tree_model.pkl')
print("Modèle exporté sous le nom 'decision_tree_model.pkl'")

```

    Dtree  1.0
    Modèle exporté sous le nom 'decision_tree_model.pkl'
    

#### RANDOM FOREST


```python
rforest=RandomForestClassifier()
rforest.fit(x_train,y_train)
print('The random forest: ',rforest.score(x_sampled,y_sampled))


```

    C:\ProgramData\anaconda3\Lib\site-packages\sklearn\base.py:1474: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)
    

    The random forest:  1.0
    

#### ADABOOST CLASSIFICATEUR


```python
adb=AdaBoostClassifier()
adb.fit(x_train,y_train)
print('the adb ',adb.score(x_sampled,y_sampled))

```

    C:\ProgramData\anaconda3\Lib\site-packages\sklearn\utils\validation.py:1300: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    C:\ProgramData\anaconda3\Lib\site-packages\sklearn\ensemble\_weight_boosting.py:519: FutureWarning: The SAMME.R algorithm (the default) is deprecated and will be removed in 1.6. Use the SAMME algorithm to circumvent this warning.
      warnings.warn(
    

    the adb  1.0
    

#### GRADIENT BOOSTING 


```python
grb=GradientBoostingClassifier()
grb.fit(x_train,y_train)
print('Gradient boosting ',grb.score(x_sampled,y_sampled))


```

    C:\ProgramData\anaconda3\Lib\site-packages\sklearn\preprocessing\_label.py:114: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    

    Gradient boosting  1.0
    

#### CLASSIFICATEUR BAGGING


```python
bag=BaggingClassifier()
bag.fit(x_train,y_train)
print('Bagging',bag.score(x_sampled,y_sampled))
```

    C:\ProgramData\anaconda3\Lib\site-packages\sklearn\ensemble\_bagging.py:782: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    

    Bagging 1.0
    


```python
import joblib
import pandas as pd

def predict_label(data):
    """
    Charge le modèle de classification et prédit les labels pour les données fournies.
    
    Args:
        data (pd.DataFrame): Un DataFrame contenant les colonnes suivantes :
            ['timestamp', 'datapath_id', 'tp_src', 'tp_dst', 'ip_proto', 
             'icmp_code', 'icmp_type', 'flow_duration_sec', 'flow_duration_nsec',
             'idle_timeout', 'hard_timeout', 'flags', 'packet_count', 
             'byte_count', 'packet_count_per_second', 'packet_count_per_nsecond',
             'byte_count_per_second', 'byte_count_per_nsecond']
    
    Returns:
        pd.Series: Les prédictions du modèle (0 pour légitime, 1 pour attaque).
    """
    # Charger le modèle exporté
    model = joblib.load('decision_tree_model.pkl')
    
    # Assurez-vous que les colonnes nécessaires sont présentes
    required_columns = [
        'timestamp', 'datapath_id', 'tp_src', 'tp_dst', 'ip_proto',
        'icmp_code', 'icmp_type', 'flow_duration_sec', 'flow_duration_nsec',
        'idle_timeout', 'hard_timeout', 'flags', 'packet_count', 
        'byte_count', 'packet_count_per_second', 'packet_count_per_nsecond',
        'byte_count_per_second', 'byte_count_per_nsecond'
    ]
    
    if not all(col in data.columns for col in required_columns):
        raise ValueError("Les colonnes nécessaires ne sont pas présentes dans les données d'entrée.")
    
    # Filtrer les colonnes nécessaires pour les prédictions
    data_filtered = data[required_columns]
    
    # Faire des prédictions
    predictions = model.predict(data_filtered)
    return pd.Series(predictions, index=data.index, name='Predictions')

```


```python
# Exemple de données
sample_data = pd.DataFrame({
    'timestamp': [123456789, 987654321],
    'datapath_id': [1, 2],
    'tp_src': [80, 8080],
    'tp_dst': [443, 53],
    'ip_proto': [6, 17],
    'icmp_code': [0, 0],
    'icmp_type': [8, 0],
    'flow_duration_sec': [5, 10],
    'flow_duration_nsec': [200, 400],
    'idle_timeout': [30, 60],
    'hard_timeout': [60, 120],
    'flags': [2, 18],
    'packet_count': [50, 100],
    'byte_count': [2048, 4096],
    'packet_count_per_second': [10, 20],
    'packet_count_per_nsecond': [1000, 2000],
    'byte_count_per_second': [500, 800],
    'byte_count_per_nsecond': [5000, 10000]
})

# Faire des prédictions
predictions = predict_label(sample_data)
print(predictions)

```

    0    0
    1    0
    Name: Predictions, dtype: int64
    


```python
import joblib

def predict_frame(frame):
    """
    Prédit le type d'une trame (légitime ou attaque) à partir de ses caractéristiques.
    
    Args:
        frame (dict): Un dictionnaire contenant les caractéristiques suivantes :
            {
                'timestamp': float,
                'datapath_id': int,
                'tp_src': int,
                'tp_dst': int,
                'ip_proto': int,
                'icmp_code': int,
                'icmp_type': int,
                'flow_duration_sec': float,
                'flow_duration_nsec': float,
                'idle_timeout': int,
                'hard_timeout': int,
                'flags': int,
                'packet_count': int,
                'byte_count': int,
                'packet_count_per_second': float,
                'packet_count_per_nsecond': float,
                'byte_count_per_second': float,
                'byte_count_per_nsecond': float
            }
    
    Returns:
        int: 0 si la trame est légitime, 1 si elle correspond à une attaque.
    """
    # Charger le modèle exporté
    model = joblib.load('decision_tree_model.pkl')
    
    # Vérifier que toutes les caractéristiques nécessaires sont présentes
    required_keys = [
        'timestamp', 'datapath_id', 'tp_src', 'tp_dst', 'ip_proto',
        'icmp_code', 'icmp_type', 'flow_duration_sec', 'flow_duration_nsec',
        'idle_timeout', 'hard_timeout', 'flags', 'packet_count', 
        'byte_count', 'packet_count_per_second', 'packet_count_per_nsecond',
        'byte_count_per_second', 'byte_count_per_nsecond'
    ]
    
    if not all(key in frame for key in required_keys):
        raise ValueError("Les caractéristiques suivantes manquent dans la trame : " +
                         ", ".join([key for key in required_keys if key not in frame]))
    
    # Organiser les caractéristiques dans le même ordre que lors de l'entraînement
    features = [
        frame['timestamp'], frame['datapath_id'], frame['tp_src'], frame['tp_dst'],
        frame['ip_proto'], frame['icmp_code'], frame['icmp_type'], 
        frame['flow_duration_sec'], frame['flow_duration_nsec'], 
        frame['idle_timeout'], frame['hard_timeout'], frame['flags'], 
        frame['packet_count'], frame['byte_count'], 
        frame['packet_count_per_second'], frame['packet_count_per_nsecond'], 
        frame['byte_count_per_second'], frame['byte_count_per_nsecond']
    ]
    
    # Prédire le type de la trame
    
    predictions = model.predict([features])[0]
    print(predictions)
    prediction=predictions[0]
    return prediction

```


```python
# Exemple d'une trame à analyser
frame = {
    'timestamp': 123456789.0,
    'datapath_id': 1,
    'tp_src': 80,
    'tp_dst': 443,
    'ip_proto': 6,
    'icmp_code': 0,
    'icmp_type': 0,
    'flow_duration_sec': 5.0,
    'flow_duration_nsec': 200.0,
    'idle_timeout': 30,
    'hard_timeout': 60,
    'flags': 2,
    'packet_count': 50,
    'byte_count': 2048,
    'packet_count_per_second': 10.0,
    'packet_count_per_nsecond': 1000.0,
    'byte_count_per_second': 500.0,
    'byte_count_per_nsecond': 5000.0
}

# Prédire le type de la trame
result = predict_frame(frame)
if result == 0:
    print("Trame légitime")
else:
    print("Trame correspondant à une attaque")

```

    0
    

    C:\ProgramData\anaconda3\Lib\site-packages\sklearn\base.py:493: UserWarning: X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names
      warnings.warn(
    


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    Cell In[25], line 24
          2 frame = {
          3     'timestamp': 123456789.0,
          4     'datapath_id': 1,
       (...)
         20     'byte_count_per_nsecond': 5000.0
         21 }
         23 # Prédire le type de la trame
    ---> 24 result = predict_frame(frame)
         25 if result == 0:
         26     print("Trame légitime")
    

    Cell In[24], line 64, in predict_frame(frame)
         62 predictions = model.predict([features])[0]
         63 print(predictions)
    ---> 64 prediction=predictions[0]
         65 return prediction
    

    IndexError: invalid index to scalar variable.



```python
import pandas as pd
import joblib

def predict_frame(frame):
    """
    Prédit le type d'une trame (légitime ou attaque) à partir de ses caractéristiques.
    
    Args:
        frame (dict): Un dictionnaire contenant les caractéristiques nécessaires.

    Returns:
        int: 0 si la trame est légitime, 1 si elle correspond à une attaque.
    """
    # Charger le modèle exporté
    model = joblib.load('decision_tree_model.pkl')
    
    # Vérifier que toutes les caractéristiques nécessaires sont présentes
    required_keys = [
        'timestamp', 'datapath_id', 'tp_src', 'tp_dst', 'ip_proto',
        'icmp_code', 'icmp_type', 'flow_duration_sec', 'flow_duration_nsec',
        'idle_timeout', 'hard_timeout', 'flags', 'packet_count', 
        'byte_count', 'packet_count_per_second', 'packet_count_per_nsecond',
        'byte_count_per_second', 'byte_count_per_nsecond'
    ]
    
    if not all(key in frame for key in required_keys):
        raise ValueError("Les caractéristiques suivantes manquent dans la trame : " +
                         ", ".join([key for key in required_keys if key not in frame]))
    
    # Convertir le dictionnaire en DataFrame avec une ligne
    frame_df = pd.DataFrame([frame])
    
    # Prédire le type de la trame
    predictions = model.predict(frame_df)
    print(predictions)
    prediction=predictions[0]
    return prediction

```


```python
# Exemple d'une trame à analyser
frame = {
    'timestamp': 9336945550,
    'datapath_id': 1,
    'tp_src': 80,
    'tp_dst': 443,
    'ip_proto': 6,
    'icmp_code': 0,
    'icmp_type': 8,
    'flow_duration_sec': 50,
    'flow_duration_nsec': 20,
    'idle_timeout': 30,
    'hard_timeout': 60,
    'flags': 2,
    'packet_count': 50,
    'byte_count': 2048,
    'packet_count_per_second': 10,
    'packet_count_per_nsecond': 1,
    'byte_count_per_second': 500,
    'byte_count_per_nsecond': 50
}

# Prédire le type de la trame
result = predict_frame(frame)
if result == 0:
    print("Trame légitime")
else:
    print("Trame correspondant à une attaque")

```

    [1]
    Trame correspondant à une attaque
    


```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Simulation of the dataset structure
# Assuming 'df' already exists with the required structure
columns = ['timestamp', 'datapath_id', 'tp_src', 'tp_dst', 'ip_proto', 'icmp_code',
           'icmp_type', 'flow_duration_sec', 'flow_duration_nsec', 'idle_timeout',
           'hard_timeout', 'flags', 'packet_count', 'byte_count',
           'packet_count_per_second', 'packet_count_per_nsecond',
           'byte_count_per_second', 'byte_count_per_nsecond', 'label']

# Mock data for demonstration purposes
np.random.seed(42)
n_samples = 200
# df = pd.DataFrame({
#     'timestamp': np.random.uniform(1, 1000, n_samples),
#     'datapath_id': np.random.randint(1, 5, n_samples),
#     'tp_src': np.random.randint(1, 65535, n_samples),
#     'tp_dst': np.random.randint(1, 65535, n_samples),
#     'ip_proto': np.random.randint(1, 256, n_samples),
#     'icmp_code': np.random.randint(0, 10, n_samples),
#     'icmp_type': np.random.randint(0, 10, n_samples),
#     'flow_duration_sec': np.random.uniform(0, 100, n_samples),
#     'flow_duration_nsec': np.random.uniform(0, 100, n_samples),
#     'idle_timeout': np.random.randint(1, 60, n_samples),
#     'hard_timeout': np.random.randint(1, 120, n_samples),
#     'flags': np.random.randint(0, 4, n_samples),
#     'packet_count': np.random.randint(1, 1000, n_samples),
#     'byte_count': np.random.randint(1, 100000, n_samples),
#     'packet_count_per_second': np.random.uniform(1, 500, n_samples),
#     'packet_count_per_nsecond': np.random.uniform(1, 1000, n_samples),
#     'byte_count_per_second': np.random.uniform(1, 1000, n_samples),
#     'byte_count_per_nsecond': np.random.uniform(1, 10000, n_samples),
#     'label': np.random.choice([0, 1], n_samples)
# })

# Extract features and labels
x = df[['timestamp', 'datapath_id', 'tp_src', 'tp_dst', 'ip_proto', 'icmp_code',
        'icmp_type', 'flow_duration_sec', 'flow_duration_nsec', 'idle_timeout',
        'hard_timeout', 'flags', 'packet_count', 'byte_count',
        'packet_count_per_second', 'packet_count_per_nsecond',
        'byte_count_per_second', 'byte_count_per_nsecond']]
y = df[['label']]

# Select relevant features for clustering
features_for_clustering = ['timestamp', 'packet_count_per_second', 'byte_count_per_second']
X_selected = x[features_for_clustering]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with cluster labels
scatter = ax.scatter(X_selected['timestamp'], 
                     X_selected['packet_count_per_second'], 
                     X_selected['byte_count_per_second'], 
                     c=clusters, cmap='viridis', s=50)

# Labels and legend
ax.set_title('3D Clustering Visualization', fontsize=14)
ax.set_xlabel('Timestamp', fontsize=12)
ax.set_ylabel('Packet Count Per Second', fontsize=12)
ax.set_zlabel('Byte Count Per Second', fontsize=12)
plt.colorbar(scatter, ax=ax, label='Cluster')

plt.show()

```


    
![png](Classificateur%20de%20trames_files/Classificateur%20de%20trames_35_0.png)
    



```python

```
