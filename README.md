# stock-portfolio


Implementation of Paper：Hypergraph-based Reinforcement Learning for Stock Portfolio Management

## Abstract



## Dataset

### Historical Price Data
We used a financial data API-https://tushare.pro to collect the historical price and relational data of stocks from China’s A-share market. We further performed a filtering step to eliminate those stocks that were traded on less than 98% of all trading days. This finally results in 758 stocks between 01/04/2013 and 12/31/2019.  

### Relation Data
We considered the industry-belonging of stocks. For the former, we grouped all stocks into 104 industry categories according to the Shenwan Industry Classification Standard. For the latter.

## Models



## Requirements

Python >= 3.6  
torch >= 1.4.0  
torchvision >= 0.1.8  
numpy  
sklearn  
  
## Hyperparameter Settings

Epoch: 600  
BatchSize: 32  
Learning Rate: 1e-3  
Dropout: 0.5
  
 ## Contact
 
If you have any questions, please contact lixiaojie199810@foxmail.com.
