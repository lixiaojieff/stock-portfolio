# stock-portfolio


Implementation of Paper：Hypergraph-based Reinforcement Learning for Stock Portfolio Selection

## Abstract

Stock portfolio selection is an important financial planning task that dynamically re-allocates the investments to stock assets to achieve the goals such as maximal profits or minimal risks. In this paper, we propose a hypergraph-based reinforcement learning method for stock portfolio selection, in which the fundamental issue is to learn a policy function generating appropriate trading actions given the current environments. The historical time-series patterns of stocks are firstly captured. Then, different from prior works ignoring or implicitly modeling stock pairwise correlations, we present a HyperGraph Attention Module (HGAM) in the portfolio policy learning, which utilizes the hypergraph structure to explicitly model the group-wise industry-belonging relationships among stocks. The attention mechanism is also introduced in HGAM that quantifies the importance of different neighbors regarding the target node to aggregate the information on the stock hypergraph adaptively. Extensive experiments on the real-world dataset collected from China’s A-share market demonstrate the significant superiority of our method, compared with state-of-the-art methods in portfolio selection, including both online learning-based methods and reinforcement learning-based methods.

## Dataset

### Historical Price Data
We used a financial data API-https://tushare.pro to collect the historical price and relational data of stocks from China’s A-share market. We further performed a filtering step to eliminate those stocks that were traded on less than 98% of all trading days. This finally results in 758 stocks between 01/04/2013 and 12/31/2019.  

### Relation Data
We considered the industry-belonging of stocks. For the former, we grouped all stocks into 104 industry categories according to the Shenwan Industry Classification Standard. For the latter.

## Models

  * `/HGAM/models.py`: the policy network framework;
  * `/HGAM/module.py`: implementation of hypergraph attention module;   
  * `/HGAM/layers.py`: implementation of attention layer;   
  * `/training/train.py`: train the overall portfolio policy network; 
  * `/loss/batch_loss.py`: maximizing the accumulated logarithmic returns in reinforcement learning.

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
