# Portfolio-Optimizer

This project provides several standard portfolio optimization methods that are widely studied and used using *cvxpy* or *scipy.minimize*, in addition,  I also add some penalization customization to avoid overly concentraded result.

Test example can be seen in the main.py, the following provides the mathmatical formula for them.

Note:
- **X** is weight
- <img src="http://latex.codecogs.com/svg.latex?\Sigma" title="http://latex.codecogs.com/svg.latex?\Sigma" /> is covariance matrix
- <img src="http://latex.codecogs.com/svg.latex?\hat{\mu}" title="http://latex.codecogs.com/svg.latex?\hat{\mu}" /> is expected return
- **L** is the uncertain level
- <img src="http://latex.codecogs.com/svg.latex?\Theta" title="http://latex.codecogs.com/svg.latex?\Theta" /> is the estimated return standard error, which is *diag(diag(\Sigma) / shrink_size)*
- <img src="http://latex.codecogs.com/svg.latex?\lambda" title="http://latex.codecogs.com/svg.latex?\lambda" /> is the penalty
- <img src="http://latex.codecogs.com/svg.latex?\alpha" title="http://latex.codecogs.com/svg.latex?\alpha" /> is the confidence level for min_CVaR_loss


1. **target_return_robustMV**
<img src="http://latex.codecogs.com/svg.latex?Min:&space;X^T&space;\Sigma&space;X&space;\\\begin{cases}X^T&space;\cdot&space;\boldsymbol{1}&space;&=&space;1&space;\\X^T&space;\cdot&space;\hat{\mu}&space;-&space;L&space;*&space;||\Theta^{0.5}&space;X||_2&&space;>=&space;target\_return&space;\\X&space;&&space;\geq&space;lower\_bound&space;\\X&space;&&space;\leq&space;upper\_bound&space;\\\end{cases}"/>
