# Portfolio-Optimizer

This project provides several standard portfolio optimization methods that are widely studied and used using *cvxpy* or *scipy.minimize*, in addition,  I also add some penalization customization to avoid overly concentraded and sensitive result.

Test example can be seen in the main.py, the following provides the mathmatical formula for them.

Note:
- **X** is weight
- <img src="http://latex.codecogs.com/svg.latex?\Sigma" title="http://latex.codecogs.com/svg.latex?\Sigma" /> is covariance matrix
- <img src="http://latex.codecogs.com/svg.latex?\hat{\mu}" title="http://latex.codecogs.com/svg.latex?\hat{\mu}" /> is expected return
- **L** is the uncertain level
- <img src="http://latex.codecogs.com/svg.latex?\Theta" title="http://latex.codecogs.com/svg.latex?\Theta" /> is the estimated return standard error, which is <img src="https://latex.codecogs.com/gif.latex?\frac{diag(diag(\Sigma))}{\text{shrink&space;size}}" title="\frac{diag(diag(\Sigma))}{\text{shrink size}}" />
- <img src="http://latex.codecogs.com/svg.latex?\lambda" title="http://latex.codecogs.com/svg.latex?\lambda" /> is the return penalty parameter
- <img src="http://latex.codecogs.com/svg.latex?\alpha" title="http://latex.codecogs.com/svg.latex?\alpha" /> is the confidence level for min_CVaR_loss
- <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{l_t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{r_t}" title="\mathbf{r_t}" /></a> is a n by 1 array of loss of day t for each asset in the portfolio


if not specified, the following optimizations all have constraints below:

<!-- <img src="http://latex.codecogs.com/svg.latex?\begin{align*}&X^T&space;\boldsymbol{1}&space;=&space;1&space;\\&X&space;\geq&space;lower\_bound&space;\\&X&space;\leq&space;upper\_bound&space;\\\end{align*}&space;" title="http://latex.codecogs.com/svg.latex?\begin{align*}&X^T \boldsymbol{1} = 1 \\&X \geq lower\_bound \\&X \leq upper\_bound \\\end{align*} " /> -->

<img src="https://latex.codecogs.com/gif.latex?\begin{cases}&space;X^T&space;\mathbf{1}&space;=&space;1&space;\\&space;X&space;\leq&space;\text{upper&space;bound}&space;\\&space;X&space;\geq&space;\text{lower&space;bound}&space;\\&space;\end{cases}" title="\begin{cases} X^T \mathbf{1} = 1 \\ X \leq upper\_bound \\ X \geq lower\_bound \\ \end{cases}" />


-  **target_return_robustMV**


<img src="https://latex.codecogs.com/gif.latex?\underset{X}{\textbf{Min:&space;}}&space;X^T&space;\Sigma&space;X&space;\\&space;\text{s.t.}&space;\quad&space;X^T&space;\hat{\mu}&space;-&space;L&space;||\Theta^{\frac{1}{2}}X||_2&space;\geq&space;\text{target&space;return}" title="\underset{X}{\textbf{Min: }} X^T \Sigma X \\ \text{s.t.} \quad X^T \hat{\mu} - \lambda ||\Theta^{\frac{1}{2}}X||_2 \geq \text{target return}" />

This is a typical Makowitz mean-variance problem in which target return is sat as a constraint to satisfy and then minimize the variance. In order to generate stable robust result, I add a L2 penalization term which relates to the standard error of estimated return to represent the low confidence towards estimated return.

- **penalized_robustMV**

<img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\begin{align*}&&space;\underset{X}{\textbf{Min:&space;}}&space;X^T&space;\Sigma&space;X&space;-&space;\lambda&space;\hat{\mu}^T&space;X&space;&plus;&space;L&space;*&space;||\Theta^{0.5}X||_2&space;\\\end{align*}&space;&space;&space;&space;" title="http://latex.codecogs.com/png.latex?\dpi{110} \begin{align*}& \underset{X}{\textbf{Min: }} X^T \Sigma X - \lambda \hat{\mu}^T X + L * ||\Theta^{0.5}X||_2 \\\end{align*} " />

Similar to the previous problem, instead of setting the hard requirement for the return, we can turn it into another penalization term in the objective to make a trade-off between risk, return, and stable result.

- **risk_parity**

<img src="https://latex.codecogs.com/gif.latex?\underset{X,\theta}{\textbf{Min:&space;}}&space;\sum_{i=1}^n&space;(X_i&space;(\Sigma&space;X)_i&space;-&space;\theta)^2&space;\\&space;\text{s.t.}&space;\quad&space;X&space;\geq&space;0" title="\underset{X,\theta}{\textbf{Min: }} \sum_{i=1}^n (X_i (\Sigma X)_i - \theta)^2 \\ \text{s.t.} \quad X \geq 0" />

<img src="https://latex.codecogs.com/gif.latex?X_i&space;(\Sigma&space;X)_i" title="X_i (\Sigma X)_i" /> is the variance contribution from the ith asset, we want to equalize the contribution from each asset (based on its variance and covariance with other assets). If no boundary condition affects, the risk contribution of each asset will be the auxiliary variable <img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" />.
note: In order to have unique solution, we can only allow long assets. 

- **max_diver_ratio**

<img src="https://latex.codecogs.com/gif.latex?\underset{X}{\textbf{Max:&space;}}&space;\frac{\sum_{i=1}^n&space;(x_i&space;(\sqrt{\Sigma_{ii}}&plus;\lambda)&space;)}{\sqrt{X^T&space;\Sigma&space;X}}&space;\\" title="\underset{X}{\textbf{Max: }} \frac{\sum_{i=1}^n (X_i (\sqrt{\Sigma_{ii}}+\lambda) )}{\sqrt{X^T \Sigma X}} \\" />

Diversition ratio is defined as weighted volatility contribution from diagonal elements as a proportion of portfolio volatility. A penalization 
(adjustment, usually > 0) on the estimated covariance matrix is added to avoid overly concentrated result.

- **min_CVaR_loss**

<img src="https://latex.codecogs.com/gif.latex?\underset{X,\gamma,z_t}{\textbf{Min:&space;}}&space;\gamma&space;&plus;&space;\frac{1}{(1-\alpha)T}&space;\sum_{t=1}^T&space;z_t&space;\\&space;\text{s.t.}&space;\quad&space;\begin{align*}&space;z_t&space;&\geq&space;0,&space;\quad&space;&t&space;=&space;1,&space;2,&space;\dots,&space;T&space;\\&space;z_t&space;&\geq&space;X^T&space;\mathbf{l_t}&space;-&space;\gamma,&space;\quad&space;&t&space;=&space;1,&space;2,&space;\dots,&space;T&space;\\&space;\end{align*}" title="\underset{X,\gamma,z_t}{\textbf{Min: }} \gamma + \frac{1}{(1-\alpha)T} \sum_{t=1}^T z_t \\ \text{s.t.} \quad \begin{align*} z_t &\geq 0, \quad &t = 1, 2, \dots, T \\ z_t &\geq X^T \mathbf{r_t} - \gamma, \quad &t = 1, 2, \dots, T \\ \end{align*}" />

This problem devotes to minimize Conditional Value-at-Risk at alpha confidence level. <img src="https://latex.codecogs.com/gif.latex?\mathbf{z}_t,\gamma" title="\mathbf{z}_t,\gamma" /> are auxiliar variables which help turn the problem into a linear programming problem. After the optimization, <img src="https://latex.codecogs.com/gif.latex?\gamma" title="" /> will be the Value-at-Risk at alpha confidence level, <img src="https://latex.codecogs.com/gif.latex?\mathbf{z}_t"  title="" /> will either be zero or excess loss over VaR at time t. 
More details can be seen at the author's paper below.
> Rockafellar, R. Tyrrell, and Stanislav Uryasev. "Optimization of conditional value-at-risk." Journal of risk 2 (2000): 21-42.

- **inverse_vol**

<img src="https://latex.codecogs.com/gif.latex?x_i&space;=&space;\begin{cases}&space;\frac{\sigma_i}{\sum_{i=1}^n&space;\sigma_i},&&space;\quad&space;\text{if&space;inverse&space;by&space;volatility}&space;\\&space;\frac{\sigma_i^2}{\sum_{i=1}^n&space;\sigma_i^2},&&space;\quad&space;\text{if&space;inverse&space;by&space;variance}&space;\\&space;\end{cases}&space;\text{for}&space;i&space;=&space;1,&space;2,\dots,&space;n" title="x_i = \begin{cases} \frac{\sigma_i}{\sum_{i=1}^n \sigma_i},& \quad \text{if inverse by volatility} \\ \frac{\sigma_i^2}{\sum_{i=1}^n \sigma_i^2},& \quad \text{if inverse by variance} \\ \end{cases} \text{for} i = 1, 2,\dots, n" />

This weighting scheme simply assign weight inversely to the risk (either volatility or variance) of each individual on the hope of equalizing the risk contribution from each asset assuming no correlation among assets.

- **equal_weight**

<img src="https://latex.codecogs.com/gif.latex?x_i&space;=&space;\frac{1}{n}&space;\text{for&space;}&space;i&space;=&space;1,&space;2,&space;\dots,&space;n" title="x_i = \frac{1}{n} \text{for } i = 1, 2, \dots, n" />

This is the most naive and straightforward (probably the most effective) weighting scheme by assigning equal weight to all assets in the portfolio.
