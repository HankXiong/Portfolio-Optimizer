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

if not specified, the following optimizations all have constraints below:

<!-- <img src="http://latex.codecogs.com/svg.latex?\begin{align*}&X^T&space;\boldsymbol{1}&space;=&space;1&space;\\&X&space;\geq&space;lower\_bound&space;\\&X&space;\leq&space;upper\_bound&space;\\\end{align*}&space;" title="http://latex.codecogs.com/svg.latex?\begin{align*}&X^T \boldsymbol{1} = 1 \\&X \geq lower\_bound \\&X \leq upper\_bound \\\end{align*} " /> -->

<img src="https://latex.codecogs.com/gif.latex?\begin{cases}&space;X^T&space;\mathbf{1}&space;=&space;1&space;\\&space;X&space;\leq&space;\text{upper&space;bound}&space;\\&space;X&space;\geq&space;\text{lower&space;bound}&space;\\&space;\end{cases}" title="\begin{cases} X^T \mathbf{1} = 1 \\ X \leq upper\_bound \\ X \geq lower\_bound \\ \end{cases}" />


- **target_return_robustMV**
<img src="https://latex.codecogs.com/gif.latex?\underset{X}{\textbf{Min:&space;}}&space;X^T&space;\Sigma&space;X&space;\\&space;\text{s.t.}&space;\quad&space;X^T&space;\hat{\mu}&space;-&space;\lambda&space;||\Theta^{\frac{1}{2}}X||_2&space;\geq&space;\text{target&space;return}" title="\underset{X}{\textbf{Min: }} X^T \Sigma X \\ \text{s.t.} \quad X^T \hat{\mu} - \lambda ||\Theta^{\frac{1}{2}}X||_2 \geq \text{target return}" />


- **penalized_robustMV**

<img src="http://latex.codecogs.com/png.latex?\dpi{110}&space;\begin{align*}&&space;\underset{X}{\textbf{Min:&space;}}&space;X^T&space;\Sigma&space;X&space;-&space;\lambda&space;\hat{\mu}^T&space;X&space;&plus;&space;L&space;*&space;||\Theta^{0.5}X||_2&space;\\\end{align*}&space;&space;&space;&space;" title="http://latex.codecogs.com/png.latex?\dpi{110} \begin{align*}& \underset{X}{\textbf{Min: }} X^T \Sigma X - \lambda \hat{\mu}^T X + L * ||\Theta^{0.5}X||_2 \\\end{align*} " />

- **risk_parity**
