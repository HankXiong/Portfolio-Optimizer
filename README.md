# Portfolio-Optimizer

This project provides several standard portfolio optimization methods that are widely studied and used using *cvxpy* or *scipy.minimize*, in addition,  I also add some penalization customization to avoid overly concentraded result.

Test example can be seen in the main.py, the following provides the mathmatical formula for them.

- **target_return_robustMV**
<img src="http://latex.codecogs.com/svg.latex?Min:&space;X^T&space;\Sigma&space;X&space;\\\begin{cases}X^T&space;\cdot&space;\boldsymbol{1}&space;&=&space;1&space;\\X^T&space;\cdot&space;\hat{\mu}&space;-&space;L&space;*&space;||\Theta&space;X||_2&&space;>=&space;target\_return&space;\\X&space;&&space;\geq&space;lower\_bound&space;\\X&space;&&space;\leq&space;upper\_bound&space;\\\end{cases}"/>
