### Filtering Generalized Additive Model

$ \text{Pr}(y_i=1 | y) = \sigma\left( s(y)_i F^T \beta + \alpha \right) $
_where_: <br>
&ensp; $y$ is a 1-dimmensional signal
&ensp; $\sigma$ is the sigmoid function
&ensp; $s$ is the segmentation function, subsequence lengths equal filter lengths
&ensp; $F$ is a matrix of filters, each row is an individual filter
&ensp; $\beta$ is a vector of weight terms, number of terms equal number of filters
&ensp; $\alpha$ is the bias term