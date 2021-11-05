### Filtering Generalized Additive Model

`$ \text{Pr}(y_i=1 | y) = \sigma\left( s_m(y)_i F^T \beta + \alpha \right) $`  
_where_: <br>
&ensp; $y$ is a 1-dimmensional signal  
&ensp; $\sigma$ is the sigmoid function  
&ensp; $s_m$ is the segmentation function, subsequence lengths ($m$) equal filter lengths  
&ensp; $F$ is a matrix of filters, each row is an individual filter  
&ensp; $\beta$ is a vector of weight terms, number of terms equal number of filters $m$   
&ensp; $\alpha$ is the bias term  

$ s_3(y) = \begin{bmatrix} y_1 & y_2 & y_3 \\ y_2 & y_3 & y_4 \\ y_3 & y_4 & y_5 \\ ... & ... & ... \\ y_{n-2} & y_{n-1} & y_n \end{bmatrix} $ $ F = \begin{bmatrix} 1 & 0 & -1 \\ 1 & -1 & 1 \end{bmatrix} $  

Filter 1: $ (1, 0, -1) $  
Filter 2: $ (1, -1, 1) $  

