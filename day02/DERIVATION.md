# Day 2 mathematical contract

Provisional notes for Parsa's independent review. Read the shorter explanation
in [README.md](README.md) first.

## Objective and gradient

State index `j` increases with wealth; `i` indexes agents. Consumption has shape
`(m,n)`. Wealth, rank, and probability vectors have shape `(m,)`; Pareto weights
have shape `(n,)`.

$$J_m(c)=\sum_{j=0}^{m-1}\sum_{i=1}^{n}p_j\lambda_iL_i(r_j,c_{ji}).$$

The rank and probabilities stay fixed when differentiating consumption:

$$g_{ji}=\partial_{c_{ji}}J_m(c)=p_j\lambda_i\partial_cL_i(r_j,c_{ji}).$$

For `h1`, divide `p_j lambda_i` by `1 + sqrt(r_j) + c_ji`. For `h2`, divide by
`1 + r_j^(2/3) + c_ji`. Each second consumption derivative is negative, so the
finite-grid objective is concave. Probability and Pareto factors are both required.

The replication's uniform midpoint convention is `r_j=(j+1/2)/m`, `s_j=r_j`,
`p_j=1/m`. It approximates a continuous quantile integral. General rank/wealth
diagnostic cases do not assume `r=s`.

## Increments and tail scores

Anchor `s_-1=0` and `c_-1,i=0`. Define

$$\Delta s_j=s_j-s_{j-1},\qquad d_{ji}=c_{ji}-c_{j-1,i}.$$

The allocation constraints are equivalent to `d_ji >= 0` and
`sum_i d_ji = Delta s_j`. Recover consumption by `c_ki=sum_{j=0}^k d_ji`.
An increment at state `j` affects every state `k >= j`, so

$$H_{ji}=\partial_{d_{ji}}J_m=\sum_{k=j}^{m-1}g_{ki}.$$

Equivalently, rearranging the linearized objective gives

$$\sum_{k,i}g_{ki}y_{ki}=\sum_{j,i}\left(\sum_{k=j}^{m-1}g_{ki}\right)d^y_{ji}.$$

Thus `g` differentiates direct consumption and `H` scores increments.

## Why the oracle is optimal for the linear problem

Each increment row independently maximizes `sum_i H_ji d_ji` over nonnegative
increments summing to `Delta s_j`. Its value is bounded by
`max_i H_ji * Delta s_j`. Assigning the increment to a maximum-score agent
attains that bound. Equal division among exact maximum-score ties also attains
it and preserves identical-agent symmetry.

All agents remain explicit. Even when every score is negative, budget equality
still requires assigning the full increment. The guide notes a residual-agent
transcription concern with the source paper's reduced formulation; this direct
all-agent construction retains every feasible option and is independently checked.

For the hand fixture, use wealth `(1/6,1/2,5/6)` and invented coefficient rows
`(4,3),(1,2),(2,1)`. Tail scores are `(7,6),(3,3),(2,1)`, yielding increment rows
`(1/6,0),(1/6,1/6),(1/3,0)` and consumption rows
`(1/6,0),(1/3,1/6),(2/3,1/6)`. This is a linear-algebra illustration rather than
the actual log-utility gradient at a specified allocation.

## Independent checks and their limits

Compare `sum_ji g_ji v_ji` with a centered difference of `J_m(c+h*v)` and
`J_m(c-h*v)` when both allocations are feasible. Use a forward feasible
direction at boundaries. The predeclared steps and tolerances are recorded.

The reference LP minimizes `-sum_ji g_ji y_ji` over direct consumption variables,
subject to `y_ji>=0`, row sums equal to wealth, and
`y_ji-y_(j+1),i <= 0`. It does not reuse increments or tail scores.
Compare optimal values; ties can produce different allocation arrays.

These checks verify the finite calculations within numerical tolerances. They
provide no nonlinear optimization result or continuous-distribution error bound.
Parsa's independent derivation and joint review are still outstanding.

Sources: five-day plan p. 2; replication guide pp. 4-6, equations (6)-(9);
Carlier-Lachapelle (2011), adjoint equation (3.5) p. 6 and Example 2 p. 12.
