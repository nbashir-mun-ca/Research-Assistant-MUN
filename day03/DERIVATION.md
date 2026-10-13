# Day 3: step, certificate, and one worked iteration

Prepared by Nouman with ChatGPT assistance under reported Parsa authorization.

## Why every step stays feasible

Let K be the finite-grid feasible set. Budgets are linear equalities;
nonnegativity and monotonicity are linear inequalities. Thus K is convex.
It is nonempty (equal sharing), closed and bounded (0 <= C_ji <= s_j), hence
compact. The continuous objective attains a maximum on K.

At current C, compute g=grad F(C). The Day 2 oracle chooses

$$V\in\arg\max_{Y\in K}\langle g,Y\rangle,\quad D=V-C.$$

Choose t in [0,1] to maximize F(C+tD). Because
C+tD=(1-t)C+tV, every candidate on this segment belongs to K.
The zero step is available, so exact line search cannot lower welfare.

For our logs, with a_ji=1+r_j^(alpha_i),

$$q(t)=\frac{dF(C+tD)}{dt}
=\sum_{j,i}\frac{p_j\lambda_iD_{ji}}{a_{ji}+C_{ji}+tD_{ji}},$$

$$q'(t)=-\sum_{j,i}\frac{p_j\lambda_iD_{ji}^2}
{(a_{ji}+C_{ji}+tD_{ji})^2}\leq0.$$

Thus q(0)<=0 permits t=0; q(1)>=0 permits t=1. Otherwise bisection locates
the interior zero. The code uses 42 halvings: the bracket width is 2^-42 and
its midpoint step error is at most 2^-43 in exact arithmetic. Floating-point
slope evaluation also contributes error; this bracket width is not a welfare certificate.

## Proof of the global finite-grid welfare bound

Let C* maximize F on K. Concavity gives

$$F(C^*)\leq F(C)+\langle g,C^*-C\rangle.$$

Since C* is feasible and V maximizes the linear objective,

$$\langle g,C^*-C\rangle\leq\langle g,V-C\rangle=:G(C).$$

Since C is itself feasible for the oracle, G(C)>=0. Combining the inequalities:

$$0\leq F(C^*)-F(C)\leq G(C),\qquad
F(C)\leq F(C^*)\leq F(C)+G(C).$$

This proof needs a feasible current allocation, a concave differentiable
objective, and an exact linear oracle. The all-agent increment oracle supplies
that exact linear maximizer algebraically. HiGHS is an additional numerical
check. Floating-point feasibility and arithmetic use declared tolerances, so
reported bounds are numerical certificates rather than interval-arithmetic proofs.
If an approximate oracle had known linear suboptimality delta, the safe bound
would use G+delta. Merely applying an approximate tie rule would need such accounting.

## One iteration from explicit scalar formulas

For equal sharing on ranks (1/6,1/2,5/6), weights one and probabilities 1/3:

| State | g for h1 | g for h2 | Tail H for h1 | Tail H for h2 |
|---|---:|---:|---:|---:|
| 1/6 | 0.223476428 | 0.240467837 | 0.536885783 | 0.562564594 |
| 1/2 | 0.170319441 | 0.177308688 | 0.313409355 | 0.322096757 |
| 5/6 | 0.143089914 | 0.144788069 | 0.143089914 | 0.144788069 |

Every increment favors h2, so V_j=(0,s_j) and D_j=(-s_j/2,s_j/2).
Hence q(0)=G(C)=0.003870827167386246. Solving the explicit q(t)=0 equation
by separate 50-digit scalar arithmetic gives t=0.110112563741513479.
The notebook's 42-halving value is 0.11011256374160894, consistent with its
step bracket and floating-point rounding.

| Wealth | New h1 consumption | New h2 consumption |
|---|---:|---:|
| 1/6 | 0.074157286355 | 0.092509380312 |
| 1/2 | 0.222471859065 | 0.277528140935 |
| 5/6 | 0.370786431774 | 0.462546901559 |

Welfare rises from 1.236217807543917768 to 1.236430915639813286.
The oracle provides local guidance; giving all wealth to h2 is not the nonlinear optimum.

## A separate analytic check for this small grid

At a fixed node s=r, let A=1+sqrt(r), B=1+r^(2/3), x=C1, C2=s-x.
Maximize log(A+x)+log(B+s-x) on 0<=x<=s. Equal marginal utilities give
A+x=B+s-x, so

$$x^*(s)=\min\{s,\max\{0,(s+B-A)/2\}\},\quad C_2^*(s)=s-x^*(s).$$

For the three chosen nodes, both agents are interior and the resulting columns
are increasing. Therefore these pointwise maxima also satisfy the coupled
monotonicity constraints and maximize the whole finite-grid objective.
This argument applies here because monotonicity was checked; pointwise
optimization need not be feasible on an arbitrary model or grid.

| Wealth | Analytic h1 | Analytic h2 |
|---|---:|---:|
| 1/6 | 0.030635904171 | 0.136030762496 |
| 1/2 | 0.211426871880 | 0.288573128120 |
| 5/6 | 0.403005605905 | 0.430327727428 |

The analytic welfare is 1.23681157934741574094. The saved solver welfare differs
by about 4.4e-16 and lies within its reported [F,F+G] interval, allowing rounding.
This scalar reference uses neither model.objective nor model.gradient.
It is a separate verification route prepared in the same assisted workflow,
not an independently conducted review by another researcher.

## What stopping proves and what it does not

The solver stops after 291 updates with G=9.597698538699184e-10, below 1e-9.
Its 292 log rows include the initial allocation. Iteration-cap and stall outcomes
are not convergence. The certificate measures fixed-grid welfare error, not
individual-share error, grid/quadrature error, or continuous-model accuracy.
The global argument cannot be carried over automatically to nonconcave utilities.

Sources: five-day plan, Day 3, p. 3; replication guide, sections 5.4–5.5,
pp. 6–7. Raw scalar values and row terms are in the completion audit's evidence.
