# Day 1 model card

Prepared by Nouman with ChatGPT assistance. Completion of the delegated theory
work follows Nouman's report of Parsa's authorization on 13 October 2026.
This document is not attributed to Parsa as an independent calculation.

## Economic problem

We divide nonnegative aggregate wealth S among agents. Consumption is wealth
received, not an insurance loss. More consumption raises utility. Choose
nonnegative, nondecreasing functions c_i(s) with sum_i c_i(s)=s at every state.
Increasing total wealth must not reduce any agent's share.

For s<t, each difference c_i(t)-c_i(s) is nonnegative and their sum is t-s.
Therefore every difference is at most t-s: each allocation is 1-Lipschitz in
wealth. This follows from monotonicity and the budget; it is not an extra rule.

## Preferences and probability convention

The continuous rank-linear objective is

$$F(c)=\sum_i\lambda_i\int_0^1 L_i(r,c_i(Q_S(r)))\,dr.$$

Rank r is a cumulative probability position. It is distinct from wealth s and is
not an additional randomization device. Monotone allocation functions allow the
consumption quantile to be written c_i(Q_S(r)), up to irrelevant quantile conventions.

For the guide's uniform aggregate S on [0,1], Q_S(r)=r. Define

$$r_j=(j+1/2)/m,\quad s_j=r_j,\quad p_j=1/m.$$

These weights approximate the continuous integral by midpoint quadrature. We
optimize that finite sum; a three-node midpoint sum is not the exact utility
integral of a separate three-atom aggregate model with varying rank utility.
For another continuous aggregate, use s_j=Q_S(r_j), keeping ranks separate.

## Finite-grid specification

| Quantity | Contract |
|---|---|
| Consumption C | shape (m,n), finite, nonnegative |
| Wealth, ranks, probabilities | shape (m,), wealth ordered, ranks strictly ordered in (0,1) |
| Probability weights | nonnegative, sum to one |
| Pareto weights | shape (n,), positive |
| Statewise budget | sum_i C_ji=s_j |
| Monotonicity | C_(j+1),i >= C_ji |
| Objective | maximize sum_j,i p_j lambda_i L_i(r_j,C_ji) |

The log utilities use natural logarithms:

$$h_1(r,c)=\log(1+\sqrt r+c),\quad h_2(r,c)=\log(1+r^{2/3}+c).$$

They are increasing and strictly concave in consumption; their consumption
marginal decreases with rank for r>0. Pareto weights count agents' utilities;
probability weights count wealth states. Equal weights do not require equal shares.

The full Example 2 target has eight agents: h1 for agents 3–7, h2 for agents
1,2,8, all weights one. Days 1–3 use a smaller illustration with one h1 agent
then one h2 agent, m=3, weights (1,1). They do not replicate full Example 2.

## Tiny example and acceptance

| Wealth and rank | Probability | Equal sharing | Other sharing |
|---|---|---|---|
| 1/6 | 1/3 | (1/12,1/12) | (1/18,1/9) |
| 1/2 | 1/3 | (1/4,1/4) | (1/6,1/3) |
| 5/6 | 1/3 | (5/12,5/12) | (5/18,5/9) |

Both allocations are feasible. Check shapes, normalized probabilities, budget,
nonnegativity and monotonicity at absolute tolerance 1e-12 for Day 1. The general
Day 2/3 routines explicitly declare their own tolerances. Numerical tolerances do
not certify continuous-distribution accuracy.

See [the explicit calculations](WORKED_EXAMPLE.md) and the
[completion audit](../audit_days01_03/README.md).

## Sources and limitations

Five-day plan, pp. 1–2; replication guide, sections 3–4, pp. 4–5;
Carlier–Lachapelle, equation (2.5) and Example 2. The companion supplied code
package is unavailable, so its baseline was not run. This independent
implementation records that limitation rather than claiming source-code replication.
