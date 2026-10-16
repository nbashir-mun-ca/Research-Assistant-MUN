# Day 4 analytic derivation

Prepared by Nouman with ChatGPT assistance under the reported authorization to
complete both roles. This is a derivation and verification of the printed model,
not a claim that Parsa conducted a separate calculation. The guide and earlier
Day 4 results were available; a blind comparison before seeing any output is not
claimed. Each rerun saves the theoretical table before calling the solver.

## 1. Reduce eight agents to two functions

There are five h1 agents (paper agents 3–7) and three h2 agents (1, 2, 8).
Each has Pareto weight one. Wealth S is uniform on [0,1], so wealth s equals rank r
in this experiment. These remain separate arrays in the program.

Within a type, utility functions and weights are identical. Strict concavity of
the logarithm means equalizing consumption within that type improves welfare
unless consumption was already equal. Averaging within a type preserves budgets,
nonnegativity, and monotonicity. Thus write b(s) for EACH h1 agent and a(s) for
EACH h2 agent. Their combined consumption must satisfy

$$5b(s)+3a(s)=s.$$

At a fixed wealth state, maximize

$$5\log(1+\sqrt{s}+b)+3\log(1+s^{2/3}+a).$$

## 2. Match the benefit of one extra unit

When both types receive positive consumption, the gain from an additional unit
must be equal; otherwise transferring a little consumption improves welfare:

$$\frac{1}{1+\sqrt{s}+b}=\frac{1}{1+s^{2/3}+a}.$$

Therefore a = b + sqrt(s) - s^(2/3). Substitute this into the budget:

$$b(s)=\max\left\{\frac{s+3(s^{2/3}-\sqrt{s})}{8},0\right\},
\qquad a(s)=\frac{s-5b(s)}{3}.$$

The maximum with zero enforces the consumption domain. The other allocation is
positive for s>0: on the interior it equals
[s+5(sqrt(s)-s^(2/3))]/8, whose terms are nonnegative on [0,1].

## 3. Explain the zero-consumption region

If b=0, the budget gives a=s/3. Keeping b at zero is optimal exactly when h1's
marginal utility is no larger than h2's marginal utility:

$$\frac{1}{1+\sqrt{s}}\leq\frac{1}{1+s^{2/3}+s/3}.$$

This is equivalent to s+3s^(2/3)-3sqrt(s)<=0. For positive s, put t=s^(1/6),
divide by t^3, and solve

$$t^3+3t-3=0,\qquad s_0=t^6\simeq0.298995684337.$$

The cubic is strictly increasing and has exactly one root in (0,1).
Below s0, each h2 agent receives s/3 and each h1 agent receives zero.
Above s0, both types receive consumption. At s=1, all eight receive 1/8.

## 4. Check monotonicity, rather than assuming it

Below s0, b'=0 and a'=1/3. Above s0:

$$b'(s)=\frac{1+2s^{-1/3}-(3/2)s^{-1/2}}{8},$$
$$a'(s)=\frac{1+(5/2)s^{-1/2}-(10/3)s^{-1/3}}{8}.$$

Multiply the numerators by t^3>0. For b', the resulting polynomial is
t^3+2t-1.5. It is increasing, and at t0 its value is 1.5-t0>0.
For a', the polynomial is t^3-(10/3)t+2.5. It is decreasing on [t0,1],
and its value at 1 is 1/6>0. Both derivatives are therefore positive.
The two formulas agree at s0, so there is no downward jump there.

The pointwise optima are feasible for the monotone problem. Consequently they
also solve the full finite-grid problem. Strict concavity gives a unique optimum
on a grid with positive probability weights. This shortcut depends on these
parameters and must be rechecked for other utilities or Pareto weights.

## 5. What the numerical certificate proves

On the fixed grid, concavity and an exact linear oracle imply

$$F(C)\leq F^*\leq F(C)+G(C),\qquad
G(C)=\sum_{j,i}\nabla F(C)_{ji}(V_{ji}-C_{ji}).$$

Here V maximizes the linear objective over all feasible allocations. The analytic
grid welfare must lie inside this interval, up to the recorded rounding tolerance.
This bounds welfare optimization error on the chosen grid. It does not bound the
error from replacing the continuous distribution by 400 midpoint nodes, and it
does not establish the same certificate for nonconcave utilities.

## Sources

Five-day plan, printed p.3, both Day 4 columns. Replication guide, sections 3–7
and milestone 5 (especially equations 12–13). Carlier–Lachapelle (2011), Example 2,
printed p.12, Figure 3 on p.7, algorithm on pp.6–8. The supplied PDFs are the
source edition; their SHA-256 hashes are recorded in source_manifest.json.
