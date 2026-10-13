"""Separate scalar, 50-digit reference for Days 1–3; no model functions used."""
import csv
import json
from decimal import Decimal as D, getcontext
from pathlib import Path

getcontext().prec = 50
ROOT = Path(__file__).resolve().parent
s = [D(1)/6, D(1)/2, D(5)/6]
offsets = [(1+r.sqrt(), 1+r**(D(2)/3)) for r in s]

def welfare(c):
    return sum(((a+x).ln()+(b+y).ln())/3 for (a,b),(x,y) in zip(offsets,c))

equal = [(r/2,r/2) for r in s]
other = [(r/3,2*r/3) for r in s]
rows = []
for name,c in [('equal',equal),('one_third_two_thirds',other)]:
    for j,((a,b),(x,y)) in enumerate(zip(offsets,c)):
        rows.append(dict(allocation=name,state=j,wealth=str(s[j]),c1=str(x),c2=str(y),
                         h1=str((a+x).ln()),h2=str((b+y).ln()),
                         weighted_total=str(((a+x).ln()+(b+y).ln())/3)))
g = [(1/(3*(a+x)),1/(3*(b+y))) for (a,b),(x,y) in zip(offsets,equal)]
h = [tuple(sum(g[k][i] for k in range(j,3)) for i in range(2)) for j in range(3)]
if not all(b>a for a,b in h): raise AssertionError('First oracle should favor h2 at every increment')
direction = [(-r/2,r/2) for r in s]

def slope(t):
    return sum(dx/(3*(a+x+t*dx))+dy/(3*(b+y+t*dy))
               for (a,b),(x,y),(dx,dy) in zip(offsets,equal,direction))
left,right=D(0),D(1)
for _ in range(160):
    middle=(left+right)/2
    if slope(middle)>0: left=middle
    else:right=middle
step=(left+right)/2
updated=[(x+step*dx,y+step*dy) for (x,y),(dx,dy) in zip(equal,direction)]
# Scalar equal-marginal optimum at each node; check it also satisfies monotonicity.
optimal=[]
for r,(a,b) in zip(s,offsets):
    x=max(D(0),min(r,(r+b-a)/2));optimal.append((x,r-x))
for i in range(2):
    if not all(optimal[j+1][i]>=optimal[j][i] for j in range(2)):
        raise AssertionError('Pointwise optimizer is not monotone')
summary=dict(precision_digits=50,equal_welfare=str(welfare(equal)),
             other_welfare=str(welfare(other)),initial_gradient=[[str(x) for x in row] for row in g],
             initial_tail_scores=[[str(x) for x in row] for row in h],
             first_gap=str(slope(D(0))),first_step=str(step),
             first_updated_allocation=[[str(x) for x in row] for row in updated],
             first_updated_welfare=str(welfare(updated)),
             analytic_allocation=[[str(x) for x in row] for row in optimal],
             analytic_welfare=str(welfare(optimal)))
# Compare with preserved floating-point evidence, not with shared objective code.
evidence=json.loads((ROOT.parent/'day03/evidence/summary.json').read_text())
f=evidence['final']; optimum=welfare(optimal)
if abs(welfare(equal)-D('1.2362178075439176'))>D('1e-15'):raise AssertionError('Day 1 mismatch')
if abs(welfare(other)-D('1.2355540193701327'))>D('1e-15'):raise AssertionError('Alternative allocation mismatch')
if not D(str(f['welfare']))-D('1e-14')<=optimum<=D(str(f['upper_bound']))+D('1e-14'):
    raise AssertionError('Analytic optimum outside numerical certificate')
summary['solver_welfare_error_vs_analytic']=str(abs(D(str(f['welfare']))-optimum))
summary['checks']='PASS: scalar welfare, first oracle, first step, feasible analytic optimum and certificate'
out=ROOT/'evidence';out.mkdir(exist_ok=True)
(out/'theory_reference.json').write_text(json.dumps(summary,indent=2)+'\n')
with (out/'tiny_example_terms.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(summary,indent=2))
