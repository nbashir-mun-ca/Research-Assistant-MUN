"""Day 5: stress tests and a reproducible handover for the checked log baseline."""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import Bounds, LinearConstraint, minimize

ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / 'day04'))
from day4_example import (ExperimentConfig, RiskSharingModel, AllocationConstraints,
                         FrankWolfeSolver, Example2Benchmark, NumericalValidator,
                         validation_matrices)

# Raise an error when a required check fails, including in optimized Python mode.
def require(condition, message):
    if not condition:
        raise AssertionError(message)

# Write a diagnostic dictionary without allowing NaN or infinity.
def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

# Save a table of dictionaries with a shared set of column names.
def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

# Evaluate the analytic Example 2 welfare at one wealth value for integration.
def analytic_integrand(s):
    b = max((s + 3*(s**(2/3) - np.sqrt(s)))/8, 0.)
    a = (s - 5*b)/3
    return 5*np.log1p(np.sqrt(s) + b) + 3*np.log1p(s**(2/3) + a)

# Extend a solved allocation to a common grid using the guide's feasible endpoints.
def common_grid_allocation(model, allocation, states):
    endpoint = allocation[-1] + (1-model.wealth[-1])/model.shape[1]
    nodes = np.r_[0., model.wealth, 1.]
    values = np.vstack((np.zeros(model.shape[1]), allocation, endpoint))
    target = (np.arange(states)+.5)/states
    interpolated = np.column_stack([np.interp(target, nodes, values[:, i])
                                   for i in range(model.shape[1])])
    common = RiskSharingModel(target, target, np.full(states, 1/states),
                              model.weights, model.agent_types)
    AllocationConstraints(common).check(interpolated)
    return common, interpolated

# Solve one case and save allocations and the complete iteration history.
def solve_case(name, model, config, exact, out, continuous=None, common_states=20000, reference_kind='analytic'):
    constraints = AllocationConstraints(model, config.feasibility_tolerance)
    constraints.check(exact)
    start = time.perf_counter()
    result = FrankWolfeSolver(model, constraints, config).solve()
    seconds = time.perf_counter()-start
    f = result.final
    error = model.objective(exact)-f['welfare']
    require(-1e-12 <= error <= f['gap']+1e-12, name+': benchmark outside certificate')
    require(all(np.diff([row['welfare'] for row in result.history]) >= -1e-12),
            name+': welfare decreased')
    folder = out/name
    folder.mkdir(parents=True, exist_ok=True)
    write_csv(folder/'iterations.csv', result.history)
    header = ['wealth','rank','probability'] + [f'numerical_{i+1}' for i in range(model.shape[1])]
    header += [f'analytic_{i+1}' for i in range(model.shape[1])]
    np.savetxt(folder/'allocations.csv', np.column_stack((model.wealth,model.ranks,
               model.probabilities,result.allocation,exact)), delimiter=',',header=','.join(header),comments='')
    write_json(folder/'config.json', dict(solver=asdict(config),wealth=model.wealth.tolist(),
               ranks=model.ranks.tolist(),probabilities=model.probabilities.tolist(),
               weights=model.weights.tolist(),agent_types=list(model.agent_types)))
    row = dict(case=name,states=model.shape[0],gap_tolerance=config.gap_tolerance,
               converged=result.converged,updates=f['iteration'],welfare=f['welfare'],gap=f['gap'],
               upper_bound=f['upper_bound'],reference_grid_welfare=model.objective(exact),reference_kind=reference_kind,
               welfare_difference_to_reference=error,maximum_allocation_error=float(np.max(abs(result.allocation-exact))),
               allocation_matches=bool(np.max(abs(result.allocation-exact)) <= config.allocation_tolerance),
               allocation_tolerance=config.allocation_tolerance,
               maximum_budget_residual=max(r['budget_residual'] for r in result.history),
               minimum_increment_over_run=min(r['minimum_increment'] for r in result.history),
               seconds=seconds)
    if continuous is not None:
        common, interpolated = common_grid_allocation(model,result.allocation,common_states)
        reference = Example2Benchmark(common).allocation(common.wealth)
        common_value = common.objective(interpolated)
        row.update(analytic_grid_minus_continuous=model.objective(exact)-continuous,
                   common_grid_welfare=common_value,
                   common_grid_analytic_welfare=common.objective(reference),
                   common_grid_welfare_error=common.objective(reference)-common_value,
                   common_grid_minus_continuous=common_value-continuous)
    write_json(folder/'summary.json',row)
    print(f"{name}: converged={result.converged}, gap={f['gap']:.3g}, "
          f"allocation error={row['maximum_allocation_error']:.3g}",flush=True)
    return row,result

# Derive the two-agent Example 5 pointwise optimum for the given Pareto weight.
def example5_benchmark(model, weight):
    s = model.wealth
    first = np.clip(weight*(1+s**(2/3)+s)-(1-weight)*(1+np.sqrt(s)),0,s)
    return np.column_stack((first,s-first))

# Check changed ranks, probabilities and weights with direct-consumption SLSQP.
def alternative_case(config, out, mixed=False):
    model = RiskSharingModel([.04,.15,.35,.65,1.1],[.07,.23,.44,.68,.93],
                            [.05,.1,.2,.25,.4],[.95,1.05,1.] if mixed else [.7,1.4,.9],('h1','h2','h1'))
    constraints = AllocationConstraints(model)
    budget, monotone = validation_matrices(model)
    reference = minimize(lambda x: -model.objective(x.reshape(model.shape)),
        model.equal_sharing().ravel(),jac=lambda x: -model.gradient(x.reshape(model.shape)).ravel(),
        bounds=Bounds(0,np.inf),constraints=[LinearConstraint(budget,model.wealth,model.wealth),
        LinearConstraint(monotone,0,np.inf)],method='SLSQP',options={'ftol':1e-13,'maxiter':1000})
    require(reference.success,reference.message)
    direct = reference.x.reshape(model.shape)
    constraints.check(direct)
    name='alternative_mixed_preferences' if mixed else 'alternative_rank_weight_probability'
    row,_ = solve_case(name,model,config,direct,out,reference_kind='SLSQP numerical reference')
    if not mixed: require(row['converged'],'Alternative case did not converge')
    return row

# Verify permutation invariance without assuming the two agent groups are contiguous.
def permutation_case(config, out):
    config = replace(config,states=40)
    model = RiskSharingModel.from_config(config)
    constraints = AllocationConstraints(model)
    original = FrankWolfeSolver(model,constraints,config).solve()
    order = np.array([7,2,0,5,1,6,3,4])
    permuted = replace(model,weights=model.weights[order],
                       agent_types=tuple(model.agent_types[i] for i in order))
    swapped = FrankWolfeSolver(permuted,AllocationConstraints(permuted),config).solve()
    error = float(np.max(abs(original.allocation-swapped.allocation[:,np.argsort(order)])))
    require(original.converged and swapped.converged and error < 1e-11,'Agent permutation failed')
    return {'maximum_allocation_difference':error,'passed':True}

# Plot grid and tolerance studies with solver and quadrature effects kept separate.
def plot_evidence(grid, tolerance, out):
    fig, axes = plt.subplots(1,2,figsize=(11,4.2))
    for profile in ('original','tight'):
        rows = [r for r in grid if r['profile']==profile]
        if profile=='original':
            axes[0].loglog([r['states'] for r in rows],
                          [r['analytic_grid_minus_continuous'] for r in rows],'o-',label='Analytic quadrature error')
        axes[0].loglog([r['states'] for r in rows],
                      [max(r['welfare_difference_to_reference'],1e-16) for r in rows],'s--',label=profile+': solver')
    axes[0].set(xlabel='Number of midpoint states',ylabel='Welfare error',title='Different sources of error')
    axes[0].legend(fontsize=8)
    axes[1].loglog([r['gap_tolerance'] for r in tolerance],
                  [r['maximum_allocation_error'] for r in tolerance],'o-')
    axes[1].axhline(.0025,color='gray',ls='--',label='Day 4 allocation tolerance')
    axes[1].set(xlabel='Requested welfare gap tolerance',ylabel='Maximum allocation error',
                title='400 states: effect of tighter stopping')
    axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'stress_tests.png',dpi=180);plt.close(fig)

# Run the full Day 5 audit, preserving failures rather than changing tolerances afterward.
def run_audit(output):
    out = Path(output)
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'run_status.json',{'status':'RUNNING'})
    base = ExperimentConfig()
    check = NumericalValidator(base).run()
    write_json(out/'day4_preflight_checks.json',check)
    benchmark = Example2Benchmark(RiskSharingModel.from_config(base))
    continuous,quad_error = quad(analytic_integrand,0,1,points=[benchmark.activation],
                                 epsabs=1e-11,epsrel=1e-11,limit=200)
    grid, tolerance, weights = [],[],[]
    for profile,tol in [('original',5e-6),('tight',1e-7)]:
        for states in (100,200,400,800,1600):
            config = replace(base,states=states,gap_tolerance=tol)
            model = RiskSharingModel.from_config(config)
            exact = Example2Benchmark(model).allocation(model.wealth)
            row,_ = solve_case(f'grid_{profile}_{states}',model,config,exact,out,continuous)
            row['profile']=profile;grid.append(row)
            require(row['converged'],row['case']+': iteration limit reached')
            if profile=='tight': require(row['allocation_matches'],row['case']+': allocation mismatch')
    for tol in (5e-6,1e-6,1e-7):
        if tol in (5e-6,1e-7):
            tolerance.append(dict(next(r for r in grid if r['states']==400 and r['gap_tolerance']==tol)))
        else:
            config=replace(base,gap_tolerance=tol);model=RiskSharingModel.from_config(config)
            row,_=solve_case('tolerance_1e-6',model,config,Example2Benchmark(model).allocation(model.wealth),out,continuous)
            row['profile']='intermediate';tolerance.append(row)
    # Deliberately stop early to verify that an iteration cap is not labeled convergence.
    cap_config=replace(base,max_updates=0)
    cap_model=RiskSharingModel.from_config(cap_config)
    cap,_=solve_case('iteration_cap_probe',cap_model,cap_config,
                    Example2Benchmark(cap_model).allocation(cap_model.wealth),out)
    require(not cap['converged'],'Iteration-cap probe incorrectly reported convergence')
    tight = replace(base,gap_tolerance=1e-7)
    alternative=alternative_case(tight,out)
    permutation=permutation_case(tight,out)
    mixed=alternative_case(replace(tight,max_updates=2000),out,mixed=True)
    # Core tight-grid checks have passed before the optional Example 5 exercise.
    utilities=[]
    for weight in (.45,.48,.50,.55):
        config=replace(base,agent_types=('h1','h2'),pareto_weights=(weight,1-weight),
                       allocation_tolerance=.006)
        model=RiskSharingModel.from_config(config)
        exact=example5_benchmark(model,weight)
        row,result=solve_case(f'example5_{weight:.2f}',model,config,exact,out)
        require(row['converged'] and row['allocation_matches'],'Example 5 acceptance failed')
        row['lambda']=weight;weights.append(row)
        # Use the analytic allocations to isolate comparative statics from solver noise.
        utilities.append(float(np.sum(model.probabilities*np.log(model.offsets[:,1]+exact[:,1]))))
    require(np.all(np.diff(utilities)<0),'Agent 2 utility should fall as agent 1 weight rises')
    dense=(np.arange(20000)+.5)/20000
    analytic_shares=[example5_benchmark(RiskSharingModel(dense,dense,np.full(20000,1/20000),
                    [w,1-w],('h1','h2')),w)[:,0] for w in (.45,.48,.50,.55)]
    require(np.all(np.diff(analytic_shares,axis=0)>=-1e-12),'Weight comparative statics failed')
    write_csv(out/'grid_study.csv',grid)
    write_csv(out/'tolerance_study.csv',tolerance)
    write_csv(out/'example5_weight_study.csv',weights)
    plot_evidence(grid,tolerance,out)
    summary={'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),
       'scope':'Concave logarithmic baseline; finite-grid certificates with floating point arithmetic',
       'continuous_welfare_reference':continuous,'quad_estimated_absolute_error':quad_error,
       'common_grid_states':20000,'original_profile_allocation_failures':
       [r['case'] for r in grid if r['profile']=='original' and not r['allocation_matches']],
       'tight_profile_all_passed':all(r['converged'] and r['allocation_matches'] for r in grid if r['profile']=='tight'),
       'iteration_cap_probe':cap,'alternative_case':alternative,'mixed_alternative_case':mixed,'permutation_check':permutation,
       'example5_agent2_analytic_utilities':utilities,'example5_cases_passed':len(weights),
       'common_grid_analytic_quadrature_bias':grid[0]['common_grid_analytic_welfare']-continuous,
       'continuous_error_certificate':False,'personal_review_by_parsa':'not claimed'}
    write_json(out/'summary.json',summary)
    write_json(out/'environment.json',{'python':sys.version,'platform':platform.platform(),
               'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__})
    files=[ROOT/'day04/day4_example.py',ROOT/'day05/day5_audit.py']
    write_json(out/'source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in files if p.exists()})
    write_json(out/'run_status.json',{'status':'PASS','note':'Original-profile allocation misses remain reported in summary.json'})
    print('Day 5 audit: PASS. See summary.json for the original-profile limitation.',flush=True)
    return summary

# Read the output-folder option and report failed audits with a nonzero exit code.
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default=str(ROOT/'day05/results'))
    args=parser.parse_args()
    try:
        run_audit(args.out)
    except Exception as error:
        Path(args.out).mkdir(parents=True,exist_ok=True)
        write_json(Path(args.out)/'run_status.json',{'status':'FAIL','error':str(error)})
        raise

if __name__=='__main__':
    main()
