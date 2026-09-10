<div align="center">

# Who polluted the harbor? A live ProbNum investigation

### Tutorial at ProbNum 2026

**Friday, 11 September 2026**

<a href="https://timwei.land"><img src="https://github.com/timweiland.png?size=300" width="120" height="120" alt="Tim Weiland" /></a>

**[Tim Weiland](https://timwei.land)**

Tübingen AI Center, University of Tübingen

[Setup](#setup) · [Notebooks](#notebooks)

</div>

**Abstract**

Together we build a probabilistic PDE solver in Julia and use it to infer the
source of a pollutant leak in a real harbor. The live demo moves from GP
conditioning on linear functional information to sparse field models and
Bayesian source inversion, with short coding exercises along the way.

## Setup

The notebook runs in Pluto with **Julia 1.12**. Install Julia with
[Juliaup](https://julialang.org/install/), then add the required version:

```sh
juliaup add 1.12
```

Clone or download this repository, then run from the repository root:

```sh
julia +1.12
```

Install Pluto in the global environment, then install the tutorial's
pinned dependencies:

```julia
import Pkg
Pkg.add("Pluto")
Pkg.activate(".")
Pkg.instantiate()
exit()
```

The notebook activates the tutorial environment itself; Pluto is launched from
the global environment.

## Run

From the repository root:

```sh
julia +1.12 -t auto
```

In Julia:

```julia
import Pluto
Pluto.run(notebook="live_demo.jl")
```

Pluto opens in your browser. The *Not yet* boxes indicate unfinished exercises.

## Notebooks

| Notebook | Use |
| --- | --- |
| [Participant notebook](live_demo.jl) | Exercises, hints, and collapsible solutions. |
| [Complete solution](live_demo_solution.jl) | The same notebook with the exercises filled in. |

To open the complete solution, use `live_demo_solution.jl` in the launch command.
If you have `make`, `make demo` and `make solution` are equivalent shortcuts.

## For the presenter

Edit `live_demo_solution.jl`; `make participant` regenerates `live_demo.jl`
by replacing `#SOL` answers with `missing`. Plot styling and harbor geometry
live in `assets/`.

Based on the [MLSS 2026 tutorial](https://github.com/probabilistic-numerics/MLSS2026Tutorial),
which builds on the [ICML 2026 tutorial](https://github.com/probabilistic-numerics/ICML2026Tutorial).

[MIT license](LICENSE) · [Citation](CITATION.bib)
