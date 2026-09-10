### A Pluto.jl notebook ###
# v1.0.1

#> [frontmatter]
#> title = "Who polluted the harbor? A live ProbNum investigation"
#> description = "ProbNum 2026 · 11 September 2026 — live tutorial with Tim Weiland"

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ a0000000-0000-4000-8000-000000000000
import Pkg

# ╔═╡ a0000001-0001-4000-8000-000000000001
# ╠═╡ show_logs = false
begin
	Pkg.activate(@__DIR__)

	using LinearAlgebra, SparseArrays
	using Random
	using Statistics
	using CairoMakie
	using PlutoUI
	using PlutoUI: Slider
	using JLD2

	using FunctionalGPs
	using FunctionalGPs.Notation
	using AbstractGPs
	using KernelFunctions: ⊗
	using Distributions
	using Distributions: MvNormal, Normal, Uniform, truncated
	import Distributions: mean as dist_mean, var as dist_var, cov as dist_cov, quantile, pdf
	using GaussianMarkovRandomFields
	using Latte
	using DynamicPPL

	# Shared colors and plotting helpers for the live demo.
	include(joinpath(@__DIR__, "assets", "plotting.jl"))
	import .HarborPlotting: COLOR_PROBLEM, COLOR_LATENT, COLOR_QoI,
		COLOR_INFORMATION_OPERATOR, plot_gp!, PNMETHODS_PROBLEM_LINES_KWARGS,
		TUE_AI_COLORS, TU_COLORS

	# Tübingen-AI theme (palette + dark spines), but with axes kept visible —
	# this is the interactive demo, not a slide figure — and a heavier linewidth
	# for on-screen readability.
	set_theme!(merge(
		Theme(Lines = (linewidth = 2,), Series = (linewidth = 2,),
			VLines = (linewidth = 2,), HLines = (linewidth = 2,)),
		HarborPlotting.THEME,
	))
end

# ╔═╡ 5d3b7d8a-b2d4-48c2-a681-a5aab956ad64
HTML("""
<h1>Follow along</h1>
<div style="display:flex; flex-wrap:wrap; align-items:center; justify-content:center;
            gap:2rem; margin:2rem 0;">
  <a href="https://github.com/timweiland/probnum26-tutorial"
     aria-label="Open the tutorial repository" style="display:block; width:300px; flex-shrink:0;">
    $(read(joinpath(@__DIR__, "assets", "qr", "tutorial.svg"), String))
  </a>
  <div style="flex:1; min-width:280px; max-width:460px;">
    <p style="font-size:1.35rem;">Please run the setup in the README to follow along! :)</p>
    <a href="https://github.com/timweiland/probnum26-tutorial"
       style="font-size:1.1rem; overflow-wrap:anywhere;">github.com/timweiland/<wbr>probnum26-tutorial</a>
  </div>
</div>
""")

# ╔═╡ b0000001-0001-4000-8000-000000000001
md"""
# Who polluted the harbor? A live ProbNum investigation

**Tim Weiland · ProbNum 2026 · 11 September 2026**

"""

# ╔═╡ 82e2cf25-aa24-4a20-bd4e-3787288f0ae5
HTML(read(joinpath(@__DIR__, "assets", "tuai_style.html"), String))

# ╔═╡ 8cd893ac-a409-4bc7-8df1-69014ad9992d
  function pkgcard(name, slug; caption = "↗ scan for the repo", w = 180, v_offset = 0)
      svg = read(joinpath(@__DIR__, "assets", "qr", slug * ".svg"), String)
      HTML("""
      <div style="position:absolute; right:-8px; width:0; transform:translateY($(v_offset)px);">
        <div style="width:$(w)px; font-family:'Plus Jakarta Sans',sans-serif; text-align:center;
                    background:white; border:1px solid var(--tuai-lightblue,#85cbd2);
                    border-radius:8px; padding:0.6em;">
          <div style="font-weight:700;color:var(--tuai-darkblue,#1a3a5b);margin-bottom:0.4em;">$name</div>
          $svg
          <div style="font-family:'Roboto Mono',monospace;font-size:0.6em;
                      color:var(--tuai-accent,#ea4b2e);margin-top:0.3em;">$caption</div>
        </div>
      </div>""")
  end;

# ╔═╡ ee000001-0001-4000-8000-000000000001
begin
	# ── Exercise helpers (feedback boxes) ──
	_adm(kind, title, body) = Markdown.MD(Markdown.Admonition(kind, title, [body]))
	correct(body = md"Nice — that's right.") = _adm("correct", "✓ Correct", body)
	keep_working(body = md"Not quite yet — have another look.") = _adm("warning", "Keep working", body)
	exercise(body) = _adm("info", "✏️ Exercise", body)
	hint(body) = _adm("hint", "Hint", body)
	optional(body) = _adm("note", "Optional", body)
	still_missing(what = "the exercise above") =
		_adm("warning", "Not yet", md"Complete $(what) first — replace `missing` with your code, and the cells below will come alive.")
	skipped(what = "this") =
		_adm("info", "Skipped", md"Tick **Run the spacetime inference** above to compute $(what) — it takes a few minutes, so it is off by default.")
	nothing
end

# ╔═╡ a0000002-0002-4000-8000-000000000002
begin
	freq_slider_1 = @bind freq Slider(1:5, default=3, show_value=true)
	freq_slider_2 = @bind freq Slider(1:5, default=3, show_value=true)
	colloc_slider_1 = @bind n_colloc Slider(1:10, default=4, show_value=true)
	colloc_slider_2 = @bind n_colloc Slider(1:10, default=4, show_value=true)
	# advection–diffusion coefficients for the 1D demo (ℒ = v·∂ₓ − κ·∂ₓ²)
	v_slider_1 = @bind v_1d Slider(0.0:0.5:3.0, default=1.0, show_value=true)
	v_slider_2 = @bind v_1d Slider(0.0:0.5:3.0, default=1.0, show_value=true)
	κ_slider_1 = @bind κ_1d Slider(0.05:0.05:0.5, default=0.2, show_value=true)
	κ_slider_2 = @bind κ_1d Slider(0.05:0.05:0.5, default=0.2, show_value=true)
	nothing
end

# ╔═╡ ee000002-0001-4000-8000-000000000002
md"""
## How this notebook works

- Cells marked **✏️ Exercise** contain a `missing` that you replace with code. A check box right below tells you whether it looks right.
- Everything downstream of an unsolved exercise shows a *"Not yet"* box instead of a plot — that is expected, not broken.
- Stuck? Every exercise has a collapsible **Hint** and, after the check, a collapsible **Solution** — but give it a real try first.
- Never written Julia? Open the cheat sheet below. If you can read Python or MATLAB, you can read this.
"""

# ╔═╡ ee000003-0001-4000-8000-000000000003
details("Julia cheat sheet (30 seconds)", md"""
| You want | Julia |
|---|---|
| a vector | `[1.0, 2.0, 3.0]` |
| a range | `0:0.1:1` or `range(0, 1, length = 10)` |
| a matrix by formula | `[f(x, x′) for x in xs, x′ in ys]` |
| apply elementwise | `sin.(xs)`, `xs .+ 1` (the dot broadcasts) |
| glue matrices | `[A B; C D]`, `hcat(A, B)`, `vcat(A, B)` |
| a one-line function | `k(x, x′) = exp(-(x - x′)^2)` |
| compose operators | `δ(X) ∘ ∂(2)` reads as "evaluate ∂²u at the points X" |
| unicode | type `\\ell` then Tab for `ℓ`, `\\delta` Tab for `δ`, `x\\prime` Tab for `x′` |

Cells re-run automatically whenever something they depend on changes.
""")

# ╔═╡ b0000004-0004-4000-8000-000000000004
md"""
## Where is the leak?
"""

# ╔═╡ 05f64818-0e35-4342-be38-d32b4242e3bf
md"""
- Oil is leaking continuously somewhere in the bay.
- Six sensors measure the steady-state concentration around the harbor.
- Currents drag the plume, diffusion spreads it, microbial degradation breaks it down.
- **Goal: infer the source location and strength from the sparse sensor readings.**

The full model is the spatiotemporal advection-diffusion-reaction PDE

```math
\frac{\partial u}{\partial t} \;+\; \mathbf{v}(\mathbf{x}) \cdot \nabla u
\;=\; \kappa\,\Delta u \;-\; \lambda\,u \;+\; f(\mathbf{x};\,\boldsymbol{\theta}_{\text{src}})
```

with Dirichlet boundary conditions ``u = 0`` along the coast and at the
open-sea boundary.
"""

# ╔═╡ c0aa0001-0000-4000-8000-000000000001
# Steady current through the bay (model km-frame): a gentle drift toward the
# open sea (south, slight east). Drives the advection term v·∇u and is drawn below.
v_current = (0.1, -0.3)

# ╔═╡ c0000003-0003-4000-8000-000000000003
md"""
## Step 1: A Gaussian process prior over ``u``

σ² $(@bind σ²_k Slider(0.5:0.5:5.0, default=2.0, show_value=true))
ℓ $(@bind ℓ_k Slider(0.1:0.05:1.0, default=0.3, show_value=true))
"""

# ╔═╡ c0000004-0004-4000-8000-000000000004
k_se(x, x′) = σ²_k * exp(-(x - x′)^2 / (2 * ℓ_k^2))

# ╔═╡ 848f8b75-14e5-4dcc-980b-f8c09b2321a2
md"""
## Gaussian Process Conditioning
"""

# ╔═╡ 6a52e349-871b-41af-8e59-58246383a7b7
md"""
### Conditioning on linear functional information

Let ``u \sim \mathcal{GP}(0,k)`` and observe ``y_i = \ell_i[u] + \varepsilon_i``,
with noise ``\boldsymbol\varepsilon \sim \mathcal{N}(0,R)`` independent of ``u``.
Each ``\ell_i`` is a linear functional: for example, a point value ``u(x_i)``,
a derivative ``u'(x_i)``, or a PDE evaluation ``(\mathcal{L}u)(x_i)``.
Assume the kernel is sufficiently regular for these functionals.

Applying the functionals to the **two arguments of the kernel** gives

```math
(K_{\mathrm{obs}})_{ij} = \ell_i^{x}\ell_j^{x'}k(x,x') + R_{ij},
\qquad
(K_{\mathrm{cross}})_{ai} = \ell_i^{x'}k(x_a^\star,x').
```

With ``(K_{\mathrm{prior}})_{ab}=k(x_a^\star,x_b^\star)``, the joint Gaussian is

```math
\begin{pmatrix}u(X_\star)\\ \mathbf y\end{pmatrix}
\sim \mathcal N\!\left(0,
\begin{pmatrix}K_{\mathrm{prior}} & K_{\mathrm{cross}}\\
K_{\mathrm{cross}}^\top & K_{\mathrm{obs}}\end{pmatrix}\right).
```

Conditioning gives the two expressions implemented in `condition_gp` below:

```math
\mu_\star = K_{\mathrm{cross}}K_{\mathrm{obs}}^{-1}\mathbf y,
\qquad
\Sigma_\star = K_{\mathrm{prior}} -
K_{\mathrm{cross}}K_{\mathrm{obs}}^{-1}K_{\mathrm{cross}}^\top.
```

Later, we mix PDE evaluations and boundary values: the same rule builds the
``K_{\mathcal L\mathcal L}``, ``K_{\mathcal L u}``, and ``K_{uu}`` blocks.
"""

# ╔═╡ c0000004-b001-4000-8000-000000000001
"""
    condition_gp(K_prior, K_cross, K_obs, y)

Posterior of a zero-mean GP after observing `y`:
- `K_prior = cov(f⋆, f⋆)` — prior covariance at the prediction points
- `K_cross = cov(f⋆, y)`  — prediction points × observations
- `K_obs   = cov(y,  y)`  — observation covariance (incl. noise)

Returns `(μ_post, Σ_post)`. We reuse this everywhere. Only the kernels that build
`K_obs`/`K_cross` change (plain evaluations for regression, derivative kernels for the PDE).
"""
function condition_gp(K_prior, K_cross, K_obs, y)
	C = cholesky(Symmetric(K_obs))                          # factor once
	μ_post = K_cross * (C \ y)                              # posterior mean
	Σ_post = Symmetric(K_prior - K_cross * (C \ K_cross'))  # posterior covariance
	return μ_post, Σ_post
end

# ╔═╡ 75ecc25f-0c82-46c9-965b-7e1a0cf991f5
md"""
## GP regression
"""

# ╔═╡ c0000005-c001-4000-8000-000000000001
md"""
We have a prior over the field. Now we **fold in measurements**: a handful of point
sensors along the transect, plus the known boundary condition ``u(0)=u(1)=0``.

Sensors: $(@bind n_sens_1d Slider(2:8; default=2, show_value=true)) ·
sensor noise σ: $(@bind σ_sens_1d Slider(0.0:0.02:0.2; default=0.04, show_value=true))

Conditioning a GP on (noisy) point evaluations is **ordinary GP regression** — nothing
probabilistic-numerics-specific yet. *You already know how to do this.* The PN leap comes
in the next step, when the "measurements" become the **PDE** itself.
"""

# ╔═╡ c0000006-0006-4000-8000-000000000006
md"""
## Advection–diffusion by hand: derivative kernels

The 1D steady advection–diffusion equation: ``\mathcal{L}u = v\,u' - \kappa\,u'' = f`` on ``(0,1)``.

$$\mathbf{K}_{\text{obs}} = \begin{pmatrix} k_{\mathcal{L}\mathcal{L}} & k_{\mathcal{L}u} \\ k_{u\mathcal{L}} & k_{uu} \end{pmatrix}$$

- Conditioning on ``\mathcal{L}u = f`` requires covariances involving ``u'`` **and** ``u''``
- For a SE kernel ``k(h) = \sigma^2 e^{-h^2/2\ell^2}`` we differentiate ``g = k`` to get ``g', g'', g^{(4)}``
"""

# ╔═╡ c0000007-0007-4000-8000-000000000007
begin
	# g'(h)  = -(h / ℓ²) · g(h)
	d1k(x, x′) = let h = x - x′
		-(h) / ℓ_k^2 * k_se(x, x′)
	end

	# g''(h) = (h² - ℓ²) / ℓ⁴ · g(h)
	d2k(x, x′) = let h = x - x′
		(h^2 - ℓ_k^2) / ℓ_k^4 * k_se(x, x′)
	end

	# g''''(h) = (h⁴ - 6h²ℓ² + 3ℓ⁴) / ℓ⁸ · g(h)
	d4k(x, x′) = let h = x - x′
		(h^4 - 6h^2 * ℓ_k^2 + 3ℓ_k^4) / ℓ_k^8 * k_se(x, x′)
	end

end

# ╔═╡ c0000007-a001-4000-8000-000000000001
md"""
Combine into operator kernels for ``\mathcal{L} = v\,\partial_x - \kappa\,\partial_x^2``
(the cross terms in ``g'''`` cancel by symmetry):
"""

# ╔═╡ c0000007-a002-4000-8000-000000000002
begin
	kLL(x, x′) = -v_1d^2 * d2k(x, x′) + κ_1d^2 * d4k(x, x′)  # Cov[ℒu, ℒu] = -v²g'' + κ²g''''
	kLu(x, x′) =  v_1d   * d1k(x, x′) - κ_1d   * d2k(x, x′)  # Cov[ℒu, u]  =  v g'  − κ g''
	kuL(x, x′) = -v_1d   * d1k(x, x′) - κ_1d   * d2k(x, x′)  # Cov[u, ℒu]  = −v g' − κ g''
end

# ╔═╡ c0000008-0008-4000-8000-000000000008
md"""
## Setup

- **PDE observations:** ``\mathcal{L}u(x_i) = v\,u'(x_i) - \kappa\,u''(x_i) = f(x_i)`` at collocation points
- **BC observations:** ``u(0) = 0``, ``u(1) = 0``
"""

# ╔═╡ 925b5bc1-2791-4637-bf64-507ccc3863e2
begin
	# A FIXED source f(x) = S·sin(fπx). The solution u then responds to the sliders:
	# stronger advection v skews it downstream, stronger diffusion κ flattens it.
	S_src = 10.0
	f₁(x) = S_src * sin(freq * π * x)

	# Exact solution of  v u′ − κ u″ + c u = S sin(ωx),  u(0) = u(1) = 0:
	# particular part A sin + B cos, homogeneous part c₁e^{r₁x} + c₂e^{r₂x} fitted to the BCs.
	function exact_advdiff_reaction(v, κ, c, S, ω)
		den = (κ*ω^2 + c)^2 + v^2*ω^2
		A = S*(κ*ω^2 + c)/den;  B = -S*v*ω/den
		up(x) = A*sin(ω*x) + B*cos(ω*x)
		disc = v^2 + 4κ*c
		if disc == 0                       # v = 0, c = 0: the homogeneous part is linear
			c₁ = -up(0.0);  c₂ = -up(1.0) - c₁
			return x -> up(x) + c₁ + c₂*x
		end
		r₁ = (v + sqrt(disc))/(2κ);  r₂ = (v - sqrt(disc))/(2κ)
		c₁, c₂ = [1.0 1.0; exp(r₁) exp(r₂)] \ [-up(0.0), -up(1.0)]
		return x -> up(x) + c₁*exp(r₁*x) + c₂*exp(r₂*x)
	end
	u_exact = exact_advdiff_reaction(v_1d, κ_1d, 0.0, S_src, freq * π)
end

# ╔═╡ c0000009-0009-4000-8000-000000000009
begin
	X_pde = range(0.0, 1.0, length = n_colloc + 2)[2:end-1]
	X_bc   = [0.0, 1.0]
	X_pred = collect(range(0, 1, length = 100));
end

# ╔═╡ c0000005-c002-4000-8000-000000000002
let
	# The (unknown) field we're measuring; zero at the coast/ends so the BC is consistent.
	u_field(x) = sin(π * x) - 0.45 * sin(3π * x)

	X_s      = collect(range(0.12, 0.88; length = n_sens_1d))           # sensor locations
	X_obs_1d = vcat(X_s, [0.0, 1.0])                                    # sensors + boundary
	y_obs_1d = vcat(u_field.(X_s), [0.0, 0.0])                          # readings + u = 0 at coast
	noise_1d = vcat(fill(max(σ_sens_1d, 1e-3)^2, n_sens_1d), [1e-8, 1e-8])  # BC ≈ exact

	# Every observation is a plain evaluation — textbook GP regression.
	K_obs   = [k_se(a, b) for a in X_obs_1d, b in X_obs_1d] + Diagonal(noise_1d)
	K_cross = [k_se(p, o) for p in X_pred,   o in X_obs_1d]
	K_prior = [k_se(p, q) for p in X_pred,   q in X_pred]
	μ_post, Σ_post = condition_gp(K_prior, K_cross, K_obs, y_obs_1d)
	σ_post = sqrt.(max.(diag(Σ_post), 0))

	fig = Figure(size = (760, 360))
	ax  = Axis(fig[1, 1]; xlabel = "position along transect", ylabel = "u(x)",
		leftspinecolor = COLOR_LATENT, rightspinecolor = COLOR_LATENT,
		topspinecolor = COLOR_LATENT, bottomspinecolor = COLOR_LATENT)
	lines!(ax, X_pred, u_field.(X_pred); color = (:gray, 0.45),
		linestyle = :dash, label = "true field")
	plot_gp!(ax, X_pred, μ_post, σ_post; color = COLOR_LATENT)
	scatter!(ax, X_s, u_field.(X_s); color = COLOR_PROBLEM,
		markersize = 11, label = "sensors")
	scatter!(ax, [0.0, 1.0], [0.0, 0.0]; color = COLOR_INFORMATION_OPERATOR,
		marker = :rect, markersize = 11, label = "boundary u = 0")
	axislegend(ax; position = :rt)
	fig
end

# ╔═╡ c0000009-a001-4000-8000-000000000001
md"""
## The block covariance matrix

$$\mathbf{K}_{\text{obs}} = \begin{pmatrix} k_{\mathcal{L}\mathcal{L}} & k_{\mathcal{L}u} \\ k_{u\mathcal{L}} & k_{uu} \end{pmatrix}$$
"""

# ╔═╡ ee000004-0001-4000-8000-000000000004
exercise(md"""
**Exercise 1 — assemble the covariance blocks.** The observations are stacked as
``[\,\mathcal{L}u(X_{\text{pde}});\; u(X_{\text{bc}})\,]`` with ``\mathcal{L} = v\,\partial_x - \kappa\,\partial_x^2``.
The diagonal blocks are given in the cell below. Fill in the two **off-diagonal blocks** `K_Lu` and `K_uL`,
and then the **cross-covariance** `K_cross` between the prediction points and the observations (two cells further down).

The four kernels `kLL`, `kLu`, `kuL`, `k_se` are defined above. Matrix comprehension: `[f(x, x′) for x in A, x′ in B]`.
""")

# ╔═╡ ee000005-0001-4000-8000-000000000005
hint(md"`K_Lu` is `Cov[ℒu(X_pde), u(X_bc)]`: rows over `X_pde`, columns over `X_bc`, operator on the *first* argument → `kLu(x, x′)`. `K_uL` is `Cov[u(X_bc), ℒu(X_pde)]`: operator on the *second* argument → `kuL(x, x′)`. Writing `kLu(x, x′)` with the ranges swapped is **not** the same: the first-derivative term flips sign.")

# ╔═╡ c0000009-a003-4000-8000-000000000003
md"""
## GP conditioning

- Cross-covariance ``\mathbf{K}_\star`` between predictions and observations
- Standard formula: ``\mu = \mathbf{K}_\star \mathbf{K}_{\text{obs}}^{-1} \mathbf{y}`` — exactly our `condition_gp`
"""

# ╔═╡ ee000007-0001-4000-8000-000000000007
details("Solution", md"""
```julia
K_Lu = [kLu(x, x′) for x in X_pde, x′ in X_bc]
K_uL = [kuL(x, x′) for x in X_bc, x′ in X_pde]
K_cross = hcat([kuL(x, x′) for x in X_pred, x′ in X_pde],
               [k_se(x, x′) for x in X_pred, x′ in X_bc])
```
""")

# ╔═╡ c0000010-a001-4000-8000-000000000001
md"""
## The result

- **Top:** GP posterior over the solution ``u(x)``
- **Bottom:** GP posterior over ``\mathcal{L}u = v\,u' - \kappa\,u''`` — the "PDE space" (recovers the source ``f``)
- Information enters in ``\mathcal{L}u`` at collocation points, then propagates to ``u`` via the kernel

Source frequency: $(freq_slider_1) Collocation points: $(colloc_slider_1)

advection ``v``: $(v_slider_1) diffusion ``κ``: $(κ_slider_1)
"""

# ╔═╡ ee000008-0001-4000-8000-000000000008
exercise(md"""
**Predict before you slide.** Where is the posterior uncertainty of ``\mathcal{L}u`` (bottom panel) smallest?

$(@bind predict_1 Radio(["at the boundary points x = 0 and x = 1", "at the collocation points", "halfway between collocation points"]))

If you *increase* the number of collocation points, does the uncertainty of ``u`` (top panel) **between** collocation points shrink too?

$(@bind predict_2 Radio(["yes — the kernel couples neighbouring points", "no — we only learn about ℒu at the collocation points"]))
""")

# ╔═╡ ee000009-0001-4000-8000-000000000009
let
	_chosen(x) = !(isnothing(x) || ismissing(x))
	a = !_chosen(predict_1) ? md"" :
		predict_1 == "at the collocation points" ?
			correct(md"We observe ``\mathcal{L}u`` there with (almost) no noise, so the band pinches to zero exactly at the collocation points.") :
			keep_working(md"Look at the bottom panel: where does the band pinch to zero?")
	b = !_chosen(predict_2) ? md"" :
		startswith(predict_2, "yes") ?
			correct(md"Right. Information enters in ``\mathcal{L}u``-space, but the kernel correlates ``\mathcal{L}u`` with ``u`` everywhere, so the top band shrinks between the points too.") :
			keep_working(md"Try the slider: watch the *top* band between two collocation points as you add more.")
	md"$(a) $(b)"
end

# ╔═╡ ee000010-0001-4000-8000-000000000010
md"""
## Optional: add the reaction term by hand
"""

# ╔═╡ ee000011-0001-4000-8000-000000000011
optional(md"""
The harbour PDE we build up to has a first-order **decay** term ``+u`` (microbial degradation). Add it to the 1D operator:
``\mathcal{L}_c u = v\,u' - \kappa\,u'' + c\,u``. Covariances are *bilinear* in the operator, so with ``g = k`` and the shorthand ``k_{\mathcal{L}\mathcal{L}}`` etc. from above,

```math
\mathrm{Cov}[\mathcal{L}_c u(x), \mathcal{L}_c u(x')] = k_{\mathcal{L}\mathcal{L}} - 2c\kappa\, g'' + c^2 k, \qquad
\mathrm{Cov}[\mathcal{L}_c u(x), u(x')] = k_{\mathcal{L}u} + c\,k, \qquad
\mathrm{Cov}[u(x), \mathcal{L}_c u(x')] = k_{u\mathcal{L}} + c\,k .
```

Write `kLL_c`, `kLu_c`, `kuL_c` below using `kLL`, `kLu`, `kuL`, `d2k`, `k_se`, `κ_1d` and `c_r`.
The source ``f`` stays the same; the decay term damps the solution (the exact solution is recomputed for the check).
""")

# ╔═╡ ee000012-0001-4000-8000-000000000012
md"c = $(@bind c_r Slider(0:1:8, default = 4, show_value = true))"

# ╔═╡ ee000013-0001-4000-8000-000000000013
begin
	kLL_c(x, x′) = missing
	kLu_c(x, x′) = missing
	kuL_c(x, x′) = missing
end

# ╔═╡ ee000014-0001-4000-8000-000000000014
let
	x, x′ = 0.3, 0.5
	if ismissing(kLL_c(x, x′)) || ismissing(kLu_c(x, x′)) || ismissing(kuL_c(x, x′))
		still_missing("the reaction-term exercise")
	else
		ok_LL = kLL_c(x, x′) ≈ kLL(x, x′) - 2c_r * κ_1d * d2k(x, x′) + c_r^2 * k_se(x, x′)
		ok_Lu = kLu_c(x, x′) ≈ kLu(x, x′) + c_r * k_se(x, x′)
		ok_uL = kuL_c(x, x′) ≈ kuL(x, x′) + c_r * k_se(x, x′)
		if ok_LL && ok_Lu && ok_uL
			correct(md"All three operator kernels are right — the plot below solves ``v u' - \kappa u'' + c u = f`` with them.")
		elseif !ok_LL
			keep_working(md"`kLL_c` is off. Expand ``(\mathcal{L} + c)(\mathcal{L}' + c)\,k``: the cross term is ``2c \cdot \mathrm{Cov}[\mathcal{L}u, u]``-like, and only the ``-\kappa g''`` part survives (the ``v g'`` parts cancel by symmetry).")
		else
			keep_working(md"`kLu_c` / `kuL_c` are off. Only *one* operator acts, and the reaction term contributes ``c\,k``.")
		end
	end
end

# ╔═╡ ee000015-0001-4000-8000-000000000015
details("Solution", md"""
```julia
kLL_c(x, x′) = kLL(x, x′) - 2c_r * κ_1d * d2k(x, x′) + c_r^2 * k_se(x, x′)
kLu_c(x, x′) = kLu(x, x′) + c_r * k_se(x, x′)
kuL_c(x, x′) = kuL(x, x′) + c_r * k_se(x, x′)
```
""")

# ╔═╡ d0000001-0001-4000-8000-000000000001
md"""
## That was tedious.

- We had to manually write out the entries of the covariance matrix
- Different kernels and different operators require new specialized implementations

### Can we do better?
- There are "probabilistic numerical computer algebra systems" ([Pförtner+ 2022](#cite-pfortner2022))
- Define actions of atomic linear operators on atomic kernels
- Users can then compose these atoms in complex ways and the system figures out how to compute the covariance matrix efficiently
- Implementations available in [Python](https://github.com/marvinpfoertner/linpde-gp) and [Julia](https://github.com/timweiland/FunctionalGPs.jl)
"""

# ╔═╡ 6a84dbb8-e00e-49a2-a43c-0c261e757985
pkgcard("FunctionalGPs.jl", "functionalgps"; v_offset = -300)

# ╔═╡ d0000005-0005-4000-8000-000000000005
md"""
## Solving the PDE with information operators

- Express the PDE and BCs as linear functionals
- Condition the GP on each piece of information
"""

# ╔═╡ d0000006-0006-4000-8000-000000000006
begin
	# GP prior
	f_prior_1d = GP(with_lengthscale(KernelFunctions.SqExponentialKernel(), ℓ_k))

	# 1. PDE: ℒu(xᵢ) = v·u′ − κ·u″ = f(xᵢ) at collocation points
	ℒ_op = v_1d * ∂(1) - κ_1d * ∂(2)
	ℒ_pde = δ(collect(X_pde)) ∘ ℒ_op
	f_pde_1d = condition_on_observation(f_prior_1d, ℒ_pde, f₁.(X_pde); noise = 1e-8)

	# 2. BCs: u(0) = u(1) = 0
	ℒ_bc = δ(X_bc)
	f_post_1d = condition_on_observation(f_pde_1d, ℒ_bc, zeros(2); noise = 1e-8)
end

# ╔═╡ d0000007-0007-4000-8000-000000000007
md"""
## Querying the posterior is also just a functional

- Want ``u(x)``? Apply the evaluation functional.
- Want ``\mathcal{L}u(x)``? Apply the operator functional.
- Want both jointly? Stack them.
"""

# ╔═╡ d0000007-a001-4000-8000-000000000001
begin
	# Query both u and ℒu at prediction points — one call
	δ_pred = δ(X_pred)
	ℒ_pred = δ(X_pred) ∘ ℒ_op
	joint_pred = StackedLinearFunctional(δ_pred, ℒ_pred)(f_post_1d; noise=1e-8)
end

# ╔═╡ d0000008-0008-4000-8000-000000000008
md"""
## Result

Same answer as the hand-derived version — but no manual kernel algebra.

frequency: $(freq_slider_2) collocation: $(colloc_slider_2) · advection ``v``: $(v_slider_2) diffusion ``κ``: $(κ_slider_2)
"""

# ╔═╡ d0000004-0004-4000-8000-000000000004
md"""
## Mix & match information operators

Every piece of knowledge is a **linear functional** we condition on — the same toolbox
we'll use on the harbor. The checkboxes right above the plot below toggle each piece of information; watch the posterior over ``u`` update.

"""

# ╔═╡ ee000018-0001-4000-8000-000000000018
exercise(md"""
**Exercise 2 — write your own information operator.** The cell above defines four functionals; the cell below is yours.
Define `ℒ_mine` as a linear functional and `y_mine` as the value it should take, in the same style. The true field `u_exact` is available as a function, so you can compute the right value. Suggestions:

- a **difference** of two evaluations, e.g. ``u(0.25) - u(0.75)``
- a second **integral sensor** over ``[0.6, 0.9]`` (copy the trapezoid rule from `∫u_true`)
- a **derivative sensor** at the left end, ``u'(0)`` (a finite difference of `u_exact` gives the value)

Then tick **Yours** in the checkboxes next to the plot below and watch the posterior. **Think:** which combinations of the five boxes are *redundant* (the plot does not change)? Which would *contradict* each other if you typed a wrong `y_mine` — and what does the GP do about a contradiction?
""")

# ╔═╡ ee000019-0001-4000-8000-000000000019
hint(md"Copy the pattern of `ℒ_cb_pde` or `ℒ_cb_int` above. Building blocks: `δ([x])`, `∂(1)`, `∫([Interval(a, b)])`, composition `∘`, and `+`, `-`, scalar `*`.")

# ╔═╡ ee000020-0001-4000-8000-000000000020
begin
	# Exercise 2: your own linear functional and its observed value
	ℒ_mine = missing
	y_mine = [u_exact(0.25) - u_exact(0.75)]
end

# ╔═╡ ee000022-0001-4000-8000-000000000022
details("Solution (difference of two evaluations)", md"""
```julia
ℒ_mine = δ([0.25]) - δ([0.75])
y_mine = [u_exact(0.25) - u_exact(0.75)]
```
""")

# ╔═╡ ee000023-0001-4000-8000-000000000023
md"""
$(@bind cb_bc CheckBox(default=true)) **Boundary** ``u(0)=u(1)=0`` — evaluation ``δ``

$(@bind cb_pde CheckBox(default=false)) **PDE** ``\mathcal{L}u = f`` at collocation points — operator ``δ\circ\mathcal{L}``

$(@bind cb_sensors CheckBox(default=false)) **Direct sensors** ``u(x_s)`` — evaluation ``δ``

$(@bind cb_integral CheckBox(default=false)) **Integral sensor** ``\int_a^b u\,dx`` — integral ``∫`` *(= Bayesian quadrature)*

$(@bind cb_mine CheckBox(default=false)) **Yours** (Exercise 2): ``\mathcal{L}_{\text{mine}}\,u = y_{\text{mine}}``
"""

# ╔═╡ 72329592-fabc-4acd-b36e-0c5deabf0416
begin
	# 2D tensor-product half-integer Matérn (ν = 7/2 per axis)
	k_2d = HalfIntegerMaternKernel(3, [0.2]) ⊗ HalfIntegerMaternKernel(3, [0.2])
	# 12×12 collocation grid inside the unit square + perimeter boundary
	xs_2d = collect(range(0.1, 0.9; length = 12))
	ys_2d = collect(range(0.1, 0.9; length = 12))
	X_c_2d = vec([[x, y] for x in xs_2d, y in ys_2d])
	X_b_2d = vcat(
			[[x, 0.0] for x in xs_2d], [[x, 1.0] for x in xs_2d],
			[[0.0, y] for y in ys_2d[2:end-1]], [[1.0, y] for y in ys_2d[2:end-1]]        )
	nothing
end

# ╔═╡ e0000001-0001-4000-8000-000000000001
md"""
## Scaling up

- So far, we've seen toy 1D examples
- Practically relevant applications are often 2D or 3D, plus a time dimension (typically). Also, the PDEs are often nonlinear.
- We also commonly want to infer parameters. Can we do this through a fully Bayesian treatment?

Next:
- How to deal with **really** large states
- Hierarchical modelling to enable parameter inference
- How to deal with **nonlinear** information
- How to treat the time dimension
"""

# ╔═╡ e0000003-0003-4000-8000-000000000003
md"""
## Dense kernel matrices don't scale

| Grid | DoF | Memory | Factorization time |
|------|-----|-------------|------|
| 30×30 | 900 | 6.5 MB | <1s |
| 150×150 | 22,500 | 4 GB | ~10s |
| 300×300 | 90,000 | 65 GB | ~minutes |
"""

# ╔═╡ d5247444-711d-47f4-b502-558280805523
md"""
# What can we do about it?
Depends on the kernel smoothness.
- For very smooth kernels (e.g. squared exponential): *Low-rank approximations*
- For less smooth kernels (e.g. Matern): *Vecchia approximations* ([Vecchia 1988](#cite-vecchia1988))
- Physical simulations can be quite rough. Thus, in the following, we cover **Vecchia approximations**.
"""

# ╔═╡ e0000004-0004-4000-8000-000000000004
md"""
## Vecchia approximations unlock sparse linear algebra

- We can find a **sparse precision matrix** ``\mathbf{Q} \approx \mathbf{K}^{-1}`` that's nearly as good
"""

# ╔═╡ e0000005-0005-4000-8000-000000000005
# TODO: Side-by-side heatmap. Left: dense K (all nonzero). Right: spy of sparse Q.
# Below: "K: 3.2 GB. Q: 0.8 MB."
let
	# Build a small dense kernel matrix for illustration
	N_demo_sparse = 200
	X_sp = collect(range(0, 1, length = N_demo_sparse))
	K_sp = [k_se(x, x′) for x in X_sp, x′ in X_sp]

	# Mock sparse precision: banded matrix (placeholder until sparse_kl_cholesky)
	bw = 5
	Q_sp = spzeros(N_demo_sparse, N_demo_sparse)
	for i in 1:N_demo_sparse, j in max(1,i-bw):min(N_demo_sparse,i+bw)
		Q_sp[i, j] = 1.0
	end
	Q_sp += 10 * I

	fig = Figure(size = (1000, 400))
	ax1 = Axis(fig[1, 1], title = "Covariance K (dense)",
		aspect = DataAspect(), yreversed = true)
	ax2 = Axis(fig[1, 2], title = "Precision Q (sparse)",
		aspect = DataAspect(), yreversed = true)

	heatmap!(ax1, K_sp, colormap = :viridis)
	spy!(ax2, Q_sp, markersize = 1, color = COLOR_LATENT)

	hidedecorations!(ax1); hidedecorations!(ax2)

	N_harbor = 20_000
	mem_K = round(N_harbor^2 * 8 / 1e9, digits = 1)
	nnz_Q = 11 * N_harbor  # ~11 nonzeros per row for a 2D mesh
	mem_Q = round(nnz_Q * 8 / 1e6, digits = 1)
	Label(fig[2, 1:2],
		"At 20k DOF:  K needs $(mem_K) GB    vs    Q needs ≈ $(mem_Q) MB",
		fontsize = 16)

	rowsize!(fig.layout, 2, Relative(0.08))
	fig
end

# ╔═╡ e0000006-0006-4000-8000-000000000006
md"""
- Zero entries in the precision matrix = **conditional independencies**
- Choose a **sparsity pattern** (which nodes are conditionally independent?)
- Find the entries that **minimize KL divergence** to the true Gaussian
- Key result: this has a **closed-form solution** ([Schäfer et al., 2021](#cite-schafer2021))
"""

# ╔═╡ 5abf86bc-2be2-4ece-8e29-1760df78eaea
pkgcard("GMRFs.jl", "gmrfs"; v_offset = -400)

# ╔═╡ e0000007-0007-4000-8000-000000000007
md"""
## Vecchia for probabilistic numerics

- This works for any kernel matrix, including the block matrices from information operators
- Result: a sparse Cholesky factor ``\mathbf{L}`` such that ``\mathbf{Q} = \mathbf{L}\mathbf{L}^\top \approx \mathbf{K}^{-1}``
"""

# ╔═╡ be6a7c2e-e6db-4021-a03e-00286b99d4b5
fg_2d = FunctionalGaussian(GP(k_2d);
			u   = δ(X_c_2d),
			uxx = δ(X_c_2d) ∘ PartialDerivative((2, 0)),
			uyy = δ(X_c_2d) ∘ PartialDerivative((0, 2)),
			u_b = δ(X_b_2d),
	)

# ╔═╡ 4cf9bbdf-c01b-4afe-af15-1c9d7483d515
fg_2d.Σ

# ╔═╡ e7ac5710-1f8f-48c0-b931-0bc6f183849c
gmrf_2d = vecchia(fg_2d; ρ = 1.0)

# ╔═╡ 74578565-b418-4bf3-87ef-d056eae3a7e6
md"""
- All operations on this so-called **Gaussian Markov Random Field** leverage the sparsity of the precision matrix
"""

# ╔═╡ b0f03964-29c2-4be6-b8d2-8b86c6d6db77
rand(gmrf_2d)

# ╔═╡ 1b54045c-6f58-42bf-815a-5ff3e2061852
std(gmrf_2d)

# ╔═╡ ee000024-0001-4000-8000-000000000024
md"""
## Stretch: solve a 2D PDE with `FunctionalGaussian`
"""

# ╔═╡ ee000025-0001-4000-8000-000000000025
optional(md"""
**Exercise (stretch) — 2D Poisson.** Solve ``\Delta u = f`` on the unit square with ``u = 0`` on the boundary, where
``f(x, y) = -2\pi^2 \sin(\pi x)\sin(\pi y)`` so that ``u = \sin(\pi x)\sin(\pi y)``.

1. Define the functional `ℒ_lap` that evaluates the **Laplacian** ``\partial_x^2 + \partial_y^2`` at the collocation points `X_c_2d` (compare with `uxx` and `uyy` in `fg_2d` above; operators can be added).
2. The cells below bundle it into a `FunctionalGaussian` with blocks `u`, `lap`, `u_b`, condition on `lap = f` and `u_b = 0`, and compare the `u` block with the exact solution.
""")

# ╔═╡ ee000026-0001-4000-8000-000000000026
hint(md"`δ(X_c_2d) ∘ PartialDerivative((2, 0))` is ``\partial_x^2`` at the points. What do you get if you add two operators before composing?")

# ╔═╡ ee000027-0001-4000-8000-000000000027
begin
	f_2d(p) = -2π^2 * sin(π * p[1]) * sin(π * p[2])
	u_2d_exact(p) = sin(π * p[1]) * sin(π * p[2])
	# slightly longer lengthscale than k_2d: the solution is a single smooth bump
	k_2d_pde = HalfIntegerMaternKernel(3, [0.3]) ⊗ HalfIntegerMaternKernel(3, [0.3])
	ℒ_lap = missing
end

# ╔═╡ ee000030-0001-4000-8000-000000000030
details("Solution", md"""
```julia
ℒ_lap = δ(X_c_2d) ∘ (PartialDerivative((2, 0)) + PartialDerivative((0, 2)))
```
""")

# ╔═╡ 24936159-6acf-44ac-934f-7fd0316ac4d5
  md"""
  # Where is the leak?

  We have a probabilistic model of the field — now we **ask a question of it**:
  given the sensor readings, *where was the pollutant released?*
  """

# ╔═╡ 6252efb4-badc-4aa2-bcac-89eb4278a391
  HTML("""
  <div style="font-family:'Plus Jakarta Sans',system-ui,sans-serif; max-width:720px; margin:0.6em auto;
              background:#fff; border:1px solid var(--tuai-lightblue,#85cbd2); border-radius:16px;
              padding:1.5em 1.7em; box-shadow:0 4px 22px rgba(26,58,91,0.10);">
 
    <div style="font-size:1.28em; font-weight:800; color:var(--tuai-darkblue,#1a3a5b); letter-spacing:-0.01em;">
      It's all one <span style="color:var(--tuai-accent,#ea4b2e);">latent Gaussian model</span>
    </div>
    <div style="font-size:0.9em; color:var(--tuai-dark,#383838); opacity:0.75; margin:0.3em 0 1.35em;">
      Probabilistic numerics doesn't <em>do</em> the inference — it supplies a prior that drops into a
      three-level hierarchical model.
    </div>
 
    <!-- θ -->
    <div style="display:flex; align-items:center; gap:1em; background:var(--tuai-gray,#f6f6f6);
                border-radius:11px; padding:0.75em 1em; border-left:6px solid #D29600;">
      <div style="font-family:'Roboto Mono',monospace; font-size:1.55em; font-weight:700; color:#D29600; width:1.6em; text-align:center;">θ</div>
      <div style="flex:1;">
        <div style="font-weight:700; color:#D29600; font-size:0.96em;">Source location <span style="opacity:0.55; font-weight:500;">— the unknown</span></div>
        <div style="font-size:0.8em; color:var(--tuai-dark,#383838); opacity:0.72;">θ = (x_src, y_src) · a flat prior over the harbor</div>
      </div>
      <div style="font-family:'Roboto Mono',monospace; font-size:1.05em; color:var(--tuai-darkblue,#1a3a5b);">p(θ)</div>
    </div>
  
    <div style="text-align:center; color:#b7c1cc; font-size:1.25em; line-height:1.1;">↓</div>

    <!-- u -->
    <div style="display:flex; align-items:center; gap:1em; background:var(--tuai-gray,#f6f6f6);
                border-radius:11px; padding:0.75em 1em; border-left:6px solid #1A3A5B;">
      <div style="font-family:'Roboto Mono',monospace; font-size:1.55em; font-weight:700; color:#1A3A5B; width:1.6em; text-align:center;">u</div>
      <div style="flex:1;">
        <div style="font-weight:700; color:#1A3A5B; font-size:0.96em;">Latent field <span style="opacity:0.55; font-weight:500;">— probabilistic numerics lives 
  here</span></div>
        <div style="font-size:0.8em; color:var(--tuai-dark,#383838); opacity:0.72;">GP / GMRF prior <span style="color:#50AAC8; font-weight:700;">× PDE 
  constraints</span></div>
      </div>
      <div style="font-family:'Roboto Mono',monospace; font-size:1.05em; color:var(--tuai-darkblue,#1a3a5b);">p(u | θ)</div>
    </div>
 
    <div style="text-align:center; color:#b7c1cc; font-size:1.25em; line-height:1.1;">↓</div>
 
    <!-- y -->
    <div style="display:flex; align-items:center; gap:1em; background:var(--tuai-gray,#f6f6f6);
                border-radius:11px; padding:0.75em 1em; border-left:6px solid #50AAC8;">
      <div style="font-family:'Roboto Mono',monospace; font-size:1.55em; font-weight:700; color:#50AAC8; width:1.6em; text-align:center;">y</div>
      <div style="flex:1;">
        <div style="font-weight:700; color:#3d8faa; font-size:0.96em;">Sensor readings <span style="opacity:0.55; font-weight:500;">— ... and here!</span></div>
        <div style="font-size:0.8em; color:var(--tuai-dark,#383838); opacity:0.72;">noisy observations of the field</div>
      </div>
      <div style="font-family:'Roboto Mono',monospace; font-size:1.05em; color:var(--tuai-darkblue,#1a3a5b);">p(y | u, θ)</div>
    </div>
 
    <!-- joint -->
    <div style="text-align:center; font-family:'Roboto Mono',monospace; font-size:1.08em;
                margin:1.25em 0 0.1em; padding-top:1em; border-top:1px dashed var(--tuai-lightblue,#85cbd2);">
      p(θ, u, y) = <span style="color:#D29600; font-weight:700;">p(θ)</span> · <span style="color:#1A3A5B; font-weight:700;">p(u | θ)</span> · <span style="color:#50AAC8;   font-weight:700;">p(y | u, θ)</span>
    </div>
  </div>
  """)

# ╔═╡ 200a9e77-74ad-49c9-a1d3-d1b9194fafa4
md"""
  # We write that program in `Latte.jl`

  Latte is a framework for **latent Gaussian modelling**. Because it knows the model
  has latent-Gaussian structure, it unlocks inference built for exactly that —
  **INLA** (integrated nested Laplace approximation; [Rue+ 2009](#cite-rue2009)) — instead of generic MCMC.
"""

# ╔═╡ 6a021524-df13-46ad-87e9-5da10bad13cc
HTML("""
    <!-- roles -->
    <div style="display:flex; justify-content:center; align-items:center; gap:0.55em; flex-wrap:wrap; margin-top:1.15em; font-size:0.82em;">
      <span style="background:#1A3A5B; color:#fff; padding:0.32em 0.75em; border-radius:999px; font-weight:600;">PN supplies the prior</span>
      <span style="color:#b7c1cc;">→</span>
      <span style="background:var(--tuai-accent,#ea4b2e); color:#fff; padding:0.32em 0.75em; border-radius:999px; font-weight:600;">Latte expresses the model</span>
      <span style="color:#b7c1cc;">→</span>
      <span style="background:#2e9e73; color:#fff; padding:0.32em 0.75em; border-radius:999px; font-weight:600;">INLA does the inference</span>
    </div>
	 """)

# ╔═╡ 4d962849-5b68-4e48-a2ff-8ceaebe5e6f5
pkgcard("Latte.jl", "latte"; v_offset = -250)

# ╔═╡ a000ce11-0000-4000-8000-000000000001
md"""
## Packaging the prior: a custom `LatentModel`

`vecchia(fg)` gives a sparse GMRF, but the inference model wants a *reusable*
prior it can re-materialise. We wrap "FunctionalGaussian → vecchia" as a
`HarborField <: LatentModel`. It drops straight into `@latte`.
"""

# ╔═╡ a000ce11-0000-4000-8000-000000000002
begin
	# A GMRF LatentModel: a vecchia-sparsified FunctionalGaussian over u, its
	# advection v·∇u and Laplacian Δu at collocation points + u at the boundary
	# + u at sensors. Parameterised by the kernel lengthscale; the current `v` is
	# fixed at construction.
	struct HarborField{NT} <: LatentModel
		X_c::Vector{Vector{Float64}}
		X_b::Vector{Vector{Float64}}
		X_sensor::Vector{Vector{Float64}}
		velocity::Tuple{Float64, Float64}   # advection current v = (vₓ, v_y)
		ρ::Float64        # vecchia sparsity radius
		n::Int            # latent dimension
		ranges::NT        # named-block layout (u, adv, Δu, u_bc, u_sensor), for nameview
	end

	# GP prior output variance, fixed to about (plume amplitude)^2. Matching the prior
	# amplitude to the field is what makes the source identifiable from sparse sensors
	# (a unit-variance prior leaves the PN solve too uncertain at the sensors).
	output_variance = 0.04


	# Build the joint FunctionalGaussian prior at a given lengthscale.
	_make_fg(X_c, X_b, X_sensor, v, ℓ) = FunctionalGaussian(
		GP(output_variance * (HalfIntegerMaternKernel(3, [ℓ]) ⊗ HalfIntegerMaternKernel(3, [ℓ])));
		u        = δ(X_c),
		adv      = δ(X_c) ∘ (v[1] * PartialDerivative((1, 0)) +
			v[2] * PartialDerivative((0, 1))),                       # v·∇u
		Δu       = δ(X_c) ∘ (PartialDerivative((2, 0)) + PartialDerivative((0, 2))),  # Δu
		u_bc     = δ(X_b),
		u_sensor = δ(X_sensor),
	)
	_make_fg(m::HarborField, ℓ) = _make_fg(m.X_c, m.X_b, m.X_sensor, m.velocity, ℓ)

	function HarborField(X_c, X_b, X_sensor; velocity, ρ = 0.7)
		fg = _make_fg(X_c, X_b, X_sensor, velocity, 1.0)   # layout is lengthscale-independent
		ks = keys(fg)
		ranges = NamedTuple{ks}(map(n -> block_range(fg, n), ks))
		return HarborField(X_c, X_b, X_sensor, velocity, Float64(ρ),
			sum(length, values(ranges)), ranges)
	end

	# GMRF LatentModel interface — materialise a vecchia GMRF at a given lengthscale.
	Base.length(m::HarborField) = m.n
	GaussianMarkovRandomFields.hyperparameters(::HarborField) = (lengthscale = Real,)
	(m::HarborField)(; lengthscale, _...) = vecchia(_make_fg(m, lengthscale); ρ = m.ρ)
	GaussianMarkovRandomFields.precision_matrix(m::HarborField; lengthscale, _...) =
		precision_matrix(m(; lengthscale))
	GaussianMarkovRandomFields.mean(m::HarborField; _...) = zeros(m.n)
	GaussianMarkovRandomFields.constraints(::HarborField; _...) = nothing
	GaussianMarkovRandomFields.model_name(::HarborField) = :harbor
	function FunctionalGPs.nameview(m::HarborField, x::AbstractVector)
		ks = keys(m.ranges)
		return NamedTuple{ks}(ntuple(i -> view(x, m.ranges[ks[i]]), length(ks)))
	end
end

# ╔═╡ 5222c42c-2613-4528-8ef8-385f5702e3f4
md"""
## The model, expressed in Latte
"""

# ╔═╡ ee000031-0001-4000-8000-000000000031
exercise(md"""
**Exercise 3 — read the model** (no code). Look at the `@latte` model two cells below. Which of these are **hyperparameters** — the handful of unknowns INLA explores on a grid, as opposed to the latent field or fixed constants?

$(@bind q_hyper MultiCheckBox(["x_src", "y_src", "x", "κ_TRUE", "lengthscale", "blocks"]))

`σ_phys_h = 1e-3` is the noise on the PDE residual. Making it *larger* means…

$(@bind q_phys Radio(["we trust the PDE less — the physics becomes a soft constraint", "the sensors become noisier", "the source becomes wider"]))
""")

# ╔═╡ ee000032-0001-4000-8000-000000000032
let
	sel = q_hyper isa AbstractVector ? Set(String.(q_hyper)) : Set{String}()
	a = isempty(sel) ? md"" :
		sel == Set(["x_src", "y_src"]) ?
			correct(md"`x_src` and `y_src` are the hyperparameters. `x` is the *latent* field (thousands of dimensions, handled by the Laplace approximation), `κ_TRUE` is a constant, the `lengthscale` is pinned to 0.25 in `field(; lengthscale = 0.25)`, and `blocks` is just a named view of `x`.") :
			keep_working(md"Hyperparameters are the ones with a `~` prior and **no** `@random` in front. There are two.")
	b = (isnothing(q_phys) || ismissing(q_phys)) ? md"" :
		startswith(q_phys, "we trust") ?
			correct(md"Yes — the PDE is an observation like any other, and its noise level says how much we believe it. Try `σ_phys_h = 0.1` later and watch the concentration field go wobbly.") :
			keep_working(md"`σ_phys_h` appears only in the `y_phys[i] ~ Normal(r_i, σ_phys_h)` line. Which observations does it affect?")
	md"$(a) $(b)"
end

# ╔═╡ ee000033-0001-4000-8000-000000000033
exercise(md"""
**Exercise 4 — fill in the physics.** In the model below, the PDE residual `r_i` is missing. The steady-state PDE is

```math
\mathbf{v}\cdot\nabla u \;-\; \kappa\,\Delta u \;+\; u \;=\; f(\mathbf{x};\,x_{\text{src}}, y_{\text{src}}),
```

so the residual at collocation point `i` is *left-hand side minus right-hand side*. Write it using the named blocks of the latent field:
`blocks.adv[i]` (that is ``\mathbf{v}\cdot\nabla u``), `blocks.Δu[i]`, `blocks.u[i]`, the constant `κ_TRUE`, and the source value `f_i` computed just above it.
""")

# ╔═╡ ee000034-0001-4000-8000-000000000034
hint(md"Translate the displayed equation term by term. The `HarborField` already contains the advection block `adv` = ``\mathbf{v}\cdot\nabla u``, so no velocity appears in your line.")

# ╔═╡ ee000037-0001-4000-8000-000000000037
details("Solution", md"""
```julia
r_i = blocks.adv[i] - κ_TRUE * blocks.Δu[i] + blocks.u[i] - f_i
```
""")

# ╔═╡ 20000003-0003-4000-8000-000000000003
md"""
## The answer
"""

# ╔═╡ 20000004-b001-4000-8000-000000000001
md"""
## Adding sensors to increase our confidence

- To obtain a more confident source estimate, we place more sensors
- Here we do it manually, but our machinery would also allow us to optimize sensor placement (experimental design)
"""

# ╔═╡ ee000038-0001-4000-8000-000000000038
md"""
## Exercise 5 — the sensor placement challenge
"""

# ╔═╡ ee000039-0001-4000-8000-000000000039
exercise(md"""
You get a budget of **two extra sensors**. Where do you put them to shrink the 90 % credible box the most?

- **Score** = area of the 90 % credible box in km² (smaller is better). The six-sensor baseline is shown below.
- **Rules:** no peeking at the true source — use only the posterior you have (the credible box and the concentration / uncertainty maps above). Sensors must be in the water.
- Read coordinates (in km) off the map below, enter them in `my_sensors`, and wait ~10 s. Then shout your score.

**Think first:** the box is much wider in ``x`` than in ``y``. Why? Where would a sensor reading change the most if the source moved along ``x``? And remember there is a current — which way does the plume drift?
""")

# ╔═╡ ee000041-0001-4000-8000-000000000041
# Exercise 5: your two extra sensor positions in km, e.g. [(0.6, 1.2), (0.3, 0.9)]
my_sensors = missing

# ╔═╡ 30000001-0001-4000-8000-000000000001
md"""
## Going nonlinear: saturating biodegradation

- So far the reaction is **linear** (`+u`, first-order decay)
- Real microbial uptake saturates: there is a maximum rate the microbes can sustain.
- Swapping the reaction for the **Monod** form ([Monod 1949](#cite-monod1949)) `R(u) = Vmax·u/(Kₘ + u)` makes the PDE constraint **nonlinear in the latent field** `u`.
"""

# ╔═╡ 30000002-0002-4000-8000-000000000002
begin
	# Matched-slope Monod: low-concentration slope Vmax/KM = 1 matches the linear `+u`;
	# the nonlinearity is the high-u saturation.
	KM_monod   = 0.08
	Vmax_monod = KM_monod
	md"Monod parameters set (matched slope; fixed GP output variance)."
end

# ╔═╡ 30000003-0003-4000-8000-000000000003
let
	us = range(0, 0.35; length = 200)
	fig = Figure(size = (520, 340))
	ax = Axis(fig[1, 1]; xlabel = "concentration u", ylabel = "reaction rate R(u)",
		title = "Reaction term: linear vs saturating (Monod)")
	lines!(ax, us, 1.0 .* us; color = TUE_AI_COLORS.darkblue, linewidth = 2.5, label = "linear  u")
	lines!(ax, us, Vmax_monod .* us ./ (KM_monod .+ us); color = TUE_AI_COLORS.accent,
		linewidth = 2.5, label = "Monod  Vmax·u/(Kₘ+u)")
	axislegend(ax; position = :lt)
	fig
end

# ╔═╡ f0000002-0002-4000-8000-000000000002
md"""
## Gauss-Newton → Laplace approximation

- **Idea:** linearize around the current best guess, solve the linear problem, repeat
- Each iteration:
  1. Compute the Jacobian of the nonlinear residual at the current mode
  2. This gives a **linear** problem → condition the GP / GMRF (sparse!)
  3. Update the mode, repeat until convergence
- The result is a **Laplace approximation** to the posterior
- Latte automatically detects nonlinear residuals and routes them to a Gauss–Newton `NonlinearLeastSquaresModel`
"""

# ╔═╡ 70000001-0001-4000-8000-000000000001
md"""
## Efficient inference along time via Kalman filtering & smoothing

- So far: steady-state **snapshots**
- A real leak *evolves*: the plume drifts, spreads, and decays
- The natural probabilistic tool for time is the **Kalman filter and smoother** ([Särkkä+ 2013](#cite-sarkka2013))


- Key connection: Certain Gaussian processes can be expressed through linear-Gaussian **state-space models**
- These are **Markov in time**: The field at time ``t`` depends on the past only through ``t\!-\!1``
- Filtering (a forward pass) and smoothing (a backward pass) have direct equivalents on the sparse precision matrix side
"""

# ╔═╡ 70000002-0002-4000-8000-000000000002
md"""
## Markov in time ⇒ (block-)tridiagonal precision

- Markov-in-time => information only flows across direct neighbours => **tridiagonal** precision matrix
- Tridiagonal factorizations and solves are ``O(T)``;  they reproduce exactly the forward–backward Kalman recursion
- Now: Insert a whole **spatial field** into each timestep
- Then the joint spacetime precision matrix becomes **block-tridiagonal**
- Here: **Integrated Wiener process** in time (position + velocity) ⊗ spatial Vecchia approximation to a Matern
"""

# ╔═╡ 70000003-0003-4000-8000-000000000003
begin
	# ── harbor geometry (km frame), reloaded under st_ names ──
	st_geo = load(joinpath(@__DIR__, "assets", "harbor_geometry.jld2"))
	st_KM = 1.0e-3
	st_pt(p) = p isa Tuple ? p : getproperty(p, :data)
	st_s(p)  = (Float64(st_pt(p)[1]) * st_KM, Float64(st_pt(p)[2]) * st_KM)
	st_coast   = st_s.(st_geo["coast"]);   st_island  = st_s.(st_geo["island"])
	st_sensors = st_s.(st_geo["sensors"]); st_source  = st_s(st_geo["source"])
	st_bb0 = st_geo["domain_bbox"]
	st_bbox = (xmin = st_bb0.xmin*st_KM, xmax = st_bb0.xmax*st_KM,
	           ymin = st_bb0.ymin*st_KM, ymax = st_bb0.ymax*st_KM)
	st_wpoly = vcat(collect(st_coast), [(st_coast[end][1], st_bbox.ymin), (st_coast[1][1], st_bbox.ymin)])
	function st_inpoly(p, poly)
		x, y = p; n = length(poly); inside = false; j = n
		for i in 1:n
			xi, yi = poly[i]; xj, yj = poly[j]
			((yi > y) != (yj > y)) && (x < (xj-xi)*(y-yi)/(yj-yi)+xi) && (inside = !inside); j = i
		end
		inside
	end
	st_iswater(p) = st_inpoly(p, st_wpoly) && (length(st_island) <= 3 || !st_inpoly(p, st_island))

	# ── physics (advection–diffusion + Monod reaction) and the time grid ──
	st_v = (0.12, -0.34); st_κ = 0.02; st_w0 = 0.25; st_Vmax = 0.12; st_KMmonod = 0.3
	st_R(u)  = st_Vmax*u/(st_KMmonod + u)
	st_dR(u) = st_Vmax*st_KMmonod/(st_KMmonod + u)^2
	st_emit(p, xs, ys) = exp(-((p[1]-xs)^2 + (p[2]-ys)^2)/(2*st_w0^2))
	st_ts = range(0, 2.0; length = 10); st_Δ = step(st_ts); st_T = length(st_ts)
	st_σdata = 0.02; st_σpde = 1.0e-4   # tight PDE pinning: source posterior ≈ on the true leak (see note)

	# ── spatial collocation (24×24 clipped to water) + Vecchia precision ──
	st_nx = 24
	st_gx = range(st_bbox.xmin, st_bbox.xmax; length = st_nx)
	st_gy = range(st_bbox.ymin, st_bbox.ymax; length = st_nx)
	st_wmask = [st_iswater((st_gx[i], st_gy[j])) for i in 1:st_nx, j in 1:st_nx]
	st_Nw = count(st_wmask); st_gid = zeros(Int, st_nx, st_nx); st_gid[st_wmask] .= 1:st_Nw
	st_X = [[st_gx[i], st_gy[j]] for i in 1:st_nx, j in 1:st_nx if st_wmask[i,j]]
	st_isbd = falses(st_Nw)
	for i in 1:st_nx, j in 1:st_nx
		st_wmask[i,j] || continue
		nb = ((i>1 ? st_wmask[i-1,j] : false) && (i<st_nx ? st_wmask[i+1,j] : false) &&
		      (j>1 ? st_wmask[i,j-1] : false) && (j<st_nx ? st_wmask[i,j+1] : false))
		st_isbd[st_gid[i,j]] = !nb
	end
	st_INTR = [p for p in 1:st_Nw if !st_isbd[p]]; st_BDRY = [p for p in 1:st_Nw if st_isbd[p]]
	st_SPT = [st_gid[argmin([(st_gx[i]-sx)^2 + (st_gy[j]-sy)^2 + (st_wmask[i,j] ? 0.0 : 1.0e6)
	          for i in 1:st_nx, j in 1:st_nx])] for (sx, sy) in st_sensors]
	st_nS = length(st_sensors)
	st_ℓ = 0.4
	st_fg = FunctionalGaussian(GP(HalfIntegerMaternKernel(3, [st_ℓ]) ⊗ HalfIntegerMaternKernel(3, [st_ℓ]));
		u  = δ(st_X),
		Lu = δ(st_X) ∘ (st_v[1]*PartialDerivative((1,0)) + st_v[2]*PartialDerivative((0,1)) -
		                st_κ*(PartialDerivative((2,0)) + PartialDerivative((0,2)))))
	st_gv = vecchia(st_fg; ρ = 1.3)
	st_Qspace = sparse(precision_matrix(st_gv)); st_Qspace = 0.5*(st_Qspace + st_Qspace')
	st_Nspace = size(st_Qspace, 1); st_ur = block_range(st_fg, :u); st_Lr = block_range(st_fg, :Lu)

	# ── block-tridiagonal spacetime precision: IWP(1) in time ⊗ Vecchia in space ──
	st_At = [1.0 st_Δ; 0.0 1.0]; st_Qnt = inv([st_Δ^3/3 st_Δ^2/2; st_Δ^2/2 st_Δ])
	st_A = kron(sparse(st_At), sparse(I, st_Nspace, st_Nspace))
	st_Qnoise = kron(sparse(st_Qnt), st_Qspace); st_Q0 = kron(sparse(I, 2, 2), st_Qspace)
	st_Ns = 2*st_Nspace; st_AtQn = st_A'*st_Qnoise
	st_dfirst = st_Q0 + st_AtQn*st_A; st_dmid = st_Qnoise + st_AtQn*st_A
	st_dlast = st_Qnoise; st_offd = -(st_Qnoise*st_A)
	st_Jst = let RR = Int[], CC = Int[], VV = Float64[]
		function place!(B, bi, bj)
			r, c, v = findnz(B); ro = (bi-1)*st_Ns; co = (bj-1)*st_Ns
			append!(RR, ro .+ r); append!(CC, co .+ c); append!(VV, v)
		end
		place!(st_dfirst, 1, 1)
		for k in 2:st_T-1; place!(st_dmid, k, k); end
		place!(st_dlast, st_T, st_T)
		for k in 1:st_T-1; place!(st_offd, k+1, k); place!(sparse(st_offd'), k, k+1); end
		sparse(RR, CC, VV, st_Ns*st_T, st_Ns*st_T)
	end
	st_toff(k) = (k-1)*st_Ns; st_uidx(p, k) = st_toff(k) + st_ur[p]; st_Luidx(p, k) = st_toff(k) + st_Lr[p]

	# ── index arrays for the PPL body ──
	st_ICP = st_INTR
	st_COLP = Int[]; st_COLK = Int[]
	for k in 1:st_T-1, p in st_INTR; push!(st_COLP, p); push!(st_COLK, k); end
	st_BCP = Int[]; st_BCK = Int[]
	for k in 1:st_T, p in st_BDRY; push!(st_BCP, p); push!(st_BCK, k); end
	st_SS = [s for s in 1:st_nS for k in 1:st_T]; st_SK = [k for s in 1:st_nS for k in 1:st_T]

	# ── honest fine-FD nonlinear Crank–Nicolson data (Dirichlet u=0, no inverse crime) ──
	st_dataS = let nf = 60
		xf = range(st_bbox.xmin, st_bbox.xmax; length = nf); yf = range(st_bbox.ymin, st_bbox.ymax; length = nf)
		hx = step(xf); hy = step(yf)
		wm = [st_iswater((xf[i], yf[j])) for i in 1:nf, j in 1:nf]; Nf = count(wm)
		gf = zeros(Int, nf, nf); gf[wm] .= 1:Nf
		bd = falses(Nf)
		for i in 1:nf, j in 1:nf
			wm[i,j] || continue
			nb = ((i>1 ? wm[i-1,j] : false) && (i<nf ? wm[i+1,j] : false) &&
			      (j>1 ? wm[i,j-1] : false) && (j<nf ? wm[i,j+1] : false)); bd[gf[i,j]] = !nb
		end
		bdi = findall(bd)
		Lr = Int[]; Lc = Int[]; Lv = Float64[]
		aL(r, c, v) = (push!(Lr, r); push!(Lc, c); push!(Lv, v))
		for i in 1:nf, j in 1:nf
			wm[i,j] || continue; r = gf[i,j]; aL(r, r, -2st_κ/hx^2 - 2st_κ/hy^2)
			i>1 && wm[i-1,j] && aL(r, gf[i-1,j], st_κ/hx^2 + st_v[1]/(2hx))
			i<nf && wm[i+1,j] && aL(r, gf[i+1,j], st_κ/hx^2 - st_v[1]/(2hx))
			j>1 && wm[i,j-1] && aL(r, gf[i,j-1], st_κ/hy^2 + st_v[2]/(2hy))
			j<nf && wm[i,j+1] && aL(r, gf[i,j+1], st_κ/hy^2 - st_v[2]/(2hy))
		end
		L = sparse(Lr, Lc, Lv, Nf, Nf); Imat = sparse(I, Nf, Nf)
		fsp = [gf[argmin([(xf[i]-sx)^2 + (yf[j]-sy)^2 + (wm[i,j] ? 0.0 : 1.0e6)
		       for i in 1:nf, j in 1:nf])] for (sx, sy) in st_sensors]
		uf = [st_emit((xf[i], yf[j]), st_source[1], st_source[2]) for i in 1:nf, j in 1:nf if wm[i,j]]
		uf[bdi] .= 0.0
		data = zeros(st_nS*st_T)
		for k in 1:st_T
			for (s, ps) in enumerate(fsp); data[(s-1)*st_T + k] = uf[ps]; end
			if k < st_T
				rhs = uf .+ st_Δ/2 .* (L*uf .- st_R.(uf)); w = copy(uf)
				for _ in 1:25
					Fw = w .- st_Δ/2 .* (L*w .- st_R.(w)) .- rhs
					Jw = Imat - st_Δ/2 .* L + st_Δ/2 .* spdiagm(0 => st_dR.(w))
					w .-= Jw \ Fw; norm(Fw, Inf) < 1.0e-11 && break
				end
				uf = w; uf[bdi] .= 0.0
			end
		end
		data .+ st_σdata .* randn(Random.MersenneTwister(7), st_nS*st_T)
	end
	st_yic = zeros(length(st_INTR)); st_ybc = zeros(length(st_BCP)); st_ycol = zeros(length(st_COLP))
	st_yall = vcat(st_yic, st_ybc, st_ycol, st_dataS); nothing
end

# ╔═╡ 70000004-0004-4000-8000-000000000004
begin
	# Package the spacetime GMRF as a LatentModel (same interface as HarborField).
	struct STField <: LatentModel
		n::Int
		gmrf::GaussianMarkovRandomFields.AbstractGMRF
	end
	Base.length(m::STField) = m.n
	GaussianMarkovRandomFields.hyperparameters(::STField) = NamedTuple()
	(m::STField)(; _...) = m.gmrf
	GaussianMarkovRandomFields.precision_matrix(m::STField; _...) = GaussianMarkovRandomFields.precision_matrix(m.gmrf)
	GaussianMarkovRandomFields.mean(m::STField; _...) = zeros(m.n)
	GaussianMarkovRandomFields.constraints(::STField; _...) = nothing
	GaussianMarkovRandomFields.model_name(::STField) = :st_field
	st_FIELD = STField(st_Ns*st_T, GMRF(zeros(st_Ns*st_T), st_Jst)); nothing
end

# ╔═╡ b0000004-a001-4000-8000-000000000001
begin
	# Load pre-processed harbor geometry (exported by explore_harbor.jl)
	geo = load(joinpath(@__DIR__, "assets", "harbor_geometry.jld2"))
	harbor_coast = geo["coast"]
	harbor_island = geo["island"]
	sensor_pos = geo["sensors"]
	source_true = geo["source"]
	domain_bbox = geo["domain_bbox"]
	harbor_name = geo["name"]
	n_sensors = length(sensor_pos)
	nothing
end

# ╔═╡ b0000005-0005-4000-8000-000000000005
let
	WATER = RGBf(0.62, 0.79, 0.90)
	LAND = RGBf(0.88, 0.85, 0.80)
	COAST = RGBf(0.28, 0.26, 0.24)
	ISLAND_COL = RGBf(0.72, 0.80, 0.68)
	SENSOR_STROKE = TUE_AI_COLORS.dark
	SOURCE_COL = COLOR_PROBLEM
	DASH_COL = RGBAf(0.3, 0.3, 0.3, 0.3)

	bb = domain_bbox
	fig = Figure(size = (1000, 700), backgroundcolor = :white)
	ax = Axis(fig[1, 1], aspect = DataAspect(), backgroundcolor = WATER)
	hidedecorations!(ax)
	hidespines!(ax)
	xlims!(ax, bb.xmin - 20, bb.xmax + 20)
	ylims!(ax, bb.ymin - 20, bb.ymax + 20)

	# ── Land & water ──
	poly!(ax, Rect2f(bb.xmin - 100, bb.ymin - 100,
		(bb.xmax - bb.xmin) + 200, (bb.ymax - bb.ymin) + 200), color = LAND)
	water_poly = vcat(harbor_coast,
		[Point2f(harbor_coast[end][1], bb.ymin)],
		[Point2f(harbor_coast[1][1], bb.ymin)])
	poly!(ax, water_poly, color = WATER)
	lines!(ax, harbor_coast, color = COAST, linewidth = 2.5)

	# ── Island ──
	if length(harbor_island) > 3
		poly!(ax, harbor_island, color = LAND, strokecolor = COAST, strokewidth = 1.5)
	end

	# ── Open sea boundary ──
	lines!(ax, [harbor_coast[1], Point2f(harbor_coast[1][1], bb.ymin)],
		color = DASH_COL, linewidth = 1.5, linestyle = :dash)
	lines!(ax, [Point2f(harbor_coast[1][1], bb.ymin),
		Point2f(harbor_coast[end][1], bb.ymin)],
		color = DASH_COL, linewidth = 1.5, linestyle = :dash)
	lines!(ax, [Point2f(harbor_coast[end][1], bb.ymin), harbor_coast[end]],
		color = DASH_COL, linewidth = 1.5, linestyle = :dash)

	# ── Sensors ──
	scatter!(ax, sensor_pos, marker = :circle, markersize = 18,
		color = :white, strokecolor = SENSOR_STROKE, strokewidth = 2.5)
	for (i, p) in enumerate(sensor_pos)
		text!(ax, p[1] + 16, p[2] + 12, text = "S$i", fontsize = 12,
			color = RGBf(0.2, 0.2, 0.2))
	end

	# ── Labels ──
	wcx = (bb.xmin + bb.xmax) / 2
	text!(ax, wcx, bb.ymax - 10, text = harbor_name,
		fontsize = 15, color = COAST, align = (:center, :top))
	text!(ax, wcx, bb.ymin + 30, text = "Open Sea",
		fontsize = 12, font = :italic, color = RGBf(0.35, 0.45, 0.60),
		align = (:center, :bottom))

	# ── Scale bar ──
	lines!(ax, Point2f[(bb.xmin + 30, bb.ymin + 50),
		(bb.xmin + 230, bb.ymin + 50)], color = COAST, linewidth = 2)
	text!(ax, bb.xmin + 130, bb.ymin + 32, text = "200 m",
		fontsize = 11, color = COAST, align = (:center, :top))

	# ── Current arrows (gentle drift toward the open sea) ──
	function _inpoly(p, poly)
		x, y = p; n = length(poly); inside = false; j = n
		for i in 1:n
			xi, yi = poly[i]; xj, yj = poly[j]
			((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) &&
				(inside = !inside)
			j = i
		end
		return inside
	end
	cur = v_current ./ hypot(v_current[1], v_current[2])
	alen = 0.07 * (bb.ymax - bb.ymin)
	cur_org = Point2f[]
	for gx in range(bb.xmin, bb.xmax; length = 6), gy in range(bb.ymin, bb.ymax; length = 6)
		p = Point2f(gx, gy)
		if _inpoly(p, water_poly) && (length(harbor_island) <= 3 || !_inpoly(p, harbor_island))
			push!(cur_org, p)
		end
	end
	arrows2d!(ax, cur_org, [Vec2f(cur[1] * alen, cur[2] * alen) for _ in cur_org],
		color = RGBAf(0.20, 0.30, 0.50, 0.30))

	fig
end

# ╔═╡ 20000002-a001-4000-8000-000000000001
begin
	# Rescale OSM geometry from metres → km so kernel lengthscales sit near 1.
	KM_SCALE = 1.0e-3
	_scale(p) = (Float64(p[1]) * KM_SCALE, Float64(p[2]) * KM_SCALE)
	COAST_km   = _scale.(harbor_coast)
	ISLAND_km  = _scale.(harbor_island)
	SENSORS_km = _scale.(sensor_pos)
	SOURCE_km  = _scale(source_true)
	BBOX_km = (
		xmin = domain_bbox.xmin * KM_SCALE, xmax = domain_bbox.xmax * KM_SCALE,
		ymin = domain_bbox.ymin * KM_SCALE, ymax = domain_bbox.ymax * KM_SCALE,
	)
	WATER_POLY_km = vcat(
		collect(COAST_km),
		[(COAST_km[end][1], BBOX_km.ymin), (COAST_km[1][1], BBOX_km.ymin)],
	); nothing
end

# ╔═╡ c0000005-0005-4000-8000-000000000005
let
	K = [k_se(x, x′) for x in X_pred, x′ in X_pred] + 1e-8I
	L_chol = cholesky(Symmetric(K)).L
	fig = Figure(size = (900, 350))
	ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u(x)")
	for _ in 1:5
		lines!(ax, X_pred, L_chol * randn(length(X_pred)), color = (COLOR_LATENT, 0.5))
	end
	band!(ax, X_pred, -2sqrt.(diag(K)), 2sqrt.(diag(K)),
		color = (COLOR_LATENT, 0.12))
	fig
end

# ╔═╡ c0000009-a002-4000-8000-000000000002
begin
	n_pde = length(X_pde)
	n_bc  = length(X_bc)
	n_obs = n_pde + n_bc

	# Diagonal blocks — given
	K_LL = [kLL(x, x′) for x in X_pde, x′ in X_pde]   # Cov[ℒu(X_pde), ℒu(X_pde)]
	K_uu = [k_se(x, x′) for x in X_bc, x′ in X_bc]    # Cov[u(X_bc),  u(X_bc)]

	# Off-diagonal blocks — your turn (Exercise 1)
	K_Lu = missing
	K_uL = missing

	# Ignore this bit — it only keeps the notebook alive until the blocks above are filled in.
	K_obs = if ismissing(K_Lu) || ismissing(K_uL)
		missing
	else
		[K_LL K_Lu; K_uL K_uu] + 1e-8I
	end
end

# ╔═╡ c0000009-a004-4000-8000-000000000004
begin
	# We want the posterior at the prediction points X_pred both in the "original space"
	# (u itself) and in the "PDE space" (ℒu). Each needs its own cross-covariance between
	# the prediction points and the observations [ℒu(X_pde); u(X_bc)].

	# Cov[ℒu(X_pred), observations] — given
	K_cross_L = hcat(
		[kLL(x, x′) for x in X_pred, x′ in X_pde],
		[kLu(x, x′) for x in X_pred, x′ in X_bc]
	)

	# Cov[u(X_pred), observations] — your turn (Exercise 1, part 2). Same column layout.
	K_cross = missing

	# Observation values
	y_obs = vcat(f₁.(X_pde), zeros(n_bc))
end

# ╔═╡ c0000009-a005-4000-8000-000000000005
byhand = if ismissing(K_obs) || ismissing(K_cross)
	missing
else let
	# GP posterior for u — same conditioning rule as the regression step,
	# now with derivative kernels building K_obs / K_cross.
	K_prior = [k_se(x, x′) for x in X_pred, x′ in X_pred]
	μ_post, Σ_post = condition_gp(K_prior, K_cross, K_obs, y_obs)
	σ_post = sqrt.(max.(diag(Σ_post), 0.0))

	# GP posterior for ℒu — condition in "PDE space" with the same helper
	K_prior_L = [kLL(x, x′) for x in X_pred, x′ in X_pred]
	μ_post_L, Σ_post_L = condition_gp(K_prior_L, K_cross_L, K_obs, y_obs)
	σ_post_L = sqrt.(max.(diag(Σ_post_L), 0.0))
	(; μ_post, σ_post, Σ_post, μ_post_L, σ_post_L, Σ_post_L)
end end

# ╔═╡ ee000006-0001-4000-8000-000000000006
let
	K_Lu_ref = [kLu(x, x′) for x in X_pde, x′ in X_bc]
	K_uL_ref = [kuL(x, x′) for x in X_bc, x′ in X_pde]
	K_uL_wrong = [kLu(x, x′) for x in X_bc, x′ in X_pde]   # operator on the wrong argument
	K_cross_ref = hcat([kuL(x, x′) for x in X_pred, x′ in X_pde],
		[k_se(x, x′) for x in X_pred, x′ in X_bc])
	if ismissing(K_obs)
		still_missing("Exercise 1 (the two off-diagonal blocks)")
	elseif size(K_Lu) != (n_pde, n_bc) || size(K_uL) != (n_bc, n_pde)
		keep_working(md"Check the shapes: `K_Lu` should be `$(n_pde)×$(n_bc)` (PDE rows, BC columns) and `K_uL` the other way round.")
	elseif !(K_Lu ≈ K_Lu_ref)
		keep_working(md"`K_Lu` has the right shape but wrong entries. It is `Cov[ℒu(x), u(x′)]` — the operator acts on the *left* argument only.")
	elseif K_uL ≈ K_uL_wrong && v_1d != 0
		keep_working(md"You used `kLu` for `K_uL` with the ranges swapped. That puts the operator on the wrong argument, and the advection term ``v\,\partial_x`` flips sign when the operator moves from one argument to the other. Use `kuL(x, x′)` — or, equivalently, the transpose of `K_Lu`.")
	elseif !(K_uL ≈ K_uL_ref)
		keep_working(md"`K_uL` has the right shape but wrong entries. It is `Cov[u(x), ℒu(x′)]` — use `kuL`.")
	elseif !isposdef(Symmetric(K_obs))
		keep_working(md"`K_obs` is assembled but not positive definite — double-check the blocks.")
	elseif ismissing(K_cross)
		correct(md"`K_obs` is right and positive definite. Now fill in `K_cross` (part 2).")
	elseif size(K_cross) != (length(X_pred), n_obs)
		keep_working(md"`K_cross` should be `$(length(X_pred))×$(n_obs)`: one row per prediction point, one column per observation.")
	elseif !(K_cross ≈ K_cross_ref)
		keep_working(md"`K_cross` has the right shape but wrong entries. Rows are `u(X_pred)`; columns are `ℒu(X_pde)` and then `u(X_bc)`.")
	else
		correct(md"All blocks match. Note that `K_uL == K_Lu'` always holds (same covariance, written the other way round), yet `kuL(x, x′) ≠ kLu(x, x′)` as functions: the odd-order derivative changes sign with the argument it acts on. Set the advection slider ``v`` to 0 below and the two functions coincide.")
	end
end

# ╔═╡ c0000010-0010-4000-8000-000000000010
if ismissing(byhand)
	still_missing("Exercise 1")
else let
	(; μ_post, σ_post, Σ_post, μ_post_L, σ_post_L, Σ_post_L) = byhand
	L_post_L = cholesky(Σ_post_L + 1e-8I).L
	fig = Figure(size = (900, 500))

	# ── Shared random draws for correlated samples ──
	n_samples = 3
	zs = [randn(length(X_pred)) for _ in 1:n_samples]

	L_post = cholesky(Σ_post + 1e-8I).L

	# ── Top: u(x) posterior ──
	ax1 = Axis(fig[1, 1], ylabel = "u(x)", xticklabelsvisible = false,
		leftspinecolor = COLOR_LATENT, rightspinecolor = COLOR_LATENT,
		topspinecolor = COLOR_LATENT, bottomspinecolor = COLOR_LATENT)
	band!(ax1, X_pred, μ_post .- 2σ_post, μ_post .+ 2σ_post,
		color = (COLOR_LATENT, 0.15))
	for z in zs
		lines!(ax1, X_pred, μ_post .+ L_post * z,
			color = (COLOR_LATENT, 0.4), linewidth = 1)
	end
	lines!(ax1, X_pred, μ_post, linewidth = 2.5, color = COLOR_LATENT, label = "GP posterior mean")
	lines!(ax1, X_pred, u_exact.(X_pred); label = "exact", PNMETHODS_PROBLEM_LINES_KWARGS...)
	vlines!(ax1, X_bc, color = (TUE_AI_COLORS.accent, 0.8), linestyle = (:dot, 0.5), label = "BCs: u = 0")
	axislegend(ax1, position = :rt)

	# ── Bottom: GP posterior of ℒu = v·u′ − κ·u″ (the "PDE space") ──
	ax2 = Axis(fig[2, 1], xlabel = "x", ylabel = "ℒu(x)",
		leftspinecolor = COLOR_INFORMATION_OPERATOR, rightspinecolor = COLOR_INFORMATION_OPERATOR,
		topspinecolor = COLOR_INFORMATION_OPERATOR, bottomspinecolor = COLOR_INFORMATION_OPERATOR)
	band!(ax2, X_pred, μ_post_L .- 2σ_post_L, μ_post_L .+ 2σ_post_L,
		color = (COLOR_INFORMATION_OPERATOR, 0.15))
	for z in zs
		lines!(ax2, X_pred, μ_post_L .+ L_post_L * z,
			color = (COLOR_INFORMATION_OPERATOR, 0.4), linewidth = 1)
	end
	lines!(ax2, X_pred, μ_post_L, linewidth = 2, color = COLOR_INFORMATION_OPERATOR,
		label = "GP posterior")
	lines!(ax2, X_pred, f₁.(X_pred); label = "f(x) (source)", PNMETHODS_PROBLEM_LINES_KWARGS...)
	scatter!(ax2, X_pde, f₁.(X_pde), markersize = 8, color = COLOR_PROBLEM,
		strokewidth = 1, strokecolor = :black, label = "observations")

	linkxaxes!(ax1, ax2)
	rowsize!(fig.layout, 2, Relative(0.3))
	fig
end end

# ╔═╡ ee000016-0001-4000-8000-000000000016
if ismissing(kLL_c(0.3, 0.5)) || ismissing(kLu_c(0.3, 0.5)) || ismissing(kuL_c(0.3, 0.5))
	still_missing("the reaction-term exercise")
else let
	u_exact_c = exact_advdiff_reaction(v_1d, κ_1d, c_r, S_src, freq * π)   # same source, new solution
	K_c = [
		[kLL_c(x, x′) for x in X_pde, x′ in X_pde]   [kLu_c(x, x′) for x in X_pde, x′ in X_bc]
		[kuL_c(x, x′) for x in X_bc, x′ in X_pde]    [k_se(x, x′) for x in X_bc, x′ in X_bc]
	] + 1e-8I
	K_cross_c = hcat([kuL_c(x, x′) for x in X_pred, x′ in X_pde],
		[k_se(x, x′) for x in X_pred, x′ in X_bc])
	y_c = vcat(f₁.(X_pde), zeros(length(X_bc)))
	K_prior = [k_se(x, x′) for x in X_pred, x′ in X_pred]
	μ_c, Σ_c = condition_gp(K_prior, K_cross_c, K_c, y_c)
	σ_c = sqrt.(max.(diag(Σ_c), 0.0))

	fig = Figure(size = (900, 350))
	ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u(x)", title = "ℒ = v∂ − κ∂² + $(c_r)")
	plot_gp!(ax, X_pred, μ_c, σ_c; color = COLOR_LATENT)
	lines!(ax, X_pred, u_exact_c.(X_pred); label = "exact (with decay)", PNMETHODS_PROBLEM_LINES_KWARGS...)
	lines!(ax, X_pred, u_exact.(X_pred); label = "exact (no decay)", color = (:gray, 0.6), linestyle = :dash)
	vlines!(ax, X_bc, color = (TUE_AI_COLORS.accent, 0.8), linestyle = (:dot, 0.5))
	axislegend(ax, position = :rt)
	fig
end end

# ╔═╡ d0000008-a001-4000-8000-000000000001
begin
	# Result: an MvNormal over [u(X_pred); ℒu(X_pred)]
	n_pred = length(X_pred)
	μ_joint = dist_mean(joint_pred)
	Σ_joint = dist_cov(joint_pred)

	μ_fgp   = μ_joint[1:n_pred]
	μ_fgp_L = μ_joint[n_pred+1:end]
	σ_fgp   = sqrt.(max.(diag(Σ_joint)[1:n_pred], 0.0))
	σ_fgp_L = sqrt.(max.(diag(Σ_joint)[n_pred+1:end], 0.0))
	L_joint  = cholesky(Symmetric(Σ_joint) + 1e-8I).L
	
	fig = Figure(size = (900, 500))
	zs = [randn(2 * n_pred) for _ in 1:3]

	# ── Top: u(x) ──
	ax1 = Axis(fig[1, 1], ylabel = "u(x)", xticklabelsvisible = false,
		leftspinecolor = COLOR_LATENT, rightspinecolor = COLOR_LATENT,
		topspinecolor = COLOR_LATENT, bottomspinecolor = COLOR_LATENT)
	band!(ax1, X_pred, μ_fgp .- 2σ_fgp, μ_fgp .+ 2σ_fgp,
		color = (COLOR_LATENT, 0.15))
	for z in zs
		s = L_joint * z
		lines!(ax1, X_pred, μ_fgp .+ s[1:n_pred],
			color = (COLOR_LATENT, 0.4), linewidth = 1)
	end
	lines!(ax1, X_pred, μ_fgp, linewidth = 2.5, color = COLOR_LATENT, label = "GP posterior")
	lines!(ax1, X_pred, u_exact.(X_pred); label = "exact", PNMETHODS_PROBLEM_LINES_KWARGS...)
	vlines!(ax1, X_bc, color = (TUE_AI_COLORS.accent, 0.8), linestyle = (:dot, 0.5), label = "BCs")
	axislegend(ax1, position = :rt)

	# ── Bottom: ℒu = v·u′ − κ·u″ (recovers the source f) ──
	ax2 = Axis(fig[2, 1], xlabel = "x", ylabel = "ℒu(x)",
		leftspinecolor = COLOR_INFORMATION_OPERATOR, rightspinecolor = COLOR_INFORMATION_OPERATOR,
		topspinecolor = COLOR_INFORMATION_OPERATOR, bottomspinecolor = COLOR_INFORMATION_OPERATOR)
	band!(ax2, X_pred, μ_fgp_L .- 2σ_fgp_L, μ_fgp_L .+ 2σ_fgp_L,
		color = (COLOR_INFORMATION_OPERATOR, 0.15))
	for z in zs
		s = L_joint * z
		lines!(ax2, X_pred, μ_fgp_L .+ s[n_pred+1:end],
			color = (COLOR_INFORMATION_OPERATOR, 0.4), linewidth = 1)
	end
	lines!(ax2, X_pred, μ_fgp_L, linewidth = 2, color = COLOR_INFORMATION_OPERATOR,
		label = "GP posterior")
	lines!(ax2, X_pred, f₁.(X_pred); label = "f(x) (source)", PNMETHODS_PROBLEM_LINES_KWARGS...)
	scatter!(ax2, collect(X_pde), f₁.(collect(X_pde)), markersize = 8,
		color = COLOR_PROBLEM, strokewidth = 1, strokecolor = :black,
		label = "observations")

	linkxaxes!(ax1, ax2)
	rowsize!(fig.layout, 2, Relative(0.3))
	fig
end

# ╔═╡ ee000017-0001-4000-8000-000000000017
begin
	# The four pieces of information: each is a linear functional + the value it should take.
	X_sensors  = [0.3, 0.65]
	int_region = (0.15, 0.55)
	∫u_true = let xs = range(int_region[1], int_region[2]; length = 4001)   # ∫ₐᵇ u_exact dx (trapezoid)
		step(xs) * (sum(u_exact, xs) - (u_exact(xs[1]) + u_exact(xs[end])) / 2)
	end

	# Boundary  u(0) = u(1) = 0  — evaluation functional δ
	ℒ_cb_bc      = δ(X_bc);                          y_cb_bc      = zeros(length(X_bc))
	# PDE  ℒu = f  at the collocation points — operator functional δ ∘ ℒ
	ℒ_cb_pde     = δ(collect(X_pde)) ∘ ℒ_op;         y_cb_pde     = f₁.(X_pde)
	# Direct sensors  u(x_s)  — evaluation functional δ
	ℒ_cb_sensors = δ(X_sensors);                     y_cb_sensors = u_exact.(X_sensors)
	# Integral sensor  ∫ₐᵇ u dx  — integral functional ∫
	ℒ_cb_int     = ∫([Interval(int_region[1], int_region[2])]);  y_cb_int = [∫u_true]
end

# ╔═╡ d0000003-0003-4000-8000-000000000003
begin
	# Condition the GP prior on whichever pieces of information are ticked above.
	# Every step is the same call — only the functional and the value change.
	f_cb = GP(with_lengthscale(KernelFunctions.SqExponentialKernel(), ℓ_k))
	if cb_bc
		f_cb = condition_on_observation(f_cb, ℒ_cb_bc, y_cb_bc; noise = 1e-8)
	end
	if cb_pde
		f_cb = condition_on_observation(f_cb, ℒ_cb_pde, y_cb_pde; noise = 1e-8)
	end
	if cb_sensors
		f_cb = condition_on_observation(f_cb, ℒ_cb_sensors, y_cb_sensors; noise = 1e-8)
	end
	if cb_integral
		f_cb = condition_on_observation(f_cb, ℒ_cb_int, y_cb_int; noise = 1e-8)
	end
	if cb_mine && !ismissing(ℒ_mine)
		f_cb = condition_on_observation(f_cb, ℒ_mine, y_mine; noise = 1e-8)
	end

	# Query the posterior on a fine grid — also just a functional
	post_cb = δ(X_pred)(f_cb; noise = 1e-8)
	μ_cb = collect(dist_mean(post_cb))
	σ_cb = sqrt.(collect(dist_var(post_cb)))
	nothing
end

# ╔═╡ ee000021-0001-4000-8000-000000000021
let
	if ismissing(ℒ_mine)
		still_missing("Exercise 2")
	elseif !(ℒ_mine isa FunctionalGPs.AbstractLinearFunctional)
		keep_working(md"`ℒ_mine` should be a linear functional (built from `δ`, `∂`, `∫`, `∘`, `+`, `-`), not a `$(typeof(ℒ_mine))`.")
	else
		d = ℒ_mine(GP(with_lengthscale(KernelFunctions.SqExponentialKernel(), ℓ_k)); noise = 1e-8)
		if length(dist_mean(d)) != length(y_mine)
			keep_working(md"`ℒ_mine` produces $(length(dist_mean(d))) number(s) but `y_mine` has $(length(y_mine)).")
		else
			correct(md"`ℒ_mine` is a valid functional. Under the prior it has mean $(round.(dist_mean(d); digits = 3)) and std $(round.(sqrt.(dist_var(d)); digits = 3)). Tick **Yours** next to the plot below to condition on it.")
		end
	end
end

# ╔═╡ 05d64fdd-e02a-4e73-be4b-b3cf8eb74e13
let
	fig = Figure(size = (900, 400))
	ax = Axis(fig[1, 1], xlabel = "x", ylabel = "u(x)",
		leftspinecolor = COLOR_LATENT, rightspinecolor = COLOR_LATENT,
		topspinecolor = COLOR_LATENT, bottomspinecolor = COLOR_LATENT)

	# Integral-sensor region: shade ∫ₐᵇ u (drawn first, behind everything)
	if cb_integral
		mask = int_region[1] .<= X_pred .<= int_region[2]
		band!(ax, X_pred[mask], zeros(count(mask)), μ_cb[mask], color = (COLOR_QoI, 0.25))
		text!(ax, (int_region[1] + int_region[2]) / 2, 0.12, text = "∫ u dx",
			fontsize = 14, color = COLOR_QoI, align = (:center, :center))
	end

	plot_gp!(ax, X_pred, μ_cb, σ_cb; color = COLOR_LATENT)
	lines!(ax, X_pred, u_exact.(X_pred); label = "true field u", PNMETHODS_PROBLEM_LINES_KWARGS...)

	# Mark the active information operators
	if cb_bc
		vlines!(ax, X_bc, color = (TUE_AI_COLORS.accent, 0.8), linestyle = (:dot, 0.5),
			label = "boundary u = 0")
	end
	if cb_pde
		scatter!(ax, X_pde, zeros(length(X_pde)), markersize = 9, marker = :utriangle,
			color = COLOR_INFORMATION_OPERATOR, strokewidth = 1, strokecolor = :black,
			label = "PDE collocation")
	end
	if cb_sensors
		scatter!(ax, X_sensors, u_exact.(X_sensors), markersize = 12, color = COLOR_PROBLEM,
			strokewidth = 2, strokecolor = :black, label = "sensors")
	end

	if cb_mine && !ismissing(ℒ_mine)
		text!(ax, 0.02, 0.95, text = "+ ℒ_mine u = $(round.(y_mine; digits = 3))", space = :relative,
			fontsize = 14, color = :black, align = (:left, :top))
	end

	axislegend(ax, position = :rt)
	fig
end

# ╔═╡ 04d7e02b-f42f-4ac3-976c-f2cfcd60913c
precision_matrix(gmrf_2d)

# ╔═╡ dd2547f2-cfae-40cc-8126-4a7a37220198
logpdf(gmrf_2d, mean(gmrf_2d))

# ╔═╡ ee000028-0001-4000-8000-000000000028
post_2d = if ismissing(ℒ_lap)
	missing
else let
	fg = FunctionalGaussian(GP(k_2d_pde); u = δ(X_c_2d), lap = ℒ_lap, u_b = δ(X_b_2d))
	posterior(fg, (; lap = f_2d.(X_c_2d), u_b = zeros(length(X_b_2d)));
		noise = (; lap = 1e-8, u_b = 1e-8))
end end

# ╔═╡ ee000029-0001-4000-8000-000000000029
if ismissing(post_2d)
	still_missing("the 2D Poisson exercise")
else let
	μ_u = dist_mean(post_2d.u)
	σ_u = sqrt.(dist_var(post_2d.u))
	err = maximum(abs.(μ_u .- u_2d_exact.(X_c_2d)))
	n = length(xs_2d)
	M(v) = reshape(v, n, n)
	fig = Figure(size = (1000, 330))
	ax1 = Axis(fig[1, 1], title = "posterior mean u", aspect = DataAspect())
	hm = heatmap!(ax1, xs_2d, ys_2d, M(μ_u), colormap = :viridis)
	Colorbar(fig[1, 2], hm)
	ax2 = Axis(fig[1, 3], title = "posterior std", aspect = DataAspect())
	hm2 = heatmap!(ax2, xs_2d, ys_2d, M(σ_u), colormap = :magma)
	Colorbar(fig[1, 4], hm2)
	ax3 = Axis(fig[1, 5], title = "|mean − exact|  (max $(round(err; sigdigits = 2)))",
		aspect = DataAspect())
	hm3 = heatmap!(ax3, xs_2d, ys_2d, M(abs.(μ_u .- u_2d_exact.(X_c_2d))), colormap = :magma)
	Colorbar(fig[1, 6], hm3)
	Label(fig[0, :], err < 0.05 ? "✓ solved: max error $(round(err; sigdigits = 2))" :
		"max error $(round(err; sigdigits = 2)) — is ℒ_lap really the Laplacian?",
		fontsize = 16)
	fig
end end

# ╔═╡ 20000002-a002-4000-8000-000000000002
begin
	# Geometry helpers (point-in-polygon, distance-to-coast, arc-resampling)
	function _inside_poly(p, poly)
		x, y = p; n = length(poly); inside = false; j = n
		@inbounds for i in 1:n
			xi, yi = poly[i]; xj, yj = poly[j]
			if ((yi > y) != (yj > y))
				xint = (xj - xi) * (y - yi) / (yj - yi) + xi
				x < xint && (inside = !inside)
			end
			j = i
		end
		inside
	end
	_is_water(p) = _inside_poly(p, WATER_POLY_km) && !_inside_poly(p, ISLAND_km)

	function _dist_to_seg(p, a, b)
		px, py = p; ax, ay = a; bx, by = b
		dx, dy = bx - ax, by - ay; len2 = dx*dx + dy*dy
		len2 == 0 && return hypot(px - ax, py - ay)
		t = clamp(((px - ax)*dx + (py - ay)*dy) / len2, 0.0, 1.0)
		hypot(px - ax - t*dx, py - ay - t*dy)
	end
	_dist_to_coast(p) = minimum(_dist_to_seg(p, COAST_km[i], COAST_km[i+1])
		for i in 1:(length(COAST_km) - 1))

	function _arc_resample(pts, target_spacing)
		out = [pts[1]]; leftover = 0.0
		for i in 1:(length(pts) - 1)
			a, b = pts[i], pts[i+1]
			dx, dy = b[1] - a[1], b[2] - a[2]; seg_len = hypot(dx, dy)
			seg_len == 0 && continue
			t = (target_spacing - leftover) / seg_len
			while t <= 1.0
				push!(out, (a[1] + t*dx, a[2] + t*dy))
				t += target_spacing / seg_len
			end
			leftover = (1.0 - (t - target_spacing/seg_len)) * seg_len
		end
		out
	end; nothing
end

# ╔═╡ 3148f4f4-99c7-4289-9db0-6c264b9ad815
begin 
import Base64
    
function _schem(emph::Set; mode = :domain)
  on(k) = k in emph
  ACC=RGBf(234/255,75/255,46/255); WATER=RGBf(0.62,0.79,0.90); LAND=RGBf(0.88,0.85,0.80)
  COASTC=RGBf(0.28,0.26,0.24); MUTE=RGBf(0.60,0.65,0.70); GOLD=RGBf(210/255,150/255,0); DB=RGBf(26/255,58/255,91/255)
  bb=BBOX_km
  WPv=vcat([Point2f(p[1],p[2]) for p in COAST_km],[Point2f(COAST_km[end][1],bb.ymin),Point2f(COAST_km[1][1],bb.ymin)])
  fig=Figure(size=(300,250), backgroundcolor=:transparent)
  ax=Axis(fig[1,1], aspect=DataAspect(), backgroundcolor=:transparent); hidedecorations!(ax); hidespines!(ax)
  xlims!(ax,bb.xmin-0.05,bb.xmax+0.05); ylims!(ax,bb.ymin-0.05,bb.ymax+0.05)
  poly!(ax, Point2f.([(bb.xmin-0.3,bb.ymin-0.3),(bb.xmax+0.3,bb.ymin-0.3),(bb.xmax+0.3,bb.ymax+0.3),(bb.xmin-0.3,bb.ymax+0.3)]), color=LAND)
  poly!(ax, WPv, color=WATER)
  length(ISLAND_km)>3 && poly!(ax, [Point2f(p[1],p[2]) for p in ISLAND_km], color=LAND)
  (on(:field)||on(:diffusion)) && scatter!(ax,[SOURCE_km[1]],[SOURCE_km[2]];marker=:circle,markersize=(on(:diffusion) ? 0.55 : 0.38)*250,color=(ACC,0.20))
  lines!(ax,[Point2f(p[1],p[2]) for p in COAST_km];color=on(:boundary) ? ACC : COASTC,linewidth=on(:boundary) ? 4 : 1.8)
  wpts=[(x,y) for x in range(bb.xmin,bb.xmax;length=7) for y in range(bb.ymin,bb.ymax;length=7) if _is_water((x,y))]
  apts=wpts[1:max(1,length(wpts)÷4):end]; nv=hypot(v_current...); L=on(:advection) ? 0.42 : 0.3
  arrows2d!(ax,[p[1] for p in apts],[p[2] for p in apts],fill(v_current[1]/nv*L,length(apts)),fill(v_current[2]/nv*L,length(apts));
      color=on(:advection) ? ACC : MUTE)
  if mode==:transect                                    # vertical transect through the source
      xs0=SOURCE_km[1]; yl=[y for y in range(bb.ymin,bb.ymax;length=160) if _is_water((xs0,y))]
      !isempty(yl) && lines!(ax,[xs0,xs0],[minimum(yl),maximum(yl)];color=DB,linewidth=2.5,linestyle=:dash)
  end
  scatter!(ax,[p[1] for p in SENSORS_km],[p[2] for p in SENSORS_km];marker=:circle,
      markersize=on(:sensors) ? 16 : 10,color=on(:sensors) ? ACC : :white,strokewidth=1.5,strokecolor=on(:sensors) ? :black : COASTC)
  scatter!(ax,[SOURCE_km[1]],[SOURCE_km[2]];marker=:star5,markersize=on(:source) ? 30 : 18,color=on(:source) ? ACC : GOLD,strokewidth=1,strokecolor=:black)
  fig
end

    
  function you_are_here(active...; done=(), step="", reveal_time=false, mode=:domain)
      A=Set(active); D=Set(done)
      acc="color:var(--tuai-accent,#ea4b2e);font-weight:700;"
      grn="color:var(--tuai-done,#2f9e60);font-weight:600;"
      gry="color:#9aa0a8;"
      lbl="font-family:'Roboto Mono',monospace;font-size:0.6em;letter-spacing:0.09em;text-transform:uppercase;color:var(--tuai-lightblue,#85cbd2);font-weight:700"
      col(k)= k in A ? acc : (k in D ? grn : gry)
      c(k,t)="<span style=\"$(col(k))\">$t</span>"; g(t)="<span style=\"$gry\">$t</span>"
      H(t)="<div style=\"$lbl;margin-top:0.35em;\">$t</div>"
      pde = reveal_time ?
          c(:time,"∂ₜu")*g(" = −")*c(:advection,"v·∇u")*g(" + ")*c(:diffusion,"κΔu")*g(" − ")*c(:reaction,"R(u)") :
          c(:advection,"v·∇u")*g(" − ")*c(:diffusion,"κΔu")*g(" + ")*c(:reaction,"R(u)")*g(" = ")*c(:source,"f(θ)")
    tmp=tempname()*".png"; save(tmp,_schem(A;mode=mode); px_per_unit=2); uri="data:image/png;base64,"*Base64.base64encode(read(tmp)); rm(tmp;force=true)
      steprow = isempty(step) ? "" : """<h2 style="font-family:'Plus Jakarta 
  Sans',sans-serif;font-size:1.12em;font-weight:700;color:var(--tuai-darkblue,#1a3a5b);background:none;padding:0;border-radius:0;margin:0 0 0.5em 
  0;letter-spacing:normal;">$step</h2>"""
      HTML("""
      <div style="font-family:'Plus Jakarta Sans',sans-serif;background:#fff;border:1px solid var(--tuai-lightblue,#85cbd2);
                  border-radius:8px;padding:0.7em 0.95em;margin:0.2em 0 0.7em 0;">
        $steprow
        <div style="display:flex;align-items:center;gap:1.1em;flex-wrap:wrap;">
          <img src="$uri" style="height:240px;width:auto;flex:0 0 auto;" alt="harbor"/>
          <div style="flex:1 1 0;min-width:0;display:flex;flex-direction:column;font-size:0.9em;line-height:1.3;">
            $(H("Goal"))<div>$(g("infer")) $(c(:theta,"θ")) $(g("and")) $(c(:field,"u"))</div>
            $(H("Sub problems"))<div>$(c(:forward,"Forward"))$(g(": simulate u for fixed θ"))</div><div>$(c(:inverse,"Inverse"))$(g(": infer θ"))</div>
            $(H("Mechanistic knowledge"))<div>$(g("•")) $pde</div><div>$(g("•")) $(c(:boundary,"u = 0 on the coast"))</div>
            $(H("Measurements"))<div>$(g("•")) $(c(:sensors,"y = ℓ(u) + ε"))</div>
          </div>
        </div>
      </div>""")
  end

  const MODULES = [
      (parts=(),                               step="The leak problem",                         mode=:domain,   rt=false),
      (parts=(:field,),                        step="Step 1 · Define a prior over u",           mode=:transect, rt=false),
      (parts=(:sensors,:boundary),             step="Step 2 · Condition on sensors + boundary", mode=:transect, rt=false),
      (parts=(:advection,:diffusion), step="Step 3 · Advection–diffusion physics",     mode=:transect, rt=false),
      (parts=(),                               step="Step 4 · Scale up to the 2D harbor",        mode=:domain,   rt=false),
      (parts=(:reaction,),                     step="Step 6 · Add the nonlinear reaction",       mode=:domain,   rt=false),
      (parts=(:theta,:source,:inverse),        step="Step 5 · Invert — infer the source θ",      mode=:domain,   rt=false),
      (parts=(:time,),                         step="Step 7 · Add time dependence",              mode=:domain,   rt=true),
  ]
  function step_banner(n)               # n = module number 0..7
      M = MODULES[n+1]
      done = reduce(union, (Set(MODULES[j+1].parts) for j in 0:n-1); init=Set{Symbol}())
      you_are_here(M.parts...; done=done, step=M.step, reveal_time=M.rt, mode=M.mode)
  end

end;

# ╔═╡ c9f25088-ce87-4618-8b20-8c5a7c83d96e
step_banner(1)

# ╔═╡ 7349a0fe-5c6a-488d-8a7b-d652452f624f
step_banner(2)

# ╔═╡ c0000006-b001-4000-8000-000000000001
step_banner(3)

# ╔═╡ 532ae585-083e-4214-a480-bc8e5999af86
step_banner(4)

# ╔═╡ 4c4d5442-8f9c-4ce0-9d65-7fc836971625
step_banner(6)

# ╔═╡ 89d16755-a925-4e33-822d-92a3711f8c52
step_banner(5)

# ╔═╡ 6d3a7f9e-345b-4076-9066-869afb6dab42
step_banner(7)

# ╔═╡ 20000002-a003-4000-8000-000000000003
begin
	# Sample collocation (regular grid clipped to water, ≥60m from coast),
	# boundary (arc-resampled coast + open-sea south edge), and sensor points.
	function _sample_collocation(nx, ny; min_coast_dist_km = 0.06)
		xs = range(BBOX_km.xmin, BBOX_km.xmax; length = nx)
		ys = range(BBOX_km.ymin, BBOX_km.ymax; length = ny)
		out = Tuple{Float64, Float64}[]
		for x in xs, y in ys
			p = (x, y)
			_is_water(p) && _dist_to_coast(p) > min_coast_dist_km && push!(out, p)
		end
		out
	end
	function _sample_boundary(; coast_spacing = 0.12, sea_spacing = 0.20)
		coast_pts = _arc_resample(collect(COAST_km), coast_spacing)
		x_east, x_west = COAST_km[end][1], COAST_km[1][1]
		n_sea = max(2, ceil(Int, abs(x_east - x_west) / sea_spacing))
		sea_pts = [(x_east + (x_west - x_east)*(i - 1)/(n_sea - 1), BBOX_km.ymin)
			for i in 1:n_sea]
		vcat(coast_pts, sea_pts)
	end
	X_c_raw = _sample_collocation(26, 25)
	X_b_raw = _sample_boundary()
	_to_vv(pts) = [Float64[p[1], p[2]] for p in pts]
	X_c      = _to_vv(X_c_raw)
	X_b      = _to_vv(X_b_raw)
	X_sensor = _to_vv(SENSORS_km); nothing
end

# ╔═╡ 20000002-a004-4000-8000-000000000004
begin
	# Kernel lengthscale for the harbor prior.
	ℓ_h = 0.25  # km, kernel lengthscale

	# Package the sparse harbor prior as a reusable LatentModel.
	harbor = HarborField(X_c, X_b, X_sensor; velocity = v_current, ρ = 0.7)
	md"`HarborField` LatentModel over the harbor geometry → $(length(harbor))-dim sparse prior."
end

# ╔═╡ 20000002-a005-4000-8000-000000000005
begin
	# Synthetic sensor data: classical FD forward solve at known κ_true / known source.
	Random.seed!(2026)
	κ_TRUE     = 0.05      # km², diffusion coefficient
	SRC_AMPL_h = 1.0
	SRC_WIDTH_h = 0.15     # km, source extent
	σ_data_TRUE = 0.02
	σ_phys_h    = 1.0e-3
	σ_bc_h      = 1.0e-3

	# Gaussian plume: emission at point p from a source at (xs, ys).
	source_emission(p, xs, ys) = SRC_AMPL_h *
		exp(-((p[1] - xs)^2 + (p[2] - ys)^2) / (2 * SRC_WIDTH_h^2))

	y_phys  = zeros(length(X_c))
	y_bc    = zeros(length(X_b))

	# Honest ground truth: an INDEPENDENT classical finite-difference solve of the LINEAR PDE
	#   v·∇u - κΔu + u = f   on a fine grid (different discretisation ⇒ no inverse crime — the
	#   direct analogue of the nonlinear FD-Newton solver, but linear ⇒ one linear solve).
	function _fd_linear_solve(; nx = 130, ny = 130)
		xs = range(BBOX_km.xmin, BBOX_km.xmax; length = nx)
		ys = range(BBOX_km.ymin, BBOX_km.ymax; length = ny)
		hx = step(xs); hy = step(ys)
		water = [_is_water((xs[i], ys[j])) for i in 1:nx, j in 1:ny]
		N = count(water); idx = zeros(Int, nx, ny); idx[water] .= 1:N
		Ii = Int[]; Jj = Int[]; Vv = Float64[]; b = zeros(N)
		pc!(r, i, j, c) = (water[i, j] && (push!(Ii, r); push!(Jj, idx[i, j]); push!(Vv, c)))
		for i in 1:nx, j in 1:ny
			water[i, j] || continue
			r = idx[i, j]; p = (xs[i], ys[j])
			pc!(r, i, j, κ_TRUE*(2/hx^2 + 2/hy^2) + 1.0)          # -κΔ diagonal + linear "+u" reaction
			i>1  && pc!(r, i-1, j, -κ_TRUE/hx^2); i<nx && pc!(r, i+1, j, -κ_TRUE/hx^2)
			j>1  && pc!(r, i, j-1, -κ_TRUE/hy^2); j<ny && pc!(r, i, j+1, -κ_TRUE/hy^2)
			i>1  && pc!(r, i-1, j, -v_current[1]/(2hx)); i<nx && pc!(r, i+1, j, v_current[1]/(2hx))
			j>1  && pc!(r, i, j-1, -v_current[2]/(2hy)); j<ny && pc!(r, i, j+1, v_current[2]/(2hy))
			b[r] = source_emission(p, SOURCE_km[1], SOURCE_km[2])
		end
		u = sparse(Ii, Jj, Vv, N, N) \ b                          # one linear solve — no Newton
		return function (p)
			i = clamp(round(Int, (p[1]-BBOX_km.xmin)/hx) + 1, 1, nx)
			j = clamp(round(Int, (p[2]-BBOX_km.ymin)/hy) + 1, 1, ny)
			water[i, j] ? u[idx[i, j]] : 0.0
		end
	end
	_samp_lin = _fd_linear_solve()
	u_at_sensors_true = [_samp_lin((p[1], p[2])) for p in X_sensor]

	y_sensor = u_at_sensors_true .+ σ_data_TRUE .* randn(length(u_at_sensors_true))
	y_all = vcat(y_phys, y_bc, y_sensor)
	md"""
	Forward solve done.
	"""
end

# ╔═╡ 20000002-a006-4000-8000-000000000006
@latte function harbor_source_model(y_phys, y_bc, y_sensor, X_c, field)
	# Unknowns: source location + sensor noise scale.
	x_src  ~ Uniform(-0.5, 1.3)
	y_src  ~ Uniform( 0.0, 1.5)

	# Latent concentration field, with the sparse HarborField prior
	@random x ~ field(; lengthscale = 0.25)   # = ℓ_h, the data-generating scale
	blocks = nameview(field, x)               # named views: u, adv, Δu, u_bc, u_sensor

	# Enforce the PDE  v·∇u - κ∇²u + u = f  by driving its residual to 0 at every
	# collocation point (y_phys is all zeros).
	for i in eachindex(X_c)
		f_i = source_emission(X_c[i], x_src, y_src)
		r_i = missing
		y_phys[i] ~ Normal(r_i, σ_phys_h)
	end
	# Boundary conditions (u = 0) and sensor observations.
	for i in eachindex(y_bc);     y_bc[i]     ~ Normal(blocks.u_bc[i],     σ_bc_h);  end
	for i in eachindex(y_sensor); y_sensor[i] ~ Normal(blocks.u_sensor[i], σ_data_TRUE);  end
end

# ╔═╡ ee000035-0001-4000-8000-000000000035
harbor_model = try
	harbor_source_model(y_phys, y_bc, y_sensor, X_c, harbor)
catch err
	missing
end

# ╔═╡ ee000036-0001-4000-8000-000000000036
if ismissing(harbor_model)
	keep_working(md"The model does not build yet — fill in the residual `r_i` in the `@latte` block above (Exercise 4).")
else
	correct(md"The model builds. Below, INLA infers the source from the six sensor readings.")
end

# ╔═╡ 20000004-a001-4000-8000-000000000001
# ╠═╡ show_logs = false
# Live INLA — about 10-20 s on first run (mostly compilation); Pluto caches the result.
# Several random starts for the mode search: the source posterior is multimodal, and a
# single start occasionally settles in a spurious mode on the far side of the bay.
res_harbor = ismissing(harbor_model) ? missing : inla(
	harbor_model, y_all;
	latent_marginalization_method = GaussianMarginal(),
	mode_init = RandomStarts(6),
	executor = ThreadedExecutor(),
	progress = false,
	accumulators = (),
)

# ╔═╡ 20000004-a002-4000-8000-000000000002
if ismissing(res_harbor)
	still_missing("Exercise 4")
else let
	κ_marg = res_harbor.hyperparameter_marginals
	mean_x = dist_mean(κ_marg[:x_src])
	q05_x, q95_x = quantile(κ_marg[:x_src], 0.05), quantile(κ_marg[:x_src], 0.95)
	mean_y = dist_mean(κ_marg[:y_src])
	q05_y, q95_y = quantile(κ_marg[:y_src], 0.05), quantile(κ_marg[:y_src], 0.95)
	md"""
	**Posterior summary**

	|         | mean   | 90 % CI                 | truth (km) |
	|---------|--------|-------------------------|------------|
	| x_src   | $(round(mean_x; digits = 3)) | [$(round(q05_x; digits = 3)), $(round(q95_x; digits = 3))] | $(round(SOURCE_km[1]; digits = 3)) |
	| y_src   | $(round(mean_y; digits = 3)) | [$(round(q05_y; digits = 3)), $(round(q95_y; digits = 3))] | $(round(SOURCE_km[2]; digits = 3)) |
	"""
end end

# ╔═╡ 30000006-0006-4000-8000-000000000006
@latte function harbor_monod_model(y_phys, y_bc, y_sensor, X_c, g)
	x_src ~ Uniform(-0.5, 1.5)
	y_src ~ Uniform( 0.0, 1.7)
	@random x ~ g                       # fixed sparse prior (output scale already set)
	blocks = nameview(g, x)
	for i in eachindex(X_c)
		f_i = source_emission(X_c[i], x_src, y_src)
		# NONLINEAR residual: the Monod reaction replaces the linear `+u` term.
		r_i = blocks.adv[i] - κ_TRUE * blocks.Δu[i] +
			Vmax_monod * blocks.u[i] / (KM_monod + blocks.u[i]) - f_i
		y_phys[i] ~ Normal(r_i, σ_phys_h; check_args = false)
	end
	for i in eachindex(y_bc);     y_bc[i]     ~ Normal(blocks.u_bc[i],     σ_bc_h;      check_args = false);  end
	for i in eachindex(y_sensor); y_sensor[i] ~ Normal(blocks.u_sensor[i], σ_data_TRUE; check_args = false);  end
end

# ╔═╡ 20000004-a003-4000-8000-000000000003
if ismissing(res_harbor)
	still_missing("Exercise 4")
else let
	κ_marg = res_harbor.hyperparameter_marginals
	q05_x, q95_x = quantile(κ_marg[:x_src], 0.05), quantile(κ_marg[:x_src], 0.95)
	q05_y, q95_y = quantile(κ_marg[:y_src], 0.05), quantile(κ_marg[:y_src], 0.95)

	WATER = RGBf(0.62, 0.79, 0.90)
	LAND  = RGBf(0.88, 0.85, 0.80)
	COAST = RGBf(0.28, 0.26, 0.24)

	fig = Figure(size = (1100, 600))

	# Map panel: harbor + 90% credible box
	ax_map = Axis(fig[1, 1]; aspect = DataAspect(), backgroundcolor = WATER,
		title = "Harbor (km) — true source + 90% credible box")
	hidedecorations!(ax_map); hidespines!(ax_map)
	xlims!(ax_map, BBOX_km.xmin - 0.05, BBOX_km.xmax + 0.05)
	ylims!(ax_map, BBOX_km.ymin - 0.05, BBOX_km.ymax + 0.05)
	poly!(ax_map, Rect2f(BBOX_km.xmin - 0.1, BBOX_km.ymin - 0.1,
		(BBOX_km.xmax - BBOX_km.xmin) + 0.2,
		(BBOX_km.ymax - BBOX_km.ymin) + 0.2); color = LAND)
	water_poly = vcat(
		[Point2f(p[1], p[2]) for p in COAST_km],
		[Point2f(COAST_km[end][1], BBOX_km.ymin),
		 Point2f(COAST_km[1][1], BBOX_km.ymin)])
	poly!(ax_map, water_poly; color = WATER)
	lines!(ax_map, [Point2f(p[1], p[2]) for p in COAST_km];
		color = COAST, linewidth = 2.0)
	if length(ISLAND_km) >= 3
		poly!(ax_map, [Point2f(p[1], p[2]) for p in ISLAND_km];
			color = LAND, strokecolor = COAST, strokewidth = 1.2)
	end
	scatter!(ax_map, [p[1] for p in SENSORS_km], [p[2] for p in SENSORS_km];
		marker = :circle, markersize = 12, color = :white,
		strokecolor = TUE_AI_COLORS.dark, strokewidth = 2.0,
		label = "sensors")
	# 90% credible box
	poly!(ax_map, Rect2f(q05_x, q05_y, q95_x - q05_x, q95_y - q05_y);
		color = (COLOR_QoI, 0.25),
		strokecolor = COLOR_QoI, strokewidth = 2.5,
		label = "90% CI")
	scatter!(ax_map, [SOURCE_km[1]], [SOURCE_km[2]];
		marker = :star5, markersize = 22, color = COLOR_PROBLEM,
		strokecolor = :black, strokewidth = 1, label = "true source")
	axislegend(ax_map; position = :lb)

	# 1D marginals
	ax_x = Axis(fig[1, 2]; title = "x_src (km)", ylabel = "p")
	xs = range(quantile(κ_marg[:x_src], 0.001),
		quantile(κ_marg[:x_src], 0.999); length = 200)
	lines!(ax_x, xs, pdf.(Ref(κ_marg[:x_src]), xs); linewidth = 2,
		color = COLOR_QoI)
	vlines!(ax_x, [SOURCE_km[1]]; label = "truth", PNMETHODS_PROBLEM_LINES_KWARGS...)
	axislegend(ax_x)

	ax_y = Axis(fig[2, 2]; title = "y_src (km)", xlabel = "km", ylabel = "p")
	ys = range(quantile(κ_marg[:y_src], 0.001),
		quantile(κ_marg[:y_src], 0.999); length = 200)
	lines!(ax_y, ys, pdf.(Ref(κ_marg[:y_src]), ys); linewidth = 2,
		color = COLOR_QoI)
	vlines!(ax_y, [SOURCE_km[2]]; label = "truth", PNMETHODS_PROBLEM_LINES_KWARGS...)
	axislegend(ax_y)

	colsize!(fig.layout, 1, Relative(0.55))
	fig
end end

# ╔═╡ ae8d17be-a783-47e5-995d-d408515e8cb3
if ismissing(res_harbor)
	still_missing("Exercise 4")
else let
	# Posterior latent concentration field u(x, y) on the harbor.
	# Pulls the `u` block of the latent vector via the named-block metadata.

	u_range = harbor.ranges.u
	u_marg  = res_harbor.latent_marginals[u_range]
	u_mean  = dist_mean.(u_marg)
	u_std   = sqrt.(dist_var.(u_marg))

	WATER = RGBf(0.62, 0.79, 0.90)
	LAND  = RGBf(0.88, 0.85, 0.80)
	COAST = RGBf(0.28, 0.26, 0.24)

	fig = Figure(size = (1200, 600))

	function draw_harbor!(ax)
			hidedecorations!(ax); hidespines!(ax)
			xlims!(ax, BBOX_km.xmin - 0.05, BBOX_km.xmax + 0.05)
			ylims!(ax, BBOX_km.ymin - 0.05, BBOX_km.ymax + 0.05)
			poly!(ax, Rect2f(BBOX_km.xmin - 0.1, BBOX_km.ymin - 0.1,
					(BBOX_km.xmax - BBOX_km.xmin) + 0.2,
					(BBOX_km.ymax - BBOX_km.ymin) + 0.2); color = LAND)
			water_poly = vcat(
					[Point2f(p[1], p[2]) for p in COAST_km],
					[Point2f(COAST_km[end][1], BBOX_km.ymin),
					 Point2f(COAST_km[1][1], BBOX_km.ymin)])
			poly!(ax, water_poly; color = WATER)
			lines!(ax, [Point2f(p[1], p[2]) for p in COAST_km];
					color = COAST, linewidth = 2.0)
			length(ISLAND_km) >= 3 && poly!(ax,
					[Point2f(p[1], p[2]) for p in ISLAND_km];
					color = LAND, strokecolor = COAST, strokewidth = 1.2)
	end

	xs_pts = [p[1] for p in X_c]
	ys_pts = [p[2] for p in X_c]

	# --- Left: posterior mean concentration ---
	ax1 = Axis(fig[1, 1]; aspect = DataAspect(),
			title = "Posterior mean concentration u(x, y)",
			backgroundcolor = WATER)
	draw_harbor!(ax1)
	scat1 = scatter!(ax1, xs_pts, ys_pts;
			color = u_mean, colormap = :viridis, markersize = 14, marker = :rect)
	Colorbar(fig[1, 2], scat1; label = "u")
	scatter!(ax1, [p[1] for p in SENSORS_km], [p[2] for p in SENSORS_km];
			marker = :circle, markersize = 12, color = :white,
			strokecolor = TUE_AI_COLORS.dark, strokewidth = 2.0)
	scatter!(ax1, [SOURCE_km[1]], [SOURCE_km[2]];
			marker = :star5, markersize = 22, color = COLOR_PROBLEM,
			strokecolor = :black, strokewidth = 1.5)

	# --- Right: posterior std (uncertainty) ---
	ax2 = Axis(fig[1, 3]; aspect = DataAspect(),
			title = "Posterior std σ_u(x, y) — uncertainty",
			backgroundcolor = WATER)
	draw_harbor!(ax2)
	scat2 = scatter!(ax2, xs_pts, ys_pts;
			color = u_std, colormap = :viridis, markersize = 14, marker = :rect)
	Colorbar(fig[1, 4], scat2; label = "σ_u")
	scatter!(ax2, [p[1] for p in SENSORS_km], [p[2] for p in SENSORS_km];
			marker = :circle, markersize = 12, color = :white,
			strokecolor = TUE_AI_COLORS.dark, strokewidth = 2.0)
	scatter!(ax2, [SOURCE_km[1]], [SOURCE_km[2]];
			marker = :star5, markersize = 22, color = COLOR_PROBLEM,
			strokecolor = :black, strokewidth = 1.5)

	fig
end end

# ╔═╡ 20000004-b002-4000-8000-000000000002
begin
	X_sensor_d_raw = _sample_collocation(6, 6; min_coast_dist_km = 0.10)
	X_sensor_d = _to_vv(X_sensor_d_raw)
	md"Dense-sensor mode: **$(length(X_sensor_d))** sensors."
end

# ╔═╡ 20000004-b003-4000-8000-000000000003
begin
	harbor_d = HarborField(X_c, X_b, X_sensor_d; velocity = v_current, ρ = 0.7)
	md"Dense-sensor `HarborField` → $(length(harbor_d))-dim sparse prior."
end

# ╔═╡ 20000004-b004-4000-8000-000000000004
begin
	Random.seed!(2026)
	# Same honest FD ground truth, sampled at the dense sensor set (reuses `_samp_lin`).
	u_at_sensors_d = [_samp_lin((p[1], p[2])) for p in X_sensor_d]
	y_sensor_d = u_at_sensors_d .+ σ_data_TRUE .* randn(length(u_at_sensors_d))
	y_all_d = vcat(y_phys, y_bc, y_sensor_d)
	md"Generated **$(length(y_sensor_d))** synthetic sensor readings."
end

# ╔═╡ 20000004-b005-4000-8000-000000000005
# ╠═╡ show_logs = false
res_harbor_d = ismissing(harbor_model) ? missing : inla(
	harbor_source_model(y_phys, y_bc, y_sensor_d, X_c, harbor_d), y_all_d;
	latent_marginalization_method = GaussianMarginal(),
	mode_init = RandomStarts(6),
	executor = ThreadedExecutor(),
	progress = false,
	accumulators = (),
)

# ╔═╡ df109776-1162-4cb4-a771-188332172be1
if ismissing(res_harbor_d)
	still_missing("Exercise 4")
else let
	hpd = res_harbor_d.hyperparameter_marginals
	xs_g = range(quantile(hpd[:x_src], 0.001),
			quantile(hpd[:x_src], 0.999); length = 200)
	ys_g = range(quantile(hpd[:y_src], 0.001),
			quantile(hpd[:y_src], 0.999); length = 200)

	fig = Figure(size = (900, 350))

	ax_x = Axis(fig[1, 1]; title = "x_src — $(length(X_sensor_d)) sensors",
			xlabel = "km", ylabel = "p")
	lines!(ax_x, xs_g, pdf.(Ref(hpd[:x_src]), xs_g);
			linewidth = 2, color = COLOR_QoI)
	vlines!(ax_x, [SOURCE_km[1]]; PNMETHODS_PROBLEM_LINES_KWARGS...)

	ax_y = Axis(fig[1, 2]; title = "y_src — $(length(X_sensor_d)) sensors",
			xlabel = "km", ylabel = "p")
	lines!(ax_y, ys_g, pdf.(Ref(hpd[:y_src]), ys_g);
			linewidth = 2, color = COLOR_QoI)
	vlines!(ax_y, [SOURCE_km[2]]; PNMETHODS_PROBLEM_LINES_KWARGS...)

	fig
end end

# ╔═╡ 20000004-b006-4000-8000-000000000006
if ismissing(res_harbor_d)
	still_missing("Exercise 4")
else let
	hp6 = res_harbor.hyperparameter_marginals
	hpd = res_harbor_d.hyperparameter_marginals
	q05_6x, q95_6x = quantile(hp6[:x_src], 0.05), quantile(hp6[:x_src], 0.95)
	q05_6y, q95_6y = quantile(hp6[:y_src], 0.05), quantile(hp6[:y_src], 0.95)
	q05_dx, q95_dx = quantile(hpd[:x_src], 0.05), quantile(hpd[:x_src], 0.95)
	q05_dy, q95_dy = quantile(hpd[:y_src], 0.05), quantile(hpd[:y_src], 0.95)

	WATER = RGBf(0.62, 0.79, 0.90)
	LAND  = RGBf(0.88, 0.85, 0.80)
	COAST = RGBf(0.28, 0.26, 0.24)

	fig = Figure(size = (1100, 700))
	ax  = Axis(fig[1, 1]; aspect = DataAspect(),
		title = "90% credible boxes: $(length(X_sensor)) vs $(length(X_sensor_d)) sensors",
		backgroundcolor = WATER)
	hidedecorations!(ax); hidespines!(ax)
	xlims!(ax, BBOX_km.xmin - 0.05, BBOX_km.xmax + 0.05)
	ylims!(ax, BBOX_km.ymin - 0.05, BBOX_km.ymax + 0.05)
	poly!(ax, Rect2f(BBOX_km.xmin - 0.1, BBOX_km.ymin - 0.1,
		(BBOX_km.xmax - BBOX_km.xmin) + 0.2,
		(BBOX_km.ymax - BBOX_km.ymin) + 0.2); color = LAND)
	water_poly = vcat(
		[Point2f(p[1], p[2]) for p in COAST_km],
		[Point2f(COAST_km[end][1], BBOX_km.ymin),
		 Point2f(COAST_km[1][1], BBOX_km.ymin)])
	poly!(ax, water_poly; color = WATER)
	lines!(ax, [Point2f(p[1], p[2]) for p in COAST_km];
		color = COAST, linewidth = 2.0)
	length(ISLAND_km) >= 3 && poly!(ax,
		[Point2f(p[1], p[2]) for p in ISLAND_km];
		color = LAND, strokecolor = COAST, strokewidth = 1.2)

	poly!(ax, Rect2f(q05_6x, q05_6y, q95_6x - q05_6x, q95_6y - q05_6y);
		color = (TU_COLORS.violet, 0.15),
		strokecolor = TU_COLORS.violet, strokewidth = 2.5,
		label = "$(length(X_sensor)) sensors (90% CI)")
	poly!(ax, Rect2f(q05_dx, q05_dy, q95_dx - q05_dx, q95_dy - q05_dy);
		color = (COLOR_QoI, 0.15),
		strokecolor = COLOR_QoI, strokewidth = 2.5,
		label = "$(length(X_sensor_d)) sensors (90% CI)")

	scatter!(ax, [p[1] for p in SENSORS_km], [p[2] for p in SENSORS_km];
		marker = :circle, markersize = 12, color = :white,
		strokecolor = TU_COLORS.violet, strokewidth = 2.0,
		label = "original sensors")
	scatter!(ax, [p[1] for p in X_sensor_d_raw], [p[2] for p in X_sensor_d_raw];
		marker = :utriangle, markersize = 10, color = :white,
		strokecolor = COLOR_QoI, strokewidth = 2.0,
		label = "dense sensors")
	scatter!(ax, [SOURCE_km[1]], [SOURCE_km[2]];
		marker = :star5, markersize = 22, color = COLOR_PROBLEM,
		strokecolor = :black, strokewidth = 1, label = "true source")

	axislegend(ax; position = :lb)
	fig
end end

# ╔═╡ ee000040-0001-4000-8000-000000000040
if ismissing(res_harbor)
	still_missing("Exercise 4")
else let
	hp = res_harbor.hyperparameter_marginals
	q05_x, q95_x = quantile(hp[:x_src], 0.05), quantile(hp[:x_src], 0.95)
	q05_y, q95_y = quantile(hp[:y_src], 0.05), quantile(hp[:y_src], 0.95)
	WATER = RGBf(0.62, 0.79, 0.90); LAND = RGBf(0.88, 0.85, 0.80); COAST = RGBf(0.28, 0.26, 0.24)
	fig = Figure(size = (900, 650))
	ax = Axis(fig[1, 1]; aspect = DataAspect(), backgroundcolor = WATER,
		xlabel = "x (km)", ylabel = "y (km)", title = "Read off coordinates here",
		xticks = -1:0.2:2, yticks = -1:0.2:2, xminorgridvisible = true, yminorgridvisible = true)
	xlims!(ax, BBOX_km.xmin - 0.05, BBOX_km.xmax + 0.05)
	ylims!(ax, BBOX_km.ymin - 0.05, BBOX_km.ymax + 0.05)
	poly!(ax, Rect2f(BBOX_km.xmin - 0.1, BBOX_km.ymin - 0.1,
		(BBOX_km.xmax - BBOX_km.xmin) + 0.2, (BBOX_km.ymax - BBOX_km.ymin) + 0.2); color = LAND)
	water_poly = vcat([Point2f(p[1], p[2]) for p in COAST_km],
		[Point2f(COAST_km[end][1], BBOX_km.ymin), Point2f(COAST_km[1][1], BBOX_km.ymin)])
	poly!(ax, water_poly; color = WATER)
	lines!(ax, [Point2f(p[1], p[2]) for p in COAST_km]; color = COAST, linewidth = 2)
	length(ISLAND_km) >= 3 && poly!(ax, [Point2f(p[1], p[2]) for p in ISLAND_km];
		color = LAND, strokecolor = COAST, strokewidth = 1.2)
	poly!(ax, Rect2f(q05_x, q05_y, q95_x - q05_x, q95_y - q05_y);
		color = (TU_COLORS.violet, 0.15), strokecolor = TU_COLORS.violet,
		strokewidth = 2.5, label = "6 sensors (90% CI)")
	scatter!(ax, [p[1] for p in SENSORS_km], [p[2] for p in SENSORS_km];
		marker = :circle, markersize = 12, color = :white,
		strokecolor = TU_COLORS.violet, strokewidth = 2, label = "sensors")
	cur = v_current ./ hypot(v_current...)
	arrows2d!(ax, [BBOX_km.xmax - 0.35], [BBOX_km.ymax - 0.15], [0.25 * cur[1]], [0.25 * cur[2]];
		color = RGBAf(0.20, 0.30, 0.50, 0.7))
	text!(ax, BBOX_km.xmax - 0.35, BBOX_km.ymax - 0.08, text = "current", fontsize = 12,
		color = RGBAf(0.20, 0.30, 0.50, 0.9))
	axislegend(ax; position = :lb)
	fig
end end

# ╔═╡ ee000042-0001-4000-8000-000000000042
# ╠═╡ show_logs = false
challenge = if ismissing(my_sensors) || ismissing(res_harbor)
	missing
else let
	pts = [Float64[p[1], p[2]] for p in my_sensors]
	bad = [p for p in my_sensors if !_is_water((Float64(p[1]), Float64(p[2])))]
	if length(pts) != 2
		(; error = "Exactly two sensors, please — you gave $(length(pts)).")
	elseif !isempty(bad)
		(; error = "These positions are on land: $(bad). Pick points in the water.")
	else
		X_sensor_c = vcat(X_sensor, pts)
		harbor_c = HarborField(X_c, X_b, X_sensor_c; velocity = v_current, ρ = 0.7)
		# Readings for the new sensors come from the same finite-difference ground truth
		Random.seed!(7)
		u_new = [_samp_lin((p[1], p[2])) for p in pts]
		y_sensor_c = vcat(y_sensor, u_new .+ σ_data_TRUE .* randn(length(u_new)))
		y_all_c = vcat(y_phys, y_bc, y_sensor_c)
		# A few random starts: the source posterior can be multimodal, and a single
		# start occasionally settles in a spurious mode on the far side of the bay.
		res_c = inla(
			harbor_source_model(y_phys, y_bc, y_sensor_c, X_c, harbor_c), y_all_c;
			latent_marginalization_method = GaussianMarginal(),
			mode_init = RandomStarts(6),
			executor = ThreadedExecutor(),
			progress = false,
			accumulators = (),
		)
		hp = res_c.hyperparameter_marginals
		box = (x0 = quantile(hp[:x_src], 0.05), x1 = quantile(hp[:x_src], 0.95),
			y0 = quantile(hp[:y_src], 0.05), y1 = quantile(hp[:y_src], 0.95))
		(; res = res_c, pts, box, area = (box.x1 - box.x0) * (box.y1 - box.y0))
	end
end end

# ╔═╡ ee000043-0001-4000-8000-000000000043
if ismissing(challenge)
	still_missing("Exercise 5 (`my_sensors`)")
elseif haskey(challenge, :error)
	keep_working(md"$(challenge.error)")
else let
	_area(r) = let hp = r.hyperparameter_marginals
		(quantile(hp[:x_src], 0.95) - quantile(hp[:x_src], 0.05)) *
			(quantile(hp[:y_src], 0.95) - quantile(hp[:y_src], 0.05))
	end
	base = _area(res_harbor)
	dense = ismissing(res_harbor_d) ? NaN : _area(res_harbor_d)
	b = challenge.box
	inside = b.x0 <= SOURCE_km[1] <= b.x1 && b.y0 <= SOURCE_km[2] <= b.y1
	md"""
	**Your score: $(round(challenge.area; digits = 3)) km²** $(inside ? "(box contains the true source ✓)" : "(⚠ the true source is *outside* your box)")

	| | box area (km²) |
	|---|---|
	| 6 sensors (baseline) | $(round(base; digits = 3)) |
	| **6 + your 2** | **$(round(challenge.area; digits = 3))** |
	| $(length(X_sensor_d)) sensors (dense grid, for reference) | $(round(dense; digits = 3)) |
	"""
end end

# ╔═╡ ee000044-0001-4000-8000-000000000044
if ismissing(challenge) || haskey(challenge, :error)
	nothing
else let
	hp6 = res_harbor.hyperparameter_marginals
	q05_6x, q95_6x = quantile(hp6[:x_src], 0.05), quantile(hp6[:x_src], 0.95)
	q05_6y, q95_6y = quantile(hp6[:y_src], 0.05), quantile(hp6[:y_src], 0.95)
	b = challenge.box
	WATER = RGBf(0.62, 0.79, 0.90); LAND = RGBf(0.88, 0.85, 0.80); COAST = RGBf(0.28, 0.26, 0.24)
	fig = Figure(size = (900, 650))
	ax = Axis(fig[1, 1]; aspect = DataAspect(), backgroundcolor = WATER,
		xlabel = "x (km)", ylabel = "y (km)", title = "Your sensors vs. the baseline")
	xlims!(ax, BBOX_km.xmin - 0.05, BBOX_km.xmax + 0.05)
	ylims!(ax, BBOX_km.ymin - 0.05, BBOX_km.ymax + 0.05)
	poly!(ax, Rect2f(BBOX_km.xmin - 0.1, BBOX_km.ymin - 0.1,
		(BBOX_km.xmax - BBOX_km.xmin) + 0.2, (BBOX_km.ymax - BBOX_km.ymin) + 0.2); color = LAND)
	water_poly = vcat([Point2f(p[1], p[2]) for p in COAST_km],
		[Point2f(COAST_km[end][1], BBOX_km.ymin), Point2f(COAST_km[1][1], BBOX_km.ymin)])
	poly!(ax, water_poly; color = WATER)
	lines!(ax, [Point2f(p[1], p[2]) for p in COAST_km]; color = COAST, linewidth = 2)
	length(ISLAND_km) >= 3 && poly!(ax, [Point2f(p[1], p[2]) for p in ISLAND_km];
		color = LAND, strokecolor = COAST, strokewidth = 1.2)
	poly!(ax, Rect2f(q05_6x, q05_6y, q95_6x - q05_6x, q95_6y - q05_6y);
		color = (TU_COLORS.violet, 0.12), strokecolor = TU_COLORS.violet,
		strokewidth = 2.5, label = "6 sensors")
	poly!(ax, Rect2f(b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0);
		color = (COLOR_QoI, 0.18), strokecolor = COLOR_QoI,
		strokewidth = 2.5, label = "6 + yours")
	scatter!(ax, [p[1] for p in SENSORS_km], [p[2] for p in SENSORS_km];
		marker = :circle, markersize = 12, color = :white,
		strokecolor = TU_COLORS.violet, strokewidth = 2, label = "original sensors")
	scatter!(ax, [p[1] for p in challenge.pts], [p[2] for p in challenge.pts];
		marker = :diamond, markersize = 18, color = COLOR_QoI,
		strokecolor = :black, strokewidth = 1, label = "your sensors")
	scatter!(ax, [SOURCE_km[1]], [SOURCE_km[2]];
		marker = :star5, markersize = 22, color = COLOR_PROBLEM,
		strokecolor = :black, strokewidth = 1, label = "true source")
	axislegend(ax; position = :lb)
	fig
end end

# ╔═╡ 30000004-0004-4000-8000-000000000004
begin
	# Honest ground truth: independent FD-Newton solve of the NONLINEAR PDE
	#   v·∇u - κΔu + Vmax·u/(Kₘ+u) = f   (different discretisation → no inverse crime).
	_R_monod(u)  = Vmax_monod * u / (KM_monod + u)
	_dR_monod(u) = Vmax_monod * KM_monod / (KM_monod + u)^2
	function _fd_monod_solve(; nx = 130, ny = 130, tol = 1.0e-11, maxit = 100)
		xs = range(BBOX_km.xmin, BBOX_km.xmax; length = nx)
		ys = range(BBOX_km.ymin, BBOX_km.ymax; length = ny)
		hx = step(xs); hy = step(ys)
		water = [_is_water((xs[i], ys[j])) for i in 1:nx, j in 1:ny]
		N = count(water); idx = zeros(Int, nx, ny); idx[water] .= 1:N
		Ii = Int[]; Jj = Int[]; Vv = Float64[]; b = zeros(N)
		pc!(r, i, j, c) = (water[i, j] && (push!(Ii, r); push!(Jj, idx[i, j]); push!(Vv, c)))
		for i in 1:nx, j in 1:ny
			water[i, j] || continue
			r = idx[i, j]; p = (xs[i], ys[j])
			pc!(r, i, j, κ_TRUE*(2/hx^2 + 2/hy^2))
			i>1  && pc!(r, i-1, j, -κ_TRUE/hx^2); i<nx && pc!(r, i+1, j, -κ_TRUE/hx^2)
			j>1  && pc!(r, i, j-1, -κ_TRUE/hy^2); j<ny && pc!(r, i, j+1, -κ_TRUE/hy^2)
			i>1  && pc!(r, i-1, j, -v_current[1]/(2hx)); i<nx && pc!(r, i+1, j, v_current[1]/(2hx))
			j>1  && pc!(r, i, j-1, -v_current[2]/(2hy)); j<ny && pc!(r, i, j+1, v_current[2]/(2hy))
			b[r] = source_emission(p, SOURCE_km[1], SOURCE_km[2])
		end
		L = sparse(Ii, Jj, Vv, N, N); u = zeros(N)
		for _ in 1:maxit
			F  = L*u .+ _R_monod.(u) .- b
			du = (L + spdiagm(0 => _dR_monod.(u))) \ (-F)
			u .+= du
			norm(du, Inf) < tol && break
		end
		return function (p)
			i = clamp(round(Int, (p[1]-BBOX_km.xmin)/hx) + 1, 1, nx)
			j = clamp(round(Int, (p[2]-BBOX_km.ymin)/hy) + 1, 1, ny)
			water[i, j] ? u[idx[i, j]] : 0.0
		end
	end
	_samp_monod = _fd_monod_solve()
	Random.seed!(2026)
	y_sensor_monod = [_samp_monod((p[1], p[2])) for p in X_sensor] .+
		σ_data_TRUE .* randn(length(X_sensor))
	y_all_monod = vcat(y_phys, y_bc, y_sensor_monod)
	md"FD-Newton ground truth solved; **$(length(X_sensor))** nonlinear sensor readings."
end

# ╔═╡ 30000005-0005-4000-8000-000000000005
begin
	# Sparse GP prior for the nonlinear model — SAME blocks as HarborField, but with the
	# kernel OUTPUT VARIANCE set to `output_variance` (a fixed scalar on the kernel).
	_k_monod = output_variance *
		(HalfIntegerMaternKernel(3, [ℓ_h]) ⊗ HalfIntegerMaternKernel(3, [ℓ_h]))
	_fg_monod = FunctionalGaussian(GP(_k_monod);
		u    = δ(X_c),
		adv  = δ(X_c) ∘ (v_current[1]*PartialDerivative((1, 0)) +
			v_current[2]*PartialDerivative((0, 1))),
		Δu   = δ(X_c) ∘ (PartialDerivative((2, 0)) + PartialDerivative((0, 2))),
		u_bc = δ(X_b), u_sensor = δ(X_sensor))
	g_monod = vecchia(_fg_monod; ρ = 0.7)
	md"Nonlinear GP prior → **$(length(g_monod))**-dim vecchia GMRF (σ²=$(output_variance))."
end

# ╔═╡ 1ea9628e-7629-43e9-ac99-76f8ef256361
res_harbor_monod = inla(
	harbor_monod_model(y_phys, y_bc, y_sensor_monod, X_c, g_monod), y_all_monod;
	latent_marginalization_method = GaussianMarginal(),
	mode_init = RandomStarts(16),
	executor = ThreadedExecutor(),
	progress = false,
	accumulators = (),
)

# ╔═╡ bf210149-ba8a-4c9d-936a-2ece93775d79
  let
        # Source posterior straight from INLA's hyperparameter marginals
        hm = res_harbor_monod.hyperparameter_marginals
        mx, sx = dist_mean(hm.x_src), sqrt(dist_var(hm.x_src))
        my, sy = dist_mean(hm.y_src), sqrt(dist_var(hm.y_src))

        WATER = RGBf(0.62, 0.79, 0.90)
        fig = Figure(size = (700, 620))
        ax = Axis(fig[1, 1]; aspect = DataAspect(), backgroundcolor = WATER,
                title = "Nonlinear (Monod) source posterior — INLA\n" *
                        "★ true   ● posterior mean   ($(round(mx;digits=2)),$(round(my;digits=2)))" *
                        " ± ($(round(sx;digits=2)),$(round(sy;digits=2)))")
        hidedecorations!(ax); hidespines!(ax)
        xlims!(ax, BBOX_km.xmin - 0.05, BBOX_km.xmax + 0.05)
        ylims!(ax, BBOX_km.ymin - 0.05, BBOX_km.ymax + 0.05)

        # INLA posterior density over leak locations (product of the two 1-D marginals)
        gx = range(BBOX_km.xmin, BBOX_km.xmax; length = 200)
        gy = range(BBOX_km.ymin, BBOX_km.ymax; length = 200)
        dens = [_is_water((x, y)) ? pdf(Normal(mx, sx), x) * pdf(Normal(my, sy), y) : NaN
                        for x in gx, y in gy]
        heatmap!(ax, gx, gy, dens; colormap = :viridis)

        lines!(ax, [Point2f(p[1], p[2]) for p in COAST_km]; color = :white, linewidth = 1.5)
        scatter!(ax, [p[1] for p in SENSORS_km], [p[2] for p in SENSORS_km];
                marker = :circle, markersize = 12, color = :white,
                strokecolor = TUE_AI_COLORS.dark, strokewidth = 2)
        scatter!(ax, [mx], [my]; color = :white, markersize = 15, strokewidth = 1.5, strokecolor = :black)
        scatter!(ax, [SOURCE_km[1]], [SOURCE_km[2]]; marker = :star5, markersize = 24,
                color = COLOR_PROBLEM, strokecolor = :black, strokewidth = 1)
        fig
  end

# ╔═╡ f72c09ac-2219-4a2d-9678-b4970cb6a7c5
  let
        # Posterior-mean latent vector → pull out the u-block (concentration at the collocation points)
        x_mean = dist_mean.(res_harbor_monod.latent_marginals)
        u      = nameview(g_monod, x_mean).u          # same block layout the model used

        WATER = RGBf(0.62, 0.79, 0.90)
        fig = Figure(size = (780, 620))
        ax = Axis(fig[1, 1]; aspect = DataAspect(), backgroundcolor = WATER,
                title = "Posterior-mean concentration field  u(x)  — Monod model")
        hidedecorations!(ax); hidespines!(ax)
        xlims!(ax, BBOX_km.xmin - 0.05, BBOX_km.xmax + 0.05)
        ylims!(ax, BBOX_km.ymin - 0.05, BBOX_km.ymax + 0.05)

        sc = scatter!(ax, [p[1] for p in X_c], [p[2] for p in X_c];
                color = u, colormap = :viridis, markersize = 18, marker = :rect)
        Colorbar(fig[1, 2], sc; label = "concentration u")

        lines!(ax, [Point2f(p[1], p[2]) for p in COAST_km]; color = :white, linewidth = 1.5)
        scatter!(ax, [p[1] for p in SENSORS_km], [p[2] for p in SENSORS_km];
                marker = :circle, markersize = 12, color = :white,
                strokecolor = TUE_AI_COLORS.dark, strokewidth = 2)
        scatter!(ax, [SOURCE_km[1]], [SOURCE_km[2]]; marker = :star5, markersize = 22,
                color = COLOR_PROBLEM, strokecolor = :black, strokewidth = 1)
        fig
  end

# ╔═╡ c1df756b-f9fc-4f9a-b186-fdd9a8d06c40
  begin
        # Dense-sensor nonlinear (Monod) prior — same blocks as g_monod, but u_sensor on X_sensor_d
        _fg_monod_d = FunctionalGaussian(GP(_k_monod);
                u    = δ(X_c),
                adv  = δ(X_c) ∘ (v_current[1]*PartialDerivative((1, 0)) +
                        v_current[2]*PartialDerivative((0, 1))),
                Δu   = δ(X_c) ∘ (PartialDerivative((2, 0)) + PartialDerivative((0, 2))),
                u_bc = δ(X_b), u_sensor = δ(X_sensor_d))
        g_monod_d = vecchia(_fg_monod_d; ρ = 0.7)

        # Honest nonlinear sensor readings at the dense sensors (reuses the FD-Newton sampler)
        Random.seed!(2026)
        y_sensor_monod_d = [_samp_monod((p[1], p[2])) for p in X_sensor_d] .+
                σ_data_TRUE .* randn(length(X_sensor_d))
        y_all_monod_d = vcat(y_phys, y_bc, y_sensor_monod_d)
        md"Dense-sensor Monod prior → **$(length(g_monod_d))**-dim; **$(length(X_sensor_d))** nonlinear readings."
  end

# ╔═╡ c222617a-c9ca-4cd8-b2fb-3b28a74f2e65
  # ╠═╡ show_logs = false
  res_harbor_monod_d = inla(
        harbor_monod_model(y_phys, y_bc, y_sensor_monod_d, X_c, g_monod_d), y_all_monod_d;
        latent_marginalization_method = GaussianMarginal(),
        mode_init = RandomStarts(16),
        executor = ThreadedExecutor(),
        progress = false,
        accumulators = (),
  )

# ╔═╡ a22bec87-c1d5-4af2-83bd-e6c4d12a9b56
  let
        hm = res_harbor_monod_d.hyperparameter_marginals
        mx, sx = dist_mean(hm.x_src), sqrt(dist_var(hm.x_src))
        my, sy = dist_mean(hm.y_src), sqrt(dist_var(hm.y_src))

        WATER = RGBf(0.62, 0.79, 0.90)
        fig = Figure(size = (700, 620))
        ax = Axis(fig[1, 1]; aspect = DataAspect(), backgroundcolor = WATER,
                title = "Monod source posterior — INLA, $(length(X_sensor_d)) sensors\n" *
                        "★ true   ● mean   ($(round(mx;digits=2)),$(round(my;digits=2)))" *
                        " ± ($(round(sx;digits=2)),$(round(sy;digits=2)))")
        hidedecorations!(ax); hidespines!(ax)
        xlims!(ax, BBOX_km.xmin - 0.05, BBOX_km.xmax + 0.05)
        ylims!(ax, BBOX_km.ymin - 0.05, BBOX_km.ymax + 0.05)

        gx = range(BBOX_km.xmin, BBOX_km.xmax; length = 200)
        gy = range(BBOX_km.ymin, BBOX_km.ymax; length = 200)
        dens = [_is_water((x, y)) ? pdf(Normal(mx, sx), x) * pdf(Normal(my, sy), y) : NaN
                        for x in gx, y in gy]
        heatmap!(ax, gx, gy, dens; colormap = :viridis)

        lines!(ax, [Point2f(p[1], p[2]) for p in COAST_km]; color = :white, linewidth = 1.5)
        scatter!(ax, [p[1] for p in X_sensor_d], [p[2] for p in X_sensor_d];
                marker = :circle, markersize = 9, color = :white,
                strokecolor = TUE_AI_COLORS.dark, strokewidth = 1.5)
        scatter!(ax, [mx], [my]; color = :white, markersize = 15, strokewidth = 1.5, strokecolor = :black)
        scatter!(ax, [SOURCE_km[1]], [SOURCE_km[2]]; marker = :star5, markersize = 24,
                color = COLOR_PROBLEM, strokecolor = :black, strokewidth = 1)
        fig
  end

# ╔═╡ 24fa5d7d-f6dd-4bb2-bc34-7041f10b5dc3
  let
        x_mean = dist_mean.(res_harbor_monod_d.latent_marginals)
        u      = nameview(g_monod_d, x_mean).u

        WATER = RGBf(0.62, 0.79, 0.90)
        fig = Figure(size = (780, 620))
        ax = Axis(fig[1, 1]; aspect = DataAspect(), backgroundcolor = WATER,
                title = "Posterior-mean concentration u(x) — Monod, $(length(X_sensor_d)) sensors")
        hidedecorations!(ax); hidespines!(ax)
        xlims!(ax, BBOX_km.xmin - 0.05, BBOX_km.xmax + 0.05)
        ylims!(ax, BBOX_km.ymin - 0.05, BBOX_km.ymax + 0.05)

        sc = scatter!(ax, [p[1] for p in X_c], [p[2] for p in X_c];
                color = u, colormap = :viridis, markersize = 18, marker = :rect)
        Colorbar(fig[1, 2], sc; label = "concentration u")
        lines!(ax, [Point2f(p[1], p[2]) for p in COAST_km]; color = :white, linewidth = 1.5)
        scatter!(ax, [p[1] for p in X_sensor_d], [p[2] for p in X_sensor_d];
                marker = :circle, markersize = 9, color = :white,
                strokecolor = TUE_AI_COLORS.dark, strokewidth = 1.5)
        scatter!(ax, [SOURCE_km[1]], [SOURCE_km[2]]; marker = :star5, markersize = 22,
                color = COLOR_PROBLEM, strokecolor = :black, strokewidth = 1)
        fig
  end

# ╔═╡ 7a513c62-13d0-41a2-8319-834c8ef0f871
  let
        hps = res_harbor_monod.hyperparameter_marginals     # original sensors
        hpd = res_harbor_monod_d.hyperparameter_marginals   # dense sensors
        q05_sx, q95_sx = quantile(hps[:x_src], 0.05), quantile(hps[:x_src], 0.95)
        q05_sy, q95_sy = quantile(hps[:y_src], 0.05), quantile(hps[:y_src], 0.95)
        q05_dx, q95_dx = quantile(hpd[:x_src], 0.05), quantile(hpd[:x_src], 0.95)
        q05_dy, q95_dy = quantile(hpd[:y_src], 0.05), quantile(hpd[:y_src], 0.95)

        WATER = RGBf(0.62, 0.79, 0.90); LAND = RGBf(0.88, 0.85, 0.80); COAST = RGBf(0.28, 0.26, 0.24)
        fig = Figure(size = (1100, 700))
        ax = Axis(fig[1, 1]; aspect = DataAspect(), backgroundcolor = WATER,
                title = "Monod: 90% credible boxes — $(length(X_sensor)) vs $(length(X_sensor_d)) sensors")
        hidedecorations!(ax); hidespines!(ax)
        xlims!(ax, BBOX_km.xmin - 0.05, BBOX_km.xmax + 0.05)
        ylims!(ax, BBOX_km.ymin - 0.05, BBOX_km.ymax + 0.05)
        poly!(ax, Rect2f(BBOX_km.xmin - 0.1, BBOX_km.ymin - 0.1,
                (BBOX_km.xmax - BBOX_km.xmin) + 0.2, (BBOX_km.ymax - BBOX_km.ymin) + 0.2); color = LAND)
        poly!(ax, vcat([Point2f(p[1], p[2]) for p in COAST_km],
                [Point2f(COAST_km[end][1], BBOX_km.ymin), Point2f(COAST_km[1][1], BBOX_km.ymin)]); color = WATER)
        lines!(ax, [Point2f(p[1], p[2]) for p in COAST_km]; color = COAST, linewidth = 2.0)
        length(ISLAND_km) >= 3 && poly!(ax, [Point2f(p[1], p[2]) for p in ISLAND_km];
                color = LAND, strokecolor = COAST, strokewidth = 1.2)

        poly!(ax, Rect2f(q05_sx, q05_sy, q95_sx - q05_sx, q95_sy - q05_sy);
                color = (TU_COLORS.violet, 0.15), strokecolor = TU_COLORS.violet, strokewidth = 2.5,
                label = "$(length(X_sensor)) sensors (90% CI)")
        poly!(ax, Rect2f(q05_dx, q05_dy, q95_dx - q05_dx, q95_dy - q05_dy);
                color = (COLOR_QoI, 0.15), strokecolor = COLOR_QoI, strokewidth = 2.5,
                label = "$(length(X_sensor_d)) sensors (90% CI)")

        scatter!(ax, [p[1] for p in SENSORS_km], [p[2] for p in SENSORS_km];
                marker = :circle, markersize = 12, color = :white,
                strokecolor = TU_COLORS.violet, strokewidth = 2.0, label = "original sensors")
        scatter!(ax, [p[1] for p in X_sensor_d_raw], [p[2] for p in X_sensor_d_raw];
                marker = :utriangle, markersize = 10, color = :white,
                strokecolor = COLOR_QoI, strokewidth = 2.0, label = "dense sensors")
        scatter!(ax, [SOURCE_km[1]], [SOURCE_km[2]]; marker = :star5, markersize = 22,
                color = COLOR_PROBLEM, strokecolor = :black, strokewidth = 1, label = "true source")
        axislegend(ax; position = :rb)
        fig
  end

# ╔═╡ 70000005-0005-4000-8000-000000000005
let
	# archetypal 1D Markov chain (Kalman): scalar tridiagonal precision
	Td = 12
	tri = spdiagm(-1 => fill(-1.0, Td-1), 0 => vcat(2.0, fill(3.0, Td-2), 2.0), 1 => fill(-1.0, Td-1))
	Pd = Float64.(abs.(Matrix(tri)) .> 0)
	fig = Figure(size = (1180, 560))
	ax1 = Axis(fig[1,1]; aspect = DataAspect(), yreversed = true,
		title = "1D Markov chain (Kalman filter/smoother)\ntridiagonal precision   →   O(T)")
	spy!(ax1, tri, color = COLOR_LATENT)
	# heatmap!(ax1, 1:Td, 1:Td, Pd'; colormap = [RGBAf(1,1,1,0), COLOR_INFORMATION_OPERATOR], colorrange = (0,1))
	# for i in 0:Td
	# 	lines!(ax1, [0.5, Td+0.5], [i+0.5, i+0.5]; color = (:gray, 0.35), linewidth = 0.6)
	# 	lines!(ax1, [i+0.5, i+0.5], [0.5, Td+0.5]; color = (:gray, 0.35), linewidth = 0.6)
	# end
	# hidedecorations!(ax1); limits!(ax1, 0.5, Td+0.5, 0.5, Td+0.5)
	ax2 = Axis(fig[1,2]; aspect = DataAspect(), yreversed = true,
		title = "spatial field × time (our harbor)\nblock-tridiagonal precision   →   O(T · space) memory")
	spy!(ax2, st_Jst; markersize = 2.0, framecolor = (:gray, 0.35), color = COLOR_LATENT)
	hidedecorations!(ax2)
	Label(fig[0, :], "Markov in time  ⇒  (block-)tridiagonal precision";
		fontsize = 18, font = :bold)
	fig
end

# ╔═╡ 70000006-0006-4000-8000-000000000006
md"""
## A spacetime harbor simulation

- PDE enters through Crank-Nicolson collocation
- Everything else as before: PPL -> INLA
"""

# ╔═╡ 70000007-0007-4000-8000-000000000007
@latte function st_pn(y_ic, y_bc, y_col, y_sensor, field)
	# Unknown: the leak location.
	x_src ~ Uniform(-0.6, 1.7)
	y_src ~ Uniform( 0.3, 1.8)

	# Latent spatiotemporal field with the block-tridiagonal spacetime prior.
	@random x ~ field(; )

	# Initial condition: the leak deposits a plume at t = 0 (interior points).
	for r in eachindex(y_ic)
		p = st_ICP[r]; f_p = st_emit(st_X[p], x_src, y_src)
		y_ic[r] ~ Normal(x[st_uidx(p, 1)] - f_p, st_σpde; check_args = false)
	end
	# Dirichlet boundary (u = 0 at the coast) at every timestep.
	for r in eachindex(y_bc)
		y_bc[r] ~ Normal(x[st_uidx(st_BCP[r], st_BCK[r])], st_σpde; check_args = false)
	end
	# Crank–Nicolson collocation of  ∂ₜu + ℒu + R(u) = 0  (couples k and k+1).
	for r in eachindex(y_col)
		p = st_COLP[r]; k = st_COLK[r]
		y_col[r] ~ Normal((x[st_uidx(p, k+1)] - x[st_uidx(p, k)]) / st_Δ
			+ 0.5*(x[st_Luidx(p, k)] + x[st_Luidx(p, k+1)])
			+ 0.5*(st_R(x[st_uidx(p, k)]) + st_R(x[st_uidx(p, k+1)])), st_σpde; check_args = false)
	end
	# Time-resolved sensor readings.
	for r in eachindex(y_sensor)
		y_sensor[r] ~ Normal(x[st_uidx(st_SPT[st_SS[r]], st_SK[r])], st_σdata; check_args = false)
	end
end

# ╔═╡ 70000008-0008-4000-8000-000000000008
st_model = st_pn(st_yic, st_ybc, st_ycol, st_dataS, st_FIELD)

# ╔═╡ ee000045-0001-4000-8000-000000000045
md"""
$(@bind run_spacetime CheckBox(default = false)) **Run the spacetime inference** — INLA over the whole space–time field plus two animations, about 2–3 minutes. Off by default so the earlier sections stay responsive; tick it when you get here.
"""

# ╔═╡ 70000009-0009-4000-8000-000000000009
st_result = if !run_spacetime
	missing
else let
	# Warm-start the source at the best point on a coarse harbor grid, then run INLA.
	st_spec = st_model.hyperparameter_spec
	st_ws = Latte.make_workspace(st_model.latent_prior; x_src = st_source[1], y_src = st_source[2])
	st_lp(x, y) = Latte.hyperparameter_logpdf(st_model, Latte.NaturalHyperparameters([x, y], st_spec), st_yall; ws = st_ws)
	st_mode_init = let
		gx = range(st_bbox.xmin+0.1, st_bbox.xmax-0.1; length = 5); gy = range(0.4, 1.6; length = 5)
		bl = -Inf; bp = (0.45, 1.0)
		for xx in gx, yy in gy
			st_iswater((xx, yy)) || continue
			l = st_lp(xx, yy); isfinite(l) || continue
			l > bl && (bl = l; bp = (xx, yy))
		end
		(x_src = bp[1], y_src = bp[2])
	end
	# At σpde=1e-4 the mode-finder trips a NaN-gradient / "did not converge" warning, but CCD
	# reliably recovers the mode — swallow those warnings so the cell output stays clean.
	st_result = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
		inla(st_model, st_yall; accumulators = (), mode_iterations = 15, mode_init = st_mode_init,
			latent_marginalization_method = Latte.GaussianMarginal(),
			executor = Latte.ThreadedExecutor(), exploration_strategy = Latte.CCDExplorationStrategy(),
			progress = false)
	end
end end

# ╔═╡ ee000046-0001-4000-8000-000000000046
if ismissing(st_result)
	skipped("the spacetime source posterior")
else let
	st_hm = st_result.hyperparameter_marginals
	st_mx = mean(st_hm.x_src); st_sx = std(st_hm.x_src)
	st_my = mean(st_hm.y_src); st_sy = std(st_hm.y_src)
	md"INLA source posterior: x = **$(round(st_mx; digits=3)) ± $(round(st_sx; digits=3))**, y = **$(round(st_my; digits=3)) ± $(round(st_sy; digits=3))** km  (true: $(round.(st_source; digits=3)))."
end end

# ╔═╡ 70000011-0011-4000-8000-000000000011
md"""
## Spatiotemporal posterior of the oil concentration
"""

# ╔═╡ 70000012-0012-4000-8000-000000000012
if ismissing(st_result)
	skipped("the spatiotemporal posterior animation")
else let
	# Condition the spatiotemporal field on the posterior source, then animate mean & std.
	umean = dist_mean.(st_result.latent_marginals)
	ustd  = sqrt.(dist_var.(st_result.latent_marginals))
	togrid(vec, k) = begin
		U = fill(NaN, st_nx, st_nx)
		for i in 1:st_nx, j in 1:st_nx
			st_wmask[i,j] || continue
			U[i,j] = vec[st_uidx(st_gid[i,j], k)]
		end
		U
	end
	MEAN = [togrid(umean, k) for k in 1:st_T]; STD = [togrid(ustd, k) for k in 1:st_T]
	umax = maximum(maximum(filter(isfinite, M)) for M in MEAN)
	smax = maximum(maximum(filter(isfinite, S)) for S in STD)
	GOLD = RGBf(0.82, 0.59, 0.0)
	sub = 6; nfr = (st_T - 1) * sub + 1          # interpolate between PN timesteps for a smooth loop
	frame_at(f) = begin
		tc = 1 + (f - 1)/sub; k = clamp(floor(Int, tc), 1, st_T - 1); a = tc - k
		((1-a) .* MEAN[k] .+ a .* MEAN[k+1], (1-a) .* STD[k] .+ a .* STD[k+1], (1-a)*st_ts[k] + a*st_ts[k+1])
	end
	over!(ax) = begin
		lines!(ax, [c[1] for c in st_coast], [c[2] for c in st_coast]; color = :white, linewidth = 1.2)
		scatter!(ax, [p[1] for p in st_sensors], [p[2] for p in st_sensors];
			color = :cyan, marker = :rect, markersize = 9, strokewidth = 1, strokecolor = :black)
		scatter!(ax, [st_source[1]], [st_source[2]]; marker = :star5, color = GOLD, markersize = 17,
			strokewidth = 1, strokecolor = :black)
		limits!(ax, st_bbox.xmin, st_bbox.xmax, st_bbox.ymin, st_bbox.ymax); hidedecorations!(ax)
	end
	M0, S0, _ = frame_at(1)
	Mo = Observable(M0); So = Observable(S0); tstr = Observable("t = 0.00")
	fig = Figure(size = (1180, 600))
	ax1 = Axis(fig[1,1]; aspect = DataAspect(), title = "posterior mean   E[ u(x,t) ]")
	hm1 = heatmap!(ax1, st_gx, st_gy, Mo; colorrange = (0, umax), colormap = :thermal, nan_color = :transparent)
	over!(ax1); Colorbar(fig[1,2], hm1)
	ax2 = Axis(fig[1,3]; aspect = DataAspect(), title = "posterior std   √Var[ u(x,t) ]")
	hm2 = heatmap!(ax2, st_gx, st_gy, So; colorrange = (0, smax), colormap = :viridis, nan_color = :transparent)
	over!(ax2); Colorbar(fig[1,4], hm2)
	Label(fig[0, :], tstr; fontsize = 22, font = :bold)
	# Record a GIF and embed it as a base64 data-URI — always renders in Pluto, auto-loops.
	gifpath = joinpath(tempdir(), "st_plume_anim.gif")
	record(fig, gifpath, 1:nfr; framerate = 12) do f
		M, S, tv = frame_at(f); Mo[] = M; So[] = S; tstr[] = "t = $(round(tv; digits = 2))"
	end
	HTML("""<img src="data:image/gif;base64,$(Base64.base64encode(read(gifpath)))" style="width:100%">""")
end end

# ╔═╡ eff530b2-06fa-4e9e-94d9-a62be0f5d6a0
md"""
## Samples
"""

# ╔═╡ 268c9a4d-0980-4943-8608-9368e8856291
samps = ismissing(st_result) ? missing : rand(st_result, 6).x

# ╔═╡ b9a3bb45-0c32-443a-ac1c-1056101ab2ab
if ismissing(samps)
	skipped("the posterior samples")
else let
    # Three JOINT posterior samples of the whole spacetime field — each a plausible plume realization.
    nsamp = size(samps, 1)
    ncol = min(nsamp, 3); nrow = cld(nsamp, ncol)   # 6 → 3 columns × 2 rows
    togrid(vec, k) = begin
            U = fill(NaN, st_nx, st_nx)
            for i in 1:st_nx, j in 1:st_nx
                    st_wmask[i,j] || continue
                    U[i,j] = vec[st_uidx(st_gid[i,j], k)]
            end
            U
    end
    SGRID = [[togrid(samps[s, :], k) for k in 1:st_T] for s in 1:nsamp]   # SGRID[s][k]
    vmax = maximum(maximum(filter(isfinite, G)) for s in 1:nsamp for G in SGRID[s])
    GOLD = RGBf(0.82, 0.59, 0.0)
    sub = 6; nfr = (st_T - 1) * sub + 1              # interpolate between PN timesteps for a smooth loop
    frame_at(f, s) = begin
            tc = 1 + (f-1)/sub; k = clamp(floor(Int, tc), 1, st_T-1); a = tc - k
            (1-a) .* SGRID[s][k] .+ a .* SGRID[s][k+1]
    end
    tval(f) = begin
            tc = 1 + (f-1)/sub; k = clamp(floor(Int, tc), 1, st_T-1); a = tc - k
            (1-a)*st_ts[k] + a*st_ts[k+1]
    end
    over!(ax) = begin
            lines!(ax, [c[1] for c in st_coast], [c[2] for c in st_coast]; color = :white, linewidth = 1.2)
            scatter!(ax, [p[1] for p in st_sensors], [p[2] for p in st_sensors];
                    color = :cyan, marker = :rect, markersize = 7, strokewidth = 1, strokecolor = :black)
            scatter!(ax, [st_source[1]], [st_source[2]]; marker = :star5, color = GOLD, markersize = 14,
                    strokewidth = 1, strokecolor = :black)
            limits!(ax, st_bbox.xmin, st_bbox.xmax, st_bbox.ymin, st_bbox.ymax); hidedecorations!(ax)
    end
    panels = [Observable(frame_at(1, s)) for s in 1:nsamp]
    tstr = Observable("t = 0.00")
    fig = Figure(size = (400*ncol, 430*nrow + 40))
    for s in 1:nsamp
            r = cld(s, ncol); c = mod1(s, ncol)
            ax = Axis(fig[r, c]; aspect = DataAspect(), title = "posterior sample $s")
            heatmap!(ax, st_gx, st_gy, panels[s]; colorrange = (0, vmax), colormap = :thermal, nan_color = :transparent)
            over!(ax)
    end
    Label(fig[0, :], tstr; fontsize = 22, font = :bold)
    gifpath = joinpath(tempdir(), "st_samples_anim.gif")
    record(fig, gifpath, 1:nfr; framerate = 12) do f
            for s in 1:nsamp; panels[s][] = frame_at(f, s); end 
            tstr[] = "t = $(round(tval(f); digits = 2))"
    end     
    HTML("""<img src="data:image/gif;base64,$(Base64.base64encode(read(gifpath)))" style="width:100%">""")
end end

# ╔═╡ 20000005-0005-4000-8000-000000000005
md"""
# What we built

| Problem | Tool | Key idea |
|---------|------|----------|
| Derivative kernels | [`FunctionalGPs.jl`](https://github.com/timweiland/FunctionalGPs.jl) | Probabilistic numerical computer algebra |
| Scaling to 2D | [`GaussianMarkovRandomFields.jl`](https://github.com/timweiland/GaussianMarkovRandomFields.jl) | Sparse linear algebra through Vecchia approximations |
| Non-Gaussian data | `gaussian_approximation` | Gauss-Newton → Laplace |
| Source inference | [`Latte.jl`](https://github.com/timweiland/Latte.jl) | Hierarchical inference via repeated Laplace |
| Temporal dynamics | Joint sparse spacetime solve | Block tridiagonal = connections to filtering / smoothing |
"""

# ╔═╡ 70000020-0020-4000-8000-000000000020
HTML("""
<style>
  .pn-refs { font-family:'Plus Jakarta Sans',system-ui,sans-serif; max-width:820px; margin:0.5em auto; color:var(--tuai-dark,#383838); }
  .pn-refs h2 { color:var(--tuai-darkblue,#1a3a5b); }
  .pn-refs h3 { color:var(--tuai-darkblue,#1a3a5b); margin:1.1em 0 0.25em; font-size:0.82em; text-transform:uppercase; letter-spacing:0.05em; opacity:0.85; }
  .pn-refs ul { padding-left:0; margin:0; }
  .pn-refs li { list-style:none; margin:0.12em 0; padding:0.4em 0.6em; border-radius:7px; scroll-margin-top:5em; transition:background 0.35s ease; font-size:0.92em; line-height:1.4; }
  .pn-refs li:target { background:#fff3cf; box-shadow:-4px 0 0 var(--tuai-accent,#ea4b2e); }
  .pn-refs a { color:var(--tuai-accent,#ea4b2e); text-decoration:none; }
  .pn-refs a:hover { text-decoration:underline; }
</style>
<div class="pn-refs">
  <h2>References</h2>

  <h3>Data</h3>
  <ul>
    <li id="cite-osm">Coastline of <b>Jinhae Bay, South Korea</b>, derived from <b>OpenStreetMap</b> — © OpenStreetMap contributors, licensed under the <a href="https://www.openstreetmap.org/copyright">Open Database License (ODbL)</a>.</li>
  </ul>

  <h3>Methods</h3>
  <ul>
    <li id="cite-hennig2022">P. Hennig, M. A. Osborne &amp; H. P. Kersting (2022). <i>Probabilistic Numerics: Computation as Machine Learning.</i> Cambridge University Press.</li>
    <li id="cite-pfortner2022">M. Pförtner, I. Steinwart, P. Hennig &amp; J. Wenger (2022). <i>Physics-Informed Gaussian Process Regression Generalizes Linear PDE Solvers.</i> arXiv:2212.12474.</li>
    <li id="cite-vecchia1988">A. V. Vecchia (1988). <i>Estimation and Model Identification for Continuous Spatial Processes.</i> J. R. Stat. Soc. B 50(2), 297–312.</li>
    <li id="cite-schafer2021">F. Schäfer, M. Katzfuss &amp; H. Owhadi (2021). <i>Sparse Cholesky Factorization by Kullback–Leibler Minimization.</i> SIAM J. Sci. Comput. 43(3), A2019–A2046.</li>
    <li id="cite-rueheld2005">H. Rue &amp; L. Held (2005). <i>Gaussian Markov Random Fields: Theory and Applications.</i> Chapman &amp; Hall/CRC.</li>
    <li id="cite-rue2009">H. Rue, S. Martino &amp; N. Chopin (2009). <i>Approximate Bayesian Inference for Latent Gaussian Models by Using Integrated Nested Laplace Approximations.</i> J. R. Stat. Soc. B 71(2), 319–392.</li>
    <li id="cite-sarkka2013">S. Särkkä, A. Solin &amp; J. Hartikainen (2013). <i>Spatiotemporal Learning via Infinite-Dimensional Bayesian Filtering and Smoothing.</i> IEEE Signal Processing Magazine 30(4), 51–61.</li>
    <li id="cite-rw2006">C. E. Rasmussen &amp; C. K. I. Williams (2006). <i>Gaussian Processes for Machine Learning.</i> MIT Press.</li>
    <li id="cite-monod1949">J. Monod (1949). <i>The Growth of Bacterial Cultures.</i> Annual Review of Microbiology 3, 371–394.</li>
  </ul>

  <h3>Software</h3>
  <ul>
    <li><a href="https://github.com/timweiland/FunctionalGPs.jl">FunctionalGPs.jl</a> · Python counterpart: <a href="https://github.com/marvinpfoertner/linpde-gp">linpde-gp</a></li>
    <li><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl">GaussianMarkovRandomFields.jl</a></li>
    <li><a href="https://github.com/timweiland/Latte.jl">Latte.jl</a></li>
  </ul>
</div>
""")

# ╔═╡ Cell order:
# ╟─5d3b7d8a-b2d4-48c2-a681-a5aab956ad64
# ╟─b0000001-0001-4000-8000-000000000001
# ╟─82e2cf25-aa24-4a20-bd4e-3787288f0ae5
# ╟─8cd893ac-a409-4bc7-8df1-69014ad9992d
# ╟─3148f4f4-99c7-4289-9db0-6c264b9ad815
# ╟─a0000000-0000-4000-8000-000000000000
# ╟─a0000001-0001-4000-8000-000000000001
# ╟─ee000001-0001-4000-8000-000000000001
# ╟─a0000002-0002-4000-8000-000000000002
# ╟─ee000002-0001-4000-8000-000000000002
# ╟─ee000003-0001-4000-8000-000000000003
# ╟─b0000004-0004-4000-8000-000000000004
# ╟─b0000005-0005-4000-8000-000000000005
# ╟─05f64818-0e35-4342-be38-d32b4242e3bf
# ╟─b0000004-a001-4000-8000-000000000001
# ╟─c0aa0001-0000-4000-8000-000000000001
# ╟─c9f25088-ce87-4618-8b20-8c5a7c83d96e
# ╟─c0000003-0003-4000-8000-000000000003
# ╠═c0000004-0004-4000-8000-000000000004
# ╟─c0000005-0005-4000-8000-000000000005
# ╟─7349a0fe-5c6a-488d-8a7b-d652452f624f
# ╟─848f8b75-14e5-4dcc-980b-f8c09b2321a2
# ╟─6a52e349-871b-41af-8e59-58246383a7b7
# ╠═c0000004-b001-4000-8000-000000000001
# ╟─75ecc25f-0c82-46c9-965b-7e1a0cf991f5
# ╟─c0000005-c001-4000-8000-000000000001
# ╠═c0000005-c002-4000-8000-000000000002
# ╟─c0000006-b001-4000-8000-000000000001
# ╟─c0000006-0006-4000-8000-000000000006
# ╠═c0000007-0007-4000-8000-000000000007
# ╟─c0000007-a001-4000-8000-000000000001
# ╠═c0000007-a002-4000-8000-000000000002
# ╟─c0000008-0008-4000-8000-000000000008
# ╠═925b5bc1-2791-4637-bf64-507ccc3863e2
# ╠═c0000009-0009-4000-8000-000000000009
# ╟─c0000009-a001-4000-8000-000000000001
# ╟─ee000004-0001-4000-8000-000000000004
# ╟─ee000005-0001-4000-8000-000000000005
# ╠═c0000009-a002-4000-8000-000000000002
# ╟─c0000009-a003-4000-8000-000000000003
# ╠═c0000009-a004-4000-8000-000000000004
# ╟─ee000006-0001-4000-8000-000000000006
# ╟─ee000007-0001-4000-8000-000000000007
# ╠═c0000009-a005-4000-8000-000000000005
# ╟─c0000010-a001-4000-8000-000000000001
# ╟─ee000008-0001-4000-8000-000000000008
# ╟─ee000009-0001-4000-8000-000000000009
# ╟─c0000010-0010-4000-8000-000000000010
# ╟─ee000010-0001-4000-8000-000000000010
# ╟─ee000011-0001-4000-8000-000000000011
# ╟─ee000012-0001-4000-8000-000000000012
# ╠═ee000013-0001-4000-8000-000000000013
# ╟─ee000014-0001-4000-8000-000000000014
# ╟─ee000015-0001-4000-8000-000000000015
# ╟─ee000016-0001-4000-8000-000000000016
# ╟─d0000001-0001-4000-8000-000000000001
# ╟─6a84dbb8-e00e-49a2-a43c-0c261e757985
# ╟─d0000005-0005-4000-8000-000000000005
# ╠═d0000006-0006-4000-8000-000000000006
# ╟─d0000007-0007-4000-8000-000000000007
# ╠═d0000007-a001-4000-8000-000000000001
# ╟─d0000008-0008-4000-8000-000000000008
# ╟─d0000008-a001-4000-8000-000000000001
# ╟─d0000004-0004-4000-8000-000000000004
# ╠═ee000017-0001-4000-8000-000000000017
# ╟─ee000018-0001-4000-8000-000000000018
# ╟─ee000019-0001-4000-8000-000000000019
# ╠═ee000020-0001-4000-8000-000000000020
# ╟─ee000021-0001-4000-8000-000000000021
# ╟─ee000022-0001-4000-8000-000000000022
# ╠═d0000003-0003-4000-8000-000000000003
# ╟─ee000023-0001-4000-8000-000000000023
# ╟─05d64fdd-e02a-4e73-be4b-b3cf8eb74e13
# ╟─72329592-fabc-4acd-b36e-0c5deabf0416
# ╟─e0000001-0001-4000-8000-000000000001
# ╟─532ae585-083e-4214-a480-bc8e5999af86
# ╟─e0000003-0003-4000-8000-000000000003
# ╟─d5247444-711d-47f4-b502-558280805523
# ╟─e0000004-0004-4000-8000-000000000004
# ╟─e0000005-0005-4000-8000-000000000005
# ╟─e0000006-0006-4000-8000-000000000006
# ╟─5abf86bc-2be2-4ece-8e29-1760df78eaea
# ╟─e0000007-0007-4000-8000-000000000007
# ╠═be6a7c2e-e6db-4021-a03e-00286b99d4b5
# ╠═4cf9bbdf-c01b-4afe-af15-1c9d7483d515
# ╠═e7ac5710-1f8f-48c0-b931-0bc6f183849c
# ╠═04d7e02b-f42f-4ac3-976c-f2cfcd60913c
# ╟─74578565-b418-4bf3-87ef-d056eae3a7e6
# ╠═b0f03964-29c2-4be6-b8d2-8b86c6d6db77
# ╠═1b54045c-6f58-42bf-815a-5ff3e2061852
# ╠═dd2547f2-cfae-40cc-8126-4a7a37220198
# ╟─ee000024-0001-4000-8000-000000000024
# ╟─ee000025-0001-4000-8000-000000000025
# ╟─ee000026-0001-4000-8000-000000000026
# ╠═ee000027-0001-4000-8000-000000000027
# ╠═ee000028-0001-4000-8000-000000000028
# ╟─ee000029-0001-4000-8000-000000000029
# ╟─ee000030-0001-4000-8000-000000000030
# ╟─4c4d5442-8f9c-4ce0-9d65-7fc836971625
# ╟─24936159-6acf-44ac-934f-7fd0316ac4d5
# ╟─6252efb4-badc-4aa2-bcac-89eb4278a391
# ╟─200a9e77-74ad-49c9-a1d3-d1b9194fafa4
# ╟─6a021524-df13-46ad-87e9-5da10bad13cc
# ╟─4d962849-5b68-4e48-a2ff-8ceaebe5e6f5
# ╟─20000002-a001-4000-8000-000000000001
# ╟─20000002-a002-4000-8000-000000000002
# ╟─20000002-a003-4000-8000-000000000003
# ╟─a000ce11-0000-4000-8000-000000000001
# ╠═a000ce11-0000-4000-8000-000000000002
# ╟─20000002-a004-4000-8000-000000000004
# ╟─20000002-a005-4000-8000-000000000005
# ╟─5222c42c-2613-4528-8ef8-385f5702e3f4
# ╟─ee000031-0001-4000-8000-000000000031
# ╟─ee000032-0001-4000-8000-000000000032
# ╟─ee000033-0001-4000-8000-000000000033
# ╟─ee000034-0001-4000-8000-000000000034
# ╠═20000002-a006-4000-8000-000000000006
# ╟─ee000035-0001-4000-8000-000000000035
# ╟─ee000036-0001-4000-8000-000000000036
# ╟─ee000037-0001-4000-8000-000000000037
# ╟─20000003-0003-4000-8000-000000000003
# ╠═20000004-a001-4000-8000-000000000001
# ╟─20000004-a002-4000-8000-000000000002
# ╟─20000004-a003-4000-8000-000000000003
# ╟─ae8d17be-a783-47e5-995d-d408515e8cb3
# ╟─20000004-b001-4000-8000-000000000001
# ╟─20000004-b002-4000-8000-000000000002
# ╟─20000004-b003-4000-8000-000000000003
# ╟─20000004-b004-4000-8000-000000000004
# ╠═20000004-b005-4000-8000-000000000005
# ╟─df109776-1162-4cb4-a771-188332172be1
# ╟─20000004-b006-4000-8000-000000000006
# ╟─ee000038-0001-4000-8000-000000000038
# ╟─ee000039-0001-4000-8000-000000000039
# ╟─ee000040-0001-4000-8000-000000000040
# ╠═ee000041-0001-4000-8000-000000000041
# ╟─ee000042-0001-4000-8000-000000000042
# ╟─ee000043-0001-4000-8000-000000000043
# ╟─ee000044-0001-4000-8000-000000000044
# ╟─89d16755-a925-4e33-822d-92a3711f8c52
# ╟─30000001-0001-4000-8000-000000000001
# ╟─30000002-0002-4000-8000-000000000002
# ╟─30000003-0003-4000-8000-000000000003
# ╟─f0000002-0002-4000-8000-000000000002
# ╟─30000004-0004-4000-8000-000000000004
# ╟─30000005-0005-4000-8000-000000000005
# ╠═30000006-0006-4000-8000-000000000006
# ╠═1ea9628e-7629-43e9-ac99-76f8ef256361
# ╟─bf210149-ba8a-4c9d-936a-2ece93775d79
# ╟─f72c09ac-2219-4a2d-9678-b4970cb6a7c5
# ╟─c1df756b-f9fc-4f9a-b186-fdd9a8d06c40
# ╟─c222617a-c9ca-4cd8-b2fb-3b28a74f2e65
# ╟─a22bec87-c1d5-4af2-83bd-e6c4d12a9b56
# ╟─24fa5d7d-f6dd-4bb2-bc34-7041f10b5dc3
# ╟─7a513c62-13d0-41a2-8319-834c8ef0f871
# ╟─6d3a7f9e-345b-4076-9066-869afb6dab42
# ╟─70000001-0001-4000-8000-000000000001
# ╟─70000002-0002-4000-8000-000000000002
# ╟─70000003-0003-4000-8000-000000000003
# ╟─70000004-0004-4000-8000-000000000004
# ╟─70000005-0005-4000-8000-000000000005
# ╟─70000006-0006-4000-8000-000000000006
# ╠═70000007-0007-4000-8000-000000000007
# ╟─70000008-0008-4000-8000-000000000008
# ╟─ee000045-0001-4000-8000-000000000045
# ╟─70000009-0009-4000-8000-000000000009
# ╟─ee000046-0001-4000-8000-000000000046
# ╟─70000011-0011-4000-8000-000000000011
# ╟─70000012-0012-4000-8000-000000000012
# ╟─eff530b2-06fa-4e9e-94d9-a62be0f5d6a0
# ╠═268c9a4d-0980-4943-8608-9368e8856291
# ╟─b9a3bb45-0c32-443a-ac1c-1056101ab2ab
# ╟─20000005-0005-4000-8000-000000000005
# ╟─70000020-0020-4000-8000-000000000020
