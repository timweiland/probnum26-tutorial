# Plot styling retained from the MLSS/ICML tutorial; see LICENSE.
module HarborPlotting
using CairoMakie: Theme, band!, series!, lines!
using Colors: RGB

# Color scheme of the University of Tübingen
const TU_COLORS = (
    # Primary Colors
    red        = RGB(165/255, 030/255, 055/255),
    gold       = RGB(180/255, 160/255, 105/255),
    dark       = RGB(050/255, 065/255, 075/255),
    gray       = RGB(175/255, 179/255, 183/255),
    # Secondary Colors
    darkblue   = RGB(065/255, 090/255, 140/255),
    blue       = RGB(000/255, 105/255, 170/255),
    lightblue  = RGB(080/255, 170/255, 200/255),
    lightgreen = RGB(130/255, 185/255, 160/255),
    green      = RGB(125/255, 165/255, 075/255),
    darkgreen  = RGB(050/255, 110/255, 030/255),
    ocre       = RGB(200/255, 080/255, 060/255),
    violet     = RGB(175/255, 110/255, 150/255),
    mauve      = RGB(180/255, 160/255, 150/255),
    beige      = RGB(215/255, 180/255, 105/255),
    orange     = RGB(210/255, 150/255, 000/255),
    brown      = RGB(145/255, 105/255, 070/255),
)

# Color scheme of the Tübingen AI Center
const TUE_AI_COLORS = (
    dark         = RGB(56/255,   56/255,  56/255),
    gray         = RGB(246/255, 246/255, 246/255),
    darkblue     = RGB( 26/255,  58/255,  91/255),
    accent       = RGB(234/255,  75/255,  46/255),
    lightblue    = RGB(133/255, 203/255, 210/255),
    oceanblue    = RGB(119/255, 221/255, 204/255),
    oceangreen   = RGB(119/255, 221/255, 159/255),
    springgreen  = RGB(186/255, 213/255,  72/255),
    brightyellow = RGB(255/255, 221/255,   0/255),
)

const TUE_AI_PALETTE = [
    TUE_AI_COLORS.darkblue,
    TUE_AI_COLORS.accent,
    TUE_AI_COLORS.dark,
    TUE_AI_COLORS.lightblue,
    TUE_AI_COLORS.oceanblue,
    TUE_AI_COLORS.oceangreen,
    TUE_AI_COLORS.springgreen,
    TUE_AI_COLORS.brightyellow,
]

const COLOR_PROBLEM = TU_COLORS.orange;
const COLOR_LATENT = TUE_AI_COLORS.darkblue;
const COLOR_QoI = TUE_AI_COLORS.oceangreen;
const COLOR_INFORMATION_OPERATOR = TU_COLORS.lightblue;

# Makie theme for all plots
const THEME = Theme(
    palette = (
        color = TUE_AI_PALETTE,
        patchcolor = TUE_AI_PALETTE,
    ),
    Axis = (
        leftspinecolor = TUE_AI_COLORS.dark,
        rightspinecolor = TUE_AI_COLORS.dark,
        topspinecolor = TUE_AI_COLORS.dark,
        bottomspinecolor = TUE_AI_COLORS.dark,
    ),
    Errorbars = (
        linewidth = 1,
    ),
    Lines = (
        linewidth = 1,
    ),
    VLines = (
        linewidth = 1,
    ),
    Series = (
        linewidth = 1,
    ),
)

function plot_gp!(
    ax,
    xs,
    mean,
    std;
    nsigma = 1.96,
    color=TUE_AI_COLORS.dark,
    bandalpha=0.2,
    samples = nothing,
)
    mean = Vector(mean)
    std = Vector(std)

    # 95% credible interval
    band!(
        ax,
        xs,
        mean .- nsigma .* std,
        mean .+ nsigma .* std;
        color = (color, bandalpha),
    )

    # Samples
    if samples !== nothing
        series!(ax, xs, samples'; solid_color = (color, 0.4))
    end

    # Mean
    lines!(ax, xs, mean; color = color)
end

const PNMETHODS_PROBLEM_LINES_KWARGS = (
    color = COLOR_PROBLEM,
    linewidth = 2,
    linestyle = (:dot, .5),
)

end
