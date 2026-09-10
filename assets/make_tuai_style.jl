# Generate a SELF-CONTAINED (offline) Pluto stylesheet matching the Tübingen AI Beamer theme.
# Plus Jakarta Sans + Roboto Mono are embedded as base64 @font-face (no internet, no system fonts),
# plus the AI-center palette. Re-run with:  julia make_tuai_style.jl   → writes tuai_style.html
using Base64
fdir = joinpath(@__DIR__, "fonts")
b64(f) = base64encode(read(joinpath(fdir, f)))
face(fam, file, w; style = "normal") =
    "@font-face{font-family:'$fam';font-style:$style;font-weight:$w;" *
    "src:url(data:font/ttf;base64,$(b64(file))) format('truetype');font-display:swap;}"

faces = join([
        face("Plus Jakarta Sans", "PlusJakartaSans-Light.ttf",       "300"),
        face("Plus Jakarta Sans", "PlusJakartaSans-LightItalic.ttf", "300"; style = "italic"),
        face("Plus Jakarta Sans", "PlusJakartaSans-Medium.ttf",      "500"),
        face("Plus Jakarta Sans", "PlusJakartaSans-Bold.ttf",        "700"),
        face("Roboto Mono",       "RobotoMono-VariableFont_wght.ttf", "100 700"),
    ], "\n")

rules = raw"""
:root{--tuai-dark:#383838;--tuai-darkblue:#1a3a5b;--tuai-accent:#ea4b2e;
      --tuai-lightblue:#85cbd2;--tuai-spring:#bad548;--tuai-gray:#f6f6f6;
      --slide-side-margin:190px;}            /* symmetric L/R margin — tune this one value */

/* centered content with symmetric, decreased margins (overrides Pluto's lopsided default) */
main{max-width:min(1500px, max(700px, calc(100% - 2 * var(--slide-side-margin)))) !important;
     margin-left:auto !important; margin-right:auto !important;}

/* body text */
pluto-output{font-family:'Plus Jakarta Sans',sans-serif;font-weight:300;
  color:var(--tuai-dark);font-size:1.05rem;line-height:1.55;}
pluto-output strong,pluto-output b{font-weight:700;}

/* headings: Jakarta bold, dark blue */
pluto-output h1,pluto-output h2,pluto-output h3,pluto-output h4,pluto-output h5,pluto-output h6{
  font-family:'Plus Jakarta Sans',sans-serif;color:var(--tuai-darkblue);
  font-weight:700;letter-spacing:-0.01em;}
/* h1 & h2 -> Beamer-style frame-title band (h2 is the per-slide title in this notebook) */
pluto-output h1,pluto-output h2{color:#fff;background:var(--tuai-darkblue);
  padding:0.35em 0.6em;border-radius:6px;margin-top:0.5em;margin-bottom:0.3em;}

/* links + emphasis */
pluto-output a{color:var(--tuai-accent);text-decoration:none;}
pluto-output a:hover{text-decoration:underline;}

/* code -> Roboto Mono (incl. the editor) */
pluto-output code,pluto-output pre,.cm-editor .cm-content{font-family:'Roboto Mono',monospace;}
pluto-output :not(pre)>code{color:var(--tuai-darkblue);}

/* list bullets in accent */
pluto-output ul li::marker{color:var(--tuai-accent);}

/* blockquote -> Beamer "block" */
pluto-output blockquote{background:color-mix(in srgb,var(--tuai-lightblue) 20%,white);
  border-left:4px solid var(--tuai-lightblue);border-radius:4px;padding:0.5em 0.9em;}

/* tables */
pluto-output table th{background:color-mix(in srgb,var(--tuai-lightblue) 25%,white);
  color:var(--tuai-darkblue);font-weight:700;}
pluto-output table td,pluto-output table th{border-color:var(--tuai-gray);}

/* ── dark mode (follows the OS / browser setting, like Pluto itself) ──────────
   The notebook is designed for light mode (presentation). In dark mode we
   re-map the palette variables and neutralise the few hard-coded light
   backgrounds in the HTML cards, so text stays readable. Makie figures are
   server-rendered images and keep their light background. */
@media (prefers-color-scheme: dark){
  :root{--tuai-dark:#e4e4e4;          /* body text */
        --tuai-darkblue:#9ec5e8;      /* headings / card titles as TEXT */
        --tuai-lightblue:#5f9aa3;     /* borders */
        --tuai-gray:#2b2b2b;          /* card inner rows */
        --tuai-card-bg:#1f1f1f;}
  /* the h1/h2 title band keeps its blue background (not the remapped text colour) */
  pluto-output h1,pluto-output h2{background:#1a3a5b;color:#fff;}
  pluto-output blockquote{background:color-mix(in srgb,#85cbd2 18%,#1f1f1f);}
  pluto-output table th{background:color-mix(in srgb,#85cbd2 22%,#1f1f1f);}
  /* HTML cards with an inline white background (hierarchy card, step banners,
     package cards, roles pill row, references) */
  pluto-output [style*="background:#fff"],pluto-output [style*="background:white"],
  pluto-output [style*="background: #fff"]{background:var(--tuai-card-bg) !important;}
  /* keep QR codes scannable: white tile behind the SVG */
  pluto-output [style*="background:white"] svg{background:#fff;border-radius:6px;padding:4px;}
  /* hard-coded dark-blue text inside the hierarchy card (row titles, the p(u|θ) term) */
  pluto-output [style*="color:#1A3A5B"],pluto-output [style*="color: #1A3A5B"]{color:#9ec5e8 !important;}
  /* references: highlighted entry */
  pluto-output .pn-refs li:target{background:#3a3520 !important;}
  /* PNG schematics inside the step banners are transparent — fine as is */
}
"""

open(joinpath(@__DIR__, "tuai_style.html"), "w") do io
    write(io, "<style>\n", faces, "\n", rules, "</style>\n")
end
let p = joinpath(@__DIR__, "tuai_style.html")
    println("wrote ", p, " (", round(filesize(p)/1024; digits=1), " KB)")
end
