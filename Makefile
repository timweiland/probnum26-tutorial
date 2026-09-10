# ProbNum 2026 — Who polluted the harbor? A live ProbNum investigation.
# Install Pluto in the global Julia 1.12 environment; the notebook activates
# this directory's project on launch.

NB ?= live_demo_solution.jl
.PHONY: help demo solution serve participant instantiate

help:
	@echo "make instantiate            install this project's dependencies (once after cloning)"
	@echo "make demo                   open the participant notebook in Pluto"
	@echo "make solution               open the solution notebook in Pluto"
	@echo "make serve                  open the solution notebook in Pluto"
	@echo "make serve NB=live_demo.jl   open the participant notebook"
	@echo "make participant            regenerate live_demo.jl from the solution"

serve:
	julia +1.12 -t auto -e 'import Pluto; Pluto.run(notebook="$(NB)")'

demo:
	julia +1.12 -t auto -e 'import Pluto; Pluto.run(notebook="live_demo.jl")'

solution:
	julia +1.12 -t auto -e 'import Pluto; Pluto.run(notebook="live_demo_solution.jl")'

participant:
	julia +1.12 make_participant.jl

instantiate:
	julia +1.12 --project=. -e 'import Pkg; Pkg.instantiate()'
