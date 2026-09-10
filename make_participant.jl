# Generate the participant notebook from the solution notebook.
#
#   julia make_participant.jl [live_demo_solution.jl] [live_demo.jl]
#
# Every line in the solution that ends with `#SOL` has its right-hand side
# (after `=` or `~`) replaced by `missing`. Nothing else changes, so the two
# notebooks never drift apart.

src_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "live_demo_solution.jl")
dst_path = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "live_demo.jl")

const SOL_RE = r"^(\s*[^=~#]*?\s*[=~]\s*)(.*?)(,?)\s*#SOL\s*$"

n = 0
out = IOBuffer()
for line in eachline(src_path; keep = true)
    m = match(SOL_RE, chomp(line))
    if m === nothing
        write(out, line)
    else
        global n += 1
        write(out, m.captures[1], "missing", m.captures[3], "\n")
    end
end

open(dst_path, "w") do io
    write(io, take!(out))
end
println("wrote $(dst_path): replaced $n solution line(s) with `missing`")
