"""
    julia utilities/coverageSummary.jl [--badge coverage.json] merged.info shard1.info [shard2.info ...]

The lcov files of the CI's test shards merged into `merged.info` (a line's hit counts added), and the summary in
Markdown on stdout (CI.yml writes it to the job summary): the lines covered of the lines with code, in all and per
file, least covered first. With `--badge`, also the total as a shields.io endpoint file (CI.yml publishes it on
gh-pages for the README's badge). No external service.
"""

"Hit counts by line, by source file."
const Counts = Dict{String, Dict{Int, Int}}

"The `DA:` records of the lcov file `path` added onto `counts`."
function readLcov!(counts::Counts, path::String)::Counts
  local lines::Dict{Int, Int} = Dict{Int, Int}()
  for record::String in eachline(path)
    if startswith(record, "SF:")
      lines = get!(counts, record[4:end], Dict{Int, Int}())
    elseif startswith(record, "DA:")
      local fields::Vector{SubString{String}} = split(record[4:end], ',')
      local line::Int = parse(Int, fields[1])
      lines[line] = get(lines, line, 0) + parse(Int, fields[2])
    end
  end
  return counts
end

"`counts` as an lcov file at `path`."
function writeLcov(path::String, counts::Counts)::Nothing
  open(path, "w") do io::IO
    for file::String in sort!(collect(keys(counts)))
      local lines::Dict{Int, Int} = counts[file]
      println(io, "SF:", file)
      for line::Int in sort!(collect(keys(lines)))
        println(io, "DA:", line, ",", lines[line])
      end
      println(io, "LH:", count(>(0), values(lines)))
      println(io, "LF:", length(lines))
      println(io, "end_of_record")
    end
  end
  return nothing
end

percent(hit::Int, total::Int)::String = total == 0 ? "-" : string(round(100 * hit / total; digits = 1), " %")

"""
The badge's color for `hit` of `total` lines: shields.io's names, from red below 50 % to brightgreen from 90 %.
"""
function badgeColor(hit::Int, total::Int)::String
  local p::Float64 = total == 0 ? 0.0 : 100 * hit / total
  for (bound::Float64, color::String) in ((90.0, "brightgreen"), (80.0, "green"), (70.0, "yellowgreen"),
                                          (60.0, "yellow"), (50.0, "orange"))
    p >= bound && return color
  end
  return "red"
end

"""
`hit` of `total` lines as a shields.io endpoint file at `path` (https://shields.io/badges/endpoint-badge).
"""
function writeBadge(path::String, hit::Int, total::Int)::Nothing
  local message::String = total == 0 ? "unknown" : string(round(100 * hit / total; digits = 1), "%")
  write(path, """{"schemaVersion": 1, "label": "coverage", "message": "$(message)", "color": "$(badgeColor(hit, total))"}\n""")
  return nothing
end

function main(argv::Vector{String})::Nothing
  local args::Vector{String} = copy(argv)
  local badge::Union{Nothing, String} = nothing
  if length(args) >= 2 && args[1] == "--badge"
    badge = args[2]
    args = args[3:end]
  end
  length(args) >= 1 || error("usage: coverageSummary.jl [--badge coverage.json] merged.info shard.info...")
  local counts::Counts = Counts()
  for path::String in args[2:end]
    readLcov!(counts, path)
  end
  writeLcov(args[1], counts)
  local rows::Vector{Tuple{String, Int, Int}} =
    Tuple{String, Int, Int}[(file, count(>(0), values(lines)), length(lines)) for (file, lines) in counts]
  sort!(rows; by = (r::Tuple{String, Int, Int}) -> (r[3] == 0 ? 1.0 : r[2] / r[3], r[1]))
  local hit::Int = sum(r[2] for r in rows; init = 0)
  local total::Int = sum(r[3] for r in rows; init = 0)
  badge === nothing || writeBadge(badge, hit, total)
  println("## Coverage of OMBackend.jl/src")
  println()
  println("**$(percent(hit, total))**: $(hit) of $(total) lines, $(length(rows)) files, $(length(args) - 1) test shards merged.")
  println()
  println("<details><summary>Per file, least covered first</summary>")
  println()
  println("| file | covered | lines | |")
  println("|---|---:|---:|---:|")
  for (file::String, h::Int, n::Int) in rows
    println("| `", file, "` | ", h, " | ", n, " | ", percent(h, n), " |")
  end
  println()
  println("</details>")
  return nothing
end

main(ARGS)
