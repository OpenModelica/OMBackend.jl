"""
    julia utilities/coverageSummary.jl merged.info shard1.info [shard2.info ...]

The lcov files of the CI's test shards merged into `merged.info` (a line's hit counts added), and the summary in
Markdown on stdout (CI.yml writes it to the job summary): the lines covered of the lines with code, in all and per
file, least covered first. No external service.
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

function main(args::Vector{String})::Nothing
  length(args) >= 1 || error("usage: coverageSummary.jl merged.info shard.info...")
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
