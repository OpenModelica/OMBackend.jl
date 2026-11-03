#= The code generators ask the references (9c): a record field's reference and identifier from its parts, the
   variables a when body writes by their references, the hoisted hvcats by their expressions; and through small models
   (Models/ReferenceLookups.mo) a table read by a discrete index, an array an initial algorithm assigns, a function
   over an array of records, a higher derivative's auxiliary beside a variable named as its code identifier. =#
using Test
import OMBackend
import OMFrontend

@testset "Code generation by references" begin
  local SC = OMBackend.SimCode
  local IR = OMBackend.OMJLIR
  local CG = OMBackend.CodeGeneration
  local part = (n::Symbol, subs::Vector{Int} = Int[], ty::IR.ModelicaType = IR.T_REAL()) ->
    IR.CrefPart(n, IR.Subscript[IR.INDEX(IR.INTEGER(k)) for k in subs], ty)
  local ref = (ps::IR.CrefPart...) -> IR.ComponentRef(IR.CrefPart[ps...])
  local read = (c::IR.ComponentRef) -> SC.EXP_CREF(SC.SimCref(Symbol(SC.derivedKey(c)), Int[], c), SC.TYPE_REAL())

  @testset "a flattened record's field by its parts" begin
    #= `u[2].re` of a function's array of records read as its field array's element `u.re[2]`, identifier `u_re[2]` =#
    local element = SC.fieldElementCref(ref(part(:u, [2]), part(:re)))
    @test element.name === Symbol("u.re[2]") && SC.derivedKey(element) == "u_re[2]"
    @test SC.fieldOfElement(part(:u, [2]), :im, IR.T_REAL()).name === Symbol("u.im[2]")
    @test SC.derivedKey(SC.fieldCref(ref(part(:r)), "a")) == "r_a"
  end

  @testset "the code view's identifiers of what is no variable" begin
    #= a field of a constant record (`Medium.data.MM`) by its derived identifier; an element of its array field
       (`Medium.data.alow[1]`, a constant's array read in an equation: MSL Media's Inverse_sh_T) by its array's =#
    local rn = SC.Renaming(Dict{String, String}())
    local data = (f::Symbol, subs::Vector{Int}) -> ref(part(:Medium, Int[], IR.T_CLASS(:M)), part(:data, Int[], IR.T_CLASS(:D)), part(f, subs))
    local field = SC.SimCref(Symbol("Medium.data.MM"), Int[], data(:MM, Int[]))
    local element = SC.SimCref(Symbol("Medium.data.alow"), Int[1], data(:alow, [1]))
    @test SC.renamed(rn, field).sym === :Medium_data_MM
    @test SC.renamed(rn, element).sym === :Medium_data_alow && SC.renamed(rn, element).subs == [1]
    #= a name of no reference (an iterator) as it is =#
    @test SC.renamed(rn, SC.SimCref(:i, Int[], nothing)).sym === :i
  end

  @testset "the variables a when body writes" begin
    local src = SC.NO_SOURCE
    local call = (args::SC.Exp...) -> SC.CALL(:f, SC.Exp[args...], SC.TYPE_REAL(), SC.CALL_FUNCTION)
    local written = CG._writtenArrays(SC.Statement[SC.ASSIGN(read(ref(part(:r))), SC.RCONST(1.0), src),
                                                   SC.ASSIGN(read(ref(part(:buf, [3]))), SC.RCONST(1.0), src)])
    #= a field of a written record, an element of a written array: written; a variable named alike: not =#
    @test CG._readsWritten(call(read(ref(part(:r), part(:a)))), written)
    @test CG._readsWritten(call(read(ref(part(:buf, [1])))), written)
    @test !CG._readsWritten(call(read(ref(part(:r_a)))), written)
    #= a record read whole when one of its fields is written =#
    @test CG._readsWritten(call(read(ref(part(:s)))), CG._writtenArrays(SC.Statement[SC.ASSIGN(read(ref(part(:s), part(:x))), SC.RCONST(1.0), src)]))
  end

  @testset "hoisted hvcats by their expressions" begin
    local U = CG.CodeGenerationUtil
    local m = (x::Any) -> :(hvcat((2, 2), $x, b, c, d))
    local lined = (line::Int) -> Expr(:block, LineNumberNode(line, :generator), :a)
    #= equal calls whose code carries different line numbers share one local; `1` and `1.0` do not =#
    local (eqs::Vector, preamble::Vector{Expr}) = U.extractCommonHvcats(Any[:(y1 = $(m(lined(1)))), :(y2 = $(m(lined(9))))])
    @test length(preamble) == 1 && !occursin("hvcat", string(eqs[1])) && !occursin("hvcat", string(eqs[2]))
    @test isempty(last(U.extractCommonHvcats(Any[:(y1 = $(m(1))), :(y2 = $(m(1.0)))])))
    #= the preamble in the order of the first occurrences =#
    local (_, ordered::Vector{Expr}) = U.extractCommonHvcats(Any[:(y1 = $(m(:z)) + $(m(:a))), :(y2 = $(m(:a)) + $(m(:z)))])
    @test occursin("hvcat((2, 2), z", string(ordered[1])) && occursin("hvcat((2, 2), a", string(ordered[2]))
  end

  local file::String = joinpath(@__DIR__, "Models", "ReferenceLookups.mo")
  local simulate = (model::String, stop::Float64) -> begin
    local (fm, cache) = OMFrontend.instantiateSCodeToFM("ReferenceLookups." * model, OMFrontend.translateToSCode(OMFrontend.parseFile(file)))
    OMBackend.translate(fm; functionList = OMFrontend.cacheToFunctionList(cache))
    local name::String = OMBackend.canonicalName("ReferenceLookups." * model)
    (OMBackend.COMPILED_MODELS_MTK[name][1], OMBackend.simulateModel(name; tspan = (0.0, stop)))
  end
  local value = (sol, v::String, k::Int) -> OMBackend.getVariableValues(sol, v)[k]

  @testset "a parameter table read by a discrete index" begin
    #= the lowering's choice among the elements (an if-equation); its values =#
    local (_, sol) = simulate("DynamicIndex", 1.0)
    @test string(sol.retcode) == "Success"
    @test value(sol, "y", 1) ≈ 10.0 && OMBackend.getVariableValues(sol, "y")[end] ≈ 30.0
    #= a reference `tab[idx]` the code generation meets (CDL's TimeTable): the table of the elements by their
       references' identifiers, indexed at run time by the variable's symbol =#
    isdefined(@__MODULE__, :BackendTestMocks) || include(joinpath(@__DIR__, "backendTestMocks.jl"))
    local M = BackendTestMocks
    local sc = M.mockSimCode(vcat(SC.SIMVAR[M.simVar("tab[$k]", SC.PARAMETER(SC.RCONST(10.0 * k))) for k in 1:3],
                                  SC.SIMVAR[M.simVar("idx", SC.DISCRETE(), SC.TYPE_INTEGER())]))
    local tabIdx = ref(IR.CrefPart(:tab, IR.Subscript[IR.INDEX(IR.CREF(ref(part(:idx, Int[], IR.T_INTEGER())), IR.T_INTEGER()))], IR.T_REAL()))
    local lookup = CG.MTK_CodeGenerationUtil._dynamicSubscriptLookup(tabIdx, sc)
    @test lookup isa Expr && lookup.args[1] == :(OMBackend.CodeGeneration.constTableLookup)
    @test lookup.args[2] == [10.0, 20.0, 30.0] && lookup.args[3] === :idx
    #= the subscript a variable that is none: nothing =#
    local tabK = ref(IR.CrefPart(:tab, IR.Subscript[IR.INDEX(IR.CREF(ref(part(:k, Int[], IR.T_INTEGER())), IR.T_INTEGER()))], IR.T_REAL()))
    @test CG.MTK_CodeGenerationUtil._dynamicSubscriptLookup(tabK, sc) === nothing
  end

  @testset "an array an initial algorithm assigns" begin
    local (_, sol) = simulate("InitialArray", 0.1)
    @test string(sol.retcode) == "Success"
    @test [value(sol, "x[$k]", 1) for k in 1:3] ≈ [1.0, 2.0, 3.0]
  end

  @testset "a higher derivative's auxiliary beside a variable named alike" begin
    #= der(der(x)) lowered through der.x (`IR.auxiliaryDerivativeCref`), `der_x` in the code as the variable der_x is:
       two variables, x = cos(t), der_x = 2 cos(t) =#
    local (_, sol) = simulate("DerivativeAuxiliaryName", 1.0)
    @test string(sol.retcode) == "Success"
    @test isapprox(OMBackend.getVariableValues(sol, "x")[end], cos(1.0); rtol = 1e-3)
    @test isapprox(OMBackend.getVariableValues(sol, "der_x")[end], 2 * cos(1.0); rtol = 1e-3)
  end

  @testset "a function over an array of records" begin
    local (_, sol) = simulate("RecordArrayFields", 1.0)
    @test string(sol.retcode) == "Success"
    @test value(sol, "y", 1) ≈ 14.0
    @test isapprox(OMBackend.getVariableValues(sol, "x")[end], 14.0; atol = 1e-6)
  end
end
