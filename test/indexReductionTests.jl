#= OMBackend's index reduction (Passes.IndexReduction) step by step on small models
   (Models/IndexReduction.mo): the plan's outcome, differentiations and states (explain), the
   choice of states, the folding of known values, symbolic partial derivatives, the commit to
   OMJLIR, the derivative unknowns solved for, a simulation, and MTK's path with the switch off. The OM.jl suite checks such models'
   values against analytic solutions and OpenModelica's states. =#
const INDEX_REDUCTION_FILE = joinpath(@__DIR__, "Models", "IndexReduction.mo")
const _INDEX_REDUCTION_SCODE = Ref{Any}(nothing)
const _IRd = OMBackend.Passes.IndexReduction
const _Ev = OMBackend.Passes.Evaluation
const _IRM = OMBackend.OMJLIR

"The flat model of `IndexReduction.<name>` and its functions."
function _irFlat(name::String)::Tuple
  _INDEX_REDUCTION_SCODE[] === nothing &&
    (_INDEX_REDUCTION_SCODE[] = OMFrontend.translateToSCode(OMFrontend.parseFile(INDEX_REDUCTION_FILE)))
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("IndexReduction." * name, _INDEX_REDUCTION_SCODE[])
  return (fm, OMFrontend.cacheToFunctionList(cache))
end

"The OMJLIR model of `IndexReduction.<name>` before index reduction (the passes through the last)."
function _irUnreduced(name::String)::_IRM.Model
  local (fm, functions) = _irFlat(name)
  local model = OMBackend.LowerFlatModel.lowerFlatModel(fm; functions = functions)
  return OMBackend.Passes.runPasses!(model; through = first(last(OMBackend.Passes.PIPELINE)))
end

"The flat model of `IndexReduction.<name>` with its arrays kept (`scalarize = false`) and its functions."
function _irArrayFlat(name::String)::Tuple
  _INDEX_REDUCTION_SCODE[] === nothing &&
    (_INDEX_REDUCTION_SCODE[] = OMFrontend.translateToSCode(OMFrontend.parseFile(INDEX_REDUCTION_FILE)))
  local (fm, cache) = OMFrontend.instantiateSCodeToFM("IndexReduction." * name, _INDEX_REDUCTION_SCODE[]; scalarize = false)
  return (fm, OMFrontend.cacheToFunctionList(cache))
end

"The OMJLIR model of `IndexReduction.<name>` with its arrays kept, before index reduction (`runPassesKeepingArrays!`, as translate)."
function _irArrayUnreduced(name::String)::_IRM.Model
  local (fm, functions) = _irArrayFlat(name)
  local model = OMBackend.LowerFlatModel.lowerFlatModel(fm; functions = functions)
  return OMBackend.Passes.runPassesKeepingArrays!(model)
end

"The values of `names` at the end of a simulation of `IndexReduction.<name>` to `stop`, scalarized or with its arrays kept."
function _irArraySimulate(name::String, names::Vector{String}, stop::Float64; kept::Bool)::Vector{Float64}
  local (fm, functions) = kept ? _irArrayFlat(name) : _irFlat(name)
  OMBackend.translate(fm; functionList = functions, scalarized = !kept)
  local sol = OMBackend.simulateModel(OMBackend.canonicalName("IndexReduction." * name); tspan = (0.0, stop))
  @test sol.retcode == ReturnCode.Success
  return Float64[OMBackend.getVariableValues(sol, n)[end] for n in names]
end

"explain of `IndexReduction.<name>`."
function _irExplain(name::String)::_IRd.Explanation
  local (fm, functions) = _irFlat(name)
  return OMBackend.explainIndexReduction(fm; functionList = functions)
end

_irStates(name::String)::Vector{String} = sort(String.(_irExplain(name).states))

"The scalar equation of `model`'s partition whose left-hand side is the variable `name`."
_irEquationOf(eqs::Vector{_IRM.Equation}, name::Symbol)::_IRM.SCALAR_EQUATION =
  only(filter(eq -> eq isa _IRM.SCALAR_EQUATION && eq.lhs isa _IRM.CREF && eq.lhs.cref.name === name, eqs))

@testset "Index reduction" begin
  @testset "explain: outcome, differentiations, states" begin
    @test _irExplain("Index1").outcome === :index1
    #= as written (the aliases not substituted): phi1 = phi2 twice (positions to accelerations), the velocity
       definitions once =#
    local e = withenv(() -> _irExplain("RigidInertias"), "OMBACKEND_ALIAS_SUBSTITUTION" => "false")
    @test e.outcome === :reduced
    @test sort(last.(e.differentiated)) == [1, 1, 2]
    @test any(p -> occursin("phi1 = phi2", first(p)) && last(p) == 2, e.differentiated)
    @test sort(String.(e.states)) == ["phi1", "w1"]
    local report = sprint(show, MIME("text/plain"), e)
    @test occursin("index 3, 2 states", report)
    @test occursin("states: phi1, w1", report)
  end

  @testset "aliases read through one representative (OMBACKEND_ALIAS_SUBSTITUTION)" begin
    #= phi2 = phi1 substituted: der(phi1) = w2 with der(phi1) = w1, index 2 (the velocity definitions once); phi2
       algebraic, its equation kept; the same states =#
    local e = withenv(() -> _irExplain("RigidInertias"), "OMBACKEND_ALIAS_SUBSTITUTION" => "true")
    @test sort(last.(e.differentiated)) == [1, 1]
    @test sort(String.(e.states)) == ["phi1", "w1"]
    local (fm, functions) = _irFlat("RigidInertias")
    local model = withenv(() -> OMBackend.omjlirModel(fm; functionList = functions), "OMBACKEND_ALIAS_SUBSTITUTION" => "true")
    @test model.variables[:phi2].kind isa _IRM.ALGEBRAIC
    @test !haskey(model.variables, Symbol("\$DER.phi2"))
    @test any(eq -> eq isa _IRM.SCALAR_EQUATION && eq.lhs isa _IRM.CREF && eq.rhs isa _IRM.CREF &&
                    Set([eq.lhs.cref.name, eq.rhs.cref.name]) == Set([:phi1, :phi2]), only(model.continuous).equations)
    #= a capacitor loop whose only constraint is the alias v1 = v2: planned as written (index 1 once substituted) =#
    @test withenv(() -> _irStates("CapacitorLoop"), "OMBACKEND_ALIAS_SUBSTITUTION" => "true") == ["v1"]
    #= the model left as written where nothing is reduced =#
    local (fm1, functions1) = _irFlat("Index1")
    local index1 = OMBackend.omjlirModel(fm1; functionList = functions1)
    @test index1.variables[:x].kind isa _IRM.STATE
  end

  @testset "the choice of states" begin
    #= StateSelect.always kept; one capacitor of a loop; the variable inside the nonlinear
       function (phi of x = L*sin(phi)); of two aliases the fixed one; an always variable the
       model does not differentiate made a state; a constraint through a function's derivative =#
    @test _irStates("RigidInertiasAlways") == ["phi2", "w2"]
    @test _irStates("CapacitorLoop") == ["v1"]
    @test _irStates("PendulumAngle") == ["phi", "phid"]
    @test _irStates("FixedAlias") == ["phi", "w"]
    @test _irStates("AlwaysUndifferentiated") == ["phi1", "w"]
    @test _irStates("FunctionConstraint") == ["x1"]
  end

  @testset "a generic point that keeps explicit relations: two angles of one rotor (MSL ActuatorWithNoise)" begin
    #= a current reference rotated by the controller's angle, back by the machine's: iR[1] = 0 only where the two
       agree, so a choice regular at a point that moves them apart (each variable's own offset) is singular on the
       trajectory; the propagated point solves each for its equation =#
    #= the rotor's angle, not the machine's (gamma = p * phi): its derivative the model's own equations read =#
    @test _irStates("RotatedCurrents") == ["iR[2]", "phi", "w"]
    local model = _irUnreduced("RotatedCurrents")
    local partition = model.continuous[1]
    local p = _IRd.plan(model, partition)
    local values = _Ev.StartValues(model.variables, model.functions)
    local functions = Dict{Symbol, _IRM.ModelicaFunction}(f.name => f for f in model.functions)
    local point = _IRd.propagatedValues(p.structure, _Ev.withKnownValues(model, partition), p.differentiated, functions,
                                        values, _IRd.naturalDerivatives(p.structure))
    local derivative = (n::Symbol) -> Symbol("\$DER.", n)
    @test point(:gamma) == point(:gammaC) ≈ 2 * point(:phi)
    @test point(derivative(:gamma)) == point(derivative(:gammaC)) ≈ 2 * point(:w)
    @test abs(point(Symbol("iR[1]"))) < 1e-12
    @test point(Symbol("iR[2]")) ≈ point(:iqRef)
    local plain = _Ev.genericValues(values)
    @test !isapprox(plain(:gamma), plain(:gammaC))
  end

  @testset "start values through functions run on numbers, and parameters an equation defines" begin
    #= R0 = planarFrame(a) becomes initial equations (a binding that reads a parameter); Q0 = frameQuaternion(R0, guess)
       runs an algorithm of array locals, for and while loops and ifs, its sign flipped by the guess (as MSL
       Frames.to_Q for a body's Q_start, MSL GyroscopicEffects) =#
    local model = _irUnreduced("FunctionStart")
    local partition = model.continuous[1]
    local values = _Ev.StartValues(model.variables, model.functions;
                                    equations = vcat(partition.equations, partition.initialEquations))
    @test values(Symbol("R0.T[1,2]")) ≈ sin(0.6)
    @test values(Symbol("R0.T[2,1]")) ≈ -sin(0.6)
    @test values(Symbol("Q0[3]")) ≈ -sin(0.3)
    @test values(Symbol("Q0[4]")) ≈ -cos(0.3)
    @test values(:q) ≈ -sin(0.3)
    #= a function argument (function scaled(k = a + 1.4)), a reduction, an array constructor of two iterators, a
       column assigned by a slice and read by a slice and size, a record's field defaults, a tuple assignment =#
    @test values(:s1) ≈ 12
    @test values(:t1) ≈ 15.25
    #= without the equations, R0 reads its default 0: the quaternion of the zero matrix =#
    @test _Ev.StartValues(model.variables, model.functions)(Symbol("Q0[4]")) ≈ -0.5
    #= flat values: a matrix product, a record field, an element of an array of a call =#
    local run = _Ev.Interpretation(values)
    local real = (x::Float64) -> _IRM.REAL(x)
    local matrix = (rows::Vector{Vector{Float64}}) ->
      _IRM.ARRAY(_IRM.Expression[_IRM.ARRAY(_IRM.Expression[real(x) for x in r], _IRM.T_ARRAY(_IRM.T_REAL(), _IRM.Dimension[_IRM.DIM_SIZE(length(r))]))
                                 for r in rows], _IRM.T_ARRAY(_IRM.T_REAL(), _IRM.Dimension[_IRM.DIM_SIZE(length(rows)), _IRM.DIM_SIZE(length(rows[1]))]))
    local product = _IRM.BINARY(matrix([[1.0, 2.0], [3.0, 4.0]]), _IRM.MATRIX_PRODUCT, matrix([[0.0, 1.0], [1.0, 0.0]]),
                                _IRM.T_ARRAY(_IRM.T_REAL(), _IRM.Dimension[_IRM.DIM_SIZE(2), _IRM.DIM_SIZE(2)]))
    @test _Ev.toArray(_Ev.valueOf(product, run)) == [2.0 1.0; 4.0 3.0]
  end

  @testset "a constraint through a record a constructor function builds (field bindings, as MSL Complex)" begin
    #= makePhasor's output is defined by its field bindings alone (Complex's 'constructor'.fromReal), scalePhasor
       calls it in its body (Complex's '*'.multiply): both differentiated through their inlined bodies =#
    local e = _irExplain("ConstructorConstraint")
    @test e.outcome === :reduced
    @test sort(String.(e.states)) == ["x1"]
    local (fmC, functionsC) = _irFlat("ConstructorConstraint")
    OMBackend.translate(fmC; functionList = functionsC)
    local solC = OMBackend.simulateModel(OMBackend.canonicalName("IndexReduction.ConstructorConstraint"); tspan = (0.0, 1.0),
                                         reltol = 1e-8, abstol = 1e-8)
    @test solC(1.0; idxs = :x1) ≈ sin(1) / 4 rtol = 1e-6
    @test solC(1.0; idxs = :x2) ≈ sin(1) / 4 rtol = 1e-6
  end

  @testset "an inlined function's derivative: a function, the record a call computes evaluated once (as IF97's rho_ph)" begin
    #= wrapped(x) := fromAux(x, auxOf(x, phase)), Inline = true; fromAux's derivative excludes the record: wrapped gets
       the derivative wrapped__inlineDer (LowerFlatModel.withInlineDerivatives), and the index reduction differentiates
       x2 = wrapped(x1) by it. Inlined, the derivative equation passed the call auxOf(x1, 0) to fromAuxDer, which the code
       generation evaluates once per field of the record (MSL MeasuringTemperature 5 times slower). =#
    local (fm, functions) = _irFlat("InlinedRecordDerivative")
    local irFunctions = OMBackend.LowerFlatModel.lowerFunctions(functions)
    local w = only(filter(f -> f.name === Symbol("IndexReduction.wrapped"), irFunctions))
    @test [(a.derivative, a.order, a.excludedInputs) for a in w.derivatives] == [(Symbol("IndexReduction.wrapped__inlineDer"), 1, Int[])]
    local d = only(filter(f -> f.name === Symbol("IndexReduction.wrapped__inlineDer"), irFunctions))
    @test [v.name for v in d.inputs] == [:x, :phase, :x__der]
    @test _irStates("InlinedRecordDerivative") == ["x1"]
    #= the structure's inliner: an annotated function inlines where its inlined form reads no call without a derivative
       (cubic: MultiBody's resolve1 as well, whose calls kept made EngineV6's Jacobian entries nested derivative calls),
       and stays a call where it does (wrapped through auxOf, as IF97's rho_props_ph through waterBaseProp_ph) =#
    local inlinedRhs = name -> begin
      local m = _irUnreduced(name)
      local fs = Dict{Symbol, _IRM.ModelicaFunction}(f.name => f for f in m.functions)
      _IRd.inlineCalls(_irEquationOf(m.continuous[1].equations, :x2).rhs, fs)
    end
    @test !_IRM.anyExp(e -> e isa _IRM.CALL && !e.attributes.builtin, inlinedRhs("FunctionConstraint"))
    local kept = inlinedRhs("InlinedRecordDerivative")
    @test kept isa _IRM.CALL && kept.name === Symbol("IndexReduction.wrapped")
    #= solving for derivative unknowns: a derivative function is linear in its derivative inputs, so its partial by
       one is the call with that input's derivative (`derivativeFunctionCall`; R134a's rho_ph_der by der(p)) =#
    local fs = Dict{Symbol, _IRM.ModelicaFunction}(f.name => f for f in _irUnreduced("InlinedRecordDerivative").functions)
    local cref = n -> _IRM.CREF(_IRM.ComponentRef(n), _IRM.T_REAL())
    local derCall = _IRM.CALL(Symbol("IndexReduction.wrapped__inlineDer"), _IRM.Expression[cref(:x1), _IRM.INTEGER(0), cref(:dx)],
                              _IRM.T_REAL(), _IRM.FUNCTION_CALL)
    local partial = _IRd.partialDerivative(_IRM.SCALAR_EQUATION(cref(:y), derCall, _IRM.T_REAL(), _IRM.NO_SOURCE, _IRM.DEFAULT_ATTRIBUTES),
                                           :dx, fs; derivativeCalls = true)
    @test partial isa _IRM.Expression && _IRM.anyExp(e -> e isa _IRM.CALL && e.name === Symbol("IndexReduction.wrapped__inlineDer"), partial)
    local eqs = OMBackend.omjlirModel(fm; functionList = functions).continuous[1].equations
    local calls = pred -> any(eq -> eq isa _IRM.SCALAR_EQUATION && (_IRM.anyExp(pred, eq.lhs) || _IRM.anyExp(pred, eq.rhs)), eqs)
    @test calls(x -> x isa _IRM.CALL && x.name === Symbol("IndexReduction.wrapped__inlineDer"))
    @test !calls(x -> x isa _IRM.CALL && x.name === Symbol("IndexReduction.fromAuxDer"))
    OMBackend.translate(fm; functionList = functions)
    local sol = OMBackend.simulateModel(OMBackend.canonicalName("IndexReduction.InlinedRecordDerivative"); tspan = (0.0, 1.0),
                                        reltol = 1e-8, abstol = 1e-8)
    @test sol.retcode == ReturnCode.Success
    #= 2 x1 + x1^2 / 3 = t =#
    @test OMBackend.getVariableValues(sol, "x1")[end] ≈ 3 * (sqrt(4 / 3) - 1) rtol = 1e-6
  end

  @testset "StateSelect.prefer an artificial state (OMBACKEND_PREFER_STATES)" begin
    #= as OpenModelica (BackendDAECreate.lowerVarkind): x gets a derivative, Pantelides differentiates m = 2 x + x^3, and
       x (prefer) is the state; off, the model is of index 1 with m the state =#
    local on = withenv(() -> _irExplain("PreferredState"), "OMBACKEND_PREFER_STATES" => "true")
    @test on.outcome === :reduced
    @test sort(String.(on.states)) == ["x"]
    @test withenv(() -> _irExplain("PreferredState"), "OMBACKEND_PREFER_STATES" => "false").outcome === :index1
    local (fm, functions) = _irFlat("PreferredState")
    local sol = withenv("OMBACKEND_PREFER_STATES" => "true") do
      OMBackend.translate(fm; functionList = functions)
      OMBackend.simulateModel(OMBackend.canonicalName("IndexReduction.PreferredState"); tspan = (0.0, 1.0), reltol = 1e-8, abstol = 1e-8)
    end
    @test sol.retcode == ReturnCode.Success
    @test OMBackend.getVariableValues(sol, "m")[end] ≈ 2 * exp(-1) rtol = 1e-6
    @test OMBackend.getVariableValues(sol, "x")[end] ≈ 0.3469902420410083 rtol = 1e-6
  end

  @testset "a function without derivative annotation, its body run symbolically" begin
    #= signedSquare's local and if-statement: x2 = if x1 < 0 then -x1^2 else x1^2, differentiated =#
    @test _irStates("IfFunctionConstraint") == ["x1"]
    local (fm, functions) = _irFlat("IfFunctionConstraint")
    OMBackend.translate(fm; functionList = functions)
    local sol = OMBackend.simulateModel(OMBackend.canonicalName("IndexReduction.IfFunctionConstraint"); tspan = (0.0, 1.0),
                                        reltol = 1e-8, abstol = 1e-8)
    @test sol.retcode == ReturnCode.Success
    #= x1 + x1^2 = t + 2 =#
    @test OMBackend.getVariableValues(sol, "x1")[end] ≈ (sqrt(13) - 1) / 2 rtol = 1e-6
  end

  @testset "known values folded" begin
    local model = _irUnreduced("KnownValues")
    local folded = _Ev.withKnownValues(model, model.continuous[1])
    local zero = e -> e isa Union{_IRM.REAL, _IRM.INTEGER} && e.value == 0
    #= k = 0: x = k * y is x = 0; c = 0 kept in its own equation; both propagate into z = c * y + x =#
    @test zero(_irEquationOf(folded.equations, :x).rhs)
    @test zero(_irEquationOf(folded.equations, :c).rhs)
    @test zero(_irEquationOf(folded.equations, :z).rhs)
    #= the model is not changed =#
    @test !zero(_irEquationOf(model.continuous[1].equations, :x).rhs)
  end

  @testset "an algorithm's record overwritten field: one equation per variable" begin
    #= p := initPair(k); p.b := p.a + 1: p.b's equation is the later assignment's only (two made the
       system structurally singular, MSL Spice3's Graetz) =#
    local eqs = _irUnreduced("AlgorithmRecordOverwrite").continuous[1].equations
    local lefts = [eq.lhs.cref.name for eq in eqs if eq isa _IRM.SCALAR_EQUATION && eq.lhs isa _IRM.CREF]
    @test count(==(Symbol("p.b")), lefts) == 1
    @test count(==(Symbol("p.a")), lefts) == 1
    @test occursin("p.a + 1", string(_irEquationOf(eqs, Symbol("p.b")).rhs))
  end

  @testset "symbolic partial derivatives" begin
    local model = _irUnreduced("Partials")
    local values = _Ev.StartValues(model.variables, model.functions)
    local functions = Dict{Symbol, _IRM.ModelicaFunction}(f.name => f for f in model.functions)
    local partial = (name, by) -> _Ev.evaluate(_IRd.partialDerivative(_irEquationOf(model.continuous[1].equations, name), by, functions), values)
    #= the residual a - (sin(x) y + x^2 / y), b - (exp(x) log(y) + sqrt(y) x - cos(x y)), at x = 0.5, y = 2 =#
    @test partial(:a, :x) ≈ -(cos(0.5) * 2 + 2 * 0.5 / 2)
    @test partial(:b, :x) ≈ -(exp(0.5) * log(2.0) + sqrt(2.0) + sin(1.0) * 2)
    #= c - (atan2(x, y) + max(x, y) + semiLinear(x, 2, 3) + x / y + x^y): y > x, x >= 0 =#
    @test partial(:c, :x) ≈ -(2 / 4.25 + 0 + 2 + 1 / 2 + 2 * 0.5)
    @test partial(:c, :y) ≈ -(-0.5 / 4.25 + 1 + 0 - 0.5 / 4 + 0.25 * log(0.5))
  end

  @testset "commit: OMJLIR's kinds and the state derivatives' link" begin
    local (fm, functions) = _irFlat("RigidInertias")
    local model = withenv(() -> OMBackend.omjlirModel(fm; functionList = functions), "OMBACKEND_ALIAS_SUBSTITUTION" => "false")
    local kind = name -> model.variables[name].kind
    @test kind(:phi1) isa _IRM.STATE && kind(:w1) isa _IRM.STATE
    @test kind(:phi2) isa _IRM.DUMMY_STATE && kind(:w2) isa _IRM.DUMMY_STATE
    #= each state's derivative an algebraic unknown, linked by $STATEDER.x = $DER.x =#
    @test kind(:phi1).derivative === Symbol("\$STATEDER.phi1")
    @test kind(Symbol("\$DER.phi1")) isa _IRM.ALGEBRAIC
  end

  @testset "derivative unknowns: a piecewise coefficient 0 in a branch is not solved through" begin
    #= MSL Media's smooth state transition (1 in one branch, 0 in the other) is not; R134a's rho_ph_der, one expression
       in the two-phase region and another outside, is (its volumes' der(p) jumped at the dew line otherwise) =#
    local x = _IRM.CREF(_IRM.ComponentRef(:x), _IRM.T_REAL())
    local cond = _IRM.CREF(_IRM.ComponentRef(:c), _IRM.T_BOOLEAN())
    @test _IRd.isPiecewise(_IRM.IF_EXPRESSION(cond, _IRM.REAL(1.0), _IRM.REAL(0.0), _IRM.T_REAL()))
    @test !_IRd.isPiecewise(_IRM.IF_EXPRESSION(cond, x, _IRM.REAL(2.0), _IRM.T_REAL()))
  end

  @testset "derivative unknowns solved for and substituted" begin
    #= (1 + v1^2) der(v1) = i1, (C2 + v2^2) der(v2) = i2, v1 = v2: der(v1) solved through the capacitance
       (the alias determines der(v2)), substituted into the state's link =#
    local (fm, functions) = _irFlat("NonlinearCapacitors")
    local reads = (e, name) -> any(c -> c.name === name, _IRM.crefsOf(e))
    local readsDerivative = e -> any(c -> startswith(String(c.name), "\$DER."), _IRM.crefsOf(e))
    local eqs = OMBackend.omjlirModel(fm; functionList = functions).continuous[1].equations
    local link = _irEquationOf(eqs, Symbol("\$STATEDER.v1"))
    @test !readsDerivative(link.rhs) && reads(link.rhs, :i1) && reads(link.rhs, :v1)
    @test !readsDerivative(_irEquationOf(eqs, Symbol("\$DER.v1")).rhs)
    local off = withenv("OMBACKEND_SOLVE_DERIVATIVES" => "false") do
      OMBackend.omjlirModel(fm; functionList = functions).continuous[1].equations
    end
    @test reads(_irEquationOf(off, Symbol("\$STATEDER.v1")).rhs, Symbol("\$DER.v1"))
    OMBackend.translate(fm; functionList = functions)
    local sol = OMBackend.simulateModel(OMBackend.canonicalName("IndexReduction.NonlinearCapacitors"); tspan = (0.0, 2.0),
                                        reltol = 1e-8, abstol = 1e-8)
    @test sol.retcode == ReturnCode.Success
    #= (3 + 2 v^2) v' = sin(t): 3 v + 2/3 v^3 = 1 - cos(t) =#
    local v = OMBackend.getVariableValues(sol, "v1")[end]
    @test 3v + 2 / 3 * v^3 ≈ 1 - cos(2.0) rtol = 1e-6
  end

  @testset "the derivatives' matching: through unknown coefficients only outside coupled blocks" begin
    local c = _IRM.CREF(_IRM.ComponentRef(:x, _IRM.T_REAL()), _IRM.T_REAL())
    local edge = (eq, constant, others) -> _IRd.DerivativeEdge(eq, c, constant, constant ? Inf : 1.0, others)
    #= a and b in two equations, each through a coefficient that reads unknowns: a coupled block, not solved =#
    local coupled = Dict(:a => [edge(1, false, [:b]), edge(2, false, [:b])], :b => [edge(1, false, [:a]), edge(2, false, [:a])])
    @test isempty(_IRd.matchDerivatives(coupled))
    #= b determined by a constant coefficient (equation 3): a through equation 1 =#
    local determined = Dict(:a => [edge(1, false, [:b])], :b => [edge(1, false, [:a]), edge(3, true, Symbol[])])
    local m = _IRd.matchDerivatives(determined)
    @test m[:b].equation == 3 && m[:a].equation == 1
  end

  @testset "review regressions: if-equation positions, recursion, array locals, Integer inputs, zero pivots" begin
    #= an if-equation's Boolean position after its rows: the else-branch's derivative by its full count =#
    local (fm, functions) = _irFlat("IfEquationPositions")
    OMBackend.translate(fm; functionList = functions)
    local sol = OMBackend.simulateModel(OMBackend.canonicalName("IndexReduction.IfEquationPositions"); tspan = (0.0, 1.0),
                                        reltol = 1e-8, abstol = 1e-8)
    @test sol.retcode == ReturnCode.Success
    @test OMBackend.getVariableValues(sol, "x1")[end] ≈ 1 / 6 + 1 / 8 rtol = 1e-6
    #= its row in the block of the choice: Jacobian entries by the branch the start values take, the numeric
       choice (an if-equation row sent it to the structural fallback: MSL SMEE_Generator's zero-sequence state) =#
    local mI = _irUnreduced("IfEquationPositions")
    local trace = _IRd.BlockTrace[]
    local plI = _IRd.plan(mI, mI.continuous[1]; trace)
    local block = only(trace)
    @test _IRd.numericChoice(plI.structure, block.rows, block.candidates, Dict{Int, Int}(c => 0 for c in block.candidates),
                             _Ev.withKnownValues(mI, mI.continuous[1]), plI.differentiated,
                             Dict{Symbol, _IRM.ModelicaFunction}(f.name => f for f in mI.functions),
                             _Ev.StartValues(mI.variables, mI.functions)) !== nothing
    #= a recursive function is not inlined into itself: its derivative is NotHandled (MTK's), no stack overflow =#
    local recursive = _irExplain("RecursiveFunctionConstraint")
    @test recursive.outcome === :notHandled && occursin("recursive", recursive.reason)
    #= an array local has no value in the symbolic run: not inlined, not read as a model variable =#
    @test _irExplain("ArrayLocalConstraint").outcome === :notHandled
    #= no derivative argument for an Integer input: powerNDer(x1, 2, der(x1)) =#
    local (fmN, functionsN) = _irFlat("IntegerInputDerivative")
    OMBackend.translate(fmN; functionList = functionsN)
    local solN = OMBackend.simulateModel(OMBackend.canonicalName("IndexReduction.IntegerInputDerivative"); tspan = (0.0, 1.0),
                                         reltol = 1e-8, abstol = 1e-8)
    @test solN.retcode == ReturnCode.Success
    @test OMBackend.getVariableValues(solN, "x1")[end] ≈ (sqrt(13) - 1) / 2 rtol = 1e-6
    #= v1 * der(v1) = i1 with v1 = 0 at the given start: der(v2) solved through (2 + v2^2), not through v1 =#
    local (fmZ, functionsZ) = _irFlat("ZeroStartCapacitance")
    local eqs = OMBackend.omjlirModel(fmZ; functionList = functionsZ).continuous[1].equations
    local link = _irEquationOf(eqs, Symbol("\$STATEDER.v1"))
    @test any(c -> c.name === :i2, _IRM.crefsOf(link.rhs)) && !any(c -> c.name === :i1, _IRM.crefsOf(link.rhs))
    #= der() of an expression by the chain rule: der(x1 + x2) reads both derivatives, x2 = 2 x1 differentiated =#
    local eDer = _irExplain("DerOfExpression")
    @test eDer.outcome === :reduced
    @test length(eDer.states) == 1
    local (fmD, functionsD) = _irFlat("DerOfExpression")
    OMBackend.translate(fmD; functionList = functionsD)
    local solDer = OMBackend.simulateModel(OMBackend.canonicalName("IndexReduction.DerOfExpression"); tspan = (0.0, 1.0),
                                           reltol = 1e-8, abstol = 1e-8)
    @test solDer(1.0; idxs = :x1) ≈ exp(-1 / 3) rtol = 1e-6
  end

  @testset "a model left to MTK or of index 1 is not changed" begin
    for name in ("RecursiveFunctionConstraint", "Index1")
      local model = _irUnreduced(name)
      local before = (string.(model.continuous[1].equations), [string(v.name, nameof(typeof(v.kind))) for v in model.variables])
      _IRd.reduceIndex!(model)
      @test (string.(model.continuous[1].equations), [string(v.name, nameof(typeof(v.kind))) for v in model.variables]) == before
    end
  end

  @testset "solved derivatives in a cycle stay unknowns; withIterator" begin
    local ref = n -> _IRM.CREF(_IRM.ComponentRef(n, _IRM.T_REAL()), _IRM.T_REAL())
    local sub = (a, b) -> _IRM.BINARY(a, _IRM.SUB, b, _IRM.T_REAL())
    #= a = f - b, b = a - g (a cycle), c = a + 1 (reads it), e = f: only e resolves =#
    local values = Dict{Symbol, _IRM.Expression}(:a => sub(ref(:f), ref(:b)), :b => sub(ref(:a), ref(:g)),
                                                 :c => _IRM.BINARY(ref(:a), _IRM.ADD, _IRM.REAL(1.0), _IRM.T_REAL()), :e => ref(:f))
    @test collect(keys(_IRd.resolvedValues(values))) == [:e]
    #= x[i] * i for i = 2: x[2] * 2; a nested constructor binding i again is left as it is =#
    local xi = _IRM.CREF(_IRM.ComponentRef(_IRM.CrefPart[_IRM.CrefPart(:x, _IRM.Subscript[_IRM.INDEX(ref(:i))], _IRM.T_REAL())]), _IRM.T_REAL())
    local body = _IRM.BINARY(xi, _IRM.MUL, ref(:i), _IRM.T_REAL())
    @test _IRM.toString(_IRd.withIterator(body, :i, 2)) == _IRM.toString(_IRM.BINARY(
      _IRM.CREF(_IRM.ComponentRef(_IRM.CrefPart[_IRM.CrefPart(:x, _IRM.Subscript[_IRM.INDEX(_IRM.INTEGER(2))], _IRM.T_REAL())]), _IRM.T_REAL()),
      _IRM.MUL, _IRM.INTEGER(2), _IRM.T_REAL()))
    local range = _IRM.RANGE(_IRM.INTEGER(1), nothing, _IRM.INTEGER(3), _IRM.T_ARRAY(_IRM.T_INTEGER(), _IRM.Dimension[_IRM.DIM_SIZE(3)]))
    local nested = _IRM.ARRAY_CONSTRUCTOR(ref(:i), _IRM.Iterator[_IRM.Iterator(:i, range)], _IRM.T_ARRAY(_IRM.T_REAL(), _IRM.Dimension[_IRM.DIM_SIZE(3)]))
    @test _IRM.toString(_IRd.withIterator(nested, :i, 2)) == _IRM.toString(nested)
  end

  @testset "dynamic state sets (OMBACKEND_DYNAMIC_STATE_SETS)" begin
    withenv("OMBACKEND_DYNAMIC_STATE_SETS" => "true") do
      #= the pendulum: one set of its positions with its velocities linked (der(x) = vx: one selection for both);
         the quaternion: 3 of 4. All candidates' variables dummy states, the states the sets' =#
      local e = _irExplain("PendulumCartesian")
      @test isempty(e.states)
      @test only(e.sets) == [:y, :x, :vy, :vx]
      @test e.setStates == [Symbol("\$STATESET1.x[1]"), Symbol("\$STATESET1.dx[1]")]
      @test occursin("index 3, 2 states", sprint(show, MIME("text/plain"), e))
      @test sort(String.(only(_irExplain("QuaternionAttitude").sets))) == ["Q[1]", "Q[2]", "Q[3]", "Q[4]"]
      #= the set's Jacobian, once the dummies of constant pivots are eliminated (der(\$DER.x) by der(\$DER.x) = der(vx)):
         2 y, 2 x up to a factor; the start values make x the dummy (|x| > |y|) =#
      local model = _irUnreduced("PendulumCartesian")
      local p = _IRd.plan(model, model.continuous[1])
      local set = only(p.sets)
      local s = p.structure
      @test [s.variables[s.antiderivative[c]] for c in set.candidates] == [:y, :x]
      @test [s.variables[s.antiderivative[c]] for c in only(set.linked)] == [:vy, :vx]
      @test s.variables[s.antiderivative[only(set.chosen)]] === :x
      local values = _Ev.StartValues(model.variables, model.functions)
      local J = Float64[_Ev.evaluate(x, values) for x in vec(set.jacobian)]
      @test J[1] / J[2] ≈ -sin(pi / 6) / cos(pi / 6)
      #= the commit: x = A * {y, x}, dx = A * {vy, vx}, A the start's selection (y) =#
      local (fm, functions) = _irFlat("PendulumCartesian")
      local reduced = OMBackend.omjlirModel(fm; functionList = functions)
      local kind = name -> reduced.variables[Symbol(name)].kind
      @test all(v -> kind(v) isa _IRM.DUMMY_STATE, ("x", "y", "vx", "vy"))
      @test kind("\$STATESET1.x[1]") isa _IRM.STATE && kind("\$STATESET1.dx[1]") isa _IRM.STATE
      @test kind("\$STATESET1.A[1,1]") isa _IRM.DISCRETE
      @test [reduced.variables[Symbol("\$STATESET1.A[1,$(i)]")].attributes.start.value for i in 1:2] == [1, 0]
      #= the ranking: a variable only index reduction differentiates and without a StateSelect is never; an implicit
         alias no higher than its set's StateSelect =#
      @test _irStates("AlgebraicNever") == ["x"]
      @test _irStates("ImplicitAlias") == ["q"]
    end
    #= switch off: the choice at the start values, the ranking as before =#
    withenv("OMBACKEND_DYNAMIC_STATE_SETS" => "false") do
      @test _irStates("PendulumCartesian") == ["vy", "y"]
      @test _irStates("AlgebraicNever") == ["s"]
      @test _irStates("ImplicitAlias") == ["p"]
    end
  end

  @testset "arrays kept: reduced in array form (OMBACKEND_ARRAY_INDEX_REDUCTION)" begin
    #= rigid pairs as array equations: the arrays phi1 and w1 states, phi2 and w2 dummy states, the derivatives of
       the derivatives whole arrays, the states' derivatives an array linked by a for-equation =#
    local m = _irArrayUnreduced("InertiaArray")
    @test _IRd.reduceIndexArrays!(m)
    local kind = name -> m.variables[Symbol(name)].kind
    @test kind("phi1") isa _IRM.STATE && kind("w1") isa _IRM.STATE && m.variables[:phi1].ty isa _IRM.T_ARRAY
    @test kind("phi2") isa _IRM.DUMMY_STATE && kind("w2") isa _IRM.DUMMY_STATE
    @test m.variables[Symbol("\$DER.\$DER.phi1")].ty isa _IRM.T_ARRAY
    @test kind("\$STATEDER.phi1") isa _IRM.STATE_DER
    #= pendulums in a for-equation: the constraint differentiated twice in array form, the iterator kept; the dynamic
       state sets of the elements one family over them (its selection matrix A[i, j, c], its when-equation in the
       for-equation); the start values choose differently per element (x the dummy of the first, y of the others) =#
    m = _irArrayUnreduced("PendulumArray")
    @test withenv(() -> _IRd.reduceIndexArrays!(m), "OMBACKEND_DYNAMIC_STATE_SETS" => "true")
    local fors = filter(eq -> eq isa _IRM.FOR_EQUATION, m.continuous[1].equations)
    @test length(fors) == 4
    @test any(eq -> occursin("\$DER.\$DER.x[i]", string(eq)), fors)
    @test count(eq -> any(b -> b isa _IRM.WHEN_EQUATION, eq.body), fors) == 1
    @test m.variables[Symbol("\$STATESET1.A")].ty == _IRM.T_ARRAY(_IRM.T_INTEGER(), _IRM.Dimension[_IRM.DIM_SIZE(3), _IRM.DIM_SIZE(1), _IRM.DIM_SIZE(2)])
    @test all(v -> m.variables[v].kind isa _IRM.DUMMY_STATE, (:x, :y, :vx, :vy))
    #= its scalarized form: the sets' states and selections per element =#
    local scalar = _IRd.scalarizeArrays(m).model
    @test scalar.variables[Symbol("\$STATESET1.x[2,1]")].kind isa _IRM.STATE
    local aStart = (i, c) -> scalar.variables[Symbol("\$STATESET1.A[$(i),1,$(c)]")].attributes.start.value
    @test [aStart(i, 1) for i in 1:3] == [1, 0, 0]
    #= a differentiated body that reads a parameter: in array form the parameter stays symbolic =#
    m = _irArrayUnreduced("AngleArray")
    @test _IRd.reduceIndexArrays!(m)
    @test any(eq -> occursin("\$DER.x[i] = (L * (cos(phi[i]) * \$DER.phi[i]))", string(eq)), m.continuous[1].equations)
    #= a boundary element differs: refused (slicing), the model unchanged =#
    m = _irArrayUnreduced("BoundaryArray")
    local before = (length(m.continuous[1].equations), [string(v.kind) for v in m.variables])
    @test_throws _IRd.NotHandled _IRd.reduceIndexArrays!(m)
    @test (length(m.continuous[1].equations), [string(v.kind) for v in m.variables]) == before
    #= translated with the arrays kept: the results of the scalarized model =#
    withenv("OMBACKEND_ARRAY_INDEX_REDUCTION" => "true") do
      #= InertiaArray, AngleArray: states; the pendulums: a family of dynamic state sets, over a vector (PendulumArray),
         a matrix (PendulumGrid), component arrays (ComponentPendulums: p[i].x), with pivots that read a parameter
         array (EllipseArray: a table by element), beside a set of scalars (MixedPendulums) =#
      for (name, names, stop) in (("InertiaArray", ["phi1[1]", "phi2[4]", "w2[3]", "tau[2]"], 2.0),
                                  ("AngleArray", ["phi[1]", "x[2]", "vx[3]"], 3.0),
                                  ("PendulumArray", ["x[1]", "y[2]", "vx[3]", "F[1]"], 3.0),
                                  ("PendulumGrid", ["x[1][2]", "y[2][1]", "F[2][2]"], 3.0),
                                  ("ComponentPendulums", ["p[1].x", "p[2].y", "p[3].F"], 3.0),
                                  ("EllipseArray", ["x[1]", "y[2]", "F[3]"], 3.0),
                                  ("MixedPendulums", ["single.x", "x[1]", "y[2]"], 3.0))
        local scalarized = _irArraySimulate(name, names, stop; kept = false)
        local kept = _irArraySimulate(name, names, stop; kept = true)
        @test OMBackend.canonicalName("IndexReduction." * name) in OMBackend.ARRAY_REDUCED_MODELS
        @test isapprox(kept, scalarized; atol = 1e-8, rtol = 1e-8)
      end
      #= `each start = sum(m)`: every element's start the sum, not m's elements =#
      @test _irArraySimulate("EachStartArray", ["phi1[1]", "phi1[3]", "phi2[2]"], 1.0; kept = true) ≈ fill(0.6, 3)
      @test OMBackend.canonicalName("IndexReduction.EachStartArray") in OMBackend.ARRAY_REDUCED_MODELS
      local kept = _irArraySimulate("BoundaryArray", ["x[1]", "x[3]"], 1.0; kept = true)
      @test !(OMBackend.canonicalName("IndexReduction.BoundaryArray") in OMBackend.ARRAY_REDUCED_MODELS)
      @test kept ≈ [exp(-1), 0.5 * exp(-1)] rtol = 1e-4
    end
    #= switch off: the frontend scalarizes =#
    withenv(() -> _irArraySimulate("InertiaArray", ["phi1[1]"], 1.0; kept = true), "OMBACKEND_ARRAY_INDEX_REDUCTION" => "false")
    @test !(OMBackend.canonicalName("IndexReduction.InertiaArray") in OMBackend.ARRAY_REDUCED_MODELS)
  end

  @testset "simulation" begin
    local (fm, functions) = _irFlat("RigidInertias")
    OMBackend.translate(fm; functionList = functions)
    local sol = OMBackend.simulateModel(OMBackend.canonicalName("IndexReduction.RigidInertias"); tspan = (0.0, 5.0))
    @test sol.retcode == ReturnCode.Success
    #= (J1 + J2) phi'' = sin(t): phi = (t - sin(t)) / 3 =#
    @test isapprox(OMBackend.getVariableValues(sol, "phi1")[end], (5 - sin(5)) / 3; rtol = 1e-3)
  end

  @testset "switch off: MTK's index reduction" begin
    local (fm, functions) = _irFlat("RigidInertias")
    local model = withenv("OMBACKEND_INDEX_REDUCTION" => "false") do
      OMBackend.omjlirModel(fm; functionList = functions)
    end
    @test model.variables[:phi2].kind isa _IRM.STATE
  end
end
