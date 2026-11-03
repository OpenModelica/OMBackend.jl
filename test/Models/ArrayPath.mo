package ArrayPath
  model Decay
    parameter Integer n = 1000;
    parameter Real k[n] = {i for i in 1:n};
    Real x[n](each start = 1, each fixed = true);
  equation
    for i in 1:n loop
      der(x[i]) = -k[i] * x[i];
    end for;
  end Decay;

  connector HeatPort
    Real T;
    flow Real Q;
  end HeatPort;

  model Cap
    parameter Real C = 1;
    parameter Real T0 = 0;
    HeatPort p;
    Real T(start = T0, fixed = true);
  equation
    T = p.T;
    C * der(T) = p.Q;
  end Cap;

  model Cond
    parameter Real G = 1;
    HeatPort a, b;
  equation
    a.Q + b.Q = 0;
    a.Q = G * (a.T - b.T);
  end Cond;

  model Chain
    parameter Integer n = 10;
    Cap c[n](T0 = {if i == 1 then 1.0 else 0.0 for i in 1:n});
    Cond g[n - 1];
  equation
    for i in 1:n - 1 loop
      connect(c[i].p, g[i].a);
      connect(g[i].b, c[i + 1].p);
    end for;
  end Chain;

  model Chain1000
    extends Chain(n = 1000);
  end Chain1000;

  model IntBoundRealArray "a Real array bound to Integer parameters (its rule gives Integers)"
    parameter Integer m = 1;
    parameter Integer n2 = 2;
    parameter Real k[2] = {m, n2};
    Real x(start = 0, fixed = true);
  equation
    der(x) = k[1] + k[2];
  end IntBoundRealArray;
  model ParamNameClash "a component's parameter c.k and a top-level c_k: one symbol P_c_k for both before (omc x(1) = 26)"
    model C
      parameter Real k = 1;
      Real y;
    equation
      y = k;
    end C;
    parameter Real c_k = 2;
    C c(k = 3 * c_k);
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    der(x) = c.y + 10 * c_k;
  end ParamNameClash;
  model ParamReference "parameters bound to references of others (c(k = gain), p2 = p1, e1 = a[2]): overrides reach them (omc x(1) = 5232)"
    model C
      parameter Real k = 1;
      Real y;
    equation
      y = k;
    end C;
    parameter Real gain = 2;
    parameter Real p1 = 3;
    parameter Real p2 = p1;
    parameter Real a[2] = {4, 5};
    parameter Real e1 = a[2];
    C c(k = gain);
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    der(x) = c.y + 10 * p2 + 100 * gain + 1000 * e1;
  end ParamReference;
  model DerCoefficient "a parameter coefficient of der(): c * der(x) = 1 (x(1) = 1 / c)"
    parameter Real c = 2;
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    c * der(x) = 1;
  end DerCoefficient;
  model StartOfParameters "starts that read parameters, one through a reference (y(1) = w(1) = p1 + 1)"
    parameter Real p1 = 3;
    parameter Real p2 = p1;
    Real v[2](each start = 0, each fixed = true);
    Real y(start = p2, fixed = true);
    Real w(start = p1, fixed = true);
  equation
    der(v) = {1, 2};
    der(y) = 1;
    der(w) = 1;
  end StartOfParameters;
  model StartOfFreeParameter "y(start = q, fixed = true) with q fixed = false, computed at the initialization (omc y(0) = 3)"
    parameter Real q(fixed = false, start = 1);
    Real v[2](each start = 0, each fixed = true);
    Real y(start = q, fixed = true);
  initial equation
    q = 3;
  equation
    der(v) = {1, 2};
    der(y) = 1;
  end StartOfFreeParameter;
  model StructuralDependency "R0 read by a structural R (an if-equation's condition): structural too, not overridable"
    parameter Real R0 = 1;
    parameter Real R = 2 * R0;
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    if R > 0 then
      der(x) = R;
    else
      der(x) = -R;
    end if;
  end StructuralDependency;
  function checkBelow "fails above lim"
    input Real x;
    input Real lim;
  algorithm
    assert(x <= lim, "checkBelow: x above lim");
  end checkBelow;
  model CallInIfBranch "a call equation under an if-equation's branch (made where it holds: fails at x > lim once x > limit)"
    parameter Real limit = 0.5;
    parameter Real lim = 0.7;
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    der(x) = 1;
    if x > limit then
      checkBelow(x, lim);
    end if;
  end CallInIfBranch;
  model ParamDep
    parameter Real a = 2;
    parameter Real b = 3 * a;
    parameter Integer n = 3;
    parameter Real k[n] = {a, 2 * a, b};
    Real x[n](each start = 1, each fixed = true);
  equation
    der(x) = -k .* x;
  end ParamDep;

  model BouncingBalls
    parameter Integer n = 5;
    parameter Real g = 9.81;
    parameter Real e[n] = {0.6 + 0.35 * (i - 1) / (n - 1) for i in 1:n};
    Real h[n](each start = 1.0, each fixed = true);
    Real v[n](each start = 0.0, each fixed = true);
    discrete Integer bounces[n](each start = 0);
  equation
    for i in 1:n loop
      der(h[i]) = v[i];
      der(v[i]) = -g;
      when h[i] <= 0.0 then
        reinit(v[i], -e[i] * pre(v[i]));
        bounces[i] = pre(bounces[i]) + 1;
      end when;
    end for;
  end BouncingBalls;

  model Rectifier
    parameter Integer n = 3;
    Real vc[n](each start = 0, each fixed = true);
    Real i[n];
  equation
    for k in 1:n loop
      i[k] = if sin(6.28 * time + k) > vc[k] then sin(6.28 * time + k) - vc[k] else 0.0;
      der(vc[k]) = i[k] - vc[k] / 10;
    end for;
  end Rectifier;

  model AlgebraicLoop "Outside the array path (a loop in a model without arrays): it is scalarized as before"
    Real x(start = 1, fixed = true);
    Real y, z;
  equation
    der(x) = -y;
    y + z = x;
    y - z = 0.5 * x;
  end AlgebraicLoop;
  model Guarded "Asserts in a loop: a warning, and an error after t = 0.5"
    parameter Integer n = 3;
    Real x[n](each start = 1, each fixed = true);
  equation
    for i in 1:n loop
      der(x[i]) = -i * x[i];
      assert(x[i] > 0.9, "x below 0.9", AssertionLevel.warning);
    end for;
    assert(time < 0.5, "time limit");
  end Guarded;
  model InitSteady
    parameter Real u = 2;
    Real x;
    Real y(start = 1, fixed = true);
  initial equation
    der(x) = 0;
  equation
    der(x) = -x + u + 0.1 * y;
    der(y) = -y;
  end InitSteady;

  model SampleZOH
    Real x(start = 0, fixed = true);
    discrete Real u(start = 1);
    discrete Integer k(start = 0);
  equation
    der(x) = -x + u;
    when sample(0.1, 0.2) then
      u = -pre(u);
      k = pre(k) + 1;
    end when;
  end SampleZOH;

  model StaticIf
    parameter Integer n = 5;
    Real x[n](start = {if i == 1 then 1.0 else 0.0 for i in 1:n}, each fixed = true);
    Real s;
    Real sl;
  equation
    for i in 1:n loop
      if i == 1 then
        der(x[i]) = -x[i];
      else
        der(x[i]) = x[i - 1] - x[i];
      end if;
    end for;
    s = sum(x);
    sl = semiLinear(x[1] - 0.5, 2.0, 0.5);
  end StaticIf;

  model DiscreteEq
    Boolean b;
    Integer m;
    Real x(start = 1, fixed = true);
    parameter Real A[2, 2] = [-1, 0.5; 0, -2];
    Real z[2](each start = 1, each fixed = true);
  equation
    b = x > 0.5;
    m = if b then 1 else 2;
    der(x) = if b then -1 else -0.1 * m;
    der(z) = A * z;
  end DiscreteEq;

  model AlgInitial
    discrete Real t0(start = -1);
    Real x(start = 0, fixed = true);
    Real z;
    discrete Integer n(start = 0);
  algorithm
    when initial() then
      t0 := time + 0.25;
    end when;
    z := 0;
    for i in 1:3 loop
      z := z + i * x;
    end for;
    when x > 0.5 then
      n := pre(n) + 1;
    end when;
  equation
    der(x) = 1 + t0;
  end AlgInitial;
  model EventFunctions "integer, floor, mod and div generate events"
    Real x(start = 0, fixed = true);
    Real y[3];
    discrete Integer n(start = 0, fixed = true);
  equation
    der(x) = 1;
    y[1] = floor(2.5 * x);
    y[2] = mod(x, 0.3);
    y[3] = div(3 * x, 1);
    when integer(4 * x) > pre(n) then
      n = pre(n) + 1;
    end when;
  end EventFunctions;
  model VectorWhen "when {c1, c2, initial()}: fires when one of them becomes true"
    Real x(start = 0, fixed = true);
    discrete Integer n(start = 0, fixed = true);
    discrete Real tl(start = -1, fixed = true);
  equation
    der(x) = 1;
    when {x > 0.3, x > 0.6, initial()} then
      n = pre(n) + 1;
      tl = time;
    end when;
  end VectorWhen;
  model FreeParam "parameters with fixed = false, determined by initial equations"
    parameter Real t0(fixed = false);
    parameter Real k(fixed = false, start = 3);
    Real x(start = 2, fixed = true);
  initial equation
    t0 = time;
    2 * k = x;
  equation
    der(x) = -k * x + t0;
  end FreeParam;
  model DoubleExpImpulse "fixed = false parameters of a double exponential impulse per element (MSL LightningImpulse, DoubleExp, 10/350 us): from the starts, undamped Newton ran to the degenerate root tau1 = tau2, eta = 0"
    parameter Integer N = 2;
    parameter Real T1 = 10e-6;
    parameter Real T2 = 350e-6;
    parameter Real T0 = T2 * T1 / (T2 - T1) * log(T2 / T1);
    parameter Real eta[N](each fixed = false, each start = 1);
    parameter Real T[N](each fixed = false, each start = T0);
    parameter Real T10[N](each fixed = false, each start = 0.01 * T0);
    parameter Real tau1[N](each fixed = false, each start = T2);
    parameter Real tau2[N](each fixed = false, each start = T1);
    Real x[N](each start = 0, each fixed = true);
    Real y[N];
  initial equation
    for i in 1:N loop
      exp(-T[i] / tau1[i]) / tau1[i] = exp(-T[i] / tau2[i]) / tau2[i];
      eta[i] = exp(-T[i] / tau1[i]) - exp(-T[i] / tau2[i]);
      0.1 * eta[i] = exp(-T10[i] / tau1[i]) - exp(-T10[i] / tau2[i]);
      0.9 * eta[i] = exp(-(T10[i] + 0.8 * T1) / tau1[i]) - exp(-(T10[i] + 0.8 * T1) / tau2[i]);
      0.5 * eta[i] = exp(-(T10[i] - 0.1 * T1 + T2) / tau1[i]) - exp(-(T10[i] - 0.1 * T1 + T2) / tau2[i]);
    end for;
  equation
    for i in 1:N loop
      y[i] = (exp(-time / tau1[i]) - exp(-time / tau2[i])) / eta[i];
      der(x[i]) = y[i];
    end for;
  end DoubleExpImpulse;
  model InitAlgPulse "an initial algorithm sets discrete values (as MSL's Pulse)"
    parameter Real period = 0.25;
    parameter Real startTime = -0.6;
    discrete Integer count;
    discrete Real T_start;
    Real y;
  initial algorithm
    count := integer((time - startTime) / period);
    T_start := startTime + count * period;
  equation
    when integer((time - startTime) / period) > pre(count) then
      count = pre(count) + 1;
      T_start = time;
    end when;
    y = if time < T_start + 0.1 then 1 else 0;
  end InitAlgPulse;
  model AssertMessage "an assert's message with String(), reported where the condition fails"
    Real x(start = 0, fixed = true);
  equation
    der(x) = 1;
    assert(x < 0.5, "x reached " + String(x, significantDigits = 6, minimumLength = 0, leftJustified = true));
  end AssertMessage;
  model IfDynamic "an if-equation on a relation of the state: y and z switch branch at x = 0.5"
    Real x(start = 0, fixed = true);
    Real y;
    Real z;
    Real w(start = 0, fixed = true);
  equation
    der(x) = 1;
    if x > 0.5 then
      y = 1;
      z = x;
    elseif x > 0.2 then
      z = 2 * x;
      y = 3;
    else
      y = 2;
      z = -x;
    end if;
    der(w) = y + z;
  end IfDynamic;
  model InitialAssert "an assert among the initial equations: checked once at the start"
    parameter Real k = -1;
    Real x(start = 1, fixed = true);
  initial equation
    assert(k > 0, "k must be positive, k = " + String(k));
  equation
    der(x) = k * x;
  end InitialAssert;
  function twoSlopes "a step with two slopes (not inlined: an algorithm with an if)"
    input Real x;
    input Real k;
    output Real y;
  protected
    Real s;
  algorithm
    s := if x > 0 then 1 else 0.5;
    y := s * k * x;
  end twoSlopes;
  function shifted "calls twoSlopes"
    input Real x;
    output Real y;
  algorithm
    y := twoSlopes(x, 3) + 1;
  end shifted;
  function squareAndNegate "two outputs"
    input Real x;
    output Real a;
    output Real b;
  algorithm
    a := x ^ 2;
    b := -x;
  end squareAndNegate;
  function norm2 "an array argument"
    input Real v[:];
    output Real y;
  algorithm
    y := sqrt(v * v);
  end norm2;
  function scaled "an array result"
    input Real v[3];
    input Real k;
    output Real w[3];
  algorithm
    w := k * v;
  end scaled;
  model Functions "Modelica functions in the equations: scalar, nested, two outputs, array argument and result"
    Real x(start = 1, fixed = true);
    Real y;
    Real z;
    Real p;
    Real q;
    Real v[3](start = {1, 2, 3}, each fixed = true);
    Real n;
    Real w[3];
  equation
    y = twoSlopes(x, 2);
    z = shifted(x);
    (p, q) = squareAndNegate(x);
    der(x) = q;
    n = norm2(v);
    w = scaled(v, 2);
    der(v) = -w / n;
  end Functions;
  function check "a call without a result: asserts its argument"
    input Real x;
  algorithm
    assert(x < 10, "x too large: " + String(x));
  end check;
  model ElseWhen "when ... elsewhen; calls without a result in an equation and an algorithm; an assert in a when body"
    Real x(start = 0, fixed = true);
    discrete Integer n(start = 0, fixed = true);
    discrete Integer m(start = 0, fixed = true);
  equation
    der(x) = 1;
    when x > 0.3 then
      n = pre(n) + 1;
      m = pre(m);
    elsewhen x > 0.6 then
      n = pre(n) + 10;
      m = pre(m) + 1;
    end when;
    when x > 0.8 then
      assert(n < 5, "n reached " + String(n), AssertionLevel.warning);
    end when;
    check(x);
  algorithm
    check(2 * x);
  end ElseWhen;
  model LocalLoops "a nonlinear loop of two unknowns per element (y, w): solved per element by Newton"
    parameter Integer N = 4;
    Real x[N](start = {1.0 + i for i in 1:N}, each fixed = true);
    Real y[N];
    Real w[N];
  equation
    for i in 1:N loop
      der(x[i]) = -y[i];
      y[i] + w[i] ^ 3 = x[i];
      w[i] = y[i] + 0.1 * sin(time);
    end for;
  end LocalLoops;
  model LocalLoops1000 = LocalLoops(N = 1000);
  model Cubic "an equation nonlinear in its unknown: y^3 + y = x per element"
    parameter Integer N = 3;
    Real x[N](start = {1.0, 2.0, 3.0}, each fixed = true);
    Real y[N];
  equation
    for i in 1:N loop
      der(x[i]) = -y[i];
      y[i] ^ 3 + y[i] = x[i];
    end for;
  end Cubic;
  model LoopWithEvent "a loop per element through a relation (events in a loop): outside the array path, scalarized as before"
    parameter Integer N = 3;
    Real x[N](each start = 1, each fixed = true);
    Real y[N];
    Real z[N];
  equation
    for i in 1:N loop
      der(x[i]) = -y[i];
      y[i] + z[i] = x[i];
      y[i] - z[i] = if x[i] > 0.5 then 0.5 * x[i] else 0.2 * x[i];
    end for;
  end LoopWithEvent;
  model CoupledLoop "an algebraic loop through all the elements (a tridiagonal system in v), solved in the right-hand side"
    parameter Integer N = 20;
    Real x[N](start = {i / N for i in 1:N}, each fixed = true);
    Real v[N];
  equation
    for i in 1:N loop
      der(x[i]) = v[i] - x[i];
    end for;
    2 * v[1] - v[2] = x[1];
    for i in 2:N - 1 loop
      -v[i - 1] + 3 * v[i] - v[i + 1] = x[i];
    end for;
    -v[N - 1] + 2 * v[N] = x[N];
  end CoupledLoop;
  model CoupledLoop1000 = CoupledLoop(N = 1000);
  model CoupledNonlinear "the loop through the elements nonlinear (v^3 terms), and an algebraic variable after it"
    parameter Integer N = 20;
    Real x[N](start = {i / N for i in 1:N}, each fixed = true);
    Real v[N];
    Real w[N];
  equation
    for i in 1:N loop
      der(x[i]) = w[i];
      w[i] = v[i] - x[i];
    end for;
    2 * v[1] - v[2] + 0.1 * v[1] ^ 3 = x[1];
    for i in 2:N - 1 loop
      -v[i - 1] + 3 * v[i] - v[i + 1] + 0.1 * v[i] ^ 3 = x[i];
    end for;
    -v[N - 1] + 2 * v[N] + 0.1 * v[N] ^ 3 = x[N];
  end CoupledNonlinear;
  model CoupledLoopEvents "CoupledLoop (an array DAE) with an event in the loop (a forcing from t = 0.5), a when-equation on a state, an assert and an initial equation for x[N]"
    parameter Integer N = 20;
    Real x[N](start = {i / N for i in 1:N}, fixed = {i < N for i in 1:N});
    Real v[N];
    Integer crossings(start = 0, fixed = true);
  initial equation
    x[N] = 0.5;
  equation
    for i in 1:N loop
      der(x[i]) = v[i] - x[i];
    end for;
    2 * v[1] - v[2] = x[1] + (if time < 0.5 then 0 else 1);
    for i in 2:N - 1 loop
      -v[i - 1] + 3 * v[i] - v[i + 1] = x[i];
    end for;
    -v[N - 1] + 2 * v[N] = x[N];
    when x[1] > 0.1 then
      crossings = pre(crossings) + 1;
    end when;
    assert(v[1] < 100, "v[1] stays below 100");
  end CoupledLoopEvents;
  model PreLatch "a discrete equation on pre() of a counter, lifted as a cluster triggered by change(pre(n)) (run at every pass, as MSL Digital's Pulse): two samples, then inactive"
    parameter Integer nmax = 2;
    discrete Integer n(start = 0, fixed = true);
    Boolean active;
    Real x[2](each start = 0, each fixed = true);
  equation
    active = nmax > pre(n);
    when active and sample(0, 0.25) then
      n = pre(n) + 1;
    end when;
    for i in 1:2 loop
      der(x[i]) = if active then i else 0;
    end for;
  end PreLatch;
  model ImplicitDiscrete "a Real assigned in a when-equation (implicitly discrete, as MSL Pulse's T_start) read by loop equations; a discrete's initial value by an initial equation; free parameters whose initial equations come before the ones they read"
    Real T0(start = 0);
    discrete Real d;
    parameter Real p2(fixed = false);
    parameter Real p1(fixed = false);
    Real x[2](each start = 0, each fixed = true);
  initial equation
    p2 = p1 + 1;
    p1 = 2;
    d = 3;
  equation
    when sample(0.1, 0.2) then
      T0 = time;
      d = pre(d) + 1;
    end when;
    for i in 1:2 loop
      der(x[i]) = i * T0 + p2;
    end for;
  end ImplicitDiscrete;
  model WhenReadsTarget "a when algorithm whose tuple assignment reads its target ((s, y) := step(s)): s from pre(s) at each run of the body (MLS 11.1.2), the event iteration settles"
    function step
      input Integer s;
      output Integer next;
      output Real y;
    algorithm
      next := s + 1;
      y := 10 * s;
    end step;
    Real x[2](each start = 0, each fixed = true);
    Integer s(start = 0, fixed = true);
    discrete Real y(start = -1, fixed = true);
  equation
    der(x) = {1, 2};
  algorithm
    when sample(0.1, 0.25) then
      (s, y) := step(s);
    end when;
  end WhenReadsTarget;
  model ReinitElements "reinit of array state elements: in a for-loop of when-equations (an iterator's subscript) and by a parameter's subscript"
    parameter Integer k = 3;
    Real v[3](each start = 0, each fixed = true);
  equation
    for i in 1:3 loop
      der(v[i]) = i;
      when v[i] > 1.5 then
        reinit(v[i], 0);
      end when;
    end for;
    when time > 0.3 then
      reinit(v[k], -1);
    end when;
  end ReinitElements;
  model RecordTupleWhen "a record target of a tuple assignment in a when algorithm: the result's elements by their positions, the record's fields in place (r.a, r.b, q: 1, 2, 3)"
    record Pair
      Real a;
      Real b;
    end Pair;
    function pairAndScalar
      input Real x;
      output Pair r;
      output Real q;
    algorithm
      r := Pair(x, 2 * x);
      q := 3 * x;
    end pairAndScalar;
    Real x[2](each start = 0, each fixed = true);
    discrete Pair r(a(start = 0, fixed = true), b(start = 0, fixed = true));
    discrete Real q(start = 0, fixed = true);
  equation
    der(x) = {1, 2};
  algorithm
    when sample(0, 0.1) then
      (r, q) := pairAndScalar(time);
    end when;
  end RecordTupleWhen;
  model RegularAlgReads "a regular algorithm reading its discrete target before assigning it at the events u changes at (lifted to whens): n from pre(n) at each pass (omc: n = 10 at 1)"
    Real x[2](each start = 0, each fixed = true);
    discrete Integer u(start = 0, fixed = true);
    discrete Integer n(start = 0, fixed = true);
    discrete Integer m(start = 0, fixed = true);
  equation
    der(x) = {1, 2};
    when sample(0.1, 0.25) then
      u = pre(u) + 1;
    end when;
  algorithm
    if u <> pre(u) then
      n := n + u;
    end if;
    m := 2 * n;
  end RegularAlgReads;
  model EdgeOfTarget "edge() of a target before its assignment in a when algorithm: false, the target its pre() (omc: cnt = 0)"
    Real x[2](each start = 0, each fixed = true);
    discrete Integer cnt(start = 0, fixed = true);
    discrete Boolean flag(start = false, fixed = true);
  equation
    der(x) = {1, 2};
  algorithm
    when sample(0.1, 0.25) then
      if edge(flag) then
        cnt := cnt + 1;
      end if;
      flag := not pre(flag);
    end when;
  end EdgeOfTarget;
  model SumOfTargets "a whole-array read of element targets before their assignment: their pre() values (omc: s = 4 at 0.5, 6 at 1)"
    Real z[2](each start = 0, each fixed = true);
    discrete Real x[3](each start = 1, each fixed = true);
    discrete Real s(start = 0, fixed = true);
  equation
    der(z) = {1, 2};
  algorithm
    when sample(0.1, 0.25) then
      s := sum(x);
      x[1] := pre(x[1]) + 1;
    end when;
  end SumOfTargets;
  model VarSubRead "a read of a target through a variable subscript before its assignment: x[2] from pre(x[2]) (omc: x[2] = 10, 20, 40 at 0.2, 0.4, 1)"
    Real z[2](each start = 0, each fixed = true);
    parameter Integer i = 2;
    discrete Real x[2](each start = 0, each fixed = true);
  equation
    der(z) = {1, 2};
  algorithm
    when sample(0.1, 0.25) then
      x[1] := 1;
      x[2] := x[i] + 10;
    end when;
  end VarSubRead;
  record P1
    Real a;
  end P1;
  model AssignedThenGuarded "an element assigned, then a guarded assignment of it through a variable index: the guard's else keeps the assigned value (omc: s[2].a = 5, 5, 7, 7 at 0.2, 0.4, 0.7, 0.9)"
    Real z[2](each start = 0, each fixed = true);
    discrete Integer k(start = 2, fixed = true);
    discrete P1 s[2](each a(start = 0, fixed = true));
  equation
    der(z) = {1, 2};
    when time > 10 then
      k = 1;
    end when;
  algorithm
    when sample(0.1, 0.25) then
      s[2].a := 5;
      if time > 0.5 then
        s[k].a := 7;
      end if;
    end when;
  end AssignedThenGuarded;
  model ReadThenIndexed "an element read, then assigned through a variable index: it starts from pre() (omc: s[2].a = 1 at 0.2, 4 at 0.9, y = 3 at 0.9)"
    Real z[2](each start = 0, each fixed = true);
    discrete Integer k(start = 2, fixed = true);
    discrete P1 s[2](each a(start = 0, fixed = true));
    discrete Real y(start = 0, fixed = true);
  equation
    der(z) = {1, 2};
    when time > 10 then
      k = 1;
    end when;
  algorithm
    when sample(0.1, 0.25) then
      y := s[2].a;
      s[k].a := s[2].a + 1;
    end when;
  end ReadThenIndexed;
  model PreIndexedRecord "pre() of an element by an index the body assigns after: the index read now (omc: y = 10, 20, 10, 20 at 0.2, 0.4, 0.7, 0.9)"
    Real z(start = 0, fixed = true);
    discrete Integer idx(start = 1, fixed = true);
    discrete P1 s[2](a(start = {10, 20}, each fixed = true));
    discrete Real y(start = 0, fixed = true);
  equation
    der(z) = 1;
    when time > 10 then
      s[1].a = 30;
      s[2].a = 40;
    end when;
  algorithm
    when sample(0.1, 0.25) then
      y := pre(s[idx].a);
      idx := if idx == 2 then 1 else idx + 1;
    end when;
  end PreIndexedRecord;
  model RingRead "read the element at idx, write it, advance idx (omc: y = 0, 0, 0.1, 0.35 at 0.2, 0.4, 0.7, 0.9)"
    Real z(start = 0, fixed = true);
    discrete Integer idx(start = 1, fixed = true);
    discrete P1 buf[2](each a(start = 0, fixed = true));
    discrete Real y(start = 0, fixed = true);
  equation
    der(z) = 1;
  algorithm
    when sample(0.1, 0.25) then
      y := buf[idx].a;
      buf[idx].a := time;
      idx := if idx == 2 then 1 else idx + 1;
    end when;
  end RingRead;
  model StaleElement "an element target by an index an equation computes from a when's output (omc: s[2].a = 0, s[1].a = 0.85 at 0.9)"
    Real w(start = 0, fixed = true);
    discrete Integer z(start = 0, fixed = true);
    Integer k;
    discrete P1 s[2](each a(start = 0, fixed = true));
  equation
    der(w) = 1;
    k = if z > 0 then 1 else 2;
    when sample(0.1, 0.25) then
      z = pre(z) + 1;
    end when;
  algorithm
    when sample(0.1, 0.25) then
      s[k].a := time;
    end when;
  end StaleElement;
  model RecBig "an element target by an index over 1001 records: every element it may assign starts from pre(), no element cap (omc: s[1].a = 0.85, s[2].a = 0.35, s[3].a = 0.6 at 1)"
    Real z(start = 0, fixed = true);
    discrete Integer k(start = 1, fixed = true);
    discrete P1 s[1001](each a(start = 0, fixed = true));
  equation
    der(z) = 1;
  algorithm
    when sample(0.1, 0.25) then
      s[k].a := time;
      k := if k == 3 then 1 else k + 1;
    end when;
  end RecBig;
  record R3
    Real a;
    Real b;
  end R3;
  function recordFirst3 "a record output, then two scalars"
    input Real t;
    output R3 r;
    output Real q;
    output Real z;
  algorithm
    r := R3(t, 2 * t);
    q := 3 * t;
    z := 4 * t;
  end recordFirst3;
  model TupleFewerTargets "a tuple equation with a record target and fewer targets than outputs (omc: q = 3 at 1)"
    Real x[2](each start = 0, each fixed = true);
    R3 r;
    Real q;
  equation
    der(x) = {1, 2};
    (r, q) = recordFirst3(time);
  end TupleFewerTargets;
  model DiscreteNoEvents "Integers equations compute in a model without events: evaluated at the start"
    parameter Integer k = 3;
    Integer a;
    Integer b;
    Real x(start = 0, fixed = true);
  equation
    a = k;
    b = a + 1;
    der(x) = b;
  end DiscreteNoEvents;
  model EnumTable "a constant table over an enumeration's dimensions (MSL Digital's AndTable) indexed by enumerations equations compute"
    type L = enumeration(u, x, z, o);
    constant L T[L, L] = [L.u, L.u, L.z, L.u; L.u, L.x, L.z, L.x; L.z, L.z, L.z, L.z; L.u, L.x, L.z, L.o];
    parameter L p = L.o;
    L a[2](each start = L.u);
    Real x(start = 0, fixed = true);
  equation
    a[1] = p;
    for i in 1:1 loop
      a[i + 1] = T[a[i], p];
    end for;
    der(x) = 1;
  end EnumTable;
  model PreAfterInit "the event iteration after the initialization: pre() takes the initial values (mode 3 -> 0 -> 1; omc: mode 1 at 0 and 0.5, x(1) = 1)"
    Integer mode(start = 3, fixed = true);
    Boolean started(start = false, fixed = true);
    Real x(start = 0, fixed = true);
  equation
    started = pre(mode) == 0 and x < 0.5 or pre(started);
    mode = if started then 1 else 0;
    der(x) = mode;
  end PreAfterInit;
  model ExactBoundaryAtEvent "a relation <= on its boundary exactly at a time event holds (MSL SimpleGearShift's brake; omc: free from 0.5, x(1) = 0.5)"
    Real f;
    Boolean free;
    Real x(start = 0, fixed = true);
  equation
    f = 1 + (if time < 0.4 then 0 else if time < 0.5 then -(time - 0.4) / 0.1 else -1);
    free = f <= 0;
    der(x) = if free then 1 else 0;
  end ExactBoundaryAtEvent;
  model SampleEquation "sample() in an equation outside the when, its start a parameter an initial equation computes (CDL Discrete.UnitDelay; omc: y = 0, 0, 0.2, 0.6 at 0.1, 0.3, 0.5, 0.9)"
    parameter Real samplePeriod = 0.2;
    Real u = time;
    Real y;
    parameter Real t0(fixed = false);
    Boolean trigger;
    discrete Real held;
  initial equation
    t0 = integer(time / samplePeriod) * samplePeriod;
    y = 0;
    held = 0;
  equation
    trigger = sample(t0, samplePeriod);
    when trigger then
      held = u;
      y = pre(held);
    end when;
  end SampleEquation;
  function fibonacciStep "the next pair of a Fibonacci sequence and a tenth of its first"
    input Integer s[2];
    output Real r;
    output Integer t[2];
  algorithm
    r := s[1] / 10;
    t := {s[1] + s[2], s[1]};
  end fibonacciStep;
  model TupleWhen "a tuple with an array output in a when-equation: (r, s) = f(pre(s)) at ten ticks (s = {89, 55}, r = 5.5 at 1)"
    Real r(start = 0, fixed = true);
    Integer s[2](start = {1, 0}, each fixed = true);
  equation
    when sample(0.05, 0.1) then
      (r, s) = fibonacciStep(pre(s));
    end when;
  end TupleWhen;
  function scaledPair "{a, 2 a}"
    input Real a;
    output Real p[2];
  algorithm
    p := {a, 2 * a};
  end scaledPair;
  model ParameterOfArrayCall "an array parameter bound to a call of literals: computed at the start (y(1) = 3)"
    parameter Real p[2] = scaledPair(1.0);
    Real y = p[1] + p[2] * time;
  end ParameterOfArrayCall;
  model StringParameterMessage "a String parameter as an assert's message (CDL Utilities.Assert; x(1) = exp(-1))"
    parameter String message = "x must stay above 0.1";
    Real x(start = 1, fixed = true);
  equation
    der(x) = -x;
    assert(x > 0.1, message);
  end StringParameterMessage;
  model StringParameterReference "a String parameter bound to another's reference as an assert's message (x(1) = exp(-1))"
    parameter String base = "x must stay above 0.1";
    parameter String message = base;
    Real x(start = 1, fixed = true);
  equation
    der(x) = -x;
    assert(x > 0.1, message);
  end StringParameterReference;
  model FallingEdgeOfSample "a sample() Boolean's falling edge (CDL FallingEdge; omc: y = 0 at 0.2 and 0.5, n = 1, 2, 4 at 0.2, 0.5, 1)"
    Real v[2](each start = 0, each fixed = true);
    Boolean u;
    Boolean y;
    Integer n(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    u = sample(0.1, 0.25);
    y = not u and pre(u);
    when y then
      n = pre(n) + 1;
    end when;
  end FallingEdgeOfSample;
  model TickNot "a when on not of a sample() Boolean (omc: n = 1, 2, 3, 5 at 0.1, 0.3, 0.6, 1)"
    Real v[2](each start = 0, each fixed = true);
    Boolean tick;
    Boolean b;
    Integer n(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    tick = sample(0, 0.25);
    b = not tick;
    when b then
      n = pre(n) + 1;
    end when;
  end TickNot;
  model SampleFirstPass "sample() true for the first pass of its instant only: change(), a counter by an equation, pre() beside it (omc: c = k = 1, 2, 4 and b = 0, 1, 3 at 0.2, 0.5, 1)"
    Real v[2](each start = 0, each fixed = true);
    Boolean u;
    Integer a(start = 0, fixed = true);
    Integer b(start = 0, fixed = true);
    Integer c(start = 0, fixed = true);
    Integer k(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    u = sample(0.1, 0.25);
    k = if u then pre(k) + 1 else pre(k);
    when u then
      a = pre(a) + 1;
    end when;
    when u and pre(a) > 0 then
      b = pre(b) + 1;
    end when;
    when change(u) then
      c = pre(c) + 1;
    end when;
  end SampleFirstPass;
  model TwoSampleTimes "sample(0, 0.2) and sample(0, 0.3): 3 * 0.2 = 0.6000000000000001 and 2 * 0.3 = 0.6 one instant (omc: a = 3, 4, 6, b = 2, 3, 4 at 0.5, 0.7, 1)"
    Real v[2](each start = 0, each fixed = true);
    Integer a(start = 0, fixed = true);
    Integer b(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    when sample(0, 0.2) then
      a = pre(a) + 1;
    end when;
    when sample(0, 0.3) then
      b = pre(b) + 1;
    end when;
  end TwoSampleTimes;
  model NearTicks "ticks 5e-10 and 2e-8 after another sample's: separate instants, pre(a) after a's tick (omc: b = c = 1, 2, 4 at 0.2, 0.5, 1)"
    Real v[2](each start = 0, each fixed = true);
    Integer a(start = 0, fixed = true);
    Integer b(start = 0, fixed = true);
    Integer c(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    when sample(0.1, 0.25) then
      a = pre(a) + 1;
    end when;
    when sample(0.1 + 5e-10, 0.25) then
      b = pre(a);
    end when;
    when sample(0.1 + 2e-8, 0.25) then
      c = pre(a);
    end when;
  end NearTicks;
  model ChainTicks "three samples 0.8e-9 apart: each tick once (omc: a = b = c = 1, 4 at 0.2, 1)"
    Real v[2](each start = 0, each fixed = true);
    Integer a(start = 0, fixed = true);
    Integer b(start = 0, fixed = true);
    Integer c(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    when sample(0.1, 0.25) then
      a = pre(a) + 1;
    end when;
    when sample(0.1 + 0.8e-9, 0.25) then
      b = pre(b) + 1;
    end when;
    when sample(0.1 + 1.6e-9, 0.25) then
      c = pre(c) + 1;
    end when;
  end ChainTicks;
  model TicksNearStartAndStop "ticks 5e-10 after the start and after the stop time: at their own times, the last none (omc: n = 0, 1, 4 at 0, 0.1, 1; tn = 5e-10 at 0.1; k = 3 at 1)"
    Real v[2](each start = 0, each fixed = true);
    Integer n(start = 0, fixed = true);
    Integer k(start = 0, fixed = true);
    discrete Real tn(start = -1, fixed = true);
  equation
    der(v) = {1, 2};
    when sample(5e-10, 0.25) then
      n = pre(n) + 1;
      tn = time;
    end when;
    when sample(0.25 + 5e-10, 0.25) then
      k = pre(k) + 1;
    end when;
  end TicksNearStartAndStop;
  model FarStartTicks "sample(-100, 0.1) and sample(0, 0.1): ticks equal up to the round-off of 100 one instant (omc: w = 1, 4, 9, 11 at 0.05, 0.35, 0.85, 1)"
    Real v[2](each start = 0, each fixed = true);
    Integer w(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    when {sample(-100, 0.1), sample(0, 0.1)} then
      w = pre(w) + 1;
    end when;
  end FarStartTicks;
  model TerminateInBranch "terminate() in an if-equation's branch: at x's crossing of 0.75 (omc: at its output step after it, 0.752)"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    der(x) = 1;
    if x > 0.75 then
      terminate("x above 0.75");
    end if;
  end TerminateInBranch;
  model TerminateAtStart "the branch's condition holds at the start: terminate() at the initialization (omc: too)"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 1, fixed = true);
  equation
    der(v) = {1, 2};
    der(x) = 1;
    if x > 0.75 then
      terminate("x above 0.75");
    end if;
  end TerminateAtStart;
  model TerminateInElse "terminate() in an if-equation of an else-branch: at 0.6 (omc ignores it, warning that if-equations are partially supported)"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
    Real y;
  equation
    der(v) = {1, 2};
    der(x) = 1;
    if x < 0.5 then
      y = 1;
    else
      y = 2;
      if x > 0.6 then
        terminate("x above 0.6");
      end if;
    end if;
  end TerminateInElse;
  function report "a warning when n reaches 3"
    input Integer n;
  algorithm
    assert(n < 3, "n reached 3", AssertionLevel.warning);
  end report;
  model TerminateInWhenIf "terminate() and a call in if-equations of a when-equation: the warning at 0.25, the end at 0.35 (omc fails to build it: lowerWhenEqn)"
    Real v[2](each start = 0, each fixed = true);
    Integer n(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    when sample(0.05, 0.1) then
      n = pre(n) + 1;
      if n >= 3 then
        report(n);
      end if;
      if n >= 4 then
        terminate("four ticks");
      end if;
    end when;
  end TerminateInWhenIf;
  model TerminateNow "terminate() outside when- and if-equations: at the initialization (omc: too)"
    Real v[2](each start = 0, each fixed = true);
  equation
    der(v) = {1, 2};
    terminate("at once");
  end TerminateNow;
  model AssertWindow "an assert violated for x in (0.7, 0.9), within one step of der(x) = 1 (omc: fails at 0.701)"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    der(x) = 1;
    assert(not (x > 0.7 and x < 0.9), "x in (0.7, 0.9)");
  end AssertWindow;
  model BranchAssertWindow "an assert in an elseif branch, violated for x in (0.7, 0.9) (omc: fails at 0.701)"
    parameter Real lim = 0.7;
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
    Real y;
  equation
    der(v) = {1, 2};
    der(x) = 1;
    if x < 0.2 then
      y = 1;
    elseif x < 0.9 then
      y = 2;
      assert(x <= lim, "x above lim");
    else
      y = 3;
    end if;
  end BranchAssertWindow;
  model AssertTouch "an assert whose relation touches its boundary: x = 1 - (t - 0.5)^2 reaches 1 at 0.5 (no failure)"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0.75, fixed = true);
  equation
    der(v) = {1, 2};
    der(x) = -2 * (time - 0.5);
    assert(x <= 1, "x above 1");
  end AssertTouch;
  model StartOfSum "a start of the sum of a parameter's elements (omc: x(0) = 6; -override=q[1]=10: 15)"
    parameter Real q[3] = {1, 2, 3};
    Real v[2](each start = 0, each fixed = true);
    Real x(start = sum(q), fixed = true);
  equation
    der(v) = {1, 2};
    der(x) = 1;
  end StartOfSum;
  model StartComponent "a component with a start of its parameter"
    parameter Real s = 1;
    Real x(start = s, fixed = true);
  equation
    der(x) = -x;
  end StartComponent;
  model StartOfComponents "rolled components' starts of their parameter bound to q (omc -override=q[1]=4,q[2]=5,q[3]=6: x(0) = 4, 5, 6)"
    parameter Real q[3] = {1, 2, 3};
    StartComponent c[3](s = q);
  end StartOfComponents;
  model DerCoefficientOfCall "a coefficient of der() that is a call of a parameter: sin(p) * der(x) = 1 (x(1) = 1 / sin(p))"
    parameter Real p = 0.5;
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    sin(p) * der(x) = 1;
  end DerCoefficientOfCall;
  model SwapInWhen "a whole-array assignment reading its own target in a when-algorithm: a swap at each tick"
    Real v[2](each start = 0, each fixed = true);
    discrete Integer x[2](start = {1, 2}, each fixed = true);
  equation
    der(v) = {1, 2};
  algorithm
    when sample(0.1, 0.25) then
      x := {x[2], x[1]};
    end when;
  end SwapInWhen;
  model RotationInWhen "x := A * x in a when-algorithm (a quarter turn at each tick)"
    parameter Real A[2, 2] = [0, -1; 1, 0];
    Real v[2](each start = 0, each fixed = true);
    discrete Real x[2](start = {1, 0}, each fixed = true);
  equation
    der(v) = {1, 2};
  algorithm
    when sample(0.1, 0.25) then
      x := A * x;
    end when;
  end RotationInWhen;
  model SwapRegular "a whole-array assignment reading its own target in a regular algorithm (Real)"
    Real v[2](each start = 0, each fixed = true);
    Real y[2];
  algorithm
    y := {v[1], v[2]};
    y := {y[2], y[1]};
  equation
    der(v) = {1, 2};
  end SwapRegular;
  model Alias "liftIfToWhen! counts x[k] and x[1] apart (omc x[1] = 3 at 0.1)"
    Real z(start = 0, fixed = true);
    discrete Integer k(start = 1, fixed = true);
    discrete Integer u(start = 3, fixed = true);
    Integer x[2](each start = 0, each fixed = true);
  equation
    der(z) = 1;
    when time > 10 then
      k = 2;
      u = 4;
    end when;
  algorithm
    if sample(0.1, 0.25) then
      x[k] := 5;
    end if;
    x[1] := u;
  end Alias;
  model ParamSample "a sample period from a parameter: n counts the ticks"
    parameter Real T = 0.25;
    Real v[2](each start = 0, each fixed = true);
    discrete Integer n(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    when sample(0, T) then
      n = pre(n) + 1;
    end when;
  end ParamSample;
  model StringParam "a String parameter in an assert's message"
    parameter String s = "abc";
    Real v[2](each start = 0, each fixed = true);
  equation
    der(v) = {1, 2};
    assert(v[1] < 0.5, "v above 0.5: " + s);
  end StringParam;
  model StartSumNested "a start of the sum of a nested literal: x(0) = 10"
    parameter Real q[2] = {1, 2};
    parameter Real r[2] = {3, 4};
    Real v[2](each start = 0, each fixed = true);
    Real x(start = sum({q, r}), fixed = true);
  equation
    der(v) = {1, 2};
    der(x) = 1;
  end StartSumNested;
  model StartSumMatrix "a start of the sum of a matrix parameter: x(0) = 10"
    parameter Real m[2, 2] = {{1, 2}, {3, 4}};
    Real v[2](each start = 0, each fixed = true);
    Real x(start = sum(m), fixed = true);
  equation
    der(v) = {1, 2};
    der(x) = 1;
  end StartSumMatrix;
  model DiscreteStartSum "a discrete start of a reduction the evaluation does not take: d = 10, then 11"
    parameter Real q[2] = {1, 2};
    parameter Real r[2] = {3, 4};
    Real v[2](each start = 0, each fixed = true);
    discrete Real d(start = sum({q, r}), fixed = true);
  equation
    der(v) = {1, 2};
    when time > 0.5 then
      d = pre(d) + 1;
    end when;
  end DiscreteStartSum;
  model GuessStartSum "a guess of a reduction the evaluation does not take: x(0) = -10 (the root it picks)"
    parameter Real q[2] = {1, 2};
    parameter Real r[2] = {3, 4};
    Real v[2](each start = 0, each fixed = true);
    Real x(start = -sum({q, r}));
  initial equation
    x ^ 2 = 100;
  equation
    der(v) = {1, 2};
    der(x) = 1;
  end GuessStartSum;
  model AssertAfterEvent "a crossing decided after an event that changes the discrete values: the assert holds throughout (omc too)"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
    Integer d(start = 0, fixed = true);
    discrete Real xs(start = 0.7, fixed = true);
  equation
    der(v) = {1, 2};
    der(x) = 1;
    when x > 0.6 then
      d = 1;
      xs = 0.58;
    end when;
    assert(x <= 0.5 or d == 0 or x > xs, "x in the window");
  end AssertAfterEvent;
  model AssertFailAtEvent "the assert fails at the event at 0.6, not at the crossing at 0.5 (omc 0.6)"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
    Integer d(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    der(x) = 1;
    when x > 0.6 then
      d = 1;
    end when;
    assert(x <= 0.5 or d == 0, "d set past 0.5");
  end AssertFailAtEvent;
  model CoefFreeParam "a der() coefficient of a parameter with fixed = false: x(1) = 1 / sin(0.5), on the array route"
    parameter Real p(fixed = false, start = 0.1);
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
  initial equation
    p = 0.5;
  equation
    der(v) = {1, 2};
    sin(p) * der(x) = 1;
  end CoefFreeParam;
  model TerminateNoEvent "terminate() under noEvent(): stops at 0.5 (omc 0.501)"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    der(x) = 1;
    if noEvent(x > 0.5) then
      terminate("x above 0.5");
    end if;
  end TerminateNoEvent;
  model WhenIfBeforeAssign "terminate() before the assignment it reads in a when-equation: stops at the third tick, 0.25, n = 3"
    Real v[2](each start = 0, each fixed = true);
    Integer n(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    when sample(0.05, 0.1) then
      if n >= 3 then
        terminate("three");
      end if;
      n = pre(n) + 1;
    end when;
  end WhenIfBeforeAssign;
  model WhenOrder "when-equations solved in order: b reads the new a (b = 2, 3, ...)"
    Real v[2](each start = 0, each fixed = true);
    Integer a(start = 0, fixed = true);
    Integer b(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    when sample(0.05, 0.1) then
      b = a + 1;
      a = pre(a) + 1;
    end when;
  end WhenOrder;
  model AssertIfExp "an assert of an if-expression: fails at 0.8"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    der(x) = 1;
    assert(if x < 0.5 then true else x < 0.8, "x reached 0.8");
  end AssertIfExp;
  model AssertSampleCount "an assert of a discrete value set at a sample tick: fails at the third tick, 0.25"
    Real v[2](each start = 0, each fixed = true);
    Integer n(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    when sample(0.05, 0.1) then
      n = pre(n) + 1;
    end when;
    assert(n < 3, "n reached 3");
  end AssertSampleCount;
  model DiscreteStartFree "a discrete start of a parameter with fixed = false: d = 2"
    parameter Real p(fixed = false, start = 0.5);
    Real v[2](each start = 0, each fixed = true);
    discrete Real d(start = p, fixed = true);
  initial equation
    p = 2;
  equation
    der(v) = {1, 2};
    when time > 0.5 then
      d = pre(d) + 1;
    end when;
  end DiscreteStartFree;
  model TermElsewhen "terminate() in if-equations of the branches of a when-equation with elsewhen: stops at 0.6, n = 2"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
    Integer n(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    der(x) = 1;
    when x > 0.3 then
      n = 1;
      if n > 5 then
        terminate("never");
      end if;
    elsewhen x > 0.6 then
      n = 2;
      if n == 2 then
        terminate("second branch");
      end if;
    end when;
  end TermElsewhen;
  model AlgTerminate "terminate() in an if-statement of an algorithm section: stops at 0.5 (omc 0.501)"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    der(x) = 1;
  algorithm
    if x > 0.5 then
      terminate("x above 0.5");
    end if;
  end AlgTerminate;
  model AlgTerminateReal "terminate() in an if-statement of an algorithm section with a Real target: stops at 0.5"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
    Real y;
  equation
    der(v) = {1, 2};
    der(x) = 1;
  algorithm
    y := 2 * x;
    if y > 1 then
      terminate("y above 1");
    end if;
  end AlgTerminateReal;
  model AlgIfAssert "an assert in an if of an algorithm without Real targets: fails at 0.7"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    der(x) = 1;
  algorithm
    if x > 0.5 then
      assert(x < 0.7, "x reached 0.7");
    end if;
  end AlgIfAssert;
  model MaxOfArrays "max over the elements of two arrays: y(1) = 2"
    Real v[2](each start = 0, each fixed = true);
    Real w[2](each start = 0, each fixed = true);
    Real y;
  equation
    der(v) = {1, 2};
    der(w) = {0.5, 0.25};
    y = max({v, w});
  end MaxOfArrays;
  model ForBranch "terminate() in an if-equation of a for-equation: at x = 0.3 (i = 1; omc 0.302)"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    der(x) = 1;
    for i in 1:2 loop
      if x > 0.3 * i then
        terminate("x above 0.3 i, i = " + String(i));
      end if;
    end for;
  end ForBranch;
  record SwapRec "a record a when swaps (review 20)"
    Real a;
    Real b;
  end SwapRec;
  function swapRec
    input SwapRec r;
    output SwapRec s;
  algorithm
    s.a := r.b;
    s.b := r.a;
  end swapRec;
  type SwapEnum = enumeration(one, two, three);
  model MixedSection "a regular Real swap and a when Real swap in one section"
    Real v[2](each start = 0, each fixed = true);
    Real y[2];
    Real z[2](start = {1, 2}, each fixed = true);
  equation
    der(v) = {1, 2};
  algorithm
    y := {v[1], v[2]};
    y := {y[2], y[1]};
    when sample(0.1, 0.25) then
      z := {z[2], z[1]};
    end when;
  end MixedSection;
  model ElemCycle "elements of one array read each other: a false cycle by array"
    Real v[2](each start = 0, each fixed = true);
    discrete Integer x[3](each start = 0, each fixed = true);
  equation
    der(v) = {1, 2};
    when sample(0.05, 0.1) then
      x[2] = x[1] + 1;
      x[1] = x[3] * 2;
      x[3] = pre(x[3]) + 1;
    end when;
  end ElemCycle;
  model IfGuardReassign "if n == 0 then n := 1; m := m + 1; else n := 0; end if; in a when: m = 2, n = 0 at 1"
    Real v[2](each start = 0, each fixed = true);
    discrete Integer n(start = 0, fixed = true);
    discrete Integer m(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
  algorithm
    when sample(0.1, 0.25) then
      if n == 0 then
        n := 1;
        m := m + 1;
      else
        n := 0;
      end if;
    end when;
  end IfGuardReassign;
  model IfGuardSwap "a swap under a guard reading the target: {1, 2} from the first tick"
    Real v[2](each start = 0, each fixed = true);
    discrete Integer x[2](start = {2, 1}, each fixed = true);
  equation
    der(v) = {1, 2};
  algorithm
    when sample(0.1, 0.25) then
      if x[1] > 1 then
        x := {x[2], x[1]};
      end if;
    end when;
  end IfGuardSwap;
  model IfGuardRegular "y := v[1]; if y < 10 then y := 20; z := 1; else z := 2; end if: z = 1"
    Real v[2](each start = 0, each fixed = true);
    Real y;
    Real z;
  equation
    der(v) = {1, 2};
  algorithm
    y := v[1];
    if y < 10 then
      y := 20;
      z := 1;
    else
      z := 2;
    end if;
  end IfGuardRegular;
  model IfGuardDiscreteRegular "n from a relation, then reassigned under a guard on it: m = 3, then 7"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
    Integer n;
    Integer m;
  equation
    der(v) = {1, 2};
    der(x) = 1;
  algorithm
    n := if x > 0.5 then 1 else 0;
    if n == 1 then
      n := 2;
      m := 7;
    else
      m := 3;
    end if;
  end IfGuardDiscreteRegular;
  model AlgAssertReassigned "an assert under an if on a value the section changes after it"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
    Integer n;
  equation
    der(v) = {1, 2};
    der(x) = 1;
  algorithm
    n := 1;
    if n == 1 then
      assert(x < 0.7, "x reached 0.7 where n is 1");
    end if;
    n := 0;
  end AlgAssertReassigned;
  model AlgAssertTopReassigned "a top-level assert on a value the section changes after it: holds (omc)"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
    Integer n;
  equation
    der(v) = {1, 2};
    der(x) = 1;
  algorithm
    n := 1;
    assert(n == 1, "n is not 1");
    n := 2;
  end AlgAssertTopReassigned;
  model AlgAssertFor "asserts in a for loop of an algorithm section: fails at 0.4"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
    Integer n;
  equation
    der(v) = {1, 2};
    der(x) = 1;
  algorithm
    n := 1;
    for i in 1:2 loop
      assert(x < 0.4 * i, "x reached 0.4 i");
    end for;
  end AlgAssertFor;
  model AlgTermFor "terminate() in an if in a for loop of an algorithm section: at 0.3"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
    Integer n;
  equation
    der(v) = {1, 2};
    der(x) = 1;
  algorithm
    n := 1;
    for i in 1:2 loop
      if x > 0.3 * i then
        terminate("x above 0.3 i");
      end if;
    end for;
  end AlgTermFor;
  model AssertBoolAlg "assert(ok) with ok = d == 0 and d set at 0.6"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
    discrete Integer d(start = 0, fixed = true);
    Boolean ok;
  equation
    der(v) = {1, 2};
    der(x) = 1;
    ok = d == 0;
    when x > 0.6 then
      d = 1;
    end when;
    assert(ok, "d set");
  end AssertBoolAlg;
  model AssertReinit "reinit over the bound at 0.5"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    der(x) = 1;
    when x > 0.5 then
      reinit(x, 0.9);
    end when;
    assert(x < 0.8, "x reached 0.8");
  end AssertReinit;
  model AssertReinitDiscrete "reinit and a discrete change at 0.5; the assert reads both: fails at 0.5"
    Real v[2](each start = 0, each fixed = true);
    Real x(start = 0, fixed = true);
    discrete Integer d(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
    der(x) = 1;
    when x > 0.5 then
      reinit(x, 0.9);
      d = 1;
    end when;
    assert(x < 0.8 or d == 0, "x reached 0.8 with d set");
  end AssertReinitDiscrete;
  model InitArmCount "n := n + 1 in when {initial(), sample}: n(0) = 1, n(1) = 5"
    Real v[2](each start = 0, each fixed = true);
    discrete Integer n(start = 0, fixed = true);
  equation
    der(v) = {1, 2};
  algorithm
    when {initial(), sample(0.1, 0.25)} then
      n := n + 1;
    end when;
  end InitArmCount;
  model InitialArmSwap "a swap in when {initial(), sample}: swapped at the start"
    Real v[2](each start = 0, each fixed = true);
    discrete Integer x[2](start = {1, 2}, each fixed = true);
  equation
    der(v) = {1, 2};
  algorithm
    when {initial(), sample(0.1, 0.25)} then
      x := {x[2], x[1]};
    end when;
  end InitialArmSwap;
  model InitWhenSwap "a swap in when initial(): x = {2, 1} throughout"
    Real v[2](each start = 0, each fixed = true);
    discrete Integer x[2](start = {1, 2}, each fixed = true);
  equation
    der(v) = {1, 2};
  algorithm
    when initial() then
      x := {x[2], x[1]};
    end when;
  end InitWhenSwap;
  model InitAlgSwap "a swap in an initial algorithm of states: x(0) = {2, 1}"
    Real v[2](each start = 0, each fixed = true);
    Real x[2](start = {1, 2});
  equation
    der(v) = {1, 2};
    der(x) = {0, 0};
  initial algorithm
    x := {1, 2};
    x := {x[2], x[1]};
  end InitAlgSwap;
  model MixedRealInt "a regular section: Real and Integer self-reading swaps"
    Real v[2](each start = 0, each fixed = true);
    Real y[2];
    Integer n[2];
  equation
    der(v) = {1, 2};
  algorithm
    y := {v[1], v[2]};
    y := {y[2], y[1]};
    n := {integer(v[1] * 10), integer(v[2] * 10)};
    n := {n[2], n[1]};
  end MixedRealInt;
  model IndexSwap "x := x[{2, 1}] in a when"
    Real v[2](each start = 0, each fixed = true);
    discrete Integer x[2](start = {1, 2}, each fixed = true);
  equation
    der(v) = {1, 2};
  algorithm
    when sample(0.1, 0.25) then
      x := x[{2, 1}];
    end when;
  end IndexSwap;
  model RecordSwapFunc "a record swapped by a function in a when"
    Real v[2](each start = 0, each fixed = true);
    discrete SwapRec r(a(start = 1, fixed = true), b(start = 2, fixed = true));
  equation
    der(v) = {1, 2};
  algorithm
    when sample(0.1, 0.25) then
      r := swapRec(r);
    end when;
  end RecordSwapFunc;
  model RecordSwapCtor "a record swapped by its constructor in a when"
    Real v[2](each start = 0, each fixed = true);
    discrete SwapRec r(a(start = 1, fixed = true), b(start = 2, fixed = true));
  equation
    der(v) = {1, 2};
  algorithm
    when sample(0.1, 0.25) then
      r := SwapRec(a = r.b, b = r.a);
    end when;
  end RecordSwapCtor;
  model EnumSwap "an enumeration array swapped in a when"
    Real v[2](each start = 0, each fixed = true);
    discrete SwapEnum e[2](start = {SwapEnum.one, SwapEnum.three}, each fixed = true);
    Integer i1 = Integer(e[1]);
  equation
    der(v) = {1, 2};
  algorithm
    when sample(0.1, 0.25) then
      e := {e[2], e[1]};
    end when;
  end EnumSwap;
  model VarSubscript "x[k] := x[3 - k] + 10 with k alternating"
    Real v[2](each start = 0, each fixed = true);
    discrete Integer k(start = 1, fixed = true);
    discrete Integer x[2](start = {1, 2}, each fixed = true);
  equation
    der(v) = {1, 2};
  algorithm
    when sample(0.1, 0.25) then
      x[k] := x[3 - k] + 10;
      k := 3 - k;
    end when;
  end VarSubscript;
  model EqInteger "an equation n = integer(10 x): events at each crossing"
    Real x(start = 0, fixed = true);
    Integer n;
  equation
    der(x) = 1;
    n = integer(x * 10);
  end EqInteger;
  model EqIntegerReal "a Real y = floor(10 x): events at each crossing"
    Real x(start = 0, fixed = true);
    Real y;
  equation
    der(x) = 1;
    y = floor(x * 10);
  end EqIntegerReal;
  model AlgIntegerTwo "integer() in a two-statement discrete section"
    Real x(start = 0, fixed = true);
    Integer n;
    Integer m;
  equation
    der(x) = 1;
  algorithm
    n := integer(x * 10);
    m := n + 1;
  end AlgIntegerTwo;
  model AlgIntegerReadsAssigned "integer() of a value assigned before it in the section"
    Real x(start = 0, fixed = true);
    Real y;
    Integer n;
  equation
    der(x) = 1;
  algorithm
    y := 2 * x;
    n := integer(y * 10);
  end AlgIntegerReadsAssigned;
  model EventFns "ceil, div, mod, rem of an oscillating argument, both signs"
    Real x(start = 1, fixed = true);
    Real v(start = 0, fixed = true);
    Real c, d, m, r;
  equation
    der(x) = v;
    der(v) = -39.47841760435743 * x;
    c = ceil(3.7 * x);
    d = div(3.7 * x, 1.3);
    m = mod(3.7 * x, 1.3);
    r = rem(3.7 * x, 1.3);
  end EventFns;
  model AlgEventFns "Integer targets of event functions in an algorithm section, a nested one"
    Real x(start = 1, fixed = true);
    Real v(start = 0, fixed = true);
    Integer i, j, k;
  equation
    der(x) = v;
    der(v) = -39.47841760435743 * x;
  algorithm
    i := integer(ceil(3.7 * x));
    j := integer(div(3.7 * x, 1.3));
    k := integer(10 * rem(3.7 * x, 1.3));
  end AlgEventFns;
  model EnumStart "discrete enumeration starts: Integer(e1) 1, Integer(e2) 2"
    discrete SwapEnum e1(start = SwapEnum.one, fixed = true);
    discrete SwapEnum e2(start = SwapEnum.two, fixed = true);
    Integer i1 = Integer(e1);
    Integer i2 = Integer(e2);
    Real x(start = 0, fixed = true);
  equation
    der(x) = 1;
    when time > 0.5 then
      e1 = SwapEnum.three;
      e2 = pre(e1);
    end when;
  end EnumStart;
  model TermDiscreteGuard "terminate() on a discrete the section computes from a relation: at 0.5 (omc)"
    Real x(start = 0, fixed = true);
    Integer n;
  equation
    der(x) = 1;
  algorithm
    n := if x > 0.5 then 5 else 0;
    if n >= 5 then
      terminate("n reached 5");
    end if;
  end TermDiscreteGuard;
  model TermBeforeLastAssignment "terminate() on a discrete the section assigns again later: at 0.5, n = 6 (omc)"
    Real x(start = 0, fixed = true);
    Integer n;
  equation
    der(x) = 1;
  algorithm
    n := if x > 0.5 then 5 else 0;
    if n >= 5 then
      terminate("n reached five before its final assignment");
    end if;
    n := n + 1;
  end TermBeforeLastAssignment;
  model TermRealThreshold "TermBeforeLastAssignment's guard against a Real literal: at 0.5, n = 6 (omc)"
    Real x(start = 0, fixed = true);
    Integer n;
  equation
    der(x) = 1;
  algorithm
    n := if x > 0.5 then 5 else 0;
    if n >= 5.0 then
      terminate("intermediate n reached five");
    end if;
    n := n + 1;
  end TermRealThreshold;
  model TermParamThreshold "... against a Real parameter: at 0.5, n = 6 (omc)"
    parameter Real th = 5;
    Real x(start = 0, fixed = true);
    Integer n;
  equation
    der(x) = 1;
  algorithm
    n := if x > 0.5 then 5 else 0;
    if n >= th then
      terminate("intermediate n reached the threshold");
    end if;
    n := n + 1;
  end TermParamThreshold;
  model TermRealBranches "... of a Real n of Real branches: at 0.5, n = 6 (MLS 11.1.2; omc runs on: its hysteresis on a value landing on its threshold)"
    Real x(start = 0, fixed = true);
    Real n;
  equation
    der(x) = 1;
  algorithm
    n := if x > 0.5 then 5.0 else 0.0;
    if n >= 5.0 then
      terminate("intermediate n reached five");
    end if;
    n := n + 1.0;
  end TermRealBranches;
  model OpaqueIntegerLanding "an Integer function result landing on its threshold at a relation's event: y = 1 after 0.5 (omc)"
    function level
      input Real u;
      input Boolean high;
      output Integer n;
    algorithm
      assert(u >= -1.0, "u outside the supported range");
      n := if high then 5 else 0;
      annotation(Inline = false);
    end level;
    Real x(start = 0, fixed = true);
    Real y;
  equation
    der(x) = 1;
    y = if level(x, x > 0.5) >= 5 then 1.0 else 0.0;
  end OpaqueIntegerLanding;
  model OpaqueSumParam "... through arithmetic with a Real parameter (the Integer promoted): y = 1 after 0.5 (omc)"
    function level
      input Real u;
      input Boolean high;
      output Integer n;
    algorithm
      assert(u >= -1.0, "u outside the supported range");
      n := if high then 5 else 0;
      annotation(Inline = false);
    end level;
    parameter Real p = 0.5;
    Real x(start = 0, fixed = true);
    Real y;
  equation
    der(x) = 1;
    y = if level(x, x > 0.5) + p >= 5.5 then 1.0 else 0.0;
  end OpaqueSumParam;
  model OpaqueNegated "... through a sign change and a Real literal: y = 1 after 0.5 (omc)"
    function level
      input Real u;
      input Boolean high;
      output Integer n;
    algorithm
      assert(u >= -1.0, "u outside the supported range");
      n := if high then 5 else 0;
      annotation(Inline = false);
    end level;
    Real x(start = 0, fixed = true);
    Real y;
  equation
    der(x) = 1;
    y = if -level(x, x > 0.5) <= -5.0 then 1.0 else 0.0;
  end OpaqueNegated;
  model TermOpaqueInteger "terminate() on an intermediate opaque Integer against a Real literal (the frontend casts it): at 0.5, n = 6 (omc)"
    function level
      input Real u;
      input Boolean high;
      output Integer n;
    algorithm
      assert(u >= -1.0, "u outside the supported range");
      n := if high then 5 else 0;
      annotation(Inline = false);
    end level;
    Real x(start = 0, fixed = true);
    Integer n;
  equation
    der(x) = 1;
  algorithm
    n := level(x, x > 0.5);
    if n >= 5.0 then
      terminate("intermediate Integer result reached five");
    end if;
    n := n + 1;
  end TermOpaqueInteger;
  model TermOpaqueIntegerAbs "terminate() on abs() of the intermediate opaque Integer plus a Real: at 0.5, n = 6 (omc)"
    function level
      input Real u;
      input Boolean high;
      output Integer n;
    algorithm
      assert(u >= -1.0, "u outside the supported range");
      n := if high then 5 else 0;
      annotation(Inline = false);
    end level;
    parameter Real p = 0.5;
    Real x(start = 0, fixed = true);
    Integer n;
  equation
    der(x) = 1;
  algorithm
    n := level(x, x > 0.5);
    if abs(n + p) >= 5.5 then
      terminate("intermediate value reached its limit");
    end if;
    n := n + 1;
  end TermOpaqueIntegerAbs;
  model IfNoEventCondition "an if-expression of a noEvent condition in a when condition: the when fires after 0.5 (MLS 8.5; omc: never)"
    Real x(start = 0, fixed = true);
    Integer hits(start = 0, fixed = true);
  equation
    der(x) = 1;
    when (if noEvent(x > 0.5) then 5.0 else 0.0) >= 4.0 then
      hits = pre(hits) + 1;
    end when;
  end IfNoEventCondition;
  model IntegerFunctionNeedsCrossing "an Integer function of a continuous argument in a when condition: it fires after 0.5 (MLS 8.5; omc: never)"
    function stepValue
      input Real u;
      output Integer k;
    algorithm
      assert(u >= -1.0, "u outside the supported range");
      k := if u > 0.5 then 2 else 0;
      annotation(Inline = false);
    end stepValue;
    Real x(start = 0, fixed = true);
    Integer hits(start = 0, fixed = true);
  equation
    der(x) = 1;
    when stepValue(x) >= 1.5 then
      hits = pre(hits) + 1;
    end when;
  end IntegerFunctionNeedsCrossing;
  model RoundedIntermediate "terminate() on a rounded sum of the intermediate (0.1 + 0.2 >= 0.3): at 0.5, n = 2 (omc)"
    parameter Real a = 0.1, b = 0.2, thr = 0.3;
    Real x(start = 0, fixed = true);
    Integer n;
  equation
    der(x) = 1;
  algorithm
    n := if x > 0.5 then 1 else 0;
    if a * n + b * n >= thr then
      terminate("rounded intermediate reached threshold");
    end if;
    n := n + 1;
  end RoundedIntermediate;
  model RoundedWhen "a when condition of a rounded sum: hits = 1 after 0.5 (omc)"
    parameter Real a = 0.1, b = 0.2, thr = 0.3;
    Real x(start = 0, fixed = true);
    Integer hits(start = 0, fixed = true);
  equation
    der(x) = 1;
    when (if x > 0.5 then a + b else 0.0) >= thr then
      hits = pre(hits) + 1;
    end when;
  end RoundedWhen;
  model StrictLanding "a strict relation on a plateau of exactly its threshold: hits = 1 at 0.25, 2 at 0.75 (omc)"
    Real x(start = 0, fixed = true);
    Integer hits(start = 0, fixed = true);
  equation
    der(x) = 1;
    when (if x > 0.75 then 6.0 elseif x > 0.5 then 5.0 elseif x > 0.25 then 6.0 else 0.0) > 5.0 then
      hits = pre(hits) + 1;
    end when;
  end StrictLanding;
  model WideIntegerIntermediate "terminate() on a 64-bit Integer intermediate: the false guard's difference does not wrap around: at 0.5, n = 6000000000000001024"
    function level
      input Real u;
      input Boolean high;
      input Integer base;
      output Integer n;
    algorithm
      assert(u >= -1.0, "u outside the supported range");
      n := 6 * base * base;
      if not high then
        n := -n;
      end if;
      annotation(Inline = false);
    end level;
    Real x(start = 0, fixed = true);
    Integer n;
  equation
    der(x) = 1;
  algorithm
    n := level(x, x > 0.5, 1000000000);
    if n >= 4 * 1000000000 * 1000000000 then
      terminate("wide intermediate reached its limit");
    end if;
    n := n + 1024;
  end WideIntegerIntermediate;
  model InactiveElseif "an elseif condition evaluated only where the earlier ones are false (sqrt of a negative d): d, n = -10, 9 at 0.95 (omc)"
    Real x(start = 0, fixed = true);
    discrete Real d(start = -1, fixed = true);
    Integer n(start = 0, fixed = true);
  equation
    der(x) = 1;
  algorithm
    when sample(0.1, 0.1) then
      if d < 0 then
        d := d - 1;
        n := pre(n) + 1;
      elseif sqrt(d) > 0 then
        d := 1;
        n := pre(n) + 2;
      else
        d := 2;
        n := pre(n) + 3;
      end if;
    end when;
  end InactiveElseif;
  model GuardedSqrt "a partial function in the branch not taken of a when's if: y = 9 at 0.95 (omc)"
    Real x(start = 0, fixed = true);
    Real d = -1 + 0 * x;
    Real y(start = 0, fixed = true);
  equation
    der(x) = 1;
  algorithm
    when sample(0.1, 0.1) then
      if d > 0 then
        y := sqrt(d);
      else
        y := pre(y) + 1;
      end if;
    end when;
  end GuardedSqrt;
  model StepThroughIfEq"round(u) of u from an if-equation (MSL IntegerNetwork1's RealToInteger): |y| <= 3, omc"
    Real u;
    Integer y;
  equation
    u = if time < 0 then 0 else 3 * sin(2 * 3.141592653589793 * 0.1 * time);
    y = if u > 0 then integer(floor(u + 0.5)) else integer(ceil(u - 0.5));
  end StepThroughIfEq;
  model StepChain "integer(floor(q)) against integer(q), a when on its change: n = m = cnt = 1, 2, 4 (omc)"
    Real x(start = 0, fixed = true);
    Integer n;
    Integer m;
    Integer dlt;
    Integer cnt(start = 0, fixed = true);
  equation
    der(x) = 1;
    n = integer(floor(x * 5));
    m = integer(x * 5);
    dlt = n - m;
    when change(n) then
      cnt = pre(cnt) + 1;
    end when;
  end StepChain;
  model StepOfDer "integer() of der(x): the scalarized route keeps it inline (no events of its own)"
    Real x(start = 0, fixed = true);
    Integer n;
  equation
    der(x) = 10 * time;
    n = integer(der(x));
  end StepOfDer;
end ArrayPath;
