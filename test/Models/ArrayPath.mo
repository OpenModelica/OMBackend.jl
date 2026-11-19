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

  model AlgebraicLoop "a loop in a model without arrays: the array code's (y = 0.75 x, z = 0.25 x, x' = -0.75 x)"
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
  model LoopWithEvent "a loop per element through a relation (events in a loop)"
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
  model AbsWindow "a relation on abs() true only in a window of nanoseconds around each zero of x: 5 in 50 ms (MLS 8.5; omc misses them)"
    Real x(start = 1, fixed = true);
    Real y(start = 0, fixed = true);
    Boolean small;
    Integer n(start = 0, fixed = true);
  equation
    der(x) = y;
    der(y) = -(2 * 3.141592653589793 * 50) ^ 2 * x;
    small = abs(x) <= 1.0e-7;
    when small then
      n = pre(n) + 1;
    end when;
  end AbsWindow;
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
  connector SlicePin
    Real v;
    flow Real i;
  end SlicePin;
  connector SlicePlug
    parameter Integer m = 3;
    SlicePin pin[m];
  end SlicePlug;
  model SliceSource
    parameter Integer m = 3;
    SlicePlug plug(m = m);
  equation
    plug.pin.v = {sin(time + k) for k in 1:m};
  end SliceSource;
  model SliceLoad
    parameter Integer m = 3;
    parameter Real R[m] = {1.0, 2.0, 4.0};
    SlicePlug ac(m = m);
    Real vAC[m] = ac.pin[:].v;
    Real iAC[m] = ac.pin[:].i;
    Real q[m](each start = 0.0, each fixed = true);
  equation
    vAC = R .* iAC;
    der(q) = iAC;
  end SliceLoad;
  model SliceConnectors "ac.pin[:].i, a field of a connector array: q[k] = (cos(k) - cos(k + 1)) / R[k] at 1"
    SliceSource source;
    SliceLoad load;
  equation
    connect(source.plug, load.ac);
  end SliceConnectors;
  model SliceCell
    parameter Real k = 1.0;
    Real x(start = 1.0, fixed = true);
  equation
    der(x) = -k * x;
  end SliceCell;
  model SliceCells "a field of a model array through a whole slice (c[:].x): s = 2 sum((1 - exp(-k)) / k) at 1"
    parameter Integer n = 3;
    SliceCell c[n](k = {1.0, 2.0, 3.0});
    Real y[n];
    Real s(start = 0.0, fixed = true);
  equation
    y = 2.0 * c[:].x;
    der(s) = sum(y);
  end SliceCells;
  model DelayCell "MSL InertialDelaySensitive, reduced: an initial() arm, a change() arm, an assignment after"
    input Integer x(start = 1, fixed = true);
    output Integer y(start = y0, fixed = true);
    parameter Real tD = 0.1;
    parameter Integer y0 = 1;
  protected
    Integer y_aux(start = y0, fixed = true);
    Integer y_old(start = y0, fixed = true);
    discrete Real t_next;
  algorithm
    when {initial(), change(x) and not initial()} then
      y_old := if initial() or pre(y) == 1 then y0 else pre(y);
      t_next := time + (if x > y_old then tD else 0.0);
      if x <= y_old then
        y_aux := x;
      end if;
    elsewhen time >= t_next then
      y_aux := x;
    end when;
    y := y_aux;
  end DelayCell;
  model DelayPair "two instances chained by an equation: x's change from its start fires after the initialization"
    DelayCell d1(tD = 0.1);
    DelayCell d2(tD = 0.2);
    Integer u = if time < 0.3 then 2 else 3;
    Real z(start = 0, fixed = true);
  equation
    der(z) = 1;
    d1.x = u;
    d2.x = d1.y;
  end DelayPair;
  model SubscriptOutside "a run-time subscript leaving its table at 0.6: an error, not a read outside it"
    constant Integer tab[2] = {5, 7};
    Real v(start = 0, fixed = true);
    discrete Integer k(start = 1, fixed = true);
    discrete Integer x(start = 0, fixed = true);
  equation
    der(v) = 1;
  algorithm
    when sample(0.1, 0.25) then
      x := tab[k];
      k := k + 1;
    end when;
  end SubscriptOutside;
  model EnumDefaultStart "an enumeration without a start: its first literal, a table's subscript (i = 10, 30)"
    constant Integer tab[SwapEnum] = {10, 20, 30};
    discrete SwapEnum e[2];
    Integer i[2];
    Real x(start = 0, fixed = true);
  equation
    der(x) = 1;
    when time > 0.5 then
      e[1] = SwapEnum.three;
      e[2] = SwapEnum.two;
    end when;
  algorithm
    for j in 1:2 loop
      i[j] := tab[e[j]];
    end for;
  end EnumDefaultStart;
  model BindCell "a declaration binding of a field (b)"
    parameter Real k = 1.0;
    Real a(start = 1.0, fixed = true);
    Real b = 2.0 * k * a;
  equation
    der(a) = -b;
  end BindCell;
  model BindCells "bindings of fields of a model array stay on the merged arrays: a[k] = exp(-2 k t)"
    BindCell c[3](k = {1.0, 2.0, 3.0});
    Real y[3] = 3.0 * c.a;
  end BindCells;
  model ModCell "an array field (r) for a modifier"
    Real r[2];
    Real x(start = 1.0, fixed = true);
  equation
    der(x) = -(r[1] + r[2]) * x;
  end ModCell;
  model ModCells "a modifier over a model array on an array field, per element: c[k].x = exp(-(2 k + 1) t)"
    parameter Real p[3] = {1.0, 2.0, 3.0};
    ModCell c[3](r = {{p[i], p[i] + 1.0} for i in 1:3});
  end ModCells;
  model ModCellsMatrix "the same with a matrix as the value"
    ModCell c[3](r = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}});
  end ModCellsMatrix;
  model WCell "an array field (w) bound in the class"
    Real a(start = 1.0, fixed = true);
    Real w[2] = {a, 2.0 * a};
  equation
    der(a) = -w[2];
  end WCell;
  model WCells "an array field of a model array bound in its class: c[k].a = exp(-2 t)"
    WCell c[3];
  end WCells;
  record Turn "a record with array fields"
    Real T[2, 2];
    Real w[2];
  end Turn;
  model TurnCell
    input Turn R;
    Real x(start = 1.0, fixed = true);
  equation
    der(x) = -R.w[2] * R.T[1, 1] * x;
  end TurnCell;
  model TurnCells "a record with array fields in a model array (c.R.T is Real[3, 2, 2]): c[k].x = exp(-2 t)"
    Turn R;
    TurnCell c[3](each R = R);
  equation
    R.T = {{1.0, 0.0}, {0.0, 1.0}};
    R.w = {time, 2.0};
  end TurnCells;
  model FixedDummy "a fixed start on a dummy state: y[i] = 2 i exp(-t), from x[i] = i at the start"
    Real x[2](start = {1, 2}, each fixed = true);
    Real y[2](each stateSelect = StateSelect.prefer);
    Real z[2];
  equation
    for i in 1:2 loop
      x[i] + y[i] = 3 * i;
      der(y[i]) = -y[i];
      der(x[i]) = z[i];
    end for;
  end FixedDummy;

  model NestedIf "an if-equation in a branch of an if-equation on a variable condition"
    Real x(start = 1, fixed = true);
    Real y[2];
  equation
    der(x) = -x;
    for i in 1:2 loop
      if time < 0.5 then
        y[i] = i * x;
      else
        if x > 0.1 then
          y[i] = 2 * i * x;
        else
          y[i] = 0;
        end if;
      end if;
    end for;
  end NestedIf;

  model ElementForms "elements of smooth() and vector(), an if-equation on a function of parameters: x = exp(-t)"
    parameter Real n[3] = {0.0, -1.0, 0.0};
    Real e[3];
    Real v[3];
    Real x(start = 1.0, fixed = true);
    Real y;
  equation
    e = smooth(0, if x > -1.0 then n * x else n);
    v = vector(cat(1, {{x}, {2.0 * x}}, {{0.0}}));
    if norm2(n) > 0.5 then
      y = 2.0 * x;
    else
      y = x;
    end if;
    der(x) = e[2] - v[2] + y;
  end ElementForms;
  record Turned
    Real T[3, 3];
    Real w[3];
  end Turned;

  function resolveBack "v in the frame R is turned from"
    input Turned R;
    input Real v[3];
    output Real u[3];
  algorithm
    u := transpose(R.T) * v;
    annotation(Inline = true);
  end resolveBack;

  model SlidingMass "a mass on a turning slide: R.T[1, i] and R.T[3, i] times parameters known as zero (e[1], e[3])"
    parameter Real e[3] = {0, 1, 0} "the slide's axis, in the turning frame";
    parameter Real g = 9.81;
    parameter Real phi0 = 0.5;
    Real phi;
    Turned R "the frame, turned by phi about z";
    Real s(start = 1, fixed = true) "s = (g/2) (cos(phi) + sin(1/2) sinh(t)) + (1 - (g/2) cos(1/2)) cosh(t)";
    Real v(start = 0, fixed = true);
    Real r[3], vr[3], f[3];
  equation
    phi = time + phi0;
    R = Turned(T = {{cos(phi), sin(phi), 0}, {-sin(phi), cos(phi), 0}, {0, 0, 1}}, w = {0, 0, der(phi)});
    r = resolveBack(R, e * s);
    v = der(s);
    der(r) = vr;
    der(vr) = f + {0, -g, 0};
    R.T[2, 1] * f[1] + R.T[2, 2] * f[2] + R.T[2, 3] * f[3] = 0;
  end SlidingMass;

  model SlidingMassAligned "the slide along y at the start: R.T[2, 1] = 0 there; s = (g/2) cos(t) + (1 - g/2) cosh(t)"
    extends SlidingMass(phi0 = 0);
  end SlidingMassAligned;

  function initialResidue "q - q0 per element, inlined (as MSL Frames.Quaternions.Orientation.equalityConstraint)"
    input Real q[2];
    input Real q0[2];
    output Real r[2];
  algorithm
    r := {q[1] - q0[1], q[2] - q0[2]};
    annotation(Inline = true);
  end initialResidue;

  function initialResidueLoop "initialResidue by a loop: not inlined, called per element"
    input Real q[2];
    input Real q0[2];
    output Real r[2];
  algorithm
    for i in 1:2 loop
      r[i] := q[i] - q0[i];
    end for;
  end initialResidueLoop;

  model ArrayInitialRefs "an initial array equation of references: q(1) = q0 exp(-1)"
    parameter Real q0[2] = {1, 2};
    Real q[2];
  initial equation
    q = q0;
  equation
    der(q) = -q;
  end ArrayInitialRefs;

  model ArrayInitialCall "an initial array equation of an inlined call against a literal (MSL Body: zeros(3) = equalityConstraint(Q, Q_start))"
    parameter Real q0[2] = {1, 2};
    Real q[2];
  initial equation
    zeros(2) = initialResidue(q, q0);
  equation
    der(q) = -q;
  end ArrayInitialCall;

  model ArrayInitialLoopCall "ArrayInitialCall, the call's body a loop (not inlined)"
    parameter Real q0[2] = {1, 2};
    Real q[2];
  initial equation
    zeros(2) = initialResidueLoop(q, q0);
  equation
    der(q) = -q;
  end ArrayInitialLoopCall;

  model ArrayInitialFree "initial array equations that no residual solves: of a parameter array with fixed = false, of a discrete array (assignments of the initialization, per element)"
    parameter Real p[2](each fixed = false, each start = 0);
    discrete Real d[2];
    Real q[2](each start = 1, each fixed = true);
  initial equation
    p = {1, 2};
    d = {3, 4};
  equation
    der(q) = -p .* q;
    when time > 0.5 then
      d[1] = 2 * pre(d[1]);
      d[2] = 2 * pre(d[2]);
    end when;
  end ArrayInitialFree;

  model StepAfterEvent "a fast transient right after an event: y follows u with Td = 1e-4 and u stops rising at 0.2, so a = der(y) falls from 20 to 0 within a millisecond; the step after the event starts small"
    parameter Real Td = 1e-4;
    Real u = if time < 0.2 then 20 * time else 4;
    Real y(start = 0, fixed = true);
    Real a = der(y);
  equation
    der(y) = (u - y) / Td;
  end StepAfterEvent;

  model HighGainKink "a state that follows its input through a gain of 15000 into a limit without an event (an op-amp differentiator): the input's slope rises at 0.1, the output is at its limit 2e-6 s later; the solver's own estimate of a first step after the event is 0.01 s"
    parameter Real V0 = 15000;
    parameter Real RC = 0.05;
    Real vin = if time < 0.1 then 0.3 + 0.1 * time else 0.31 + 500 * (time - 0.1);
    Real x(start = 0.3, fixed = true);
    Real vn = vin - x;
    Real vout = smooth(0, noEvent(if V0 * vn > 15 then -15 else if V0 * vn < -15 then 15 else -V0 * vn));
  equation
    RC * der(x) = vn - vout;
  end HighGainKink;

  model SingularStart "an algebraic loop through the elements whose Jacobian is singular at the start values: (v[k] + i[k])^3 = P has no derivative at v = i = 0; one solution, v + i = 2, i = (E - 2) / (1 + 5 c)"
    parameter Real E = 3;
    parameter Real P = 8;
    parameter Real c = 0.01;
    Real v[5];
    Real i[5];
    Real s;
    Real x(start = 0, fixed = true);
  equation
    s = c * (i[1] + i[2] + i[3] + i[4] + i[5]);
    for k in 1:5 loop
      (v[k] + i[k])^3 = P;
      v[k] + 2 * i[k] = E - s;
    end for;
    der(x) = i[1] - x;
  end SingularStart;

  model ZeroCoefficient "an unknown times a parameter that is zero (a resistance of 0 in series): i[k] * R[k] = v[k] - vi[k] is not solved for i[k] where R[k] = 0; v' = -v / 2 there, -v / 3 with R = 1"
    parameter Real R[3] = {0, 0, 1};
    Real v[3](each start = 1, each fixed = true);
    Real vi[3];
    Real i[3];
  equation
    for k in 1:3 loop
      i[k] * R[k] = v[k] - vi[k];
      i[k] = vi[k] / 2;
      der(v[k]) = -i[k];
    end for;
  end ZeroCoefficient;

  function swapStep "two outputs, the second an array"
    input Real s[2];
    output Real r;
    output Real t[2];
  algorithm
    r := s[1] + s[2];
    t := {s[2], s[1] + 1};
  end swapStep;

  model WhenArrayTarget "a when-equation that assigns a whole array (not array code: scalarized by the backend, an assignment per element); d doubled at 0.5"
    discrete Real d[2](start = {3, 4}, each fixed = true);
    Real x(start = 0, fixed = true);
  equation
    der(x) = 1;
    when x > 0.5 then
      d = 2 * pre(d);
    end when;
  end WhenArrayTarget;

  model WhenIfArrayTarget "WhenArrayTarget, the array assigned in the branches of an if: d + 1 at 0.25, doubled at 0.75"
    discrete Real d[2](start = {3, 4}, each fixed = true);
    Real x(start = 0, fixed = true);
  equation
    der(x) = 1;
    when {x > 0.25, x > 0.75} then
      if x > 0.5 then
        d = 2 * pre(d);
      else
        d = pre(d) + {1, 1};
      end if;
    end when;
  end WhenIfArrayTarget;

  model TupleArrayTarget "an algorithm's tuple with an array target (the reference kept as the frontend's scalarization has it); at 0.25, 0.5, 0.75: r = 3, 4, 5, s = {2, 2}, {2, 3}, {3, 3}"
    discrete Real s[2](start = {1, 2}, each fixed = true);
    discrete Real r(start = 0, fixed = true);
    Real x(start = 0, fixed = true);
  equation
    der(x) = 1;
  algorithm
    when sample(0.25, 0.25) then
      (r, s) := swapStep(pre(s));
    end when;
  end TupleArrayTarget;

  function seedPair "a start for a pair of states from a seed (large, as a random generator's)"
    input Integer seed;
    output Integer s[2];
  algorithm
    for k in 1:2 loop
      s[k] := seed + k - 1;
    end for;
    s[2] := s[2] + 2000000000;
  end seedPair;

  function nextPair "two outputs, the second an Integer array"
    input Integer s[2];
    output Real r;
    output Integer t[2];
  algorithm
    r := s[1] + s[2];
    t := {s[2], s[1] + 1};
  end nextPair;

  model PreArrayStart "the start of pre() of an array by an initial array equation (MSL Noise: pre(state) = initialState(...)); its elements pre(s[k]) = seedPair(seed)[k] in this orientation for the initialization"
    parameter Integer seed = 1;
    discrete Integer s[2];
    discrete Real r(start = 0, fixed = true);
    Real x(start = 0, fixed = true);
  initial equation
    pre(s) = seedPair(seed);
  equation
    der(x) = 1;
  algorithm
    when sample(0.25, 0.25) then
      (r, s) := nextPair(pre(s));
    end when;
  end PreArrayStart;

  model RecordForBody "a Complex equation in a for body, arrays kept: z[k] = Complex(x[k], -x[k]); x' = -1.5 x"
    parameter Integer m = 3;
    Complex z[m];
    Real x[m](each start = 1, each fixed = true);
  equation
    for k in 1:m loop
      z[k] = Complex(x[k], -x[k]);
      der(x[k]) = -z[k].re + 0.5 * z[k].im;
    end for;
  end RecordForBody;

  model RecordForSum "Complex arithmetic in a for body: w = z + z j with z = x (1 + 2 j); x' = -(w.re + w.im) = -2 x"
    parameter Integer m = 3;
    Complex z[m];
    Complex w[m];
    Real x[m](each start = 1, each fixed = true);
  equation
    for k in 1:m loop
      z[k] = Complex(x[k], 2 * x[k]);
      w[k] = z[k] + z[k] * Complex(0, 1);
      der(x[k]) = -w[k].re - w[k].im;
    end for;
  end RecordForSum;

  model RecordForIf "a Complex equation in a for body with an if-expression on the iterator (MSL PlugToPin): i[2] = -i0, the others 0; x' = -x"
    parameter Integer m = 3;
    parameter Integer k0 = 2;
    Complex i0;
    Complex i[m];
    Real x(start = 1, fixed = true);
  equation
    i0 = Complex(x, -x);
    for j in 1:m loop
      i[j] = if j == k0 then -i0 else Complex(0);
    end for;
    der(x) = i[2].re + i[1].im;
  end RecordForIf;

  model ComplexResistor "v = R i of Complex values"
    Complex v;
    Complex i;
    parameter Real R = 2;
  equation
    v = R * i;
  end ComplexResistor;

  model RecordComponents "a component array with a Complex equation (a for-equation with the arrays kept); x' = -x (1 + 1/2 + 1/4)"
    ComplexResistor r[3](R = {1, 2, 4});
    Real x(start = 1, fixed = true);
  equation
    for k in 1:3 loop
      r[k].v = Complex(x, 0);
    end for;
    der(x) = -(r[1].i.re + r[2].i.re + r[3].i.re);
  end RecordComponents;

  model ComplexArrays "array equations of Complex values: of two arrays (by their fields' arrays), of an array constructor (a for-equation of the parts), of the elements written out (an equation per element and part); i = u j, x' = -i.im = -x"
    parameter Integer m = 3;
    Complex v[m];
    Complex u[m];
    Complex i[m];
    Complex s[m];
    Real x[m](each start = 1, each fixed = true);
  equation
    v = {Complex(x[k], 0) for k in 1:m};
    u = v;
    i = {u[k] * Complex(0, 1) for k in 1:m};
    s = {i[1] + u[1], i[2] + u[2], i[3] + u[3]};
    for k in 1:m loop
      der(x[k]) = -s[k].im;
    end for;
  end ComplexArrays;

  model DerivativeTwoCycle "a stray inductance in series with a main one (MSL Machines' air gaps, per axis): psi = Lm is is no state; Ls der(is) + der(psi) = u - R is and der(psi) = Lm der(is) read each other's derivative, one solved with the other's value put in; (Ls + Lm) is' = u - R is"
    parameter Real Ls = 0.1;
    parameter Real Lm = 1;
    parameter Real R = 2;
    Real is[2](each start = 0, each fixed = true);
    Real psi[2];
  equation
    Ls * der(is) + der(psi) = {sin(time), cos(time)} - R * is;
    psi = Lm * is;
  end DerivativeTwoCycle;

  model ElementwisePowerSum "the sum of an element-wise power of an array (MSL Machines' quasiRMS: sqrt(sum(i .^ 2 / m))); i[k] = k x, x' = -sqrt(14 / 3) x"
    parameter Integer m = 3;
    Real i[m];
    Real x(start = 1, fixed = true);
  equation
    for k in 1:m loop
      i[k] = k * x;
    end for;
    der(x) = -sqrt(sum(i .^ 2 / m));
  end ElementwisePowerSum;

  model ScalarLoop "a nonlinear loop of scalars (no arrays): y + z^3 = x, y - z = x / 2"
    Real x(start = 1, fixed = true);
    Real y(start = 1);
    Real z;
  equation
    der(x) = -y;
    y + z^3 = x;
    y - z = 0.5 * x;
  end ScalarLoop;

  model ScalarIndexTwo "two masses rigidly coupled, scalars (no arrays, index above one): (m1 + m2) x1'' = -k x1"
    parameter Real m1 = 1;
    parameter Real m2 = 2;
    parameter Real k = 10;
    Real x1(start = 1, fixed = true);
    Real v1(start = 0, fixed = true);
    Real x2;
    Real v2;
    Real f;
  equation
    der(x1) = v1;
    m1 * der(v1) = -k * x1 + f;
    der(x2) = v2;
    m2 * der(v2) = -f;
    x2 = x1;
  end ScalarIndexTwo;

  model ClockedSampleHold "a periodic clock of 0.1 s samples x, a unit delay, held into the continuous system (a clocked partition of index 1, no arrays)"
    Real x(start = 1, fixed = true);
    Real xs(start = 0);
    Real xd(start = 0);
    Real y;
  equation
    der(x) = -x;
    xs = sample(x, Clock(0.1));
    xd = previous(xs);
    y = hold(xd);
  end ClockedSampleHold;

  model ClockedLeftLimit "sample(u, clock) at a tick where u jumps takes u's left limit (MLS 16.5.1): u and the Boolean b step at 0.2, a tick of the clock of 0.1 s, so ys and bs step at the tick at 0.3; a hold's output sampled back into its partition is last tick's: n counts the ticks (no algebraic loop through the hold)"
    Real u = if time < 0.2 then 0 else 1;
    Boolean b = time >= 0.2;
    Real ys(start = 0);
    Boolean bs(start = false);
    Real n(start = 0);
    Integer k = integer(floor(10 * time + 0.5)) "an Integer the equations compute (MSL's realToInteger of a table), sampled: 2 at 0.2, 3 at 0.3";
    Integer ks(start = 0);
    Real y;
    Boolean by;
    Real nh;
    Integer kh;
  equation
    ys = sample(u, Clock(0.1));
    bs = sample(b, Clock(0.1));
    n = sample(nh, Clock(0.1)) + 1;
    ks = sample(k, Clock(0.1));
    y = hold(ys);
    by = hold(bs);
    nh = hold(n);
    kh = hold(ks);
  end ClockedLeftLimit;

  model PeriodMean "MSL Blocks.Math.Mean: u integrated over a period, the mean taken and the state reset at its end; the start of the state and of the output by initial equations"
    parameter Real f = 5;
    parameter Real x0 = 0;
    parameter Real y0 = 0;
    Real u;
    Real x;
    discrete Real y;
  initial equation
    x = x0;
    y = y0;
  equation
    der(x) = u;
    when sample(1 / f, 1 / f) then
      y = f * pre(x);
      reinit(x, 0);
    end when;
  end PeriodMean;

  model RolledReinit "two separately declared instances rolled into arrays: the reinit of each one's state, the initial equations of their discrete outputs (for-equations over the instances). At 0.95: m1.y = 1, m2.y = 0.85, m1.x = 0.15, m2.x = 0.04625"
    PeriodMean m1(f = 5);
    PeriodMean m2(f = 10);
  equation
    m1.u = 1;
    m2.u = time;
  end RolledReinit;

  model PreStartCell "a counter whose value before the first event is given by an initial equation on pre()"
    discrete Real e;
    Real x(start = 0, fixed = true);
  initial equation
    pre(e) = 0.5;
  equation
    der(x) = 1;
    when x > 0.3 then
      e = pre(e) + 1;
    end when;
  end PreStartCell;

  model PreStarts "pre(e) = 0.5 of a component array (MSL Timer's pre(entryTime) = 0 in a rolled component): an initial for-equation that no residual solves; e = 1.5 after 0.3"
    PreStartCell c[2];
  end PreStarts;

  model PreStartParameter "pre(y) = a parameter's value, of variables an equation computes (MSL Hysteresis' pre(y) = pre_y_start, MultiSwitch's pre(y) = y_default): y[1] starts true and x[1] rises, y[2] starts false and x[2] falls; with pre() the start value (false) both fell"
    parameter Boolean y0 = true;
    Real x[2](each start = 0.5, each fixed = true);
    Boolean y[2];
  initial equation
    pre(y[1]) = y0;
    pre(y[2]) = not y0;
  equation
    for i in 1:2 loop
      der(x[i]) = if y[i] then 1 else -1;
      y[i] = not pre(y[i]) and x[i] > 1 or pre(y[i]) and x[i] >= 0;
    end for;
  end PreStartParameter;

  class TimeTable "Modelica.Blocks.Types.ExternalCombiTimeTable of MSL 4.1"
    extends ExternalObject;
    function constructor
      input String tableName;
      input String fileName;
      input Real table[:, :];
      input Real startTime;
      input Integer columns[:];
      input Integer smoothness;
      input Integer extrapolation;
      input Real shiftTime;
      input Integer timeEvents;
      input Boolean verboseRead;
      input String delimiter;
      input Integer nHeaderLines;
      output TimeTable externalCombiTimeTable;
    external "C" externalCombiTimeTable = ModelicaStandardTables_CombiTimeTable_init3(fileName, tableName,
      table, size(table, 1), size(table, 2), startTime, columns, size(columns, 1), smoothness, extrapolation,
      shiftTime, timeEvents, verboseRead, delimiter, nHeaderLines)
      annotation(Library = {"ModelicaStandardTables", "ModelicaIO", "ModelicaMatIO", "zlib"});
    end constructor;

    function destructor
      input TimeTable externalCombiTimeTable;
    external "C" ModelicaStandardTables_CombiTimeTable_close(externalCombiTimeTable)
      annotation(Library = {"ModelicaStandardTables", "ModelicaIO", "ModelicaMatIO", "zlib"});
    end destructor;
  end TimeTable;

  function timeTableValue "Modelica.Blocks.Tables.Internal.getTimeTableValueNoDer"
    input TimeTable tableID;
    input Integer icol;
    input Real timeIn;
    input Real nextTimeEvent;
    input Real pre_nextTimeEvent;
    output Real y;
  external "C" y = ModelicaStandardTables_CombiTimeTable_getValue(tableID, icol, timeIn, nextTimeEvent, pre_nextTimeEvent)
    annotation(Library = {"ModelicaStandardTables", "ModelicaIO", "ModelicaMatIO", "zlib"});
  end timeTableValue;

  function timeTableTmax "Modelica.Blocks.Tables.Internal.getTimeTableTmax"
    input TimeTable tableID;
    output Real timeMax;
  external "C" timeMax = ModelicaStandardTables_CombiTimeTable_maximumTime(tableID)
    annotation(Library = {"ModelicaStandardTables", "ModelicaIO", "ModelicaMatIO", "zlib"});
  end timeTableTmax;

  model TableDrive "an external object (a time table, linear from 0 to 2 over [0, 1]) read by the equations of an array and by a parameter's binding (MSL CombiTimeTable's tableID and t_max): x[i] = i t^2, tmax = 1"
    parameter Real data[2, 2] = [0, 0; 1, 2];
    parameter TimeTable tableID = TimeTable("NoName", "NoName", data, 0.0, {2}, 1, 2, 0.0, 3, false, ",", 0);
    parameter Real tmax = timeTableTmax(tableID);
    Real x[2](each start = 0, each fixed = true);
    Real z(start = 0, fixed = true);
  equation
    for i in 1:2 loop
      der(x[i]) = i * timeTableValue(tableID, 1, time, 1e60, 1e60);
    end for;
    der(z) = tmax;
  end TableDrive;

  model TableIntegrator "a component holding its own table (MSL RadioButtonSource's BooleanTable)"
    parameter Real k = 1;
    parameter Real data[2, 2] = [0, 0; 1, 2];
    parameter TimeTable tableID = TimeTable("NoName", "NoName", data, 0.0, {2}, 1, 2, 0.0, 3, false, ",", 0);
    Real x(start = 0, fixed = true);
  equation
    der(x) = k * timeTableValue(tableID, 1, time, 1e60, 1e60);
  end TableIntegrator;

  model OverflowingProbe "a capacitor charged through two opposed junctions from 0 V (MSL Spice3BenchmarkRtlInverter's transistor): OrdinaryDiffEq's first-step probe lands at 1e9 V, where both currents overflow, their difference NaN, the step NaN; v -> 0.025 ln(I / 0.9e-14) = 0.98123"
    parameter Real I = 1e3;
    parameter Real C = 1e-12;
    Real v(start = 0, fixed = true);
    Real i1 = 1e-14 * exp(v / 0.025);
    Real i2 = 1e-15 * exp(v / 0.025);
  equation
    C * der(v) = I - i1 + i2;
  end OverflowingProbe;

  model RolledTables "two components each with an external object (MSL Interaction1's start and stop): not rolled, a constructor call per instance; x = k t^2: 1, 2"
    TableIntegrator a(k = 1);
    TableIntegrator b(k = 2);
  end RolledTables;

  model ShiftedTable "a component with a table shifted in time by its parameter: its constructor reads a field of its component"
    parameter Real k = 1;
    parameter Real shift = 0;
    parameter TimeTable tableID = TimeTable("NoName", "NoName", [0, 0; 1, 2], 0.0, {2}, 1, 2, shift, 3, false, ",", 0);
    Real x(start = 0, fixed = true);
  equation
    der(x) = k * timeTableValue(tableID, 1, time, 1e60, 1e60);
  end ShiftedTable;

  model TableArray "an array of components, each with its own external object (MSL CCCV_Stack's cells' tables): y = 2 (t - shift), x = k (1 - 2 shift): 1, 1, 0"
    ShiftedTable c[3](k = {1, 2, 3}, shift = {0, 0.25, 0.5});
  end TableArray;

  function timeTableNextEvent "Modelica.Blocks.Tables.Internal.getNextTimeEvent"
    input TimeTable tableID;
    input Real timeIn;
    output Real nextTimeEvent;
  external "C" nextTimeEvent = ModelicaStandardTables_CombiTimeTable_nextTimeEvent(tableID, timeIn)
    annotation(Library = {"ModelicaStandardTables", "ModelicaIO", "ModelicaMatIO", "zlib"});
  end timeTableNextEvent;

  model TableSegments "a time table of constant segments with its time events (MSL CombiTimeTable, ConstantSegments): y = 0, 10, 2 from 0, 0.3, 0.6; x[i] = i (10 * 0.3 + 2 * 0.4) at 1"
    parameter Real data[4, 2] = [0, 0; 0.3, 10; 0.6, 2; 1, 2];
    parameter TimeTable tableID = TimeTable("NoName", "NoName", data, 0.0, {2}, 3, 2, 0.0, 1, false, ",", 0);
    discrete Real nextEvent(start = 0, fixed = true);
    Real y;
    Real x[2](each start = 0, each fixed = true);
  equation
    when {time >= pre(nextEvent), initial()} then
      nextEvent = timeTableNextEvent(tableID, time);
    end when;
    y = timeTableValue(tableID, 1, time, nextEvent, pre(nextEvent));
    for i in 1:2 loop
      der(x[i]) = i * y;
    end for;
  end TableSegments;

  function differences "two outputs, the first an array (MSL Machines' ToSpacePhasor)"
    input Real x[3];
    output Real y[2];
    output Real mean;
  algorithm
    for k in 1:2 loop
      y[k] := x[k] - x[k + 1];
    end for;
    mean := (x[1] + x[2] + x[3]) / 3;
  end differences;

  model FirstArrayOutput "the array output of a function of two outputs, the first taken (i_ = ToSpacePhasor(i)): y[k] = x[k] - x[k + 1], x[i] = x0[i] exp(-i t)"
    Real x[3](start = {1, 2, 4}, each fixed = true);
    Real y[2];
    Real w[2] "of an array expression as the argument (ToSpacePhasor(v_p - v_n)): w = 2 y";
  equation
    for i in 1:3 loop
      der(x[i]) = -i * x[i];
    end for;
    y = differences(x);
    w = differences(2 * x - {1, 1, 1});
  end FirstArrayOutput;

  function twoGains "two outputs, the first an array (MSL Filter's coefficient functions)"
    input Real k;
    output Real a[2];
    output Real b;
  algorithm
    a := {k, 2 * k};
    b := 3 * k;
  end twoGains;

  model FreeParameterTuple "(a, b) = f(k) in an initial equation, a and b parameters with fixed = false (MSL Filter's (cr, c0, c1) = coefficients.lowPass(...)): der(x[i]) = -a[i] x[i] + b, x[i] = b / a[i] + (1 - b / a[i]) exp(-a[i] t)"
    parameter Real k = 1;
    parameter Real a[2](each fixed = false);
    parameter Real b(fixed = false);
    Real x[2](each start = 1, each fixed = true);
  initial equation
    (a, b) = twoGains(k);
  equation
    for i in 1:2 loop
      der(x[i]) = -a[i] * x[i] + b;
    end for;
  end FreeParameterTuple;

  model FreeParameterTupleSteady "(a, b) = f(k) in an initial equation read by the steady-state start der(x) = 0 (MSL Blocks.Examples.Filter): the tuple runs before the initial system; x[i] = b / a[i] at all times: 3, 1.5"
    parameter Real k = 1;
    parameter Real a[2](each fixed = false);
    parameter Real b(fixed = false);
    Real x[2];
  initial equation
    (a, b) = twoGains(k);
    der(x) = zeros(2);
  equation
    for i in 1:2 loop
      der(x[i]) = -a[i] * x[i] + b;
    end for;
  end FreeParameterTupleSteady;

  function seedState "a generator's state from its seed (MSL Xorshift128plus.initialState)"
    input Integer seed;
    output Integer state[2];
  algorithm
    state := {seed, 2 * seed};
  end seedState;

  function drawNumber "a number and the next state (MSL Xorshift128plus.random)"
    input Integer stateIn[2];
    output Real r;
    output Integer stateOut[2];
  algorithm
    r := 0.1 * stateIn[1] + 0.01 * stateIn[2];
    stateOut := {stateIn[2], stateIn[1] + 1};
  end drawNumber;

  model InitialDiscreteChain "the initial values of discrete variables from parameters, read by the initial system (MSL NormalNoiseProperties: localSeed, pre(state), r_raw, r, then Variance's mu = u): they are given before it; mu = y at the start, 0.72, at all times"
    parameter Integer fixedSeed = 3;
    final parameter Integer localSeed(fixed = false);
    Integer state[2];
    discrete Real r_raw;
    discrete Real r;
    Real y;
    Real mu;
  initial equation
    localSeed = fixedSeed;
    pre(state) = seedState(localSeed);
    r_raw = drawNumber(pre(state));
    r = 2 * r_raw;
    mu = y;
  equation
    when sample(0, 0.1) then
      (r_raw, state) = drawNumber(pre(state));
      r = 2 * r_raw;
    end when;
    y = r;
    der(mu) = 0;
  end InitialDiscreteChain;

  model LeftLimitAtTimeEvent "pre(u) of u = if time < tStep then .. else ..: u's value before the time relation switches (MSL DemonstrateContinuousSignalExtrema's pulse edges): the step at 0.3 gives y_max = 1.6; with a discrete step time moved from 0.3 to 0.9 at 0.6, z_max = 2.8"
    parameter Real T = 1e-6;
    parameter Real tStep = 0.3;
    discrete Real tEnd(start = 0.3, fixed = true);
    Real u = 2 * time + (if time < tStep then 1 else -2);
    Real v = 2 * time + (if time < tEnd then 1 else -2);
    Real x(start = 1, fixed = true);
    Real w(start = 1, fixed = true);
    Real y_max(start = -10, fixed = true);
    Real z_max(start = -10, fixed = true);
  equation
    der(x) = (u - x) / T;
    der(w) = (v - w) / T;
    when time >= 0.6 then
      tEnd = 0.9;
    end when;
    when {u <= x, u >= x} then
      y_max = max({pre(y_max), u, pre(u)});
    end when;
    when {v <= w, v >= w} then
      z_max = max({pre(z_max), v, pre(v)});
    end when;
  end LeftLimitAtTimeEvent;

  model SaturatedLoop "an amplifier with a limited output in a loop (MSL IdealizedOpAmpLimited as a voltage follower): after the input's step at 0.5 the last solution is on the saturated piece (y = 15), from where Newton's method cycles between two points that are no solutions; y = K / (1 + K) u after it, x(1) = 7.5 + 0.5 K / (1 + K)"
    parameter Real K = 1000;
    Real u = if time < 0.5 then 100 else 1;
    Real e;
    Real y;
    Real x(start = 0, fixed = true);
  equation
    e = u - y;
    y = smooth(0, noEvent(if K * e > 15 then 15 else if K * e < -15 then -15 else K * e));
    der(x) = y;
  end SaturatedLoop;

  model BilinearStart "a loop whose Jacobian is singular at the start values: R * i = v with R and i both 0 there (MSL's resistor with a heat port: R_actual without a start); R = 100 (1 + 1e-3 (T - 300)), T - 300 = 0.01 v i, x integrates i: about sin(1) / 10"
    Real v = 10 * cos(time);
    Real R;
    Real i;
    Real T(start = 300);
    Real x(start = 0, fixed = true);
  equation
    R * i = v;
    R = 100 * (1 + 1e-3 * (T - 300));
    T - 300 = 0.01 * v * i;
    der(x) = i;
  end BilinearStart;

  model ZeroOrientation "MSL FundamentalWave's converter at orientation 0: -v = Nre * der(Phire) + Nim * der(Phiim) with Nim = Nt * sin(orientation) = 0, so der(Phiim) has a zero coefficient only and Phiim is no state; a coil of 1 ohm and a reluctance of 1: Phire = exp(-t), Phiim = 0"
    parameter Real Nt = 1;
    parameter Real orientation = 0;
    final parameter Real Nre = Nt * cos(orientation);
    final parameter Real Nim = Nt * sin(orientation);
    Real Phire(start = 1, fixed = true);
    Real Phiim;
    Real i;
    Real v;
    Real Vre;
    Real Vim;
  equation
    Vre = Nre * i;
    Vim = Nim * i;
    -v = Nre * der(Phire) + Nim * der(Phiim);
    v = i;
    Vre = Phire;
    Vim = Phiim;
  end ZeroOrientation;

  model ZeroOrientationPhases "three converters at the orientations 0, 2 pi / 3, 4 pi / 3 (MSL FundamentalWave's polyphase converter): Nim[1] = 0 takes der(Phiim[1]) out of the first iteration only, so Phiim[1] is no state while Phiim[2] and Phiim[3] are; every phase: a coil of 1 ohm and a reluctance of 1, Phire[k]^2 + Phiim[k]^2 = exp(-2 t)"
    parameter Real Nt = 1;
    parameter Real orientation[3] = {0, 2.0943951023931953, 4.1887902047863905};
    final parameter Real Nre[3] = {Nt * cos(orientation[k]) for k in 1:3};
    final parameter Real Nim[3] = {Nt * sin(orientation[k]) for k in 1:3};
    Real Phire[3](start = {cos(orientation[k]) for k in 1:3}, each fixed = true);
    Real Phiim[3];
    Real i[3];
    Real v[3];
    Real Vre[3];
    Real Vim[3];
    Real w[24](each start = 1, each fixed = true) "arrays that stay arrays beside the split ones";
  equation
    for k in 1:3 loop
      Vre[k] = Nre[k] * i[k];
      Vim[k] = Nim[k] * i[k];
      -v[k] = Nre[k] * der(Phire[k]) + Nim[k] * der(Phiim[k]);
      v[k] = i[k];
      Vre[k] = Phire[k];
      Vim[k] = Phiim[k];
    end for;
    for j in 1:24 loop
      der(w[j]) = -w[j];
    end for;
  end ZeroOrientationPhases;

  model ZeroConductanceCore "MSL FundamentalWave's eddy-current core without conductance, in series with a winding: Vm = G * der(PhiCore) with G = 0 gives Vm = 0, and loss = Vm * der(PhiCore) is the only other read of der(PhiCore), so PhiCore is no state (it is the winding's flux); a winding of 2 turns at v = sin(t): Phi = (1 - cos(t)) / 2"
    parameter Real G = 0 "the core's conductance: none";
    parameter Real N = 2;
    Real Phi(start = 0, fixed = true);
    Real PhiCore;
    Real Vm;
    Real v;
    Real loss;
  equation
    v = sin(time);
    v = N * der(Phi);
    PhiCore = Phi;
    Vm = G * der(PhiCore);
    loss = Vm * der(PhiCore);
  end ZeroConductanceCore;

  model IdealDiodeLoop "MSL IdealSemiconductor's equations, two diodes on sources of opposite phase: the Boolean off[k] = s[k] < 0 is in the loop of s, vd and i (a discrete unknown in an algebraic loop); conducting i = V / (R + Ron), blocking i = Goff V / (1 + R Goff); q(1) = (1 - Goff) / (pi (R + Ron))"
    parameter Real R = 1;
    parameter Real Ron = 1e-5;
    parameter Real Goff = 1e-5;
    Real s[2](each start = 0);
    Real vd[2];
    Real i[2];
    Boolean off[2](each start = true);
    Real q(start = 0, fixed = true);
  equation
    for k in 1:2 loop
      off[k] = s[k] < 0;
      vd[k] = s[k] * (if off[k] then 1 else Ron);
      i[k] = s[k] * (if off[k] then Goff else 1);
      (if k == 1 then 1 else -1) * sin(2 * 3.141592653589793 * time) = R * i[k] + vd[k];
    end for;
    der(q) = i[1];
  end IdealDiodeLoop;

  model ThyristorCommutation "three ideal thyristors from three phase sources to one load (a center tap rectifier), fired 20 degrees after the natural commutation: the loop of all (a discrete unknown per thyristor, cut there); when one fires the last still conducts, and for a pass of the event iteration both do: a near short (2 Ron = 2e-5 ohm under 0.6 V), a linear loop of condition 1e10; q(1) = 3 sqrt(3) / (2 pi) cos(20 degrees) within 1e-3"
    constant Real pi = 3.141592653589793;
    parameter Real R = 1;
    parameter Real Ron = 1e-5;
    parameter Real Goff = 1e-5;
    Real v[3];
    Real s[3](each start = 0);
    Real vd[3];
    Real i[3];
    Real vL;
    Boolean fire[3];
    Boolean off[3](each start = true);
    Real q(start = 0, fixed = true);
  equation
    for k in 1:3 loop
      v[k] = sin(2 * pi * time - (k - 1) * 2 * pi / 3);
      fire[k] = sin(2 * pi * time - (k - 1) * 2 * pi / 3 - pi / 6 - pi / 9) > 0;
      off[k] = s[k] < 0 or pre(off[k]) and not fire[k];
      vd[k] = s[k] * (if off[k] then 1 else Ron);
      i[k] = s[k] * (if off[k] then Goff else 1);
      v[k] - vL = vd[k];
    end for;
    vL = R * (i[1] + i[2] + i[3]);
    der(q) = vL;
  end ThyristorCommutation;

  function complexMeanMagnitude "the mean magnitude of a vector of Complex values (MSL QuasiStatic's quasiRMS)"
    input Complex u[:];
    output Real y;
  algorithm
    y := 0;
    for i in 1:size(u, 1) loop
      y := y + sqrt(u[i].re ^ 2 + u[i].im ^ 2);
    end for;
    y := y / size(u, 1);
  end complexMeanMagnitude;

  function conjugatedProduct "sum(conj(a[k]) * b[k]): MSL 4's Complex scalarProduct, a record result of vectors of records"
    input Complex a[:];
    input Complex b[size(a, 1)];
    output Complex c;
  algorithm
    c := Complex(0, 0);
    for i in 1:size(a, 1) loop
      c := Complex(c.re + a[i].re * b[i].re + a[i].im * b[i].im, c.im + a[i].re * b[i].im - a[i].im * b[i].re);
    end for;
  end conjugatedProduct;

  model ComplexVectorArguments "vectors of Complex values as function arguments (the variables are their fields' vectors) and a Complex result: u = {x, j x}, k = {1, j}; y = conj(k) . u = 2 x (unconjugated it were 0), r = x"
    parameter Complex k[2] = {Complex(1, 0), Complex(0, 1)};
    Real x(start = 1, fixed = true);
    Complex u[2];
    Complex y;
    Real r;
  equation
    der(x) = -x;
    u = {Complex(x, 0), Complex(0, x)};
    y = conjugatedProduct(k, u);
    r = complexMeanMagnitude(u);
  end ComplexVectorArguments;

  model IfBranchResiduals "if-equations on a variable condition whose branches set different variables and have a left side that is no reference (MSL CCCVcharger, IdealSwitchWithArc, DCPowerSupply's -i = ...): a charger of constant current, then constant voltage, on R and C, two elements; vc = t until u = 1 + t reaches Vend, then vc' = Vend - vc: vc(3) = {2 - exp(-2), 3 - exp(-1)}"
    parameter Real Vend[2] = {2, 3};
    parameter Real I = 1;
    parameter Real R = 1;
    parameter Real C = 1;
    Real vc[2](each start = 0, each fixed = true);
    Real u[2];
    Real i[2];
    Boolean cv[2](each start = false, each fixed = true);
  equation
    for k in 1:2 loop
      C * der(vc[k]) = i[k];
      u[k] = R * i[k] + vc[k];
      cv[k] = time >= k;
      if cv[k] then
        u[k] = Vend[k];
      else
        -i[k] = -I;
      end if;
    end for;
  end IfBranchResiduals;

  model InitialCluster "discrete equations whose initial() term decides them at the initialization (MSL PartialFriction's startBackward = ... or initial() and w < 0, mode start Unknown): back and mode = -1 there, x rises from -1 with slope 1; with pre(mode) the start value again in the start's iteration, mode was 0 and x stayed"
    Real x(start = -1, fixed = true);
    Boolean back(start = false, fixed = true);
    Integer mode(start = 3, fixed = true);
  equation
    back = pre(mode) == 0 and x < -2 or initial() and x < 0;
    mode = if (pre(mode) == -1 or back) and x < 0 then -1 else 0;
    der(x) = if mode == -1 then 1 else 0;
  end InitialCluster;

  model StickSlip "a mass with Coulomb friction and a stiction peak (MSL PartialFriction's equations), pushed by 2 sin(t): stuck until 2 sin(t) > Fmax = 1.5 at asin(0.75), then sliding against F0 = 1: v(2) = 2 (cos(asin(0.75)) - cos(2)) - (2 - asin(0.75)) = 1.00324; sa, a and f in a loop with the discrete startForward and locked (a mixed system)"
    parameter Real m = 1;
    parameter Real F0 = 1;
    parameter Real Fmax = 1.5;
    Real u = 2 * sin(time);
    Real v(start = 0, fixed = true);
    Real a;
    Real sa;
    Real f;
    Boolean startForward(start = false, fixed = true);
    Boolean startBackward(start = false, fixed = true);
    Boolean locked;
    Integer mode(start = 0, fixed = true);
  equation
    der(v) = a;
    m * a = u - f;
    startForward = pre(mode) == 0 and (sa > Fmax or pre(startForward) and sa > F0) or pre(mode) == -1 and v > 1e-10 or initial() and v > 0;
    startBackward = pre(mode) == 0 and (sa < -Fmax or pre(startBackward) and sa < -F0) or pre(mode) == 1 and v < -1e-10 or initial() and v < 0;
    locked = not (pre(mode) == 1 or startForward or pre(mode) == -1 or startBackward);
    a = if locked then 0 else if startForward then sa - Fmax else if startBackward then sa + Fmax else if pre(mode) == 1 then sa - Fmax else sa + Fmax;
    f = if locked then sa else if startForward then F0 else if startBackward then -F0 else if pre(mode) == 1 then F0 else -F0;
    mode = if (pre(mode) == 1 or startForward) and v > 0 then 1 else if (pre(mode) == -1 or startBackward) and v < 0 then -1 else 0;
  end StickSlip;

  record FlowState "a record of two Reals (as a medium's ThermodynamicState)"
    Real p;
    Real T;
  end FlowState;

  function flowState "the state at a pressure and an enthalpy (not inlined: an algorithm with an if)"
    input Real p;
    input Real h;
    output FlowState state;
  algorithm
    if h > 0 then
      state := FlowState(p = p, T = h / 2);
    else
      state := FlowState(p = p, T = 0);
    end if;
  end flowState;

  model RecordArrayElements "an array of records set by an array of calls (MSL Fluid's flow models: states = {setState_phX(...), setState_phX(...)}): an equation per field of each element, states[k].p and states[k].T; x = exp(-t), y = states[1].p + states[2].T = 5 x"
    FlowState states[2];
    Real x(start = 1, fixed = true);
    Real y;
  equation
    der(x) = -x;
    states = {flowState(x, 4 * x), flowState(2 * x, 8 * x)};
    y = states[1].p + states[2].T;
  end RecordArrayElements;

  model ColorShape "a shape with a Real color (as MSL MultiBody's visualizers)"
    input Real color[3];
    Real brightness;
  equation
    brightness = color[1] + color[2] + color[3];
  end ColorShape;

  model ColoredBody "Integer color inputs given on to Real ones, the second by the first (MSL MultiBody Body's sphereColor and cylinderColor)"
    input Integer sphereColor[3] = {0, 128, 255};
    input Integer cylinderColor[3] = sphereColor;
    parameter Real m = 1;
    ColorShape sphere(color = sphereColor);
    ColorShape cylinder(color = cylinderColor);
    Real v(start = 1, fixed = true);
  equation
    m * der(v) = -v;
  end ColoredBody;

  model RolledBodies "two bodies of one class (rolled into arrays over the instances), the second's mass the first's: the equalities of the Integer colors stay as written, and the mass array's second element reads its first; v = exp(-t / 2) in both, the shapes' brightness 383"
    ColoredBody body1(m = 2);
    ColoredBody body2(m = body1.m);
  end RolledBodies;

  function squareBothWays "k1 x^2 for x >= 0, -k2 x^2 below (as MSL Fluid's regSquare2 outside its regularized band): one expression, which the backend inlines (no Inline annotation: the frontend leaves the call)"
    input Real x;
    input Real k1;
    input Real k2;
    output Real y;
  algorithm
    y := if x >= 0 then k1 * x ^ 2 else -k2 * x ^ 2;
  end squareBothWays;

  model InlinedRelation "a function with a relation inlined into an equation of a loop: the relation is the function's, no event (MLS 3.7.5), so the loop is solved on the branch of its iterate, not on the branch held since the last event. Two tanks of area 1 joined by a valve, q the flow from 1 to 2: h1 - h2 = squareBothWays(q, 1, 4). From h = {2, 0}: q = sqrt(h1 - h2), the difference d = h1 - h2 follows der(d) = -2 sqrt(d), d = (sqrt(2) - t)^2, zero at t = sqrt(2)"
    Real h[2](start = {2, 0}, each fixed = true);
    Real q(start = 1);
    Real dp;
  equation
    der(h[1]) = -q;
    der(h[2]) = q;
    dp = h[1] - h[2];
    dp = squareBothWays(q, 1, 4);
  end InlinedRelation;

  model PreStartInLoop "pre() of a discrete array's elements given in an initial for-equation (MSL Fluid's tank: pre(aboveLevel[i]) = level_start >= height[i]); above[i] keeps its value inside a band of 0.1 around height[i]. x = 0.5 + 0.4 cos(t): at t = 0, x = 0.9, above = {true, true} by pre() (without it above[1] is false: 0.9 is inside its band); at t = 1, x = 0.716: {false, true}"
    parameter Real height[2] = {0.85, 0.6};
    parameter Real x0 = 0.9;
    Real x;
    Boolean above[2];
    Real count;
  initial equation
    for i in 1:2 loop
      pre(above[i]) = x0 >= height[i];
    end for;
  equation
    x = 0.5 + 0.4 * cos(time);
    for i in 1:2 loop
      above[i] = x >= height[i] + 0.1 or pre(above[i]) and x >= height[i] - 0.1;
    end for;
    count = (if above[1] then 1 else 0) + (if above[2] then 10 else 0);
  end PreStartInLoop;

  function quarterTurn "j^k as a Complex number (as MSL ComplexBlocks' powerOfJ)"
    input Integer k;
    output Complex z;
  algorithm
    z := if mod(k, 4) == 0 then Complex(1, 0) else if mod(k, 4) == 1 then Complex(0, 1) else if mod(k, 4) == 2 then Complex(-1, 0) else Complex(0, -1);
    annotation(Inline = true);
  end quarterTurn;

  model RecordIfField "a field of an if-expression of records (an inlined function's value): w = {x j^1, x j^2}, x = exp(-t)"
    Real x(start = 1, fixed = true);
    Complex w[2];
  equation
    der(x) = -x;
    w = {Complex(x, 0) * quarterTurn(1), Complex(x, 0) * quarterTurn(2)};
  end RecordIfField;

  model SampleCounter "counts the samples of its period, in an algorithm section"
    parameter Real period = 0.25;
    Integer n(start = 0, fixed = true);
  algorithm
    when sample(period, period) then
      n := pre(n) + 1;
    end when;
  end SampleCounter;

  model ComponentAlgorithms "an array of components with an algorithm section each: with the arrays kept the sections come as one for-statement around the when-statement (MSL Digital's InertialDelaySensitive); until t = 0.95 the periods 0.25 and 0.1 give 3 and 9 samples"
    SampleCounter c[2](period = {0.25, 0.1});
    Real x(start = 1, fixed = true);
  equation
    der(x) = -x;
  end ComponentAlgorithms;

  type Logic3 = enumeration(Unknown, Low, High);

  model EnumSubscriptCluster "an array over an enumeration read at a literal (MSL Digital's ResetMap[Logic.'U']), and a discrete equation with a relation of parameters beside one of time: its lifted when-equation has change(tHigh[1] > 0) among its triggers, which never fires. y = gain[High] x + k: 2 exp(-1) + 2 at t = 1"
    parameter Real gain[Logic3] = {0, 1, 2};
    parameter Real tBase = 1;
    parameter Real tHigh[2] = {tBase, 2 * tBase};
    Boolean b;
    Integer k;
    Real x(start = 1, fixed = true);
    Real y;
  equation
    der(x) = -x;
    b = time > 0.5;
    k = if tHigh[1] > 0 then (if b then 2 else 1) else 0;
    y = gain[Logic3.High] * x + k;
  end EnumSubscriptCluster;
end ArrayPath;
