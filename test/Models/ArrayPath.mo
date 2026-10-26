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
end ArrayPath;
