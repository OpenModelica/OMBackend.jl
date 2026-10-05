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

  model AlgebraicLoop "Outside the array path: it is scalarized as before"
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
end ArrayPath;
