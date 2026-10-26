package OMBackendWorkload "The models of OMBackend's precompile workload for the arrays-kept paths (precompile.jl): small, but with the constructs MSL models take those paths with, so that the passes' code compiles into the package image"

  record Frame "an orientation: a rotation matrix and an angular velocity (as MSL Frames.Orientation)"
    Real T[3, 3];
    Real w[3];
  end Frame;

  function planarFrame "the frame rotated by angle about z, turning at speed (as MSL Frames.planarRotation): a record result"
    input Real angle;
    input Real speed;
    output Frame R;
  algorithm
    R := Frame(T = {{cos(angle), sin(angle), 0}, {-sin(angle), cos(angle), 0}, {0, 0, 1}}, w = {0, 0, speed});
  end planarFrame;

  function resolve1 "a vector of the frame resolved in the world (as MSL Frames.resolve1): a record argument"
    input Frame R;
    input Real v[3];
    output Real v0[3];
  algorithm
    v0 := transpose(R.T) * v;
  end resolve1;

  function shaped "x + x^3/10, no derivative annotation: a constraint through it differentiated through its body"
    input Real x;
    output Real y;
  algorithm
    y := x + x ^ 3 / 10;
  end shaped;

  model Link "a link turning by phi against its base: its frame from a function, its tip resolved through it; a pair of inertias rigidly coupled through a function (index 3)"
    parameter Real L = 1;
    parameter Real k = 1;
    Real phi_a "the base's angle";
    Real w_a "the base's speed";
    Real phi(start = 0) "no fixed start: the reduction's state choice, its dummy states free of one";
    Real w(start = 0);
    Real phi2(start = 0);
    Real w2(start = 0);
    Real tau;
    Frame R;
    Real r_a[3];
    Real r_b[3];
  equation
    R = planarFrame(phi_a + phi, w_a + w);
    r_b = r_a + resolve1(R, {L, 0, 0});
    der(phi) = w;
    der(phi2) = w2;
    der(w) = k * sin(time) - tau;
    2 * der(w2) = tau;
    shaped(phi) = shaped(phi2);
  end Link;

  model Chain "three Links declared one by one (rolled into arrays), each on the one before's tip and angle, the first on the origin: elements solved differently (slicing, the frames' matrices along their inner dimensions)"
    Link a(k = 1);
    Link b(k = 2);
    Link c(k = 3);
    Real tip[3] "c's tip";
  equation
    a.r_a = {0, 0, 0};
    a.phi_a = 0;
    a.w_a = 0;
    b.r_a = a.r_b;
    b.phi_a = a.phi_a + a.phi;
    b.w_a = a.w_a + a.w;
    c.r_a = b.r_b;
    c.phi_a = b.phi_a + b.phi;
    c.w_a = b.w_a + b.w;
    tip = c.r_b;
  end Chain;

  connector FluidPort "MSL Media's test port: an enthalpy flow beside the mass flow"
    flow Real m_flow;
    Real h;
    flow Real H_flow;
  end FluidPort;

  model Volume "an enthalpy store of mass M (index 2: U = M h, both differentiated)"
    parameter Real M = 1;
    Real U;
    Real h(start = 0, fixed = true);
    Real dh;
    FluidPort port;
  equation
    der(U) = port.H_flow;
    U = M * h;
    der(h) = dh;
    port.h = h;
    port.m_flow = 0;
  end Volume;

  model Pipe "MSL Media's ShortPipe without pressures"
    FluidPort a;
    FluidPort b;
  equation
    a.H_flow = semiLinear(a.m_flow, a.h, b.h);
    a.H_flow + b.H_flow = 0;
    a.m_flow + b.m_flow = 0;
  end Pipe;

  model Source "mass flow k t out at enthalpy hs; k a free parameter of an initial equation"
    parameter Real hs = 1;
    parameter Real k(fixed = false, start = 2);
    FluidPort port;
  initial equation
    k * k = 1;
  equation
    port.m_flow = -k * time;
    port.H_flow = semiLinear(port.m_flow, port.h, hs);
  end Source;

  model Ambient "MSL Media's FixedAmbient: enthalpy h0 at inflow"
    parameter Real h0 = 2;
    FluidPort port;
  equation
    port.H_flow = semiLinear(port.m_flow, port.h, h0);
  end Ambient;

  model Volumes "a source, two volumes and pipes (rolled), an ambient: a semiLinear chain through the aliases rolling leaves; a when-equation"
    Source source;
    Volume v1;
    Volume v2;
    Pipe p1;
    Pipe p2;
    Ambient ambient;
    Integer filled(start = 0, fixed = true);
    discrete Real tFilled(start = -1, fixed = true);
  equation
    connect(source.port, v1.port);
    connect(v1.port, p1.a);
    connect(p1.b, v2.port);
    connect(v2.port, p2.a);
    connect(p2.b, ambient.port);
    when v2.h > 0.05 then
      filled = pre(filled) + 1;
      tFilled = time;
    end when;
  end Volumes;

  model LoopEvents "an algebraic loop through all the elements (torn, solved in the right-hand side) with an event in it, a when-equation on a state, an assert and an initial equation"
    parameter Integer N = 12;
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
  end LoopEvents;

end OMBackendWorkload;
