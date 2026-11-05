package OMBackendTranslateWorkload "The models of OMBackend's precompile workload for the scalarized path (precompile.jl): small, but with the constructs MSL's examples without arrays take it with (Rotational First, Blocks, Thermal, Analog, MultiBody's index reduction), so that their translate, build and solve compile into the package image"

  connector Flange "a rotational flange"
    Real phi;
    flow Real tau;
  end Flange;

  model Inertia
    parameter Real J = 1;
    Flange a;
    Flange b;
    Real phi(start = 0, fixed = true);
    Real w(start = 0, fixed = true);
  equation
    phi = a.phi;
    phi = b.phi;
    der(phi) = w;
    J * der(w) = a.tau + b.tau;
  end Inertia;

  model IdealGear "a.phi = ratio b.phi: the inertias on both sides rigidly coupled (index two)"
    parameter Real ratio = 10;
    Flange a;
    Flange b;
  equation
    a.phi = ratio * b.phi;
    0 = ratio * a.tau + b.tau;
  end IdealGear;

  model SpringDamper
    parameter Real c = 1e4;
    parameter Real d = 10;
    Flange a;
    Flange b;
    Real phi_rel;
    Real tau;
  equation
    phi_rel = b.phi - a.phi;
    tau = c * phi_rel + d * der(phi_rel);
    b.tau = tau;
    a.tau = -tau;
  end SpringDamper;

  model Fixed
    Flange flange;
  equation
    flange.phi = 0;
  end Fixed;

  function saturate "u limited to [-uMax, uMax]: an algorithm with an if-statement"
    input Real u;
    input Real uMax;
    output Real y;
  algorithm
    if u > uMax then
      y := uMax;
    elseif u < -uMax then
      y := -uMax;
    else
      y := u;
    end if;
  end saturate;

  model Torque "a sine torque through a function"
    parameter Real amplitude = 10;
    parameter Real f = 5;
    Flange flange;
    Real tau;
  equation
    tau = saturate(amplitude * sin(6.283185307179586 * f * time), 0.8 * amplitude);
    flange.tau = -tau;
  end Torque;

  model Drive "Rotational First's structure: a torque on an inertia, an ideal gear to a second, a spring-damper to a third, damped to a fixed flange"
    Torque torque;
    Inertia j1(J = 0.1);
    IdealGear gear;
    Inertia j2(J = 2, phi(fixed = false), w(fixed = false));
    SpringDamper spring;
    Inertia j3(J = 1);
    SpringDamper damper(c = 0, d = 1);
    Fixed fixed;
  equation
    connect(torque.flange, j1.a);
    connect(j1.b, gear.a);
    connect(gear.b, j2.a);
    connect(j2.b, spring.a);
    connect(spring.b, j3.a);
    connect(j3.b, damper.a);
    connect(damper.b, fixed.flange);
  end Drive;

  type Mode = enumeration(off "heating off", on "heating on");

  model Thermostat "a heated mass under an on-off controller (when/elsewhen on an enumeration), a sampled PI (discrete states), a bouncing state (reinit), an Integer counter of edges, noEvent and smooth, an assert"
    parameter Real C = 1;
    parameter Real G = 1;
    parameter Real Tset = 1;
    parameter Real band = 0.1;
    Real T(start = 0, fixed = true);
    Mode mode(start = Mode.on, fixed = true);
    Boolean on;
    Real Q;
    discrete Real integral(start = 0, fixed = true);
    discrete Real u(start = 0, fixed = true);
    Integer switches(start = 0, fixed = true);
    Real x(start = 1, fixed = true);
    Real v(start = 0, fixed = true);
    Real r;
    Real s;
  equation
    when T < Tset - band then
      mode = Mode.on;
    elsewhen T > Tset + band then
      mode = Mode.off;
    end when;
    on = mode == Mode.on;
    Q = if on then 5 + u else 0;
    C * der(T) = Q - G * T;
    when sample(0, 0.1) then
      integral = pre(integral) + 0.1 * (Tset - T);
      u = max(0, min(2, 0.5 * (Tset - T) + pre(integral)));
    end when;
    when edge(on) then
      switches = pre(switches) + 1;
    end when;
    der(x) = v;
    der(v) = -9.81;
    when x < 0 then
      reinit(v, -0.8 * pre(v));
    end when;
    r = noEvent(if T > 0 then sqrt(T) else 0);
    s = smooth(1, if T > 0.5 then (T - 0.5) ^ 2 else 0);
    assert(T < 100, "the temperature stays below 100");
  end Thermostat;

  connector Pin
    Real v;
    flow Real i;
  end Pin;

  model Resistor
    parameter Real R = 1;
    Pin p;
    Pin n;
  equation
    p.v - n.v = R * p.i;
    p.i + n.i = 0;
  end Resistor;

  model Capacitor
    parameter Real C = 1e-3;
    Pin p;
    Pin n;
    Real v(start = 0, fixed = true);
  equation
    v = p.v - n.v;
    C * der(v) = p.i;
    p.i + n.i = 0;
  end Capacitor;

  model Diode "exponential: a nonlinear algebraic loop with the resistor before it"
    parameter Real Is = 1e-6;
    parameter Real Vt = 0.04;
    Pin p;
    Pin n;
    Real v;
  equation
    v = p.v - n.v;
    p.i = Is * (exp(v / Vt) - 1);
    p.i + n.i = 0;
  end Diode;

  model SineVoltage
    parameter Real V = 1;
    parameter Real f = 10;
    Pin p;
    Pin n;
  equation
    p.v - n.v = V * sin(6.283185307179586 * f * time);
    p.i + n.i = 0;
  end SineVoltage;

  model Ground
    Pin p;
  equation
    p.v = 0;
  end Ground;

  model Rectifier "a source and a resistor feeding a diode that charges a loaded capacitor (Analog's examples: a torn nonlinear loop)"
    SineVoltage source;
    Resistor r(R = 10);
    Diode diode;
    Capacitor c;
    Resistor load(R = 100);
    Ground ground;
  equation
    connect(source.p, r.p);
    connect(r.n, diode.p);
    connect(diode.n, c.p);
    connect(c.p, load.p);
    connect(c.n, ground.p);
    connect(load.n, ground.p);
    connect(source.n, ground.p);
  end Rectifier;

  record Polar
    Real r;
    Real angle;
  end Polar;

  function toPolar "a record result"
    input Real x;
    input Real y;
    output Polar p;
  algorithm
    p := Polar(r = sqrt(x ^ 2 + y ^ 2), angle = atan2(y, x));
  end toPolar;

  function bounds "two outputs (a tuple call)"
    input Real u;
    output Real lo;
    output Real hi;
  algorithm
    lo := u - 1;
    hi := u + 1;
  end bounds;

  model Pendulum "a mass on a rod (index three, dummy derivatives: MultiBody's reduction scalarized), its position through functions with a record and a tuple result"
    parameter Real L = 1;
    parameter Real m = 1;
    parameter Real g = 9.81;
    Real x(start = 1, fixed = true);
    Real y(start = 0);
    Real vx(start = 0, fixed = true);
    Real vy;
    Real F;
    Polar position;
    Real lo;
    Real hi;
  equation
    der(x) = vx;
    der(y) = vy;
    m * der(vx) = -x / L * F;
    m * der(vy) = -y / L * F - m * g;
    x ^ 2 + y ^ 2 = L ^ 2;
    position = toPolar(x, y);
    (lo, hi) = bounds(x);
  end Pendulum;

end OMBackendTranslateWorkload;
