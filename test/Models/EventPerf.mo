package EventPerf "The event iteration's compiled parts: a cluster body, the event Newton, assert crossings, eager evaluation"
  model HalfWave "A sine source, an ideal diode (MSL IdealDiode's equations, a discrete cluster) and an RC load"
    constant Real pi = 3.141592653589793;
    parameter Real Ron = 1e-5;
    parameter Real Goff = 1e-5;
    parameter Real R = 10;
    parameter Real C = 1e-3;
    Real vC(start = 0, fixed = true) "the capacitor's voltage";
    Real v "the diode's voltage";
    Real i "the diode's current";
    Real s(start = 0) "the diode's characteristic's parameter";
    Boolean off(start = true);
  equation
    v = sin(2 * pi * 50 * time) - vC;
    off = s < 0;
    v = s * (if off then 1 else Ron);
    i = s * (if off then Goff else 1);
    C * der(vC) = i - vC / R;
  end HalfWave;

  model Asserts "Time-varying asserts with crossings: x = time"
    Real x(start = 0, fixed = true);
    Real y = 2 * x;
  equation
    der(x) = 1;
    assert(x < 2, "x below 2");
    assert(y > -1 or x < 0.5, "y above -1");
    assert(x >= 0 and y <= 4, "x non-negative, y at most 4", AssertionLevel.warning);
  end Asserts;

  function clipped "u, at least 0: a condition on its argument"
    input Real u;
    output Real y;
  algorithm
    if u > 0 then
      y := u;
    else
      y := 0;
    end if;
  end clipped;

  model Eager "clipped(x) twice with the same argument: its eager symbolic evaluation fails once"
    Real x(start = 1, fixed = true);
    Real y1;
    Real y2;
  equation
    der(x) = -1;
    y1 = clipped(x);
    y2 = clipped(x) + 1;
  end Eager;
end EventPerf;
