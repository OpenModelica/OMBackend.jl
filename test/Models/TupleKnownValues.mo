package TupleKnownValues "a variable a function's tuple output defines, which the simulation code finds constant: index reduction reads its value (MSL Spice3's diodes without a junction capacitance, `(cc, m_dCap) = diodeNoBypassCode(...)` 0, whose `icap = m_dCap * der(v)` was solved for der(v): a division by 0)"
  function junction "a junction's current, and its capacitance from the parameters: none for cj = 0"
    input Real v;
    input Real cj;
    output Real i;
    output Real c;
  algorithm
    i := 1e-3 * v;
    c := 2 * cj;
  end junction;

  model Clamp "a capacitor C with a junction across it: the junction's capacitance c is 0"
    constant Real cj = 0 "folded into the call, as the Spice3 diode's parameters";
    parameter Real C = 1e-6;
    Real v(start = 1, fixed = true);
    Real i, c, icap;
  equation
    (i, c) = junction(v, cj);
    icap = c * der(v);
    C * der(v) = -v / 1e3 - i - icap;
  end Clamp;
end TupleKnownValues;
