package HeldInBand
  model Port
    "A level rising to a port, with Modelica.Fluid.Vessels' port logic: the blocked port's s and the level's
     relation cross zero together, s's hysteresis edge first. Held false in its band, regularFlow leaves no
     consistent inFlow mode (inflow: s = m < 0; blocked: s > 0). The level's band is 7 times the tolerance (its
     scale 1 + h), s's edge comes at a quarter of it. The level reaches h at t = 0.1, then rises at 0.5:
     level(1) = 6.45."
    parameter Real h = 6;
    Real level(start = 5.9, fixed = true);
    Real m "flow through the port into the vessel";
    Real s;
    Boolean regularFlow;
    Boolean inFlow;
  equation
    der(level) = 1 + m;
    regularFlow = level >= h;
    inFlow = not regularFlow and s > 0;
    if regularFlow then
      m = -0.5;
      s = level - h;
    elseif inFlow then
      m = -0.2;
      s = m;
    else
      m = 0;
      s = 2*(level - h);
    end if;
  end Port;
end HeldInBand;
