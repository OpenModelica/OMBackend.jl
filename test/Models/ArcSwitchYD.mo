package ArcSwitchYD "MSL 4.1.0's SwitchYDwithArc (Machines.Utilities) on three RL windings, its components' equations copied (MSL's `quenched` is `arcOff` here): the Y openers open at 25 ms (arcs), the D closers close 20 ms later"
  model LogicalDelay "Modelica.Blocks.Logical.LogicalDelay"
    parameter Real delayTime = 0;
    input Boolean u;
    output Boolean y1 "delayed falling edge";
    output Boolean y2 "delayed rising edge";
  protected
    discrete Real tSwitch;
  initial equation
    tSwitch = time - 2 * delayTime;
  equation
    when {u, not u} then
      tSwitch = time;
    end when;
    y1 = if u then true else not (time >= tSwitch + delayTime);
    y2 = if not u then false else (time >= tSwitch + delayTime);
  end LogicalDelay;

  model OpenerWithArc "Modelica.Electrical.Analog.Ideal.OpenerWithArc, its voltage v and current i"
    parameter Real Ron = 1e-5;
    parameter Real Goff = 1e-5;
    parameter Real V0 = 10;
    parameter Real dVdt = 1e4;
    parameter Real Vmax = 100;
    Real v;
    Real i;
    input Boolean control "true: open";
    Boolean off(start = true, fixed = true) = control;
    Boolean on = not off;
  protected
    discrete Real tSwitch(start = -1e60);
    Boolean arcOff(start = true, fixed = true) "the arc has gone out (MSL: quenched)";
  equation
    when edge(off) then
      tSwitch = time;
    end when;
    arcOff = off and (abs(i) <= abs(v) * Goff or pre(arcOff));
    if on then
      v = Ron * i;
    else
      if arcOff then
        i = Goff * v;
      else
        v = min(Vmax, V0 + dVdt * (time - tSwitch)) * sign(i);
      end if;
    end if;
  end OpenerWithArc;

  model CloserWithArc "Modelica.Electrical.Analog.Ideal.CloserWithArc, its voltage v and current i"
    parameter Real Ron = 1e-5;
    parameter Real Goff = 1e-5;
    parameter Real V0 = 10;
    parameter Real dVdt = 1e4;
    parameter Real Vmax = 100;
    Real v;
    Real i;
    input Boolean control "true: closed";
    Boolean off(start = true, fixed = true) = not control;
    Boolean on = not off;
  protected
    discrete Real tSwitch(start = -1e60);
    Boolean arcOff(start = true, fixed = true) "the arc has gone out (MSL: quenched)";
  equation
    when edge(on) then
      tSwitch = time;
    end when;
    arcOff = off and (abs(i) <= abs(v) * Goff or pre(arcOff));
    if on then
      v = Ron * i;
    else
      if arcOff then
        i = Goff * v;
      else
        v = min(Vmax, V0 + dVdt * (time - tSwitch)) * sign(i);
      end if;
    end if;
  end CloserWithArc;

  model Switching "phase k's winding (R, L) from its supply terminal sp[k] to sn[k]; the opener sn[k] to the star point, the closer sn[k] to sp[k + 1]"
    constant Real pi = 3.141592653589793;
    parameter Real V = 81.64965809277261 "sqrt(2/3) 100";
    parameter Real R = 0.5;
    parameter Real L = 1e-3;
    LogicalDelay delay[3](each delayTime = 0.02);
    OpenerWithArc opener[3];
    CloserWithArc closer[3];
    Real i[3](each start = 0, each fixed = true) "winding currents";
    Real sp[3] "supply terminals";
    Real sn[3] "winding ends";
    Real star;
  equation
    for k in 1:3 loop
      delay[k].u = time >= 0.025;
      opener[k].control = delay[k].y1;
      closer[k].control = delay[k].y2;
      sp[k] = V * sin(2 * pi * 50 * time - 2 * pi * (k - 1) / 3);
      R * i[k] + L * der(i[k]) = sp[k] - sn[k];
      opener[k].v = sn[k] - star;
      closer[k].v = sn[k] - sp[mod(k, 3) + 1];
      i[k] = opener[k].i + closer[k].i;
    end for;
    opener[1].i + opener[2].i + opener[3].i = 0;
  end Switching;
end ArcSwitchYD;
