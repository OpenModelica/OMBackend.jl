package EventOrder
  model RadioButtons
    "Two of MSL Blocks.Sources.RadioButtonSource, each resetting the other, pressed at the same instant t = 1 (start's
     table toggles there, stop's too), and the step and transition of MSL StateGraph.Examples.ControlledTanks that
     read them: stop sets the step s2, the transition t4 from s2 fires on start. In the first event iteration at t = 1
     start and stop go on; in the second start is reset (pre(stop.on) rose) as s2 becomes active: t4 does not fire,
     s2 stays active (MLS 8.6: each iteration evaluates all equations with pre() of the previous one)."
    Boolean startTable = time >= 0.5 and time < 1 "start's BooleanTable: pressed at 0.5 and at 1";
    Boolean stopTable = time >= 1 "stop's BooleanTable: pressed at 1";
    Boolean startOn(start = false, fixed = true);
    Boolean stopOn(start = false, fixed = true);
    Boolean startPreReset;
    Boolean stopPreReset;
    Boolean s2Active "s2's localActive";
    Boolean s2New(start = false, fixed = true) "s2's newActive";
    Boolean t4 "t4 fires";
    Integer t4Fired(start = 0, fixed = true);
    Real x(start = 0, fixed = true);
  initial equation
    pre(startPreReset) = false;
    pre(stopPreReset) = false;
  algorithm
    startPreReset := pre(stopOn);
    when startPreReset then
      startOn := false;
    end when;
    when change(startTable) then
      startOn := true;
    end when;
  algorithm
    stopPreReset := pre(startOn);
    when stopPreReset then
      stopOn := false;
    end when;
    when change(stopTable) then
      stopOn := true;
    end when;
  equation
    s2Active = pre(s2New);
    t4 = startOn and s2Active;
    s2New = stopOn and not s2Active or s2Active and not t4;
    when t4 then
      t4Fired = pre(t4Fired) + 1;
    end when;
    der(x) = if s2Active then 1 else 0;
  end RadioButtons;

  model StepCount
    "The integer() of a Real count of active steps, as MSL StateGraph's PartialCompositeStep counts its inner steps
     (activeSteps = -integer(...activeSteps)): at t = 1 step b becomes active in the first event iteration and a
     inactive in the second, so the count reads 2 for one iteration (1, 2, 1). integer() changes with it (2, then 1
     again, MLS 3.7.4.2): n = 1 after t = 1, and y grows on. The steps read the count (q < 3, always true), as
     ControlledTanks' transitions read what the steps drive: their equations are solved with the algebraic unknowns,
     and q moves before n's when reads it."
    Boolean aNew(start = true, fixed = true);
    Boolean bNew(start = false, fixed = true);
    Boolean aActive "a's localActive";
    Boolean bActive "b's localActive";
    Real q "the count of active steps";
    Integer n;
    Real y(start = 0, fixed = true);
  equation
    aActive = pre(aNew);
    bActive = pre(bNew);
    aNew = aActive and not bActive and q < 3;
    bNew = bActive or time >= 1;
    q = (if aActive then 1.0 else 0.0) + (if bActive then 1.0 else 0.0);
    n = integer(q);
    der(y) = n;
  end StepCount;
end EventOrder;
