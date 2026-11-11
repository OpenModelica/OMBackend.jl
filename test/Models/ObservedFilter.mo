package ObservedFilter "Observed equations kept for an observed filter (performStructuralSimplify)"
  model AliasOfObserved "u = z: z the alias representative, observed after the structural simplification, read by the alias observation u = z; with an if-equation's event, the alias and eliminated variables' observations are added after the structural simplification (odeSystemWithEvents); read with an observed filter that names u only"
    Real x(start = 1, fixed = true);
    Real u;
    Real y;
    Real z;
    Real w;
  equation
    der(x) = -x;
    u = 2 * x + sin(x);
    y = sqrt(u^2 + 1);
    z = u;
    if x > 0.5 then
      w = 1;
    else
      w = 0;
    end if;
  end AliasOfObserved;
  model RelationOfObserved "tEnd observed after the structural simplification and read by the relation of y only (MSL FullRobot's path planning); read with an observed filter that names y only"
    Real x(start = 1, fixed = true);
    Real tEnd;
    Real y;
  equation
    der(x) = -x;
    tEnd = 0.3 + 0.1 * x;
    y = if time >= tEnd then 1 else 0;
  end RelationOfObserved;
  model ClusterOfObserved "reference observed after the structural simplification and read only by the discrete cluster of on (a hysteresis, MSL OnOffController's shape); no equation and no ModelingToolkit event reads it; read with an observed filter that names y only"
    Real x(start = 0, fixed = true);
    Real reference;
    Boolean on(start = true, fixed = true);
    Real y;
  equation
    reference = 0.5 + 0.1 * time;
    on = pre(on) and x < reference + 0.1 or x < reference - 0.1;
    der(x) = if on then 1 else -1;
    y = 2 * x;
  end ClusterOfObserved;
  function isIdeal "false for the table {1, 0}"
    input Real table[:];
    output Boolean ideal;
  algorithm
    ideal := table[1] > 1 and table[2] > 1;
  end isIdeal;
  model ClusterOfConstantObserved "ideal a constant ModelingToolkit substitutes (ideal ~ 0), read only by the discrete cluster of up (MSL LossyGearDemo1's gear.ideal); read with an observed filter that names y only"
    parameter Real table[2] = {1, 0};
    Real x(start = 0, fixed = true);
    Boolean ideal;
    Boolean up(start = false, fixed = true);
    Real y;
  equation
    ideal = isIdeal(table);
    up = not ideal and (x > 0.3 or pre(up) and x > 0.1);
    der(x) = if up then 2 else 1;
    y = 2 * x;
  end ClusterOfConstantObserved;
end ObservedFilter;
