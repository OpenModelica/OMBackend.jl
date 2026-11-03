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
end ObservedFilter;
