package ObservedFilter "Observed equations kept for an observed filter (performStructuralSimplify)"
  model AliasOfObserved "z = u: z the alias representative, observed after the structural simplification, read by the observations u = z and y = sqrt(1 + z^2); read with an observed filter that names z only"
    Real x(start = 1, fixed = true);
    Real u;
    Real y;
    Real z;
  equation
    der(x) = -x;
    u = 2 * x + sin(x);
    y = sqrt(u^2 + 1);
    z = u;
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
