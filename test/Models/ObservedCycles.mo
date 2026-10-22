package ObservedCycles "observed equations, each solved for its variable, need an order: the variables of a loop among them stay unknowns, their residuals a loop (SimCode.cyclicNames)"
  model Folded "foldExplicitSingleAssign: u = 2y + 0.1x and y = noEvent(if u > 1 ...), nothing else reads them (InitialEquationTests IEQ54 without homotopy): y = 1 from the start values"
    Real u(start = 2);
    Real y(start = 1);
    Real x(start = 0, fixed = true);
  equation
    u = 2 * y + 0.1 * x;
    y = noEvent(if u > 1 then 1 elseif u < -1 then -1 else u);
    der(x) = -x;
  end Folded;

  model OutputOnly "eliminateNonDynamic: 3u = 2y + 0.1x and 2y = noEvent(...), no explicit definitions: the one root u = y = 0"
    Real u(start = 0.5);
    Real y(start = 0.5);
    Real x(start = 0, fixed = true);
  equation
    2 * y = noEvent(if u > 1 then 2 elseif u < -1 then -2 else 2 * u);
    3 * u = 2 * y + 0.1 * x;
    der(x) = -x;
  end OutputOnly;
end ObservedCycles;
