package InitializationNames "The initialization's parameters and pins matched by the variables' names"

  model FreeArrayParameter "free parameters that are array elements (printed var\"k[1]\"): unknowns of the initialization"
    parameter Real k[2](each fixed = false, each start = 1);
    Real x[2](start = {1, 2}, each fixed = true);
    Real y[2];
  equation
    der(x) = -k .* x;
    y = k .* x;
  initial equation
    y = {3, 8};
  end FreeArrayParameter;

  model RelationAtStart "positive feedback around a limiter, simplified = its upper limit: y = 1 (OpenModelica); the initialization settles the if-expression's relations (the entries' ifCond names are Symbols)"
    Real u;
    Real y;
    Real x(start = 0, fixed = true);
  equation
    u = 2 * y + 0.1 * x;
    y = homotopy(actual = smooth(0, if u > 1 then 1 elseif u < -1 then -1 else u), simplified = 1);
    der(x) = -x;
  end RelationAtStart;

end InitializationNames;
