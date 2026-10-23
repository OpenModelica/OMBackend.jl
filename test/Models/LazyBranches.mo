package LazyBranches "an if-expression evaluates only the branch its condition chooses (MLS 3.6.5): a function in the other branch that fails for the values there is not called (MSL PsychrometricData's `if x < 5e-6 then 200 else Medium.saturationTemperature(ps_phi[i])` with ps_phi = 0 at the start)"
  function positiveLog "log(x); an error for x <= 0, as MSL's saturationTemperature outside its bracket"
    input Real x;
    output Real y;
  algorithm
    assert(x > 0, "positiveLog: x must be positive");
    y := log(x);
  end positiveLog;

  model Guarded "y's else branch fails at the start (x = 0); the condition chooses the other one until x reaches 0.5"
    Real x(start = 0, fixed = true);
    Real y;
    Real z "a nested if-expression: the inner condition guards the call as well";
  equation
    der(x) = 1;
    y = if x < 0.5 then 0 else positiveLog(x);
    z = if x < 0.25 then 1 else if x < 0.5 then 2 else 3 + positiveLog(2 * x);
  end Guarded;
end LazyBranches;
