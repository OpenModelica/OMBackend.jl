package InitialAlgorithms "Parameters an initial algorithm assigns"

  model AssignedParameter "MSL Fluid TraceSubstances' ind (its substance's index), DCPowerSupply's vLim, iLim, cp"
    parameter Integer ind(fixed = false);
    parameter Real k(fixed = false);
    Real x(start = 1, fixed = true);
    Real y(start = 1) "implicit: the initialization solves it, reading k";
  initial algorithm
    ind := 0;
    for i in 1:3 loop
      if i == 2 then
        ind := i;
      end if;
    end for;
    k := 2 * ind;
  equation
    der(x) = -k * x;
    y + y ^ 3 = k * x;
  end AssignedParameter;

end InitialAlgorithms;
