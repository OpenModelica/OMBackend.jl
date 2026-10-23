package SolverClocks "clocked discretized continuous-time partitions (MLS 16.8.2): continuous equations on a solver clock, values from omc 1.27.1 (MSL Clocked ClockedWithDiscretizedContinuousController)"
  model ImplicitIntegrator "der(x) = u on a clock of 0.1 s with ImplicitEuler: x(ti) = x(ti-1) + 0.1 u(ti)"
    Real u = 1 + time;
    Real us;
    Real x(start = 0);
    Real y;
  equation
    us = sample(u, Clock(Clock(0.1), solverMethod = "ImplicitEuler"));
    der(x) = us;
    y = hold(x);
  end ImplicitIntegrator;

  model ExplicitDecay "der(x) = -x + u on a clock of 0.1 s with ExplicitEuler: x(ti) = x(ti-1) + 0.1 der(x)(ti-1)"
    Real u = 1;
    Real us;
    Real x(start = 2);
    Real y;
  equation
    us = sample(u, Clock(Clock(0.1), solverMethod = "ExplicitEuler"));
    der(x) = -x + us;
    y = hold(x);
  end ExplicitDecay;
end SolverClocks;
