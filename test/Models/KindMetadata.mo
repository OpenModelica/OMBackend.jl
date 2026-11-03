package KindMetadata "What the generated model's variables are in the simulation code (SimCodeKind)"

  model Pendulum "a pendulum MTK index-reduces (OMBACKEND_INDEX_REDUCTION=false): its derivative variable xˍt is MTK's, not x"
    parameter Real L = 1;
    parameter Real g = 9.81;
    Real x(start = 1, fixed = true);
    Real y(start = 0);
    Real vx(start = 0, fixed = true);
    Real vy;
    Real F;
  equation
    der(x) = vx;
    der(y) = vy;
    der(vx) = -F * x / L;
    der(vy) = -F * y / L - g;
    x ^ 2 + y ^ 2 = L ^ 2;
  end Pendulum;

  model Limiter "if-expression relations with a homotopy: the hysteresis and the homotopy parameter are declared with the conditions' parameters"
    Real u;
    Real y;
    Real x(start = 0, fixed = true);
  equation
    u = 2 * y + 0.1 * x;
    y = homotopy(actual = smooth(0, if u > 1 then 1 elseif u < -1 then -1 else u), simplified = 1);
    der(x) = -x;
  end Limiter;

end KindMetadata;
