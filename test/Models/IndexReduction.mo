package IndexReduction "Models for OMBackend's index reduction step by step (Passes.IndexReduction)"

  model Index1 "index 1: nothing to differentiate"
    Real x(start = 1, fixed = true);
    Real y;
  equation
    der(x) = -x + y;
    y = sin(time);
  end Index1;

  model RigidInertias "index 3: two inertias joined rigidly, phi1 = phi2; the first declared stay states"
    parameter Real J1 = 1;
    parameter Real J2 = 2;
    Real phi1(start = 0, fixed = true);
    Real w1(start = 0, fixed = true);
    Real phi2;
    Real w2;
    Real tau;
  equation
    der(phi1) = w1;
    der(phi2) = w2;
    J1 * der(w1) = sin(time) - tau;
    J2 * der(w2) = tau;
    phi1 = phi2;
  end RigidInertias;

  model RigidInertiasAlways "as RigidInertias, the second inertia's StateSelect.always"
    parameter Real J1 = 1;
    parameter Real J2 = 2;
    Real phi1;
    Real w1;
    Real phi2(stateSelect = StateSelect.always, start = 0, fixed = true);
    Real w2(stateSelect = StateSelect.always, start = 0, fixed = true);
    Real tau;
  equation
    der(phi1) = w1;
    der(phi2) = w2;
    J1 * der(w1) = sin(time) - tau;
    J2 * der(w2) = tau;
    phi1 = phi2;
  end RigidInertiasAlways;

  model CapacitorLoop "index 2: two capacitors in parallel"
    parameter Real C1 = 1;
    parameter Real C2 = 3;
    Real v1(start = 0, fixed = true);
    Real v2;
    Real i1;
    Real i2;
  equation
    C1 * der(v1) = i1;
    C2 * der(v2) = i2;
    v1 = v2;
    i1 + i2 = sin(time);
  end CapacitorLoop;

  model PendulumAngle "x = L*sin(phi): phi stays the state (an entry that reads the states can vanish)"
    parameter Real g = 9.81;
    parameter Real L = 2;
    Real x;
    Real vx;
    Real phi(start = 2.356194490192345, fixed = true);
    Real phid(start = 0, fixed = true);
  equation
    x = L * sin(phi);
    der(x) = vx;
    der(phi) = phid;
    der(phid) = -g / L * sin(phi);
  end PendulumAngle;

  model FixedAlias "of two aliases the one with a fixed start stays the state"
    Real phiSource;
    Real wSource;
    Real phi(start = 0, fixed = true);
    Real w(start = 10, fixed = true);
  equation
    der(phiSource) = wSource;
    der(phi) = w;
    der(w) = 0;
    phiSource = phi;
  end FixedAlias;

  model AlwaysUndifferentiated "w = 2 * w1 of StateSelect.always, which the model does not differentiate: a state"
    parameter Real J1 = 1;
    parameter Real J2 = 2;
    Real phi1(start = 0, fixed = true);
    Real w1(start = 0, fixed = true);
    Real phi2;
    Real w2;
    Real tau;
    Real w(stateSelect = StateSelect.always);
  equation
    der(phi1) = w1;
    der(phi2) = w2;
    J1 * der(w1) = sin(time) - tau;
    J2 * der(w2) = tau;
    phi1 = phi2;
    w = 2 * w1;
  end AlwaysUndifferentiated;

  function cubic "y = x^3 + x, with its derivative"
    input Real x;
    output Real y;
  algorithm
    y := x ^ 3 + x;
    annotation(derivative = cubicDer);
  end cubic;

  function cubicDer
    input Real x;
    input Real der_x;
    output Real der_y;
  algorithm
    der_y := (3 * x ^ 2 + 1) * der_x;
  end cubicDer;

  model FunctionConstraint "index 2 through a function with a derivative annotation: x2 = cubic(x1)"
    Real x1(start = 0, fixed = true);
    Real x2;
  equation
    der(x1) + der(x2) = 1;
    x2 = cubic(x1);
  end FunctionConstraint;

  function signedSquare "x |x| by an if-statement on a local, no derivative annotation"
    input Real x;
    output Real y;
  protected
    Real a;
  algorithm
    a := x;
    if x < 0 then
      y := -a * a;
    else
      y := a * a;
    end if;
  end signedSquare;

  model IfFunctionConstraint "index 2 through a function body with a local and an if-statement: x2 = signedSquare(x1) (x1 + x1^2 = t + 2)"
    Real x1(start = 1, fixed = true);
    Real x2;
  equation
    der(x1) + der(x2) = 1;
    x2 = signedSquare(x1);
  end IfFunctionConstraint;

  model IfEquationPositions "an if-equation with a Boolean position after its rows, only its first position differentiated (x1(1) = 1/6 + 1/8)"
    Real x1(start = 0, fixed = true);
    Real x2;
    Real i2;
    Boolean on;
  equation
    der(x1) + der(x2) = 1;
    if time < 0.5 then
      x2 = 2 * x1;
      i2 = x1;
      on = true;
    else
      x2 = 3 * x1;
      i2 = 0;
      on = false;
    end if;
  end IfEquationPositions;

  function chainPower "x^n through itself (recursive), no derivative annotation"
    input Real x;
    input Integer n;
    output Real y;
  algorithm
    y := if n <= 1 then x else x * chainPower(x, n - 1);
  end chainPower;

  model RecursiveFunctionConstraint "index 2 through a recursive function: left to MTK, not inlined without end"
    parameter Integer n = 2;
    Real x1(start = 1, fixed = true);
    Real x2;
  equation
    der(x1) + der(x2) = 1;
    x2 = chainPower(x1, n);
  end RecursiveFunctionConstraint;

  function sumOfLocalArray "an array local, which the symbolic run keeps no value of"
    input Real x;
    output Real y;
  protected
    Real c[2] = {x, 2 * x};
  algorithm
    y := c[1] + c[2];
  end sumOfLocalArray;

  model ArrayLocalConstraint "index 2 through a function with an array local: not inlined (c[1] is no model variable)"
    Real x1(start = 1, fixed = true);
    Real x2;
  equation
    der(x1) + der(x2) = 1;
    x2 = sumOfLocalArray(x1);
  end ArrayLocalConstraint;

  function powerN "x^n, its derivative by annotation (an Integer input: no derivative argument for it)"
    input Real x;
    input Integer n;
    output Real y;
  algorithm
    y := x ^ n;
    annotation(derivative = powerNDer);
  end powerN;

  function powerNDer
    input Real x;
    input Integer n;
    input Real der_x;
    output Real der_y;
  algorithm
    der_y := n * x ^ (n - 1) * der_x;
  end powerNDer;

  model IntegerInputDerivative "index 2 through powerN(x1, 2) (x1 + x1^2 = t + 2)"
    Real x1(start = 1, fixed = true);
    Real x2;
  equation
    der(x1) + der(x2) = 1;
    x2 = powerN(x1, 2);
  end IntegerInputDerivative;

  model ZeroStartCapacitance "der(v1) through v1 * der(v1) = i1 (0 at the given start) or (2 + v2^2) * der(v2) = i2: solved through the second"
    Real v1(start = 0, fixed = true);
    Real v2;
    Real i1;
    Real i2;
  equation
    v1 * der(v1) = i1;
    (2 + v2 ^ 2) * der(v2) = i2;
    v1 = v2;
    i1 + i2 = sin(time);
  end ZeroStartCapacitance;

  model DerOfExpression "der() of an expression by the chain rule: index 2 through der(x1 + x2) with x2 = 2 x1 (x1 = exp(-t/3))"
    Real x1(start = 1, fixed = true);
    Real x2;
  equation
    der(x1 + x2) = -x1;
    x2 = 2 * x1;
  end DerOfExpression;

  model KnownValues "x = k * y with k = 0 reads no y once folded (x = 0); c = 0 and x = 0 propagate into z = c * y + x"
    parameter Real k = 0;
    Real x;
    Real y(start = 0, fixed = true);
    Real c;
    Real z;
  equation
    der(y) = 1;
    x = k * y;
    c = 0;
    z = c * y + x;
  end KnownValues;

  model Partials "equations to differentiate by x and y at the start values (x = 0.5, y = 2)"
    Real x(start = 0.5);
    Real y(start = 2);
    Real a;
    Real b;
    Real c;
  equation
    a = sin(x) * y + x ^ 2 / y;
    b = exp(x) * log(y) + sqrt(y) * x - cos(x * y);
    c = atan2(x, y) + max(x, y) + semiLinear(x, 2, 3) + x / y + x ^ y;
    der(x) = 1;
    der(y) = 1;
  end Partials;

  model NonlinearCapacitors "index 2: capacitors in parallel, capacitances depending on the voltage (3 v + 2/3 v^3 = 1 - cos(t))"
    parameter Real C2 = 2;
    Real v1(start = 0, fixed = true);
    Real v2;
    Real i1;
    Real i2;
  equation
    (1 + v1 ^ 2) * der(v1) = i1;
    (C2 + v2 ^ 2) * der(v2) = i2;
    v1 = v2;
    i1 + i2 = sin(time);
  end NonlinearCapacitors;

  record Pair
    Real a;
    Real b;
  end Pair;

  function initPair
    input Real k;
    output Pair p;
  algorithm
    p := Pair(k, 2 * k);
  end initPair;

  model AlgorithmRecordOverwrite "an algorithm assigns a record whole, then one of its fields (MSL Spice3's semiconductor resistor)"
    parameter Real k = 3;
    Pair p;
    Real y;
  algorithm
    p := initPair(k);
    p.b := p.a + 1;
    y := p.b * time;
  end AlgorithmRecordOverwrite;

  model RotatedCurrents "a current reference rotated by the controller's angle, back by the machine's (MSL ActuatorWithNoise): the two angles tied by explicit equations, at the start no current"
    parameter Real p = 2;
    Real phi(start = 0, fixed = true) "the rotor's angle";
    Real w(start = 0, fixed = true);
    Real gamma "the machine's electrical angle";
    Real gammaC "the controller's electrical angle";
    Real iqRef(start = 0, fixed = true) "the filtered current reference";
    Real iS[2] "the stator currents the source imposes";
    Real iR[2](each stateSelect = StateSelect.prefer) "the currents in the rotor frame";
    Real vR[2];
  equation
    der(phi) = w;
    der(w) = iR[2] - w;
    gamma = p * phi;
    gammaC = p * phi;
    der(iqRef) = 1 - iqRef;
    iS = {-sin(gammaC) * iqRef, cos(gammaC) * iqRef};
    iR = {cos(gamma) * iS[1] + sin(gamma) * iS[2], -sin(gamma) * iS[1] + cos(gamma) * iS[2]};
    vR = {0.1 * iR[1] + 0.01 * der(iR[1]) - 0.01 * w * iR[2], 0.1 * iR[2] + 0.01 * der(iR[2]) + 0.01 * w * iR[1]};
  end RotatedCurrents;

  record Frame "an orientation: a rotation matrix and an angular velocity (as MSL Frames.Orientation)"
    Real T[3, 3];
    Real w[3];
  end Frame;

  function planarFrame "the frame rotated by angle about z"
    input Real angle;
    output Frame R;
  algorithm
    R := Frame(T = {{cos(angle), sin(angle), 0}, {-sin(angle), cos(angle), 0}, {0, 0, 1}}, w = zeros(3));
  end planarFrame;

  function frameQuaternion "the quaternion of a frame by the largest diagonal term (as MSL Frames.Quaternions.from_T): array locals, loops, an if, a sign by a scalar product"
    input Frame R;
    input Real guess[4] = {0, 0, 0, 1};
    output Real Q[4];
  protected
    Real c[4];
    Integer k;
    Integer n;
    Real p;
  algorithm
    c[4] := 1 + R.T[1, 1] + R.T[2, 2] + R.T[3, 3];
    for i in 1:3 loop
      c[i] := 1 + 2 * R.T[i, i] - (R.T[1, 1] + R.T[2, 2] + R.T[3, 3]);
    end for;
    k := 4;
    for i in 1:3 loop
      if c[i] > c[k] then
        k := i;
      end if;
    end for;
    n := 0;
    while n < 3 loop
      n := n + 1;
    end while;
    if k == 4 then
      p := sqrt(c[4]) / 2;
      Q := {(R.T[2, 3] - R.T[3, 2]) / (4 * p), (R.T[3, 1] - R.T[1, 3]) / (4 * p), (R.T[1, 2] - R.T[2, 1]) / (4 * p), p};
    else
      Q := guess * n / 3;
    end if;
    if Q * guess < 0 then
      Q := -Q;
    end if;
  end frameQuaternion;

  partial function Scalar "a scalar function: the type of a function argument"
    input Real x;
    output Real y;
  end Scalar;

  function scaled "k x"
    extends Scalar;
    input Real k = 1;
  algorithm
    y := k * x;
  end scaled;

  record Offset "a record with field defaults"
    Real a = 0.25;
    Real b[2] = {1, 2};
  end Offset;

  function sumOf "f(1) + ... + f(n) by a reduction; a matrix by two iterators, a column assigned by a slice, read by a slice and size, a record's defaults"
    input Scalar f;
    input Integer n;
    output Real s;
    output Real t;
  protected
    Real v[n];
    Real w[n, 2];
    Offset o;
  algorithm
    v := {f(i) for i in 1:n};
    s := sum(v[i] for i in 1:n);
    w := {v[i] * j for i in 1:n, j in 1:2};
    w[:, 2] := zeros(n);
    t := sum(w[2:n, 1]) + size(w, 1) + o.a + o.b[2];
  end sumOf;

  function sumOfSecond "the second output of sumOf: a tuple assignment, the function input passed on"
    input Scalar f;
    input Integer n;
    output Real t;
  protected
    Real s;
  algorithm
    (s, t) := sumOf(f, n);
  end sumOfSecond;

  model FunctionStart "start values through functions: a record of a function, a quaternion of that record, its sign flipped by the guess (Q0 = -{0, 0, sin(0.3), cos(0.3)})"
    parameter Real a = 0.6;
    parameter Frame R0 = planarFrame(a);
    parameter Real Q0[4] = frameQuaternion(R0, {0, 0, -1, -1});
    parameter Real s1 = sumOf(function scaled(k = a + 1.4), 3) "2 + 4 + 6 = 12";
    parameter Real t1 = sumOfSecond(function scaled(k = a + 1.4), 3) "4 + 6 + 3 + 0.25 + 2 = 15.25";
    Real q(start = Q0[3], fixed = true);
  equation
    der(q) = -q;
  end FunctionStart;

  model PendulumCartesian "index 3: a pendulum in Cartesian coordinates, released at 30 degrees from the horizontal (dynamic state sets: x or y, vx or vy)"
    parameter Real L = 1;
    parameter Real m = 1;
    parameter Real g = 9.81;
    Real x(start = cos(0.5235987755982988), fixed = true);
    Real y(start = -sin(0.5235987755982988));
    Real vx(start = 0, fixed = true);
    Real vy;
    Real F;
  equation
    der(x) = vx;
    der(y) = vy;
    m * der(vx) = -x / L * F;
    m * der(vy) = -y / L * F - m * g;
    x ^ 2 + y ^ 2 = L ^ 2;
  end PendulumCartesian;

  model QuaternionAttitude "index 2: an orientation as a quaternion (Q[4] the scalar part) at a given rate (a dynamic state set: 3 of 4)"
    Real Q[4](start = {0, 0, 0, 1});
    Real w[3];
  equation
    w = {1, 0.5, 0.2};
    w = 2 * {Q[4] * der(Q[1]) + Q[3] * der(Q[2]) - Q[2] * der(Q[3]) - Q[1] * der(Q[4]),
             -Q[3] * der(Q[1]) + Q[4] * der(Q[2]) + Q[1] * der(Q[3]) - Q[2] * der(Q[4]),
             Q[2] * der(Q[1]) - Q[1] * der(Q[2]) + Q[4] * der(Q[3]) - Q[3] * der(Q[4])};
    0 = Q * Q - 1;
  initial equation
    Q[1] = 0;
    Q[2] = 0;
    Q[3] = 0;
  end QuaternionAttitude;

  model AlgebraicNever "s only index reduction differentiates (no StateSelect): with the dynamic state sets' ranking StateSelect.never, so x (avoid) stays the state; otherwise s (default)"
    Real x(stateSelect = StateSelect.avoid, start = 0.5, fixed = true);
    Real z(stateSelect = StateSelect.never);
    Real s;
  equation
    der(x) + der(z) = 1;
    s = x ^ 2;
    z = s + 1;
  end AlgebraicNever;

  model ImplicitAlias "p (no StateSelect, its alias pa avoid) and q (avoid): with the dynamic state sets' ranking p no higher than avoid, so p's constant entry makes it the dummy (MSL Body's frame_a.r_0 against its quaternion); otherwise p (default) stays the state"
    Real q(stateSelect = StateSelect.avoid, start = 0.5, fixed = true);
    Real p;
    Real pa(stateSelect = StateSelect.avoid);
    Real f;
  equation
    pa = p;
    der(q) = 1 - f;
    der(p) = f;
    p = q ^ 2 + 1;
  end ImplicitAlias;

  model PendulumArray "three Cartesian pendulums in a for-equation (index 3), released at different angles (the start values choose x as the dummy of the first, y of the others); arrays kept: the constraint differentiated in array form, the dynamic state sets of the elements one family"
    parameter Integer N = 3;
    parameter Real L = 1;
    parameter Real g = 9.81;
    Real x[N](start = {0.8, 0.3, 0.6}, each fixed = true);
    Real y[N](start = {-0.6, -0.95, -0.8});
    Real vx[N](each start = 0, each fixed = true);
    Real vy[N];
    Real F[N];
  equation
    for i in 1:N loop
      der(x[i]) = vx[i];
      der(y[i]) = vy[i];
      der(vx[i]) = -x[i] / L * F[i];
      der(vy[i]) = -y[i] / L * F[i] - g;
      x[i] ^ 2 + y[i] ^ 2 = L ^ 2;
    end for;
  end PendulumArray;

  model InertiaArray "N rigidly coupled pairs of inertias as array equations (index 3 as written); arrays kept: phi1 and w1 the states as arrays"
    parameter Integer N = 4;
    Real phi1[N](each start = 0, each fixed = true);
    Real w1[N](each start = 0, each fixed = true);
    Real phi2[N];
    Real w2[N];
    Real tau[N];
  equation
    der(phi1) = w1;
    der(phi2) = w2;
    der(w1) = {sin(time) * i for i in 1:N} - tau;
    2 * der(w2) = tau;
    phi1 = phi2;
  end InertiaArray;

  model BoundaryArray "the first two elements of x tied: their kinds differ from the third's, which the array form does not take (slicing); the frontend scalarizes"
    parameter Integer N = 3;
    Real x[N](start = {1, 1, 0.5}, fixed = {true, false, true});
    Real f;
  equation
    for i in 2:N loop
      der(x[i]) = -x[i];
    end for;
    der(x[1]) = -x[1] + f;
    x[1] = x[2];
  end BoundaryArray;

  model BoundaryWithArray "BoundaryArray's tie splits x into its elements (slicing); the array y stays an array"
    parameter Integer N = 3;
    parameter Integer M = 4;
    Real x[N](start = {1, 1, 0.5}, fixed = {true, false, true});
    Real f;
    Real y[M](each start = 1, each fixed = true);
  equation
    for i in 2:N loop
      der(x[i]) = -x[i];
    end for;
    der(x[1]) = -x[1] + f;
    x[1] = x[2];
    for j in 1:M loop
      der(y[j]) = -j * y[j];
    end for;
  end BoundaryWithArray;

  model AngleArray "PendulumAngle per element in a for-equation: the differentiated body x[i] = L * sin(phi[i]) reads the parameter L (kept symbolic in array form)"
    parameter Integer N = 3;
    parameter Real g = 9.81;
    parameter Real L = 2;
    Real x[N];
    Real vx[N];
    Real phi[N](start = {2.356194490192345, 2.0, 1.0}, each fixed = true);
    Real phid[N](each start = 0, each fixed = true);
  equation
    for i in 1:N loop
      x[i] = L * sin(phi[i]);
      der(x[i]) = vx[i];
      der(phi[i]) = phid[i];
      der(phid[i]) = -g / L * sin(phi[i]);
    end for;
  end AngleArray;

  model PendulumGrid "a 2 x 2 grid of Cartesian pendulums of different lengths in nested for-equations: a family of dynamic state sets over a matrix"
    parameter Real g = 9.81;
    parameter Real L[2, 2] = {{1, 1.5}, {2, 0.5}};
    parameter Real angle[2, 2] = {{0.5, 0.7}, {0.9, 0.3}};
    Real x[2, 2](start = {{L[i, j] * cos(angle[i, j]) for j in 1:2} for i in 1:2}, each fixed = true);
    Real y[2, 2](start = {{-L[i, j] * sin(angle[i, j]) for j in 1:2} for i in 1:2});
    Real vx[2, 2](each start = 0, each fixed = true);
    Real vy[2, 2];
    Real F[2, 2];
  equation
    for i in 1:2 loop
      for j in 1:2 loop
        der(x[i, j]) = vx[i, j];
        der(y[i, j]) = vy[i, j];
        der(vx[i, j]) = -x[i, j] / L[i, j] * F[i, j];
        der(vy[i, j]) = -y[i, j] / L[i, j] * F[i, j] - g;
        x[i, j] ^ 2 + y[i, j] ^ 2 = L[i, j] ^ 2;
      end for;
    end for;
  end PendulumGrid;

  model EllipseArray "masses on ellipses x^2 + c y^2 = 1, c a parameter array: the sets' pivots (2 x, 2 c[i] y) read it"
    parameter Integer N = 3;
    parameter Real g = 9.81;
    parameter Real c[N] = {1, 2, 0.5};
    Real x[N](start = {0.8, 0.5, 0.9}, each fixed = true);
    Real y[N](start = {-0.6, -sqrt((1 - 0.25) / 2), -sqrt((1 - 0.81) / 0.5)});
    Real vx[N](each start = 0, each fixed = true);
    Real vy[N];
    Real F[N];
  equation
    for i in 1:N loop
      der(x[i]) = vx[i];
      der(y[i]) = vy[i];
      der(vx[i]) = -x[i] * F[i];
      der(vy[i]) = -c[i] * y[i] * F[i] - g;
      x[i] ^ 2 + c[i] * y[i] ^ 2 = 1;
    end for;
  end EllipseArray;

  model Pendulum "a Cartesian pendulum, a component"
    parameter Real L = 1;
    parameter Real g = 9.81;
    parameter Real angle = 0.5;
    Real x(start = L * cos(angle), fixed = true);
    Real y(start = -L * sin(angle));
    Real vx(start = 0, fixed = true);
    Real vy;
    Real F;
  equation
    der(x) = vx;
    der(y) = vy;
    der(vx) = -x / L * F;
    der(vy) = -y / L * F - g;
    x ^ 2 + y ^ 2 = L ^ 2;
  end Pendulum;

  model ComponentPendulums "an array of pendulum components (p[i].x: the dimension on the component), released at different angles"
    Pendulum p[3](angle = {0.4, 0.6, 0.8});
  end ComponentPendulums;

  model MixedPendulums "a scalar pendulum beside an array of two: a dynamic state set of scalars and a family"
    Pendulum single(angle = 0.3);
    parameter Real g = 9.81;
    Real x[2](start = {0.8, 0.6}, each fixed = true);
    Real y[2](start = {-0.6, -0.8});
    Real vx[2](each start = 0, each fixed = true);
    Real vy[2];
    Real F[2];
  equation
    for i in 1:2 loop
      der(x[i]) = vx[i];
      der(y[i]) = vy[i];
      der(vx[i]) = -x[i] * F[i];
      der(vy[i]) = -y[i] * F[i] - g;
      x[i] ^ 2 + y[i] ^ 2 = 1;
    end for;
  end MixedPendulums;

  model PendulumsFirstForceLaw "PendulumArray, the first pendulum's force law twice the others': its derivatives solved from other bodies, solvedInArrays splits the arrays in place"
    parameter Integer n = 3;
    Real x[n](each start = 0.6, each fixed = true);
    Real y[n](each start = -0.8);
    Real vx[n](each start = 0, each fixed = true);
    Real vy[n];
    Real F[n];
  equation
    for i in 1:n loop
      der(x[i]) = vx[i];
      der(y[i]) = vy[i];
      x[i]^2 + y[i]^2 = 1;
    end for;
    der(vx[1]) = -2 * F[1] * x[1];
    der(vy[1]) = -2 * F[1] * y[1] - 9.81;
    for i in 2:n loop
      der(vx[i]) = -F[i] * x[i];
      der(vy[i]) = -F[i] * y[i] - 9.81;
    end for;
  end PendulumsFirstForceLaw;

  model EachStartArray "each start = sum(m): one start for every element of phi1 (not m's elements); rigidly coupled pairs at rest"
    parameter Real m[3] = {0.1, 0.2, 0.3};
    Real phi1[3](each start = sum(m), each fixed = true);
    Real w1[3](each start = 0, each fixed = true);
    Real phi2[3];
    Real w2[3];
    Real tau[3];
  equation
    der(phi1) = w1;
    der(phi2) = w2;
    der(w1) = -tau;
    der(w2) = tau;
    phi1 = phi2;
  end EachStartArray;

  model PendulumsPrefer "Cartesian pendulums, x and vx the states (StateSelect.prefer): the reduced model solves y, the root of x^2 + y^2 = 1 near its start, per element by Newton"
    parameter Integer N = 3;
    parameter Real g = 9.81;
    Real x[N](start = {0.5, 0.6, 0.7}, each fixed = true, each stateSelect = StateSelect.prefer);
    Real vx[N](each start = 0, each fixed = true, each stateSelect = StateSelect.prefer);
    Real y[N](each start = -0.8);
    Real vy[N];
    Real F[N];
  equation
    for i in 1:N loop
      der(x[i]) = vx[i];
      der(y[i]) = vy[i];
      der(vx[i]) = -x[i] * F[i];
      der(vy[i]) = -y[i] * F[i] - g;
      x[i] ^ 2 + y[i] ^ 2 = 1;
    end for;
  end PendulumsPrefer;

  record Phasor "a record of two Reals (as MSL Complex)"
    Real re;
    Real im;
  end Phasor;

  function makePhasor "a constructor function: its output defined by field bindings, no statement (as MSL Complex's 'constructor'.fromReal)"
    input Real re;
    input Real im = 0;
    output Phasor result(re = re, im = im);
  algorithm
  end makePhasor;

  function scalePhasor "a constructor called in a body (as MSL Complex's '*'.multiply)"
    input Phasor c;
    input Real k;
    output Phasor d;
  algorithm
    d := makePhasor(k * c.re, k * c.im);
  end scalePhasor;

  model ConstructorConstraint "index 2 through a record a constructor function builds: 2 (x1 + x2) = sin(t), i1 = i2 (x1 = x2 = sin(t)/4)"
    Real x1(start = 0, fixed = true);
    Real x2;
    Real i1;
    Real i2;
    Phasor p;
  equation
    der(x1) = i1;
    der(x2) = i2;
    p = scalePhasor(makePhasor(x1, x2), 2.0);
    p.re + p.im = sin(time);
    i1 = i2;
  end ConstructorConstraint;

  record Aux "an auxiliary property record (as IF97's ThermoProperties)"
    Real s;
    Real c;
  end Aux;

  function auxOf "the auxiliary record of x, no derivative annotation (as IF97's waterBaseProp_ph)"
    input Real x;
    input Integer phase = 0;
    output Aux aux;
  algorithm
    aux.s := x ^ 2;
    aux.c := 3;
  end auxOf;

  function fromAux "x + aux.s / aux.c, its derivative excluding the record (as IF97's rho_props_ph: noDerivative = aux)"
    input Real x;
    input Aux aux;
    output Real y;
  algorithm
    y := x + aux.s / aux.c;
    annotation(derivative(noDerivative = aux) = fromAuxDer, Inline = false, LateInline = true);
  end fromAux;

  function fromAuxDer
    input Real x;
    input Aux aux;
    input Real der_x;
    output Real der_y;
  algorithm
    der_y := (1 + 2 * x / aux.c) * der_x;
  end fromAuxDer;

  function wrapped "fromAux of x's auxiliary record, inlined, no derivative annotation (as IF97's rho_ph)"
    input Real x;
    input Integer phase = 0;
    output Real y;
  algorithm
    y := fromAux(x, auxOf(x, phase));
    annotation(Inline = true);
  end wrapped;

  function property "wrapped(x), inlined (as Water's density_ph): the frontend inlines it, and wrapped stays a call"
    input Real x;
    output Real y;
  algorithm
    y := wrapped(x, 0);
    annotation(Inline = true);
  end property;

  model InlinedRecordDerivative "index 2 through an inlined function passing a record a call computes to a function with a derivative annotation (as IF97's rho_ph): x1 + x2 = t, x2 = x1 + x1^2/3"
    Real x1(start = 0, fixed = true);
    Real x2;
  equation
    der(x1) + der(x2) = 1;
    x2 = property(x1);
  end InlinedRecordDerivative;

  model PreferredState "x of StateSelect.prefer, which the model does not differentiate, tied to the natural state m (as MSL Fluid's medium p and a volume's mass): an artificial state with OMBACKEND_PREFER_STATES (m = 2 exp(-t), 2 x + x^3 = m)"
    Real m(start = 2, fixed = true);
    Real x(stateSelect = StateSelect.prefer);
  equation
    der(m) = -m;
    m = 2 * x + x ^ 3;
  end PreferredState;

  record Gain "a record with an array field, passed to functions by its fields"
    Real k;
    Real offset[2];
  end Gain;

  function gainTimes "a record argument and an array argument: g.k x[1] + g.offset[1] x[2]"
    input Gain g;
    input Real x[2];
    output Real y;
  algorithm
    y := g.k * x[1] + g.offset[1] * x[2];
  end gainTimes;

  function gainOf "a record result"
    input Real k;
    output Gain g;
  algorithm
    g := Gain(k = k, offset = {0.5 * k, 0});
  end gainOf;

  model RecordsArray "InertiaArray driven through a record and an array argument, a record result, a fixed = false parameter, an if-equation on time, a literal indexed by the iterator; a record variable's array field read by a for-equation: array code after index reduction"
    parameter Integer N = 3;
    parameter Gain gain(k = 2, offset = {1, 0});
    parameter Real c(fixed = false) "6";
    Real phi1[N](each start = 0, each fixed = true);
    Real w1[N](each start = 0, each fixed = true);
    Real phi2[N];
    Real w2[N];
    Real tau[N];
    Real drive[N] "i (2 sin(t) + 1) before 0.5, 3 i after";
    Gain observed "k = w1[1], offset = phi1[1:2]";
    Gain fromC "gainOf(c): offset[1] = 3";
    Real sum1 "w1[1] + phi1[1]";
    Real weighted[N] "{1, 0.5, 2} .* w1: rolled, the literal indexed by the iterator";
  initial equation
    c = 3 * gain.k;
  equation
    der(phi1) = w1;
    der(phi2) = w2;
    for i in 1:N loop
      if time < 0.5 then
        drive[i] = gainTimes(gain, {sin(time), 1.0}) * i;
      else
        drive[i] = fromC.offset[1] * i;
      end if;
      der(w1[i]) = drive[i] - tau[i];
    end for;
    2 * der(w2) = tau;
    phi1 = phi2;
    observed.k = w1[1];
    observed.offset = phi1[1:2];
    sum1 = gainTimes(observed, {1.0, 1.0});
    weighted = {1, 0.5, 2} .* w1;
    fromC = gainOf(c);
  end RecordsArray;

  model RigidPair "a rigidly coupled pair of inertias driven by k sin(t) (index 3): phi1 = k (t - sin(t)) / 3"
    parameter Real k = 1;
    Real phi1(start = 0, fixed = true);
    Real w1(start = 0, fixed = true);
    Real phi2;
    Real w2;
    Real tau;
  equation
    der(phi1) = w1;
    der(phi2) = w2;
    der(w1) = k * sin(time) - tau;
    2 * der(w2) = tau;
    phi1 = phi2;
  end RigidPair;

  model SeparatePairs "three RigidPairs declared one by one: rolled into arrays over the instances (OMBACKEND_ROLL_COMPONENTS), reduced in array form; total reads all three"
    RigidPair a(k = 1);
    RigidPair b(k = 2);
    RigidPair c(k = 3);
    Real total "2 (t - sin(t))";
  equation
    total = a.phi1 + b.phi1 + c.phi1;
  end SeparatePairs;

  model DampedPair "a RigidPair damped above w1 = 0.5, by an if-expression the passes lift to an auxiliary (ifEq_tmp<k>)"
    parameter Real k = 1;
    Real phi1(start = 0, fixed = true);
    Real w1(start = 0, fixed = true);
    Real phi2;
    Real w2;
    Real tau;
    Real damping;
  equation
    der(phi1) = w1;
    der(phi2) = w2;
    der(w1) = k * sin(time) - tau - damping;
    2 * der(w2) = tau;
    phi1 = phi2;
    damping = 2 * (if w1 > 0.5 then 0.1 * (w1 - 0.5) else 0);
  end DampedPair;

  model SeparateDampedPairs "three DampedPairs declared one by one: their auxiliaries rolled into one array with them"
    DampedPair a(k = 1);
    DampedPair b(k = 2);
    DampedPair c(k = 3);
  end SeparateDampedPairs;

  model TimedPair "a RigidPair that records when phi1 passes 0.01 (k (t - sin(t)) / 3 = 0.01) and counts it"
    extends RigidPair;
    discrete Real tCross(start = -1, fixed = true);
    Integer crossings(start = 0, fixed = true);
  equation
    when phi1 > 0.01 then
      tCross = time;
      crossings = pre(crossings) + 1;
    end when;
  end TimedPair;

  model SeparateTimedPairs "three TimedPairs declared one by one: rolled, reduced in array form, the when-equation over the instances in the array code"
    TimedPair a(k = 1);
    TimedPair b(k = 2);
    TimedPair c(k = 3);
  end SeparateTimedPairs;

  connector FluidPort "MSL Media's test port: an enthalpy flow beside the mass flow"
    flow Real m_flow;
    Real h;
    flow Real H_flow;
  end FluidPort;

  model FlowSource "mass flow m_flow out of the source at enthalpy hs"
    input Real m_flow;
    parameter Real hs = 1;
    FluidPort port;
  equation
    port.m_flow = -m_flow;
    port.H_flow = semiLinear(port.m_flow, port.h, hs);
  end FlowSource;

  model EnthalpyVolume "an enthalpy store of mass M (index 2: U = M h, both differentiated)"
    parameter Real M = 1;
    Real U;
    Real h(start = 0, fixed = true);
    Real dh;
    FluidPort port;
  equation
    der(U) = port.H_flow;
    U = M * h;
    der(h) = dh;
    port.h = h;
    port.m_flow = 0;
  end EnthalpyVolume;

  model EnthalpyPipe "MSL Media's ShortPipe without pressures: the enthalpy carried by the flow"
    FluidPort a;
    FluidPort b;
  equation
    a.H_flow = semiLinear(a.m_flow, a.h, b.h);
    a.H_flow + b.H_flow = 0;
    a.m_flow + b.m_flow = 0;
  end EnthalpyPipe;

  model EnthalpyAmbient "MSL Media's FixedAmbient: enthalpy h0 at inflow"
    parameter Real h0 = 2;
    FluidPort port;
  equation
    port.H_flow = semiLinear(port.m_flow, port.h, h0);
  end EnthalpyAmbient;

  model SemiLinearChain "source -> v1 -> p1 -> v2 -> p2 -> ambient, the flow time (0 at the start); v and p rolled; p2's and the ambient's semiLinear a chain through p2.b.h (MLS 3.7.2.5). h1 = 1 - exp(-t^2/2), h2 = 1 - (1 + t^2/2) exp(-t^2/2)"
    FlowSource source(m_flow = time);
    EnthalpyVolume v1;
    EnthalpyVolume v2;
    EnthalpyPipe p1;
    EnthalpyPipe p2;
    EnthalpyAmbient amb;
  equation
    connect(source.port, v1.port);
    connect(v1.port, p1.a);
    connect(p1.b, v2.port);
    connect(v2.port, p2.a);
    connect(p2.b, amb.port);
  end SemiLinearChain;

end IndexReduction;
