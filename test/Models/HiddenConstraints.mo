package HiddenConstraints "A linear constraint among the states that no single equation shows"
  model FloatingBridge "Three chokes feed a floating resistor bridge (no ground: the balances of its five nodes add up to i1 + i2 + i3 = 0), next to two rigidly coupled masses (index 3, so the index is reduced)"
    parameter Real L = 1e-3;
    parameter Real R[6] = {1, 2, 2, 1, 1.5, 0.5};
    parameter Real Rload = 10;
    parameter Real V = 100;
    constant Real pi = 3.141592653589793;
    Real i1(start = 0, fixed = true);
    Real i2(start = 0, fixed = true);
    Real i3(start = 0);
    Real va, vb, vc, vp, vn;
    Real iap, ian, ibp, ibn, icp, icn, iload;
    Real x1(start = 0, fixed = true);
    Real v1(start = 0, fixed = true);
    Real x2, v2, f;
  equation
    L * der(i1) = V * sin(2 * pi * 50 * time) - va;
    L * der(i2) = V * sin(2 * pi * 50 * time - 2 * pi / 3) - vb;
    L * der(i3) = V * sin(2 * pi * 50 * time - 4 * pi / 3) - vc;
    i1 = iap + ian;
    i2 = ibp + ibn;
    i3 = icp + icn;
    iap + ibp + icp = iload;
    ian + ibn + icn + iload = 0;
    va - vp = R[1] * iap;
    va - vn = R[2] * ian;
    vb - vp = R[3] * ibp;
    vb - vn = R[4] * ibn;
    vc - vp = R[5] * icp;
    vc - vn = R[6] * icn;
    vp - vn = Rload * iload;
    der(x1) = v1;
    der(x2) = v2;
    der(v1) = 1 - f;
    2 * der(v2) = f;
    x1 = x2;
    annotation(experiment(StopTime = 0.02, Interval = 1e-4, Tolerance = 1e-8));
  end FloatingBridge;
end HiddenConstraints;
