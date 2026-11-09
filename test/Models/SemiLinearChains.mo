package SemiLinearChains
  model ZeroFlowPump
    "MSL FluidHeatFlow OneMass's flow path in enthalpies: ambient -> pump (no storage, Q = 0) -> pipe (storage) ->
     ambient, connected by semiLinear energy flows; the flow starts from zero at t = 0.2. The pump's enthalpy is
     determined at zero flow only by the chain rule (MLS 3.7.2.5): hPump = hAmb. Then
     hPipe = hAmb + (hPipe0 - hAmb)*exp(-(t - 0.2)^2/(2*C)): 293.9652 at t = 1."
    parameter Real hAmb = 293.15;
    parameter Real C = 0.1;
    Real V "volume flow";
    Real H1 "ambient 1's port enthalpy flow";
    Real hA "enthalpy at ambient 1's and the pump's ports";
    Real Ha;
    Real Hb;
    Real Q "the pump's heat flow";
    Real hPump(start = 288.15);
    Real hB "enthalpy at the pump's and the pipe's ports";
    Real Hpa;
    Real Hpb;
    Real hPipe(start = 313.15, fixed = true);
    Real hC "enthalpy at the pipe's and ambient 2's ports";
    Real H2;
  equation
    V = if time < 0.2 then 0 else time - 0.2;
    H1 = semiLinear(-V, hA, hAmb);
    Ha + H1 = 0;
    Ha = semiLinear(V, hA, hPump);
    Hb = semiLinear(-V, hB, hPump);
    Q = 0;
    Ha + Hb + Q = 0;
    Hpa + Hb = 0;
    Hpa = semiLinear(V, hB, hPipe);
    Hpb = semiLinear(-V, hC, hPipe);
    C*der(hPipe) = Hpa + Hpb;
    H2 + Hpb = 0;
    H2 = semiLinear(V, hC, hAmb);
  end ZeroFlowPump;

  model ZeroFlowPumpT
    "ZeroFlowPump with the pump's enthalpy from its temperature, hPump = cp*TPump, as MSL FluidHeatFlow's
     components: after the alias elimination the chain's node at the pump is the expression cp*TPump. At zero
     flow TPump = hAmb/cp (the chain rule); hPipe as ZeroFlowPump's."
    parameter Real hAmb = 293.15;
    parameter Real C = 0.1;
    parameter Real cp = 4.2;
    Real V;
    Real H1;
    Real hA;
    Real Ha;
    Real Hb;
    Real Q;
    Real TPump(start = 288.15);
    Real hPump;
    Real hB;
    Real Hpa;
    Real Hpb;
    Real hPipe(start = 313.15, fixed = true);
    Real hC;
    Real H2;
  equation
    V = if time < 0.2 then 0 else time - 0.2;
    H1 = semiLinear(-V, hA, hAmb);
    Ha + H1 = 0;
    hPump = cp*TPump;
    Ha = semiLinear(V, hA, hPump);
    Hb = semiLinear(-V, hB, hPump);
    Q = 0;
    Ha + Hb + Q = 0;
    Hpa + Hb = 0;
    Hpa = semiLinear(V, hB, hPipe);
    Hpb = semiLinear(-V, hC, hPipe);
    C*der(hPipe) = Hpa + Hpb;
    H2 + Hpb = 0;
    H2 = semiLinear(V, hC, hAmb);
  end ZeroFlowPumpT;
end SemiLinearChains;
