package ZeroFlow "Zero flow and zero speed: MSL FluidHeatFlow WaterPump's components, reduced"

  connector FlowPort
    Real p;
    flow Real m_flow;
    Real h;
    flow Real H_flow;
  end FlowPort;

  model Source "a pressure source: its pressure falls through the sink's"
    parameter Real p0 = 1e5;
    parameter Real dp0 = 1e3;
    parameter Real h0 = 1e6;
    FlowPort port;
  equation
    port.p = p0 + dp0 * (0.5 - time);
    port.H_flow = semiLinear(port.m_flow, port.h, h0);
  end Source;

  model Sink
    parameter Real p0 = 1e5;
    parameter Real h0 = 2e6;
    FlowPort port;
  equation
    port.p = p0;
    port.H_flow = semiLinear(port.m_flow, port.h, h0);
  end Sink;

  model ZeroMass "a component without mass (FluidHeatFlow's TwoPort with m = 0): its friction loss fed to the medium"
    parameter Real R = 1e4;
    parameter Real frictionLoss = 0;
    FlowPort a;
    FlowPort b;
    Real dp = a.p - b.p;
    Real h;
    Real Q_flow;
  equation
    a.m_flow + b.m_flow = 0;
    a.H_flow + b.H_flow + Q_flow = 0;
    a.H_flow = semiLinear(a.m_flow, a.h, h);
    b.H_flow = semiLinear(b.m_flow, b.h, h);
    Q_flow = frictionLoss * a.m_flow * dp;
    dp = R * a.m_flow;
  end ZeroMass;

  model ZeroMassChain "three components without mass in series, the flow through zero at t = 0.5"
    Source source;
    ZeroMass c1;
    ZeroMass c2;
    ZeroMass c3;
    Sink sink;
    Real E(start = 0, fixed = true) "the enthalpy the sink takes in";
  equation
    der(E) = sink.port.H_flow;
    connect(source.port, c1.a);
    connect(c1.b, c2.a);
    connect(c2.b, c3.a);
    connect(c3.b, sink.port);
  end ZeroMassChain;

  model NoEventSpeed "a pump's characteristic on noEvent(abs(w) < small) (FluidHeatFlow IdealPump): w exactly zero after 0.5"
    parameter Real small = 1e-60;
    discrete Real d(start = 1, fixed = true);
    Real w;
    Real V(start = 1, fixed = true);
    Real dp;
    Real tau;
  equation
    when time > 0.5 then
      d = 0;
    end when;
    w = 2 * d;
    der(V) = -V;
    if noEvent(abs(w) < small) then
      dp = 0;
      tau = 0;
    else
      dp = w * V;
      tau * w = dp * V;
    end if;
  end NoEventSpeed;

  model SmoothStream "the frontend's actualStream(h) of a port, m_flow * smooth(0, if m_flow > 0 then inStream(h) else h): the flow through zero at t = 0.5 with an event (smooth of order 0: its derivative jumps)"
    Real m_flow(start = 0.5, fixed = true);
    Real hIn = 1 + time;
    parameter Real h = 2;
    Real H_flow;
    Real E(start = 0, fixed = true);
  equation
    der(m_flow) = -1;
    H_flow = m_flow * smooth(0, if m_flow > 0 then hIn else h);
    der(E) = H_flow;
  end SmoothStream;

  model SmoothQuadraticLoad "a quadratic load torque through zero speed at t = 0.25, smooth(1, if w >= 0 then w^2 else -w^2) (MSL QuadraticSpeedDependentTorque): no event (smooth of order 1, MLS 3.7.4.2)"
    Real w(start = -0.25, fixed = true);
    Real tau;
    Real E(start = 0, fixed = true);
  equation
    der(w) = 1;
    tau = smooth(1, if w >= 0 then w^2 else -w^2);
    der(E) = tau;
  end SmoothQuadraticLoad;

  model FlatZeroFlow "two quadratic losses in series from 110 kPa to 100 kPa, the flow from its start 0, where both are flat to the initialization's finite differences (MSL MomentumBalanceFittings): 3 m_flow^2 = 1e4"
    parameter Real pa = 110000;
    parameter Real pb = 100000;
    Real m_flow(start = 0);
    Real p(start = 5e6);
    Real V(start = 0, fixed = true);
  equation
    pa - p = m_flow * abs(m_flow);
    p - pb = 2 * m_flow * abs(m_flow);
    der(V) = m_flow;
  end FlatZeroFlow;

end ZeroFlow;
