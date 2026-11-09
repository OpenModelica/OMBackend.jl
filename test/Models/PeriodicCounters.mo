package PeriodicCounters
  model Trapezoid
    "Modelica.Blocks.Sources.Trapezoid's period counter (MSL 4.1): a when on time >= (pre(count) + 1)*period + startTime.
     MSL OpAmps.Differentiator starts it before time 0 (startTime = -(rising + width/2) = -0.035): the counter steps
     at t = 0.065 + k*0.1, and y is the trapezoid between -5 and 5 (5 from T_start + 0.02 to T_start + 0.05)."
    parameter Real period = 0.1;
    parameter Real startTime = -0.035;
    parameter Real rising = 0.02;
    parameter Real width = 0.03;
    parameter Real falling = 0.02;
    parameter Real amplitude = 10;
    parameter Real offset = -5;
    Real T_start;
    Integer count;
    Real y;
  initial algorithm
    count := integer((time - startTime)/period);
    T_start := startTime + count*period;
  equation
    when time >= (pre(count) + 1)*period + startTime then
      count = pre(count) + 1;
      T_start = time;
    end when;
    y = offset + (if time < startTime then 0 else if time < T_start + rising then amplitude*(time - T_start)/rising
                  else if time < T_start + rising + width then amplitude
                  else if time < T_start + rising + width + falling then amplitude*(T_start + rising + width + falling - time)/falling
                  else 0);
  end Trapezoid;

  model TrapezoidDifferentiator
    "MSL OpAmps.Differentiator's circuit: the capacitor from the source to the op-amp's inverting input, the
     resistor to its output, a finite gain V0 (index 1; the array code's path). vOut = -R*C*der(y) to 1/V0:
     -R*C*amplitude/rising = -50 on the rising edges, 50 on the falling, 0 between."
    parameter Real C = 1e-3;
    parameter Real R = 100;
    parameter Real V0 = 15000;
    Trapezoid signalSource;
    Real vc(start = 5, fixed = true);
    Real i;
    Real vn;
    Real vOut;
  equation
    C*der(vc) = i;
    vc = signalSource.y - vn;
    i = (vn - vOut)/R;
    vOut = -V0*vn;
  end TrapezoidDifferentiator;
end PeriodicCounters;
