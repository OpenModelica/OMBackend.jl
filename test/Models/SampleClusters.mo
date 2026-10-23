package SampleClusters "sample() read by a discrete cluster (an RS flip-flop: MSL BooleanNetwork1), and an initial algorithm section reading a variable the initialization determines (MSL MathBoolean.OnDelay); values from omc 1.27.1"
  model FlipFlop "set at 0, 0.5, ...; reset at 0.3, 0.8, ... (MSL Logical.RSFlipFlop's two Nor gates and Pre)"
    Boolean set = sample(0, 0.5);
    Boolean reset = sample(0.3, 0.5);
    Boolean Q;
    Boolean QI;
    Boolean preQI(start = true, fixed = true);
  equation
    preQI = pre(QI);
    Q = not (reset or preQI);
    QI = not (set or Q);
  end FlipFlop;

  model LateSection "delaySignal from u at the initialization, u a relation the initialization evaluates"
    Boolean u = time >= 0;
    Boolean delaySignal(start = false, fixed = true);
    discrete Real t_next;
    Boolean y;
  algorithm
    when initial() then
      delaySignal := u;
      t_next := time - 1;
    elsewhen not u then
      delaySignal := false;
      t_next := time - 1;
    end when;
  equation
    y = if delaySignal then time >= t_next else false;
  end LateSection;
end SampleClusters;
