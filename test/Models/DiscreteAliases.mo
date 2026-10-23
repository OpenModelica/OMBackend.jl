package DiscreteAliases "pure aliases of two held discretes, both read in when conditions (MSL RadioButtonSource's reset inputs: each button's `on` in two other buttons' reset vectors), their initial equations pre(reset) = false; values from omc 1.27.1"
  block Button "on at pressAt, off when an element of reset was true at the previous event (MSL RadioButtonSource)"
    parameter Real pressAt;
    input Boolean reset[2];
    output Boolean on(start = false, fixed = true);
  protected
    Boolean pre_reset[2];
  initial equation
    pre(pre_reset) = fill(false, 2);
    pre(reset) = fill(false, 2);
  algorithm
    pre_reset := pre(reset);
    when pre_reset then
      on := false;
    end when;
    when time >= pressAt then
      on := true;
    end when;
  end Button;

  model Three "start, stop, shut resetting each other"
    Button start(pressAt = 1, reset = {stop.on, shut.on});
    Button stop(pressAt = 2, reset = {start.on, shut.on});
    Button shut(pressAt = 3, reset = {start.on, stop.on});
  end Three;
end DiscreteAliases;
