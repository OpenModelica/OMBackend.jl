package HeldDiscretes
  model NextEventBetweenCrossings
    "A discrete Real, next, written at t = k/3, and a relation that crosses 13 times in [0, 2]. next changes at its
     when only: 1/3 + k/3 by repeated addition, 7 values. y is an algebraic unknown (a singular mass matrix: each
     event re-initializes)."
    parameter Real step = 1/3;
    discrete Real next(start = 1/3, fixed = true);
    Real x(start = 0, fixed = true);
    Real y;
    Boolean up;
  equation
    der(x) = 1 - x;
    up = sin(40*time) > 0.3;
    y = if up then x else -x;
    when time >= pre(next) then
      next = pre(next) + step;
    end when;
  end NextEventBetweenCrossings;

  model NextEventBetweenWhens
    "Two discrete Reals written by two whens, no algebraic unknown (an ODE: no re-initialization at an event). Each
     changes at its own when only: next has 7 values in [0, 2], count 14 (0, then 0.1 added at each of the 13 rising
     crossings of sin(40*time) > 0.3)."
    parameter Real step = 1/3;
    discrete Real next(start = 1/3, fixed = true);
    discrete Real count(start = 0, fixed = true);
    Real x(start = 0, fixed = true);
  equation
    der(x) = 1 - x;
    when sin(40*time) > 0.3 then
      count = pre(count) + 0.1;
    end when;
    when time >= pre(next) then
      next = pre(next) + step;
    end when;
  end NextEventBetweenWhens;
end HeldDiscretes;
