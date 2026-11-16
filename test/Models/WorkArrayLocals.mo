package WorkArrayLocals
  function f "a work array used by literal subscripts only, as MSL IF97's o[20]"
    input Real x;
    output Real y;
  protected
    Real o[3];
  algorithm
    o[1] := x * x;
    o[2] := o[1] + 1.0;
    o[3] := 2.0 * o[2];
    y := o[3] - o[1];
  end f;

  function g "a work array also used by a computed subscript"
    input Real x;
    output Real y;
  protected
    Real o[2];
  algorithm
    for i in 1:2 loop
      o[i] := i * x;
    end for;
    y := o[1] + o[2];
  end g;

  model M
    Real y;
    Real z;
  equation
    y = f(time + 1.0);
    z = g(time);
  end M;
end WorkArrayLocals;
