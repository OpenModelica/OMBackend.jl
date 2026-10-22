package RecordArrayLocals "a function's protected array of records bound by record-valued calls (MSL Ice09's Complex[2] t)"
  record R
    Real re;
    Real im;
  end R;

  function mk "a record-valued call, as Complex's 'constructor'.fromReal"
    input Real a;
    input Real b;
    output R r;
  algorithm
    r := R(a, b);
  end mk;

  function f
    input Real x;
    output Real y;
  protected
    R c[2] = {mk(1, 2), mk(3, 4)};
  algorithm
    y := x * (c[1].re + c[2].im);
  end f;

  model M
    Real y;
  equation
    y = f(time + 1);
  end M;
end RecordArrayLocals;
