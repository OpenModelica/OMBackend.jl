package CompiledFunctions "the model's functions compiled (OMBACKEND_COMPILED_FUNCTIONS): a comprehension, a nested call"
  function norm2 "an array constructor with an iterator: a comprehension"
    input Real u[:];
    output Real y;
  algorithm
    y := sqrt(sum({u[k]^2 for k in 1:size(u, 1)}));
  end norm2;

  function scaled "a call of another of the model's functions"
    input Real a;
    input Real b;
    output Real y;
  algorithm
    y := 2 * norm2({a, b});
  end scaled;

  model M
    Real y;
  equation
    y = scaled(3 * time, 4);
  end M;

  partial function Scalar
    input Real x;
    output Real y;
  end Scalar;

  function applyTo "f(x)"
    input Scalar f;
    input Real x;
    output Real y;
  algorithm
    y := f(x);
  end applyTo;

  function weighted
    extends Scalar;
    input Real w[:];
  algorithm
    y := x * sum(w);
  end weighted;

  package Medium
    constant Integer n = 2;
    function g "x (X[1] + X[2]) + X[3]: a partial application that reads the package's constant (MSL Fluid's T_phX, its nXi), the model's constant h_Med_n"
      input Real x;
      input Real X[3];
      output Real y;
    protected
      Real z;
    algorithm
      z := applyTo(function weighted(w = X[1:n]), x);
      y := z + X[3];
      annotation(Inline = false);
    end g;

    function f "g inlined: X[1:n] of the literal, a slice of a constant table"
      input Real x;
      input Real X[3];
      output Real y;
    algorithm
      y := applyTo(function weighted(w = X[1:n]), x) + X[3];
    end f;
  end Medium;

  model Holder
    replaceable package Med = Medium;
    Real s(start = 1, fixed = true);
    Real y;
    Real w;
  equation
    der(s) = Med.g(s, {-1, 0, 0});
    y = Med.g(time + 1, {1, 2, 4});
    w = Med.f(time + 1, {1, 2, 4});
  end Holder;

  model WithConstant "s = exp(-t); y = w = 3 (t + 1) + 4"
    Holder h;
  end WithConstant;
end CompiledFunctions;
