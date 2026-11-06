package EnumIndexedParameters "Parameter arrays read by literal subscripts (an enumeration's, a Boolean's) and by a variable subscript"
  type E = enumeration(a, b, c);

  model Read "der(x) reads p[E.b] (2) and q[true] (10): x(1) = 12"
    parameter Real p[E] = {1, 2, 3};
    parameter Real q[Boolean] = {5, 10};
    Real x(start = 0, fixed = true);
  equation
    der(x) = p[E.b] + q[true];
  end Read;

  model Dynamic "der(y) reads r[i], i stepping from 1 to 3 at 0.5: y(1) = 0.5 * 1 + 0.5 * 3 = 2"
    parameter Real r[3] = {1, 2, 3};
    discrete Integer i(start = 1, fixed = true);
    Real y(start = 0, fixed = true);
  equation
    when time > 0.5 then
      i = 3;
    end when;
    der(y) = r[i];
  end Dynamic;
end EnumIndexedParameters;
