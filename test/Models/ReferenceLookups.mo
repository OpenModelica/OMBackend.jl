package ReferenceLookups "Code generation that asks the references: a table by a discrete index, an assigned array, a record array's fields"

  model DynamicIndex "A parameter table read by a discrete index (CDL's Integer TimeTable): tab[idx], idx a discrete"
    parameter Real tab[3] = {10, 20, 30};
    discrete Integer idx(start = 1, fixed = true);
    Real y;
  algorithm
    when sample(0.1, 0.2) then
      idx := if pre(idx) < 3 then pre(idx) + 1 else 3;
    end when;
  equation
    y = tab[idx];
  end DynamicIndex;

  model InitialArray "An array an initial algorithm assigns as a whole: its element variables' starts"
    Real x[3];
  initial algorithm
    x := {1, 2, 3};
  equation
    der(x) = -x;
  end InitialArray;

  record R
    Real a;
    Real b;
  end R;

  function weighted "An array of records' fields read by element (`r[k].a`: the field array `r.a`, element k)"
    input R r[2];
    output Real y;
  algorithm
    y := r[1].a * r[1].b + r[2].a * r[2].b;
  end weighted;

  model RecordArrayFields
    Real y;
    Real x(start = 0, fixed = true);
  equation
    y = weighted({R(1, 2), R(3, 4)});
    der(x) = y;
  end RecordArrayFields;

  model DerivativeAuxiliaryName "der(der(x)) beside a variable named der_x: the auxiliary for der(x), der.x, is a variable of its own"
    Real x(start = 1, fixed = true);
    Real der_x;
  initial equation
    der(x) = 0;
  equation
    der(der(x)) = -x;
    der_x = 2 * x;
  end DerivativeAuxiliaryName;

end ReferenceLookups;
