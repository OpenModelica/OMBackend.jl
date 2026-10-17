model ProtectedAttributes "protected variables of each type keep their attributes"
  type E = enumeration(one, two, three);
  Real x(start = 1, fixed = true);
protected
  E e(start = E.two, fixed = true);
  Integer k(start = 2, fixed = true, min = 0, max = 5);
  Boolean b(start = true, fixed = true);
  Integer m[3](each start = 4, each fixed = true);
equation
  der(x) = -x;
  when x < 0.5 then
    e = E.three;
    k = pre(k) + 1;
    b = false;
    m = {1, 2, 3};
  end when;
end ProtectedAttributes;
