package InitFixStates "underdetermined initializations: the unfixed states the equations leave free fixed at their start values (OpenModelica, Dymola)"
  model PartlyFixed "one initial equation for two states: OpenModelica fixes x (x = 1, y = 4)"
    Real x(start = 1);
    Real y(start = 2);
  initial equation
    x + y = 5;
  equation
    der(x) = -x;
    der(y) = -2*y;
  end PartlyFixed;

  model WithAlgebraic "the initial equation on an algebraic variable: OpenModelica fixes x (x = 1, y = 1.5, z = 4)"
    Real x(start = 1);
    Real y(start = 2);
    Real z;
  initial equation
    z = 4;
  equation
    z = x + 2*y;
    der(x) = -x;
    der(y) = -2*y;
  end WithAlgebraic;
end InitFixStates;
