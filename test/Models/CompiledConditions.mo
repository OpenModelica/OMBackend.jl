package CompiledConditions "When conditions compiled: a continuous when (WhenCondition), whens on one relation, a discrete when (TieredCondition)"
  function twice "2 u"
    input Real u;
    output Real y;
  algorithm
    y := 2 * u;
  end twice;

  model Whens "s = time; the whens record s where their conditions become true (as MSL SoftStartControl's whens)"
    parameter Real k = 0.6;
    Real s(start = 0, fixed = true);
    discrete Real sA(start = 0, fixed = true) "twice(s) >= 0.6 (a continuous when, it has an elsewhen): 0.3";
    discrete Real sB(start = 0, fixed = true) "s >= 0.8 (its elsewhen, one relation): 0.8";
    discrete Real sK(start = 0, fixed = true) "s > k (one relation): 0.6";
    Boolean b = s > 0.45;
    Integer n(start = 0, fixed = true) "the times b became true: 1";
    discrete Real sE(start = 0, fixed = true) "when b (a discrete when): 0.45";
  equation
    der(s) = 1;
    when twice(s) >= 0.6 then
      sA = s;
      sB = pre(sB);
    elsewhen s >= 0.8 then
      sA = pre(sA);
      sB = s;
    end when;
    when s > k then
      sK = s;
    end when;
    when b then
      n = pre(n) + 1;
      sE = s;
    end when;
  end Whens;
end CompiledConditions;
