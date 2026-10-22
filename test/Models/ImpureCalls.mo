package ImpureCalls "impure functions as MSL's Modelica.Math.Random.Utilities: a hidden C state"
  function setInternalState
    input Integer rngState[:];
    input Integer id;
  external "C" ModelicaRandom_setInternalState_xorshift1024star(rngState, size(rngState, 1), id);
  end setInternalState;

  impure function initializeImpureRandom "seeds the generator's state, as MSL's (a simpler state)"
    input Integer seed;
    output Integer id;
  protected
    Integer rngState[33];
  algorithm
    for i in 1:33 loop
      rngState[i] := seed + 7919 * i;
    end for;
    setInternalState(rngState, 715827883);
    id := 715827883;
  end initializeImpureRandom;

  impure function impureRandom "the next draw: a new value at each call"
    input Integer id;
    output Real y;
  external "C" y = ModelicaRandom_impureRandom_xorshift1024star(id);
  end impureRandom;

  model ImpureSample "as MSL's GlobalSeed and ImpureGenerator: a seed set once, one draw per sample"
    parameter Integer seed = 7;
    parameter Integer id(fixed = false);
    discrete Real r(start = 0, fixed = true);
    Real x[3](each start = 1, each fixed = true);
  initial equation
    id = initializeImpureRandom(seed);
  equation
    der(x) = -x;
    when sample(0, 0.1) then
      r = impureRandom(id);
    end when;
  end ImpureSample;
end ImpureCalls;
