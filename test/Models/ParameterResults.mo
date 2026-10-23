package ParameterResults "a model without continuous equations: its parameters are its result (MSL readRealParameterModel); one computed through an external C function's String output, a const char** (ModelicaStrings_scanIdentifier)"
  function scanIdent "the identifier of s from startIndex (MSL Strings.scanIdentifier)"
    input String s;
    input Integer startIndex;
    output Integer nextIndex;
    output String identifier;
  external "C" ModelicaStrings_scanIdentifier(s, startIndex, nextIndex, identifier) annotation(Library = "ModelicaExternalC");
  end scanIdent;

  function identEnd "the index after the identifier of s if it is myVar, else -1"
    input String s;
    output Real y;
  protected
    Integer next;
    String r;
  algorithm
    (next, r) := scanIdent(s, 1);
    y := if r == "myVar" then next else -1;
  end identEnd;

  model OnlyParameters "no equations: c and d are its result (from a String parameter, as readRealParameterModel's from its file)"
    parameter String s = "myVar = 42";
    parameter Real c = identEnd(s);
    parameter Real d = 2 * c;
  end OnlyParameters;
end ParameterResults;
