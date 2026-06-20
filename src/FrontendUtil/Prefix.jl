module Prefix

using MetaModelica
#= ExportAll is not good practice but it makes it so that we do not have to write export after each function :( =#
using ExportAll
  #= Necessary to write declarations for your uniontypes until Julia adds support for mutually recursive types =#

@UniontypeDecl ComponentPrefix
@UniontypeDecl ClassPrefix

import SCode
import DAE.ClassInf

#= A Prefix has a component prefix and a class prefix.
The component prefix consist of a name an a list of constant valued subscripts.
The class prefix contains the variability of the class, i.e unspecified, parameter or constant. =#
@Uniontype PrefixType begin
  @Record NOPRE begin
  end

  @Record PREFIX begin
    compPre #= component prefixes are stored in inverse order c.b.a =#::ComponentPrefix
    classPre #= the class prefix, i.e. variability, var, discrete, param, const =#::ClassPrefix
  end
end

ComponentPrefixOpt = Option  #= a type alias for an optional component prefix =#

#= Prefix for component name, e.g. a.b[2].c.
NOTE: Component prefixes are stored in inverse order c.b[2].a! =#
@Uniontype ComponentPrefix begin
  @Record PRE begin
    prefix #= prefix name =#::String
    dimensions #= dimensions =#::List
    subscripts #= subscripts =#::List
    next #= next prefix =#::ComponentPrefix
    ci_state #= to be able to at least partially fill in type information properly for DAE.VAR =#::ClassInf.SMNode
    info::SourceInfo
  end

  @Record NOCOMPPRE begin
  end
end

#= Prefix for classes is its variability =#
@Uniontype ClassPrefix begin
  @Record CLASSPRE begin
    variability #= VAR, DISCRETE, PARAM, or CONST =#::SCode.Variability
  end
end

#= So that we can use wildcard imports and named imports when they do occur. Not good Julia practice =#
@exportAll()
end
