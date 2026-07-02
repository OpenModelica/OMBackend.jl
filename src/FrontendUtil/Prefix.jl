#= /*
* This file is part of OpenModelica.
*
* Copyright (c) 1998-2026, Open Source Modelica Consortium (OSMC),
* c/o Linköpings universitet, Department of Computer and Information Science,
* SE-58183 Linköping, Sweden.
*
* All rights reserved.
*
* THIS PROGRAM IS PROVIDED UNDER THE TERMS OF AGPL VERSION 3 LICENSE OR
* THIS OSMC PUBLIC LICENSE (OSMC-PL) VERSION 1.8.
* ANY USE, REPRODUCTION OR DISTRIBUTION OF THIS PROGRAM CONSTITUTES
* RECIPIENT'S ACCEPTANCE OF THE OSMC PUBLIC LICENSE OR THE GNU AGPL
* VERSION 3, ACCORDING TO RECIPIENTS CHOICE.
*
* The OpenModelica software and the OSMC (Open Source Modelica Consortium)
* Public License (OSMC-PL) are obtained from OSMC, either from the above
* address, from the URLs:
* http://www.openmodelica.org or
* https://github.com/OpenModelica/ or
* http://www.ida.liu.se/projects/OpenModelica,
* and in the OpenModelica distribution.
*
* GNU AGPL version 3 is obtained from:
* https://www.gnu.org/licenses/licenses.html#GPL
*
* This program is distributed WITHOUT ANY WARRANTY; without
* even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE, EXCEPT AS EXPRESSLY SET FORTH
* IN THE BY RECIPIENT SELECTED SUBSIDIARY LICENSE CONDITIONS OF OSMC-PL.
*
* See the full OSMC Public License conditions for more details.
*
*/ =#

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
