#= The protection of a variable, marked on its attributes (BDAECreate._maybeMarkAttrProtected;
   dropObservationOnlyVariables reads it): an enumeration's, a String's and a Clock's
   attributes became an empty Real's, their start, fixed, min and max lost, and a variable
   without attributes got a Real's whatever its type. =#
using Test
import OMBackend
import DAE
import MetaModelica

@testset "Protected variables keep their attributes" begin
  local mark = OMBackend.Backend.BDAECreate._maybeMarkAttrProtected
  local nil = MetaModelica.nil
  local NONE = MetaModelica.NONE
  local SOME = MetaModelica.SOME
  local start = SOME(DAE.ICONST(2))
  local enumAttrs = SOME(DAE.VAR_ATTR_ENUMERATION(NONE(), NONE(), NONE(), start, SOME(DAE.BCONST(true)),
                                                  NONE(), NONE(), NONE(), NONE()))
  local marked = mark(enumAttrs, DAE.PROTECTED(), DAE.T_ENUMERATION_DEFAULT).data
  @test marked isa DAE.VAR_ATTR_ENUMERATION
  @test marked.start === enumAttrs.data.start
  @test marked.isProtected.data === true
  #= Without attributes: the empty ones of the variable's type (of an array's elements). =#
  local intArray = DAE.T_ARRAY(DAE.T_INTEGER(nil), MetaModelica.list(DAE.DIM_INTEGER(3)))
  for (ty, attrType) in ((DAE.T_REAL(nil), DAE.VAR_ATTR_REAL), (DAE.T_INTEGER(nil), DAE.VAR_ATTR_INT),
                         (DAE.T_BOOL(nil), DAE.VAR_ATTR_BOOL), (DAE.T_STRING(nil), DAE.VAR_ATTR_STRING),
                         (DAE.T_ENUMERATION_DEFAULT, DAE.VAR_ATTR_ENUMERATION), (intArray, DAE.VAR_ATTR_INT))
    local attrs = mark(NONE(), DAE.PROTECTED(), ty).data
    @test attrs isa attrType
    @test attrs.isProtected.data === true
  end
  #= A public variable's attributes stay as they are. =#
  @test mark(enumAttrs, DAE.PUBLIC(), DAE.T_ENUMERATION_DEFAULT) === enumAttrs
  @test mark(NONE(), DAE.PUBLIC(), DAE.T_REAL(nil)) === NONE()
end
