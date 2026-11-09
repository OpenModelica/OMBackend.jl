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

#= What OMFrontend.jl's flat model gives the backend: functions OMBackend declares and the extension
   OMBackendOMFrontendExt (loaded with OMFrontend) defines on the frontend's FlatModel and its function list. The core
   takes OMJLIR (`translate(::OMJLIR.Model)`); without OMFrontend these have no methods. =#

"""
    lowerFlatModel(fm; functions = MetaModelica.nil) -> OMJLIR.Model

The OMJLIR model of OMFrontend's flat model `fm` and its functions (the frontend's function list), before the passes
(`OMBackendOMFrontendExt.LowerFlatModel`).
"""
function lowerFlatModel end

"""
    lowerFunctions(functions) -> Vector{OMJLIR.ModelicaFunction}

The functions of OMFrontend's function list in OMJLIR, the derivatives synthesized for inlined functions after them
(`LowerUtil.withInlineDerivatives`).
"""
function lowerFunctions end

"""
    explainIndexReduction(fm; functionList = nothing) -> Passes.IndexReduction.Explanation

What OMBackend's index reduction does to the flat model `fm` and why
(`Passes.IndexReduction.explain`): the equations it differentiates and how often, the candidates
of each choice with their StateSelect and the dummy derivatives among them, the states, the
dynamic state sets; or why the model needs no reduction or is left to MTK. Planned as a
translation would plan it, nothing translated.
"""
function explainIndexReduction end

"""
    omjlirModel(fm; functionList = nothing, scalarized = true, arrayReduction = true) -> OMJLIR.Model

The OMJLIR model of the flat model `fm` and its functions (the frontend's function list),
through the passes (Passes.PIPELINE): what the simulation code is lowered from. A flat model
with arrays kept (`scalarized = false`, OM's `scalarize = false`) whose index OMBackend reduces in
array form is lowered with its arrays (`arrayReducedModel`; not with `arrayReduction` false: a caller
that tried it); another is scalarized first: the backend takes scalars.
"""
function omjlirModel end

"""
    arrayReducedModel(fm; functionList = nothing) -> Union{Nothing, OMJLIR.Model}

The OMJLIR model of a flat model with arrays kept whose index OMBackend reduces in array form (the
switch OMBACKEND_ARRAY_INDEX_REDUCTION, with OMBACKEND_INDEX_REDUCTION), for the scalar simulation
code: reduced with its arrays (`reducedArrayModel`), then scalarized (`scalarizedReducedModel`).
`nothing` where the array form does not take the model: the caller scalarizes in the frontend as
before.
"""
function arrayReducedModel end

"""
    reducedArrayModel(fm; functionList = nothing) -> Union{Nothing, OMJLIR.Model}

A flat model with arrays kept lowered with its arrays, through the passes (`runPassesKeepingArrays!`:
its array equations as for-equations), its index reduced in array form where above one, its states
committed in array form (`Passes.IndexReduction.reduceIndexArrays!`; a model without arrays to keep of
index 1 the same, `commitIndexOne!`, for the array code only): what the array code
(`generateReducedArrayODETargetCode`) or the scalarized form (`scalarizedReducedModel`) is made from.
`nothing` where the array form does not take the model (switched off, no arrays to keep and of index
above one, of index 1 with clocked partitions, of index above one with an if-expression that reads a
for-equation's iterator, what it does not handle or the lowering does not support: logged; another
error: a warning), the reason recorded (`declineArrayPath!`).
"""
function reducedArrayModel end

"""
    translateReducedArrays(fm; functionList, BackendMode, warnMissingStartValues, eliminateNonDynamic,
                           observedFilter, checkSimCode) -> Union{Nothing, Tuple}

`translate` of a flat model with arrays kept whose index OMBackend reduces in array form
(`arrayReducedModel`); `nothing` where it does not (OM's translate then flattens with
scalarization).
"""
function translateReducedArrays end

"""
    resolveModelicaURI(uri::String) -> String

The file of a `modelica://` URI (an external function's IncludeDirectory), where the frontend loaded the library from
(OMFrontend.resolveModelicaURI). A model of another frontend has none (`applicable` is false).
"""
function resolveModelicaURI end
