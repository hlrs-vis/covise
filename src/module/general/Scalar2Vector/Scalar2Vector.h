/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SCALAR2VECTOR_H
#define _SCALAR2VECTOR_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2000 Vircinity  ++
// ++ Description: Unify three scalar 3D-data-object to one vectorial     ++
// ++  3D-data-object                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                           Christof Schwenzer                        ++
// ++                        Vircinity GmbH Stuttgart                     ++
// ++                            Nobelstrasze 15                          ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  12.09.2000  V1.0                                             ++
// ++**********************************************************************/

#include <api/coSimpleModule.h>
using namespace covise;
#include <do/coDoData.h>

class Scalar2Vector : public coSimpleModule
{

private:
    //////////  member functions

    /// this module has only the compute call-back
    virtual int compute(const char *port);

    ////////// parameters

    ////////// ports
    coInputPort *p_inPortU;
    coInputPort *p_inPortV;
    coInputPort *p_inPortW;
    coInputPort *p_inPortA;
    coOutputPort *p_outPort;
    coOutputPort *p_outPortPacked;
    coChoiceParam *p_paramNormalizeChoice;
    coFloatParam *p_paramNormalizeIgnore;
    coFloatVectorParam *p_paramNormalizeMin;
    coFloatVectorParam *p_paramNormalizeMax;

    ///////// Additional data structures for
    //Unifying attributes
    int summarized;
    const char **summarizedNames;
    const char **summarizedValues;

    void summarizedValuesPushBack(char *value);
    void summarizedNamesPushBack(char *name);
    void removeNewLine(char *string);
    void summarizeAttributes(
        int xSize,
        const char **xNames,
        const char **xValues,
        int ySize,
        const char **yNames,
        const char **yValues,
        int zSize,
        const char **zNames,
        const char **zValues);
    void summarizeAttributes(
        const coDistributedObject *objU,
        const coDistributedObject *objV,
        const coDistributedObject *objW);
    int matchAttributeNames(
        const char **xName,
        const char **yName,
        const char **zName,
        char *prefixOrSuffix);
    void summarizeValue(
        const char **xValue,
        const char **yValue,
        const char **zValue);

    void summarizeEntry(
        const char **xName,
        const char **xValue,
        const char **yName,
        const char **yValue,
        const char **zName,
        const char **zValue,
        int manner,
        char *prefixOrSuffix);
    void checkInPort(
        const coDistributedObject *obj,
        coInputPort *p_port,
        int &retVal,
        char *errMsg);
    int checkInPorts(
        const coDistributedObject *objU,
        const coDistributedObject *objV,
        const coDistributedObject *objW,
        char *errMsg);
    coDistributedObject *computeStructured(
        const coDoFloat *uData,
        const coDoFloat *vData,
        const coDoFloat *wData);
    coDistributedObject *computeStructuredPacked(
        const coDoFloat *uData,
        const coDoFloat *vData,
        const coDoFloat *wData,
        const coDoFloat *aData);
    coDistributedObject *computeUnstructured(
        const coDoFloat *uData,
        const coDoFloat *vData,
        const coDoFloat *wData);

public:
    Scalar2Vector(int argc, char *argv[]);
};
#endif
