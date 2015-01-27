/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ISOS_H
#define _ISOS_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description:  COVISE Isosurface application module                     **
 **                                                                        **
 **                                                                        **
 **                             (C) 1995                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Uwe Woessner                                                  **
 **                                                                        **
 **                                                                        **
 ** Date:  23.02.95  V1.0                                                  **
\**************************************************************************/

#include <api/coSimpleModule.h>
using namespace covise;
#include <util/coviseCompat.h>
#include <float.h>

#include <do/coDoData.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoUnstructuredGrid.h>

#include "IsoPoint.h"
#ifdef _COMPLEX_MODULE_
#include <alg/coColors.h>
#endif

class IsoSurface : public coSimpleModule
{

private:
    float isovalueHack[3];
    float &isovalue;

    float startp[3];

    // Input ports
    coInputPort *p_GridIn;
    coInputPort *p_IsoDataIn;
    coInputPort *p_DataIn, *p_IBlankIn;
    // Output ports
    coOutputPort *p_GridOut;
    coOutputPort *p_NormalsOut;
    coOutputPort *p_DataOut;

    const short int shiftOut;
    void addFeedbackParams(coDistributedObject *obj);
#ifdef _COMPLEX_MODULE_
    coInputPort *p_ColorMapIn;
    coOutputPort *p_GeometryOut;
    coBooleanParam *p_color_or_texture;
    void StaticParts(coDistributedObject **geopart,
                     coDistributedObject **normpart,
                     coDistributedObject **colorpart,
                     const coDistributedObject *geo,
                     const coDistributedObject *data,
                     string geometryOutName,
                     bool ColorMapAttrib = true,
                     const ScalarContainer *SCont = NULL);
    coFloatSliderParam *p_scale;
    coChoiceParam *p_length;
    coIntScalarParam *p_num_sectors;
    coChoiceParam *p_vector;
#endif
    virtual void copyAttributesToOutObj(coInputPort **input_ports,
                                        coOutputPort **output_ports, int i);

    // Parameters
    coBooleanParam *p_gennormals;
    coBooleanParam *p_genstrips;
    enum
    {
        POINT = 0,
        VALUE = 1
    };
    coChoiceParam *p_pointOrValue;
    coFloatVectorParam *p_isopoint;
    coFloatSliderParam *p_isovalue;
    coBooleanParam *p_autominmax_;

    // previous scalar object name
    void UpdateIsoValue();
    string _scalarName;
    bool _autominmax;
    float _min, _max;

    // ratio for vertex allocation
    float vertexRatio;

    // Flag set when in selfExec loop to prevent errorMessage
    bool inSelfExec;

    // automatically create module title? if != NULL, mask for titles
    bool autoTitle;

    // config variable for autoTitle set?
    bool autoTitleConfigured;

    // use polyhedra support or not
    bool Polyhedra;

protected:
    myPair find_isovalueU(const coDoUniformGrid *, const coDoFloat *);
    myPair find_isovalueR(const coDoRectilinearGrid *, const coDoFloat *);
    myPair find_isovalueS(const coDoStructuredGrid *, const coDoFloat *);
    myPair find_isovalueUU(const coDoUnstructuredGrid *, const coDoFloat *);
    myPair find_isovalueUS(const coDoUnstructuredGrid *, const coDoFloat *);
    void fillThePoint()
    {
        startp[0] = p_isopoint->getValue(0);
        startp[1] = p_isopoint->getValue(1);
        startp[2] = p_isopoint->getValue(2);
    }
    myPair find_isovalueT(const coDistributedObject *inObj,
                          const coDistributedObject *idata);
    myPair find_isovalueSG(const coDistributedObject *inObj,
                           const coDistributedObject *idata);
    myPair find_isovalueSGoT(const coDistributedObject *inObj,
                             const coDistributedObject *idata);
    myList objLabVal;
    void setUpIsoList(coInputPort **inPorts);
    int lookUp;
    int level;
    float getValue(int lookup)
    {
        float ret;
        if (p_pointOrValue->getValue() == POINT)
        {
            ret = objLabVal.getValue(lookup);
        }
        else
        {
            ret = p_isovalue->getValue();
        }
        return ret;
    }

    virtual void preHandleObjects(coInputPort **);
    virtual void setIterator(coInputPort **, int);

    virtual void postHandleObjects(coOutputPort **);

    // which feedback to send, maybe both?
    enum FeedbackStyle
    {
        FEED_NONE,
        FEED_OLD,
        FEED_NEW,
        FEED_BOTH
    };
    FeedbackStyle fbStyle_;

public:
    IsoSurface(int argc, char *argv[]);

    virtual int compute(const char *port);
    virtual void param(const char *paramName, bool inMapLoading);

    // void postInst(){setCopyNonSetAttributes(0);}

    virtual ~IsoSurface()
    {
    }
};
#endif // _ISOS_H
