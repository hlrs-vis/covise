/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _COLORS_H
#define _COLORS_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: Universal Colormap module                              ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Andreas Werner                           ++
// ++               Computer Center University of Stuttgart               ++
// ++                            Allmandring 30                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  03.08.2000  V3.0                                             ++
// ++**********************************************************************/

#include <api/coModule.h>
using namespace covise;

class Colors : public coModule
{
public:
    typedef float FlColor[5];

private:
    //////////  inherited mamber functions

    virtual int compute(const char *port);
    virtual void param(const char *portName, bool inMapLoading);

    ////////// static data members : pre-defined colormaps

    // values for Spike Removal Selection
    enum
    {
        SPIKE_NONE = 0,
        SPIKE_ADAPTIVE = 1,
        SPIKE_INTERVAL = 2,
        SPIKE_ELEMENTS = 3
    };

    TColormapChoice *colormaps;
    struct ColormapAttributes
    {
        bool isAbsolute;
        float min, max;
    };
    std::vector<ColormapAttributes> colormapAttributes;

    vector<float> d_cmap;
    int numColors, numColormaps;

    ////////// maximum size of colormap
    enum
    {
        MAX_CMAP_SIZE = 2048
    };

    ////////// ports
    coInputPort *p_data, *p_cmapIn, *p_alphaIn, *p_histoIn;
    coOutputPort *p_color, *p_texture, *p_cmapOut;

#ifdef NO_COLORMAP_PARAM
    coChoiceParam *p_colorNames;
#else
    coColormapParam *p_colorMap;
    coColormapChoiceParam *p_colorNames;
#endif

    //coChoiceParam       *p_selectMap;
    coFloatVectorParam *p_minmax;
    coIntScalarParam *p_numSteps;
    coStringParam *p_annotation;
    coBooleanParam *p_autoScale;
    coBooleanParam *p_scaleNow;
    coFloatParam *p_alpha;

    // ports for Spike Removal
    coChoiceParam *p_spikeAlgo;
    coFloatParam *p_spikeBot;
    coFloatParam *p_spikeTop;

    // Number  of components in Textures: 3 or 4
    int textureComponents;

    // color to be used for non-existing data values, nomally 0x00000000
    unsigned long d_noDataColor;

    // color output styles
    enum Outstyle
    {
        RGBA = 1,
        TEX = 4
    };

    // struct captures all info for one data input object
    struct recObj
    {
        const coDistributedObject *obj; // the object itself
        const coDistributedObject *const *objList; // if object was a set: list of subobj.
        recObj *subObj; //                      recurse to subobj
        int numElem; //                      number
        float *data; // otherwise: pointer to data
        bool doDelete; // whether we have to delete this data
        // (from vect or int)
        recObj()
        {
            obj = 0;
            objList = 0;
            subObj = 0;
            numElem = 0;
            data = 0;
            doDelete = false;
        }
        ~recObj()
        {
            if (subObj)
                delete[] subObj;
            if (doDelete)
                delete[] data;
        }
    };

    int openObj(recObj &base, const coDistributedObject *obj, const char *&species);

    void readMaps();

    void getMinMax(const recObj &base, float &min, float &max);

    FlColor *interpolateColormap(int numSteps, float alphaMult);

    coDistributedObject *createColors(recObj &base, recObj &alpha, FlColor *map,
                                      float min, float max, int numSteps,
                                      const char *name, int outStyle);

    void addColormapAttrib(const char *objName, float min, float max,
                           coDistributedObject *outObj, const char *annotation,
                           const FlColor *map, int numSteps);

    // initialize the different port setups
    void initCOLORS();
    void initCMAPTEX();
    void initOldStyle();

    // update min and max value convenience fct
    void updateMinMax(float min, float max);

    // new functions for Spike Removal

    // Number of bins used in corersponding algorithm
    int numBinsAdaptive, numBinsElements;

    /// correct get min/max values of all values within minV/maxV
    void getMinMax(const recObj &base, float &min, float &max,
                   float minV, float maxV);

    /// count all data values between min and max into bins
    void countBins(const recObj &base, float min, float max,
                   int numBins, int *bins);

    /// Spike removal algorithms
    void removeSpikesAdaptive(const recObj &base, float &min, float &max);
    void removeSpikesInterval(const recObj &base, float &min, float &max);
    void removeSpikesElements(const recObj &base, float &min, float &max);

public:
    Colors(int argc, char *argv[]);
};
#endif
