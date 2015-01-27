/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <api/coModule.h>
using namespace covise;
#include "include/booth.h"

class sc2004booth : public coModule
{

private:
    enum
    {
        MAX_CUBES = 20
    };

#define GEO_SEC "GeometrySection"

    // Gate.cpp
    virtual int compute(const char *port);
    virtual void param(const char *, bool inMapLoading);
    virtual void postInst();
    virtual void quit();
    virtual void CreateUserMenu(void);

    coOutputPort *grid;
    coOutputPort *surface;
    coOutputPort *bcin;
    coOutputPort *bcout;
    coOutputPort *bcwall;
    coOutputPort *boco;
    coOutputPort *inpoints;
    coOutputPort *airpoints;
    coOutputPort *venpoints;

    coOutputPort *feedback_info;

    coFileBrowserParam *startFile;

    int isInitialized;

    struct geometry *geo;

    // Button for triggering Grid Generation
    coBooleanParam *p_makeGrid;
    coBooleanParam *p_lockmakeGrid;
    coBooleanParam *p_createGeoRbFile;
    coBooleanParam *p_openDoor;
    coFloatParam *p_gridSpacing;
    coIntScalarParam *p_nobjects;
    coFloatVectorParam *p_booth_size;
    coFloatSliderParam *p_v_in;
    coFloatSliderParam *p_v_aircond_front;
    coFloatSliderParam *p_v_aircond_middle;
    coFloatSliderParam *p_v_aircond_back;
    coFloatSliderParam *p_v_ven;
    coStringParam *p_geofile;
    coStringParam *p_rbfile;
    coFloatVectorParam *p_cubes_size[MAX_CUBES];
    coFloatVectorParam *p_cubes_pos[MAX_CUBES];

    struct sc_booth *booth;
    struct sc2004grid *sg;
    struct covise_info *ci;

    // some Menues ...
    coChoiceParam *m_Geometry;
    char **geo_labels;

    char *IndexedParameterName(const char *name, int index);
    void createFeedbackObjects();
    int GetParamsFromControlPanel(struct sc_booth *booth);
    int setGeoParamsStandardForSC2004();

public:
    sc2004booth(int argc, char *argv[]);
};
