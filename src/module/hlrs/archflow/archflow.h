/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <api/coModule.h>
using namespace covise;
#include "include/model.h"

class archflow : public coModule
{

private:
    enum
    {
        MAX_CUBES = 20
    };

#define GEO_SEC "GeometrySection"

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
    coFloatVectorParam *p_model_size;
    coFloatVectorParam *p_v_in;
    coFloatParam *p_zScale;
    coStringParam *p_geofile;
    coStringParam *p_rbfile;
    coFloatVectorParam *p_cubes_size[MAX_CUBES];
    coFloatVectorParam *p_cubes_pos[MAX_CUBES];

    struct arch_model *model;
    struct archgrid *ag;
    struct covise_info *ci;

    // some Menues ...
    coChoiceParam *m_Geometry;
    char **geo_labels;

    char *IndexedParameterName(const char *name, int index);
    void createFeedbackObjects();
    int GetParamsFromControlPanel(struct arch_model *model);
    int setGeoParamsStandardForArchFlow();

public:
    archflow(int argc, char *argv[]);
};
