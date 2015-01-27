/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PLMXMLRECHENRAUM_H
#define PLMXMLRECHENRAUM_H
#include <ctime>
#include <api/coModule.h>
using namespace covise;
#include "include/model.h"
#include "include/IOutil.h"

#define NONCOMPLIENTPARSING 0

//#define USE_PLMXML
class PLMXMLRechenraum : public coModule
{

private:
//	enum { MAX_CUBES=64 };

#define GEO_SEC "Geometry_Rack"
#define FLOW_SEC "Flowrate_Rack"

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
    coOutputPort *intypes;
    coOutputPort *bccheck;

    coOutputPort *feedback_info;

    coFileBrowserParam *startFile;

    int isInitialized;

    int fileChanged;
    struct stat fileSt;
    time_t fileTimestamp;

    struct geometry *geo;

    // Button for triggering Grid Generation
    coBooleanParam *p_makeGrid;
    coBooleanParam *p_lockmakeGrid;
    coBooleanParam *p_createGeoRbFile;
    coFileBrowserParam *p_PLMXMLfile;
    coFloatParam *p_gridSpacing;
    coIntScalarParam *p_nobjects;
    coFloatVectorParam *p_model_size;
    coFloatParam *p_Q_total;
    coStringParam *p_BCFile;
    coStringParam *p_geofile;
    coStringParam *p_rbfile;
    coFloatVectorParam *p_cubes_size[MAX_CUBES];
    coFloatVectorParam *p_cubes_pos[MAX_CUBES];
    coFloatParam *p_flowrate[MAX_CUBES];

    coFloatParam *p_power[MAX_CUBES];
    coStringParam *p_preFile;
    coStringParam *p_preFileNew;

    coStringParam *p_direction[MAX_CUBES];

    struct rech_model *model;
    struct rechgrid *rg;
    struct covise_info *ci;

    // some Menues ...
    coChoiceParam *m_Geometry;
    char **geo_labels;

    coChoiceParam *m_FlowRate;
    char **flow_labels;

    char *IndexedParameterName(const char *name, int index);
    void createFeedbackObjects();
    int GetParamsFromControlPanel(struct rech_model *model);
    int getValues();

    void createPreFile();
    string addPower(string string);

    BC_TYPE readDirection(const char *str);

public:
    PLMXMLRechenraum(int argc, char *argv[]);
};

#endif
