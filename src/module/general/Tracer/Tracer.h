/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  MODULE Tracer
//
//  Universal tracer
//
//  Initial version: 2001-12-07 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2001 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _TRACER_H_
#define _TRACER_H_

#include <api/coModule.h>
using namespace covise;
#include "HTask.h"
#include "Fifo.h"

#include "BBoxAdmin.h"

#ifdef _COMPLEX_MODULE_
#include <api/coFeedback.h>
#endif

namespace covise
{
class coDoPoints;
class coDoGeometry;
}

// Do not use Pthreads on hp1020
// #define CO_hp1020

class Tracer : public coFunctionModule
{
    COMODULE

public:
    HTask::time_direction td_;
    Tracer(int argc, char **argv);
    virtual ~Tracer()
    {
#ifndef CO_hp1020
        delete[] read_task_;
#endif
    }
    enum HTaskTyp
    {
        STREAMLINES = 0,
        MOVING_POINTS = 1,
        GROWING_LINES = 2,
        STREAKLINES = 3
    };
#ifdef _COMPLEX_MODULE_
    enum StartTyp
    {
        LINE = 0,
        SQUARE = 1,
        FREE = 2
    };
#else
    enum StartTyp
    {
        LINE = 0,
        SQUARE = 1,
        CYLINDER = 2
    };
#endif

protected:
private:
    // Ports
    coInputPort *p_grid;
    coInputPort *p_velo;
    coInputPort *p_ini_points;
    coInputPort *p_octtrees;
    coInputPort *p_field;
#ifdef _COMPLEX_MODULE_
    coInputPort *p_ColorMapIn;
    coInputPort *p_SampleGeom_, *p_SampleData_;
    coOutputPort *p_GeometryOut;
    coFloatParam *p_radius;
    coStringParam *p_free_start_points_;
    coFloatVectorParam *p_minmax;
    coBooleanParam *p_autoScale;
    void ComplexObject();
    void addFeedbackParams(coFeedback &feedback, const char *&oldstyleAttrib);
    coDoGeometry *SampleToGeometry(const coDistributedObject *grid, const coDistributedObject *data);
#endif
    coOutputPort *p_line;
    coOutputPort *p_mag;
    coOutputPort *p_start;
    // Parameters
    coIntSliderParam *p_no_startp;
    coFloatVectorParam *p_startpoint1;
    coFloatVectorParam *p_startpoint2;
    coFloatVectorParam *p_direction;
    coFloatVectorParam *p_cyl_axis;
    coFloatParam *p_cyl_radius;
    coFloatParam *p_cyl_height;
    coFloatVectorParam *p_cyl_axispoint;
    coFloatVectorParam *p_verschiebung_;
    coChoiceParam *p_tdirection;
    coChoiceParam *p_whatout;
    coChoiceParam *p_taskType;
    coChoiceParam *p_startStyle;
    coFloatParam *p_trace_eps;
    coFloatParam *p_trace_abs;
    coFloatParam *p_grid_tol;
    coFloatParam *p_trace_len;
    coFloatParam *p_min_vel;
    coFloatParam *p_stepDuration;
    coFloatParam *p_tube_width;
    coIntScalarParam *p_MaxPoints;
    coIntScalarParam *p_cycles;
    coIntScalarParam *p_trailLength;
    coBooleanParam *p_control;
    coBooleanParam *p_newParticles;
    coBooleanParam *p_randomOffset;
    coBooleanParam *p_randomStartpoints;
    coFloatParam *p_timeNewParticles;
    coFloatParam *p_divide_cell;
    coFloatParam *p_max_out_of_cell;
    coIntScalarParam *p_no_threads_w;
    coIntScalarParam *p_search_level_polygons_;
    coIntScalarParam *p_skip_initial_steps_;
    coStringParam *p_color;
    ///////////////////////////////////////
    int crewSize_;
    int findCrewSize();
#ifndef CO_hp1020
    PTask **read_task_; // list of pointers to PTask objects.
    // used for task assignation
    Fifo<int> lazyThreads_;

    //  virtual void quit();
    void terminateThreads();
    void startThreads();
    void lockMMutex();
#endif
    BBoxAdmin BBoxAdmin_;
    bool GoodOctTrees();
    bool GoodOctTrees(const coDistributedObject *grid, const coDistributedObject *otree);
    HTask *createHTask();
    virtual int compute(const char *port);
    virtual void postInst();
    virtual void param(const char *paramname, bool inMapLoading);
    void fillLine(float **x_ini, float **y_ini, float **z_ini);
    void fillSquare(float **x_ini, float **y_ini, float **z_ini);
    void fillCylinder(float **x_ini, float **y_ini, float **z_ini);
    void fillWhatOut();
    int computeGlobals();
    // extract point information from object obj
    coDoPoints *extractPoints(const coDistributedObject *obj);
    float *xValues, *yValues, *zValues;
    int numExtractedStartpoints;
    // recurse into set and add startpoints
    void addPoints(const coDistributedObject *obj);

    ///////////////////////////////////////
    // Used to simplify diagnostics: if we get
    // the same input as in the previous computation,
    // then we need not repeat it.
    ///////////////////////////////////////
    std::string gridName_;
    std::string veloName_;
    std::string iniPName_;
    std::string octTreeName_;
    std::string fieldName_;

    // automatically create module title? if != NULL, mask for titles
    bool autoTitle;

    // config variable for autoTitle set?
    bool autoTitleConfigured;

    std::string complexObjectType;

    // which feedback to send, maybe both?
    enum FeedbackStyle
    {
        FEED_NONE,
        FEED_OLD,
        FEED_NEW,
        FEED_BOTH
    };
    FeedbackStyle fbStyle_;

    // attach old-style attributes
    void AddInteractionAttributes();
};

class WristWatch
{
private:
    timeval myClock;

public:
    WristWatch();
    ~WristWatch();

    void start();
    void stop(char *s);
};
#endif
