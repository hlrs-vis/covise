/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ReadTsunami_H
#define ReadTsunami_H

#include <api/coModule.h>
#ifdef OLD_NETCDFCXX
#include <netcdfcpp.h>
#else
#include <ncFile.h>
#include <ncVar.h>
#include <ncDim.h>
using namespace netCDF;
#endif

#define numParams 6

// -------------------------------------------------------------------
// class ReadTsunami
// -------------------------------------------------------------------
class ReadTsunami : public covise::coModule
{
public:
    /// default constructor
    ReadTsunami(int argc, char *argv[]);

    /// destructor
    virtual ~ReadTsunami();

    /// change of parameters (callback)
    virtual void param(const char *paramName, bool inMapLoading);

    /// compute callback
    virtual int compute(const char *);

private:
    /// open and check the netCFD File
    bool openNcFile();
    void readData(float *data, NcVar *var, int nDims, long *edges);

    // parameters
    covise::coFileBrowserParam *p_fileBrowser;
    covise::coChoiceParam *p_variables[numParams];
    covise::coChoiceParam *p_grid_choice_x;
    covise::coChoiceParam *p_grid_choice_y;
    covise::coChoiceParam *p_grid_choice_z;
    covise::coFloatParam *p_verticalScale;

    // ports
    covise::coOutputPort *p_grid_out;
    //covise::coOutputPort *p_data_outs[numParams];
    covise::coOutputPort* p_surface_out;
    covise::coOutputPort* p_seeSurface_out;
    covise::coOutputPort* p_maxHeight;
    covise::coOutputPort* p_waterSurface_out;

    // the netCDF File to be read from
    NcFile *ncDataFile;
};

// -------------------------------------------------------------------

#endif
