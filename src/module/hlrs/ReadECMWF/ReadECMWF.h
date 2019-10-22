/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef READECMWF_H
#define READECMWF_H

#include <api/coModule.h>
#include <netcdfcpp.h>

#define numParams 6

// -------------------------------------------------------------------
// class ReadECMWF
// -------------------------------------------------------------------
class ReadECMWF : public covise::coModule
{
public:
    /// default constructor
    ReadECMWF(int argc, char *argv[]);

    /// destructor
    virtual ~ReadECMWF();

    /// change of parameters (callback)
    virtual void param(const char *paramName, bool inMapLoading);

    /// compute callback
    virtual int compute(const char *);
    enum CoordType {
        PRESSURE = 0,
        DEPTH = 1
    };

private:
    /// open and check the netCFD File
    bool openNcFile();
    void readData(float *data, NcVar *var, int nDims, long *edges);

    // parameters
    covise::coFileBrowserParam *p_fileBrowser;
    covise::coChoiceParam *p_variables[numParams];
    covise::coChoiceParam *p_coord_type;
    covise::coChoiceParam *p_grid_lat;
    covise::coChoiceParam *p_grid_lon;
    covise::coChoiceParam *p_grid_pressure_level;
    covise::coChoiceParam *p_grid_depth;
    covise::coFloatParam *p_verticalScale;
  //  covise::coChoiceParam *p_date_choice;
    covise::coIntScalarParam *p_numTimesteps;
    // ports
    covise::coOutputPort *p_grid_out;
    covise::coOutputPort *p_unigrid_out;
    covise::coOutputPort *p_data_outs[numParams];
  //  covise::coOutputPort *p_surface_out;

    float pressureAltitude(float p);
    // the ECMWF File to be read from
    NcFile *ncDataFile;

    int has_timesteps;
};

// -------------------------------------------------------------------

#endif
