/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ReadTsunami.h"

#include <iostream>
#include <api/coFeedback.h>
#include "do/coDoStructuredGrid.h"
#include "do/coDoData.h"
#include "do/coDoSet.h"

using namespace covise;

const char *NoneChoices[] = { "none" };
std::vector<std::string> varNames;
char *VarDisplayList[100];
char *AxisChoices[100];

// -----------------------------------------------------------------------------
// constructor
// -----------------------------------------------------------------------------
ReadTsunami::ReadTsunami(int argc, char *argv[])
    : coModule(argc, argv, "NetCDF Reader")
{
    ncDataFile = NULL;

    // define parameters

    // File browser
    p_fileBrowser = addFileBrowserParam("NC_file", "NC File");
    p_fileBrowser->setValue("/data/hpcsdela/harmonie/fc-0000.nc",
                            "*.nc");

    // Variables to visualise
    p_verticalScale = addFloatParam("VerticalScale", "VerticalScale");
    p_verticalScale->setValue(1.0);
    p_step = addInt32Param("StepWidth","time step step with");
    p_step->setValue(1);

    // define ports

    // 2D Surface
    p_surface_out = addOutputPort("surfaceOut", "Polygons", "2D Grid output");
    p_seaSurface_out
        = addOutputPort("seaSurfaceOut", "Polygons", "2D See floor");
    p_waterSurface_out
        = addOutputPort("waterSurfaceOut", "Polygons", "2D water surface");

    p_maxHeight = addOutputPort("maxHeight", "Float", "Maxx water height");
}

// -----------------------------------------------------------------------------
// destructor
// -----------------------------------------------------------------------------
ReadTsunami::~ReadTsunami()
{
    if (ncDataFile)
    {
        delete ncDataFile;
    }
}

// -----------------------------------------------------------------------------
// change of parameters callback
// -----------------------------------------------------------------------------
void ReadTsunami::param(const char *paramName, bool inMapLoading)
{
}

// -----------------------------------------------------------------------------
// compute callback
// -----------------------------------------------------------------------------
int ReadTsunami::compute(const char *)
{
    sendInfo("compute call");
    if (openNcFile())
    {
        // get variable name from choice parameters
        NcVar var;
        int numdims = 0;

        NcVar latvar = ncDataFile->getVar("lat");
        NcVar lonvar = ncDataFile->getVar("lon");
        NcVar grid_latvar = ncDataFile->getVar("grid_lat");
        NcVar grid_lonvar = ncDataFile->getVar("grid_lon");
        NcVar bathymetryvar = ncDataFile->getVar("bathymetry");
        NcVar max_height = ncDataFile->getVar("max_height");
        NcVar eta = ncDataFile->getVar("eta");

        int snx = latvar.getDim(0).getSize();
        int sny = lonvar.getDim(0).getSize();
		int nz = 0;
        float* latVals = new float[snx];
        float* lonVals = new float[sny];
        latvar.getVar(latVals);
        lonvar.getVar(lonVals);

        // Now for the 2D variables, we create a surface
        int snumPolygons = (snx - 1) * (sny - 1);
        coDoPolygons* outSurface = new coDoPolygons(p_surface_out->getObjName(),
            snx * sny, snumPolygons * 4, snumPolygons);
        int* svl, * spl;
        float *sx_coord, *sy_coord, *sz_coord;
        outSurface->getAddresses(&sx_coord, &sy_coord, &sz_coord, &svl, &spl);
        // Fill the _coord arrays (memcpy faster?)
        // FIXME: Should depend on var?->num_dims(). (fix with pointers?)
        int n = 0;
        n = 0;
		for (int j = 0; j < snx; j++)
			for (int k = 0; k < sny; k++, n++)
			{
				sx_coord[n] = latVals[j];
				sy_coord[n] = lonVals[k];
				sz_coord[n] = 0; //zVals[0];
			}
        // Fill the vertex list
        n = 0;
        for (int j = 1; j < snx; j++)
            for (int k = 1; k < sny; k++)
            {
                svl[n++] = (j - 1) * sny + (k - 1);
                svl[n++] = j * sny + (k - 1);
                svl[n++] = j * sny + k;
                svl[n++] = (j - 1) * sny + k;
            }
        // Fill the polygon list
        for (int p = 0; p < snumPolygons; p++)
            spl[p] = p * 4;

        // Delete buffers from grid replication
        delete[] latVals;
        delete[] lonVals;

        int nx = grid_latvar.getDim(0).getSize();
        int ny = grid_lonvar.getDim(0).getSize();
        latVals = new float[nx];
        lonVals = new float[ny];
        float *depthVals = new float[nx * ny];
        grid_latvar.getVar(latVals);
        grid_lonvar.getVar(lonVals);
        bathymetryvar.getVar(depthVals);
        int* vl, * pl;
        float* x_coord, * y_coord, * z_coord;

        // Now for the 2D variables, we create a surface
        int numPolygons = (nx - 1) * (ny - 1);
        coDoPolygons* outSeeSurface = new coDoPolygons(p_seaSurface_out->getObjName(),
            nx * ny, numPolygons * 4, numPolygons);
        outSeeSurface->getAddresses(&x_coord, &y_coord, &z_coord, &vl, &pl);
        // Fill the _coord arrays (memcpy faster?)
        // FIXME: Should depend on var?->num_dims(). (fix with pointers?)
        n = 0;
        for (int j = 0; j < nx; j++)
            for (int k = 0; k < ny; k++, n++)
            {
                x_coord[n] = latVals[j];
                y_coord[n] = lonVals[k];
                z_coord[n] = -depthVals[j*ny+k];
            }
        // Fill the vertex list
        n = 0;
        for (int j = 1; j < nx; j++)
            for (int k = 1; k < ny; k++)
            {
                vl[n++] = (j - 1) * ny + (k - 1);
                vl[n++] = j * ny + (k - 1);
                vl[n++] = j * ny + k;
                vl[n++] = (j - 1) * ny + k;
            }
        // Fill the polygon list
        for (int p = 0; p < numPolygons; p++)
            pl[p] = p * 4;

        // Delete buffers from grid replication
        delete[] latVals;
        delete[] lonVals;
        delete[] depthVals;

        
        float* floatData;
        coDoFloat *mh = new coDoFloat(p_maxHeight->getObjName(), max_height.getDim(0).getSize()* max_height.getDim(1).getSize());
        mh->getAddress(&floatData);
        max_height.getVar(floatData);

        floatData = new float[eta.getDim(0).getSize()* eta.getDim(1).getSize() *eta.getDim(2).getSize()];
        int numTimesteps = eta.getDim(0).getSize();
        eta.getVar(floatData);

        coDistributedObject** objs = new coDistributedObject * [numTimesteps+1];
        std::string baseName = p_waterSurface_out->getObjName();
        objs[numTimesteps] = nullptr;
        sendInfo("numTimesteps %d!",numTimesteps);
        int i=0;
        float zScale = p_verticalScale->getValue();
        for (int t = 0; t < numTimesteps; t+=p_step->getValue())
        {
			// Now for the 2D variables, we create a surface
			int snumPolygons = (snx - 1) * (sny - 1);
			coDoPolygons* outSurface = new coDoPolygons(baseName + std::to_string(i),
				snx * sny, snumPolygons * 4, snumPolygons);
			int* vl, * pl;
			float* x_coord, * y_coord, * z_coord;
			outSurface->getAddresses(&x_coord, &y_coord, &z_coord, &vl, &pl);
            int n = 0;
            for (int j = 0; j < snx; j++)
            {
                for (int k = 0; k < sny; k++, n++)
                {
                    n++;
                    x_coord[n] = sx_coord[n];
                    y_coord[n] = sy_coord[n];
                    if (floatData[t * snx * sny + n] <= -depthVals[j * ny + k])
                    {
						z_coord[n] = -depthVals[j * ny + k] - 5;
                    }
                    else
                    {
                        z_coord[n] = floatData[t * snx * sny + n] * zScale;
                    }
                }
            }
            for (int j = 0; j < snumPolygons * 4; j++)
            {
                vl[j]=svl[j];
            }
			// Fill the polygon list
			for (int p = 0; p < snumPolygons; p++)
				pl[p] = spl[p];

            objs[i] = outSurface;
            objs[i + 1] = nullptr;
            i++;
        }

        coDoSet* set = new coDoSet(baseName, objs);
        set->addAttribute("TIMESTEP", "-1 -1");

    }

    return CONTINUE_PIPELINE;
}

// -----------------------------------------------------------------------------
// open the nc file
// -----------------------------------------------------------------------------
bool ReadTsunami::openNcFile()
{
    string sFileName = p_fileBrowser->getValue();

    if (sFileName.empty())
    {
        sendInfo("NetCDF filename is empty!");
        return false;
    }
    else
    {
        try {
            ncDataFile = new NcFile(sFileName.c_str(), NcFile::read);
        }
        catch(...)
        {
            sendInfo("Couldn't open NetCDF file!");
            return false;
        }

        if (ncDataFile->getVarCount()==0)
        {
            sendInfo("empty NetCDF file!");
            return false;
        }
        else
        {
            return true;
        }
    }
}

// -----------------------------------------------------------------------------
// main: create and start the module
// -----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    covise::coModule *application = new ReadTsunami(argc, argv);
    application->start(argc, argv);

    return 0;
}

// -----------------------------------------------------------------------------
