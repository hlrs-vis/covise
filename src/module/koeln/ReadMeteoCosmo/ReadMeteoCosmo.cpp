/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*! \file
    \brief  read star data of 1st physical institute
 
    \author Daniel Wickeroth <wickeroth@uni-koeln.de>
    \author (C) 2011, University of Cologne, Germany
 
    \date   10.08.2011
 */

#include <do/coDoSet.h>
#include <do/coDoData.h>
#include <api/coModule.h>
#include <climits>
#include <cfloat>
#include <cassert>
#include <map>
#include <string>
#include <ncFile.h>
#include <ncVar.h>
#include <ncDim.h>
using namespace netCDF;
#include <do/coDoUniformGrid.h>

using namespace covise;

class ReadAstro : public coModule
{
public:
    ReadAstro(int argc, char *argv[]);

private:
    // ports:
    coOutputPort *m_poGrid;
    coOutputPort *m_poQS;

    // parameters:
    coFileBrowserParam *m_paramFilename; ///< name of NetCDF file

    // methods
    virtual int compute(const char *port);
    bool open(const char *netcdfpath);
    bool read(int name, float time);
    bool close();

    // data
    NcFile *m_ncfile;
    NcVar m_ncdata;
    int m_currentIndex;
};

/// Constructor
ReadAstro::ReadAstro(int argc, char *argv[])
    : coModule(argc, argv, "Read Meteo Cosmo Data.")
    , m_ncfile(NULL)
    , m_currentIndex(0)
{

    // Create ports:
    m_poGrid = addOutputPort("grid", "UniformGrid", "Grid for volume data");
    m_poGrid->setInfo("Grid for volume Data");

    m_poQS = addOutputPort("QS", "Float", "Mass fraction of snow in the air");

    // Create parameters:
    m_paramFilename = addFileBrowserParam("Filename", "NetCDF file");
    m_paramFilename->setValue("data/", "*.nc;*.NC/*");
}

bool ReadAstro::close()
{

    delete m_ncfile;
    m_ncfile = NULL;
    return true;
}

bool ReadAstro::open(const char *path)
{
    assert(!m_ncfile);
    try {
        m_ncfile = new NcFile(path, NcFile::read);
    }
    catch (...)
    {
        close();
        sendError("failed to open NetCDF file %s", path);
        return false;
    }


    fprintf(stderr, "dims=%d, vars=%d, attrs=%d\n",
            m_ncfile->getDimCount(), m_ncfile->getVarCount(), m_ncfile->getAttCount());

    std::multimap<std::string, NcDim> allDims = m_ncfile->getDims();
    for(const auto &dim:allDims)
    {
        fprintf(stderr, "%s: %zd\n",
                dim.second.getName().c_str(),
                dim.second.getSize());
    }

    std::multimap<std::string, NcVar> allVars = m_ncfile->getVars();
    for (const auto& var : allVars)
    {
        fprintf(stderr, "%s: dims=%d atts=%d\n",
            var.second.getName().c_str(),
            var.second.getDimCount(),
            var.second.getAttCount());
        //int dims = m_ncfile->get_var(i)->num_dims();
        for (int j = 0; j < var.second.getDimCount(); ++j)
        {
            fprintf(stderr, "   %s: %zd\n",
                var.second.getDim(j).getName().c_str(),
                var.second.getDim(j).getSize());
        }
    }

    std::multimap<std::string, NcGroupAtt> allAtts = m_ncfile->getAtts();
    for (const auto& att : allAtts)
    {
        fprintf(stderr, "%s\n", att.second.getName().c_str());
    }

    return true;
}

bool ReadAstro::read(int name, float time)
{
    return false;
}

/// Compute routine: load checkpoint file
int ReadAstro::compute(const char *)
{
    const char *path = m_paramFilename->getValue();
    if (!open(path))
    {
        return STOP_PIPELINE;
    }

    //TODO: init variables from netcdf file info
    int x = 880, y = 50, z = 880;
    float xmin = 0, xmax = 880, ymin = 0, ymax = 50, zmin = 0, zmax = 880;

    coDoUniformGrid *gridData = new coDoUniformGrid(m_poGrid->getObjName(), x, y, z, xmin, xmax, ymin, ymax, zmin, zmax);

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(IO, ReadAstro)
