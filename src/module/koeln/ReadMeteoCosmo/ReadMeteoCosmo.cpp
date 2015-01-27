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
#include <netcdfcpp.h>
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
    NcVar *m_ncdata;
    int m_currentIndex;
};

/// Constructor
ReadAstro::ReadAstro(int argc, char *argv[])
    : coModule(argc, argv, "Read Meteo Cosmo Data.")
    , m_ncfile(NULL)
    , m_ncdata(NULL)
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
    //delete m_ncdata;
    m_ncdata = NULL;

    delete m_ncfile;
    m_ncfile = NULL;
    return true;
}

bool ReadAstro::open(const char *path)
{
    assert(!m_ncfile);
    m_ncfile = new NcFile(path, NcFile::ReadOnly);
    if (!m_ncfile->is_valid())
    {
        close();
        sendError("failed to open NetCDF file %s", path);
        return false;
    }

    if (m_ncfile->get_format() == NcFile::BadFormat)
    {
        close();
        sendError("bad NetCDF file");
        return false;
    }

    fprintf(stderr, "dims=%d, vars=%d, attrs=%d\n",
            m_ncfile->num_dims(), m_ncfile->num_vars(), m_ncfile->num_atts());

    for (int i = 0; i < m_ncfile->num_dims(); ++i)
    {
        fprintf(stderr, "%s: %ld\n",
                m_ncfile->get_dim(i)->name(),
                m_ncfile->get_dim(i)->size());
    }

    for (int i = 0; i < m_ncfile->num_vars(); ++i)
    {
        fprintf(stderr, "%s: dims=%d atts=%d vals=%ld type=%d\n",
                m_ncfile->get_var(i)->name(),
                m_ncfile->get_var(i)->num_dims(),
                m_ncfile->get_var(i)->num_atts(),
                m_ncfile->get_var(i)->num_vals(),
                m_ncfile->get_var(i)->type());
        //int dims = m_ncfile->get_var(i)->num_dims();
        NcVar *var = m_ncfile->get_var(i);
        for (int j = 0; j < var->num_dims(); ++j)
        {
            fprintf(stderr, "   %s: %ld edge=%ld\n",
                    var->get_dim(j)->name(),
                    var->get_dim(j)->size(),
                    var->edges()[j]);
        }
    }

    for (int i = 0; i < m_ncfile->num_atts(); ++i)
    {
        fprintf(stderr, "%s\n", m_ncfile->get_att(i)->name());
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
