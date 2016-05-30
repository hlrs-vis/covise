/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef READFOAM_H
#define READFOAM_H
/**************************************************************************\
 **                                                           (C)2013 RUS  **
 **                                                                        **
 ** Description: Read FOAM data format                                     **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** History:                                                               **
 ** May   13        C.Kopf          V1.0                                   **
 *\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <api/coModule.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>

#include "foamtoolbox.h"

using namespace covise;
typedef int index_t;

class ReadFOAM : public coModule
{

private:
    const char *casedir; //Path to FOAM-Case Directory
    CaseInfo m_case;
    int num_ports;
    int num_boundary_data_ports;
    std::vector<std::string> lastDataPortSelection;
    std::vector<std::string> lastBoundaryPortSelection;

    //Output Ports
    coOutputPort *meshOutPort;
    coOutputPort *boundaryOutPort;
    std::vector<coOutputPort *> outPorts;
    std::vector<coOutputPort *> boundaryDataPorts;

    //Parameters
    coFileBrowserParam *filenameParam;
    coFloatParam *starttimeParam, *stoptimeParam;
    coIntScalarParam *skipfactorParam;
    coBooleanParam *meshParam;
    std::vector<coChoiceParam *> portChoice;
    std::vector<coChoiceParam *> boundaryDataChoice;
    coStringParam *patchesStringParam;
    coBooleanParam *boundaryParam;

    //  member functions
    virtual int compute(const char *port);
    bool vectorsAreFilled();

public:
    ReadFOAM(int argc, char *argv[]); //Constructor
    virtual ~ReadFOAM(); //Destructor
    std::vector<const char *> getFieldList();
    virtual void param(const char *, bool);

    coDoUnstructuredGrid *loadMesh(const std::string &meshdir,
                                   const std::string &pointsdir,
                                   const std::string &meshObjName,
                                   const index_t Processor = -1);
    coDoPolygons *loadPatches(const std::string &meshdir,
                              const std::string &pointsdir,
                              const std::string &boundObjName,
                              const std::string &selection,
                              const index_t Processor = -1,
                              const index_t saveMapTo = -1);
    coDistributedObject *loadField(const std::string &timedir,
                               const std::string &file,
                               const std::string &vecObjName,
                               const std::string &meshdir);
    coDistributedObject *loadBoundaryField(const std::string &timedir,
                                      const std::string &meshdir,
                                      const std::string &file,
                                      const std::string &vecObjName,
                                      const std::string &selection);
    std::map<int, coDoUnstructuredGrid *> basemeshs;
    std::map<int, coDoPolygons *> basebounds;
    std::vector<std::map<int, int> > pointmaps;
};
#endif // READFOAM_H
