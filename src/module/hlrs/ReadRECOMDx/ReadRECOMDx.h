/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READRECOMDX_H
#define _READRECOMDX_H

#include <api/coModule.h>
using namespace covise;

#include <api/coStepFile.h>

#include "parser.h"
#include "action.h"
#include "MultiGridMember.h"

const int maxDataPorts = 10;

class ReadRECOMDx : public coModule
{
private:
    //  member functions
    virtual int compute(const char *port);

    //  Ports
    coOutputPort *p_UnstrGrid;
    coOutputPort *p_ScalarData[maxDataPorts];

    // Path of the file to be read in
    coFileBrowserParam *p_filePath;

    coBooleanParam *p_defaultIsLittleEndian;

    std::map<std::string, coDistributedObject *> grid;
    std::map<std::string, std::vector<coDistributedObject *> *> data;
    std::map<std::string, int> num;

    std::map<std::string, std::string> objects;

    coChoiceParam *p_ScalarChoice[maxDataPorts];

    actionClass *action;

    void traverse(std::vector<std::vector<DxObject *> > &ports,
                  bool generateOutput);
    coDistributedObject *generateData(DxObject *object, coOutputPort *port,
                                      float &min, float &max);
    coDoUnstructuredGrid *generateUSG(DxObject *positions,
                                      DxObject *connections,
                                      coOutputPort *port);

    void addToGroup(std::string object, std::string group);

public:
    ReadRECOMDx(int argc, char *argv[]);
    virtual ~ReadRECOMDx();

    virtual void param(const char *name, bool inMapLoading);
};

#endif
