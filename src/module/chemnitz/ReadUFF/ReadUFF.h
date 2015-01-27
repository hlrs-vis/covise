/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ReadUFF_H
#define _ReadUFF_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: ReadUFF universal file reader                          ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Andreas Funke                            ++
// ++               University of technology Chemnitz	                  ++
// ++											                          ++
// ++										                              ++
// ++                                                                     ++
// ++ Date:  16.04.2007  V1.0                                             ++
// ++**********************************************************************/

#include <map>
#include <list>
#include <vector>
#include <api/coModule.h>
using namespace covise;
#include <do/coDoSet.h>
#include "datasets.h"
#include "FortranData.h"

using namespace std;
#define NUM_PORTS 4 // specify the number of output ports

class ReadUFF : public coModule
{

private:
    FILE *uffFile; //the universal file

    vector<dataset15> old_nodes; //is marked as obsolete
    vector<dataset2411> nodes;
    vector<dataset58> DOFs; //function at nodal DOFs
    dataset82 *traceLines; //tracelines
    dataset151 fileHeader; //file header
    dataset164 units; //units

    unsigned int numNodes;
    unsigned int numTraceLines;

    void Clean();

    char loadedFile[256];
    bool forceUpdate;

    virtual int compute(const char *port);
    virtual void param(const char *paramName, bool inMapLoading);
    virtual void postInst();

    int ReadFile(const char *name);
    int ReadFileHeaders(const char *name);

    bool GetDelimiter(bool resetPosition = false); //read a delimiter, return true if found
    int GetDatasetNr(); //read a dataset number, return 0 if unsuccessful
    coDoSet *PackDataset(const char *datasetName, const char *portName); //return a coDoSet which contains the dataset information

    map<unsigned int, char *> choices;

    coOutputPort *outputPorts[NUM_PORTS];
    coChoiceParam *portChoices[NUM_PORTS];

    coFileBrowserParam *fileName;

public:
    ReadUFF(int argc, char *argv[]);
    virtual ~ReadUFF();
};
#endif
