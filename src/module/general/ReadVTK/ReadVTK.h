/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2006 ZAIK/RRZK  ++
// ++ Description: ReadVTK module                                         ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                       Thomas van Reimersdahl                        ++
// ++               Institute for Computer Science (Prof. Lang)           ++
// ++                        University of Cologne                        ++
// ++                         Robert-Koch-Str. 10                         ++
// ++                             50931 Koeln                             ++
// ++                                                                     ++
// ++ Date:  2006                                                         ++
// ++**********************************************************************/

#ifndef _READ_VTK_H
#define _READ_VTK_H

// includes
#include <stdlib.h>
#include <stdio.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <api/coModule.h>
using namespace covise;
#include <util/coviseCompat.h>

#include <vtkSmartPointer.h>
#include <vtkDataObject.h>

class vtkDataSet;
class vtkDataSetReader;
class vtkFieldData;

class ReadVTK : public coModule
{
    static const int NumPorts = 3;

public:
    ReadVTK(int argc, char *argv[]);
    virtual ~ReadVTK();


private:
    // main
    int compute(const char *);

    // virtual methods
    virtual void param(const char *name, bool inMapLoading);

    // local methods
    bool FileExists(const char *filename);

    void update();

private:
    void setChoices();
    char *m_filename;
    int blockSize;

    vtkSmartPointer<vtkDataObject> m_dataSet;

    coOutputPort *m_portGrid, *m_portNormals;
    coOutputPort *m_portPointData[NumPorts], *m_portCellData[NumPorts];

    coFileBrowserParam *m_pParamFile;
    coStringParam *m_pParamFilePattern;
    coChoiceParam *m_pointDataChoice[NumPorts];
    coChoiceParam *m_cellDataChoice[NumPorts];
    coDistributedObject* sdogrid, *sdopoint[NumPorts], *sdocell[NumPorts], *sdonormal;
    void createDataObjects(std::string &grid_name, std::string &normal_name, std::string pointDataName[NumPorts], std::string cellDataName[NumPorts], vtkDataSet* vdata);

    coBooleanParam *m_pTime;
    coIntSliderParam *m_pTimeMin;
    coIntSliderParam* m_pTimeMax;
    coIntSliderParam* m_pTimeSkip;


    int m_iTimestep;
    int m_iTimestepMin;
    int m_iTimestepMax;
    int m_iTimestepSkip;
};
#endif // _READ_VTK_H
