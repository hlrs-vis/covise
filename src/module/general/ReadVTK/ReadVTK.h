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

class vtkDataSet;
class vtkDataSetReader;
class vtkFieldData;

// defines
#define _FILE_TYPE_BINARY 1
#define _FILE_TYPE_FORTRAN 2
#define _FILE_TYPE_ASCII 3

#define _FILE_STRUCTURED_GRID 4
#define _FILE_UNSTRUCTURED_GRID 5
#define _FILE_IBLANKED 6

#define _FILE_SOLUTION 7
#define _FILE_DATA 8
#define _FILE_FUNCTION 9

#define _FILE_SINGLE_ZONE 10
#define _FILE_MULTI_ZONE 11

#define _READ_GRID 12
#define _READ_DATA 13

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
    void setChoices(vtkDataSet *dataSet);
    char *m_filename;
    int blockSize;

    vtkSmartPointer<vtkDataSet> m_dataSet;
    vtkFieldData *m_fieldData;

    coOutputPort *m_portGrid, *m_portNormals;
    coOutputPort *m_portPointData[NumPorts], *m_portCellData[NumPorts];

    coFileBrowserParam *m_pParamFile;
    coStringParam *m_pParamFilePattern;
    coChoiceParam *m_pointDataChoice[NumPorts];
    coChoiceParam *m_cellDataChoice[NumPorts];

    coBooleanParam *m_pTime;
    coIntSliderParam *m_pTimeMin;
    coIntSliderParam *m_pTimeMax;

    int m_iTimestep;
    int m_iTimestepMin;
    int m_iTimestepMax;
};
#endif // _READ_VTK_H
