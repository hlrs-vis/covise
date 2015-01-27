/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_WRITEVTK_H
#define CO_WRITEVTK_H 1

#include <api/coSimpleModule.h>
#include <util/coviseCompat.h>
#include <vtkDataSetWriter.h>
#include <vector>

using namespace covise;

class vtkMultiPieceDataSet;

class WriteVtk : public coSimpleModule
{
    static const int NumFields = 6;

    std::vector<coInputPort *> m_inPorts;
    virtual int compute(const char *port);
    virtual void preHandleObjects(coInputPort **);
    virtual void postHandleObjects(coOutputPort **);

    std::vector<const coDistributedObject *> m_savedInput;
    coInputPort **m_savedInputPorts;

    coFileBrowserParam *m_FileNameParam;
    coBooleanParam *m_BinaryFileTypeParam;
    coBooleanParam *m_filePerTimeStep;

    coStringParam *m_FieldNameParam[NumFields];
    coStringParam *m_NormalsNameParam;
    coStringParam *m_TCoordsNameParam;
#if 0
   coStringParam *m_GlobalIdsNameParam;
   coStringParam *m_LookupTableNameParam;
   coStringParam *m_PedigreeIdsNameParam;
   coStringParam *m_TensorsNameParam;
#endif

    std::vector<vtkMultiPieceDataSet *> m_collection;
    void write(vtkDataObject *obj, const char *filename);

public:
    WriteVtk(int argc, char *argv[]);
    virtual ~WriteVtk();
};
#endif
