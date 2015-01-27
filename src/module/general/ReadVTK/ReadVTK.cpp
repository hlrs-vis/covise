/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2004 ZAIK/RRZK  ++
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
// ++ Date:  26.09.2004                                                   ++
// ++**********************************************************************/

#include <iostream>

#include "ReadVTK.h"
#include <alg/coVtk.h>
#include "vtkDataArray.h"
#include "vtkDataSetReader.h"
#include "vtkDataSet.h"
#include "vtkUnstructuredGrid.h"
#include "vtkStructuredGrid.h"
#include "vtkUnstructuredGrid.h"
#include "vtkStructuredGrid.h"
#include "vtkRectilinearGrid.h"
//#include "vtkUniformGrid.h"
#include "vtkStructuredPoints.h"
#include "vtkPolyData.h"
#include "vtkPointData.h"
#include "vtkCellData.h"
#include "vtkDataArray.h"
#include "vtkCellArray.h"
#if VTK_MAJOR_VERSION < 5
#include "vtkIdType.h"
#endif
#include "vtkDataSetAttributes.h"
#include "vtkUnsignedCharArray.h"
#include "vtkUnstructuredGridWriter.h"
#include "vtkStructuredGridWriter.h"
#include "vtkDataSetWriter.h"

#include <do/coDoData.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoSet.h>

//////
////// we must provide main to init covise
//////

static const char *choiceVal0[] = { "1. scalar field" };
static const char *choiceVal1[] = { "1. vector field" };
static const char **choiceVal[2];
static const int choiceLen[2] = { 1, 1 };

ReadVTK::ReadVTK(int argc, char *argv[])
    : coModule(argc, argv, "Read Visualization Toolkit (VTK) data files")
{
    m_filename = new char[256];

    m_portGrid = addOutputPort("GridOut0", "UniformGrid|StructuredGrid|UnstructuredGrid|RectilinearGrid|Polygons|Lines", "Grid output");

    m_portScalars = addOutputPort("DataOut0", "Float|Int|Byte", "first (or selected) scalar field in data file");
    m_portVectors = addOutputPort("DataOut1", "Vec3|RGBA", "first (or selected) vector field in data file");
    m_portNormals = addOutputPort("DataOut2", "Vec3", "normals");

    m_pParamFile = addFileBrowserParam("filename", "Specify the filename of the VTK data file(s).");
    m_pParamFile->setValue("./", "*.vtk");

    choiceVal[0] = choiceVal0;
    choiceVal[1] = choiceVal1;

    m_pParamScalar = this->addChoiceParam("scalar", "Select one scalar field included in the VTK file. Default is the first one.");
    m_pParamScalar->setValue(choiceLen[0], choiceVal[0], 0);

    m_pParamVector = this->addChoiceParam("vector", "Select one vector field included in the VTK file. Default is the first one.");
    m_pParamVector->setValue(choiceLen[1], choiceVal[1], 0);

    m_pTime = addBooleanParam("timesteps", "Read several timesteps at once.");
    m_pTime->setValue(0);

    m_pParamFilePattern = addStringParam("filenamepattern", "Specify the filename pattern allowing to read in several vtk datafiles/timesteps.");

    m_pTimeMin = addIntSliderParam("timesteps_min", "Adjust minimal timestep.");
    m_pTimeMin->setValue(0);

    m_pTimeMax = addIntSliderParam("timesteps_max", "Adjust maximal timestep.");
    m_pTimeMax->setValue(0);

    m_iTimestep = 0;
    m_iTimestepMin = 0;
    m_iTimestepMax = 0;

    // setup the vtk reader
    m_pReader = vtkDataSetReader::New();
}

ReadVTK::~ReadVTK()
{
    if (m_filename != NULL)
    {
        delete[] m_filename;
        m_filename = NULL;
    }
    if (m_pReader != NULL)
    {

        m_pReader->vtkDataSetReader::Delete();
        m_pReader = NULL;
    }
}

//////
////// this is our compute-routine
//////

int ReadVTK::compute(const char *)
{
    FILE *pFile = Covise::fopen(m_pParamFile->getValue(), "r");
    if (!pFile)
    {
        Covise::sendError("ERROR: can't open file %s", m_filename);
        return FAIL;
    }
    fclose(pFile);

    update();

    return SUCCESS;
}

//////
////// the read-function itself
//////

bool ReadVTK::FileExists(const char *filename)
{
    FILE *fp = NULL;
    if (!(fp = fopen(filename, "r")))
    {
        std::cout << "File: " << filename << " does not exist." << std::endl;
        return false;
    }
    else
    {
        std::cout << "File: " << filename << " ok." << std::endl;
        fclose(fp);
    }
    return true;
}

void ReadVTK::update()
{

    coDoFloat *str_s3d_out = NULL;
    coDoFloat *unstr_s3d_out = NULL;
    coDoVec3 *str_v3d_out = NULL;
    coDoVec3 *unstr_v3d_out = NULL;

    char *out_obj_scalar;
    char *out_obj_vector;
    char *obj_name;
    std::string sInfo;
    int iScalar = 0, iVector = 0;
    //char *cValue=NULL;

    // here we go
    m_pReader->SetFileName(m_filename);
    m_pReader->Update();
    const int iNrScalarsInFile = m_pReader->GetNumberOfScalarsInFile();

    bool timesteps = m_pTime->getValue();

    std::vector<coDistributedObject *> dogrid, doscalar, dovector, donormal;
    for (int iTimestep = m_iTimestepMin; iTimestep <= m_iTimestepMax; iTimestep++)
    {
        std::string grid_name = m_portGrid->getObjName();
        std::string scalar_name = m_portScalars->getObjName();
        std::string vector_name = m_portVectors->getObjName();
        std::string normal_name = m_portNormals->getObjName();

        if (timesteps)
        {
            const char *filenamepattern = m_pParamFilePattern->getValue();
            if (!std::string(filenamepattern).find("%"))
            {
                Covise::sendInfo("no valid filename pattern - does not contain %% for printf format string");
                return;
            }

            sprintf(m_filename, filenamepattern, iTimestep);
            std::cout << "New Filename is " << m_filename << std::endl;

            m_pReader->SetFileName(m_filename);
            m_pReader->Update();

            char buf[256];
            snprintf(buf, sizeof(buf), "%d", iTimestep - m_iTimestepMin);

            grid_name += "_";
            scalar_name += "_";
            vector_name += "_";

            grid_name += buf;
            scalar_name += buf;
            vector_name += buf;
        }
        else
        {
            m_pReader->SetFileName(m_pParamFile->getValue());
        }

        vtkDataSet *vdata = m_pReader->GetOutput();
        coDoGrid *grid = coVtk::vtkGrid2Covise(grid_name, vdata);
        if (!vdata)
        {
            Covise::sendInfo("could not read: %s", m_pReader->GetFileName());
            return;
        }
        /* if (!grid)
      {
         Covise::sendInfo("not supported: %s", vdata->GetClassName());
         return;
      }*/

        dogrid.push_back(grid);

        if (iNrScalarsInFile > 0)
        {
            Covise::sendInfo("Reading the data. Please wait ...");
            Covise::sendInfo("...filename is %s", m_filename);

            Covise::get_choice_param("scalar", &iScalar);
            m_pReader->SetScalarsName(m_pReader->GetScalarsNameInFile(iScalar - 1));
            Covise::sendInfo("...activated scalar field is %s", m_pReader->GetScalarsNameInFile(iScalar - 1));
        }

        const int iNrVectorsInFile = m_pReader->GetNumberOfVectorsInFile();

        if (iNrVectorsInFile > 0)
        {
            Covise::get_choice_param("vector", &iVector);
            m_pReader->SetVectorsName(m_pReader->GetVectorsNameInFile(iVector - 1));
            Covise::sendInfo("...activated vector field is %s", m_pReader->GetVectorsNameInFile(iVector - 1));
        }
        m_pReader->Update();

        if (coDistributedObject *scaldata = coVtk::vtkData2Covise(scalar_name, vdata, coVtk::Scalars, m_pReader->GetScalarsNameInFile(iScalar - 1), dynamic_cast<coDoAbstractStructuredGrid *>(grid)))
            doscalar.push_back(scaldata);
        if (coDistributedObject *vecdata = coVtk::vtkData2Covise(vector_name, vdata, coVtk::Vectors, m_pReader->GetVectorsNameInFile(iVector - 1), dynamic_cast<coDoAbstractStructuredGrid *>(grid)))
            dovector.push_back(vecdata);
        if (coDistributedObject *normdata = coVtk::vtkData2Covise(normal_name, vdata, coVtk::Normals, NULL, dynamic_cast<coDoAbstractStructuredGrid *>(grid)))
            donormal.push_back(normdata);

        Covise::sendInfo("The input data was read (%d scalars, %d vectors).", iNrScalarsInFile, iNrVectorsInFile);
    }

    coDistributedObject *outGrid = NULL, *outScalars = NULL, *outVectors = NULL, *outNormals = NULL;
    if (timesteps)
    {
        char buf[1024];
        snprintf(buf, sizeof(buf), "%d %d", m_iTimestepMin, m_iTimestepMax);
        std::cout << "TIMESTEP is " << std::string(buf) << std::endl;
        outGrid = NULL;
        if (dogrid.size() > 0)
        {
            outGrid = new coDoSet(m_portGrid->getObjName(), dogrid.size(), &dogrid[0]);
            outGrid->addAttribute("TIMESTEP", buf);
        }
        outScalars = NULL;
        if (doscalar.size() > 0)
        {
            outScalars = new coDoSet(m_portScalars->getObjName(), doscalar.size(), &doscalar[0]);
            outScalars->addAttribute("TIMESTEP", buf);
        }

        outVectors = NULL;
        if (dovector.size() > 0)
        {
            outVectors = new coDoSet(m_portVectors->getObjName(), dovector.size(), &dovector[0]);
            outVectors->addAttribute("TIMESTEP", buf);
        }

        outNormals = NULL;
        if (donormal.size() > 0)
        {
            outNormals = new coDoSet(m_portNormals->getObjName(), donormal.size(), &donormal[0]);
            outNormals->addAttribute("TIMESTEP", buf);
        }
    }
    else
    {
        if (!dogrid.empty())
            outGrid = dogrid[0];
        if (!doscalar.empty())
            outScalars = doscalar[0];
        if (!dovector.empty())
            outVectors = dovector[0];
        if (!donormal.empty())
            outNormals = donormal[0];
    }

    // Assign sets to output ports:
    if (outGrid)
        m_portGrid->setCurrentObject(outGrid);
    if (outScalars)
        m_portScalars->setCurrentObject(outScalars);
    if (outVectors)
        m_portVectors->setCurrentObject(outVectors);
    if (outNormals)
        m_portNormals->setCurrentObject(outNormals);
}

void ReadVTK::postInst()
{
    m_pParamScalar->show();
}

void ReadVTK::param(const char *name, bool /*inMapLoading*/)
{
    cerr << "\n ------- Parameter Callback for '"
         << name
         << "'" << endl;

    if (strcmp(name, m_pParamFile->getName()) == 0)
    {
        cerr << "---------------" << endl;
        //m_pReader->DebugOn();
        m_filename = (char *)m_pParamFile->getValue();
        m_pReader->ReadAllScalarsOn();
        m_pReader->ReadAllVectorsOn();
        m_pReader->ReadAllFieldsOn();

        m_pReader->SetFileName(m_filename);
        m_pReader->Update();

        m_pParamFilePattern->setValue(m_filename);

        // determine the scalar fields in the dataset
        const int iNrScalarsInFile = m_pReader->GetNumberOfScalarsInFile();
        int iNrFieldDataInFile = m_pReader->GetNumberOfFieldDataInFile();
        if (iNrFieldDataInFile > 0)
        {
            m_pReader->SetFieldDataName(m_pReader->GetFieldDataNameInFile(0));
            m_pReader->Update();
            vtkDataSet *dataSet = m_pReader->GetOutput();
            m_fieldData = dataSet->GetFieldData();
            iNrFieldDataInFile = m_fieldData->GetNumberOfArrays();
        }
        if ((iNrScalarsInFile + iNrFieldDataInFile) > 0)
        {
            int i = 0;
            char **cScalarNames;
            cScalarNames = (char **)malloc((iNrScalarsInFile + iNrFieldDataInFile) * sizeof(char *));
            for (i = 0; i < iNrScalarsInFile; i++)
            {
                cScalarNames[i] = (char *)malloc((std::string(m_pReader->GetScalarsNameInFile(i)).length() + 1) * sizeof(char));
                strcpy(cScalarNames[i], m_pReader->GetScalarsNameInFile(i));
            }
            for (i = 0; i < iNrFieldDataInFile; i++)
            {
                cScalarNames[i + iNrScalarsInFile] = (char *)malloc((std::string(m_fieldData->GetArrayName(i)).length() + 1) * sizeof(char));
                strcpy(cScalarNames[i + iNrScalarsInFile], m_fieldData->GetArrayName(i));
            }
            m_pParamScalar->setValue((iNrScalarsInFile + iNrFieldDataInFile), cScalarNames, 0);
        }
        //m_pReader->DebugOff();
        // determine the vector fields in the dataset
        const int iNrVectorsInFile = m_pReader->GetNumberOfVectorsInFile();

        if (iNrVectorsInFile > 0)
        {
            int i = 0;
            char **cVectorNames;
            cVectorNames = (char **)malloc(iNrVectorsInFile * sizeof(char *));
            for (i = 0; i < iNrVectorsInFile; i++)
            {
                cVectorNames[i] = (char *)malloc((std::string(m_pReader->GetVectorsNameInFile(i)).length() + 1) * sizeof(char));
                strcpy(cVectorNames[i], m_pReader->GetVectorsNameInFile(i));
            }
            m_pParamVector->setValue(iNrVectorsInFile, cVectorNames, 0);
        }
    }
    else if (strcmp(name, m_pParamFilePattern->getName()) == 0)
    {
        int result = 0, i = 0;
        char *cTmp = new char[1024];
        // per definitionem, the fully specified filename is the minimum
        result = sscanf(m_filename, m_pParamFilePattern->getValue(), &i);
        if (result == 1)
        {
            m_iTimestepMin = i;
            sprintf(cTmp, m_pParamFilePattern->getValue(), i);
            while (FileExists(cTmp))
            {
                i++;
                sprintf(cTmp, m_pParamFilePattern->getValue(), i);
            }
            m_iTimestepMax = i - 1;
        }
        std::cout << "Timestep Min: " << m_iTimestepMin << std::endl;
        std::cout << "Timestep Max: " << m_iTimestepMax << std::endl;

        m_pTimeMin->setValue(m_iTimestepMin);
        m_pTimeMin->setMin(m_iTimestepMin);
        m_pTimeMin->setMax(m_iTimestepMax);
        m_pTimeMax->setValue(m_iTimestepMax);
        m_pTimeMax->setMin(m_iTimestepMin);
        m_pTimeMax->setMax(m_iTimestepMax);
    }
    else if (strcmp(name, m_pTimeMax->getName()) == 0)
    {
        m_iTimestepMax = m_pTimeMax->getValue();
        if (m_iTimestepMax < m_iTimestepMin)
        {
            m_pTimeMax->setValue(m_iTimestepMin);
            m_iTimestepMax = m_iTimestepMin;
            Covise::sendInfo("TimestepMax is not allowed to be smaller than TimestepMin. Thus, the value of TimestepMax is changed to TimestepMin.");
        }
        if (m_iTimestepMax > m_pTimeMax->getMax())
        {
            m_pTimeMax->setValue(m_pTimeMax->getMax());
            m_iTimestepMax = m_pTimeMax->getMax();
            Covise::sendInfo("TimestepMax changed to maximum timestep.");
        }
        if (m_iTimestepMax < m_pTimeMax->getMin())
        {
            m_pTimeMax->setValue(m_pTimeMax->getMin());
            m_iTimestepMax = m_pTimeMax->getMin();
            Covise::sendInfo("TimestepMax changed.");
        }

        m_pTimeMin->setMax(m_iTimestepMax);
        std::cout << "iTimestepMax = " << m_iTimestepMax << std::endl;
    }
    else if (strcmp(name, m_pTimeMin->getName()) == 0)
    {
        m_iTimestepMin = m_pTimeMin->getValue();
        if (m_iTimestepMin > m_iTimestepMax)
        {
            m_pTimeMin->setValue(m_iTimestepMax);
            m_iTimestepMin = m_iTimestepMax;
            Covise::sendInfo("TimestepMin is not allowed to be greater than TimestepMax. Thus, the value of TimestepMin is changed to TimestepMax.");
        }
        if (m_iTimestepMin < m_pTimeMin->getMin())
        {
            m_pTimeMax->setValue(m_pTimeMin->getMin());
            m_iTimestepMin = m_pTimeMin->getMin();
            Covise::sendInfo("TimestepMin changed to minimum timestep.");
        }
        if (m_iTimestepMin > m_pTimeMin->getMax())
        {
            m_pTimeMax->setValue(m_pTimeMin->getMax());
            m_iTimestepMin = m_pTimeMin->getMax();
            Covise::sendInfo("TimestepMin changed.");
        }
        m_pTimeMax->setMin(m_iTimestepMin);
        std::cout << "iTimestepMin = " << m_iTimestepMin << std::endl;
    }
}

MODULE_MAIN(Reader, ReadVTK)
