/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2004 ZAIK/RRZK  ++
// ++ Description: ReadPDB module                                         ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                       Thomas van Reimersdahl                        ++
// ++               Institute for Computer Science (Prof. Lang)           ++
// ++                        University of Cologne                        ++
// ++                         Robert-Koch-Str. 10                         ++
// ++                             50931 Köln                              ++
// ++                                                                     ++
// ++ Date:  26.09.2004                                                   ++
// ++**********************************************************************/

#include <appl/ApplInterface.h>
#include <vector>
#include "ReadPDB.h"
#include "vtkMyPDBReader.h"
#include "vtkPolyData.h"
#include "vtkCellArray.h"
#include "vtkPointData.h"
#include <config/CoviseConfig.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>

ReadPDB::ReadPDB(int argc, char *argv[])
    : coModule(argc, argv, "Read PDB files")
{
    m_filename = new char[256];

    m_portPoints = addOutputPort("points", "Points", "points Output");
    m_portPoints->setInfo("points Output");

    m_portAtomType = addOutputPort("AtomType", "Int", "atom type");
    m_portAtomType->setInfo("Atom type");

    m_portBondsLines = addOutputPort("bonds", "Lines", "Lines (bonds) Output");
    m_portBondsLines->setInfo("Lines (bonds) Output");

    m_portGroupsLines = addOutputPort("groups", "Lines", "Lines (groups) Output");
    m_portGroupsLines->setInfo("Lines (groups) Output");

    m_pParamFile = addFileBrowserParam("filename", "name of PDB file");
    m_pParamFile->setValue("", "*.pdb/*.*");

    // setup the vtk reader
    m_pReader = vtkMyPDBReader::New();
    m_pReader->SetFileName(m_filename);
    m_pReader->SetHBScale(1.0);
    m_pReader->SetBScale(1.0);

    m_pTime = addBooleanParam("timesteps", "test");
    m_pTime->setValue(0);

    m_pTimeMin = addInt32Param("timesteps_min", "test");
    m_pTimeMin->setValue(0);

    m_pTimeMax = addInt32Param("timesteps_max", "test");
    m_pTimeMax->setValue(0);

    m_iTimestep = 0;
    m_iTimestepMin = 0;
    m_iTimestepMax = 0;
}

//////
////// this is our compute-routine
//////

int ReadPDB::compute(const char *)
{
    FILE *pFile;

    // get parameters
    strcpy(m_filename, m_pParamFile->getValue());

    // compute parameters
    if ((pFile = Covise::fopen(m_filename, "r")) <= 0)
    {
        sendError("ERROR: can't open file %s", m_filename);
        return STOP_PIPELINE;
    }
    fclose(pFile);

    m_iTimestepMax = m_pTimeMax->getValue();
    if (m_iTimestepMax < m_iTimestepMin)
        m_iTimestepMax = m_iTimestepMin;
    readPDBFile();

    return CONTINUE_PIPELINE;
}

//////
////// the read-function itself
//////

bool ReadPDB::fileExists(const char *filename)
{
    FILE *fp = NULL;
    if (!(fp = fopen(filename, "r")))
    {
        //      std::cout << "File: " << filename << " does not exist." << std::endl;
        return false;
    }
    else
    {
        //		std::cout << "File: " << filename << " ok." << std::endl;
        fclose(fp);
    }
    return true;
}

void ReadPDB::readPDBFile()
{
    const char *obj_points;
    const char *obj_scalar_colors_name;
    const char *obj_lines_name;
    const char *obj_groups_lines_name;
    std::string sInfo;

    // here we go
    m_pReader->SetFileName(m_filename);
    m_pReader->Update();

    Covise::sendInfo("The input data was read.");

    std::string strTemp;
    std::string strClassName = std::string(m_pReader->GetOutput()->GetClassName());

    if (!strClassName.compare("vtkPolyData"))
    {
        std::vector<coDistributedObject *> dataPoints;
        std::vector<coDistributedObject *> dataAtomType;
        std::vector<coDistributedObject *> dataLines;
        std::vector<coDistributedObject *> dataGroupsLines;

        vtkPolyData *pVtkData = dynamic_cast<vtkPolyData *>(m_pReader->GetOutput());

        if (pVtkData == NULL)
            return;

        // name handles
        obj_points = m_portPoints->getObjName();
        obj_scalar_colors_name = m_portAtomType->getObjName();
        obj_lines_name = m_portBondsLines->getObjName();
        obj_groups_lines_name = m_portGroupsLines->getObjName();

        std::string obj_points_prefix = std::string(obj_points);
        std::string obj_scalar_color_prefix = std::string(obj_scalar_colors_name);
        std::string obj_lines_prefix = std::string(obj_lines_name);
        std::string obj_groups_lines_prefix = std::string(obj_groups_lines_name);

        std::string obj_points_current = obj_points_prefix;
        std::string obj_scalar_color_current = obj_scalar_color_prefix;
        std::string obj_lines_current = obj_lines_prefix;
        std::string obj_groups_lines_current = obj_groups_lines_prefix;

        char filenamepattern[1024];
        strcpy(filenamepattern, "");
        int result, result2;
        result = strcspn(m_filename, "0123456789");
        strncat(filenamepattern, m_filename, result);
        sprintf(filenamepattern, "%s%s", filenamepattern, "%d");
        result2 = strspn(m_filename + result, "0123456789");

        strncat(filenamepattern, m_filename + result + result2, sizeof(m_filename) - result - result2);

        // activated timesteps?
        bool iTime = m_pTime->getValue();

        // for each timestep, assemble its path and filename,
        // read the file and build the covise data structures
        int iTimestepMax = 0;
        int iTimestepMin = 0;

        if (iTime)
        {
            iTimestepMax = m_iTimestepMax;
            iTimestepMin = m_iTimestepMin;
        }
        else
        {
            iTimestepMax = m_iTimestepMin;
            iTimestepMin = m_iTimestepMin;
        }

        std::cout << "\n iTimestepMin ist " << iTimestepMin << std::endl;
        std::cout << "\n iTimestepMax ist " << iTimestepMax << std::endl;

        // for each timestep, assemble its path and filename,
        // read the file and build the covise data structures
        for (int iTimestep = m_iTimestepMin;
             iTimestep <= m_iTimestepMax && ((!iTime && iTimestep == m_iTimestepMin) || iTime);
             iTimestep++)
        {
            if (iTime)
            {
                if (!std::string(filenamepattern).find("%"))
                    return;

                sprintf(m_filename, filenamepattern, iTimestep);
                if (!fileExists(m_filename))
                    continue;

                std::cout << "Read File: " << m_filename;
                m_pReader->SetFileName(m_filename);
                m_pReader->Update();
                std::cout << " finished." << std::endl;

                char buf[256];
                snprintf(buf, sizeof(buf), "%d", iTimestep - m_iTimestepMin);

                obj_points_current = obj_points_prefix;
                obj_scalar_color_current = obj_scalar_color_prefix;
                obj_lines_current = obj_lines_prefix;
                obj_groups_lines_current = obj_groups_lines_prefix;

                obj_points_current.append("_").append(std::string(buf));
                obj_scalar_color_current.append("_").append(std::string(buf));
                obj_lines_current.append("_").append(std::string(buf));
                obj_groups_lines_current.append("_").append(std::string(buf));
            }

            coDoPoints *pCovisePoints = new coDoPoints(obj_points_current.c_str(), pVtkData->GetNumberOfPoints());

            int sz = pVtkData->GetNumberOfPoints();
            coDoInt *pCoviseAtomType = new coDoInt(obj_scalar_color_current.c_str(), sz);

            coDoLines *pCoviseLines = new coDoLines(obj_lines_current.c_str(),
                                                    pVtkData->GetNumberOfPoints(),
                                                    pVtkData->GetLines()->GetNumberOfConnectivityEntries() - pVtkData->GetNumberOfLines(),
                                                    pVtkData->GetNumberOfLines());

            coDoLines *pCoviseGroupsLines = new coDoLines(obj_groups_lines_current.c_str(),
                                                          pVtkData->GetNumberOfPoints(),
                                                          m_pReader->GetGroupsLines()->GetNumberOfConnectivityEntries() - m_pReader->GetNumberOfGroupsLines(),
                                                          m_pReader->GetNumberOfGroupsLines());

            /*				std::cout << "NrPoints: " << pVtkData->GetNumberOfPoints()
                                        << " 2: " << m_pReader->GetGroupsLines()->GetNumberOfConnectivityEntries() - m_pReader->GetNumberOfGroupsLines()
                                        << " 3: " << m_pReader->GetNumberOfGroupsLines()
                                        << " GroupName: " << obj_groups_lines_current << std::endl;
                                        */
            float *fcoordx = NULL, *fcoordy = NULL, *fcoordz = NULL;
            float *fcoordx2 = NULL, *fcoordy2 = NULL, *fcoordz2 = NULL;
            float *fcoordx3 = NULL, *fcoordy3 = NULL, *fcoordz3 = NULL;
            int *icorner2 = NULL, *ipolygon2 = NULL;
            int *icorner3 = NULL, *ipolygon3 = NULL;
            int *iType = NULL;

            pCovisePoints->getAddresses(&fcoordx, &fcoordy, &fcoordz);
            pCoviseLines->getAddresses(&fcoordx2, &fcoordy2, &fcoordz2, &icorner2, &ipolygon2);
            pCoviseGroupsLines->getAddresses(&fcoordx3, &fcoordy3, &fcoordz3, &icorner3, &ipolygon3);
            pCoviseAtomType->getAddress(&iType);

            for (int i = 0; i < pVtkData->GetNumberOfPoints(); i++)
            {
                fcoordx[i] = fcoordx2[i] = fcoordx3[i] = static_cast<float>(pVtkData->GetPoint(i)[0]);
                fcoordy[i] = fcoordy2[i] = fcoordy3[i] = static_cast<float>(pVtkData->GetPoint(i)[1]);
                fcoordz[i] = fcoordz2[i] = fcoordz3[i] = static_cast<float>(pVtkData->GetPoint(i)[2]);
                iType[i] = m_pReader->GetAtomType(i) + 1;
            }

            vtkIdType npts = 0, *pts = NULL;

            // VTK LINES
            int k = 0;
            if (pVtkData->GetLines() != NULL)
            {
                pVtkData->GetLines()->InitTraversal();
                for (int i = 0;; i++)
                {
                    if (!pVtkData->GetLines()->GetNextCell(npts, pts))
                        break;

                    if (i == 0)
                        ipolygon2[0] = 0;
                    else
                        ipolygon2[i] = ipolygon2[i - 1] + npts;

                    for (int j = 0; j < npts; j++)
                    {
                        icorner2[k] = pts[j];
                        ++k;
                    }
                }
            }

            // VTK LINES
            k = 0;
            if (m_pReader->GetGroupsLines() != NULL)
            {
                m_pReader->GetGroupsLines()->InitTraversal();
                for (int i = 0; i < m_pReader->GetNumberOfGroupsLines(); i++)
                {
                    m_pReader->GetGroupsLines()->GetNextCell(npts, pts);
                    if (i == 0)
                        ipolygon3[0] = 0;
                    else
                        ipolygon3[i] = ipolygon3[i - 1] + npts;

                    for (int j = 0; j < npts; j++)
                    {
                        icorner3[k] = pts[j];
                        ++k;
                    }
                }
            }

            //				std::cout << "CheckSpheres=" << pCoviseSpheres->checkObject() << std::endl;

            dataPoints.push_back(pCovisePoints);
            dataAtomType.push_back(pCoviseAtomType);
            dataLines.push_back(pCoviseLines);
            dataGroupsLines.push_back(pCoviseGroupsLines);
        }

        if (iTime)
        {
            char buf[1024];
            snprintf(buf, sizeof(buf), "%d %d", m_iTimestepMin - 1, m_iTimestepMax - 1);
            //			std::cout << buf << std::endl;

            // Create set objects:
            coDoSet *setPoints = new coDoSet(m_portPoints->getObjName(), dataPoints.size(), &dataPoints[0]);
            coDoSet *setAtomType = new coDoSet(m_portAtomType->getObjName(), dataAtomType.size(), &dataAtomType[0]);
            coDoSet *setLines = new coDoSet(m_portBondsLines->getObjName(), dataLines.size(), &dataLines[0]);
            coDoSet *setGroupsLines = new coDoSet(m_portGroupsLines->getObjName(), dataGroupsLines.size(), &dataGroupsLines[0]);

            dataPoints.clear();
            dataAtomType.clear();
            dataLines.clear();
            dataGroupsLines.clear();

            // Set timestep attribute:
            setPoints->addAttribute("TIMESTEP", buf);
            setAtomType->addAttribute("TIMESTEP", buf);
            setLines->addAttribute("TIMESTEP", buf);
            setGroupsLines->addAttribute("TIMESTEP", buf);

            // Assign sets to output ports:
            m_portPoints->setCurrentObject(setPoints);
            m_portAtomType->setCurrentObject(setAtomType);
            m_portBondsLines->setCurrentObject(setLines);
            m_portGroupsLines->setCurrentObject(setGroupsLines);
        }
        else
        {
            // Assign single instances to output ports:
            if (dataPoints.size())
                m_portPoints->setCurrentObject((coDistributedObject *)(dataPoints[0]));
            if (dataAtomType.size())
                m_portAtomType->setCurrentObject((coDistributedObject *)dataAtomType[0]);
            if (dataLines.size())
                m_portBondsLines->setCurrentObject((coDistributedObject *)dataLines[0]);
            if (dataGroupsLines.size())
                m_portGroupsLines->setCurrentObject((coDistributedObject *)dataGroupsLines[0]);
        }
    }
    else
    {
        Covise::sendInfo("not supported data type: %s", strClassName.c_str());
    }
}

void ReadPDB::param(const char *name, bool inMapLoading)
{
    /*   cerr << "\n ------- Parameter Callback for '"
      << name
      << "'" << endl;
*/
    if (strcmp(name, m_pParamFile->getName()) == 0)
    {
        m_filename = (char *)m_pParamFile->getValue();
        m_pReader->SetFileName(m_filename);

        int iResultDecimal = 0, i = 0;
        //	   i=iResultDecimal/0;
        char cTmp[1024] = "";
        // per definitionem, the fully specified filename is the minimum
        iResultDecimal = strcspn(m_filename, "0123456789");
        strncpy(cTmp, m_filename, iResultDecimal);
        cTmp[iResultDecimal] = '\0';

        const char *result;
        char result2[1024] = "";
        result = strpbrk((const char *)m_filename, "0123456789");
        if (result != NULL)
        {
            if (sscanf(result, "%d%s", &iResultDecimal, result2) == 2)
            {
                strcat(cTmp, "%d");
                strcat(cTmp, result2);
                m_iTimestepMin = m_iTimestepMax = iResultDecimal;
            }
            else
            {
                strcpy(cTmp, m_filename);
                m_iTimestepMin = m_iTimestepMax = -1;
            }

            char cTmpSearch[1024];
            if (m_iTimestepMin > -1)
            {
                i = m_iTimestepMin;
                m_iTimestepMax = i;

                sprintf(cTmpSearch, cTmp, i);
                while (fileExists(cTmpSearch))
                {
                    m_iTimestepMax = i;
                    i++;
                    sprintf(cTmpSearch, cTmp, i);
                }
                if (inMapLoading)
                {
                    m_pTimeMax->setValue(m_iTimestepMax);
                    m_pTimeMin->setValue(m_iTimestepMin);
                }
            }
        } /* result != NULL */
    }
    else if (strcmp(name, m_pTimeMax->getName()) == 0)
    {
        if (m_pTimeMax->getValue() <= m_iTimestepMax)
            m_iTimestepMax = m_pTimeMax->getValue();
        //      std::cout << "iTimestepMax = " << m_iTimestepMax << std::endl;
    }
    else if (strcmp(name, m_pTimeMin->getName()) == 0)
    {
        if (m_pTimeMin->getValue() >= m_iTimestepMin)
            m_iTimestepMin = m_pTimeMin->getValue();

        //      std::cout << "iTimestepMin = " << m_iTimestepMin << std::endl;
    }
}

MODULE_MAIN(IO, ReadPDB)
