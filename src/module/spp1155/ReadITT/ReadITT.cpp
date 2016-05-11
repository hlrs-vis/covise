/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2004 ZAIK/RRZK  ++
// ++ Description: ReadITT module                                         ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                       Thomas van Reimersdahl                        ++
// ++               Institute for Computer Science (Prof. Lang)           ++
// ++                        University of Cologne                        ++
// ++                         Robert-Koch-Str. 10                         ++
// ++                             50931 Kln                              ++
// ++                                                                     ++
// ++ The sources for reading the molecule structures are based on        ++
// ++ the VRMoleculeViewer plugin of OpenCOVER.                           ++
// ++                                                                     ++
// ++ Date:  26.12.2004                                                   ++
// ++**********************************************************************/

#include <do/coDoSet.h>
#include <do/coDoData.h>
#include "ReadITT.h"

#include <float.h>
#include <limits.h>
#include <api/coFeedback.h>
#include <util/coMatrix.h>
#include <util/coVector.h>

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

ReadITT::ReadITT(int argc, char *argv[])
    : coModule(argc, argv, "Read ITT's simulation data")
{
    // module parameters
    char *pDataPath = getenv("COVISE_DATA_ITT");

    m_pParamFile = addFileBrowserParam("Filename", "dummy");
    if (pDataPath != NULL)
    {
        m_pParamFile->setValue(pDataPath, "*.via;*.vim;*.vis/*.via/*.vim/*.vis/*");
    }
    else
    {
        m_pParamFile->setValue("./", "*.via;*.vim;*.vis/*.via/*.vim/*.vis/*");
    }

    m_pLookAhead = addBooleanParam("Lookahead", "lookahead");
    m_pLookAhead->setValue(0);

    m_pLookAheadValue = addInt32Param("LookaheadValue", "test");
    m_pLookAheadValue->setValue(0);

    m_pSleepSeconds = addInt32Param("SleepSeconds", "seconds to sleep");
    m_pSleepSeconds->setValue(10);

    // Output ports
    m_portPoints = addOutputPort("points", "Points", "points Output");
    m_portPoints->setInfo("points Output");

    m_portRadii = addOutputPort("radii", "Float", "Atom Radii Output");
    m_portRadii->setInfo("Radii Output");

    m_portColors = addOutputPort("colors", "RGBA", "Atom Colors Output");
    m_portColors->setInfo("Colors Output");

    m_portVolumeBox = addOutputPort("Boundingbox", "Lines", "Bounding Box Output");
    m_portVolumeBox->setInfo("BoundingBox Output");

    m_bDoSelfExec = false;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int ReadITT::compute(const char * /*port*/)
{

    // get parameters
    string m_filename = m_pParamFile->getValue();
    m_bLookAhead = m_pLookAhead->getValue();
    m_iLookAhead = m_pLookAheadValue->getValue();

    // compute parameters
    FILE *pFile;
    if ((pFile = Covise::fopen(m_filename.c_str(), "r")) <= 0)
    {
        sendError("ERROR: can't open file: %s", m_filename.c_str());
        return FAIL;
    }
    fclose(pFile);

    if (!m_bDoSelfExec)
        if (m_bLookAhead == 1)
        {
            loadData(m_filename.c_str());
            m_bDoSelfExec = true;
            return SUCCESS;
        }
        else
        {
            loadData(m_filename.c_str());
        }
    else
    {
        m_bLookAhead = 0;
        loadData(m_filename.c_str());
        m_bDoSelfExec = false;
    }
    return SUCCESS;
}

float ReadITT::idle()
{
    if (m_bDoSelfExec)
    {
        sleep(m_iSleepSeconds);
        selfExec();
    }
    return -1;
}

void ReadITT::enableLookAhead(bool bOn)
{
    m_bLookAhead = bOn;
}

void ReadITT::setLookAhead(int iTimestep)
{
    m_iLookAhead = iTimestep;
}

void ReadITT::loadData(const char *moleculepath)
{
    //char *moleculepath;
    FILE *molecule_fp = fopen(moleculepath, "r");

    if (molecule_fp == NULL)
    {
        sendError("Cannot open file %s", moleculepath);
        return;
    }
    std::cout << "Reading file " << moleculepath << "..." << std::endl;

    // read in the data
    // read the molecule structure first

    structure = new MoleculeStructure(molecule_fp);

    // now we will read in the simulation data
    // we will perform a first scan to determine the
    // number of timesteps and the maximum number of
    // molecules appearing in a timestep

    // getting number of molecule types
    int noOfTypes = structure->getNumberOfMolecules();

    // getting the count of atoms per molecule
    std::vector<int> atomsPerType(noOfTypes);
    for (int i = 0; i < noOfTypes; i++)
    {
        std::vector<int> noOfAtoms;
        if (structure->getMolIDs(i + 1, &noOfAtoms))
            atomsPerType[i] = noOfAtoms.size();
    }
    //printf("noOfTypes: %d, atomsPerType[i]: %d\n", noOfTypes, atomsPerType[0]);

    printf("getting number of timesteps and\nnumber of atoms per timestep\n\n");

    char buf[512];
    int maxAtomCount = 0;
    // vector holding the number of atoms in every timestep
    std::vector<int> anzAtoms;
    while (fgets(buf, LINE_SIZE, molecule_fp) != NULL && ((m_bLookAhead == 0) || (m_bLookAhead == 1 && anzAtoms.size() < m_iLookAhead)))
    {
        // skip blank lines and comments
        if (*buf == '\0')
            // read the next line
            continue;

        // store the data according to the parts...
        if (*buf == '!')
        {
            int type;
            if (sscanf(buf, "%*s %d", &type) == 1)
                // counting number of atoms per timestep
                anzAtoms.back() += atomsPerType[type - 1];
        }
        else if (*buf == '#')
        {
            float tmpf;
            if (sscanf(buf, "%*s %f", &tmpf) == 1)
            {
                // adding a now frame initialised with 0 atoms
                if (!anzAtoms.empty())
                {
                    if (anzAtoms.back() > maxAtomCount)
                        maxAtomCount = anzAtoms.back();
                }
                anzAtoms.push_back(0);
            }
        }
        else
            continue;
    }

    int noOfTimesteps = anzAtoms.size();
    fprintf(stderr, "number of timesteps:\t\t\t%d\n", noOfTimesteps);
    fprintf(stderr, "maximum number of atoms per timestep:\t%d\n\n", maxAtomCount);

    //reset file pointer to beginning
    fseek(molecule_fp, 0, SEEK_SET);

    // getting prefix of output object names
    const char *obj_points = m_portPoints->getObjName();
    const char *obj_lines_name = m_portVolumeBox->getObjName();
    const char *obj_atomcolors = m_portColors->getObjName();
    const char *obj_radii = m_portRadii->getObjName();

    printf("reading timestep data...\n\n");
    float cubeSize = 0.0;
    float *fcoordx = NULL, *fcoordy = NULL, *fcoordz = NULL;
    float *fRadii = NULL;
    int *pc = NULL;

    coDoPoints **dataPoints = new coDoPoints *[noOfTimesteps];
    coDoLines **dataLines = new coDoLines *[noOfTimesteps];
    coDoRGBA **dataColors = new coDoRGBA *[noOfTimesteps];
    coDoFloat **dataRadii = new coDoFloat *[noOfTimesteps];

    int molCount = -1;
    int currentFrame = -1; //quite the same as numberOfTimesteps...
    int currAtom = -1;
    while (fgets(buf, LINE_SIZE, molecule_fp) != NULL && ((m_bLookAhead == 0 && currentFrame < noOfTimesteps) || (m_bLookAhead == 1 && currentFrame < m_iLookAhead)))
    {
        int type;
        float x, y, z;
        float q0, q1, q2, q3;

        // skip blank lines and comments
        if (*buf == '\0')
            continue;
        else if (*buf == '#')
        {
            molCount = -1;
            currAtom = -1;
            ++currentFrame;
            if (sscanf(buf, "%*s %f", &cubeSize) != 1)
            {
                cubeSize = 0.0;
                continue;
            }

            // generate timestep dependent name of actual object
            stringstream obj_points_current;
            obj_points_current << obj_points << "_" << currentFrame;
            stringstream obj_lines_current;
            obj_lines_current << obj_lines_name << "_" << currentFrame;
            stringstream obj_atomcolors_current;
            obj_atomcolors_current << obj_atomcolors << "_" << currentFrame;
            stringstream obj_radii_current;
            obj_radii_current << obj_radii << "_" << currentFrame;

            // generate objects for actual timestep
            dataPoints[currentFrame] = new coDoPoints(obj_points_current.str().c_str(), anzAtoms[currentFrame]);

            dataLines[currentFrame] = createBox(obj_lines_current.str().c_str(),
                                                0, 0, 0, cubeSize, cubeSize, cubeSize);

            dataColors[currentFrame] = new coDoRGBA(obj_atomcolors_current.str().c_str(), anzAtoms[currentFrame]);

            dataRadii[currentFrame] = new coDoFloat(obj_radii_current.str().c_str(), anzAtoms[currentFrame]);

            // get references to internal data structure
            dataPoints[currentFrame]->getAddresses(&fcoordx, &fcoordy, &fcoordz);
            dataRadii[currentFrame]->getAddress(&fRadii);
            dataColors[currentFrame]->getAddress(&pc);

            continue;
        }
        else if (*buf == '!')
        {
            int n = sscanf(buf, "%*s %d %f %f %f %f %f %f %f", &type, &x, &y, &z, &q0, &q1, &q2, &q3);
            //printf("n: %d\n", n);
            if (n == 4)
            {
                x = x / 10.;
                y = y / 10.;
                z = z / 10.;
                q0 = 0.0f;
                q1 = 0.0f;
                q2 = 0.0f;
                q3 = 1000.0f;
            }
            else if (n == 7)
            {
                q0 = q0 / 1000.;
                q1 = q1 / 1000.;
                q2 = q2 / 1000.;
                q3 = sqrt(1.0f - q0 * q0 - q1 * q1 - q2 * q2);
            }
            else if (n == 8)
            {
                q0 = q0 / 1000.;
                q1 = q1 / 1000.;
                q2 = q2 / 1000.;
                q3 = q3 / 1000.;
            }
            else
            {
                printf("badly formed line\n");
                continue;
            }
            ++molCount;
        }
        else
        {
            continue;
        }
        //getting the rotation matrix of quaternion components
        coMatrix quatRot;
        quatRot.fromQuat(q1, q2, q3, q0);

        std::vector<int> liIDgroup;
        structure->getMolIDs(type, &liIDgroup);

        int typeSize = liIDgroup.size();
        //printf("typeSize: %d\n", typeSize);
        for (int j = 0; j < typeSize; ++j)
        {
            ++currAtom;
            //printf("noOfTimesteps: %d,\tmolCount: %d\n", noOfTimesteps, molCount);
            //getting the midpoint of a component of the molecule
            float molX, molY, molZ;
            structure->getXYZ(liIDgroup[j], &molX, &molY, &molZ);

            //Setting point of molecule
            coVector pointOfMolecule(molX, molY, molZ, 1.0f);

            //Transform point of molecule
            pointOfMolecule = pointOfMolecule * (quatRot);

            //Modified point
            float fpos[3];
            pointOfMolecule.get(fpos);

            /*Translating and storing of the component of the molecule
           according to the midpoint of the whole molecule*/
            fcoordx[currAtom] = fpos[0] + (x / 1000. * cubeSize);
            fcoordy[currAtom] = fpos[1] + (y / 1000. * cubeSize);
            fcoordz[currAtom] = fpos[2] + (z / 1000. * cubeSize);

            //Getting radius of the molecule component
            float size;
            structure->getSigma(liIDgroup[j], &size);
            fRadii[currAtom] = size / 2.;

            //Setting the color of the molecule component
            if (structure->getVersion() == 0)
            {
                int color;
                structure->getColor(liIDgroup[j], &color);
                //printf("color: %d,\tliIDgroup[j]: %d\n", color, liIDgroup[j]);
                static const uint32_t lut[16] = {
                    0xddddddff, // grau
                    0x202020ff, // schwarz
                    0xff0000ff, // rot
                    0xff7f00ff, // orange
                    0xffff00ff, // gelb
                    0xb2ff00ff, // zitron
                    0x00ff00ff, // gruen
                    0x00ff7fff, // tuerkis
                    0x00ffffff, // cyan
                    0x007fffff, // wasser
                    0x0000ffff, // blau
                    0x7f00ffff, // lila
                    0xff00ffff, // magenta
                    0xff007fff, // kirsch
                    0xddddddff, // blau
                    0x202020ff, // blau
                };
                pc[currAtom] = lut[color & 0x0f];
            }
            else
            {
                int fColorR, fColorG, fColorB, fColorA;
                structure->getColorRGBA(liIDgroup[j], &fColorR, &fColorG, &fColorB, &fColorA);
                dataColors[currentFrame]->setIntRGBA(currAtom, fColorR, fColorG, fColorB, fColorA);
            }
        }
    }
    fclose(molecule_fp);
    fprintf(stderr, "file operation complete!\n\n");

    fprintf(stderr, "packing and sending data to covise.\n\n");
    coDoSet *setPoints = new coDoSet(m_portPoints->getObjName(), noOfTimesteps, (coDistributedObject **)dataPoints);
    coDoSet *setLines = new coDoSet(m_portVolumeBox->getObjName(), noOfTimesteps, (coDistributedObject **)dataLines);
    coDoSet *setColors = new coDoSet(m_portColors->getObjName(), noOfTimesteps, (coDistributedObject **)dataColors);
    coDoSet *setRadii = new coDoSet(m_portRadii->getObjName(), noOfTimesteps, (coDistributedObject **)dataRadii);

    // Set timestep attribute:
    stringstream timestep;
    timestep << "0 " << noOfTimesteps - 1;
    setPoints->addAttribute("TIMESTEP", timestep.str().c_str());
    setLines->addAttribute("TIMESTEP", timestep.str().c_str());
    setColors->addAttribute("TIMESTEP", timestep.str().c_str());
    setRadii->addAttribute("TIMESTEP", timestep.str().c_str());

    std::string module_id("!");
    module_id += std::string(Covise::get_module()) + std::string("\n") + Covise::get_instance() + std::string("\n");
    module_id += std::string(Covise::get_host()) + std::string("\n");
    setPoints->addAttribute("ITTFEEDBACK", module_id.c_str());

#if 0
   // Create Feedback
   coFeedback feedback("FileBrowserParam");
   feedback.addPara(m_pParamFile);
   feedback.apply(setPoints);
#endif
    // Assign sets to output ports:
    m_portPoints->setCurrentObject(setPoints);
    m_portVolumeBox->setCurrentObject(setLines);
    m_portColors->setCurrentObject(setColors);
    m_portRadii->setCurrentObject(setRadii);

    fprintf(stderr, "ready!");
    delete structure;
}

void ReadITT::param(const char *name, bool /*inMapLoading*/)
{
    if (strcmp(name, m_pSleepSeconds->getName()) == 0)
    {
        m_iSleepSeconds = m_pSleepSeconds->getValue();
    }
}

coDoLines *ReadITT::createBox(const char *objectName,
                              float ox, float oy, float oz, float size_x, float size_y, float size_z)
{
    coDoLines *linesObj;
    float xCoords[8], yCoords[8], zCoords[8]; // the box coordinates
    int vertexList[24] = // the index list
        {
          0, 3, 7, 4, 3, 2, 6, 7, 0, 1, 2, 3, 0, 4, 5, 1, 1, 5, 6, 2, 7, 6, 5, 4
        };
    int linesList[6] = { 0, 4, 8, 12, 16, 20 };

    // compute the vertex coordinates

    //      5.......6
    //    .       . .
    //  4.......7   .  z
    //  .       .   .
    //  .   1   .   2
    //  .       . .  y
    //  0.......3
    //      x

    xCoords[0] = ox;
    yCoords[0] = oy;
    zCoords[0] = oz;

    xCoords[1] = ox;
    yCoords[1] = oy + size_y;
    zCoords[1] = oz;

    xCoords[2] = ox + size_x;
    yCoords[2] = oy + size_y;
    zCoords[2] = oz;

    xCoords[3] = ox + size_x;
    yCoords[3] = oy;
    zCoords[3] = oz;

    xCoords[4] = ox;
    yCoords[4] = oy;
    zCoords[4] = oz + size_z;

    xCoords[5] = ox;
    yCoords[5] = oy + size_y;
    zCoords[5] = oz + size_z;

    xCoords[6] = ox + size_x;
    yCoords[6] = oy + size_y;
    zCoords[6] = oz + size_z;

    xCoords[7] = ox + size_x;
    yCoords[7] = oy;
    zCoords[7] = oz + size_z;

    // create the lines data object
    linesObj = new coDoLines(objectName, 8,
                             xCoords, yCoords, zCoords,
                             24, vertexList,
                             6, linesList);

    return (linesObj);
}

MODULE_MAIN(IO, ReadITT)
