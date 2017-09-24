/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1998 RUS  **
 **                                                     (C) 2000 VirCinity **
 ** Description: Read module for Nastran data                              **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Franz Maurer                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30a                             **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  07.04.1998  V1.0                                                **
 ** Data:  08.03.2001  Sven Kufer: corrected transformation matrix         **
\**************************************************************************/

#include "ReadNastran.h"
#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>
#include <algorithm>
// some useful defines
// -------------------
#ifdef M_PI
#define PI M_PI
#else
#define PI 3.14159265358979323846
#endif

#undef DEEPVERBOSE
#undef VERBOSE

#define POS(a) ((a) > 0 ? (a) : 0)

#ifdef __MINGW32__
#define min(X, Y) ((X) < (Y) ? (X) : (Y))
#endif

// --------------
// some constants
// --------------
// the type of the coordinate system
const int RECTANGULAR = 1;
const int CYLINDRICAL = 2;
const int SPHERICAL = 3;

// the default temperature for a temperature record
const float NULL_TEMPERATURE = 293.15f;

// the degrees of freedom
const int TRANSLATION_X = 1;
const int TRANSLATION_Y = 2;
const int TRANSLATION_Z = 3;
const int ROTATION_X = 4;
const int ROTATION_Y = 5;
const int ROTATION_Z = 6;

// some inliners
// -------------
// radians to degrees
// inline double r2d(double radians) {return(radians * 180.0f / PI);};
// degrees to radians
inline double d2r(double degrees) { return (degrees * PI / 180.0f); };
// calc the square
inline double square(double val) { return val * val; };
// normalize a vector
inline void normalize(float *pin, float *pout)
{
    float l = sqrt(pin[0] * pin[0] + pin[1] * pin[1] + pin[2] * pin[2]);
    pout[0] = pin[0] / l;
    pout[1] = pin[1] / l;
    pout[2] = pin[2] / l;
};

//-------------------------------------------------------------------------
//
// constructor
//
//-------------------------------------------------------------------------
ReadNastran::ReadNastran(int argc, char *argv[])
    : coModule(argc, argv, "Read NASTRAN output2 files")
    , runCnt_(0)
    , block(NULL)
{
    char *cov_path = getenv("COVISEDIR");
    if (cov_path)
        sprintf(buf, "%s/data/ *.op2", cov_path);
    else
        sprintf(buf, "./ *.op2");

    // input
    output2Path = addFileBrowserParam("output2_path", "NASTRAN output2 file path");
    output2Path->setValue(buf, "*.op2");
    plotelColor = addStringParam("plotel_color", "colorname for PLOTEL");
    plotelColor->setValue("orange");
    conm2Color = addStringParam("conm2_color", "colorname for CONM2");
    conm2Color->setValue("purple");
    conm2Scale = addFloatParam("conm2_scale", "CONM2 scaling factor");
    conm2Scale->setValue(0.2f);
    forceColor = addStringParam("force_color", "colorname for FORCE");
    forceColor->setValue("red");
    gravColor = addStringParam("grav_color", "colorname for GRAV");
    gravColor->setValue("green");
    momentColor = addStringParam("moment_color", "colorname for MOMENT");
    momentColor->setValue("blue");
    rbarColor = addStringParam("rbar_color", "colorname for RBAR");
    rbarColor->setValue("cyan");
    rbe2Color = addStringParam("rbe2_color", "colorname for RBE2");
    rbe2Color->setValue("magenta");
    spc1Color = addStringParam("spc1_color", "colorname for SPC1");
    spc1Color->setValue("yellow");
    spc1Scale = addFloatParam("spc1_scale", "SPC1 scaling factor");
    spc1Scale->setValue(1.0);
    modeParam = addInt32Param("mode", "mode number");
    modeParam->setValue(1);
    fibreDistanceParam = addInt32Param("fibre_distance", "fibre distance");
    fibreDistanceParam->setValue(1);
    trySkipping = addBooleanParam("try_skipping", "try");
    trySkipping->setValue(false);
    dispTransient = addBooleanParam("disp_transient", "transient");
    dispTransient->setValue(false);

    // output
    meshOut = addOutputPort("MESH", "UnstructuredGrid", "Mesh output");
    typeOut = addOutputPort("TYPE", "IntArr", "IDs");
    plotelOut = addOutputPort("PLOTEL", "Lines", "PLOTEL output");
    conm2Out = addOutputPort("CONM2", "Polygons", "CONM2 output");
    forceOut = addOutputPort("FORCE", "Lines", "FORCE output");
    momentOut = addOutputPort("MOMENT", "Lines", "MOMENT output");
    gravOut = addOutputPort("GRAV", "Lines", "GRAV output");
    tempOut = addOutputPort("TEMP", "Float", "TEMP output");
    rbarOut = addOutputPort("RBAR", "Lines", "RBAR output");
    rbe2Out = addOutputPort("RBE2", "Lines", "RBE2 output");
    spc1Out = addOutputPort("SPC1", "Lines", "SPC output");
    oqg1Out = addOutputPort("OQG1", "Vec3", "reaction forces");
    ougv1Out = addOutputPort("OUGV1", "Vec3", "displacements");
    oef1Out = addOutputPort("OEF1", "Vec3", "element forces");
    oes1Out = addOutputPort("OES1", "Float", "element stress");

    // initialization
    op2Path[0] = '0';
    charSize = 1;
    intSize = 4;
    floatSize = 4;
    doubleSize = 8;
    wordSize = intSize;

    numDisplacements = 0;
    numDisplacementSets = 0;

    numStresses = 0;
    numStressSets = 0;

    fibreDistance = 1;
    byte_swapped = false;
}

//-------------------------------------------------------------------------
//
// destructor
//
//-------------------------------------------------------------------------
ReadNastran::~ReadNastran()
{
    // just die!
    freeResources();
}

//-------------------------------------------------------------------------
//
// compute the module
//
//-------------------------------------------------------------------------
int ReadNastran::compute(const char *)
{
    const char *output2_path, *colorname;
    int *clPtr, *elPtr;
    float *xPtr, *yPtr, *zPtr;
    int i;
    long mode = 1;

    runCnt_++;

    try_skipping = trySkipping->getValue();
    //
    // there are two possible fibre distances: 1 and 2
    // with different element stresses
    //
    // retrieve the fibre distance parameter
    fibreDistance = fibreDistanceParam->getValue();
    // check input
    if (fibreDistance > 2)
        fibreDistance = 2;
    if (fibreDistance < 1)
    {
        fibreDistance = 1;
    }
#ifdef VERBOSE
    fprintf(stdout, "fibre distance = %ld\n", fibreDistance);
#endif

    output2_path = output2Path->getValue();
    if (strcmp(op2Path, output2_path) != 0)
    {
        freeResources();
        cleanup();
        load(output2_path);
        strcpy(op2Path, output2_path);
    }

#ifdef VERBOSE
    // print out some statistics
    fprintf(stdout, "-------------------------------------------------\n");
    fprintf(stdout, "NASTRAN output2 statistics\n");
    fprintf(stdout, "-------------------------------------------------\n");
    fprintf(stdout, "Got %d grid points!\n", gridID.size());
    fprintf(stdout, "Got %d grid connections!\n", connectionList.size());
    fprintf(stdout, "Got %d grid elements!\n", elementList.size());
    fprintf(stdout, "Got %d grid element types!\n", typeList.size());
    fprintf(stdout, "Got %d grid element ids!\n", elementID.size());
    fprintf(stdout, "Got %d PLOTEL elements!\n", plotelList.size());
    fprintf(stdout, "Got %d CONM2 elements!\n", conm2List.size());
    fprintf(stdout, "Got %d FORCE elements!\n", forceList.size());
    fprintf(stdout, "Got %d GRAV elements!\n", gravList.size());
    fprintf(stdout, "Got %d MOMENT elements!\n", momentList.size());
    fprintf(stdout, "Got %d RBAR elements!\n", rbarList.size());
    fprintf(stdout, "Got %d RBE2 elements!\n", rbe2List.size());
    fprintf(stdout, "Got %d SPC1 elements!\n", spc1Trans.size() + spc1Rot.size());
    fprintf(stdout, "Got %d OUGV displacements!\n", dispX.size());
    fprintf(stdout, "Got %d OUGV displacement modes!\n", numDisplacementSets);
    //   elementID.print("elementIDs");
    //   propertyID.print("properties");
    fprintf(stdout, "-------------------------------------------------\n");
#endif

    // anyway, put some informations in the message area,
    // so that the user gets an impression which NASTRAN cards
    // are present
    if (plotelList.size())
        sendInfo("PLOTEL card detected.");
    if (conm2List.size())
        sendInfo("CONM2 card detected.");
    if (forceList.size())
        sendInfo("FORCE card detected.");
    if (gravList.size())
        sendInfo("GRAV card detected.");
    if (momentList.size())
        sendInfo("MOMENT card detected.");
    if (rbarList.size())
        sendInfo("RBAR card detected.");
    if (rbe2List.size())
        sendInfo("RBE2 card detected.");
    if (spc1Trans.size() + spc1Rot.size())
        sendInfo("SPC1 card detected.");
    if (numDisplacementSets)
    {
        sendInfo("%d mode(s) detected.", numDisplacementSets);
    }
    if (numDisplacements)
    {
        sendInfo("Displacements OUGV1 detected.");
    }
    if (numStresses)
    {
        sendInfo("Stresses OES1 detected.");
    }

    // --------
    // the mesh
    // --------
    if (elementList.size())
    {
        coDoUnstructuredGrid *mesh = new coDoUnstructuredGrid(meshOut->getNewObjectInfo(),
                                                              elementList.size(),
                                                              connectionList.size(),
                                                              gridID.size(),
                                                              elementList.getDataPtr(),
                                                              connectionList.getDataPtr(),
                                                              gridX.getDataPtr(),
                                                              gridY.getDataPtr(),
                                                              gridZ.getDataPtr(),
                                                              typeList.getDataPtr());

        if (!mesh->objectOk())
        {
            sendError("ERROR: creation of data object 'mesh' failed");
            return STOP_PIPELINE;
        }

        delete mesh;
    }

    // --------
    // the id's
    // --------
    if (elementID.size() || propertyID.size())
    {
        int size[2];
        size[0] = elementList.size();
        size[1] = 3; // [element id, property id, component id]

        int *id;
        coDoIntArr *type = new coDoIntArr(typeOut->getNewObjectInfo(), 2, size);

        if (!type->objectOk())
        {
            sendError("ERROR: creation of data object 'type' failed");
            return STOP_PIPELINE;
        }
        type->getAddress(&id);

        // set ids to -1
        memset((void *)id, -1, 3 * elementList.size() * sizeof(int));

        // copy element ids
        for (i = 0; i < elementID.size(); i++)
        {
            id[i] = elementID[i];
        }
        // copy property ids
        for (i = 0; i < propertyID.size(); i++)
        {
            id[elementList.size() + i] = propertyID[i];
        }
        // set element type ids -> will be displayed through the
        // component selection in SelectUSG (Covise version 4.5)
        // 1 CBAR
        // 2 CTRIA3
        // 3 CQUAD4
        // 4 CPENTA
        // 5 CHEXA
        // 6 CTETRA
        for (i = 0; i < typeList.size(); i++)
        {
            if (typeList[i] == TYPE_PRISM)
            {
                id[2 * elementList.size() + i] = 4;
            }
            else if (typeList[i] == TYPE_HEXAEDER)
            {
                id[2 * elementList.size() + i] = 5;
            }
            else if (typeList[i] == TYPE_TETRAHEDER)
            {
                id[2 * elementList.size() + i] = 6;
            }
            else
            {
                // rest matches: TYPE_BAR=1, TYPE_TRIANGLE=2, TYPE_QUAD=3
                id[2 * elementList.size() + i] = typeList[i];
            }
        }

        delete type;
    }

    // ----------
    // the plotel
    // ----------
    if (plotelList.size())
    {
        coDoLines *plotel = new coDoLines(plotelOut->getNewObjectInfo(),
                                          plotelList.size() * 2,
                                          plotelList.size() * 2,
                                          plotelList.size());

        if (!plotel->objectOk())
        {
            sendError("ERROR: creation of data object 'plotel' failed");
            return STOP_PIPELINE;
        }

        plotel->getAddresses(&xPtr, &yPtr, &zPtr, &clPtr, &elPtr);

        for (i = 0; i < plotelList.size(); i++)
        {
            int idx1 = gridID.findBinary(plotelList[i]->g[0]);
            int idx2 = gridID.findBinary(plotelList[i]->g[1]);
            xPtr[2 * i] = gridX[idx1];
            xPtr[2 * i + 1] = gridX[idx2];
            yPtr[2 * i] = gridY[idx1];
            yPtr[2 * i + 1] = gridY[idx2];
            zPtr[2 * i] = gridZ[idx1];
            zPtr[2 * i + 1] = gridZ[idx2];

            clPtr[2 * i] = 2 * i;
            clPtr[2 * i + 1] = 2 * i + 1;

            elPtr[i] = 2 * i;
        }

        colorname = plotelColor->getValue();
        if ((strcmp(colorname, "none") != 0) && (colorname[0] != '\0'))
            plotel->addAttribute("COLOR", colorname);

        delete plotel;
    }

    // ---------
    // the conm2
    // ---------
    if (conm2List.size())
    {
        coDoPolygons *conm2 = new coDoPolygons(conm2Out->getNewObjectInfo(),
                                               conm2List.size() * 8,
                                               conm2List.size() * 24,
                                               conm2List.size() * 6);

        if (!conm2->objectOk())
        {
            sendError("ERROR: creation of data object 'conm2' failed");
            return STOP_PIPELINE;
        }

        conm2->getAddresses(&xPtr, &yPtr, &zPtr, &clPtr, &elPtr);

        float conm2_scale_factor = conm2Scale->getValue();

        for (i = 0; i < conm2List.size(); i++)
        {

            float x = gridX[conm2List[i]];
            float y = gridY[conm2List[i]];
            float z = gridZ[conm2List[i]];

#ifdef DEEPVERBOSE
            fprintf(stdout, "CONM2 at [%f %f %f]\n", x, y, z);
#endif

            xPtr[8 * i] = x - conm2_scale_factor * 1.0f;
            yPtr[8 * i] = y + conm2_scale_factor * 1.0f;
            zPtr[8 * i] = z + conm2_scale_factor * 1.0f;
            xPtr[8 * i + 1] = x - conm2_scale_factor * 1.0f;
            yPtr[8 * i + 1] = y + conm2_scale_factor * 1.0f;
            zPtr[8 * i + 1] = z - conm2_scale_factor * 1.0f;
            xPtr[8 * i + 2] = x + conm2_scale_factor * 1.0f;
            yPtr[8 * i + 2] = y + conm2_scale_factor * 1.0f;
            zPtr[8 * i + 2] = z - conm2_scale_factor * 1.0f;
            xPtr[8 * i + 3] = x + conm2_scale_factor * 1.0f;
            yPtr[8 * i + 3] = y + conm2_scale_factor * 1.0f;
            zPtr[8 * i + 3] = z + conm2_scale_factor * 1.0f;
            xPtr[8 * i + 4] = x - conm2_scale_factor * 1.0f;
            yPtr[8 * i + 4] = y - conm2_scale_factor * 1.0f;
            zPtr[8 * i + 4] = z + conm2_scale_factor * 1.0f;
            xPtr[8 * i + 5] = x - conm2_scale_factor * 1.0f;
            yPtr[8 * i + 5] = y - conm2_scale_factor * 1.0f;
            zPtr[8 * i + 5] = z - conm2_scale_factor * 1.0f;
            xPtr[8 * i + 6] = x + conm2_scale_factor * 1.0f;
            yPtr[8 * i + 6] = y - conm2_scale_factor * 1.0f;
            zPtr[8 * i + 6] = z - conm2_scale_factor * 1.0f;
            xPtr[8 * i + 7] = x + conm2_scale_factor * 1.0f;
            yPtr[8 * i + 7] = y - conm2_scale_factor * 1.0f;
            zPtr[8 * i + 7] = z + conm2_scale_factor * 1.0f;

            clPtr[24 * i] = i * 8 + 3;
            clPtr[24 * i + 1] = i * 8 + 2;
            clPtr[24 * i + 2] = i * 8 + 1;
            clPtr[24 * i + 3] = i * 8 + 0;
            clPtr[24 * i + 4] = i * 8 + 7;
            clPtr[24 * i + 5] = i * 8 + 6;
            clPtr[24 * i + 6] = i * 8 + 2;
            clPtr[24 * i + 7] = i * 8 + 3;
            clPtr[24 * i + 8] = i * 8 + 4;
            clPtr[24 * i + 9] = i * 8 + 5;
            clPtr[24 * i + 10] = i * 8 + 6;
            clPtr[24 * i + 11] = i * 8 + 7;
            clPtr[24 * i + 12] = i * 8 + 0;
            clPtr[24 * i + 13] = i * 8 + 1;
            clPtr[24 * i + 14] = i * 8 + 5;
            clPtr[24 * i + 15] = i * 8 + 4;
            clPtr[24 * i + 16] = i * 8 + 7;
            clPtr[24 * i + 17] = i * 8 + 3;
            clPtr[24 * i + 18] = i * 8 + 0;
            clPtr[24 * i + 19] = i * 8 + 4;
            clPtr[24 * i + 20] = i * 8 + 5;
            clPtr[24 * i + 21] = i * 8 + 1;
            clPtr[24 * i + 22] = i * 8 + 2;
            clPtr[24 * i + 23] = i * 8 + 6;

            elPtr[6 * i] = i * 24;
            elPtr[6 * i + 1] = i * 24 + 4;
            elPtr[6 * i + 2] = i * 24 + 8;
            elPtr[6 * i + 3] = i * 24 + 12;
            elPtr[6 * i + 4] = i * 24 + 16;
            elPtr[6 * i + 5] = i * 24 + 20;
        }

        colorname = conm2Color->getValue();
        if ((strcmp(colorname, "none") != 0) && (colorname[0] != '\0'))
            conm2->addAttribute("COLOR", colorname);

        delete conm2;
    }

    // ---------
    // the force
    // ---------
    if (forceList.size())
    {
        coDoLines *force = new coDoLines(forceOut->getNewObjectInfo(),
                                         forceList.size() * 2,
                                         forceList.size() * 2,
                                         forceList.size());

        if (!force->objectOk())
        {
            sendError("ERROR: creation of data object 'force' failed");
            return STOP_PIPELINE;
        }

        force->getAddresses(&xPtr, &yPtr, &zPtr, &clPtr, &elPtr);

        for (i = 0; i < forceList.size(); i++)
        {
            float p[3];
            int idx = gridID.findBinary(forceList[i]->gid);
            normalize(forceList[i]->n, p);

            xPtr[2 * i] = gridX[idx];
            xPtr[2 * i + 1] = gridX[idx] + p[0];
            yPtr[2 * i] = gridY[idx];
            yPtr[2 * i + 1] = gridY[idx] + p[1];
            zPtr[2 * i] = gridZ[idx];
            zPtr[2 * i + 1] = gridZ[idx] + p[2];

            clPtr[2 * i] = 2 * i;
            clPtr[2 * i + 1] = 2 * i + 1;

            elPtr[i] = 2 * i;
        }

        colorname = forceColor->getValue();
        if ((strcmp(colorname, "none") != 0) && (colorname[0] != '\0'))
            force->addAttribute("COLOR", colorname);

        delete force;
    }

    // --------
    // the grav
    // --------
    if (gravList.size())
    {
        coDoLines *grav = new coDoLines(gravOut->getNewObjectInfo(),
                                        gravList.size() * 2,
                                        gravList.size() * 2,
                                        gravList.size());

        if (!grav->objectOk())
        {
            sendError("ERROR: creation of data object 'grav' failed");
            return STOP_PIPELINE;
        }

        grav->getAddresses(&xPtr, &yPtr, &zPtr, &clPtr, &elPtr);

        for (i = 0; i < gravList.size(); i++)
        {
            float p[3];
            normalize(gravList[i]->n, p);

            xPtr[2 * i] = 0.0;
            xPtr[2 * i + 1] = p[0];
            yPtr[2 * i] = 0.0;
            yPtr[2 * i + 1] = p[1];
            zPtr[2 * i] = 0.0;
            zPtr[2 * i + 1] = p[2];

            clPtr[2 * i] = 2 * i;
            clPtr[2 * i + 1] = 2 * i + 1;

            elPtr[i] = 2 * i;
        }

        colorname = gravColor->getValue();
        if ((strcmp(colorname, "none") != 0) && (colorname[0] != '\0'))
            grav->addAttribute("COLOR", colorname);

        delete grav;
    }

    // ----------
    // the moment
    // ----------
    if (momentList.size())
    {
        coDoLines *moment = new coDoLines(momentOut->getNewObjectInfo(),
                                          momentList.size() * 2,
                                          momentList.size() * 2,
                                          momentList.size());

        if (!moment->objectOk())
        {
            sendError("ERROR: creation of data object 'moment' failed");
            return STOP_PIPELINE;
        }

        moment->getAddresses(&xPtr, &yPtr, &zPtr, &clPtr, &elPtr);

        for (i = 0; i < momentList.size(); i++)
        {
            float p[3];
            int idx = gridID.findBinary(momentList[i]->gid);
            normalize(momentList[i]->n, p);

            xPtr[2 * i] = gridX[idx];
            xPtr[2 * i + 1] = gridX[idx] + p[0];
            yPtr[2 * i] = gridY[idx];
            yPtr[2 * i + 1] = gridY[idx] + p[1];
            zPtr[2 * i] = gridZ[idx];
            zPtr[2 * i + 1] = gridZ[idx] + p[2];

            clPtr[2 * i] = 2 * i;
            clPtr[2 * i + 1] = 2 * i + 1;

            elPtr[i] = 2 * i;
        }

        colorname = momentColor->getValue();
        if ((strcmp(colorname, "none") != 0) && (colorname[0] != '\0'))
            moment->addAttribute("COLOR", colorname);

        delete moment;
    }

    // --------
    // the temp
    // --------
    if (tempList.size())
    {
        coDoFloat *temp = new coDoFloat(tempOut->getNewObjectInfo(),
                                        tempList.size(),
                                        tempList.getDataPtr());

        if (!temp->objectOk())
        {
            sendError("ERROR: creation of data object 'temp' failed");
            return STOP_PIPELINE;
        }

        delete temp;
    }

    // --------
    // the rbar
    // --------
    if (rbarList.size())
    {
        coDoLines *rbar = new coDoLines(rbarOut->getNewObjectInfo(),
                                        rbarList.size() * 2,
                                        rbarList.size() * 2,
                                        rbarList.size());

        if (!rbar->objectOk())
        {
            sendError("ERROR: creation of data object 'rbar' failed");
            return STOP_PIPELINE;
        }

        rbar->getAddresses(&xPtr, &yPtr, &zPtr, &clPtr, &elPtr);

        for (i = 0; i < rbarList.size(); i++)
        {
            int idx1 = gridID.findBinary(rbarList[i]->g[0]);
            int idx2 = gridID.findBinary(rbarList[i]->g[1]);
            if (idx1 == -1 || idx2 == -1)
                fprintf(stdout, "ERROR in RBAR: Can't find grid index!!!\n");
            xPtr[2 * i] = gridX[idx1];
            xPtr[2 * i + 1] = gridX[idx2];
            yPtr[2 * i] = gridY[idx1];
            yPtr[2 * i + 1] = gridY[idx2];
            zPtr[2 * i] = gridZ[idx1];
            zPtr[2 * i + 1] = gridZ[idx2];

            clPtr[2 * i] = 2 * i;
            clPtr[2 * i + 1] = 2 * i + 1;

            elPtr[i] = 2 * i;
        }

        colorname = rbarColor->getValue();
        if ((strcmp(colorname, "none") != 0) && (colorname[0] != '\0'))
            rbar->addAttribute("COLOR", colorname);

        delete rbar;
    }

    // --------
    // the rbe2
    // --------
    if (rbe2List.size())
    {
        coDoLines *rbe2 = new coDoLines(rbe2Out->getNewObjectInfo(),
                                        rbe2List.size() * 2,
                                        rbe2List.size() * 2,
                                        rbe2List.size());

        if (!rbe2->objectOk())
        {
            sendError("ERROR: creation of data object 'rbe2' failed");
            return STOP_PIPELINE;
        }

        rbe2->getAddresses(&xPtr, &yPtr, &zPtr, &clPtr, &elPtr);

        for (i = 0; i < rbe2List.size(); i++)
        {
            int idx1 = gridID.findBinary(rbe2List[i]->g[0]);
            int idx2 = gridID.findBinary(rbe2List[i]->g[1]);
            if (idx1 == -1 || idx2 == -1)
            {
                fprintf(stdout, "ERROR in RBE2: Can't find grid index!!!\n");
                if (idx1 == -1)
                    fprintf(stdout, "Can't find grid ID %d (index1)\n", rbe2List[i]->g[0]);
                else
                    fprintf(stdout, "Can't find grid ID %d (index2)\n", rbe2List[i]->g[1]);
                xPtr[2 * i] = 0.0f;
                xPtr[2 * i + 1] = 0.0f;
                yPtr[2 * i] = 0.0f;
                yPtr[2 * i + 1] = 0.0f;
                zPtr[2 * i] = 0.0f;
                zPtr[2 * i + 1] = 0.0f;
            }
            else
            {
                xPtr[2 * i] = gridX[idx1];
                xPtr[2 * i + 1] = gridX[idx2];
                yPtr[2 * i] = gridY[idx1];
                yPtr[2 * i + 1] = gridY[idx2];
                zPtr[2 * i] = gridZ[idx1];
                zPtr[2 * i + 1] = gridZ[idx2];
            }

            clPtr[2 * i] = 2 * i;
            clPtr[2 * i + 1] = 2 * i + 1;

            elPtr[i] = 2 * i;
        }

        colorname = rbe2Color->getValue();
        if ((strcmp(colorname, "none") != 0) && (colorname[0] != '\0'))
            rbe2->addAttribute("COLOR", colorname);

        delete rbe2;
    }

    // --------
    // the spc1
    // --------
    if (spc1Trans.size() || spc1Rot.size())
    {
        coDoSet *spc1 = new coDoSet(spc1Out->getNewObjectInfo(), SET_CREATE);

        if (!spc1->objectOk())
        {
            sendError("ERROR: creation of data object 'spc1' failed");
            return STOP_PIPELINE;
        }

        float spc1_scale_factor = spc1Scale->getValue();

        if (spc1Trans.size())
        {
            // create a unique name for temp. DO_.. obj.

            runCnt_ = runCnt_ % 365;
            sprintf(buf, "TMP%s%s%dTRANS", Covise::get_module(), Covise::get_instance(), runCnt_);
            char *tmpObjName = new char[1 + strlen(buf)];
            strcpy(tmpObjName, buf);
            // the translation dof's
            coDoLines *spc1_trans = new coDoLines(tmpObjName,
                                                  spc1Trans.size() * 4,
                                                  spc1Trans.size() * 6,
                                                  spc1Trans.size() * 3);

            if (!spc1_trans->objectOk())
            {
                sendError("ERROR: creation of data object spc1_trans failed");
                return STOP_PIPELINE;
            }

            spc1_trans->getAddresses(&xPtr, &yPtr, &zPtr, &clPtr, &elPtr);

            for (i = 0; i < spc1Trans.size(); i++)
            {
                int idx = gridID.findBinary(spc1Trans[i]->gid);
                switch (spc1Trans[i]->dof)
                {
                case TRANSLATION_X:
                    xPtr[4 * i] = gridX[idx];
                    yPtr[4 * i] = gridY[idx];
                    zPtr[4 * i] = gridZ[idx];
                    xPtr[4 * i + 1] = gridX[idx] + spc1_scale_factor * 1.0f;
                    yPtr[4 * i + 1] = gridY[idx];
                    zPtr[4 * i + 1] = gridZ[idx];
                    xPtr[4 * i + 2] = gridX[idx] + spc1_scale_factor * 0.8f;
                    yPtr[4 * i + 2] = gridY[idx] + spc1_scale_factor * 0.2f;
                    zPtr[4 * i + 2] = gridZ[idx];
                    xPtr[4 * i + 3] = gridX[idx] + spc1_scale_factor * 0.8f;
                    yPtr[4 * i + 3] = gridY[idx] - spc1_scale_factor * 0.2f;
                    zPtr[4 * i + 3] = gridZ[idx];
                    break;
                case TRANSLATION_Y:
                    xPtr[4 * i] = gridX[idx];
                    yPtr[4 * i] = gridY[idx];
                    zPtr[4 * i] = gridZ[idx];
                    xPtr[4 * i + 1] = gridX[idx];
                    yPtr[4 * i + 1] = gridY[idx] + spc1_scale_factor * 1.0f;
                    zPtr[4 * i + 1] = gridZ[idx];
                    xPtr[4 * i + 2] = gridX[idx] - spc1_scale_factor * 0.2f;
                    yPtr[4 * i + 2] = gridY[idx] + spc1_scale_factor * 0.8f;
                    zPtr[4 * i + 2] = gridZ[idx];
                    xPtr[4 * i + 3] = gridX[idx] + spc1_scale_factor * 0.2f;
                    yPtr[4 * i + 3] = gridY[idx] + spc1_scale_factor * 0.8f;
                    zPtr[4 * i + 3] = gridZ[idx];
                    break;
                case TRANSLATION_Z:
                    xPtr[4 * i] = gridX[idx];
                    yPtr[4 * i] = gridY[idx];
                    zPtr[4 * i] = gridZ[idx];
                    xPtr[4 * i + 1] = gridX[idx];
                    yPtr[4 * i + 1] = gridY[idx];
                    zPtr[4 * i + 1] = gridZ[idx] + spc1_scale_factor * 1.0f;
                    xPtr[4 * i + 2] = gridX[idx] + spc1_scale_factor * 0.2f;
                    yPtr[4 * i + 2] = gridY[idx];
                    zPtr[4 * i + 2] = gridZ[idx] + spc1_scale_factor * 0.8f;
                    xPtr[4 * i + 3] = gridX[idx] - spc1_scale_factor * 0.2f;
                    yPtr[4 * i + 3] = gridY[idx];
                    zPtr[4 * i + 3] = gridZ[idx] + spc1_scale_factor * 0.8f;
                    break;
                }

                clPtr[6 * i] = 4 * i;
                clPtr[6 * i + 1] = 4 * i + 1;
                clPtr[6 * i + 2] = 4 * i + 1;
                clPtr[6 * i + 3] = 4 * i + 2;
                clPtr[6 * i + 4] = 4 * i + 1;
                clPtr[6 * i + 5] = 4 * i + 3;

                elPtr[3 * i] = 6 * i;
                elPtr[3 * i + 1] = 6 * i + 2;
                elPtr[3 * i + 2] = 6 * i + 4;
            }

            colorname = spc1Color->getValue();
            if ((strcmp(colorname, "none") != 0) && (colorname[0] != '\0'))
            {
                spc1_trans->addAttribute("COLOR", colorname);
            }

            spc1->addElement(spc1_trans);

            delete spc1_trans;
            delete[] tmpObjName;
        }

        if (spc1Rot.size())
        {
            // create a unique name for temp. DO_.. obj.

            runCnt_ = runCnt_ % 365;
            sprintf(buf, "TMP%s%s%dROT", Covise::get_module(), Covise::get_instance(), runCnt_);
            char *tmpObjName = new char[1 + strlen(buf)];
            strcpy(tmpObjName, buf);
            // the rotation dof's
            coDoLines *spc1_rot = new coDoLines(tmpObjName,
                                                spc1Rot.size() * 7,
                                                spc1Rot.size() * 10,
                                                spc1Rot.size() * 5);

            if (!spc1_rot->objectOk())
            {
                sendError("ERROR: creation of data object spc1_rot failed");
                return STOP_PIPELINE;
            }

            spc1_rot->getAddresses(&xPtr, &yPtr, &zPtr, &clPtr, &elPtr);

            for (i = 0; i < spc1Rot.size(); i++)
            {
                int idx = gridID.findBinary(spc1Rot[i]->gid);
                switch (spc1Rot[i]->dof)
                {
                case ROTATION_X:
                    xPtr[7 * i] = gridX[idx];
                    yPtr[7 * i] = gridY[idx];
                    zPtr[7 * i] = gridZ[idx];
                    xPtr[7 * i + 1] = gridX[idx] + spc1_scale_factor * 0.9f;
                    yPtr[7 * i + 1] = gridY[idx];
                    zPtr[7 * i + 1] = gridZ[idx];
                    xPtr[7 * i + 2] = gridX[idx] + spc1_scale_factor * 1.1f;
                    yPtr[7 * i + 2] = gridY[idx];
                    zPtr[7 * i + 2] = gridZ[idx];
                    xPtr[7 * i + 3] = gridX[idx] + spc1_scale_factor * 0.9f;
                    yPtr[7 * i + 3] = gridY[idx] + spc1_scale_factor * 0.2f;
                    zPtr[7 * i + 3] = gridZ[idx];
                    xPtr[7 * i + 4] = gridX[idx] + spc1_scale_factor * 0.9f;
                    yPtr[7 * i + 4] = gridY[idx] - spc1_scale_factor * 0.2f;
                    zPtr[7 * i + 4] = gridZ[idx];
                    xPtr[7 * i + 5] = gridX[idx] + spc1_scale_factor * 0.7f;
                    yPtr[7 * i + 5] = gridY[idx] + spc1_scale_factor * 0.2f;
                    zPtr[7 * i + 5] = gridZ[idx];
                    xPtr[7 * i + 6] = gridX[idx] + spc1_scale_factor * 0.7f;
                    yPtr[7 * i + 6] = gridY[idx] - spc1_scale_factor * 0.2f;
                    zPtr[7 * i + 6] = gridZ[idx];
                    break;
                case ROTATION_Y:
                    xPtr[7 * i] = gridX[idx];
                    yPtr[7 * i] = gridY[idx];
                    zPtr[7 * i] = gridZ[idx];
                    xPtr[7 * i + 1] = gridX[idx];
                    yPtr[7 * i + 1] = gridY[idx] + spc1_scale_factor * 0.9f;
                    zPtr[7 * i + 1] = gridZ[idx];
                    xPtr[7 * i + 2] = gridX[idx];
                    yPtr[7 * i + 2] = gridY[idx] + spc1_scale_factor * 1.1f;
                    zPtr[7 * i + 2] = gridZ[idx];
                    xPtr[7 * i + 3] = gridX[idx] - spc1_scale_factor * 0.2f;
                    yPtr[7 * i + 3] = gridY[idx] + spc1_scale_factor * 0.9f;
                    zPtr[7 * i + 3] = gridZ[idx];
                    xPtr[7 * i + 4] = gridX[idx] + spc1_scale_factor * 0.2f;
                    yPtr[7 * i + 4] = gridY[idx] + spc1_scale_factor * 0.9f;
                    zPtr[7 * i + 4] = gridZ[idx];
                    xPtr[7 * i + 5] = gridX[idx] - spc1_scale_factor * 0.2f;
                    yPtr[7 * i + 5] = gridY[idx] + spc1_scale_factor * 0.7f;
                    zPtr[7 * i + 5] = gridZ[idx];
                    xPtr[7 * i + 6] = gridX[idx] + spc1_scale_factor * 0.2f;
                    yPtr[7 * i + 6] = gridY[idx] + spc1_scale_factor * 0.7f;
                    zPtr[7 * i + 6] = gridZ[idx];
                    break;
                case ROTATION_Z:
                    xPtr[7 * i] = gridX[idx];
                    yPtr[7 * i] = gridY[idx];
                    zPtr[7 * i] = gridZ[idx];
                    xPtr[7 * i + 1] = gridX[idx];
                    yPtr[7 * i + 1] = gridY[idx];
                    zPtr[7 * i + 1] = gridZ[idx] + spc1_scale_factor * 0.9f;
                    xPtr[7 * i + 2] = gridX[idx];
                    yPtr[7 * i + 2] = gridY[idx];
                    zPtr[7 * i + 2] = gridZ[idx] + spc1_scale_factor * 1.1f;
                    xPtr[7 * i + 3] = gridX[idx] + spc1_scale_factor * 0.2f;
                    yPtr[7 * i + 3] = gridY[idx];
                    zPtr[7 * i + 3] = gridZ[idx] + spc1_scale_factor * 0.9f;
                    xPtr[7 * i + 4] = gridX[idx] - spc1_scale_factor * 0.2f;
                    yPtr[7 * i + 4] = gridY[idx];
                    zPtr[7 * i + 4] = gridZ[idx] + spc1_scale_factor * 0.9f;
                    xPtr[7 * i + 5] = gridX[idx] + spc1_scale_factor * 0.2f;
                    yPtr[7 * i + 5] = gridY[idx];
                    zPtr[7 * i + 5] = gridZ[idx] + spc1_scale_factor * 0.7f;
                    xPtr[7 * i + 6] = gridX[idx] - spc1_scale_factor * 0.2f;
                    yPtr[7 * i + 6] = gridY[idx];
                    zPtr[7 * i + 6] = gridZ[idx] + spc1_scale_factor * 0.7f;
                    break;
                }

                clPtr[10 * i] = 7 * i;
                clPtr[10 * i + 1] = 7 * i + 2;
                clPtr[10 * i + 2] = 7 * i + 2;
                clPtr[10 * i + 3] = 7 * i + 3;
                clPtr[10 * i + 4] = 7 * i + 2;
                clPtr[10 * i + 5] = 7 * i + 4;
                clPtr[10 * i + 6] = 7 * i + 1;
                clPtr[10 * i + 7] = 7 * i + 5;
                clPtr[10 * i + 8] = 7 * i + 1;
                clPtr[10 * i + 9] = 7 * i + 6;

                elPtr[5 * i] = 10 * i;
                elPtr[5 * i + 1] = 10 * i + 2;
                elPtr[5 * i + 2] = 10 * i + 4;
                elPtr[5 * i + 3] = 10 * i + 6;
                elPtr[5 * i + 4] = 10 * i + 8;
            }

            colorname = spc1Color->getValue();
            if ((strcmp(colorname, "none") != 0) && (colorname[0] != '\0'))
            {
                spc1_rot->addAttribute("COLOR", colorname);
            }

            spc1->addElement(spc1_rot);

            delete spc1_rot;
            delete[] tmpObjName;
        }

        delete spc1;
    }

    // -------------------
    // the reaction forces
    // -------------------
    if (rfX.size())
    {
        coDoVec3 *oqg1 = new coDoVec3(oqg1Out->getNewObjectInfo(),
                                      rfX.size(),
                                      rfX.getDataPtr(),
                                      rfY.getDataPtr(),
                                      rfZ.getDataPtr());

        if (!oqg1->objectOk())
        {
            sendError("ERROR: Creation of data object 'oqg1' failed");
            return STOP_PIPELINE;
        }

        delete oqg1;
    }

    // -----------------
    // the displacements
    // -----------------
    if (dispX.size())
    {

        const char *name = ougv1Out->getObjName();
        if (name == NULL)
        {
            sendError("Can't find the COVISE 'OUGV1' name");
            return STOP_PIPELINE;
        }

        mode = modeParam->getValue();
        // check input
        if (mode > numDisplacementSets)
            mode = numDisplacementSets;
        if (mode < 1)
        {
            mode = 1;
        }

        bool data_is_transient = dispTransient->getValue();

        if (data_is_transient && numDisplacementSets > 1)
        {
            coDistributedObject **ougv_set = new coDistributedObject *[numDisplacementSets + 1];
            ougv_set[numDisplacementSets] = NULL;
            int i;
            char obj_name[256];
            for (i = 0; i < numDisplacementSets; i++)
            {
                sprintf(obj_name, "%s_%d", name, i);

                ougv_set[i] = new coDoVec3(obj_name,
                                           gridID.size(),
                                           dispX.getDataPtr(i * gridID.size()),
                                           dispY.getDataPtr(i * gridID.size()),
                                           dispZ.getDataPtr(i * gridID.size()));
            }

            coDoSet *ougv1 = new coDoSet(ougv1Out->getNewObjectInfo(), ougv_set);

            sprintf(obj_name, "1 %d", numDisplacementSets);
            ougv1->addAttribute("TIMESTEP", obj_name);

            delete ougv1;
        }
        else
        {
            coDoVec3 *ougv1 = new coDoVec3(name,
                                           gridID.size(),
                                           dispX.getDataPtr((mode - 1) * gridID.size()),
                                           dispY.getDataPtr((mode - 1) * gridID.size()),
                                           dispZ.getDataPtr((mode - 1) * gridID.size()));
            if (!ougv1->objectOk())
            {
                sendError("ERROR: creation of data object 'ougv1' failed");
                return STOP_PIPELINE;
            }
            delete ougv1;
        }
    }

    // -------------------
    // the element forces
    // -------------------
    if (efX.size())
    {
        coDoVec3 *oef1 = new coDoVec3(oef1Out->getNewObjectInfo(),
                                      efX.size(),
                                      efX.getDataPtr(),
                                      efY.getDataPtr(),
                                      efZ.getDataPtr());

        if (!oef1->objectOk())
        {
            sendError("ERROR: creation of data object 'oef1' failed");
            return STOP_PIPELINE;
        }

        delete oef1;
    }

    // ------------------
    // the element stress
    // ------------------
    if (stressList.size())
    {
#ifdef OLD
        coDoFloat *oes1 = new coDoFloat(oes1Out->getNewObjectInfo(),
                                        stressList.size(),
                                        stressList.getDataPtr());
#else
        if (dispX.size() == 0)
        {
            sendError("ERROR: not all required data objects available");
            return STOP_PIPELINE;
        }
        coDoFloat *oes1 = new coDoFloat(oes1Out->getNewObjectInfo(),
                                        elementID.size(),
                                        stressList.getDataPtr((mode - 1) * elementID.size()));
#endif

        if (!oes1->objectOk())
        {
            sendError("ERROR: creation of data object 'oes1' failed");
            return STOP_PIPELINE;
        }

        delete oes1;
    }

    return CONTINUE_PIPELINE;
}

//-------------------------------------------------------------------------
//
// cleanup
//
//-------------------------------------------------------------------------
void ReadNastran::cleanup()
{

    // empty all lists
    conm2List.clear();
    connectionList.clear();
    csID.clear();
    csList.clear();
    dispX.clear();
    dispY.clear();
    dispZ.clear();
    efX.clear();
    efY.clear();
    efZ.clear();
    elementID.clear();
    elementList.clear();
    forceList.clear();
    gravList.clear();
    gridCID.clear();
    gridID.clear();
    gridX.clear();
    gridY.clear();
    gridZ.clear();
    momentList.clear();
    plotelList.clear();
    propertyID.clear();
    rbarList.clear();
    rbe2List.clear();
    rfX.clear();
    rfY.clear();
    rfZ.clear();
    spc1Trans.clear();
    spc1Rot.clear();
    stressList.clear();
    tempList.clear();
    typeList.clear();

    // reset variables
    numDisplacements = 0;
    numDisplacementSets = 0;
    numStresses = 0;
    numStressSets = 0;
}

//-------------------------------------------------------------------------
//
// free all resources
//
//-------------------------------------------------------------------------
void ReadNastran::freeResources()
{
    int i;

    // free all resources for the module
    for (i = 0; i < csList.size(); i++)
        delete csList[i];
    for (i = 0; i < forceList.size(); i++)
        delete forceList[i];
    for (i = 0; i < gravList.size(); i++)
        delete gravList[i];
    for (i = 0; i < momentList.size(); i++)
        delete momentList[i];
    for (i = 0; i < plotelList.size(); i++)
        delete plotelList[i];
    for (i = 0; i < rbarList.size(); i++)
        delete rbarList[i];
    for (i = 0; i < rbe2List.size(); i++)
        delete rbe2List[i];
    for (i = 0; i < spc1Trans.size(); i++)
        delete spc1Trans[i];
    for (i = 0; i < spc1Rot.size(); i++)
        delete spc1Rot[i];
}

//-------------------------------------------------------------------------
//
// load the NASTRAN output2 file
//
//-------------------------------------------------------------------------
bool ReadNastran::load(const char *filename)
{
    fp = fopen((char *)filename, "rb");
    if (fp == NULL)
    {
        sendError("ERROR: Couldn't open NASTRAN output2 file %s\n", filename);
        return false;
    }

    // init members
    blockdesc[0] = 0;
    state = START;
    resultID = 0;
    resElementID = 0;
    deviceCode = 0;
    recordNr = 0;
    flBrokenRecord = false;

    // read first NASTRAN block desc
    if (fread((void *)&blockdesc[1], intSize, (BLOCKDESCSIZE - 1), fp) != BLOCKDESCSIZE - 1)
    {
        fprintf(stderr, "ReadNastran::load: fread failed\n");
    }
    int bd = blockdesc[4];
    byteSwap(bd);
    if (blockdesc[4] >= bd)
    {
        byte_swapped = true;
        byteSwap((unsigned int *)(blockdesc + 1), (BLOCKDESCSIZE - 1));
    }
#ifdef DEEPVERBOSE
    fprintf(stdout, "byte swap: %d\n", (int)byte_swapped);
    fprintf(stdout, "BLOCKDESC: %x %x %x %x %x\n", blockdesc[0], blockdesc[1], blockdesc[2], blockdesc[3], blockdesc[4]);
#endif

    while (!feof(fp))
    {
        if (blockdesc[4] >= BLOCKSIZE)
        {
            sendError("NASTRAN block size %d exceeds the internal block size %d! Giving up...", blockdesc[4], BLOCKSIZE);
            return false;
        }
        // read and process a block
        delete[] block;
        block = new char[blockdesc[2] * intSize * charSize];

        if (fread(block, blockdesc[2] * intSize, charSize, fp) != charSize)
        {
            fprintf(stderr, "ReadNastran::load: fread2 failed\n");
        }
        if (byte_swapped)
        {
            byteSwap((unsigned int *)block, blockdesc[2] * charSize);
        }

        int i = 0;

        if (fread((void *)blockdesc, intSize, 1, fp) != 1)
        {
            fprintf(stderr, "ReadNastran::load: fread3 failed\n");
        }
        if (byte_swapped)
        {
            byteSwap((unsigned int *)blockdesc, 1);
        }

        while (blockdesc[0] != blockdesc[4] && !feof(fp))
        {
            memcpy(block + blockdesc[2] * intSize + i, blockdesc, intSize);
            i++;
            fseek(fp, -3, SEEK_CUR);
            //fprintf(stdout, "%x ", ftell(fp) );
            if (fread((void *)blockdesc, intSize, 1, fp) != 1)
            {
                fprintf(stderr, "ReadNastran::load: fread4 failed\n");
            }
            if (byte_swapped)
            {
                byteSwap((unsigned int *)blockdesc, 1);
            }
        }

        if (!processBlock(blockdesc[2] * intSize + i))
        {
            sendError("Error detected, while processing a NASTRAN data block! Giving up...");
            return false;
        }

        char c;
        if (try_skipping)
        {
            c = 1;
            while (c != 0)
            {
                if (fread(&c, 1, 1, fp) != 1)
                {
                    fprintf(stderr, "ReadNastran::load: fread5 failed\n");
                }
            }
            fseek(fp, -1, SEEK_CUR);
        }

        // read next NASTRAN block description
        if (fread((void *)(blockdesc + 1), intSize, BLOCKDESCSIZE - 1, fp) != BLOCKDESCSIZE - 1)
        {
            fprintf(stderr, "ReadNastran::load: fread6 failed\n");
        }
        if (byte_swapped)
        {
            byteSwap((unsigned int *)(blockdesc + 1), BLOCKDESCSIZE - 1);
        }
#ifdef DEEPVERBOSE
        fprintf(stdout, "BLOCKDESC at 0 %lx: %d %d %d %d %d\n", (long)(ftell(fp) - 20), blockdesc[0], blockdesc[1], blockdesc[2], blockdesc[3], blockdesc[4]);
#endif

        while (blockdesc[2] <= 0 && !feof(fp))
        {
            fseek(fp, -8, SEEK_CUR);
            int ret = int(fread((void *)blockdesc, intSize, BLOCKDESCSIZE, fp));
            if (ret != BLOCKDESCSIZE)
            {
                fprintf(stderr, "ReadNastran::load: fread7 failed: %d instead of %d\n",
                        ret, BLOCKDESCSIZE);
            }
            if (byte_swapped)
            {
                byteSwap((unsigned int *)blockdesc, BLOCKDESCSIZE);
            }

#ifdef DEEPVERBOSE
            fprintf(stdout, "BLOCKDESC at 1 %lx: %d %d %d %d %d\n", (long)(ftell(fp) - 20), blockdesc[0], blockdesc[1], blockdesc[2], blockdesc[3], blockdesc[4]);
#endif
        }
    }
    delete[] block;
    block = NULL;

    fclose(fp);
    return true;
}

//-------------------------------------------------------------------------
//
// print the coordinate system
//
//-------------------------------------------------------------------------
void ReadNastran::printCS(CoordinateSystem2 *cs, const char *str)
{
    cout << "----------------------------------------" << endl;
    if (str)
        cout << "Coordinate sytem: " << str << endl;
    else
        cout << "Coordinate sytem:" << endl;
    cout << "----------------------------------------" << endl;
    cout << "id:" << cs->cid << endl;
    switch (cs->type)
    {
    case CYLINDRICAL:
        cout << "type:   CYLINDRICAL" << endl;
        break;
    case RECTANGULAR:
        cout << "type:   RECTANGULAR" << endl;
        break;
    case SPHERICAL:
        cout << "type:   SPHERICAL" << endl;
        break;
    }
    cout << "Translation vector: [" << cs->t[0] << " " << cs->t[1] << " " << cs->t[2] << "]" << endl;
    cout << "Rotation matrix: [" << cs->r[0][0] << " " << cs->r[0][1] << " " << cs->r[0][2] << ","
         << cs->r[1][0] << " " << cs->r[1][1] << " " << cs->r[1][2] << ","
         << cs->r[2][0] << " " << cs->r[2][1] << " " << cs->r[2][2] << "]" << endl;

    cout << "----------------------------------------" << endl;
}

//-------------------------------------------------------------------------
//
// process a NASTRAN block
//
//-------------------------------------------------------------------------
int ReadNastran::processBlock(int numBytes)
{

    switch (state)
    {

    case BLOCKDESCRIPTION:
        if (numBytes == 28)
        {
            // we ignore the block information
        }
        if (numBytes == 4)
        {
            state = DATABLOCK;
        }
        break;

    case DATABLOCK:
        switch (numBytes)
        {

        case 4:
            // end of something, start of something
            if (substate == MYDEFINITION)
            {
#ifdef DEEPVERBOSE
                cout << "End of card detected." << endl;
#endif
                cardFinished = true;
            }
            else if (substate == MYRESULT)
            {
                if (recordNr == RECORD2)
                    recordNr = RECORD1;
                else
                    recordNr++;
            }
            break;

        case 8:
            // reset the resultID
            if (byte_swapped)
            {
                byteSwap((unsigned int *)block, 8 / intSize);
            }
#ifdef VERBOSE
            // the data block name
            block[8] = '\0';
            fprintf(stdout, "NASTRAN output2 data block name: %s\n", block);
#endif
            if (strncmp("GEOM1S  ", block, 8) == 0)
            {
                state = BLOCKDESCRIPTION;
            }
            else if (strncmp("GEOM2S  ", block, 8) == 0)
            {
                state = BLOCKDESCRIPTION;
            }
            else if (strncmp("GEOM3S  ", block, 8) == 0)
            {
                state = BLOCKDESCRIPTION;
            }
            else if (strncmp("GEOM4S  ", block, 8) == 0)
            {
                state = BLOCKDESCRIPTION;
            }
            else if (strncmp("GEOM1   ", block, 8) == 0)
            {
                substate = MYDEFINITION;
            }
            else if (strncmp("IGEOM2X ", block, 8) == 0)
            {
                substate = MYDEFINITION;
            }
            else if (strncmp("GEOM3   ", block, 8) == 0)
            {
                substate = MYDEFINITION;
            }
            else if (strncmp("GEOM4   ", block, 8) == 0)
            {
                substate = MYDEFINITION;
            }
            else if (strncmp("GPDT    ", block, 8) == 0)
            {
                substate = MYRESULT;
                resultID = GPDT;
                recordNr = DESCRIPTION;
                flBrokenRecord = false;
            }
            else if (strncmp("OQG1    ", block, 8) == 0)
            {
                substate = MYRESULT;
                resultID = OQG1;
                recordNr = DESCRIPTION;
                flBrokenRecord = false;
            }
            else if (strncmp("OUGV1   ", block, 8) == 0)
            {
                substate = MYRESULT;
                resultID = OUGV1;
                recordNr = DESCRIPTION;
                flBrokenRecord = false;
            }
            /**************not implemented ************************************************************
                              else if (strncmp("OEF1X   ", block, 8) == 0 || strncmp("OEF1    ", block, 8) == 0) {
                                 substate = MYRESULT;
                                 resultID = OEF1X;
                                 recordNr = DESCRIPTION;
                                 flBrokenRecord = false;
                              }
               *******************************************************************************************/
            else if (strncmp("OES1X   ", block, 8) == 0 || strncmp("OES1    ", block, 8) == 0)
            {
                substate = MYRESULT;
                //substate = MYIGNORE;
                resultID = OES1X;
                recordNr = DESCRIPTION;
                flBrokenRecord = false;
            }
            else
            {
                // ignore the rest
                substate = MYIGNORE;
                flBrokenRecord = false;
            }

            break;

        default:
            if (substate == MYDEFINITION)
            {
                // this looks like a card
                readCard(numBytes);
            }
            else if (substate == MYRESULT)
            {
                readResult(numBytes);
            }
            break;
        }
        break;

    case HEADER:
        switch (numBytes)
        {
        case 8:
#ifdef VERBOSE
            // the file label
            block[8] = '\0';
            fprintf(stdout, "NASTRAN output2 label: %s\n", block);
#endif
            state = DATABLOCK;
            break;

        case 12:
        {
            int date[3];
            memcpy(date, block, numBytes);
#ifdef VERBOSE
            fprintf(stdout, "NASTRAN output2 file created at %d/%d/%d\n", date[0], date[1], date[2]);
#endif
            break;
        }
        case 28:
            // the tape id
            if (byte_swapped)
            {
                byteSwap((unsigned int *)block, 28 / intSize);
            }
#ifdef VERBOSE
            block[28] = '\0';
            fprintf(stdout, "NASTRAN output2 tape id: %s\n", block);
#endif
            if (strncmp("NASTRAN FORT", block, 12) != 0)
            {
                fprintf(stderr, "File doesn't seem to be a NASTRAN output2 file!\n");
                return false;
            }
            break;
        }
        break;

    case START:
        if (numBytes == 12)
        {
            state = HEADER;
            processBlock(numBytes);
        }
        if (numBytes == 8)
        {
            state = DATABLOCK;
            processBlock(numBytes);
        }
        break;

    case TRAILER:
        // do nothing
        break;
    }

    return true;
}

//-------------------------------------------------------------------------
//
// read a NASTRAN card
//
//-------------------------------------------------------------------------
int ReadNastran::readCard(int numBytes)
{
    int bytesToRead;
    int validRecord;
    int recSize;
    int flDoublePrecision = false;

    if (cardFinished)
    {
        memcpy(card, block, 3 * intSize);
        cardFinished = false;
        bytesToRead = numBytes - 3 * intSize;
        dataPtr = block + 3 * intSize;
    }
    else
    {
        bytesToRead = numBytes;
        dataPtr = block;
    }

    //
    // we use memcpy, because else we will get alignment troubles
    //

    switch (card[0])
    {

    case 1501:
#ifdef DEEPVERBOSE
        cout << "Got CONM2 block." << endl;
#endif
        // ------------------------------------------------------------
        // CONM2 card: 13 words (3 ints, 10 floats)
        // ------------------------------------------------------------
        struct
        {
            int eid; // the element id
            int g; // the grid point
            int cid; // the id of the coordinate system
            float ign[10]; // we ignore the rest
        } conm2;
        // ------------------------------------------------------------
        recSize = 3 * intSize + 10 * floatSize;
        while (bytesToRead)
        {
            if (flBrokenRecord)
            {
                memcpy(&record[recPartSize], block, POS(recSize - recPartSize));
                bytesToRead = bytesToRead - POS(recSize - recPartSize);
                flBrokenRecord = false;
                dataPtr = record;
                validRecord = true;
            }
            else if (bytesToRead < recSize)
            {
                recPartSize = bytesToRead;
                memcpy(record, &block[numBytes - bytesToRead], bytesToRead);
                bytesToRead = 0;
                flBrokenRecord = true;
                validRecord = false;
            }
            else
            {
                dataPtr = &block[numBytes - bytesToRead];
                bytesToRead -= recSize;
                validRecord = true;
            }
            if (validRecord)
            {
                memcpy(&conm2, dataPtr, recSize);
#ifdef DEEPVERBOSE
                fprintf(stdout, "CONM2 at grid point %d.\n", conm2.g);
#endif
                int idx = gridID.findBinary(conm2.g);
                if (idx != -1)
                {
                    conm2List.add(idx);
                }
            }
        }
        // ------------------------------------------------------------
        break;

    case 2001:
    {
#ifdef DEEPVERBOSE
        cout << "Got CORD2C block." << endl;
#endif
        // ------------------------------------------------------------
        // CORD2C card: 13 words (4 ints, 9 doubles)
        // ------------------------------------------------------------
        struct
        {
            int cid; // the id of the coordinate system
            int ign[3]; // we ignore this
            double a[3]; // point A
            double b[3]; // point B
            double c[3]; // point C
        } cord2c_dp;
        struct
        {
            int cid; // the id of the coordinate system
            int ign[3]; // we ignore this
            float a[3]; // point A
            float b[3]; // point B
            float c[3]; // point C
        } cord2c_sp;
        // ------------------------------------------------------------
        if (card[2] != 9)
        {
            flDoublePrecision = true;
            recSize = 4 * intSize + 9 * doubleSize;
        }
        else
        {
            flDoublePrecision = false;
            recSize = 4 * intSize + 9 * floatSize;
        }
        while (bytesToRead)
        {
            CoordinateSystem2 *cs = new CoordinateSystem2;
            if (flBrokenRecord)
            {
                strncpy(&record[recPartSize], block, POS(recSize - recPartSize));
                bytesToRead = bytesToRead - POS(recSize - recPartSize);
                flBrokenRecord = false;
                if (flDoublePrecision)
                {
                    memcpy(&cord2c_dp, record, recSize);
                    if (byte_swapped)
                    {
                        swap_double((double *)cord2c_dp.a, 9);
                    }
                }
                else
                    memcpy(&cord2c_sp, record, recSize);
                validRecord = true;
            }
            else if (bytesToRead < recSize)
            {
                delete cs;
                recPartSize = bytesToRead;
                strncpy(record, &block[numBytes - bytesToRead], bytesToRead);
                bytesToRead = 0;
                flBrokenRecord = true;
                validRecord = false;
            }
            else
            {
                if (flDoublePrecision)
                {
                    memcpy(&cord2c_dp, &block[numBytes - bytesToRead], recSize);
                    if (byte_swapped)
                    {
                        swap_double((double *)cord2c_dp.a, 9);
                    }
                }
                else
                    memcpy(&cord2c_sp, &block[numBytes - bytesToRead], recSize);
                bytesToRead -= recSize;
                validRecord = true;
            }
            if (validRecord)
            {
                cs->type = CYLINDRICAL;
                csList.add(cs);
                if (flDoublePrecision)
                {
                    cs->cid = cord2c_dp.cid;
                    calcTransformMatrix(cs, cord2c_dp.a, cord2c_dp.b, cord2c_dp.c);
                    csID.add(cord2c_dp.cid);
                }
                else
                {
                    double a[3], b[3], c[3];
                    a[0] = cord2c_sp.a[0];
                    a[1] = cord2c_sp.a[1];
                    a[2] = cord2c_sp.a[2];
                    b[0] = cord2c_sp.b[0];
                    b[1] = cord2c_sp.b[1];
                    b[2] = cord2c_sp.b[2];
                    c[0] = cord2c_sp.c[0];
                    c[1] = cord2c_sp.c[1];
                    c[2] = cord2c_sp.c[2];
                    cs->cid = cord2c_sp.cid;
                    calcTransformMatrix(cs, a, b, c);
                    csID.add(cord2c_sp.cid);
                }
#ifdef VERBOSE
                printCS(cs);
#endif
            }
        }
        // ------------------------------------------------------------
        break;
    }

    case 2101:
    {
#ifdef DEEPVERBOSE
        cout << "Got CORD2R block." << endl;
#endif
        // ------------------------------------------------------------
        // CORD2R card: 13 words (4 ints, 9 doubles)
        // ------------------------------------------------------------
        struct
        {
            int cid; // the id of the coordinate system
            int ign[3]; // we ignore this
            double a[3]; // point A
            double b[3]; // point B
            double c[3]; // point C
        } cord2r_dp;
        struct
        {
            int cid; // the id of the coordinate system
            int ign[3]; // we ignore this
            float a[3]; // point A
            float b[3]; // point B
            float c[3]; // point C
        } cord2r_sp;
        // ------------------------------------------------------------
        if (card[2] != 8)
        {
            flDoublePrecision = true;
            recSize = 4 * intSize + 9 * doubleSize;
        }
        else
        {
            flDoublePrecision = false;
            recSize = 4 * intSize + 9 * floatSize;
        }
        while (bytesToRead)
        {
            CoordinateSystem2 *cs = new CoordinateSystem2;
            if (flBrokenRecord)
            {
                strncpy(&record[recPartSize], block, POS(recSize - recPartSize));
                bytesToRead = bytesToRead - POS(recSize - recPartSize);
                flBrokenRecord = false;
                if (flDoublePrecision)
                {
                    memcpy(&cord2r_dp, record, recSize);
                    if (byte_swapped)
                    {
                        swap_double((double *)(cord2r_dp.a), 9);
                    }
                }
                else
                    memcpy(&cord2r_sp, record, recSize);
                validRecord = true;
            }
            else if (bytesToRead < recSize)
            {
                delete cs;
                recPartSize = bytesToRead;
                strncpy(record, &block[numBytes - bytesToRead], bytesToRead);
                bytesToRead = 0;
                flBrokenRecord = true;
                validRecord = false;
            }
            else
            {
                if (flDoublePrecision)
                {
                    memcpy(&cord2r_dp, &block[numBytes - bytesToRead], recSize);
                    if (byte_swapped)
                    {
                        swap_double((double *)(cord2r_dp.a), 9);
                    }
                }
                else
                    memcpy(&cord2r_sp, &block[numBytes - bytesToRead], recSize);
                bytesToRead -= recSize;
                validRecord = true;
            }
            if (validRecord)
            {
                cs->type = RECTANGULAR;
                csList.add(cs);
                if (flDoublePrecision)
                {
                    cs->cid = cord2r_dp.cid;
                    calcTransformMatrix(cs, cord2r_dp.a, cord2r_dp.b, cord2r_dp.c);
                    csID.add(cord2r_dp.cid);
                }
                else
                {
                    double a[3], b[3], c[3];
                    a[0] = cord2r_sp.a[0];
                    a[1] = cord2r_sp.a[1];
                    a[2] = cord2r_sp.a[2];
                    b[0] = cord2r_sp.b[0];
                    b[1] = cord2r_sp.b[1];
                    b[2] = cord2r_sp.b[2];
                    c[0] = cord2r_sp.c[0];
                    c[1] = cord2r_sp.c[1];
                    c[2] = cord2r_sp.c[2];
                    cs->cid = cord2r_sp.cid;
                    calcTransformMatrix(cs, a, b, c);
                    csID.add(cord2r_sp.cid);
                }
#ifdef VERBOSE
                printCS(cs);
#endif
            }
        }
        // ------------------------------------------------------------
        break;
    }

    case 2201:
    {
#ifdef DEEPVERBOSE
        cout << "Got CORD2S block." << endl;
#endif
        // ------------------------------------------------------------
        // CORD2S card: 13 words (4 ints, 9 doubles)
        // ------------------------------------------------------------
        struct
        {
            int cid; // the id of the coordinate system
            int ign[3]; // we ignore this
            double a[3]; // point A
            double b[3]; // point B
            double c[3]; // point C
        } cord2s_dp;
        struct
        {
            int cid; // the id of the coordinate system
            int ign[3]; // we ignore this
            float a[3]; // point A
            float b[3]; // point B
            float c[3]; // point C
        } cord2s_sp;
        // ------------------------------------------------------------
        if (card[2] != 9)
        {
            flDoublePrecision = true;
            recSize = 4 * intSize + 9 * doubleSize;
        }
        else
        {
            flDoublePrecision = false;
            recSize = 4 * intSize + 9 * floatSize;
        }
        while (bytesToRead)
        {
            CoordinateSystem2 *cs = new CoordinateSystem2;
            if (flBrokenRecord)
            {
                strncpy(&record[recPartSize], block, POS(recSize - recPartSize));
                bytesToRead = bytesToRead - POS(recSize - recPartSize);
                flBrokenRecord = false;
                if (flDoublePrecision)
                {
                    memcpy(&cord2s_dp, record, recSize);
                    if (byte_swapped)
                    {
                        swap_double((double *)(cord2s_dp.a), 9);
                    }
                }
                else
                    memcpy(&cord2s_sp, record, recSize);
                validRecord = true;
            }
            else if (bytesToRead < recSize)
            {
                delete cs;
                recPartSize = bytesToRead;
                strncpy(record, &block[numBytes - bytesToRead], bytesToRead);
                bytesToRead = 0;
                flBrokenRecord = true;
                validRecord = false;
            }
            else
            {
                if (flDoublePrecision)
                {
                    memcpy(&cord2s_dp, &block[numBytes - bytesToRead], recSize);
                    if (byte_swapped)
                    {
                        swap_double((double *)(cord2s_dp.a), 9);
                    }
                }
                else
                    memcpy(&cord2s_sp, &block[numBytes - bytesToRead], recSize);
                bytesToRead -= recSize;
                validRecord = true;
            }
            if (validRecord)
            {
                cs->type = SPHERICAL;
                csList.add(cs);
                if (flDoublePrecision)
                {
                    cs->cid = cord2s_dp.cid;
                    calcTransformMatrix(cs, cord2s_dp.a, cord2s_dp.b, cord2s_dp.c);
                    csID.add(cord2s_dp.cid);
                }
                else
                {
                    double a[3], b[3], c[3];
                    a[0] = cord2s_sp.a[0];
                    a[1] = cord2s_sp.a[1];
                    a[2] = cord2s_sp.a[2];
                    b[0] = cord2s_sp.b[0];
                    b[1] = cord2s_sp.b[1];
                    b[2] = cord2s_sp.b[2];
                    c[0] = cord2s_sp.c[0];
                    c[1] = cord2s_sp.c[1];
                    c[2] = cord2s_sp.c[2];
                    cs->cid = cord2s_sp.cid;
                    calcTransformMatrix(cs, a, b, c);
                    csID.add(cord2s_sp.cid);
                }
#ifdef VERBOSE
                printCS(cs);
#endif
            }
        }
        // ------------------------------------------------------------
        break;
    }

    case 2408:
    {
#ifdef DEEPVERBOSE
        cout << "Got CBAR block." << endl;
#endif
        // ------------------------------------------------------------
        // CBAR card: 16 words (10 ints, 6 floats)
        // ------------------------------------------------------------
        struct
        {
            int eid; // the element id
            int pid; // the property id
            int ga; // grid point identification number
            int gb; // grid point identification number
            int r1[6]; // we ignore the rest
            float r2[6]; // we ignore the rest
        } cbar;
        // ------------------------------------------------------------
        recSize = 10 * intSize + 6 * floatSize;
        while (bytesToRead)
        {
            if (flBrokenRecord)
            {
                memcpy(&record[recPartSize], block, POS(recSize - recPartSize));
                bytesToRead = bytesToRead - POS(recSize - recPartSize);
                flBrokenRecord = false;
                dataPtr = record;
                validRecord = true;
            }
            else if (bytesToRead < recSize)
            {
                recPartSize = bytesToRead;
                memcpy(record, &block[numBytes - bytesToRead], bytesToRead);
                bytesToRead = 0;
                flBrokenRecord = true;
                validRecord = false;
            }
            else
            {
                dataPtr = &block[numBytes - bytesToRead];
                bytesToRead -= recSize;
                validRecord = true;
            }
            if (validRecord)
            {
                memcpy(&cbar, dataPtr, recSize);
                int idx1 = gridID.findBinary(cbar.ga);
                int idx2 = gridID.findBinary(cbar.gb);
                typeList.add(TYPE_BAR);
                elementList.add(connectionList.size());
                elementID.add(cbar.eid);
                propertyID.add(cbar.pid);
                connectionList.add(idx1);
                connectionList.add(idx2);
            }
        }
        // ------------------------------------------------------------
        break;
    }

    case 2958:
    {
#ifdef DEEPVERBOSE
        cout << "Got CQUAD block." << endl;
#endif
        // ------------------------------------------------------------
        // CQUAD4 card: 14 words (8 ints, 6 floats)
        // ------------------------------------------------------------
        struct
        {
            int eid; // the element id
            int pid; // the property id
            int g[4]; // grid point identification number
            float r1[2]; // we ignore the rest
            int r2[2]; // we ignore the rest
            float r3[4]; // we ignore the rest
        } cquad4;
        // ------------------------------------------------------------
        recSize = 8 * intSize + 6 * floatSize;
        while (bytesToRead)
        {
            if (flBrokenRecord)
            {
                memcpy(&record[recPartSize], block, POS(recSize - recPartSize));
                bytesToRead = bytesToRead - POS(recSize - recPartSize);
                flBrokenRecord = false;
                dataPtr = record;
                validRecord = true;
            }
            else if (bytesToRead < recSize)
            {
                recPartSize = bytesToRead;
                memcpy(record, &block[numBytes - bytesToRead], bytesToRead);
                bytesToRead = 0;
                flBrokenRecord = true;
                validRecord = false;
            }
            else
            {
                dataPtr = &block[numBytes - bytesToRead];
                bytesToRead -= recSize;
                validRecord = true;
            }
            if (validRecord)
            {
                memcpy(&cquad4, dataPtr, recSize);

                if (gridID.findBinary(cquad4.g[0]) == -1)
                {
                    //fprintf(stdout, "%x\n ", ftell(fp) );
                    return true;
                }

                int idx;
                elementList.add(connectionList.size());
                elementID.add(cquad4.eid);
                propertyID.add(cquad4.pid);
                for (int i = 0; i < 4; i++)
                {
                    idx = gridID.findBinary(cquad4.g[i]);
                    //cout << cquad4.g[i] << " " << endl;
                    connectionList.add(idx);
                }

                typeList.add(TYPE_QUAD);
            }
        }
        // ------------------------------------------------------------
        break;
    }

    case 10808:
    {
#ifdef DEEPVERBOSE
        cout << "Got CHBDYG block." << endl;
#endif
        // ------------------------------------------------------------
        // CTETRA card: 6 words (6 ints)
        // ------------------------------------------------------------
        struct
        {
            int eid; // the element id
            int pid; // the property id
            int d[6];
            int g[4]; // grid point identification number
            int r[4]; // unused
        } chbdyg;
        // ------------------------------------------------------------
        recSize = 16 * intSize;
        flBrokenRecord = false;

        while (bytesToRead)
        {
            if (flBrokenRecord)
            {
                memcpy(&record[recPartSize], block, POS(recSize - recPartSize));
                bytesToRead = bytesToRead - POS(recSize - recPartSize);
                flBrokenRecord = false;
                dataPtr = record;
                validRecord = true;
            }
            else if (bytesToRead < recSize)
            {
                recPartSize = bytesToRead;
                memcpy(record, &block[numBytes - bytesToRead], bytesToRead);
                bytesToRead = 0;
                flBrokenRecord = true;
                validRecord = false;
            }
            else
            {
                dataPtr = &block[numBytes - bytesToRead];
                bytesToRead -= recSize;
                validRecord = true;
            }
            if (validRecord)
            {
                memcpy(&chbdyg, dataPtr, recSize);

                int idx;
                elementList.add(connectionList.size());
                elementID.add(chbdyg.eid);
                propertyID.add(chbdyg.pid);
                for (int i = 0; i < 4; i++)
                {
                    idx = gridID.findBinary(chbdyg.g[i]);
                    if (idx == -1)
                    {
                        cout << "idx == -1 for " << chbdyg.g[i] << ", " << i << ", " << chbdyg.eid << endl;
                        cout << "take " << (idx = gridID.findBinary(chbdyg.g[0])) << " instead " << endl;
                    }

                    connectionList.add(idx);
                }
                typeList.add(TYPE_TETRAHEDER);
            }
        }
        // ------------------------------------------------------------
        break;
    }
    case 5508:
    {
#ifdef DEEPVERBOSE
        cout << "Got CTETRA block." << endl;
#endif
        // ------------------------------------------------------------
        // CTETRA card: 6 words (6 ints)
        // ------------------------------------------------------------
        struct
        {
            int eid; // the element id
            int pid; // the property id
            int g[4]; // grid point identification number
            int r[6]; // unused
        } ctetra;
        // ------------------------------------------------------------
        recSize = 12 * intSize;
        while (bytesToRead)
        {
            if (flBrokenRecord)
            {
                memcpy(&record[recPartSize], block, POS(recSize - recPartSize));
                bytesToRead = bytesToRead - POS(recSize - recPartSize);
                flBrokenRecord = false;
                dataPtr = record;
                validRecord = true;
            }
            else if (bytesToRead < recSize)
            {
                recPartSize = bytesToRead;
                memcpy(record, &block[numBytes - bytesToRead], bytesToRead);
                bytesToRead = 0;
                flBrokenRecord = true;
                validRecord = false;
            }
            else
            {
                dataPtr = &block[numBytes - bytesToRead];
                bytesToRead -= recSize;
                validRecord = true;
            }
            if (validRecord)
            {
                memcpy(&ctetra, dataPtr, recSize);

                int idx;
                elementList.add(connectionList.size());
                elementID.add(ctetra.eid);
                propertyID.add(ctetra.pid);
                for (int i = 0; i < 4; i++)
                {
                    idx = gridID.findBinary(ctetra.g[i]);
                    //                 if(idx == -1)
                    //                      cout << "idx == -1 for " << ctetra.g[i] << ", " << i << endl;
                    connectionList.add(idx);
                }
                typeList.add(TYPE_TETRAHEDER);
            }
        }
        // ------------------------------------------------------------
        break;
    }

    case 4108:
    {
#ifdef DEEPVERBOSE
        cout << "Got CPENTA block." << endl;
#endif
        // ------------------------------------------------------------
        // CPENTA card: 17 words (17 ints)
        // ------------------------------------------------------------
        struct
        {
            int eid; // the element id
            int pid; // the property id
            int g[15]; // grid point identification number
        } cpenta;
        // ------------------------------------------------------------
        recSize = 17 * intSize;
        while (bytesToRead)
        {
            if (flBrokenRecord)
            {
                memcpy(&record[recPartSize], block, POS(recSize - recPartSize));
                bytesToRead = bytesToRead - POS(recSize - recPartSize);
                flBrokenRecord = false;
                dataPtr = record;
                validRecord = true;
            }
            else if (bytesToRead < recSize)
            {
                recPartSize = bytesToRead;
                memcpy(record, &block[numBytes - bytesToRead], bytesToRead);
                bytesToRead = 0;
                flBrokenRecord = true;
                validRecord = false;
            }
            else
            {
                dataPtr = &block[numBytes - bytesToRead];
                bytesToRead -= recSize;
                validRecord = true;
            }
            if (validRecord)
            {
                memcpy(&cpenta, dataPtr, recSize);
                int idx;
                elementList.add(connectionList.size());
                elementID.add(cpenta.eid);
                propertyID.add(cpenta.pid);
                for (int i = 0; i < 6; i++)
                {
                    idx = gridID.findBinary(cpenta.g[i]);
                    connectionList.add(idx);
                }
                typeList.add(TYPE_PRISM);
            }
        }
        // ------------------------------------------------------------
        break;
    }

    case 4201:
    {
#ifdef DEEPVERBOSE
        cout << "Got FORCE block." << endl;
#endif
        // ------------------------------------------------------------
        // FORCE card: 7 words (3 ints, 4 floats)
        // ------------------------------------------------------------
        struct
        {
            int sid; // the element id
            int g; // the grid point
            int cid; // the id of the coordinate system
            float f; // the force value
            float n[3]; // the force direction
        } force;
        // ------------------------------------------------------------
        recSize = 3 * intSize + 4 * floatSize;
        while (bytesToRead)
        {
            if (flBrokenRecord)
            {
                memcpy(&record[recPartSize], block, POS(recSize - recPartSize));
                bytesToRead = bytesToRead - POS(recSize - recPartSize);
                flBrokenRecord = false;
                dataPtr = record;
                validRecord = true;
            }
            else if (bytesToRead < recSize)
            {
                recPartSize = bytesToRead;
                memcpy(record, &block[numBytes - bytesToRead], bytesToRead);
                bytesToRead = 0;
                flBrokenRecord = true;
                validRecord = false;
            }
            else
            {
                dataPtr = &block[numBytes - bytesToRead];
                bytesToRead -= recSize;
                validRecord = true;
            }
            if (validRecord)
            {
                memcpy(&force, dataPtr, recSize);
                VectorArrow *vector = new VectorArrow;
                vector->gid = force.g;
                vector->length = force.f;
                vector->n[0] = force.n[0];
                vector->n[1] = force.n[1];
                vector->n[2] = force.n[2];
                forceList.add(vector);
#ifdef DEEPVERBOSE
                fprintf(stdout, "FORCE at %d: %f [%f %f %f]\n", force.g, force.f, force.n[0], force.n[1], force.n[2]);
#endif
            }
        }
        // ------------------------------------------------------------
        break;
    }

    case 4401:
    {
#ifdef DEEPVERBOSE
        cout << "Got GRAV block." << endl;
#endif
        // ------------------------------------------------------------
        // GRAV card: 7 words (3 ints, 4 floats)
        //       REM: docu says 6 words, data file has 7 words!!!
        // ------------------------------------------------------------
        struct
        {
            int sid; // the element id
            int cid; // the id of the coordinate system
            float g; // the grav value
            float n[3]; // the grav direction
            int r; // this is not documented, but it is there
        } grav;
        // ------------------------------------------------------------
        recSize = 3 * intSize + 4 * floatSize;
        while (bytesToRead)
        {
            if (flBrokenRecord)
            {
                memcpy(&record[recPartSize], block, POS(recSize - recPartSize));
                bytesToRead = bytesToRead - POS(recSize - recPartSize);
                flBrokenRecord = false;
                dataPtr = record;
                validRecord = true;
            }
            else if (bytesToRead < recSize)
            {
                recPartSize = bytesToRead;
                memcpy(record, &block[numBytes - bytesToRead], bytesToRead);
                bytesToRead = 0;
                flBrokenRecord = true;
                validRecord = false;
            }
            else
            {
                dataPtr = &block[numBytes - bytesToRead];
                bytesToRead -= recSize;
                validRecord = true;
            }
            if (validRecord)
            {
                memcpy(&grav, dataPtr, recSize);
                VectorArrow *vector = new VectorArrow;
                vector->length = grav.g;
                vector->n[0] = grav.n[0];
                vector->n[1] = grav.n[1];
                vector->n[2] = grav.n[2];
                gravList.add(vector);
#ifdef DEEPVERBOSE
                fprintf(stdout, "GRAV: %f [%f %f %f]\n", grav.g, grav.n[0], grav.n[1], grav.n[2]);
#endif
            }
        }
        // ------------------------------------------------------------
        break;
    }

    case 4501:
    {
#ifdef DEEPVERBOSE
        cout << "Got GRID block." << endl;
#endif
        //         cout << "Got GRID block." << card[0] << endl;
        //         cout << "Got GRID block." << card[1] << endl;
        //         cout << "Got GRID block." << card[2] << endl;

        // ------------------------------------------------------------
        // GRID card: 8 words (5 ints, 3 doubles)
        // ------------------------------------------------------------
        struct
        {
            int id; // the grid id
            int cid; // the id of the coordinate system
            double p[3]; // the grid point coordinates
            int r[3]; // we ignore the rest
        } grid_dp;

        struct
        {
            int id; // the grid id
            int cid; // the id of the coordinate system
            float p[3]; // the grid point coordinates
            int r[3]; // we ignore the rest
        } grid_sp;
        // ------------------------------------------------------------
        if (card[2] != 1)
        {
            flDoublePrecision = true;
            recSize = 5 * intSize + 3 * doubleSize;
        }
        else
        {
            flDoublePrecision = false;
            recSize = 5 * intSize + 3 * floatSize;
        }
#ifdef DEEPVERBOSE
        fprintf(stdout, "GRID record size: %d\n", recSize);
#endif

        while (bytesToRead)
        {

            if (flBrokenRecord)
            {
                memcpy(&record[recPartSize], block, POS(recSize - recPartSize));
                bytesToRead = bytesToRead - POS(recSize - recPartSize);
                flBrokenRecord = false;
                dataPtr = record;
                validRecord = true;
            }
            else if (bytesToRead < recSize)
            {
                recPartSize = bytesToRead;
                memcpy(record, &block[numBytes - bytesToRead], bytesToRead);
                bytesToRead = 0;
                flBrokenRecord = true;
                validRecord = false;
            }
            else
            {
                dataPtr = &block[numBytes - bytesToRead];
                bytesToRead -= recSize;
                validRecord = true;
            }
            if (validRecord)
            {
                float p[3];
                if (flDoublePrecision)
                {
                    memcpy(&grid_dp, dataPtr, recSize);
                    if (byte_swapped)
                    {
                        swap_double((double *)(grid_dp.p), 3);
                    }
                    gridID.add(grid_dp.id);
                    transformCoordinate(grid_dp.cid, grid_dp.p, p);
                }
                else
                {
                    double pd[3];
                    memcpy(&grid_sp, dataPtr, recSize);
                    gridID.add(grid_sp.id);
                    pd[0] = grid_sp.p[0];
                    pd[1] = grid_sp.p[1];
                    pd[2] = grid_sp.p[2];
                    transformCoordinate(grid_sp.cid, pd, p);
                }
                gridX.add(p[0]);
                gridY.add(p[1]);
                gridZ.add(p[2]);
            }
        }
        // ------------------------------------------------------------
        break;
    }

    case 4801:
    {
#ifdef DEEPVERBOSE
        cout << "Got MOMENT block." << endl;
#endif
        // ------------------------------------------------------------
        // MOMENT card: 7 words (3 ints, 4 floats)
        // ------------------------------------------------------------
        struct
        {
            int sid; // the element id
            int g; // the grid point
            int cid; // the id of the coordinate system
            float m; // the moment value
            float n[3]; // the moment direction
        } moment;
        // ------------------------------------------------------------
        recSize = 3 * intSize + 4 * floatSize;
        while (bytesToRead)
        {
            if (flBrokenRecord)
            {
                memcpy(&record[recPartSize], block, POS(recSize - recPartSize));
                bytesToRead = bytesToRead - POS(recSize - recPartSize);
                flBrokenRecord = false;
                dataPtr = record;
                validRecord = true;
            }
            else if (bytesToRead < recSize)
            {
                recPartSize = bytesToRead;
                memcpy(record, &block[numBytes - bytesToRead], bytesToRead);
                bytesToRead = 0;
                flBrokenRecord = true;
                validRecord = false;
            }
            else
            {
                dataPtr = &block[numBytes - bytesToRead];
                bytesToRead -= recSize;
                validRecord = true;
            }
            if (validRecord)
            {
                memcpy(&moment, dataPtr, recSize);
                VectorArrow *vector = new VectorArrow;
                vector->gid = moment.g;
                vector->length = moment.m;
                vector->n[0] = moment.n[0];
                vector->n[1] = moment.n[1];
                vector->n[2] = moment.n[2];
                momentList.add(vector);
#ifdef DEEPVERBOSE
                fprintf(stdout, "MOMENT at %d: %f [%f %f %f]\n", moment.g, moment.m, moment.n[0], moment.n[1], moment.n[2]);
#endif
            }
        }
        // ------------------------------------------------------------
        break;
    }

    case 5201:
    {
#ifdef DEEPVERBOSE
        cout << "Got PLOTEL block." << endl;
#endif
        // ------------------------------------------------------------
        // PLOTEL card: 3 words (3 ints)
        // ------------------------------------------------------------
        struct
        {
            int eid; // the element id
            int g[2]; // the grid points
        } plotel;
        // ------------------------------------------------------------
        recSize = 3 * intSize;
        while (bytesToRead)
        {
            if (flBrokenRecord)
            {
                memcpy(&record[recPartSize], block, POS(recSize - recPartSize));
                bytesToRead = bytesToRead - POS(recSize - recPartSize);
                flBrokenRecord = false;
                dataPtr = record;
                validRecord = true;
            }
            else if (bytesToRead < recSize)
            {
                recPartSize = bytesToRead;
                memcpy(record, &block[numBytes - bytesToRead], bytesToRead);
                bytesToRead = 0;
                flBrokenRecord = true;
                validRecord = false;
            }
            else
            {
                dataPtr = &block[numBytes - bytesToRead];
                bytesToRead -= recSize;
                validRecord = true;
            }
            if (validRecord)
            {
                memcpy(&plotel, dataPtr, recSize);
                ConnectionLine *line = new ConnectionLine;
                line->g[0] = plotel.g[0];
                line->g[1] = plotel.g[1];
                plotelList.add(line);
            }
        }
        // ------------------------------------------------------------
        break;
    }

    case 5481:
    {
#ifdef DEEPVERBOSE
        cout << "Got SPC1 block." << endl;
#endif
        // ------------------------------------------------------------
        // SPC1 card: 2 forms
        //             1. open ended: sid c 0 g1 g2 gn -1
        //             2. sid c 1 g1 g2
        // ------------------------------------------------------------
        struct
        {
            int sid; //
            int c; //
            int form; // form 0 or 1
        } spc1;
        while (bytesToRead > 0)
        {
            int value = 0;
            memcpy(&spc1, dataPtr, 3 * intSize);
            dataPtr += 3 * intSize;
            bytesToRead -= 3 * intSize;
            //
            // es scheint so, als ob Form 0 die Repraesentation
            // der THRU Anweisung darstellt
            //
            if (spc1.form == 0)
            {
                while (value != -1 && bytesToRead > 0)
                {
                    memcpy(&value, dataPtr, intSize);
                    dataPtr += intSize;
                    bytesToRead -= intSize;
                    // fprintf(stdout, "SPC1: dof = %d\n", spc1.c);
                    if (value != -1)
                    {
                        div_t dofid = div(spc1.c, 10);
                        while (dofid.rem != 0)
                        {
                            DOF *dof = new DOF;
                            dof->gid = value;
                            dof->dof = dofid.rem;
                            // fprintf(stdout, "SPC1: gid = %d, dof = %d\n", dof->gid, dof->dof);
                            if (dof->dof < ROTATION_X)
                                spc1Trans.add(dof);
                            else
                                spc1Rot.add(dof);
                            dofid = div(dofid.quot, 10);
                        }
                    }
                }
#ifdef DEEPVERBOSE
                fprintf(stdout, "Got SPC1 (form 0)\n");
#endif
            }
            else
            {
                int points[2];
                memcpy(points, dataPtr, 2 * intSize);
                dataPtr += 2 * intSize;
                bytesToRead -= 2 * intSize;
                for (int i = 0; i < 2; i++)
                {
                    div_t dofid = div(spc1.c, 10);
                    while (dofid.rem != 0)
                    {
                        DOF *dof = new DOF;
                        dof->gid = points[i];
                        dof->dof = dofid.rem;
                        if (dof->dof < ROTATION_X)
                            spc1Trans.add(dof);
                        else
                            spc1Rot.add(dof);
                        dofid = div(dofid.quot, 10);
                    }
                }
#ifdef DEEPVERBOSE
                fprintf(stdout, "Got SPC1 (form 1): %d %d\n", points[0], points[1]);
#endif
            }
        }
        break;
    }

    case 5701:
    {
#ifdef DEEPVERBOSE
        cout << "Got TEMP block." << endl;
#endif
        // ------------------------------------------------------------
        // TEMP card: 3 words (2 ints, 1 float)
        // ------------------------------------------------------------
        struct
        {
            int sid; // the element id
            int g; // the grid point
            float t; // the temp value
        } temp;
        // ------------------------------------------------------------
        recSize = 2 * intSize + floatSize;
        while (bytesToRead)
        {
            if (flBrokenRecord)
            {
                memcpy(&record[recPartSize], block, POS(recSize - recPartSize));
                bytesToRead = bytesToRead - POS(recSize - recPartSize);
                flBrokenRecord = false;
                dataPtr = record;
                validRecord = true;
            }
            else if (bytesToRead < recSize)
            {
                recPartSize = bytesToRead;
                memcpy(record, &block[numBytes - bytesToRead], bytesToRead);
                bytesToRead = 0;
                flBrokenRecord = true;
                validRecord = false;
            }
            else
            {
                dataPtr = &block[numBytes - bytesToRead];
                bytesToRead -= recSize;
                validRecord = true;
            }
            if (validRecord)
            {
                memcpy(&temp, dataPtr, recSize);
                if (tempList.size() == 0)
                {
                    tempList.newSize(gridID.size());
                    tempList.init(NULL_TEMPERATURE);
                }
                int idx;
                if (gridID.size() > 0)
                    idx = gridID.findBinary(temp.g);
                else
                    break;
                if (idx != -1)
                    tempList.set(idx, temp.t);
#ifdef DEEPVERBOSE
                fprintf(stdout, "TEMP at %d: %f\n", temp.g, temp.t);
#endif
            }
        }
        // ------------------------------------------------------------
        break;
    }

    case 5959:
    {
#ifdef DEEPVERBOSE
        cout << "Got CTRIA3 block." << endl;
#endif
        // ------------------------------------------------------------
        // CTRIA3 card: 13 words (8 ints, 5 floats)
        // ------------------------------------------------------------
        struct
        {
            int eid; // the element id
            int pid; // the property id
            int g[3]; // grid point identification number
            float r1[2]; // we ignore the rest
            int r2[3]; // we ignore the rest
            float r3[3]; // we ignore the rest
        } ctria3;
        // ------------------------------------------------------------
        recSize = 8 * intSize + 5 * floatSize;

        while (bytesToRead)
        {
            if (flBrokenRecord)
            {
                memcpy(&record[recPartSize], block, POS(recSize - recPartSize));
                bytesToRead = bytesToRead - POS(recSize - recPartSize);
                flBrokenRecord = false;
                dataPtr = record;
                validRecord = true;
            }
            else if (bytesToRead < recSize)
            {
                recPartSize = bytesToRead;
                memcpy(record, &block[numBytes - bytesToRead], bytesToRead);
                bytesToRead = 0;
                flBrokenRecord = true;
                validRecord = false;
            }
            else
            {
                dataPtr = &block[numBytes - bytesToRead];
                bytesToRead -= recSize;
                validRecord = true;
            }
            if (validRecord)
            {
                memcpy(&ctria3, dataPtr, recSize);
                int idx;
                elementList.add(connectionList.size());
                elementID.add(ctria3.eid);
                propertyID.add(ctria3.pid);
                for (int i = 0; i < 3; i++)
                {
                    idx = gridID.findBinary(ctria3.g[i]);
                    connectionList.add(idx);
                }
                typeList.add(TYPE_TRIANGLE);
            }
        }
        // ------------------------------------------------------------
        break;
    }

    case 6601:
    {
#ifdef DEEPVERBOSE
        cout << "Got RBAR block." << endl;
#endif
        // ------------------------------------------------------------
        // RBAR card: 7 words (7 ints)
        // ------------------------------------------------------------
        struct
        {
            int eid; // the element id
            int g[2]; // grid points
            int ign[4]; // we ignore the rest
        } rbar;
        // ------------------------------------------------------------
        recSize = 7 * intSize;

        while (bytesToRead)
        {
            if (flBrokenRecord)
            {
                memcpy(&record[recPartSize], block, POS(recSize - recPartSize));
                bytesToRead = bytesToRead - POS(recSize - recPartSize);
                flBrokenRecord = false;
                dataPtr = record;
                validRecord = true;
            }
            else if (bytesToRead < recSize)
            {
                recPartSize = bytesToRead;
                memcpy(record, &block[numBytes - bytesToRead], bytesToRead);
                bytesToRead = 0;
                flBrokenRecord = true;
                validRecord = false;
            }
            else
            {
                dataPtr = &block[numBytes - bytesToRead];
                bytesToRead -= recSize;
                validRecord = true;
            }
            if (validRecord)
            {
                memcpy(&rbar, dataPtr, recSize);
                ConnectionLine *line = new ConnectionLine;
                line->g[0] = rbar.g[0];
                line->g[1] = rbar.g[1];
                rbarList.add(line);
            }
        }
        // ------------------------------------------------------------
        break;
    }

    case 6901:
    {
#ifdef DEEPVERBOSE
        cout << "Got RBE2 block." << endl;
#endif
        // ------------------------------------------------------------
        // RBE2 card: (open ended)
        // ------------------------------------------------------------
        struct
        {
            int eid; // the element id
            int gn; // the father grid point
            int cn; // ?
        } rbe2;
        // ------------------------------------------------------------
        while (bytesToRead > 0)
        {
            int value = 0;
            memcpy(&rbe2, dataPtr, 3 * intSize);
            dataPtr += 3 * intSize;
            bytesToRead -= 3 * intSize;
            while (value != -1 && bytesToRead > 0)
            {
                memcpy(&value, dataPtr, intSize);
                dataPtr += intSize;
                bytesToRead -= intSize;
                if (value != -1)
                {
                    ConnectionLine *line = new ConnectionLine;
                    line->g[0] = rbe2.gn;
                    line->g[1] = value;
                    rbe2List.add(line);
                    // fprintf(stdout, "Draw line from %d to %d\n", rbe2.gn, value);
                }
            }
        }
        break;
    }

    case 7308:
    {
#ifdef DEEPVERBOSE
        cout << "Got CHEXA block." << endl;
#endif
        // ------------------------------------------------------------
        // CHEXA card: 22 words (22 ints)
        // ------------------------------------------------------------
        struct
        {
            int eid; // the element id
            int pid; // the property id
            int g[20]; // grid point identification numbers
        } chexa;
        // ------------------------------------------------------------
        recSize = 22 * intSize;
        while (bytesToRead)
        {
            if (flBrokenRecord)
            {
                memcpy(&record[recPartSize], block, POS(recSize - recPartSize));
                bytesToRead = bytesToRead - POS(recSize - recPartSize);
                flBrokenRecord = false;
                dataPtr = record;
                validRecord = true;
            }
            else if (bytesToRead < recSize)
            {
                recPartSize = bytesToRead;
                memcpy(record, &block[numBytes - bytesToRead], bytesToRead);
                bytesToRead = 0;
                flBrokenRecord = true;
                validRecord = false;
            }
            else
            {
                dataPtr = &block[numBytes - bytesToRead];
                bytesToRead -= recSize;
                validRecord = true;
            }
            if (validRecord)
            {
                memcpy(&chexa, dataPtr, recSize);
                int idx;
                elementList.add(connectionList.size());
                elementID.add(chexa.eid);
                propertyID.add(chexa.pid);
                for (int i = 0; i < 8; i++)
                {
                    idx = gridID.findBinary(chexa.g[i]);
                    connectionList.add(idx);
                }
                typeList.add(TYPE_HEXAEDER);
            }
        }
        // ------------------------------------------------------------
        break;
    }

    case 65535:
#ifdef DEEPVERBOSE
        cout << "Got END of data block." << endl;
#endif
        break;

    default:
#ifdef VERBOSE
        cout << "couldn't identify card id " << card[0] << endl;
#endif
        // ignore the rest
        break;
    }
    return true;
}

//-------------------------------------------------------------------------
//
// read a NASTRAN result block
//
//-------------------------------------------------------------------------
int ReadNastran::readResult(int numBytes)
{
    int recSize;
    int bytesToRead;
    int validRecord;
    int idx;
    int correct;

#ifdef VERBOSE
    int *test_device;
#endif

    if (numBytes == 0)
        return true;

    bytesToRead = numBytes;

    switch (resultID)
    {
    //----------------
    // gridPoints
    //----------------
    case GPDT:
#ifdef VERBOSE
        cout << numBytes << " bytes in GPDT result set." << endl;
#endif
        // ------------------------------------------------------------
        // GRID card: 8 words (5 ints, 3 doubles)
        // ------------------------------------------------------------
        struct
        {
            int id; // the grid id
            int cid; // the id of the coordinate system
            float p[3]; // the grid point coordinates
            int r[2]; // we ignore the rest
        } grid_point;

        recSize = 4 * intSize + 3 * floatSize;

        switch (recordNr)
        {

        case DESCRIPTION:

            break;

        case RECORD0:
        {
            while (bytesToRead)
            {

                if (flBrokenRecord)
                {
                    memcpy(&record[recPartSize], block, POS(recSize - recPartSize));
                    bytesToRead = bytesToRead - POS(recSize - recPartSize);
                    flBrokenRecord = false;
                    dataPtr = record;
                    validRecord = true;
                }
                else if (bytesToRead < recSize)
                {
                    recPartSize = bytesToRead;
                    memcpy(record, &block[numBytes - bytesToRead], bytesToRead);
                    bytesToRead = 0;
                    flBrokenRecord = true;
                    validRecord = false;
                }
                else
                {
                    dataPtr = &block[numBytes - bytesToRead];
                    bytesToRead -= recSize;
                    validRecord = true;
                }
                if (validRecord)
                {
                    float p[3];
                    double pd[3];
                    int ret = false;

                    memcpy(&grid_point, dataPtr, recSize);

                    pd[0] = grid_point.p[0];
                    pd[1] = grid_point.p[1];
                    pd[2] = grid_point.p[2];
                    if (transformCoordinate(grid_point.cid, pd, p) == false)
                    {
                        ret = true;
                    }

                    gridID.add(grid_point.id);
#ifdef VERBOSE
                    cout << "GPDT: " << grid_point.id << " " << grid_point.cid << " : " << p[0] << " " << p[1] << " " << p[2] << " " << endl;
#endif
                    gridX.add(p[0]);
                    gridY.add(p[1]);
                    gridZ.add(p[2]);
                    if (ret)
                    {
                        return true;
                    }
                }
            }
        }
        }
        break;

    //----------------
    // reaction forces
    //----------------
    case OQG1:
#ifdef VERBOSE
        cout << numBytes << " bytes in OQG1 result set." << endl;
#endif
        switch (recordNr)
        {

        case DESCRIPTION:
        case RECORD0:
            // skip that
            break;

        case RECORD1:
        {
            // detect element type
            int arr[10];
            memcpy(arr, block, 10 * intSize);
            // fprintf(stdout, "Format code = %d\n", arr[8]);
            // fprintf(stdout, "Number of words per record = %d\n", arr[9]);
            break;
        }

        case RECORD2:
        {
            // ------------------------------------------------------------
            // OQG1 result record2: 8 words (2 ints + 6 floats)
            // ------------------------------------------------------------
            struct
            {
                int id; // the point id
                int pt; // the point type id
                float r[6]; // the forces
            } oqg;
            // ------------------------------------------------------------
            recSize = 2 * intSize + 6 * floatSize;
            while (bytesToRead)
            {
                if (flBrokenRecord)
                {
                    memcpy(&record[recPartSize], block, POS(recSize - recPartSize));
                    bytesToRead = bytesToRead - POS(recSize - recPartSize);
                    flBrokenRecord = false;
                    dataPtr = record;
                    validRecord = true;
                }
                else if (bytesToRead < recSize)
                {
                    recPartSize = bytesToRead;
                    memcpy(record, &block[numBytes - bytesToRead], bytesToRead);
                    bytesToRead = 0;
                    flBrokenRecord = true;
                    validRecord = false;
                }
                else
                {
                    dataPtr = &block[numBytes - bytesToRead];
                    bytesToRead -= recSize;
                    validRecord = true;
                }
                if (validRecord)
                {
                    memcpy(&oqg, dataPtr, recSize);
                    if (rfX.size() == 0)
                    {
                        rfX.newSize(gridID.size());
                        rfY.newSize(gridID.size());
                        rfZ.newSize(gridID.size());
                        rfX.init(0.0f);
                        rfY.init(0.0f);
                        rfZ.init(0.0f);
                        sendInfo("Reaction forces OQG detected.");
                    }
                    idx = gridID.findBinary(oqg.id / 10);
                    if (idx != -1)
                    {
                        // TODO: check format code
                        // (this is real format)
                        rfX.set(idx, oqg.r[0]);
                        rfY.set(idx, oqg.r[1]);
                        rfZ.set(idx, oqg.r[2]);
                    }
                    else
                    {
                        fprintf(stdout, "ERROR: Can't find grid id for OQG1!\n");
                    }
                }
            }
            break;
        }
        }
        break;

    //--------------
    // displacements
    //--------------
    case OUGV1:
#ifdef VERBOSE
        cout << numBytes << " bytes in OUGV1 result set." << endl;
#endif
        switch (recordNr)
        {

        case DESCRIPTION:
        case RECORD0:
            // skip that
            break;

        case RECORD1:
        {
            // detect element type
            int arr[10];
            memcpy(arr, block, 10 * intSize);
            //fprintf(stdout, "Format code = %d\n", arr[8]);
            //fprintf(stdout, "Number of words per record = %d\n", arr[9]);
            break;
        }

        case RECORD2:
        {
            // ------------------------------------------------------------
            // OUGV result record2: 8 words (2 ints + 6 floats)
            // ------------------------------------------------------------
            struct
            {
                int id; // the point id
                int pt; // the point type id
                float r[6]; // the displacements
            } ougv;
            // ------------------------------------------------------------
            recSize = 2 * intSize + 6 * floatSize;
            while (bytesToRead)
            {
                if (flBrokenRecord)
                {
                    memcpy(&record[recPartSize], block, POS(recSize - recPartSize));
                    bytesToRead = bytesToRead - POS(recSize - recPartSize);
                    flBrokenRecord = false;
                    dataPtr = record;
                    validRecord = true;
                }
                else if (bytesToRead < recSize)
                {
                    recPartSize = bytesToRead;
                    memcpy(record, &block[numBytes - bytesToRead], bytesToRead);
                    bytesToRead = 0;
                    flBrokenRecord = true;
                    validRecord = false;
                }
                else
                {
                    dataPtr = &block[numBytes - bytesToRead];
                    bytesToRead -= recSize;
                    validRecord = true;
                }
                if (validRecord)
                {
                    memcpy(&ougv, dataPtr, recSize);
                    if (dispX.size() == 0)
                    {
                        numDisplacementSets = 1;
                        dispX.newSize(gridID.size());
                        dispY.newSize(gridID.size());
                        dispZ.newSize(gridID.size());
                        dispX.init(0.0f);
                        dispY.init(0.0f);
                        dispZ.init(0.0f);
                    }
                    if (gridID.size() > 0)
                        idx = gridID.findBinary(ougv.id / 10);
                    else
                        break;

                    if (idx != -1)
                    {
                        numDisplacements++;
                        if (numDisplacements > (numDisplacementSets * gridID.size()))
                        {
                            // grow
                            numDisplacementSets++;
                            dispX.newSize(gridID.size() * numDisplacementSets);
                            dispY.newSize(gridID.size() * numDisplacementSets);
                            dispZ.newSize(gridID.size() * numDisplacementSets);
                        }

                        // TODO: check format code
                        // (this is real format)

                        dispX.set(gridID.size() * (numDisplacementSets - 1) + idx, ougv.r[0]);
                        dispY.set(gridID.size() * (numDisplacementSets - 1) + idx, ougv.r[1]);
                        dispZ.set(gridID.size() * (numDisplacementSets - 1) + idx, ougv.r[2]);
                    }
                }
            }
            break;
        }
        }
        break;

    //---------------
    // element forces
    //---------------
    case OEF1X:
#ifdef VERBOSE
        cout << numBytes << " bytes in OEF1X result set." << endl;
#endif

        switch (recordNr)
        {

        case DESCRIPTION:
        case RECORD0:
            // skip that
            nwds = 0;
            break;
        case RECORD1:
        {
            // detect element type
            int arr[10];
            memcpy(arr, block, 10 * intSize);
            resElementID = arr[2];
            nwds = arr[9];
            deviceCode = arr[0] % 10;
#ifdef VERBOSE
            fprintf(stdout, "Element type = %d\n", resElementID);
            fprintf(stdout, "Load set id = %d\n", arr[4]);
            fprintf(stdout, "Format code = %d\n", arr[8]);
            fprintf(stdout, "Number of words per record = %d\n", nwds);
            fprintf(stdout, "DeviceCode = %d, %d\n", deviceCode, arr[0]);
#endif
            break;
        }

        case RECORD2:
        {
            // ------------------------------------------------------------
            // OEF1X result record2
            // ------------------------------------------------------------
            struct
            {
                int id; // the element id
                float v[16]; // the data
            } oef;
            // ------------------------------------------------------------

            assert(0 && "Something is missing here.");
            //recSize is uninitialized. What should happen?
            recSize = 0;
            if (recSize == 0)
                bytesToRead = 0;
            while (bytesToRead > 0)
            {
                recSize = nwds * wordSize;
                if (flBrokenRecord)
                {
#ifdef VERBOSE
                    fprintf(stdout, "OEF1 broken record detected.\n");
#endif
                    memcpy(&record[recPartSize], block, POS(recSize - recPartSize));
                    bytesToRead = bytesToRead - POS(recSize - recPartSize);
                    flBrokenRecord = false;
                    dataPtr = record;
                    validRecord = true;
                }
                else if (bytesToRead < recSize)
                {
#ifdef VERBOSE
                    fprintf(stdout, "OEF1 assemble broken record.\n");
#endif
                    recPartSize = bytesToRead;
                    memcpy(record, &block[numBytes - bytesToRead], bytesToRead);
                    bytesToRead = 0;
                    flBrokenRecord = true;
                    validRecord = false;
                }
                else
                {
                    dataPtr = &block[numBytes - bytesToRead];
                    correct = 0;

#ifdef VERBOSE
                    test_device = (int *)dataPtr;
#endif

                    while (dataPtr[0] != 0)
                    {
                        correct++;
                        dataPtr = &block[numBytes - bytesToRead + correct];
#ifdef VERBOSE

                        cerr << *test_device << " " << endl;
                        test_device = (int *)dataPtr;
#endif
                    }
                    bytesToRead -= recSize + correct;
                    validRecord = true;
                }
                if (validRecord)
                {
#ifdef _WIN32
                    memcpy(&oef, dataPtr, std::min(recSize, (int)(sizeof(int) + 16 * sizeof(float))));
#else
                    memcpy(&oef, dataPtr, MIN(recSize, sizeof(int) + 16 * sizeof(float)));
#endif
                    // check list existence
                    if (efX.size() == 0)
                    {
                        efX.newSize(elementID.size());
                        efY.newSize(elementID.size());
                        efZ.newSize(elementID.size());
                        efX.init(0.0f);
                        efY.init(0.0f);
                        efZ.init(0.0f);
                        sendInfo("Element force OEF detected.");
                    }
                    if (elementID.size() > 0)
                        idx = elementID.findBinary(oef.id / 10);
                    else
                        break;
#ifdef VERBOSE
                    fprintf(stdout, "Element force for element %d\n", oef.id / 10);
#endif
                    if (idx != -1)
                    {
                        switch (resElementID)
                        {
                        case CBAR:
#ifdef VERBOSE
                            fprintf(stdout, "Try to add %f as element force (CBAR type).\n", oef.v[8]);
#endif
                            break;

                        case CQUAD4:
#ifdef VERBOSE
                            fprintf(stdout, "Try to add %f as element force (CQUAD4 type).\n", oef.v[8]);
#endif
                            break;

                        case CTRIA1:
#ifdef VERBOSE
                            fprintf(stdout, "Try to add %f as element force (CTRIA1 type).\n", oef.v[8]);
#endif
                            break;
                        }
                    }
                }
            }
            nwds = 0;
            break;
        }
        }

        break;

    //---------------
    // element stress
    //---------------
    case OES1X:
#ifdef VERBOSE
        cout << numBytes << " bytes in OES1X result set." << endl;
        cout << " recodNr.: " << recordNr << " " << block << endl;
#endif

        switch (recordNr)
        {

        case DESCRIPTION:
        case RECORD0:
            // skip that
            break;
        case RECORD1:
        {
            // detect element type
            int arr[10];
            memcpy(arr, block, 10 * intSize);
            resElementID = arr[2];
            deviceCode = arr[0] % 10;
            nwds = arr[9];
#ifdef VERBOSE
            fprintf(stdout, "Element type = %d\n", resElementID);
            fprintf(stdout, "Load set id = %d\n", arr[4]);
            fprintf(stdout, "Format code = %d\n", arr[8]);
            fprintf(stdout, "Number of words per record = %d\n", nwds);
            fprintf(stdout, "DeviceCode + 10*approach = %d\n", deviceCode);
#endif
            break;
        }

        case RECORD2:
        {
            // ------------------------------------------------------------
            // OES1X result record2
            // ------------------------------------------------------------
            struct
            {
                int id; // the element id
                float v[16]; // the data
            } oes;
            // ------------------------------------------------------------

            while (bytesToRead > 0)
            {
                recSize = nwds * wordSize;
                //cout << " bytesToRead " << bytesToRead << " " << stressList.size() << endl ;
                if (flBrokenRecord)
                {
                    memcpy(&record[recPartSize], block, POS(recSize - recPartSize));
                    bytesToRead = bytesToRead - POS(recSize - recPartSize);
                    flBrokenRecord = false;
                    dataPtr = record;
                    validRecord = true;
                }
                else if (bytesToRead < recSize)
                {
                    recPartSize = bytesToRead;
                    memcpy(record, &block[numBytes - bytesToRead], bytesToRead);
                    bytesToRead = 0;
                    flBrokenRecord = true;
                    validRecord = false;
                }
                else
                {
                    correct = 0;
                    dataPtr = &block[numBytes - bytesToRead];
                    while (dataPtr[0] != 0)
                    {
                        correct++;
                        if (numBytes - bytesToRead + correct >= 0)
                            dataPtr = &block[numBytes - bytesToRead + correct];
                        else
                            return true;
                    }
                    bytesToRead -= recSize + correct;
                    validRecord = true;
                }
                //cmwmodi29.2.00: the oes structure could be to small for other elements!!

                if (resElementID != CQUAD4 && resElementID != CTRIA1)
                {
                    sendInfo("Elementtype: %d not supported ", resElementID);
                    break;
                }
                //cmwmodi29.2.00:

                if (validRecord)
                {

                    memcpy(&oes, dataPtr, recSize);
                    // check list existence

                    if (stressList.size() == 0)
                    {
                        numStressSets = 1;
                        stressList.newSize(elementID.size());
                        stressList.init(0.0f);
                    }

#ifdef VERBOSE
                    fprintf(stdout, "Element stress for element %d\n", oes.id / 10);
#endif
                    switch (resElementID)
                    {

                    case CQUAD4:
                    case CTRIA1:
                        // calculate van Mises stress
                        idx = elementID.findBinary(oes.id / 10);
                        if (idx != -1)
                        {
                            numStresses++;
                            if (numStresses > (numStressSets * elementID.size()))
                            {
                                // grow
                                numStressSets++;
                                stressList.newSize(elementID.size() * numStressSets);
                            }
                            if (fibreDistance == 1)
                            {
                                float vms = (float)sqrt(0.5f * square(oes.v[1] - oes.v[2]) + square(oes.v[1]) + square(oes.v[2]) + 3 * square(oes.v[3]));
                                stressList.set(elementID.size() * (numStressSets - 1) + idx, vms);
                            }
                            else
                            {
                                float vms = (float)sqrt(0.5f * square(oes.v[9] - oes.v[10]) + square(oes.v[9]) + square(oes.v[10]) + 3 * square(oes.v[11]));
                                stressList.set(elementID.size() * (numStressSets - 1) + idx, vms);
                            }
                        }
                        //else
                        // return true;
                        break;
                    }
                }

                //cout << " bytesToRead after" << bytesToRead << " " << stressList.size() << endl ;
                if (bytesToRead < 0)
                    return true;
            }
            break;
        }
        }
        break;

    default:
        // we ignore the rest
        break;
    }

    return true;
}

//-------------------------------------------------------------------------
//
// transform a coordinate
//
//-------------------------------------------------------------------------
int ReadNastran::transformCoordinate(int csid, double pin[3], float *pout)
{
    float pn[3] = { 0.f, 0.f, 0.f };

    if (csid == 0)
    {

        // this is the world coordinate system => coordinate unchanged
        pout[0] = (float)pin[0];
        pout[1] = (float)pin[1];
        pout[2] = (float)pin[2];

        return true;
    }

    int index = csID.find(csid);
    if (index == -1)
        return false;

    switch (csList[index]->type)
    {
    case CYLINDRICAL:
        pn[0] = (float)(pin[0] * cos(d2r(pin[1])));
        pn[1] = (float)(pin[0] * sin(d2r(pin[1])));
        pn[2] = (float)pin[2];
        break;
    case RECTANGULAR:
        // nothing to do here, so only copy it
        pn[0] = (float)pin[0];
        pn[1] = (float)pin[1];
        pn[2] = (float)pin[2];
        break;
    case SPHERICAL:
        pn[0] = (float)(pin[0] * cos(d2r(pin[1])) * cos(d2r(pin[2])));
        pn[1] = (float)(pin[0] * sin(d2r(pin[1])) * sin(d2r(pin[2])));
        pn[2] = (float)(pin[0] * sin(d2r(pin[1])));
        break;
    default:
        fprintf(stderr, "unknown coord type in ReadNastran::transformCoordinate\n");
        break;
    }

    pout[0] = (float)(csList[index]->t[0] + pn[0] * csList[index]->r[0][0] + pn[1] * csList[index]->r[0][1] + pn[2] * csList[index]->r[0][2]);
    pout[1] = (float)(csList[index]->t[1] + pn[0] * csList[index]->r[1][0] + pn[1] * csList[index]->r[1][1] + pn[2] * csList[index]->r[1][2]);
    pout[2] = (float)(csList[index]->t[2] + pn[0] * csList[index]->r[2][0] + pn[1] * csList[index]->r[2][1] + pn[2] * csList[index]->r[2][2]);
    return true;
}

//-------------------------------------------------------------------------
//
// calculate the transformation matrix
//
//-------------------------------------------------------------------------
void ReadNastran::calcTransformMatrix(CoordinateSystem2 *cs, double a[3], double b[3], double c[3])
{
    double xd[3], yd[3], zd[3];
    double xdd[3], ydd[3];

#ifdef DEEPVERBOSE
    fprintf(stdout, "a1 = %lf\n", a[0]);
    fprintf(stdout, "a2 = %lf\n", a[1]);
    fprintf(stdout, "a3 = %lf\n", a[2]);
    fprintf(stdout, "b1 = %lf\n", b[0]);
    fprintf(stdout, "b2 = %lf\n", b[1]);
    fprintf(stdout, "b3 = %lf\n", b[2]);
    fprintf(stdout, "c1 = %lf\n", c[0]);
    fprintf(stdout, "c2 = %lf\n", c[1]);
    fprintf(stdout, "c3 = %lf\n", c[2]);
#endif

    // perform B - A
    zd[0] = b[0] - a[0];
    zd[1] = b[1] - a[1];
    zd[2] = b[2] - a[2];

    float zlength = (float)sqrt(zd[0] * zd[0] + zd[1] * zd[1] + zd[2] * zd[2]);

    zd[0] /= zlength;
    zd[1] /= zlength;
    zd[2] /= zlength;
    // perform C - A
    xdd[0] = c[0] - a[0];
    xdd[1] = c[1] - a[1];
    xdd[2] = c[2] - a[2];

    float xlength = (float)sqrt(xdd[0] * xdd[0] + xdd[1] * xdd[1] + xdd[2] * xdd[2]);

    xdd[0] /= xlength;
    xdd[1] /= xlength;
    xdd[2] /= xlength;

    // perform zd x xd
    ydd[0] = zd[1] * xdd[2] - zd[2] * xdd[1];
    ydd[1] = zd[2] * xdd[0] - zd[0] * xdd[2];
    ydd[2] = zd[0] * xdd[1] - zd[1] * xdd[0];

    float ylength = (float)sqrt(ydd[0] * ydd[0] + ydd[1] * ydd[1] + ydd[2] * ydd[2]);

    yd[0] = ydd[0] / ylength;
    yd[1] = ydd[1] / ylength;
    yd[2] = ydd[2] / ylength;

    // perform yd x zd

    xd[0] = yd[1] * zd[2] - yd[2] * zd[1];
    xd[1] = yd[2] * zd[0] - yd[0] * zd[2];
    xd[2] = yd[0] * zd[1] - yd[1] * zd[0];

    xlength = (float)sqrt(xd[0] * xd[0] + xd[1] * xd[1] + xd[2] * xd[2]);

    xd[0] /= xlength;
    xd[1] /= xlength;
    xd[2] /= xlength;

#ifdef DEEPVERBOSE
    fprintf(stdout, "xd[0] = %lf\n", xd[0]);
    fprintf(stdout, "xd[1] = %lf\n", xd[1]);
    fprintf(stdout, "xd[2] = %lf\n", xd[2]);
    fprintf(stdout, "yd[0] = %lf\n", yd[0]);
    fprintf(stdout, "yd[1] = %lf\n", yd[1]);
    fprintf(stdout, "yd[2] = %lf\n", yd[2]);
    fprintf(stdout, "zd[0] = %lf\n", zd[0]);
    fprintf(stdout, "zd[1] = %lf\n", zd[1]);
    fprintf(stdout, "zd[2] = %lf\n", zd[2]);
#endif
    // set the transform matrix
    cs->r[0][0] = xd[0];
    cs->r[1][0] = xd[1];
    cs->r[2][0] = xd[2];
    cs->r[0][1] = yd[0];
    cs->r[1][1] = yd[1];
    cs->r[2][1] = yd[2];
    cs->r[0][2] = zd[0];
    cs->r[1][2] = zd[1];
    cs->r[2][2] = zd[2];

    // the translation vector
    cs->t[0] = a[0];
    cs->t[1] = a[1];
    cs->t[2] = a[2];
}

void ReadNastran::swap_double(double *d, int no)
{
    int32_t *tmp0, *tmp1;
    int32_t swap;

    for (int i = 0; i < no; i++, d++)
    {
        tmp0 = (int32_t *)d;
        tmp1 = tmp0 + 1;

        swap = *tmp0;
        *tmp0 = *tmp1;
        *tmp1 = swap;
    }
}

MODULE_MAIN(IO, ReadNastran)
