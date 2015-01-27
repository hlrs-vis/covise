/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                   	      (C)1999 RUS **
 **                                                                        **
 ** Description: Read Optres V6.0C binary files      	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: Uwe Woessner                                                   **
 **                                                                        **
 ** History:                                                               **
 ** December 99         v1                                                    **                               **
 **                                                                        **
\**************************************************************************/
#include "ReadOptres.h"
#include <util/coRestraint.h>
#include <ctype.h>

#include <stdio.h>

#define D_NONE 1
#define D_MATERIAL_NUM 2
#define D_THICKNESS 3
#define D_THICKING_VELOCITY 4
#define D_INITIAL_ENERGY 5
#define D_K_F 6
#define D_SIGMA_M 7
#define D_PHI_V 8
#define D_PLASTIC_STRAIN 9
#define D_PLASTIC_STRAIN_VELOCITY 10
#define D_EQUIVALENT_STRESS 11
#define D_SPEED 12
#define D_NORMAL_CONTACT_P 13
#define D_TANGENTIAL_CONTACT_P 14
#define D_Crush 15

#define NS 1
#define NV 2
#define ES 3
#define EV 4
const int dataType[] = // 0 if variable is per element, 1 if variable is per node
    {
      0, 0, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, NV, NS, NS, NS
    };

#include "optres.h"

int main(int argc, char *argv[])
{
    ReadOptres *application = new ReadOptres();
    application->start(argc, argv);

    return 0;
}

ReadOptres::ReadOptres()
{

    int i;
    char buf[1024];
    char buf2[1024];

    // this info appears in the module setup window
    set_module_description("Read Optres V6.0 binary files");

    // select the file name with a file browser
    p_filename = addFileBrowserParam("inFile", "V6.0 Optres file");
    const char *covisedir = getenv("COVISEDIR");
    if (!covisedir)
        covisedir = ".";
    strcpy(buf, covisedir);
    strcat(buf, "/data/sfb374/ifu/optres/Napf_Huhn_1");
    p_filename->setValue(buf, "*");

    p_numTimesteps = addInt32Param("numTimesteps", "Maximum Number of Timesteps");
    p_numTimesteps->setValue(200);

    // the output ports
    p_bars = addOutputPort("bars", "coDoLines", "Bars Objects");
    p_patches = addOutputPort("patches", "coDoPolygons", "flat Objects");
    p_volumes = addOutputPort("volumes", "coDoUnstructuredGrid", "Volume Objects");
    p_geometry = addOutputPort("geometry", "coDoPolygons", "selected geometry objects(no data)");
    for (i = 0; i < NUM_PORTS; i++)
    {
        sprintf(buf, "bar_data%d", i);
        sprintf(buf2, "data[%d] on bars", i);
        p_bar_data[i] = addOutputPort(buf, "coDoFloat|coDoVec3", buf2);
    }
    for (i = 0; i < NUM_PORTS; i++)
    {
        sprintf(buf, "patch_data%d", i);
        sprintf(buf2, "data[%d] on patches", i);
        p_patch_data[i] = addOutputPort(buf, "coDoFloat|coDoVec3", buf2);
    }
    for (i = 0; i < NUM_PORTS; i++)
    {
        sprintf(buf, "volume_data%d", i);
        sprintf(buf2, "data[%d] on volumes", i);
        p_volume_data[i] = addOutputPort(buf, "coDoFloat|coDoVec3", buf2);
    }

    // the data coice
    const char *choiceVal[] = {
        "--None--", "Material Number", "Thickness",
        "Thicking Velocity", "Initial Energy",
        "v. Mises Kf", "Sigma m",
        "Phi v", "Plastic Strain",
        "Plastic Strain Velocity", "Equivalent Stress",
        "Speed", "Normal Contact Pressure",
        "Tangential Contact Pressure", "Crush"
    };
    for (i = 0; i < NUM_PORTS; i++)
    {
        sprintf(buf, "dataSelection%d", i);
        sprintf(buf2, "data to be read in on port [%d]", i);
        p_selection[i] = addChoiceParam(buf, buf2);
        p_selection[i]->setValue(14, choiceVal, 0);
    }

    p_partSelection = addStringParam("selection", "Selected parts (m;n-o;...)");
    p_partSelection->setValue("1;2");
    p_partSelection->enable();
    p_partMaterial = addStringParam("material", "material for geometry parts");
    p_partMaterial->setValue("metal metal.30");
    p_dataSelection = addStringParam("dataSelection", "Selected parts with data(m;n-o;...)");
    p_dataSelection->setValue("6-1000");
    p_dataSelection->enable();

    // the layer coice
    p_layer = addChoiceParam("layer", "IntegrationPoint");
    const char *layerChoiceVal[] = { "lower", "middle", "upper" };
    p_layer->setValue(3, layerChoiceVal, 1);
}

void ReadOptres::postInst()
{
    int i;
    for (i = 0; i < NUM_PORTS; i++)
        p_selection[i]->show();

    //p_filename->show();
    p_partSelection->show();
    p_dataSelection->show();
}

int ReadOptres::compute()
{
    int i;
    char buf[1024];
    coRestraint selection;
    selection.add(p_partSelection->getValue());
    coRestraint dataPartSelection;
    dataPartSelection.add(p_dataSelection->getValue());

    //const char *objName = p_data->getObjName();

    /*****************************
    ** STATE/RESTART files data **
    ******************************/

    flagNdNb = 1; // Number of Nodes
    flagNdNum = 1; // Node Numbers
    flagNdCoord = 1; // Node Coordinates

    // Bars
    flagEl2Nb = 1; // Number of Elements
    flagEl2Num = 1; // Element Numbers, Material numbers aund Connectivities

    // Triangles
    flagEl3Nb = 1; // Number of Elements
    flagEl3Num = 1; // Element Numbers, Material numbers aund Connectivities

    // Quads
    flagEl4Nb = 1; // Number of Elements
    flagEl4Num = 1; // Element Numbers, Material numbers aund Connectivities

    // Hexaeders
    flagEl8Nb = 1; // Number of Elements
    flagEl8Num = 1; // Element Numbers, Material numbers aund Connectivities

    for (i = 0; i < NUM_PORTS; i++)
    {
        switch (p_selection[i]->getValue())
        {
        case D_MATERIAL_NUM:
            // material numbers are already read in
            break;
        case D_THICKNESS:
            flagEl3Thk = 1; // Thickness
            flagEl4Thk = 1; // Thickness
            break;
        case D_THICKING_VELOCITY:
            flagEl3DThk = 1; // Thicking Velocity
            flagEl4DThk = 1; // Thicking Velocity
            break;
        case D_INITIAL_ENERGY:
            flagEl3Eint = 1; // Initial Energy
            flagEl4Eint = 1; // Initial Energy
            break;
        case D_K_F:
            flagEl3Thk = 1; // Thickness
            flagEl4Thk = 1; // Thickness
            flagEl3Sig = 1; // Stress Tensor
            flagEl4Sig = 1; // Stress Tensor
            break;
        case D_SIGMA_M:
            flagEl3Thk = 1; // Thickness
            flagEl4Thk = 1; // Thickness
            flagEl3Sig = 1; // Stress Tensor
            flagEl4Sig = 1; // Stress Tensor
            break;
        case D_PHI_V:
            flagEl3Thk = 1; // Thickness
            flagEl4Thk = 1; // Thickness
            flagEl3Sig = 1; // Stress Tensor
            flagEl4Sig = 1; // Stress Tensor
            break;
        case D_PLASTIC_STRAIN:
            flagEl3Eps = 1; // Plastic Strain
            flagEl4Eps = 1; // Plastic Strain
            break;
        case D_PLASTIC_STRAIN_VELOCITY:
            flagEl3Epp = 1; // Plastic Strain
            flagEl4Epp = 1; // Plastic Strain
            break;
        case D_EQUIVALENT_STRESS:
            flagEl3Seq = 1; // Equivalent Stress
            flagEl4Seq = 1; // Equivalent Stress
            flagEl8Seq = 1; // Equivalent Stress
            break;
        case D_SPEED:
            flagNdSpeed = 1; // Node Speed
            break;
        case D_NORMAL_CONTACT_P:
            flagNdNorPress = 1; // Normal Pressure
            break;
        case D_TANGENTIAL_CONTACT_P:
            flagNdTgPress = 1; // Tangential Pressure
            break;
        case D_Crush:
            flagNdCrush = 1; // Crush
            break;
        default:
            break;
        }
    }
    int numTimesteps = p_numTimesteps->getValue();

    coDistributedObject **Bar_sets = new coDistributedObject *[numTimesteps + 1];
    coDistributedObject **Patch_sets = new coDistributedObject *[numTimesteps + 1];
    coDistributedObject **Volume_sets = new coDistributedObject *[numTimesteps + 1];
    coDistributedObject **Geometry_sets = new coDistributedObject *[numTimesteps + 1];
    coDistributedObject ***BarData_sets = new coDistributedObject **[NUM_PORTS];
    coDistributedObject ***PatchData_sets = new coDistributedObject **[NUM_PORTS];
    coDistributedObject ***VolumeData_sets = new coDistributedObject **[NUM_PORTS];
    for (i = 0; i < NUM_PORTS; i++)
    {
        BarData_sets[i] = new coDistributedObject *[numTimesteps + 1];
        PatchData_sets[i] = new coDistributedObject *[numTimesteps + 1];
        VolumeData_sets[i] = new coDistributedObject *[numTimesteps + 1];
        BarData_sets[i][0] = NULL;
        PatchData_sets[i][0] = NULL;
        VolumeData_sets[i][0] = NULL;
    }
    Bar_sets[0] = NULL;
    Patch_sets[0] = NULL;
    Volume_sets[0] = NULL;
    Geometry_sets[0] = NULL;

    // now read the selected Data
    int err = 0;
    int timestep = 0;
    int lastTimestep = p_numTimesteps->getValue();
    char *name;
    char *baseName;
    char *nameEnd;
    int baseNumber;

    // create basename (name without number);
    baseName = new char[strlen(p_filename->getValue()) + 1];
    strcpy(baseName, p_filename->getValue());
    nameEnd = new char[strlen(baseName) + 1];
    name = new char[strlen(baseName) + 100];

    i = strlen(baseName) - 1;
    while ((baseName[i] < '0') || (baseName[i] > '9'))
        i--;

    // baseName[i] ist jetzt die letzte Ziffer, alles danach ist Endung
    strcpy(nameEnd, baseName + i + 1); // nameEnd= Endung;
    baseName[i + 1] = 0;
    while ((baseName[i] >= '0') && (baseName[i] <= '9'))
        i--;
    sscanf(baseName + i + 1, "%d", &baseNumber); //baseNumber = Aktueller Zeitschritt

    baseName[i + 1] = 0; // baseName ist jetzt fertig

    while ((err == 0) && (lastTimestep >= timestep))
    {
        sprintf(name, "%s%d", baseName, baseNumber + timestep);
        err = readStateFile(name);
        if (err == 1) // could not open file, so try "file"end
        {
            sprintf(name, "%send", baseName);
            err = readStateFile(name);
            lastTimestep = timestep;
        }
        if (err == 1)
        {
            if (timestep == 0)
            {
                sprintf(buf, "Could not open file: %s", name);
                Covise::sendError(buf);
                break;
            }
            else
            {
                // last timestep read, do nothing
                break;
            }
        }
        if (err == 0)
        {
            // no error, so all data is read, now create the COVISE objects
            sprintf(buf, "Reading Timestep %d", baseNumber + timestep);
            Covise::sendInfo(buf);

            int *vertexNumbers = new int[NdNb]; // array which holds new vertex numbers or -1
            // Size is number of total nodes
            int *twoNodeElementNumbers = NULL; // arrays which holds new element numbers or -1
            int *threeNodeElementNumbers = NULL; // Size is number of elements
            int *fourNodeElementNumbers = NULL;
            int *eightNodeElementNumbers = NULL;

            if (El8Nb)
                eightNodeElementNumbers = new int[El8Nb];

            memset(eightNodeElementNumbers, -1, El8Nb * sizeof(int));

            //
            //
            //   Geometry parts (if selection was made)
            //
            //

            if (selection.getNumGroups())
            {
                // Count number of polygons and vertices
                int numPoly = 0, numVert = 0, numNodes = 0;
                int oldMaterial = -1000;
                int selected = 0;
                memset(vertexNumbers, -1, NdNb * sizeof(int));
                for (i = 0; i < El3Nb; i++)
                {
                    if (El3Mat[i] != oldMaterial)
                    {
                        oldMaterial = El3Mat[i];
                        selected = selection(oldMaterial);
                    }
                    if (selected)
                    {
                        numPoly++;
                        if (vertexNumbers[El3N1[i]] < 0)
                        {
                            vertexNumbers[El3N1[i]] = numNodes;
                            numNodes++;
                        }
                        if (vertexNumbers[El3N2[i]] < 0)
                        {
                            vertexNumbers[El3N2[i]] = numNodes;
                            numNodes++;
                        }
                        if (vertexNumbers[El3N3[i]] < 0)
                        {
                            vertexNumbers[El3N3[i]] = numNodes;
                            numNodes++;
                        }
                        numVert += 3;
                    }
                }
                for (i = 0; i < El4Nb; i++)
                {
                    if (El4Mat[i] != oldMaterial)
                    {
                        oldMaterial = El4Mat[i];
                        selected = selection(oldMaterial);
                    }
                    if (selected)
                    {
                        numPoly++;
                        if (vertexNumbers[El4N1[i]] < 0)
                        {
                            vertexNumbers[El4N1[i]] = numNodes;
                            numNodes++;
                        }
                        if (vertexNumbers[El4N2[i]] < 0)
                        {
                            vertexNumbers[El4N2[i]] = numNodes;
                            numNodes++;
                        }
                        if (vertexNumbers[El4N3[i]] < 0)
                        {
                            vertexNumbers[El4N3[i]] = numNodes;
                            numNodes++;
                        }
                        if (vertexNumbers[El4N4[i]] < 0)
                        {
                            vertexNumbers[El4N4[i]] = numNodes;
                            numNodes++;
                        }
                        numVert += 4;
                    }
                }
                // if anything was selected, then create the output
                if (numPoly)
                {
                    sprintf(buf, "%s_%d", p_geometry->getObjName(), baseNumber + timestep);
                    int n = 0;
                    while (Geometry_sets[n])
                        n++;
                    Geometry_sets[n + 1] = NULL;
                    Geometry_sets[n] = new coDoPolygons(buf, numNodes, numVert, numPoly);
                    float *x, *y, *z;
                    int *vl, *pl;
                    ((coDoPolygons *)Geometry_sets[n])->getAddresses(&x, &y, &z, &vl, &pl);
                    /*Real*/
                    Geometry_sets[n]->addAttribute("vertexOrder", "2");
                    Geometry_sets[n]->addAttribute("MATERIAL", (char *)p_partMaterial->getValue());

                    numNodes = numVert = numPoly = 0;
                    memset(vertexNumbers, -1, NdNb * sizeof(int));
                    for (i = 0; i < El3Nb; i++)
                    {
                        if (El3Mat[i] != oldMaterial)
                        {
                            oldMaterial = El3Mat[i];
                            selected = selection(oldMaterial);
                        }
                        if (selected)
                        {
                            pl[numPoly] = numVert;
                            numPoly++;
                            if (vertexNumbers[El3N1[i]] < 0)
                            {
                                vertexNumbers[El3N1[i]] = numNodes;
                                x[numNodes] = NdX[El3N1[i]];
                                y[numNodes] = NdY[El3N1[i]];
                                z[numNodes] = NdZ[El3N1[i]];
                                numNodes++;
                            }
                            if (vertexNumbers[El3N2[i]] < 0)
                            {
                                vertexNumbers[El3N2[i]] = numNodes;
                                x[numNodes] = NdX[El3N2[i]];
                                y[numNodes] = NdY[El3N2[i]];
                                z[numNodes] = NdZ[El3N2[i]];
                                numNodes++;
                            }
                            if (vertexNumbers[El3N3[i]] < 0)
                            {
                                vertexNumbers[El3N3[i]] = numNodes;
                                x[numNodes] = NdX[El3N3[i]];
                                y[numNodes] = NdY[El3N3[i]];
                                z[numNodes] = NdZ[El3N3[i]];
                                numNodes++;
                            }
                            vl[numVert] = vertexNumbers[El3N1[i]];
                            numVert++;
                            vl[numVert] = vertexNumbers[El3N2[i]];
                            numVert++;
                            vl[numVert] = vertexNumbers[El3N3[i]];
                            numVert++;
                        }
                    }
                    for (i = 0; i < El4Nb; i++)
                    {
                        if (El4Mat[i] != oldMaterial)
                        {
                            oldMaterial = El4Mat[i];
                            selected = selection(oldMaterial);
                        }
                        if (selected)
                        {
                            pl[numPoly] = numVert;
                            numPoly++;
                            if (vertexNumbers[El4N1[i]] < 0)
                            {
                                vertexNumbers[El4N1[i]] = numNodes;
                                x[numNodes] = NdX[El4N1[i]];
                                y[numNodes] = NdY[El4N1[i]];
                                z[numNodes] = NdZ[El4N1[i]];
                                numNodes++;
                            }
                            if (vertexNumbers[El4N2[i]] < 0)
                            {
                                vertexNumbers[El4N2[i]] = numNodes;
                                x[numNodes] = NdX[El4N2[i]];
                                y[numNodes] = NdY[El4N2[i]];
                                z[numNodes] = NdZ[El4N2[i]];
                                numNodes++;
                            }
                            if (vertexNumbers[El4N3[i]] < 0)
                            {
                                vertexNumbers[El4N3[i]] = numNodes;
                                x[numNodes] = NdX[El4N3[i]];
                                y[numNodes] = NdY[El4N3[i]];
                                z[numNodes] = NdZ[El4N3[i]];
                                numNodes++;
                            }
                            if (vertexNumbers[El4N4[i]] < 0)
                            {
                                vertexNumbers[El4N4[i]] = numNodes;
                                x[numNodes] = NdX[El4N4[i]];
                                y[numNodes] = NdY[El4N4[i]];
                                z[numNodes] = NdZ[El4N4[i]];
                                numNodes++;
                            }
                            vl[numVert] = vertexNumbers[El4N1[i]];
                            numVert++;
                            vl[numVert] = vertexNumbers[El4N2[i]];
                            numVert++;
                            vl[numVert] = vertexNumbers[El4N3[i]];
                            numVert++;
                            vl[numVert] = vertexNumbers[El4N4[i]];
                            numVert++;
                        }
                    }
                }
            }

            //
            //
            // now comes the part with data
            //
            //
            //

            if (El2Nb) // we do have 2 node elements
            {
                if (El2Nb)
                    twoNodeElementNumbers = new int[El2Nb];
                memset(twoNodeElementNumbers, -1, El2Nb * sizeof(int));
                /*i=0;
            while(Bar_sets[i]) i++;
            Bar_sets[i+1]=NULL;
            Bar_sets[i] = new coDoLines(*/
                delete[] twoNodeElementNumbers;
            }
            if ((El3Nb) || (El4Nb)) // we have patches
            {

                if (El3Nb)
                    threeNodeElementNumbers = new int[El3Nb];
                if (El4Nb)
                    fourNodeElementNumbers = new int[El4Nb];
                memset(threeNodeElementNumbers, -1, El3Nb * sizeof(int));
                memset(fourNodeElementNumbers, -1, El4Nb * sizeof(int));

                if (dataPartSelection.getNumGroups())
                {
                    // Count number of polygons and vertices
                    int numPoly = 0, numVert = 0, numNodes = 0;
                    int oldMaterial = -1000;
                    int selected = 0;
                    memset(vertexNumbers, -1, NdNb * sizeof(int));
                    int numthreeNode = 0;
                    for (i = 0; i < El3Nb; i++)
                    {
                        if (El3Mat[i] != oldMaterial)
                        {
                            oldMaterial = El3Mat[i];
                            selected = dataPartSelection(oldMaterial);
                        }
                        if (selected)
                        {
                            threeNodeElementNumbers[i] = numthreeNode;
                            numthreeNode++;
                            numPoly++;
                            if (vertexNumbers[El3N1[i]] < 0)
                            {
                                vertexNumbers[El3N1[i]] = numNodes;
                                numNodes++;
                            }
                            if (vertexNumbers[El3N2[i]] < 0)
                            {
                                vertexNumbers[El3N2[i]] = numNodes;
                                numNodes++;
                            }
                            if (vertexNumbers[El3N3[i]] < 0)
                            {
                                vertexNumbers[El3N3[i]] = numNodes;
                                numNodes++;
                            }
                            numVert += 3;
                        }
                    }
                    int numfourNode = 0;
                    for (i = 0; i < El4Nb; i++)
                    {
                        if (El4Mat[i] != oldMaterial)
                        {
                            oldMaterial = El4Mat[i];
                            selected = dataPartSelection(oldMaterial);
                        }
                        if (selected)
                        {
                            fourNodeElementNumbers[i] = numfourNode;
                            numfourNode++;
                            numPoly++;
                            if (vertexNumbers[El4N1[i]] < 0)
                            {
                                vertexNumbers[El4N1[i]] = numNodes;
                                numNodes++;
                            }
                            if (vertexNumbers[El4N2[i]] < 0)
                            {
                                vertexNumbers[El4N2[i]] = numNodes;
                                numNodes++;
                            }
                            if (vertexNumbers[El4N3[i]] < 0)
                            {
                                vertexNumbers[El4N3[i]] = numNodes;
                                numNodes++;
                            }
                            if (vertexNumbers[El4N4[i]] < 0)
                            {
                                vertexNumbers[El4N4[i]] = numNodes;
                                numNodes++;
                            }
                            numVert += 4;
                        }
                    }
                    // if anything was selected, then create the output
                    if (numPoly)
                    {
                        sprintf(buf, "%s_%d", p_patches->getObjName(), baseNumber + timestep);
                        int n = 0;
                        while (Patch_sets[n])
                            n++;
                        Patch_sets[n + 1] = NULL;
                        Patch_sets[n] = new coDoPolygons(buf, numNodes, numVert, numPoly);
                        float *x, *y, *z;
                        int *vl, *pl;
                        ((coDoPolygons *)Patch_sets[n])->getAddresses(&x, &y, &z, &vl, &pl);
                        Patch_sets[n]->addAttribute("vertexOrder", "2");
                        //Geometry_sets[n]->addAttribute("MATERIAL", (char *)p_partMaterial->getValue());

                        numNodes = numVert = numPoly = 0;
                        memset(vertexNumbers, -1, NdNb * sizeof(int));
                        for (i = 0; i < El3Nb; i++)
                        {
                            if (El3Mat[i] != oldMaterial)
                            {
                                oldMaterial = El3Mat[i];
                                selected = dataPartSelection(oldMaterial);
                            }
                            if (selected)
                            {
                                pl[numPoly] = numVert;
                                numPoly++;
                                if (vertexNumbers[El3N1[i]] < 0)
                                {
                                    vertexNumbers[El3N1[i]] = numNodes;
                                    x[numNodes] = NdX[El3N1[i]];
                                    y[numNodes] = NdY[El3N1[i]];
                                    z[numNodes] = NdZ[El3N1[i]];
                                    numNodes++;
                                }
                                if (vertexNumbers[El3N2[i]] < 0)
                                {
                                    vertexNumbers[El3N2[i]] = numNodes;
                                    x[numNodes] = NdX[El3N2[i]];
                                    y[numNodes] = NdY[El3N2[i]];
                                    z[numNodes] = NdZ[El3N2[i]];
                                    numNodes++;
                                }
                                if (vertexNumbers[El3N3[i]] < 0)
                                {
                                    vertexNumbers[El3N3[i]] = numNodes;
                                    x[numNodes] = NdX[El3N3[i]];
                                    y[numNodes] = NdY[El3N3[i]];
                                    z[numNodes] = NdZ[El3N3[i]];
                                    numNodes++;
                                }
                                vl[numVert] = vertexNumbers[El3N1[i]];
                                numVert++;
                                vl[numVert] = vertexNumbers[El3N2[i]];
                                numVert++;
                                vl[numVert] = vertexNumbers[El3N3[i]];
                                numVert++;
                            }
                        }
                        for (i = 0; i < El4Nb; i++)
                        {
                            if (El4Mat[i] != oldMaterial)
                            {
                                oldMaterial = El4Mat[i];
                                selected = dataPartSelection(oldMaterial);
                            }
                            if (selected)
                            {
                                pl[numPoly] = numVert;
                                numPoly++;
                                if (vertexNumbers[El4N1[i]] < 0)
                                {
                                    vertexNumbers[El4N1[i]] = numNodes;
                                    x[numNodes] = NdX[El4N1[i]];
                                    y[numNodes] = NdY[El4N1[i]];
                                    z[numNodes] = NdZ[El4N1[i]];
                                    numNodes++;
                                }
                                if (vertexNumbers[El4N2[i]] < 0)
                                {
                                    vertexNumbers[El4N2[i]] = numNodes;
                                    x[numNodes] = NdX[El4N2[i]];
                                    y[numNodes] = NdY[El4N2[i]];
                                    z[numNodes] = NdZ[El4N2[i]];
                                    numNodes++;
                                }
                                if (vertexNumbers[El4N3[i]] < 0)
                                {
                                    vertexNumbers[El4N3[i]] = numNodes;
                                    x[numNodes] = NdX[El4N3[i]];
                                    y[numNodes] = NdY[El4N3[i]];
                                    z[numNodes] = NdZ[El4N3[i]];
                                    numNodes++;
                                }
                                if (vertexNumbers[El4N4[i]] < 0)
                                {
                                    vertexNumbers[El4N4[i]] = numNodes;
                                    x[numNodes] = NdX[El4N4[i]];
                                    y[numNodes] = NdY[El4N4[i]];
                                    z[numNodes] = NdZ[El4N4[i]];
                                    numNodes++;
                                }
                                vl[numVert] = vertexNumbers[El4N1[i]];
                                numVert++;
                                vl[numVert] = vertexNumbers[El4N2[i]];
                                numVert++;
                                vl[numVert] = vertexNumbers[El4N3[i]];
                                numVert++;
                                vl[numVert] = vertexNumbers[El4N4[i]];
                                numVert++;
                            }
                        }
                        float *s, *u, *v, *w;
                        for (i = 0; i < NUM_PORTS; i++)
                        {
                            int n = 0, m = 0;
                            while (PatchData_sets[i][n])
                                n++;
                            PatchData_sets[i][n + 1] = NULL;

                            sprintf(buf, "%s_%d", p_patch_data[i]->getObjName(), baseNumber + timestep);
                            if (dataType[p_selection[i]->getValue()] == NS)
                            {
                                PatchData_sets[i][n] = new coDoFloat(buf, numNodes);
                                ((coDoFloat *)PatchData_sets[i][n])->getAddress(&s);
                            }
                            else if (dataType[p_selection[i]->getValue()] == ES)
                            {
                                PatchData_sets[i][n] = new coDoFloat(buf, numPoly);
                                ((coDoFloat *)PatchData_sets[i][n])->getAddress(&s);
                            }
                            else if (dataType[p_selection[i]->getValue()] == NV)
                            {
                                PatchData_sets[i][n] = new coDoVec3(buf, numNodes);
                                ((coDoVec3 *)PatchData_sets[i][n])->getAddresses(&u, &v, &w);
                            }
                            else if (dataType[p_selection[i]->getValue()] == EV)
                            {
                                PatchData_sets[i][n] = new coDoVec3(buf, numPoly);
                                ((coDoVec3 *)PatchData_sets[i][n])->getAddresses(&u, &v, &w);
                            }
                            else
                            {
                                continue;
                            }
                            switch (p_selection[i]->getValue())
                            {
                            case D_MATERIAL_NUM:
                                for (m = 0; m < El3Nb; m++)
                                {
                                    if (threeNodeElementNumbers[m] >= 0)
                                        s[threeNodeElementNumbers[m]] = El3Mat[m];
                                }
                                for (m = 0; m < El4Nb; m++)
                                {
                                    if (fourNodeElementNumbers[m] >= 0)
                                        s[fourNodeElementNumbers[m] + numthreeNode] = El4Mat[m];
                                }
                                break;
                            case D_THICKNESS:
                                for (m = 0; m < El3Nb; m++)
                                {
                                    if (threeNodeElementNumbers[m] >= 0)
                                        s[threeNodeElementNumbers[m]] = El3Thk[m];
                                }
                                for (m = 0; m < El4Nb; m++)
                                {
                                    if (fourNodeElementNumbers[m] >= 0)
                                        s[fourNodeElementNumbers[m] + numthreeNode] = El4Thk[m];
                                }
                                break;
                            case D_THICKING_VELOCITY:
                                for (m = 0; m < El3Nb; m++)
                                {
                                    if (threeNodeElementNumbers[m] >= 0)
                                        s[threeNodeElementNumbers[m]] = El3DThk[m];
                                }
                                for (m = 0; m < El4Nb; m++)
                                {
                                    if (fourNodeElementNumbers[m] >= 0)
                                        s[fourNodeElementNumbers[m] + numthreeNode] = El4DThk[m];
                                }
                                break;
                            case D_INITIAL_ENERGY:
                                for (m = 0; m < El3Nb; m++)
                                {
                                    if (threeNodeElementNumbers[m] >= 0)
                                        s[threeNodeElementNumbers[m]] = El3Eint[m];
                                }
                                for (m = 0; m < El4Nb; m++)
                                {
                                    if (fourNodeElementNumbers[m] >= 0)
                                        s[fourNodeElementNumbers[m] + numthreeNode] = El4Eint[m];
                                }
                                break;
                            case D_K_F:
                                if (p_layer->getValue() == 1)
                                {
                                    for (m = 0; m < El3Nb; m++)
                                    {
                                        if (threeNodeElementNumbers[m] >= 0)
                                            s[threeNodeElementNumbers[m]] = El3Thk[m] * sqrt(El3Sxx1[m] * El3Sxx1[m] + El3Syy1[m] * El3Syy1[m] + 3 * (El3Sxy1[m]));
                                    }
                                    for (m = 0; m < El4Nb; m++)
                                    {
                                        if (fourNodeElementNumbers[m] >= 0)
                                            s[threeNodeElementNumbers[m]] = El4Thk[m] * sqrt(El4Sxx1[m] * El4Sxx1[m] + El4Syy1[m] * El4Syy1[m] + 3 * (El4Sxy1[m]));
                                    }
                                }
                                else if (p_layer->getValue() == 2)
                                {
                                    for (m = 0; m < El3Nb; m++)
                                    {
                                        if (threeNodeElementNumbers[m] >= 0)
                                            s[threeNodeElementNumbers[m]] = El3Thk[m] * sqrt(El3Sxx3[m] * El3Sxx3[m] + El3Syy3[m] * El3Syy3[m] + 3 * (El3Sxy3[m]));
                                    }
                                    for (m = 0; m < El4Nb; m++)
                                    {
                                        if (fourNodeElementNumbers[m] >= 0)
                                            s[threeNodeElementNumbers[m]] = El4Thk[m] * sqrt(El4Sxx3[m] * El4Sxx3[m] + El4Syy3[m] * El4Syy3[m] + 3 * (El4Sxy3[m]));
                                    }
                                }
                                else if (p_layer->getValue() == 3)
                                {
                                    for (m = 0; m < El3Nb; m++)
                                    {
                                        if (threeNodeElementNumbers[m] >= 0)
                                            s[threeNodeElementNumbers[m]] = El3Thk[m] * sqrt(El3Sxx5[m] * El3Sxx5[m] + El3Syy5[m] * El3Syy5[m] + 3 * (El3Sxy5[m]));
                                    }
                                    for (m = 0; m < El4Nb; m++)
                                    {
                                        if (fourNodeElementNumbers[m] >= 0)
                                            s[threeNodeElementNumbers[m]] = El4Thk[m] * sqrt(El4Sxx5[m] * El4Sxx5[m] + El4Syy5[m] * El4Syy5[m] + 3 * (El4Sxy5[m]));
                                    }
                                }
                                break;
                            case D_SIGMA_M:
                                if (p_layer->getValue() == 1)
                                {
                                    for (m = 0; m < El3Nb; m++)
                                    {
                                        if (threeNodeElementNumbers[m] >= 0)
                                            s[threeNodeElementNumbers[m]] = El3Thk[m] * (El3Sxx1[m] + El3Syy1[m]) / 3.0;
                                    }
                                    for (m = 0; m < El4Nb; m++)
                                    {
                                        if (fourNodeElementNumbers[m] >= 0)
                                            s[threeNodeElementNumbers[m]] = El4Thk[m] * (El4Sxx1[m] + El4Syy1[m]) / 3.0;
                                    }
                                }
                                else if (p_layer->getValue() == 2)
                                {
                                    for (m = 0; m < El3Nb; m++)
                                    {
                                        if (threeNodeElementNumbers[m] >= 0)
                                            s[threeNodeElementNumbers[m]] = El3Thk[m] * (El3Sxx3[m] + El3Syy3[m]) / 3.0;
                                    }
                                    for (m = 0; m < El4Nb; m++)
                                    {
                                        if (fourNodeElementNumbers[m] >= 0)
                                            s[threeNodeElementNumbers[m]] = El4Thk[m] * (El4Sxx3[m] + El4Syy3[m]) / 3.0;
                                    }
                                }
                                else if (p_layer->getValue() == 3)
                                {
                                    for (m = 0; m < El3Nb; m++)
                                    {
                                        if (threeNodeElementNumbers[m] >= 0)
                                            s[threeNodeElementNumbers[m]] = El3Thk[m] * (El3Sxx5[m] + El3Syy5[m]) / 3.0;
                                    }
                                    for (m = 0; m < El4Nb; m++)
                                    {
                                        if (fourNodeElementNumbers[m] >= 0)
                                            s[threeNodeElementNumbers[m]] = El4Thk[m] * (El4Sxx5[m] + El4Syy5[m]) / 3.0;
                                    }
                                }
                                break;
                            case D_PHI_V:
                                flagEl3Sig = 1; // Stress Tensor
                                flagEl4Sig = 1; // Stress Tensor
                                break;
                            case D_PLASTIC_STRAIN:
                                if (p_layer->getValue() == 1)
                                {
                                    for (m = 0; m < El3Nb; m++)
                                    {
                                        if (threeNodeElementNumbers[m] >= 0)
                                            s[threeNodeElementNumbers[m]] = El3Ep1[m];
                                    }
                                    for (m = 0; m < El4Nb; m++)
                                    {
                                        if (fourNodeElementNumbers[m] >= 0)
                                            s[fourNodeElementNumbers[m] + numthreeNode] = El4Ep1[m];
                                    }
                                }
                                else if (p_layer->getValue() == 2)
                                {
                                    for (m = 0; m < El3Nb; m++)
                                    {
                                        if (threeNodeElementNumbers[m] >= 0)
                                            s[threeNodeElementNumbers[m]] = El3Ep3[m];
                                    }
                                    for (m = 0; m < El4Nb; m++)
                                    {
                                        if (fourNodeElementNumbers[m] >= 0)
                                            s[fourNodeElementNumbers[m] + numthreeNode] = El4Ep3[m];
                                    }
                                }
                                else if (p_layer->getValue() == 3)
                                {
                                    for (m = 0; m < El3Nb; m++)
                                    {
                                        if (threeNodeElementNumbers[m] >= 0)
                                            s[threeNodeElementNumbers[m]] = El3Ep5[m];
                                    }
                                    for (m = 0; m < El4Nb; m++)
                                    {
                                        if (fourNodeElementNumbers[m] >= 0)
                                            s[fourNodeElementNumbers[m] + numthreeNode] = El4Ep5[m];
                                    }
                                }
                                break;
                            case D_PLASTIC_STRAIN_VELOCITY:
                                if (p_layer->getValue() == 1)
                                {
                                    for (m = 0; m < El3Nb; m++)
                                    {
                                        if (threeNodeElementNumbers[m] >= 0)
                                            s[threeNodeElementNumbers[m]] = El3Epp1[m];
                                    }
                                    for (m = 0; m < El4Nb; m++)
                                    {
                                        if (fourNodeElementNumbers[m] >= 0)
                                            s[fourNodeElementNumbers[m] + numthreeNode] = El4Epp1[m];
                                    }
                                }
                                else if (p_layer->getValue() == 2)
                                {
                                    for (m = 0; m < El3Nb; m++)
                                    {
                                        if (threeNodeElementNumbers[m] >= 0)
                                            s[threeNodeElementNumbers[m]] = El3Epp3[m];
                                    }
                                    for (m = 0; m < El4Nb; m++)
                                    {
                                        if (fourNodeElementNumbers[m] >= 0)
                                            s[fourNodeElementNumbers[m] + numthreeNode] = El4Epp3[m];
                                    }
                                }
                                else if (p_layer->getValue() == 3)
                                {
                                    for (m = 0; m < El3Nb; m++)
                                    {
                                        if (threeNodeElementNumbers[m] >= 0)
                                            s[threeNodeElementNumbers[m]] = El3Epp5[m];
                                    }
                                    for (m = 0; m < El4Nb; m++)
                                    {
                                        if (fourNodeElementNumbers[m] >= 0)
                                            s[fourNodeElementNumbers[m] + numthreeNode] = El4Epp5[m];
                                    }
                                }
                                break;
                            case D_EQUIVALENT_STRESS:
                                if (p_layer->getValue() == 1)
                                {
                                    for (m = 0; m < El3Nb; m++)
                                    {
                                        if (threeNodeElementNumbers[m] >= 0)
                                            s[threeNodeElementNumbers[m]] = El3Seq1[m];
                                    }
                                    for (m = 0; m < El4Nb; m++)
                                    {
                                        if (fourNodeElementNumbers[m] >= 0)
                                            s[fourNodeElementNumbers[m] + numthreeNode] = El4Seq1[m];
                                    }
                                }
                                else if (p_layer->getValue() == 2)
                                {
                                    for (m = 0; m < El3Nb; m++)
                                    {
                                        if (threeNodeElementNumbers[m] >= 0)
                                            s[threeNodeElementNumbers[m]] = El3Seq3[m];
                                    }
                                    for (m = 0; m < El4Nb; m++)
                                    {
                                        if (fourNodeElementNumbers[m] >= 0)
                                            s[fourNodeElementNumbers[m] + numthreeNode] = El4Seq3[m];
                                    }
                                }
                                else if (p_layer->getValue() == 3)
                                {
                                    for (m = 0; m < El3Nb; m++)
                                    {
                                        if (threeNodeElementNumbers[m] >= 0)
                                            s[threeNodeElementNumbers[m]] = El3Seq5[m];
                                    }
                                    for (m = 0; m < El4Nb; m++)
                                    {
                                        if (fourNodeElementNumbers[m] >= 0)
                                            s[fourNodeElementNumbers[m] + numthreeNode] = El4Seq5[m];
                                    }
                                }
                                break;
                            case D_SPEED:
                                for (m = 0; m < NdNb; m++)
                                {
                                    if (vertexNumbers[m] >= 0)
                                    {
                                        u[vertexNumbers[m]] = NdVx[m];
                                        v[vertexNumbers[m]] = NdVy[m];
                                        w[vertexNumbers[m]] = NdVz[m];
                                    }
                                }
                                break;
                            case D_NORMAL_CONTACT_P:
                                for (m = 0; m < NdNb; m++)
                                {
                                    if (vertexNumbers[m] >= 0)
                                    {
                                        s[vertexNumbers[m]] = NdNorPress[m];
                                    }
                                }
                                break;
                            case D_TANGENTIAL_CONTACT_P:
                                for (m = 0; m < NdNb; m++)
                                {
                                    if (vertexNumbers[m] >= 0)
                                    {
                                        s[vertexNumbers[m]] = NdTgPress[m];
                                    }
                                }
                                break;
                            case D_Crush:
                                for (m = 0; m < NdNb; m++)
                                {
                                    if (vertexNumbers[m] >= 0)
                                    {
                                        s[vertexNumbers[m]] = NdCrush[m];
                                    }
                                }
                                break;
                            }
                        }
                    }
                }

                delete[] threeNodeElementNumbers;
                delete[] fourNodeElementNumbers;
            }
            if (El8Nb) // we do have 8 node elements
            {
                if (El8Nb)
                    eightNodeElementNumbers = new int[El8Nb];
                memset(eightNodeElementNumbers, -1, El8Nb * sizeof(int));

                delete[] eightNodeElementNumbers;
            }
            delete[] vertexNumbers;
        }
        else if (err == 1)
        {
            sprintf(buf, "Could not open file: %s", name);
            Covise::sendError(buf);
            break;
        }
        else if ((err > 2) && (err < 9))
        {
            sprintf(buf, "Error %d reading file %s", err, name);
            Covise::sendError(buf);
            break;
        }
        else if (err == 9)
        {
            sprintf(buf, "Wrong Version of file %s, expected V6.0C", name);
            Covise::sendError(buf);
            break;
        }
        else if (err == 10)
        {
            sprintf(buf, "no material corresponds to the material number of an element inf file %s", name);
            Covise::sendError(buf);
            break;
        }
        else
        {
            sprintf(buf, "Unknown Error %d reading file %s", err, name);
            Covise::sendError(buf);
            break;
        }

        freeStateData(); // free the data read in by the readStateFile command
        timestep++;
    }

    // Create Output objects
    coDoSet *set;
    if (Bar_sets[0])
    {
        set = new coDoSet(p_bars->getObjName(), Bar_sets);
        set->addAttribute("TIMESTEP", "1 100");
        p_bars->setCurrentObject(set);
    }
    if (Patch_sets[0])
    {
        set = new coDoSet(p_patches->getObjName(), Patch_sets);
        set->addAttribute("TIMESTEP", "1 100");
        p_patches->setCurrentObject(set);
    }
    if (Volume_sets[0])
    {
        set = new coDoSet(p_volumes->getObjName(), Volume_sets);
        set->addAttribute("TIMESTEP", "1 100");
        p_volumes->setCurrentObject(set);
    }
    if (Geometry_sets[0])
    {
        set = new coDoSet(p_geometry->getObjName(), Geometry_sets);
        set->addAttribute("TIMESTEP", "1 100");
        p_geometry->setCurrentObject(set);
    }
    for (i = 0; i < NUM_PORTS; i++)
    {
        if (BarData_sets[i][0])
            p_bar_data[i]->setCurrentObject(new coDoSet(p_bar_data[i]->getObjName(), BarData_sets[i]));
        if (PatchData_sets[i][0])
            p_patch_data[i]->setCurrentObject(new coDoSet(p_patch_data[i]->getObjName(), PatchData_sets[i]));
        if (VolumeData_sets[i][0])
            p_volume_data[i]->setCurrentObject(new coDoSet(p_volume_data[i]->getObjName(), VolumeData_sets[i]));
    }

    // Create Output arrays (___NOT___ the objects)
    delete[] Bar_sets;
    delete[] Patch_sets;
    delete[] Volume_sets;
    delete[] Geometry_sets;
    for (i = 0; i < NUM_PORTS; i++)
    {
        delete[] BarData_sets[i];
        delete[] PatchData_sets[i];
        delete[] VolumeData_sets[i];
    }
    delete[] BarData_sets;
    delete[] PatchData_sets;
    delete[] VolumeData_sets;

    return SUCCESS;
}

ReadOptres::~ReadOptres()
{
}
