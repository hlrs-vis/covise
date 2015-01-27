/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Animate.h"
#include "Matrix.h"

Animate::Animate(int argc, char *argv[])
    : coModule(argc, argv, "Rotate timedependent grid and velocities")
{
    p_inGrid = addInputPort("Grid", "UnstructuredGrid|Polygons|StructuredGrid|RectilinearGrid", "Grid");
    p_inVelo = addInputPort("velocity", "Vec3", "velocity");
    p_inGrid->setRequired(1);
    p_inVelo->setRequired(0);
    p_outGrid = addOutputPort("outGrid", "UnstructuredGrid|UniformGrid|StructuredGrid|RectilinearGrid", "Grid");
    // 0 is as many as velocities set
    p_outVelo = addOutputPort("outVelo", "Vec3", "velocity");
    p_outVelo->setDependencyPort(p_inVelo);
    p_angle = addFloatParam("angle", "angle of rotation in degrees per timestep");
    p_angle->setValue(7.2f); // 0 is as many as velocities set
    p_startAngle = addFloatParam("startAngle", "start angle of rotation in degrees");
    p_startAngle->setValue(7.2f);
    p_axis = addFloatVectorParam("axis", "axis of rotation");
    p_axis->setValue(0, 1, 0);
    p_pos = addFloatVectorParam("pos", "position of rotation axis");
    p_pos->setValue(0, 0, 0);
};

int Animate::Diagnose()
{
    if (!(setGeoIn = (coDoSet *)p_inGrid->getCurrentObject()))
    {
        sendError("Did not receive any input object");
        return -1;
    }
    else if (!setGeoIn->objectOk())
    {
        sendError("Did not get a good input object");
        return -1;
    }
    if (!(setIn = (coDoSet *)p_inVelo->getCurrentObject()))
    {
        sendWarning("No Velocity to rotate");
    }

    int nt = 0, ntg = 0;
    // if we have a hierarchy of sets, the upper level should be the timesteps
    ntg = setGeoIn->getNumElements();
    numTimesteps = ntg;
    if (setIn)
    {
        nt = setIn->getNumElements();
        if (numTimesteps != nt)
        {
            sendError("Numer of Timesteps do not match");
            return -1;
        }
    }
    if (!dynamic_cast<const coDoSet *>(p_inGrid->getCurrentObject()))
    {
        sendError("only sets are acceptable for input");
        return -1;
    }
    return 0;
}

int Animate::compute(const char *)
{
    int i;
    Matrix mat;

    if (Diagnose() < 0)
        return FAIL;

    int numVelos = 0;
    int numGeos = 0;
    const coDistributedObject *const *inGeos = setGeoIn->getAllElements(&numGeos);

    polygons = false;
    trianglestrips = false;
    if (dynamic_cast<const coDoPolygons *>(inGeos[0]))
        polygons = true;

    if (inGeos[0]->isType("SETELE"))
    {
        fprintf(stderr, "we have another set!\n");
        return -1;
    }

    if (inGeos[0]->isType("TRIANG"))
    {
        fprintf(stderr, "we have triangle strips!\n");
    }
    if (inGeos[0]->isType("POLYGN"))
    {
        fprintf(stderr, "we have polygons!\n");
    }

    coDistributedObject **set_list = new coDistributedObject *[numTimesteps + 1];
    set_list[numTimesteps] = 0;
    currentAngle = p_startAngle->getValue();

    const coDistributedObject *const *inVelos = NULL;
    coDistributedObject **setVelo = NULL;

    if (setIn)
    {
        inVelos = setIn->getAllElements(&numVelos);
        setVelo = new coDistributedObject *[numTimesteps + 1];
        setVelo[numTimesteps] = 0;
    }

    for (i = 0; i < numTimesteps; ++i)
    {
        float pos[3];
        p_pos->getValue(pos[0], pos[1], pos[2]);
        float axis[3];
        p_axis->getValue(axis[0], axis[1], axis[2]);
        mat.RotateMatrix(currentAngle, pos, axis);
        currentAngle += p_angle->getValue();
        int no_pl, no_vl, no_points;
        int *plist, *vlist, *tlist = NULL;
        float *xCoordsIn, *yCoordsIn, *zCoordsIn;
        float *xCoords, *yCoords, *zCoords;
        if (polygons)
        {
            polyInObj = (coDoPolygons *)inGeos[i];
            no_pl = polyInObj->getNumPolygons();
            no_vl = polyInObj->getNumVertices();
            no_points = polyInObj->getNumPoints();
            if ((xCoords = new float[no_points]) == NULL)
                cout << "Not enough memory for array xCoords!" << endl;
            if ((yCoords = new float[no_points]) == NULL)
                cout << "Not enough memory for array yCoords!" << endl;
            if ((zCoords = new float[no_points]) == NULL)
                cout << "Not enough memory for array zCoords!" << endl;

            polyInObj->getAddresses(&xCoordsIn, &yCoordsIn, &zCoordsIn, &vlist, &plist);
        }
        else
        {
            gridInObj = (coDoUnstructuredGrid *)inGeos[i];
            gridInObj->getGridSize(&no_pl, &no_vl, &no_points);
            if ((xCoords = new float[no_points]) == NULL)
                cout << "Not enough memory for array xCoords!" << endl;
            if ((yCoords = new float[no_points]) == NULL)
                cout << "Not enough memory for array yCoords!" << endl;
            if ((zCoords = new float[no_points]) == NULL)
                cout << "Not enough memory for array zCoords!" << endl;

            gridInObj->getAddresses(&plist, &vlist, &xCoordsIn, &yCoordsIn, &zCoordsIn);
            gridInObj->getTypeList(&tlist);
        }

        mat.transformCoordinates(no_points, xCoords, yCoords, zCoords, xCoordsIn, yCoordsIn, zCoordsIn);

        int numDigits;
        if (i < 10)
            numDigits = 1;
        else
            numDigits = (int)ceil(log10f((float)(i + 1)));

        char *name = new char[strlen(p_outGrid->getObjName()) + 2 + numDigits];
        sprintf(name, "%s_%d", p_outGrid->getObjName(), i);
        if (polygons)
        {
            if ((polyOutObj = new coDoPolygons(name, no_points, xCoords, yCoords, zCoords, no_vl, vlist, no_pl, plist)) == NULL)
                sendError("Not enough memory for polyOutObj!");
            set_list[i] = polyOutObj;
        }
        else
        {
            if ((gridOutObj = new coDoUnstructuredGrid(name, no_pl, no_vl, no_points, plist, vlist, xCoords, yCoords, zCoords, tlist)) == NULL)
                sendError("Not enough memory for gridOutObj!");
            set_list[i] = gridOutObj;
        }
        delete[] name;

        if (numVelos)
        {
            veloInObj = (coDoVec3 *)inVelos[i];

            name = new char[strlen(p_outVelo->getObjName()) + 2 + numDigits];
            no_points = veloInObj->getNumPoints();
            veloInObj->getAddresses(&xCoordsIn, &yCoordsIn, &zCoordsIn);
            sprintf(name, "%s_%d", p_outVelo->getObjName(), i);
            mat.transformCoordinates(no_points, xCoords, yCoords, zCoords, xCoordsIn, yCoordsIn, zCoordsIn);
            veloOutObj = new coDoVec3(name, no_points, xCoords, yCoords, zCoords);
            delete[] name;

            setVelo[i] = veloOutObj;
        }

        delete[] xCoords;
        delete[] yCoords;
        delete[] zCoords;
    }

    setOut = new coDoSet(p_outGrid->getObjName(), set_list);
    delete[] set_list;

    char buf[16];
    sprintf(buf, "1 %d", numTimesteps);
    setOut->addAttribute("TIMESTEP", buf);
    p_outGrid->setCurrentObject(setOut);

    if (numVelos)
    {
        setVeloOut = new coDoSet(p_outVelo->getObjName(), setVelo);
        delete[] setVelo;
        p_outVelo->setCurrentObject(setVeloOut);
        setVeloOut->addAttribute("TIMESTEP", buf);
    }

    return SUCCESS;
}

MODULE_MAIN(Tools, Animate)
