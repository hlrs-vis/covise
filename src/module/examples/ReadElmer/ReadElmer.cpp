/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1998 RUS  **
 **                                                                        **
 ** Description: Read module Elmer data format         	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** History:                                                               **
 ** May   98	    U. Woessner	    V1.0                                      **
 ** March 99	    D. Rainer	    added comments                            **
 ** September 99 D. Rainer       new api                                   **
 *\**************************************************************************/

#include "ReadElmer.h"
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>

ReadElmer::ReadElmer(int argc, char *argv[])
    : coModule(argc, argv, "Read Elmer Data") // description in the module setup window
{
    // file browser parameter
    filenameParam = addFileBrowserParam("file_path", "Data file path");
    filenameParam->setValue("file_path", "data/visit/tube.dat *.dat*");

    // the output ports
    meshOutPort = addOutputPort("mesh", "UnstructuredGrid", "unstructured grid");
    velOutPort = addOutputPort("velocity", "Vec3", "velocity data");
    pressOutPort = addOutputPort("pressure", "Float", "pressure data");
    keOutPort = addOutputPort("ke", "Float", "kinetic energy data");
}

ReadElmer::~ReadElmer()
{
}

int ReadElmer::compute(const char *port)
{
    (void)port;

    FILE *fp;
    const char *fileName;
    int num_coord, num_elem, num_conn;
    int i, tmpi;
    char buf[300];

    int *el, *vl, *tl; // element list, vertex list, type list
    float *x_coord, *y_coord, *z_coord; // coordinate lists
    float *k, *u, *v, *w, *p; // data lists

    // names of the COVISE output objects
    const char *meshName;
    const char *velocName;
    const char *pressName;
    const char *keName;

    // the COVISE output objects (located in shared memory)
    coDoUnstructuredGrid *meshObj;
    coDoVec3 *velocObj;
    coDoFloat *pressObj;
    coDoFloat *keObj;

    // read the file browser parameter
    fileName = filenameParam->getValue();

    // open the file
    if ((fp = fopen(fileName, "r")) == NULL)
    {
        sendError("ERROR: Can't open file >> %s", fileName);
        return STOP_PIPELINE;
    }

    // get the ouput object names from the controller
    // the output object names have to be assigned by the controller
    meshName = meshOutPort->getObjName();
    velocName = velOutPort->getObjName();
    pressName = pressOutPort->getObjName();
    keName = keOutPort->getObjName();

    // read the dimensions from the header line
    if (fgets(buf, 300, fp) == NULL)
    {
        cerr << "ReadElmer::compute: fgets failed" << endl;
    }
    if (sscanf(buf, "%d%d", &num_coord, &num_elem) != 2)
    {
        cerr << "ReadElmer::compute: sscanf1 failed" << endl;
    }
    num_conn = num_elem * 8; // we have only hexaeder cells

    // create the unstructured grid object for the mesh
    if (meshName != NULL)
    {
        // the last parameters needs to be 1
        meshObj = new coDoUnstructuredGrid(meshName, num_elem, num_conn, num_coord, 1);
        meshOutPort->setCurrentObject(meshObj);

        if (meshObj->objectOk())
        {
            // get pointers to the element, vertex and coordinate lists
            meshObj->getAddresses(&el, &vl, &x_coord, &y_coord, &z_coord);

            // get a pointer to the type list
            meshObj->getTypeList(&tl);

            // read the coordinate lines
            for (i = 0; i < num_coord; i++)
            {
                // read the line which contains the coordinates and scan it
                if (fgets(buf, 300, fp) != NULL)
                {
                    if (sscanf(buf, "%f%f%f\n", x_coord, y_coord, z_coord) != 2)
                    {
                        cerr << "ReadElmer::compute: sscanf2 failed" << endl;
                    }
                    x_coord++;
                    y_coord++;
                    z_coord++;
                }
                else
                {
                    sendError("ERROR: unexpected end of file");
                    return STOP_PIPELINE;
                }
            }

            // read the element lines
            for (i = 0; i < num_elem; i++)
            {
                if (fgets(buf, 300, fp) != NULL)
                {
                    // and create the vertex lists
                    if (sscanf(buf, "%d%d%d%d%d%d%d%d%d%d\n", &tmpi, &tmpi, vl, vl + 1, vl + 2, vl + 3, vl + 4, vl + 5, vl + 6, vl + 7) != 10)
                    {
                        cerr << "ReadElmer::compute; sscanf3 failed" << endl;
                    }
                    vl += 8;

                    // the element list
                    *el++ = i * 8;

                    // the type lists
                    *tl++ = TYPE_HEXAGON;
                }
                else
                {
                    Covise::sendError("ERROR: unexpected end of file");
                    return STOP_PIPELINE;
                }
            }
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'meshObj' failed");
            return STOP_PIPELINE;
        }
    }
    else
    {
        Covise::sendError("ERROR: object name not correct for 'mesh'");
        return STOP_PIPELINE;
    }

    // create the vector data object for the velocity
    if (velocName != NULL)
    {
        velocObj = new coDoVec3(velocName, num_coord);
        velOutPort->setCurrentObject(velocObj);

        if (velocObj->objectOk())
        {
            velocObj->getAddresses(&u, &v, &w);
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'velocObj' failed");
            return STOP_PIPELINE;
        }
    }
    else
    {
        Covise::sendError("ERROR: object name not correct for 'velocity'");
        return STOP_PIPELINE;
    }

    // create the scalar data object for pressure
    if (pressName != NULL)
    {
        pressObj = new coDoFloat(pressName, num_coord);
        pressOutPort->setCurrentObject(pressObj);

        if (pressObj->objectOk())
        {
            pressObj->getAddress(&p);
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'pressObj' failed");
            return STOP_PIPELINE;
        }
    }
    else
    {
        Covise::sendError("ERROR: object name not correct for 'pressure'");
        return STOP_PIPELINE;
    }

    // create the scalar data object for pressure
    if (keName != NULL)
    {
        keObj = new coDoFloat(keName, num_coord);
        keOutPort->setCurrentObject(keObj);

        if (keObj->objectOk())
        {
            keObj->getAddress(&k);
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'keObj' failed");
            return STOP_PIPELINE;
        }
    }
    else
    {
        Covise::sendError("ERROR: object name not correct for 'ke'");
        return STOP_PIPELINE;
    }

    // now read the variables
    for (i = 0; i < num_coord; i++)
    {
        if (fgets(buf, 300, fp) != NULL)
        {
            if (sscanf(buf, "%f%f%f%f%f", p, u, v, w, k) != 5)
            {
                cerr << "ReadElmer::compute: sscanf4 failed" << endl;
            }
            u++;
            v++;
            w++;
            k++;
            p++;
        }
        else
        {
            Covise::sendError("ERROR: unexpected end of file");
            return STOP_PIPELINE;
        }
    }

    // close the file
    fclose(fp);
    return CONTINUE_PIPELINE;
}

MODULE_MAIN(IO, ReadElmer)
