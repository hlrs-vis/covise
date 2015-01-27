/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//#define test
/* ****************************************
 ** Read module for MEQSICO data format **
 ** Author: Tobias Bachran              **
 **************************************** */

#include <ReadMeqsico.h>

coMEQSICO::coMEQSICO(int argc, char *argv[])
    : coModule(argc, argv, "Read MEQSICO Data")
{
    // file browser parameter
    filenameParam = addFileBrowserParam("file_path", "Data file Path");
    filenameParam->setValue("file_path", "home/tobi/covise/ *.comeq*");

    // the output ports
    meshOutPort = addOutputPort("mesh", "UnstructuredGrid", "unstructured grid");
    mesh_dualOutPort = addOutputPort("mesh_dual", "UnstructuredGrid", "unstructured grid dual");
    Voltage_scalarOutPort = addOutputPort("Voltage", "Float", "Voltage data");
    EField_vectorOutPort = addOutputPort("E-Field", "Vec3", "E-Field data");
}

coMEQSICO::~coMEQSICO()
{
}

int coMEQSICO::compute(const char *port)
{
#ifdef test
    FILE *fptest;
    fptest = fopen("test-error.txt", "w");
    if (fptest == NULL)
    {
        sendError("nicht moeglich test-error.txt zu oeffnen");
    }
    fprintf(fptest, "TEST:compute laeuft an\n");
#endif
    (void)port;

    FILE *fp;
    fpos_t fp_dual;
    const char *fileName;
    int num_coord, num_elem, num_conn;
    int num_coord_dual;
    int i;
    char buf[300];

    int *el, *vl, *tl;
    int *el_dual, *vl_dual, *tl_dual;
    float *x_coord, *y_coord, *z_coord;
    float *x_coord_dual, *y_coord_dual, *z_coord_dual;
    float *u, *v, *w, *volt;

    // names of the COVISE output objects
    const char *meshName;
    const char *mesh_dualName;
    const char *Voltage_scalarName;
    const char *EField_vectorName;

    // the COVISE output objects (located in shared memory)
    coDoUnstructuredGrid *meshObj;
    coDoUnstructuredGrid *mesh_dualObj;
    coDoFloat *Voltage_scalarObj;
    coDoVec3 *EField_vectorObj;

    // read the file browser parameter
    fileName = filenameParam->getValue();

    // open the file
    if ((fp = fopen(fileName, "r")) == NULL)
    {
        strcpy(buf, "ERROR: Can't open file >>");
        strcpy(buf, fileName);
        sendError(buf);
        return STOP_PIPELINE;
    }

#ifdef test
    else
        fprintf(fptest, "TEST:Datei geoeffnet\n");
#endif
    // get the output object names from the controller
    // the output object names have to be assigned by tho controller
    meshName = meshOutPort->getObjName();
    mesh_dualName = mesh_dualOutPort->getObjName();
    Voltage_scalarName = Voltage_scalarOutPort->getObjName();
    EField_vectorName = EField_vectorOutPort->getObjName();

    //read the dimensions from the header line
    if (fgets(buf, 300, fp) == NULL)
    {
        cerr << "coMEQSICO::compute: fgets failed" << endl;
    }
    if (sscanf(buf, "%d%d%d", &num_coord, &num_coord_dual, &num_elem) != 3)
    {
        cerr << "coMEQSICO::compute: sscanf (num_coord/num_coord_dual/num_elem) failed" << endl;
    }
    num_conn = num_elem * 4;

#ifdef test
    fprintf(fptest, "TEST:Dimensionen eingelesen: num_coord=%d, num_coord_dual=%d, num_elem=%d\n", num_coord, num_coord_dual, num_elem);
#endif
    // create the unstructured grid object for the mesh
    if (meshName != NULL)
    {
        /*
		 el = new int[num_elem];  
         vl = new int[4*num_elem];  
         tl = new int[num_elem];  
         x_coord = new float[num_coord];  
         y_coord = new float[num_coord];  
         z_coord = new float[num_coord]; 
		 */
        meshObj = new coDoUnstructuredGrid(meshName, num_elem, num_conn, num_coord, 1);
        meshOutPort->setCurrentObject(meshObj);

        if (meshObj->objectOk())
        {
            // get pointers to the element, vertex and coordinate lists
            meshObj->getAddresses(&el, &vl, &x_coord, &y_coord, &z_coord);

            // get a pointer to the type list
            meshObj->getTypeList(&tl);

// read the coordinate lines
#ifdef test
            fprintf(fptest, "TEST:meshObj erstellt\n");
#endif

            for (i = 0; i < num_coord; i++)
            {
                // read the lines which contains the coordinates and scan it
                if (fgets(buf, 300, fp) != NULL)
                {
                    if (sscanf(buf, "%f%f%f\n", x_coord, y_coord, z_coord) != 3)
                    {
                        cerr << "coMEQSICO::compute: sscanf (x_coord/y_coord/z_coord) failed" << endl;
                    }
                    x_coord++;
                    y_coord++;
                    z_coord++;
                }
                else
                {
                    sendError("ERROR: unexpected end of file(mesh:coord)");
                    return STOP_PIPELINE;
                }
            }
#ifdef test
            fprintf(fptest, "TEST:mesh: Koordinaten eingelesen\n");
            fprintf(fptest, "TEST:mesh: index:%d, num_coord:%d, buf:%s\n", i, num_coord, (buf + 0));
#endif

            // read the element lines
            for (i = 0; i < num_elem; i++)
            {
                if (fgets(buf, 300, fp) != NULL)
                {
                    // and creates the vertey lists
                    if (sscanf(buf, "%d%d%d%d\n", vl, vl + 1, vl + 2, vl + 3) != 4)
                    {
                        cerr << "coMEQSICO::compute: sscanf (vl/.../vl+3) failed" << endl;
                    }
                    vl += 4;
                    // the element list
                    *el = i * 4;
                    el++;
                    // the type lists
                    *tl = 4;
                    tl++;
                }
                else
                {
                    Covise::sendError("ERROR: unexpected end of file(mesh:elements)");
                    return STOP_PIPELINE;
                }
            }

#ifdef test
            fprintf(fptest, "TEST:mesh: Elemente eingelesen\n");
            fprintf(fptest, "TEST:mesh: index:%d, num_elem:%d, buf:%s\n", i, num_elem, (buf + 0));
            fprintf(fptest, "TEST:mesh: buf hier als fp_dual=fp\n");
#endif
            // save filepoint for dualgrid
            //fp_dual=fp;
            fgetpos(fp, &fp_dual);
            for (i = 0; i < num_coord_dual; i++)
            {
                if (fgets(buf, 300, fp) == 0)
                {
                    sendError("ERROR: unexpected end of file(savepoint)");
                    return STOP_PIPELINE;
                }
            }

#ifdef test
            fprintf(fptest, "TEST:mesh: Filepointer fuer duales mesh gespeichert\n");
            fprintf(fptest, "TEST:mesh: index:%d, num_coord_dual:%d, buf:%s\n", i, num_coord_dual, (buf + 0));
            fprintf(fptest, "TEST:mesh: buf hier als fp fuer volt\n");
#endif
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

#ifdef test
    fprintf(fptest, "TEST:mesh fertig\n");
#endif

    // create the scalar data object for Voltage
    if (Voltage_scalarName != NULL)
    {
        Voltage_scalarObj = new coDoFloat(Voltage_scalarName, num_coord);
        Voltage_scalarOutPort->setCurrentObject(Voltage_scalarObj);

        if (Voltage_scalarObj->objectOk())
        {
#ifdef test
            fprintf(fptest, "TEST:Voltage_scalarObj erstellt\n");
#endif
            Voltage_scalarObj->getAddress(&volt);

            // read the Voltage_scalar lines
            for (i = 0; i < num_coord; i++)
            {
                // read the lines which contains the Voltage_scalar and scan it
                if (fgets(buf, 300, fp) != NULL)
                {
                    if (sscanf(buf, "%f\n", volt) != 1)
                    {
                        cerr << "coMEQSICO::compute: sscanf (volt) failed" << endl;
                    }
                    volt++;
                }
                else
                {
                    sendError("ERROR: unexpected end of file(volt)");
                    return STOP_PIPELINE;
                }
            }
#ifdef test
            fprintf(fptest, "TEST:Voltage_scalar: Werte eingelesen\n");
            fprintf(fptest, "TEST:Voltage_scalar: index:%d, num_coord:%d, buf:%s\n", i, num_coord, (buf + 0));
#endif
        }
        else
        {
            Covise::sendError("ERROR: object name not correct for 'Voltage_scalar'");
            return STOP_PIPELINE;
        }
#ifdef test
        fprintf(fptest, "TEST:Voltage_scalarObj fertig\n");
#endif
    }
    else
    {
        Covise::sendError("ERROR: object name not correct for 'Voltage_scalar'");
        return STOP_PIPELINE;
    }

    // create the vector data object for EField
    if (EField_vectorName != NULL)
    {
        EField_vectorObj = new coDoVec3(EField_vectorName, num_coord_dual);
        EField_vectorOutPort->setCurrentObject(EField_vectorObj);

        if (EField_vectorObj->objectOk())
        {
#ifdef test
            fprintf(fptest, "TEST:EField_vectorObj erstellt\n");
#endif
            EField_vectorObj->getAddresses(&u, &v, &w);

            // read the EField_vector lines
            for (i = 0; i < num_coord_dual; i++)
            {
                // read the lines which contains the EField_vector and scan it
                if (fgets(buf, 300, fp) != NULL)
                {
                    if (sscanf(buf, "%f%f%f\n", u, v, w) != 3)
                    {
                        cerr << "coMEQSICO::compute: sscanf (u/v/w) failed" << endl;
                    }
                    u++;
                    v++;
                    w++;
                }
                else
                {
                    sendError("ERROR: unexpected end of file(efield)");
                    return STOP_PIPELINE;
                }
            }
#ifdef test
            fprintf(fptest, "TEST:EField_vector: Daten eingelesen");
            fprintf(fptest, "TEST:EField_vector: index:%d, num_coord_dual:%d, buf:%s\n", i, num_coord_dual, (buf + 0));
#endif
        }
        else
        {
            Covise::sendError("ERROR: object name not correct for 'EField_vector'");
            return STOP_PIPELINE;
        }
#ifdef test
        fprintf(fptest, "TEST:EField_vector fertig\n");
#endif
    }
    else
    {
        Covise::sendError("ERROR: object name not correct for 'EField_vector'");
        return STOP_PIPELINE;
    }

    // create the unstructured grid object for the mesh_dual
    if (mesh_dualName != NULL)
    {
        fsetpos(fp, &fp_dual);
        mesh_dualObj = new coDoUnstructuredGrid(mesh_dualName, 0, 0, num_coord_dual, 1);
        mesh_dualOutPort->setCurrentObject(mesh_dualObj);

        if (mesh_dualObj->objectOk())
        {
#ifdef test
            fprintf(fptest, "TEST:mesh_dualObj erstellt\n");
#endif
            // get pointers to the element, vertex and coordinate lists
            mesh_dualObj->getAddresses(&el_dual, &vl_dual, &x_coord_dual, &y_coord_dual, &z_coord_dual);
            // get a pointer to the type list
            mesh_dualObj->getTypeList(&tl_dual);
            // read the coordinate lines
            for (i = 0; i < num_coord_dual; i++)
            {
                // read the lines which contains the coordinates and scan it
                if (fgets(buf, 300, fp) != NULL)
                {
                    if (sscanf(buf, "%f%f%f\n", x_coord_dual, y_coord_dual, z_coord_dual) != 3)
                    {
                        cerr << "coMEQSICO::compute: sscanf (x_coord_dual/.../z_coord_dual) failed" << endl;
                    }
                    x_coord_dual++;
                    y_coord_dual++;
                    z_coord_dual++;
                }
                else
                {
                    sendError("ERROR: unexpected end of file(mesh_dual)");
#ifdef test
                    fprintf(fptest, "ERROR: unexpected end of file(mesh_dual)\n");
                    fprintf(fptest, "TEST:Datei geschlossen\n");
                    fprintf(fptest, "TEST:index:%d, num_coord_dual:%d, x:%f, y:%f ,z:%f", i, num_coord_dual, --(*x_coord_dual), --(*y_coord_dual), --(*z_coord_dual));
                    fprintf(fptest, "TEST:%s", (buf + 0));
                    fclose(fptest);
#endif
                    return STOP_PIPELINE;
                }
            }
#ifdef test
            fprintf(fptest, "TEST:mesh_dual: Koordinaten eingelesen\n");
#endif
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'mesh_dualObj' failed");
            return STOP_PIPELINE;
        }
#ifdef test
        fprintf(fptest, "TEST:mesh_dual fertig\n");
#endif
    }
    else
    {
        Covise::sendError("ERROR: object name not correct for 'mesh_dual'");
        return STOP_PIPELINE;
    }

    //close the file
    fclose(fp);
#ifdef test
    fprintf(fptest, "TEST:Datei geschlossen\n");
    fclose(fptest);
#endif
    return CONTINUE_PIPELINE;
}
MODULE_MAIN(IO, coMEQSICO)
