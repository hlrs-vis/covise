/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2002 OSR  **
 **                                                                        **
 ** Description: Read module Rect Norsk Hydro FLACS data format            **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 *\**************************************************************************/

#include "ReadRect.h"
#include <api/coStepFile.h>

void main(int argc, char *argv[])
{

    Application *application = new Application();

    application->start(argc, argv);
}

Application::Application()
    : coModule("Read Flacs Data") // description in the module setup window
{
    // file browser parameter
    gridfileParam = addFileBrowserParam("grid", "Grid file path");
    //gridfileParam->setValue("data","data *.dat*");
    gridfileParam->setValue("/h/osr/covise/src/application/OwnReaders/", "grid gr*.asc*");

    sdatafileParam = addFileBrowserParam("sdata", "Scalar data file path");
    //sdatafileParam->setValue("data","data *.dat*");
    sdatafileParam->setValue("/h/osr/covise/src/application/OwnReaders/", "sdata sc*.*");

    vdatafileParam = addFileBrowserParam("vdata", "Vector data file path");
    //vdatafileParam->setValue("data","data *.dat*");
    vdatafileParam->setValue("/h/osr/covise/src/application/OwnReaders/", "vdata ve*.*");

    p_timesteps = addInt32Param("timesteps", "timesteps");
    p_timesteps->setValue(1);
    p_skip = addInt32Param("skipped_files", "number of skipped files for each timestep");
    p_skip->setValue(0);

    // the output ports
    p_outPort1 = addOutputPort("mesh", "coDoStructuredGrid | Set_StructuredGrid", "structured grid");
    //p_outPort1 = addOutputPort("mesh","coDoRectilinearGrid | Set_RectilinearGrid","rectilinear grid");
    p_outPort2 = addOutputPort("scalar data", "coDoFloat | Set_Float", "scalar data");
    p_outPort3 = addOutputPort("vector data", "coDoVec3 | Set_Vec3", "vector data");
}

Application::~Application()
{
}

int Application::compute()
{

    FILE *fp;
    const char *gridfileName;
    const char *sdatafileName;
    const char *vdatafileName;
    int i, numSteps;
    char buf[300];

    int x_dim, y_dim, z_dim; // dimensions

    // read the file browser parameter
    gridfileName = gridfileParam->getValue();
    sdatafileName = sdatafileParam->getValue();
    vdatafileName = vdatafileParam->getValue();
    int timesteps = p_timesteps->getValue();

    if (timesteps < 1)
    {
        timesteps = 1;
        p_timesteps->setValue(timesteps);
    }

    int skip_value = p_skip->getValue();

    char *next_path = NULL;

    // the COVISE output objects (located in shared memory)
    coDoRectilinearGrid *gridObj;
    coDoFloat *scalarObj;
    coDoVec3 *vectorObj;

    coDistributedObject **time_outputgrid;
    coDistributedObject **time_outputscalar;
    coDistributedObject **time_outputvector;

    // get the ouput object names from the controller
    // the output object names have to be assigned by the controller

    time_outputgrid = new coDistributedObject *[timesteps + 1];
    time_outputgrid[timesteps] = NULL;

    time_outputscalar = new coDistributedObject *[timesteps + 1];
    time_outputscalar[timesteps] = NULL;

    time_outputvector = new coDistributedObject *[timesteps + 1];
    time_outputvector[timesteps] = NULL;

    coStepFile *step_grid = new coStepFile(gridfileName);
    step_grid->set_skip_value(skip_value);
    step_grid->set_delta(3000);

    coStepFile *step_scalar = new coStepFile(sdatafileName);
    step_scalar->set_skip_value(skip_value);
    step_scalar->set_delta(3000);

    coStepFile *step_vector = new coStepFile(vdatafileName);
    step_vector->set_skip_value(skip_value);
    step_vector->set_delta(3000);

    numSteps = 0;

    int error = 0;

    for (i = 0; i < timesteps; i++)
    {
        //======================GRID====================================
        step_grid->get_nextpath(&next_path);

        if (next_path)
        {
            if ((fp = fopen(next_path, "r")) == NULL)
            {
                sprintf(buf, "ERROR: Can't open file >> %s", next_path);
                sendError(buf);
                return STOP_PIPELINE;
            }
            fgets(buf, 300, fp);
            sscanf(buf, "%d %d %d", &x_dim, &y_dim, &z_dim);
            //cerr <<"x_dim= "<<x_dim<<" y_dim= "<<y_dim<<" z_dim= "<<z_dim<<endl;

            sprintf(buf, "Reading the grid file %s (step %d) ...", next_path, i + 1);
            sendInfo(buf);

            if (timesteps > 1)
                sprintf(buf, "%s_%d", p_outPort1->getObjName(), i);
            else
            {
                time_outputgrid[i] = NULL;
                strcpy(buf, p_outPort1->getObjName());
            }

            //read data from next_path file
            if (read_grid(fp, buf, x_dim, y_dim, z_dim, &gridObj) == STOP_PIPELINE)
                return STOP_PIPELINE;
            // close the file
            fclose(fp);
            sendInfo("The file was successfuly read!");

            time_outputgrid[i] = gridObj;
            delete[] next_path;
        }
        else
        {
            error = 1;
            time_outputgrid[i] = NULL;
            cerr << "the indicated number of timesteps is bigger than the number of available grid files" << endl;
        }

        //=====================SCALAR DATA=================================
        step_scalar->get_nextpath(&next_path);

        if (next_path && !error)
        {
            if ((fp = fopen(next_path, "r")) == NULL)
            {
                sprintf(buf, "ERROR: Can't open file >> %s", next_path);
                sendError(buf);
                return STOP_PIPELINE;
            }

            sprintf(buf, "Reading the scalar data file %s (step %d) ...", next_path, i + 1);
            sendInfo(buf);

            if (timesteps > 1)
                sprintf(buf, "%s_%d", p_outPort2->getObjName(), i);
            else
            {
                time_outputscalar[i] = NULL;
                strcpy(buf, p_outPort2->getObjName());
            }

            //read data from next_path file
            if (read_scalar(fp, buf, x_dim, y_dim, z_dim, &scalarObj) == STOP_PIPELINE)
                return STOP_PIPELINE;
            fclose(fp);

            sendInfo("The file was successfuly read!");

            time_outputscalar[i] = scalarObj;
            delete[] next_path;
        }
        else
        {
            error = 1;
            time_outputscalar[i] = NULL;
            cerr << "the indicated number of timesteps is bigger than the number of available scalar files" << endl;
        }

        //=====================VECTOR DATA=================================
        step_vector->get_nextpath(&next_path);

        if (next_path && !error)
        {
            if ((fp = fopen(next_path, "r")) == NULL)
            {
                sprintf(buf, "ERROR: Can't open file >> %s", next_path);
                sendError(buf);
                return STOP_PIPELINE;
            }

            sprintf(buf, "Reading the vector data file %s (step %d) ...", next_path, i + 1);
            sendInfo(buf);

            if (timesteps > 1)
                sprintf(buf, "%s_%d", p_outPort3->getObjName(), i);
            else
                strcpy(buf, p_outPort3->getObjName());

            //read data from next_path file
            if (read_vector(fp, buf, x_dim, y_dim, z_dim, &vectorObj) == STOP_PIPELINE)
                return STOP_PIPELINE;
            fclose(fp);

            sendInfo("The file was successfuly read!");

            time_outputvector[i] = vectorObj;
            delete[] next_path;
        }
        else
        {
            error = 1;
            cerr << "the indicated number of timesteps is bigger than the number of available vector files" << endl;
            time_outputvector[i] = NULL;
        }

        if (!error)
        {
            numSteps++;
        }
        else
        {
            if (time_outputgrid[i] != NULL)
                delete time_outputgrid[i];
            if (time_outputscalar[i] != NULL)
                delete time_outputscalar[i];
            if (time_outputvector[i] != NULL)
                delete time_outputvector[i];
        }
    }

    delete step_grid;
    delete step_scalar;
    delete step_vector;

    if (timesteps > 1)
    {
        time_outputgrid[numSteps] = NULL;
        coDoSet *time_grd = new coDoSet(p_outPort1->getObjName(), time_outputgrid);
        sprintf(buf, "1 %d", numSteps);
        time_grd->addAttribute("TIMESTEP", buf);

        for (i = 0; i < numSteps; i++)
            delete time_outputgrid[i];
        delete[] time_outputgrid;

        p_outPort1->setCurrentObject(time_grd);

        time_outputscalar[numSteps] = NULL;
        coDoSet *time_scalar = new coDoSet(p_outPort2->getObjName(), time_outputscalar);
        time_scalar->addAttribute("TIMESTEP", buf);

        for (i = 0; i < numSteps; i++)
            delete time_outputscalar[i];
        delete[] time_outputscalar;

        p_outPort2->setCurrentObject(time_scalar);

        time_outputvector[numSteps] = NULL;
        coDoSet *time_vector = new coDoSet(p_outPort3->getObjName(), time_outputvector);
        time_vector->addAttribute("TIMESTEP", buf);

        for (i = 0; i < numSteps; i++)
            delete time_outputvector[i];
        delete[] time_outputvector;

        p_outPort3->setCurrentObject(time_vector);
    }
    else
    {
        p_outPort1->setCurrentObject(gridObj);
        p_outPort2->setCurrentObject(scalarObj);
        p_outPort3->setCurrentObject(vectorObj);
    }

    return CONTINUE_PIPELINE;
}

int Application::read_grid(FILE *fp, const char *gridName, int x_dim, int y_dim, int z_dim, coDoRectilinearGrid **gridObj)
{
    float *x_coord, *y_coord, *z_coord; // coordinate lists
    int i;
    char buf[300];

    // create the structured grid object for the mesh
    if (gridName != NULL)
    {
        *gridObj = new coDoRectilinearGrid(gridName, x_dim, y_dim, z_dim);
        if ((*gridObj)->objectOk())
        {
            // get pointers to the element, vertex and coordinate lists
            (*gridObj)->getAddresses(&x_coord, &y_coord, &z_coord);
            //cerr<<"got the addresses...";

            // read the xu-coordinate lines
            for (i = 0; i < x_dim; i++)
            {
                // read the line which contains the coordinates and scan it
                if (fgets(buf, 300, fp) != NULL)
                {
                    sscanf(buf, "%f", &x_coord[i]);
                }
                else
                {
                    sendError("ERROR: unexpected end of file");
                    return STOP_PIPELINE;
                }
                //cerr<<"x_coord...: "<<x_coord[i]<<endl;
            }

            // read the yv-coordinate lines
            for (i = 0; i < y_dim; i++)
            {
                // read the line which contains the coordinates and scan it
                if (fgets(buf, 300, fp) != NULL)
                {
                    sscanf(buf, "%f\n", &y_coord[i]);
                }
                else
                {
                    sendError("ERROR: unexpected end of file");
                    return STOP_PIPELINE;
                }
                //cerr<<"y_coord...: "<<y_coord[i]<<endl;
            }

            //fgets(buf, 300, fp);

            // read the zw-coordinate lines
            for (i = 0; i < z_dim; i++)
            {
                // read the line which contains the coordinates and scan it
                if (fgets(buf, 300, fp) != NULL)
                {
                    sscanf(buf, "%f\n", &z_coord[i]);
                }
                else
                {
                    sendError("ERROR: unexpected end of file");
                    return STOP_PIPELINE;
                }
                //cerr<<"z_coord...: "<<z_coord[i]<<endl;
            }

            //fgets(buf, 300, fp);
        }
        else
        {
            sendError("Failed to create the object '%s' for the port '%s'", gridName, p_outPort1->getName());
            return STOP_PIPELINE;
        }
    }
    else
    {
        return STOP_PIPELINE;
    }

    // get the grid size
    int numX, numY, numZ;
    (*gridObj)->getGridSize(&numX, &numY, &numZ);
    //cerr<<"got the grid size...";
    //cerr<<"numX...: "<<numX<<endl;
    //cerr<<"numY...: "<<numY<<endl;
    //cerr<<"numZ...: "<<numZ<<endl;

    // get the point coordinates
    int ix, jy, kz;
    float x_size, y_size, z_size; // coordinate lists
    for (kz = 0; kz < z_dim; kz++)
    {
        for (jy = 0; jy < y_dim; jy++)
        {
            for (ix = 0; ix < x_dim; ix++)
            {
                (*gridObj)->getPointCoordinates(ix, &x_size, jy, &y_size, kz, &z_size);
                // cerr<<"got the point coordinate...";
                // cerr<<"cv_position["<<ix<<","<<jy<<","<<kz<<"]....: "<<x_size<<" "<<y_size<<" "<<z_size<<endl;
            }
        }
    }

    return CONTINUE_PIPELINE;
}

int Application::read_scalar(FILE *fp, const char *scalarName, int x_dim, int y_dim, int z_dim, coDoFloat **scalarObj)
{
    int i;
    char buf[300];
    float *sdata;

    if (scalarName != NULL)
    {
        *scalarObj = new coDoFloat(scalarName, x_dim, y_dim, z_dim);
        if ((*scalarObj)->objectOk())
        {
            // get pointers to the element, vertex and coordinate lists
            (*scalarObj)->getAddress(&sdata);
            //cerr<<"got the addresses...";
            // read the coordinate lines
            fgets(buf, 300, fp);
            fgets(buf, 300, fp);

            for (i = 0; i < x_dim * y_dim * z_dim; i++)
            {
                // read the line which contains the coordinates and scan it
                if (fgets(buf, 300, fp) != NULL)
                {
                    sscanf(buf, "%f", &sdata[i]);
                }
                else
                {
                    sendError("ERROR: unexpected end of file");
                    return STOP_PIPELINE;
                }
                //cerr<<"s_data["<<i<<"] ..: "<<sdata[i]<<endl;
            }

            //fgets(buf, 300, fp);
        }
        else
        {
            sendError("Failed to create the object '%s' for the port '%s'", scalarName, p_outPort2->getName());
            return STOP_PIPELINE;
        }
    }
    else
    {
        return STOP_PIPELINE;
    }

    return CONTINUE_PIPELINE;
}

int Application::read_vector(FILE *fp, const char *vectorName, int x_dim, int y_dim, int z_dim, coDoVec3 **vectorObj)
{
    int i;
    char buf[300];
    float *vu_data, *vv_data, *vw_data;

    if (vectorName != NULL)
    {
        (*vectorObj) = new coDoVec3(vectorName, x_dim, y_dim, z_dim);
        if ((*vectorObj)->objectOk())
        {
            // get pointers to the element, vertex and coordinate lists
            (*vectorObj)->getAddresses(&vu_data, &vv_data, &vw_data);
            //cerr<<"got the addresses...";
            // read the vector data lines
            fgets(buf, 300, fp);
            fgets(buf, 300, fp);

            for (i = 0; i < x_dim * y_dim * z_dim; i++)
            {
                // read the line which contains the coordinates and scan it
                if (fgets(buf, 300, fp) != NULL)
                {
                    sscanf(buf, "%f%f%f", &vu_data[i], &vv_data[i], &vw_data[i]);
                }
                else
                {
                    sendError("ERROR: unexpected end of file");
                    return STOP_PIPELINE;
                }
                //cerr<<"v_data["<<i<<"] ..: "<<vu_data[i]<<","<<vv_data[i]<<","<<vw_data[i]<<endl;
            }

            //fgets(buf, 300, fp);
        }
        else
        {
            sendError("Failed to create the object '%s' for the port '%s'", vectorName, p_outPort3->getName());
            return STOP_PIPELINE;
        }
    }
    else
    {
        return STOP_PIPELINE;
    }

    return CONTINUE_PIPELINE;
}
