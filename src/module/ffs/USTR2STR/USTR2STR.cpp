/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
 **                                                                        **
 ** Description:                                                           **
 **              samples scattered data to volume data                     **
 **                                                                        **
 **                                                                        **
 **                               (C) 1998                                 **
 **                             Paul Benoelken                             **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 ** Author: Paul Benoelken                                                 **
 **                                                                        **
 ** Date:  11.08.98  V1.0                                                  **
 ****************************************************************************/

// C library stuff
#include <stdlib.h>
#include <stdio.h>
#include <iostream.h>
#include <string.h>
#include <math.h>

#define Min(a, b) (a < b) ? a : b;
#define Max(a, b) (a > b) ? a : b;

// COVISE include stuff
#include <appl/ApplInterface.h>

#define INDEX(i, j, k) i *SizeY *SizeZ + j *SizeZ + k

int SizeX, SizeY, SizeZ;
float XMin = 0.0, YMin = 0.0, ZMin = 0.0;
float XMax = 0.0, YMax = 0.0, ZMax = 0.0;
float DX, DY, DZ;

void Compute(void *userData, void *callbackData);
void mkStructured(coDoUnstructuredGrid *input_grid,
                  coDistributedObject *input_data,
                  coDoUniformGrid **output_grid,
                  coDistributedObject **output_data,
                  char *gridname,
                  char *dataname);
void ComputeMinMax(int n, float *px, float *py, float *pz);
float *SampleNN(int np, float *px, float *py, float *pz, float *buffer);

int main(int argc, char *argv[])
{
    // Initialize the software environment

    Covise::set_module_description("sampling module");

    Covise::add_port(INPUT_PORT, "InputGrid", "coDoUnstructuredGrid", "input grid");
    Covise::add_port(INPUT_PORT,
                     "InputData",
                     "coDoFloat|coDoFloat coDoVec3|coDoVec3",
                     "input data");
    Covise::add_port(OUTPUT_PORT, "OutputGrid", "coDoUniformGrid", "output grid");
    Covise::add_port(OUTPUT_PORT,
                     "OutputData",
                     "coDoFloat|coDoVec3",
                     "output data");

    Covise::add_port(PARIN, "SizeX", "Scalar", "Gridsize X");
    Covise::add_port(PARIN, "SizeY", "Scalar", "Gridsize Y");
    Covise::add_port(PARIN, "SizeZ", "Scalar", "Gridsize Z");

    Covise::set_port_default("SizeX", "64");
    Covise::set_port_default("SizeY", "64");
    Covise::set_port_default("SizeZ", "64");
    Covise::init(argc, argv);

    // Supply your work routine here, it is called whenever a
    // message from the COVISE environment arrives

    Covise::set_start_callback(Compute, NULL);

    // wait for events and never return from here

    Covise::main_loop();
    return EXIT_SUCCESS;
}

/****************************************************************************
 *						COVISE callback									    *
 ****************************************************************************/

void Compute(void *userData, void *callbackData)
{
    coDoUniformGrid *sgrid = NULL;
    coDistributedObject *sdata = NULL;

    /* getting grid-size */
    Covise::get_scalar_param("SizeX", &SizeX);
    Covise::get_scalar_param("SizeY", &SizeY);
    Covise::get_scalar_param("SizeZ", &SizeZ);
    char info[255];
    sprintf(info, "sampling to grid %d x %d x %d.", SizeX, SizeY, SizeZ);
    Covise::sendInfo(info);

    /* getting data from shared memory */

    char *objectname = Covise::get_object_name("InputGrid");
    coDistributedObject *tmp_obj = new coDistributedObject(objectname);
    coDistributedObject *d_obj = tmp_obj->createUnknown();

    if (d_obj == NULL)
    {
        char info[255];
        sprintf(info, "can't create %s !", objectname);
        Covise::sendError(info);
        return;
    }
    char *objecttype = d_obj->getType();
    if (strcmp(objecttype, "UNSGRD") == 0)
    {
        coDoUnstructuredGrid *ugrid = (coDoUnstructuredGrid *)d_obj;
        if (!ugrid->objectOk())
        {
            Covise::sendError("error creating grid !");
            return;
        }
        /*
       * getting scalar data from shm
       */
        objectname = Covise::get_object_name("InputData");
        tmp_obj = new coDistributedObject(objectname);
        coDistributedObject *udata = tmp_obj->createUnknown();
        if (udata == NULL)
        {
            char info[255];
            sprintf(info, "can't create %s !", objectname);
            Covise::sendError(info);
            return;
        }
        mkStructured(ugrid,
                     udata,
                     &sgrid,
                     &sdata,
                     Covise::get_object_name("OutputGrid"),
                     Covise::get_object_name("OutputData"));

        if (sgrid == NULL || sdata == NULL)
        {
            cerr << "no volume created" << endl;
            return;
        }
        if (!sgrid->objectOk())
        {
            cerr << "problems in creation of structured grid";
            return;
        }
        if (!sdata->objectOk())
        {
            cerr << "problems in creation of structured data";
            return;
        }
    }
    else
    {
        if (strcmp(objecttype, "SETELE") == 0)
        {
            /*
          * getting point set from shm
          */
            coDoSet *ugrid_set = (coDoSet *)d_obj;
            if (!ugrid_set->objectOk())
            {
                Covise::sendError("error in getting point data !");
                return;
            }
            int num_of_ugrids;
            coDistributedObject **ugrid_array = ugrid_set->getAllElements(&num_of_ugrids);
            /*
          * getting set data from shm
          */
            cerr << "getting data set from shm" << endl;
            objectname = Covise::get_object_name("InputData");
            tmp_obj = new coDistributedObject(objectname);
            d_obj = tmp_obj->createUnknown();
            if (d_obj == NULL)
            {
                char info[255];
                sprintf(info, "can't create %s !", objectname);
                Covise::sendError(info);
                return;
            }
            objecttype = d_obj->getType();
            if (strcmp(objecttype, "SETELE") != 0)
            {
                char info[255];
                sprintf(info, "objecttype %s doesn't match!", objecttype);
                Covise::sendError(info);
                return;
            }
            coDoSet *udata_set = (coDoSet *)d_obj;
            if (udata_set == NULL)
            {
                cerr << "udata_set is NULL!" << endl;
                return;
            }
            if (!udata_set->objectOk())
            {
                Covise::sendError("error in getting scalar data !");
                return;
            }
            int num_of_udata;
            coDistributedObject **udata_array = udata_set->getAllElements(&num_of_udata);
            /*
          * computing set of grids and data
          */
            if (num_of_ugrids != num_of_udata)
            {
                Covise::sendError("number of set items don't match !");
                return;
            }
            else
            {
                char info[255];
                sprintf(info, "got a set with %d elements", num_of_ugrids);
                Covise::sendInfo(info);
            }
            char *sgrid_name = Covise::get_object_name("OutputGrid");
            coDoSet *sgrid_set = new coDoSet(sgrid_name, SET_CREATE);
            char *sdata_name = Covise::get_object_name("OutputData");
            coDoSet *sdata_set = new coDoSet(sdata_name, SET_CREATE);
            if (!sgrid_set->objectOk())
            {
                Covise::sendError("error in creating scalar grid set !");
                return;
            }
            for (int i = 0; i < num_of_ugrids; i++)
            {
                char tmp_grid_name[255];
                char tmp_data_name[255];
                sprintf(tmp_grid_name, "%s_%d", sgrid_name, i);
                sprintf(tmp_data_name, "%s_%d", sdata_name, i);
                mkStructured((coDoUnstructuredGrid *)ugrid_array[i],
                             udata_array[i],
                             &sgrid,
                             &sdata,
                             tmp_grid_name,
                             tmp_data_name);

                if (sgrid == NULL)
                {
                    Covise::sendError("couldn't get structured grid !");
                    return;
                }
                if (sdata == NULL)
                {
                    Covise::sendError("couldn't get structured data !");
                    return;
                }

                if (!sgrid->objectOk())
                {
                    Covise::sendError("error in creating structured grid !");
                    return;
                }
                if (!sgrid->objectOk())
                {
                    Covise::sendError("error in creating structured data !");
                    return;
                }
                Covise::sendInfo("adding grid to set");
                sgrid_set->addElement(sgrid);
                sdata_set->addElement(sdata);
                delete sgrid;
                delete sdata;
                Covise::sendInfo("ok adding data set");
            }
        }
        else
        {
            char buffer[255];
            sprintf(buffer, "object type %s doesn't match !", objecttype);
            Covise::sendError(buffer);
            return;
        }
    }
}

void mkStructured(coDoUnstructuredGrid *input_grid,
                  coDistributedObject *input_data,
                  coDoUniformGrid **output_grid,
                  coDistributedObject **output_data,
                  char *gridname,
                  char *dataname)
{
    char info[255];
    float *scalar_data;
    float *vx, *vy, *vz;
    int num_scalars = 0;
    int num_vectors = 0;

    /* getting grid points */

    int num_points = 0, num_elements, num_connections;
    float *px, *py, *pz;
    int *elements, *connections;
    input_grid->getGridSize(&num_elements, &num_connections, &num_points);
    input_grid->getAddresses(&elements, &connections, &px, &py, &pz);

    ComputeMinMax(num_points, px, py, pz);

    /* getting shm-data */

    char *objecttype = input_data->getType();
    if (strcmp(objecttype, "STRSDT") == 0)
    {
        cerr << "got structured scalar data !" << endl;
        coDoFloat *structured_scalars = (coDoFloat *)input_data;
        int dim_x, dim_y, dim_z;
        structured_scalars->getGridSize(&dim_x, &dim_y, &dim_z);
        num_scalars = dim_x * dim_y * dim_z;
        sprintf(info, "got %d structured scalar data from shm", num_scalars);
        Covise::sendInfo(info);
        structured_scalars->getAddress(&scalar_data);
    }
    else if (strcmp(objecttype, "USTSDT") == 0)
    {
        cerr << "got unstructured scalar data !" << endl;
        coDoFloat *unstructured_scalars = (coDoFloat *)input_data;
        num_scalars = unstructured_scalars->getNumPoints();
        sprintf(info, "got %d unstructured scalar data from shm", num_scalars);
        Covise::sendInfo(info);
        unstructured_scalars->getAddress(&scalar_data);
    }
    else if (strcmp(objecttype, "STRVDT") == 0)
    {
        cerr << "got structured vector data !" << endl;
        coDoVec3 *structured_vectors = (coDoVec3 *)input_data;
        int dim_x, dim_y, dim_z;
        structured_vectors->getGridSize(&dim_x, &dim_y, &dim_z);
        num_vectors = dim_x * dim_y * dim_z;
        sprintf(info, "got %d structured vector data from shm", num_vectors);
        Covise::sendInfo(info);
        structured_vectors->getAddresses(&vx, &vy, &vz);
    }
    else if (strcmp(objecttype, "USTVDT") == 0)
    {
        cerr << "got unstructured vector data !" << endl;
        coDoVec3 *unstructured_vectors = (coDoVec3 *)input_data;
        num_vectors = unstructured_vectors->getNumPoints();
        sprintf(info, "got %d unstructured vector data from shm", num_vectors);
        Covise::sendInfo(info);
        unstructured_vectors->getAddresses(&vx, &vy, &vz);
    }
    else
    {
        char buffer[255];
        sprintf(buffer, "object type %s doesn't match !", objecttype);
        Covise::sendError(buffer);
        return;
    }

    if (num_scalars != num_points && num_vectors != num_points)
    {
        Covise::sendError("number of points doesn't match number of scalars !");
        return;
    }

    /* 
    * sampling  data
    */

    if (num_scalars)
    {
        float *sdata = SampleNN(num_scalars, px, py, pz, scalar_data);
        if (sdata == NULL)
        {
            Covise::sendError("sampling scalar data failed !");
            return;
        }
        coDoFloat *do_sdata = new coDoFloat(dataname,
                                            SizeX,
                                            SizeY,
                                            SizeZ,
                                            sdata);
        *output_data = (coDistributedObject *)do_sdata;
    }
    else if (num_vectors)
    {
        float *vdata[3];
        vdata[0] = SampleNN(num_vectors, px, py, pz, vx);
        vdata[1] = SampleNN(num_vectors, px, py, pz, vy);
        vdata[2] = SampleNN(num_vectors, px, py, pz, vz);
        if (vdata[0] == NULL || vdata[1] == NULL || vdata[2] == NULL)
        {
            Covise::sendError("sampling vector data failed !");
            return;
        }
        coDoVec3 *do_vdata = new coDoVec3(dataname,
                                          SizeX,
                                          SizeY,
                                          SizeZ,
                                          vdata[0],
                                          vdata[1],
                                          vdata[2]);
        *output_data = (coDistributedObject *)&do_vdata;
    }
    /*
    * create uniform grid
    */

    *output_grid = new coDoUniformGrid(gridname, SizeX, SizeY, SizeZ, XMin, YMin, ZMin, XMax, YMax, ZMax);
}

void ComputeMinMax(int n, float *px, float *py, float *pz)
{
    cerr << "compute min/max positions and scalar values" << endl;
    for (int i = 0; i < n; i++)
    {
        XMin = Min(XMin, px[i]);
        YMin = Min(YMin, py[i]);
        ZMin = Min(ZMin, pz[i]);

        XMax = Max(XMax, px[i]);
        YMax = Max(YMax, py[i]);
        ZMax = Max(ZMax, pz[i]);
        //cerr << px[i] << ' ' << py[i] << ' ' << pz[i] << endl;
    }
    DX = XMax - XMin;
    DY = YMax - YMin;
    DZ = ZMax - ZMin;
    cerr << XMin << ' ' << YMin << ' ' << ZMin << endl;
    cerr << XMax << ' ' << YMax << ' ' << ZMax << endl;
}

/**************************************************************************
 *						NEAREST NEIGHBOUR SAMPLING  					  *
 **************************************************************************/

float *SampleNN(int n, float *px, float *py, float *pz, float *scalars)
{

    cerr << "sampling " << n << " points" << endl;

    int size = SizeX * SizeY * SizeZ;
    int *numofparticles = new int[size];
    float *scalarsum = new float[size];
    float *grid_value = new float[size];

    for (int index = 0; index < size; index++)
    {
        numofparticles[index] = 0;
        scalarsum[index] = 0.0;
    }

    cerr << "compute nearest voxels " << endl;
    for (int c = 0; c < n; c++)
    {
        int i = (SizeX - 1) * (px[c] - XMin) / DX;
        int j = (SizeY - 1) * (py[c] - YMin) / DY;
        int k = (SizeZ - 1) * (pz[c] - ZMin) / DZ;
        int index = INDEX(i, j, k);
        numofparticles[index]++;
        scalarsum[index] += scalars[c];
    }

    cerr << "compute voxel values" << endl;
    for (int i = 0; i < SizeX; i++)
        for (int j = 0; j < SizeY; j++)
            for (int k = 0; k < SizeZ; k++)
            {
                int index = INDEX(i, j, k);
                if (numofparticles[index] != 0)
                    grid_value[index] = scalarsum[index] / (float)numofparticles[index];
                else
                    grid_value[index] = 0.0;
            }

    delete[] numofparticles;
    delete[] scalarsum;
    cerr << "done sampling" << endl;
    return grid_value;
}
