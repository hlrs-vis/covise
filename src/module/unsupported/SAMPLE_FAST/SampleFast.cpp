/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
 **                                                                        **
 ** Description:                                                           **
 **              samples scattered data to volume data/Structured Data     **
 **                                                                        **
 **                                                                        **
 **                               (C) 1998                                 **
 **                       Uwe Woessner,Paul Benoelken                      **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 ** Author: Paul Benoelken, Uwe Woessner                                   **
 **                                                                        **
 ** Date:  11.08.98  V1.0                                                  **
 ** Date:  23.10.98  V2.0                                                  **
 ****************************************************************************/

#include "SampleFast.h"

#define Min(a, b) (a < b) ? a : b;
#define Max(a, b) (a > b) ? a : b;

#define INDEX(i, j, k) i *size_j *size_k + j *size_k + k

int main(int argc, char *argv[])
{
    // init
    Sample *application = new Sample(argc, argv);

    Covise::main_loop();
}

/****************************************************************************
 *						COVISE callback									    *
 ****************************************************************************/

coDistributedObject **Sample::compute(coDistributedObject **in, char **outNames)
{
    coDistributedObject **returnObject = NULL;
    DO_Volume_Data *volume = NULL;
    coDoUniformGrid *gridOut = NULL;
    coDoVec3 *vDataOut = NULL;
    coDoFloat *sDataOut = NULL;
    createVolume = 0;
    returnObject = new coDistributedObject *[4];
    returnObject[0] = NULL;
    returnObject[1] = NULL;
    returnObject[2] = NULL;
    returnObject[3] = NULL;
    // getting grid-size
    int size_i, size_j, size_k;
    int num_points;
    float *px, *py, *pz;
    int mode;
    sentWarning = 0;
    Covise::get_scalar_param("SizeX", &size_i);
    Covise::get_scalar_param("SizeY", &size_j);
    Covise::get_scalar_param("SizeZ", &size_k);
    Covise::get_scalar_param("fillDistance", &fillDistance);
    Covise::get_choice_param("Mode", &mode);
    Covise::get_boolean_param("createVolume", &createVolume);
    if (createVolume)
        Density = new char[size_i * size_j * size_k];
    else
        Density = NULL;
    char info[600];
    sprintf(info, "sampling to grid %d x %d x %d.", size_i, size_j, size_k);
    Covise::sendInfo(info);

    // getting data from shared memory

    coDistributedObject *data_obj = in[0];

    if (data_obj == NULL)
    {
        sprintf(info, "can't get Points !");
        Covise::sendError(info);
        return returnObject;
    }
    char *objecttype = data_obj->getType();
    if (strcmp(objecttype, "POINTS") == 0)
    {
        coDoPoints *points = (coDoPoints *)data_obj;
        if (!points->objectOk())
        {
            Covise::sendError("error retrieving points !");
            return returnObject;
        }
        num_points = points->getNumPoints();
        points->getAddresses(&px, &py, &pz);
    }
    else if (strcmp(objecttype, "UNSGRD") == 0)
    {
        int tmpi, *tmpip;
        coDoUnstructuredGrid *grid_in = (coDoUnstructuredGrid *)data_obj;
        if (!grid_in->objectOk())
        {
            Covise::sendError("error retrieving points !");
            return returnObject;
        }
        grid_in->getGridSize(&tmpi, &tmpi, &num_points);
        grid_in->getAddresses(&tmpip, &tmpip, &px, &py, &pz);
    }
    else if (strcmp(objecttype, "STRGRD") == 0)
    {
        int xs, ys, zs;
        coDoStructuredGrid *grid_in = (coDoStructuredGrid *)data_obj;
        if (!grid_in->objectOk())
        {
            Covise::sendError("error retrieving points !");
            return returnObject;
        }
        grid_in->getGridSize(&xs, &ys, &zs);
        num_points = xs * ys * zs;
        grid_in->getAddresses(&px, &py, &pz);
    }
    else
    {
        char buffer[255];
        sprintf(buffer, "object type %s not supported !", objecttype);
        Covise::sendError(buffer);
        return returnObject;
    }

    data_obj = in[1];

    if (data_obj == NULL)
    {
        sprintf(info, "can't get Data!");
        Covise::sendError(info);
        return returnObject;
    }

    objecttype = data_obj->getType();
    float *scalar_data;
    float *vector_length = NULL;
    num_scalars = 0;
    num_vectors = 0;
    coDoFloat *structured_scalars = NULL;
    coDoFloat *unstructured_scalars = NULL;
    coDoVec3 *structured_vectors = NULL;
    coDoVec3 *unstructured_vectors = NULL;

    if (strcmp(objecttype, "STRSDT") == 0)
    {
        cerr << "got structured scalar data !" << endl;
        structured_scalars = (coDoFloat *)data_obj;
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
        unstructured_scalars = (coDoFloat *)data_obj;
        num_scalars = unstructured_scalars->getNumPoints();
        sprintf(info, "got %d unstructured scalar data from shm", num_scalars);
        Covise::sendInfo(info);
        unstructured_scalars->getAddress(&scalar_data);
    }
    else if (strcmp(objecttype, "STRVDT") == 0)
    {
        cerr << "got structured vector data !" << endl;
        structured_vectors = (coDoVec3 *)data_obj;
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
        unstructured_vectors = (coDoVec3 *)data_obj;
        num_vectors = unstructured_vectors->getNumPoints();
        sprintf(info, "got %d unstructured vector data from shm", num_vectors);
        Covise::sendInfo(info);
        unstructured_vectors->getAddresses(&vx, &vy, &vz);
    }
    else
    {
        char buffer[255];
        sprintf(buffer, "object type %s not supported !", objecttype);
        Covise::sendError(buffer);
        return returnObject;
    }

    if (num_scalars != num_points && num_vectors != num_points)
    {
        Covise::sendError("number of points doesn't match number of scalars/vectors !");
        return returnObject;
    }

    p_min[0] = 0.0; // get x,y,z min and max
    p_max[0] = 0.0;
    p_min[1] = 0.0;
    p_max[1] = 0.0;
    p_min[2] = 0.0;
    p_max[2] = 0.0;
    for (int c = 0; c < num_points; c++)
    {
        p_min[0] = Min(p_min[0], px[c]);
        p_min[1] = Min(p_min[1], py[c]);
        p_min[2] = Min(p_min[2], pz[c]);

        p_max[0] = Max(p_max[0], px[c]);
        p_max[1] = Max(p_max[1], py[c]);
        p_max[2] = Max(p_max[2], pz[c]);
    }
    gridOut = new coDoUniformGrid(outNames[0],
                                  size_i,
                                  size_j,
                                  size_k, p_min[0], p_max[0], p_min[1], p_max[1], p_min[2], p_max[2]);
    if (gridOut == NULL)
    {
        Covise::sendError("no grid created!");
        return returnObject;
    }
    if (!gridOut->objectOk())
    {
        Covise::sendError("creation of grid in shm  failed!");
        return returnObject;
    }
    gridOut->addAttribute("DataObject", outNames[1]);

    char *DataIn = Covise::get_object_name("SData");
    //fprintf(stderr, "SAMPLE: obj name is %p\n", DataIn);
    gridOut->addAttribute("DataObjectName", DataIn);
    //fprintf(stderr, "SAMPLE: set don attrib\n");

    if (createVolume)
    {
        if (structured_vectors != NULL || unstructured_vectors != NULL)
        {
            /*
          * compute length of vectors
          */
            vector_length = new float[num_vectors];
            for (int i = 0; i < num_vectors; i++)
                vector_length[i] = sqrt(vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
            num_scalars = num_vectors;
            scalar_data = vector_length;
        }
    }
    if (num_vectors)
    {
        vDataOut = new coDoVec3(outNames[1],
                                size_i,
                                size_j,
                                size_k);
        if (vDataOut == NULL)
        {
            Covise::sendError("no data created!");
            return returnObject;
        }
        if (!vDataOut->objectOk())
        {
            Covise::sendError("creation of data in shm  failed!");
            return returnObject;
        }
        vDataOut->getAddresses(&vxo, &vyo, &vzo);

        /* 
       * sampling  data
       */
        if (mode == 1)
            SampleMeanv(num_vectors, num_vectors,
                        px,
                        py,
                        pz,
                        scalar_data,
                        size_i,
                        size_j,
                        size_k);
    }
    else
    {
        sDataOut = new coDoFloat(outNames[1],
                                 size_i,
                                 size_j,
                                 size_k);
        if (sDataOut == NULL)
        {
            Covise::sendError("no data created!");
            return returnObject;
        }
        if (!sDataOut->objectOk())
        {
            Covise::sendError("creation of data in shm  failed!");
            return returnObject;
        }
        sDataOut->getAddress(&so);

        /* 
       * sampling  data
       */
        if (mode == 1)
            SampleMean(num_scalars, num_vectors,
                       px,
                       py,
                       pz,
                       scalar_data,
                       size_i,
                       size_j,
                       size_k);
        else
            SampleDensity(num_scalars, num_vectors,
                          px,
                          py,
                          pz,
                          size_i,
                          size_j,
                          size_k);
    }

    if (createVolume)
    {

        if (Density == NULL)
        {
            Covise::sendError("getScalarVolume::Sample failed !");
            cerr << "Sample failed !" << endl;
            return returnObject;
        }
        volume = new DO_Volume_Data(outNames[2],
                                    size_i,
                                    size_j,
                                    size_k,
                                    Density);

        if (volume == NULL)
        {
            Covise::sendError("no volume data created!");
            return returnObject;
        }
        if (!volume->objectOk())
        {
            Covise::sendError("creation of volume data in shm  failed!");
            return returnObject;
        }
        delete[] Density;
        delete[] vector_length;
    }
    // assemble output
    returnObject[0] = gridOut;
    if (sDataOut)
        returnObject[1] = sDataOut;
    else
        returnObject[1] = vDataOut;
    returnObject[2] = volume;
    returnObject[3] = NULL;

    // done
    return (returnObject);
}

/**************************************************************************
 *						SAMPLING-FUNCTIONS								  *
 **************************************************************************/

void Sample::SampleMean(int n, int nv, float *px, float *py, float *pz, float *scalars, int size_i, int size_j, int size_k)
{

    cerr << "sampling " << n << " points" << endl;
    float s_min = 0.0;
    float s_max = 0.0;
    int size = size_i * size_j * size_k;
    int i, j, k, idx, c;
    int *numofparticles = new int[size];
    float *scalarsum = new float[size];
    for (idx = 0; idx < size; idx++)
    {
        numofparticles[idx] = 0;
        scalarsum[idx] = 0.0;
    }

    float dp[3] = {
        p_max[0] - p_min[0],
        p_max[1] - p_min[1],
        p_max[2] - p_min[2]
    };

    for (c = 0; c < n; c++)
    {
        int i = (size_i - 1) * (px[c] - p_min[0]) / dp[0];
        int j = (size_j - 1) * (py[c] - p_min[1]) / dp[1];
        int k = (size_k - 1) * (pz[c] - p_min[2]) / dp[2];
        int idx = INDEX(i, j, k);
        numofparticles[idx]++;
        scalarsum[idx] += scalars[c];
    }

    for (idx = 0; idx < size; idx++)
    {
        if (numofparticles[idx] != 0)
            scalarsum[idx] /= (float)numofparticles[idx];
    }

    for (i = 0; i < size_i; i++)
        for (j = 0; j < size_j; j++)
            for (k = 0; k < size_k; k++)
            {
                idx = INDEX(i, j, k);
                if (numofparticles[idx] != 0)
                {
                    so[idx] = scalarsum[idx];
                }
                else
                {
                    so[idx] = getAverage(i, j, k, size_i, size_j, size_k, numofparticles, scalarsum);
                }
            }

    if (createVolume)
    {
        cerr << "compute min/max positions and scalar values" << endl;
        for (c = 0; c < n; c++)
        {
            s_min = Min(s_min, scalars[c]);
            s_max = Max(s_max, scalars[c]);
        }
        float ds = s_max - s_min;
        cerr << "compute voxel values" << endl;
        for (i = 0; i < size_i; i++)
            for (j = 0; j < size_j; j++)
                for (k = 0; k < size_k; k++)
                {
                    idx = INDEX(i, j, k);
                    Density[idx] = 0;
                    Density[idx] = 255 * scalarsum[idx] / ds;
                }
    }

    delete[] numofparticles;
    delete[] scalarsum;
    cerr << "done sampling" << endl;
    // sprintf(Filename,"volume%d.den",num_volume);
    //  WriteDensity(Filename,Density, size);
}

void Sample::SampleMeanv(int n, int nv, float *px, float *py, float *pz, float *scalars, int size_i, int size_j, int size_k)
{

    cerr << "sampling " << n << " points" << endl;
    int size = size_i * size_j * size_k;
    int i, j, k, c, idx;
    int *numofparticles = new int[size];
    float *xs = new float[size];
    float *ys = new float[size];
    float *zs = new float[size];
    for (idx = 0; idx < size; idx++)
    {
        numofparticles[idx] = 0;
        xs[idx] = 0.0;
        ys[idx] = 0.0;
        zs[idx] = 0.0;
    }

    float dp[3] = {
        p_max[0] - p_min[0],
        p_max[1] - p_min[1],
        p_max[2] - p_min[2]
    };

    for (c = 0; c < n; c++)
    {
        int i = (size_i - 1) * (px[c] - p_min[0]) / dp[0];
        int j = (size_j - 1) * (py[c] - p_min[1]) / dp[1];
        int k = (size_k - 1) * (pz[c] - p_min[2]) / dp[2];
        int idx = INDEX(i, j, k);
        numofparticles[idx]++;
        xs[idx] += vx[c];
        ys[idx] += vy[c];
        zs[idx] += vz[c];
    }

    for (idx = 0; idx < size; idx++)
    {
        xs[idx] /= (float)numofparticles[idx];
        ys[idx] /= (float)numofparticles[idx];
        zs[idx] /= (float)numofparticles[idx];
    }

    for (i = 0; i < size_i; i++)
        for (j = 0; j < size_j; j++)
            for (k = 0; k < size_k; k++)
            {
                idx = INDEX(i, j, k);
                if (numofparticles[idx] != 0)
                {
                    vxo[idx] = xs[idx];
                    vyo[idx] = ys[idx];
                    vzo[idx] = zs[idx];
                }
                else
                {
                    getAverage(i, j, k, size_i, size_j, size_k, numofparticles, xs, ys, zs);
                }
            }

    delete[] numofparticles;
    delete[] xs;
    delete[] ys;
    delete[] zs;
    cerr << "done sampling" << endl;
    // sprintf(Filename,"volume%d.den",num_volume);
    //  WriteDensity(Filename,Density, size);
}

void Sample::getAverage(int i, int j, int k, int size_i, int size_j, int size_k, int *numofparticles, float *xs, float *ys, float *zs)
{
    float xsumm = 0.0;
    float ysumm = 0.0;
    float zsumm = 0.0;
    int n = 0, l, m, o, num, idx, dist;
    for (dist = 0; dist < fillDistance; dist++)
    {
        num = (dist * 2) + 1;
        for (l = 0; l < num; l++)
        {
            if (((i - dist + l) < 0) || ((i - dist + l) >= size_i))
                continue;
            for (m = 0; m < num; m++)
            {
                if (((j - dist + m) < 0) || ((j - dist + m) >= size_j))
                    continue;
                for (o = 0; o < num; o++)
                {
                    if (((k - dist + o) < 0) || ((k - dist + o) >= size_k))
                        continue;
                    idx = INDEX((i - dist + l), (j - dist + m), (k - dist + o));
                    if (numofparticles[idx])
                    {
                        n++;
                        xsumm += xs[idx];
                        ysumm += ys[idx];
                        zsumm += zs[idx];
                    }
                }
            }
        }
        if (n > 0)
            break;
    }

    if (n)
    {
        vxo[INDEX(i, j, k)] = xsumm / (float)n;
        vyo[INDEX(i, j, k)] = ysumm / (float)n;
        vzo[INDEX(i, j, k)] = zsumm / (float)n;
        return;
    }
    vxo[INDEX(i, j, k)] = 0.0;
    vyo[INDEX(i, j, k)] = 0.0;
    vzo[INDEX(i, j, k)] = 0.0000000000001;
    if (sentWarning == 0)
    {
        Covise::sendInfo("found cell with no Data in it (use corser Grid or more fillDistance)");
        sentWarning = 1;
    }
}

float Sample::getAverage(int i, int j, int k, int size_i, int size_j, int size_k, int *numofparticles, float *xs)
{
    float xsumm = 0.0;
    int n = 0, l, m, o, num, idx, dist;
    for (dist = 0; dist < fillDistance; dist++)
    {
        num = (dist * 2) + 1;
        for (l = 0; l < num; l++)
        {
            if (((i - dist + l) < 0) || ((i - dist + l) >= size_i))
                continue;
            for (m = 0; m < num; m++)
            {
                if (((j - dist + m) < 0) || ((j - dist + m) >= size_j))
                    continue;
                for (o = 0; o < num; o++)
                {
                    if (((k - dist + o) < 0) || ((k - dist + o) >= size_k))
                        continue;
                    idx = INDEX((i - dist + l), (j - dist + m), (k - dist + o));
                    if (numofparticles[idx])
                    {
                        n++;
                        xsumm += xs[idx];
                    }
                }
            }
        }
        if (n > 0)
            break;
    }

    if (n)
    {
        return (xsumm / (float)n);
    }
    if (sentWarning == 0)
    {
        Covise::sendInfo("found cell with no Data in it (use corser Grid or more fillDistance)");
        sentWarning = 1;
    }
    return (0.0);
}

void Sample::SampleDensity(int n, int nv, float *px, float *py, float *pz, int size_i, int size_j, int size_k)
{

    cerr << "sampling " << n << " points" << endl;
    // sampling n points
    int size = size_i * size_j * size_k;
    int c;
    int *numofparticles = new int[size];
    for (int idx = 0; idx < size; idx++)
        numofparticles[idx] = 0;
    // compute min/max positions
    float dp[3] = {
        p_max[0] - p_min[0],
        p_max[1] - p_min[1],
        p_max[2] - p_min[2]
    };

    // compute nearest voxels
    int nmax = 0;
    for (c = 0; c < n; c++)
    {
        int i = (size_i - 1) * (px[c] - p_min[0]) / dp[0];
        int j = (size_j - 1) * (py[c] - p_min[1]) / dp[1];
        int k = (size_k - 1) * (pz[c] - p_min[2]) / dp[2];
        int idx = INDEX(i, j, k);
        numofparticles[idx]++;
        nmax = Max(nmax, numofparticles[idx]);
    }

    if (createVolume)
    {
        // compute voxel values
        for (int i = 0; i < size_i; i++)
            for (int j = 0; j < size_j; j++)
                for (int k = 0; k < size_k; k++)
                {
                    int idx = INDEX(i, j, k);
                    Density[idx] = 0;
                    if (numofparticles[idx] != 0)
                        Density[idx] = 255 * numofparticles[idx] / nmax;
                }
    }

    delete[] numofparticles;
}

void Sample::WriteDensity(char *filename, char *density, int size)
{
    int density_fd;
    if ((density_fd = creat(filename, 0755)) < 0)
    {
        cerr << "can't create file " << filename << endl;
        return;
    }
    if ((density_fd = Covise::open(filename, O_WRONLY)) < 0)
    {
        cerr << "could not open " << filename << endl;
        return;
    }
    /* write  the raw data */
    if (write(density_fd, density, size) != size)
    {
        cerr << "could not write data to " << filename << endl;
        return;
    }
    close(density_fd);
}
