/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
 **                                                                        **
 **                                                                        **
 **                               (C) 1997                                 **
 **                             Paul Benoelken                             **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 ** Author: Paul Benoelken                                                 **
 **                                                                        **
 ** Date:  05.02.97  V1.0                                                  **
 ****************************************************************************/

// C library stuff
#include <stdlib.h>
#include <stdio.h>
#include <iostream.h>
#include <string.h>

#define Min(a, b) (a < b) ? a : b;
#define Max(a, b) (a > b) ? a : b;

// COVISE include stuff
#include <appl/ApplInterface.h>
#include <covise/covise_volumedata.h>

#define INDEX(i, j, k) i *size_y *size_z + j *size_z + k

void Scalar2Volume(void *userData, void *callbackData);

coDoFloat *ScalarData;
DO_Volume_Data *VolumeData;

void main(int argc, char *argv[])
{
    // Initialize the software environment

    Covise::set_module_description("conversion module for scalar-data");
    Covise::add_port(INPUT_PORT, "ScalarData", "coDoFloat", "scalar data");
    Covise::add_port(OUTPUT_PORT, "Volume", "DO_Volume_Data", "volume data");

    Covise::init(argc, argv);

    Covise::set_start_callback(Scalar2Volume, NULL);

    Covise::main_loop();
}

void Scalar2Volume(void *userData, void *callbackData)
{
    char *name;
    // getting scalar data - object from shared memory
    name = Covise::get_object_name("ScalarData");
    ScalarData = new coDoFloat(name);
    if (!ScalarData->objectOk())
    {
        Covise::sendError("Error creating Scalar Data !");
        return;
    }
    int size_x, size_y, size_z;
    // getting grid sizes
    ScalarData->getGridSize(&size_x, &size_y, &size_z);

    // getting scalar values
    float *scalar_values;
    ScalarData->getAddress(&scalar_values);
    // computing min and max values
    float min = 0.0, max = 0.0;
    int i, j, k;
    for (i = 0; i < size_x; i++)
        for (j = 0; j < size_y; j++)
            for (k = 0; k < size_z; k++)
            {
                min = Min(scalar_values[INDEX(i, j, k)], min);
                max = Max(scalar_values[INDEX(i, j, k)], max);
            }
    cerr << "min :" << min << endl;
    cerr << "max :" << max << endl;
    char *density = new char[size_x * size_y * size_z];
    // converting floating point scalar data to byte volume data
    float dividor = max - min;
    for (i = 0; i < size_x; i++)
        for (j = 0; j < size_y; j++)
            for (k = 0; k < size_z; k++)
                density[INDEX(i, j, k)] = 255 * (scalar_values[INDEX(i, j, k)] / dividor);
    // putting volume data to shared memory
    name = Covise::get_object_name("Volume");
    VolumeData = new DO_Volume_Data(name, size_z, size_y, size_x, density);

    if (!VolumeData->objectOk())
    {
        Covise::sendError("Error creating Volume Data !");
        return;
    }
}
