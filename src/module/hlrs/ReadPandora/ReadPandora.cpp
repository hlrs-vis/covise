// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: "Hello, world!" in COVISE API                          ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Andreas Werner                           ++
// ++               Computer Center University of Stuttgart               ++
// ++                            Allmandring 30                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  10.01.2000  V2.0                                             ++
// ++**********************************************************************/

// this includes our own class's headers
#include "ReadHDF5.h"
#include <hdf5.h>
#include <hdf5_hl.h>
#include <do/coDoPoints.h>
#include <do/coDoData.h>

using namespace covise;

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ReadHDF5::ReadHDF5(int argc, char *argv[])
    : coModule(argc, argv, "simple reader for HDF5 data")
{
    filename = addFileBrowserParam("filename", "HDF5 file name");
    pointsOut = addOutputPort("points", "Points", "positions");
    uOut = addOutputPort("u", "Float", "x velocity");
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once every time the module is executed
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int ReadHDF5::compute(const char *port)
{
    (void)port;
    hsize_t dims[3];
    hsize_t dimU;
    herr_t status;
    size_t i, j, nrow, n_values;

    /* open file from ex_lite1.c */
    hid_t file_id = H5Fopen(filename->getValue(), H5F_ACC_RDONLY, H5P_DEFAULT);
    sendInfo("file_id: %d", (int)file_id);

    /* get the dimensions of the dataset */
    status = H5LTget_dataset_info(file_id, "/Step#1/X", &dims[0], NULL, NULL);
    status = H5LTget_dataset_info(file_id, "/Step#1/Y", &dims[1], NULL, NULL);
    status = H5LTget_dataset_info(file_id, "/Step#1/Z", &dims[2], NULL, NULL);
    if (dims[0] != dims[1])
        return STOP_PIPELINE;
    if (dims[0] != dims[2])
        return STOP_PIPELINE;

    status = H5LTget_dataset_info(file_id, "/Step#1/u", &dimU, NULL, NULL);
    if (dims[0] != dimU)
        return STOP_PIPELINE;

    coDoPoints *points = new coDoPoints(pointsOut->getObjName(), dims[0]);
    float *x, *y, *z;
    points->getAddresses(&x, &y, &z);
    coDoFloat *u = new coDoFloat(uOut->getObjName(), dimU);
    float *du = u->getAddress();

    /* read dataset */
    status = H5LTread_dataset_float(file_id, "/Step#1/X", x);
    status = H5LTread_dataset_float(file_id, "/Step#1/Y", y);
    status = H5LTread_dataset_float(file_id, "/Step#1/Z", z);
    status = H5LTread_dataset_float(file_id, "/Step#1/u", du);

    pointsOut->setCurrentObject(points);
    uOut->setCurrentObject(u);

    /* close file */
    status = H5Fclose(file_id);

    return CONTINUE_PIPELINE;
}

// instantiate an object of class ReadHDF5 and register with COVISE
MODULE_MAIN(Module, ReadHDF5)
