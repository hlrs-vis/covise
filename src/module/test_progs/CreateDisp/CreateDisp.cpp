/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:CreateDisp - creates a displacement field               ++
// ++             Implementation of class <Skel>                          ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 12.02.01                                                      ++
// ++**********************************************************************/

#include "CreateDisp.h"
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
CreateDisp::CreateDisp(int argc, char **argv)
    : coModule(argc, argv, "Create a homegenious displacement field at the points of an unstructured grid")
{

    pTrace1_ = addFloatParam("Eps_xx", "1st element of the trace of the strain matrix ");
    pTrace2_ = addFloatParam("Eps_yy", "2st element of the trace of the strain matrix ");
    pTrace3_ = addFloatParam("Eps_zz", "3st element of the trace of the strain matrix ");

    pTrace4_ = addFloatParam("Eps_xy", "shear in the x-y plane");
    pTrace5_ = addFloatParam("Eps_xz", "shear in the x-z plane");
    pTrace6_ = addFloatParam("Eps_yz", "shear in the y-z plane");

    pRigDisp_ = addFloatVectorParam("disp", "rigid body displacement");

    const float initv = 0.0;

    pTrace1_->setValue(initv);
    pTrace2_->setValue(initv);
    pTrace3_->setValue(initv);
    pTrace4_->setValue(initv);
    pTrace5_->setValue(initv);
    pTrace6_->setValue(initv);

    int i;
    for (i = 0; i < 3; ++i)
        pRigDisp_->setValue(i, initv);

    p_GridInPort_ = addInputPort("gridIn", "coDoUnstructuredGrid|coDoPolygons", "Unstructured Grid or Polygon mesh");

    p_vDataOutPort_ = addOutputPort("vodata", "Set_Vec3", "displacement field");
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
int
CreateDisp::compute(const char *)
{

    const coDistributedObject *gridInObj = p_GridInPort_->getCurrentObject();

    // we should have an object
    if (!gridInObj)
    {
        sendError("Did not receive object at port '%s'", p_GridInPort_->getName());
        return FAIL;
    }
    // it should be the correct type
    if (!(gridInObj->isType("UNSGRD") || gridInObj->isType("POLYGN")))
    {
        sendError("Received illegal type at port '%s'", p_GridInPort_->getName());
        return FAIL;
    }

    // retrieve parameters
    float sXX = pTrace1_->getValue();
    float sYY = pTrace2_->getValue();
    float sZZ = pTrace3_->getValue();

    float sXY = pTrace4_->getValue();
    float sXZ = pTrace5_->getValue();
    float sYZ = pTrace6_->getValue();

    float dX = pRigDisp_->getValue(0);
    float dY = pRigDisp_->getValue(1);
    float dZ = pRigDisp_->getValue(2);

    float *iXCo = NULL, *iYCo = NULL, *iZCo = NULL;
    int numCoord = 0;

    if (gridInObj->isType("UNSGRD"))
    {

        const coDoUnstructuredGrid *inGrid = (const coDoUnstructuredGrid *)gridInObj;

        int *inElemList, *inConnList;
        int numElem, numConn;

        inGrid->getGridSize(&numElem, &numConn, &numCoord);

        inGrid->getAddresses(&inElemList, &inConnList,
                             &iXCo, &iYCo, &iZCo);
    }

    if (gridInObj->isType("POLYGN"))
    {

        const coDoPolygons *inGrid;
        if (!(inGrid = dynamic_cast<const coDoPolygons *>(gridInObj)))
        {
            // this should never occur
            cerr << "FATAL error:  GetSetElem::compute( ) dynamic cast failed in line "
                 << __LINE__ << " of file " << __FILE__ << endl;
        }

        numCoord = inGrid->getNumPoints();

        int *inElemList, *inConnList;
        inGrid->getAddresses(&iXCo, &iYCo, &iZCo, &inElemList, &inConnList);
    }

    const char *vDataName = p_vDataOutPort_->getObjName();
    coDoVec3 *vDataOut = new coDoVec3(vDataName, numCoord);

    if (!vDataOut->objectOk())
    {
        sendError("Failed to create object '%s' for port '%s'",
                  p_vDataOutPort_->getObjName(), p_vDataOutPort_->getName());
        return FAIL;
    }

    // now calculate the displacement field
    float *u;
    float *v;
    float *w;

    vDataOut->getAddresses(&u, &v, &w);

    int i;
    for (i = 0; i < numCoord; ++i)
    {

        float xi = iXCo[i];
        float yi = iYCo[i];
        float zi = iZCo[i];

        u[i] = sXX * xi + sXY * yi + sXZ * zi + dX;
        v[i] = sYY * yi + sXY * xi + sYZ * zi + dY;
        w[i] = sZZ * zi + sXZ * xi + sYZ * yi + dZ;
    }

    p_vDataOutPort_->setCurrentObject(vDataOut);

    return SUCCESS;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  What's left to do for the Main program:
// ++++                                    create the module and start it
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/*
int main( int argc, char *argv[])

{
   // create the module
   CreateDisp *application = new CreateDisp;

   // this call leaves with exit(), so we ...
   application->start(argc,argv);

   // ... never reach this point
   return 0;

}
*/

MODULE_MAIN(Tools, CreateDisp)
