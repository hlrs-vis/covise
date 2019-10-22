/******************************************************************
 *
 *    NNSample
 *
 *
 *  Description: Sample a surface from points using nearest neighbors
 *  Date: 04.06.19
 *  Author: Leyla Kern
 *
 *******************************************************************/


#include "NNSample.h"
#include "api/coFeedback.h"
#include <do/coDoSet.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoData.h>

NNSample::NNSample(int argc, char *argv[])
    :coSimpleModule (argc, argv, "Sample values of points to a grid using neares neighbor method")
{
    Points_In_Port = addInputPort("PointsIn", "Points", "Points input");
    Data_In_Port = addInputPort("DataIn", "Float|Vec3", "Data input");
    Reference_Grid_In_Port = addInputPort("ReferenceGridIn", "StructuredGrid", "Reference Grid containing true coordinates");
    Unigrid_In_Port =  addInputPort("UniformReferenceGridIn", "StructuredGrid", "Uniform Grid");
    Unigrid_In_Port->setRequired(0);

    Grid_Out_Port = addOutputPort("GridOut", "StructuredGrid", "Grid Output Port");
    Data_Out_Port = addOutputPort("DataOut", "Float|Vec3", "Data Output Port");
}

NNSample::~NNSample(){

}

void NNSample::param(const char *name, bool inMapLoading)
{

}

//compare squared distance for efficiency
float distance_sq(float x, float y, float z, float p_x, float p_y, float p_z)
{
    // 2D distance
    float x_ = x-p_x;
    float y_ = y-p_y;
   // float z_ = z-p_z;
    return (x_*x_+ y_*y_/*+ z_*z_*/);
}

int NNSample::nearestNeighborIDX(float x, float y, float z)
{
    float min_val = 100000;
    int min_idx = numPoints;

    for (int j = 0; j < numPoints; ++j) //loop over points
    {
       float d = distance_sq(x, y, z, p_x[j], p_y[j], p_z[j]);
       if (d < min_val)
       {
           min_idx = j;
           min_val = d;
       }
    }
    return min_idx;
}

int NNSample::compute(const char *)
{
    bool unigr = false;
    //TODO: handle TIMESTEP data
    const coDistributedObject *refObj = Reference_Grid_In_Port->getCurrentObject();
    if (!refObj)
    {
        sendError("cannot retrieve reference grid");
        return STOP_PIPELINE;

    }
    const coDoStructuredGrid *refGrid = dynamic_cast<const coDoStructuredGrid*>(refObj);
    if (!refGrid)
    {
        sendError("Wrong type of reference object. expected grid");
    }

    const coDistributedObject *pointObj = Points_In_Port->getCurrentObject();
    if (!pointObj)
    {
        sendError("could not retrieve input for points");
        return STOP_PIPELINE;
    }
    const coDoPoints *pointList = dynamic_cast<const coDoPoints *>(pointObj);
    if (!pointList)
    {
        sendError("wrong type for point object");
        return STOP_PIPELINE;
    }

    const coDistributedObject *dataObj = Data_In_Port->getCurrentObject();
    if (!dataObj)
    {
        sendError("could not retrieve input for data");
        return STOP_PIPELINE;
    }
    const coDoFloat *dataList = dynamic_cast<const coDoFloat *>(dataObj);
    if (!dataList)
    {
        sendError("wrong type for data object. expected float");
        return STOP_PIPELINE;
    }

    const char *Grid_outObjName = Grid_Out_Port->getObjName();
    const char *Data_outObjName = Data_Out_Port->getObjName();

    float *x_c, *y_c, *z_c, *x_o, *y_o, *z_o, *val, *p_val;
    int x_s, y_s, z_s;

    refGrid->getAddresses(&x_c, &y_c, &z_c);
    refGrid->getGridSize(&x_s, &y_s, &z_s);
    int numCoords = refGrid->getNumPoints();

    coDoStructuredGrid *outGrid = new coDoStructuredGrid(Grid_outObjName, x_s, y_s, z_s);
    outGrid->getAddresses(&x_o, &y_o, &z_o);

    coDoFloat *outData = new coDoFloat(Data_outObjName, numCoords);
    outData->getAddress(&val);

    numPoints = pointList->getNumPoints();
    pointList->getAddresses(&p_x, &p_y, &p_z);
    dataList->getAddress(&p_val);


    float *u_x, *u_y, *u_z;
    const coDistributedObject *uniGrid = Unigrid_In_Port->getCurrentObject();
    if (uniGrid)
    {
        unigr = true;
        const coDoStructuredGrid *refUniGrid = dynamic_cast<const coDoStructuredGrid*>(uniGrid);
        int nx, ny, nz;
        refUniGrid->getGridSize(&nx, &ny, &nz);
        refUniGrid->getAddresses(&u_x, &u_y, &u_z);

    }

    // loop over vertices in refGrid (=outGrid)
    for (int i = 0; i < numCoords ; ++i)
    {

        int idx = nearestNeighborIDX(x_c[i], y_c[i], z_c[i]);
        if (unigr)
        {
            x_o[i] = u_x[i]; //NOTE: troubles here when mapping points: swap u_x and u_y
            y_o[i] = u_y[i];
            z_o[i] = u_z[i];
        }else
        {
            x_o[i] = x_c[i];
            y_o[i] = y_c[i];
            z_o[i] = z_c[i];
        }

        if (idx < numPoints)
        {
            //set val of outGrid  at 'j' to val of points at idx
            val[i] = p_val[idx];
        }else
        {
            val[i] = 0.0;
        }
    }

    return CONTINUE_PIPELINE;
}


MODULE_MAIN(Interpolator, NNSample)
