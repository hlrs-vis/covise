


/******************************************************************
 *
 *    SNAP To Surface
 *
 *
 *  Description: Snap points to a surface
 *  Date: 02.06.19
 *  Author: Leyla Kern
 *
 *******************************************************************/


#include "Snap2Surface.h"
#include <do/coDoPoints.h>
#include <do/covise_gridmethods.h>

Snap2Surface::Snap2Surface(int argc, char *argv[])
                           :coSimpleModule (argc, argv,"Snap points to a polygonal surface")
{
      p_pointsIn = addInputPort("points", "Points", "point data");
      p_surfaceIn = addInputPort("surface","Polygons","surface to snap points to");

      p_pointsOut = addOutputPort("pointsOut", "Points", "point data");

      p_axis = addChoiceParam("axis","snap along selected axis");
      p_axis->setValue(3, {"x","y","z"}, 2);
      p_delta = addFloatParam("distance","lift object from surface by distance");
      p_delta->setValue(0.0f);
}

bool Snap2Surface::outOfBox(grid_methods::POINT3D pt , int ax, grid_methods::POINT3D max, grid_methods::POINT3D min)
{
    switch (ax) {
    case 0:
        return ((pt.y < min.y) || (pt.z < min.z) || (pt.y > max.y) || (pt.z > max.z));
    case 1:
        return ((pt.x < min.x) || (pt.z < min.z) || (pt.x > max.x) || (pt.z > max.z));
    case 2:
        return ((pt.x < min.x) || (pt.y < min.y) || (pt.x > max.x) || (pt.y > max.y));
    }
    return false;
}


bool Snap2Surface::outOfQuad(int* idx, grid_methods::POINT3D pt, int ax)
{
    float* x_ = new float[4];
    float* y_ = new float[4];
    float* z_ = new float[4];

    grid_methods::POINT3D min, max;

    for (int i = 0; i < 4; ++i)
    {
        x_[i] = x_surf[idx[i]];
        y_[i] = y_surf[idx[i]];
        z_[i] = z_surf[idx[i]];
    }

    max.x = *std::max_element(&x_[0], &x_[0] + 4);
    max.y = *std::max_element(&y_[0], &y_[0] + 4);
    max.z = *std::max_element(&z_[0], &z_[0] + 4);
    min.x = *std::min_element(&x_[0], &x_[0] + 4);
    min.y = *std::min_element(&y_[0], &y_[0] + 4);
    min.z = *std::min_element(&z_[0], &z_[0] + 4);

    switch (ax) {
    case 0:
        return ((pt.y < min.y) || (pt.z < min.z) || (pt.y > max.y) || (pt.z > max.z));
    case 1:
        return ((pt.x < min.x) || (pt.z < min.z) || (pt.x > max.x) || (pt.z > max.z));
    case 2:
        return ((pt.x < min.x) || (pt.y < min.y) || (pt.x > max.x) || (pt.y > max.y));
    }

    return false;
}

int Snap2Surface::compute(const char *)
{
    float *x_in,*y_in,*z_in;
    float *x_out,*y_out,*z_out;
    int *vl_surf, *pl_surf;

    float delta;
    int axis = 2;

    const coDistributedObject *pointsIn = p_pointsIn->getCurrentObject();
    if (!pointsIn)
    {
        sendError("Could not get input points");
        return STOP_PIPELINE;
    }
    const coDoPoints *points = dynamic_cast<const coDoPoints *>(pointsIn);
    if (!points)
    {
        sendError("Wrong type for input points");
        return STOP_PIPELINE;
    }

    const coDistributedObject *surfIn = p_surfaceIn->getCurrentObject();
    if(!surfIn)
    {
        sendError("Could not get input surface");
        return  STOP_PIPELINE;
    }
    const coDoPolygons *surf = dynamic_cast<const coDoPolygons*>(surfIn);
    if (!surf)
    {
        sendError("Wrong type for input polygons");
        return STOP_PIPELINE;
    }

    int numPoints = points->getNumPoints();
    points->getAddresses(&x_in, &y_in, &z_in);

    int numPolyCoords = surf->getNumPoints();
    int numPoly = surf->getNumPolygons();
    //int numPolyVert = surf->getNumVertices();
    surf->getAddresses(&x_surf, &y_surf, &z_surf, &vl_surf, &pl_surf);
    grid_methods::POINT3D max, min;
    max.x = *std::max_element(&x_surf[0], &x_surf[0] + numPolyCoords);
    max.y = *std::max_element(&y_surf[0], &y_surf[0] + numPolyCoords);
    max.z = *std::max_element(&z_surf[0], &z_surf[0] + numPolyCoords);
    min.x = *std::min_element(&x_surf[0], &x_surf[0] + numPolyCoords);
    min.y = *std::min_element(&y_surf[0], &y_surf[0] + numPolyCoords);
    min.z = *std::min_element(&z_surf[0], &z_surf[0] + numPolyCoords);

    delta = p_delta->getValue();
    axis = p_axis->getValue();
    //printf("numPoly %d\n", numPoly);

    coDoPoints *pointsOut = new coDoPoints(p_pointsOut->getNewObjectInfo(), numPoints); //TODO: size
    pointsOut->getAddresses(&x_out, &y_out, &z_out);

    for (int i = 0; i < numPoints; ++i)
    {
        x_out[i]=x_in[i];
        y_out[i]=y_in[i];
        z_out[i]=z_in[i];
    }

    //compute intersection of each pt with each traingle of the surface
    grid_methods::POINT3D end_pt, inter_pt;
    grid_methods::POINT3D query_pt;
    query_pt.x = 0.0;
    query_pt.y = 0.0;
    query_pt.z = 0.0;

    float* tr_x = new float[3];
    float* tr_y = new float[3];
    float* tr_z = new float[3];
   // int idx0, idx1, idx2, idx3;
    int* idx = new int[4];
    char code = '0';


    //TODO: sort polygons into tree structure
    for (int i = 0; i < numPoints; ++i)
    {
        // set query and end such that a ray parallel to chosen axis is created through the query point
        query_pt.x = x_in[i];
        query_pt.y = y_in[i];
        query_pt.z = z_in[i];

        end_pt.x = query_pt.x;
        end_pt.y = query_pt.y;
        end_pt.z = query_pt.z;

        //only snap if point lies within bounding box of surface, o.w. skip this point
        if (outOfBox( query_pt , axis, max,  min))
            continue;

        switch (axis)
        {
        case 0:
             query_pt.x = max.x + 1;
             end_pt.x = min.x - 1;
            break;
        case 1:
            query_pt.y = max.y + 1;
            end_pt.y = min.y -1;
            break;
        case 2:
            query_pt.z = max.z + 1;
            end_pt.z = min.z - 1 ;
            break;
        }

        //TODO: change to polygon -> tesselate?
        for (int j = 0; j < numPoly; ++j)
        {
            //compute indices for coords of vertices
            for (int k = 0; k < 4; ++k)
            {
                idx[k] = vl_surf[pl_surf[j] + k];
            }
            for (int k = 0; k < 3; ++k)
            {
                tr_x[k] = x_surf[idx[k]];
                tr_y[k] = y_surf[idx[k]];
                tr_z[k] = z_surf[idx[k]];
            }

            //only check intersection if point is within x,y boundaries of current quadrangle
            if (outOfQuad(idx, query_pt, axis))
            {
                continue;
            }else
            { //if pt within boundaries -> get intersection with plane spaned by quadrangle
                int comp_idx;
                code = grid_methods::RayPlaneIntersection(tr_x, tr_y, tr_z, query_pt, end_pt, inter_pt, comp_idx);
                if (code != '0')
                    break;
            }
        }

        //set coordinate of surface for pt, only if an intersection was found
        if (code != '0')
        {
            switch (axis)
            {
            case 0:
                x_out[i] = inter_pt.x + delta;
                break;
            case 1:
                y_out[i] = inter_pt.y + delta;
                break;
            case 2:
                z_out[i] = inter_pt.z + delta;
                break;
            }
        }
    }

    p_pointsOut->setCurrentObject(pointsOut);
    return CONTINUE_PIPELINE;
}
MODULE_MAIN(Tools,Snap2Surface)
