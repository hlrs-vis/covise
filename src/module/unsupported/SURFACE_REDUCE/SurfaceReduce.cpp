/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description:  COVISE Surface reduction application module              **
 **                                                                        **
 **                                                                        **
 **                             (C) 1997                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30a                             **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Karin Frank                                                   **
 **                                                                        **
 **                                                                        **
 ** Date:  April 1997  V1.0                                                **
\**************************************************************************/
#define _IEEE 1

#include <appl/ApplInterface.h>
#include "SurfaceReduce.h"
#include "SurfEdge.h"
#include <math.h>
#include <stdio.h>
#include <time.h>

#include "HandleSet.h"

int *vl_in, *pl_in, *tl;
int set_num_elem = 0;
float percent;
float feature_angle;
int new_point;
float *x_in = NULL;
float *y_in = NULL;
float *z_in = NULL;
float *nu_in = NULL;
float *nv_in = NULL;
float *nw_in = NULL;

char *dtype, *gtype, *ntype;
char *MeshIn, *MeshOut;
char *NormalsIn, *NormalsOut;

//  Shared memory data
coDoTriangleStrips *tmesh_in = NULL;
coDoTriangleStrips *tmesh_out = NULL;
coDoPolygons *mesh_in = NULL;
coDoPolygons *mesh_out = NULL;
coDoVec3 *normals_in = NULL;
coDoVec3 *normals_out = NULL;

//
// static stub callback functions calling the real class
// member functions
//

void main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();
}

void Application::quitCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->quit(callbackData);
}

void Application::computeCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->compute(callbackData);
}

//======================================================================
// Called before module exits
//======================================================================
void Application::quit(void *)
{
    //
    // ...... delete your data here .....
    //
    Covise::log_message(__LINE__, __FILE__, "Quitting now");
}

//======================================================================
// Computation routine (called when START message arrives)
//======================================================================
void Application::compute(void *)
{
    char *inportnames[] = { "meshIn", "normalsIn", NULL };
    char *outportnames[] = { "meshOut", "normalsOut", NULL };

    //	get parameter
    Covise::get_scalar_param("percent", &percent);
    Covise::get_scalar_param("feature_angle", &feature_angle);
    Covise::get_choice_param("new_point", &new_point);

    if ((percent > 100) || (percent < 0))
    {
        Covise::sendWarning("Parameter 'percent' out of range, set to default");
        percent = 30.0;
    }

    if ((feature_angle < 0) || (feature_angle > 180))
    {
        Covise::sendWarning("Parameter 'feature_angle' out of range, set to default");
        feature_angle = 40.0;
    }

    Compute(2, inportnames, outportnames);
}

coDistributedObject **Application::ComputeObject(coDistributedObject **data_in, char **out_name, int)
{
    static coDistributedObject **r;

    r = HandleObjects(data_in[0], data_in[1], out_name[0], out_name[1]);

    return (r);
}

coDistributedObject **Application::HandleObjects(coDistributedObject *mesh_object, coDistributedObject *normal_object, char *Mesh_out_name, char *Normal_out_name)
{
    int numpoints, numvertices, numpolygons;
    int red_tri, red_points;
    SurfaceEdgeCollapse *surf = NULL;
    int ndata;

    coDistributedObject **DO_return;

    if (mesh_object == NULL)
        return (NULL);

    if (mesh_object->objectOk())
    {
        gtype = mesh_object->getType();

        if (strcmp(gtype, "POLYGN") == 0)
        {
            mesh_in = (coDoPolygons *)mesh_object;
            numpoints = mesh_in->getNumPoints();
            numvertices = mesh_in->getNumVertices();
            numpolygons = mesh_in->getNumPolygons();
            mesh_in->getAddresses(&x_in, &y_in, &z_in, &vl_in, &pl_in);
        }
        else if (strcmp(gtype, "TRIANG") == 0)
        {
            tmesh_in = (coDoTriangleStrips *)mesh_object;
            numpoints = tmesh_in->getNumPoints();
            numvertices = tmesh_in->getNumVertices();
            numpolygons = tmesh_in->getNumStrips();
            tmesh_in->getAddresses(&x_in, &y_in, &z_in, &vl_in, &pl_in);
        }
        else
        {
            Covise::sendError("ERROR: Data object 'meshIn' has wrong data type");
            return (NULL);
        }
    }
    else
    {
        Covise::sendError("ERROR: Data object 'meshIn' can't be accessed in shared memory");
        return (NULL);
    }
    if (numpoints == 0 || numvertices == 0 || (numpolygons == 0))
    {
        return (NULL);
    }

    if (normal_object == NULL)
    {
        Covise::sendError("ERROR: Data object 'normalsIn' can't be accessed in shared memory");
        return (NULL);
    }
    else
    {
        if (normal_object->objectOk())
        {
            ntype = normal_object->getType();

            if (strcmp(ntype, "USTVDT") == 0)
            {
                normals_in = (coDoVec3 *)normal_object;
                ndata = normals_in->getNumPoints();
                normals_in->getAddresses(&nu_in, &nv_in, &nw_in);
                if (ndata != numpoints)
                {
                    Covise::sendError("ERROR: Size of data object 'NormalsIn' does not match mesh size");
                    return (NULL);
                }
            }
            else
            {
                Covise::sendError("ERROR: Data object 'NormalsIn' has wrong data type");
                return (NULL);
            }
        }
        else
        {
            Covise::sendError("ERROR: Data object 'NormalsIn' can't be accessed in shared memory");
            return (NULL);
        }
    }

    surf = new SurfaceEdgeCollapse(numpoints, numvertices, numpolygons, gtype, pl_in, vl_in, x_in, y_in, z_in, nu_in, nv_in, nw_in);
    surf->Set_Percent(percent);
    surf->Set_FeatureAngle(feature_angle);
    surf->Set_NewPoint(new_point);
    surf->Reduce(red_tri, red_points);
    DO_return = surf->createcoDistributedObjects(red_tri, red_points, Mesh_out_name, NULL, Normal_out_name);
    delete surf;

    DO_return[1] = DO_return[2];

    return (DO_return);
}
