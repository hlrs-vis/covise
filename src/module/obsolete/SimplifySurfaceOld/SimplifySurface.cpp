/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1998 RUS  **
 **                                                                        **
 ** Description:  COVISE Surface reduction SimplifySurface module          **
 **                                                                        **
 **                                                                        **
 **                             (C) 1998                                   **
 **                Computing Center University of Stuttgart                **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Author:  Karin Frank                                                   **
 **                                                                        **
 ** Date:  March 1998  V1.0                                                **
 **									  **
 ** changed to new API:   22. 02. 2001					  **
 **  	 Sven Kufer							  **
 **	 (C) VirCinity IT-Consulting GmbH				  **
 **       Nobelstrasse 15						  **
 **       D- 70569 Stuttgart                                           	  **
\**************************************************************************/

#define _IEEE 1

#include "SimplifySurface.h"
#include "SurfVertex.h"
#include "SurfEdge.h"
#include "SurfVertexData.h"
#include "SurfEdgeData.h"
#include <util/coviseCompat.h>

int *vl_in, *pl_in;
int set_num_elem = 0;

float percent;
float feature_angle;
float volume_bound;
//float alpha;
//int which_curv;
//int which_grad;
int strategy;
//int new_point;
float *x_in;
float *y_in;
float *z_in;
float *s_in = NULL;
float *u_in = NULL;
float *v_in = NULL;
float *w_in = NULL;
float *nu_in = NULL;
float *nv_in = NULL;
float *nw_in = NULL;

char *dtype, *gtype, *ntype;
char *MeshIn, *MeshOut;
char *DataIn, *DataOut;
char *NormalsIn, *NormalsOut;
char *colorn;
char *color[10] = {
    "yellow", "green", "blue", "red", "violet", "chocolat",
    "linen", "pink", "crimson", "indigo"
};

//  Shared memory data
coDoPolygons *mesh_in = NULL;
coDoPolygons *mesh_out = NULL;
coDoTriangleStrips *tmesh_in = NULL;
coDoVec3 *normals_in = NULL;
coDoVec3 *normals_out = NULL;
coDoFloat *data_in = NULL;
coDoFloat *data_out = NULL;
coDoVec3 *vdata_in = NULL;
coDoVec3 *vdata_out = NULL;

SimplifySurface::SimplifySurface(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Data-dependent reduction of a triangulated surface")
{
    p_meshIn = addInputPort("meshIn", "Polygons|TriangleStrips", "Geometry");
    p_dataIn = addInputPort("dataIn", "Float|Vec3", "Vertice-attached data");
    p_dataIn->setRequired(0);

    p_normalsIn = addInputPort("normalsIn", "Vec3", "Vertice-attached normals");
    p_normalsIn->setRequired(0);
    p_meshOut = addOutputPort("meshOut", "Polygons", "The reduced geometry");
    p_dataOut = addOutputPort("dataOut", "Float|Vec3", "The interpolated scalar data");
    //p_dataOut->setDependency(p_dataIn);
    p_normalsOut = addOutputPort("normalsOut", "Vec3", "The interpolated normals");

    param_strategy = addChoiceParam("strategy", "Which strategy to use");
    char *choices[] = { "EdgeCollapse", "VertexRemoval" };
    param_strategy->setValue(2, choices, 0);

    param_percent = addFloatParam("percent", "Percentage of triangles to be left after simplification");
    param_percent->setValue(30.0);

    param_volume_bound = addFloatParam("volume_bound", "Do not remove vertices causing a greater loss of volume");
    param_volume_bound->setValue(0.0001);

    param_feature_angle = addFloatParam("feature_angle", "Preserve feature edges enclosing a less angle");
    param_feature_angle->setValue(120.0);

    //Covise::add_port(PARIN,"new_point","Choice","How to compute the new point when edge collapsing");
    //Covise::set_port_default("new_point","1 Midpoint Volume_preservation Endpoint");
    //Covise::add_port(PARIN,"which_curvature","Choice","Which curvature criterion shall be used");
    //Covise::set_port_default("which_curvature","3 L1 Discrete Taubin Hamann");
    //Covise::add_port(PARIN,"which_gradient","Choice","Which gradient criterion shall be used");
    //Covise::set_port_default("which_gradient","1 Directional_Derivatives Least_Square_Gradient Data_Deviation");
    //Covise::add_port(PARIN,"alpha","Scalar","Weight: alpha * curvature + (1 - alpha) * gradient");
    //Covise::set_port_default("alpha","0.5");
}

//======================================================================
// Computation routine (called when START message arrives)
//======================================================================
int SimplifySurface::compute()
{

    //	get parameter
    strategy = param_strategy->getValue();
    percent = param_percent->getValue();
    feature_angle = param_feature_angle->getValue();
    volume_bound = param_volume_bound->getValue();

    //Covise::get_choice_param("new_point", &new_point);
    //Covise::get_choice_param("which_curvature", &which_curv);
    //Covise::get_choice_param("which_gradient", &which_grad);
    //Covise::get_scalar_param("alpha", &alpha);

    if ((percent > 100) || (percent < 0))
    {
        sendWarning("Parameter 'percent' out of range, set to default");
        percent = 40.0;
    }

    if (volume_bound < 0)
    {
        sendWarning("Parameter 'volume_bound' out of range, set to zero");
        volume_bound = 0.0;
    }

    if ((feature_angle < 0) || (feature_angle > 180))
    {
        sendWarning("Parameter 'feature_angle' out of range, set to default");
        feature_angle = 40.0;
    }

    coDistributedObject **returns = HandleObjects(p_meshIn->getCurrentObject(), p_dataIn->getCurrentObject(), p_normalsIn->getCurrentObject(),
                                                  p_meshOut->getObjName(), p_dataOut->getObjName(), p_normalsOut->getObjName());

    if (returns == NULL)
        return STOP_PIPELINE;

    p_meshOut->setCurrentObject(returns[0]);
    p_dataOut->setCurrentObject(returns[1]);
    p_normalsOut->setCurrentObject(returns[2]);

    return CONTINUE_PIPELINE;
}

coDistributedObject **SimplifySurface::HandleObjects(coDistributedObject *mesh_object,
                                                     coDistributedObject *data_object,
                                                     coDistributedObject *normal_object,
                                                     const char *Mesh_out_name,
                                                     const char *Data_out_name,
                                                     const char *Normal_out_name)
{
    int n_pnt, n_vert, n_poly, ndata = 0, nndata;
    int red_tri, red_pnt;
    SurfaceVertexRemoval *gvr = NULL;
    SurfaceEdgeCollapse *gec = NULL;
    ScalarDataVertexRemoval *svr = NULL;
    VectorDataVertexRemoval *vvr = NULL;
    ScalarDataEdgeCollapse *sec = NULL;
    VectorDataEdgeCollapse *vec = NULL;
    int no_data;
    int no_normals;

    coDistributedObject **DO_return;

    if (mesh_object && mesh_object->objectOk())
    {
        gtype = mesh_object->getType();

        if (strcmp(gtype, "POLYGN") == 0)
        {
            mesh_in = (coDoPolygons *)mesh_object;
            n_pnt = mesh_in->getNumPoints();
            n_vert = mesh_in->getNumVertices();
            n_poly = mesh_in->getNumPolygons();
            mesh_in->getAddresses(&x_in, &y_in, &z_in, &vl_in, &pl_in);
            if ((colorn = mesh_in->getAttribute("COLOR")) == NULL)
            {
                colorn = new char[20];
                strcpy(colorn, "yellow");
            }
        }
        else if (strcmp(gtype, "TRIANG") == 0)
        {
            tmesh_in = (coDoTriangleStrips *)mesh_object;
            n_pnt = tmesh_in->getNumPoints();
            n_vert = tmesh_in->getNumVertices();
            n_poly = tmesh_in->getNumStrips();
            tmesh_in->getAddresses(&x_in, &y_in, &z_in, &vl_in, &pl_in);
            if ((colorn = tmesh_in->getAttribute("COLOR")) == NULL)
            {
                colorn = new char[20];
                strcpy(colorn, "yellow");
            }
        }
        else
        {
            Covise::sendError("ERROR: Data object 'meshIn' has wrong data type");
            return (NULL);
        }
    }
    else
    {
        //Covise::sendError("ERROR: Data object 'meshIn' can't be accessed in shared memory");
        //return(NULL);
        n_pnt = n_vert = n_poly = 0;
    }

    if (n_pnt == 0 || n_vert == 0 || n_poly == 0)
    {
        //  Covise::sendWarning("WARNING: Data object 'meshIn' is empty");
        // aw: this is allowed now !!
    }

    dtype = NULL;
    if (data_object != NULL)
    {
        no_data = 0;
        if (data_object->objectOk())
        {
            dtype = data_object->getType();
            if (strcmp(dtype, "USTSDT") == 0)
            {
                data_in = (coDoFloat *)data_object;
                data_in->getAddress(&s_in);
                ndata = data_in->getNumPoints();
            }
            else if (strcmp(dtype, "USTVDT") == 0)
            {
                vdata_in = (coDoVec3 *)data_object;
                vdata_in->getAddresses(&u_in, &v_in, &w_in);
                ndata = vdata_in->getNumPoints();
            }
            else
            {
                Covise::sendError("ERROR: Data object 'DataIn' has wrong data type");
                return NULL;
            }
            if (ndata > 0 && ndata != n_pnt)
            {
                Covise::sendError("ERROR: Size of data object 'DataIn' does not match mesh size");
                return (NULL);
                // no_data=1;
            }
            if (ndata == 0)
                s_in = 0;
        }
        else
        {
            Covise::sendError("ERROR: Data object 'DataIn' can't be accessed in shared memory");
            return (NULL);
        }
    }
    else
    {
        no_data = 1;
        s_in = 0;
    }

    if (normal_object != NULL)
    {
        no_normals = 0;
        if (normal_object->objectOk())
        {
            ntype = normal_object->getType();

            if (strcmp(ntype, "USTVDT") == 0)
            {
                normals_in = (coDoVec3 *)normal_object;
                nndata = normals_in->getNumPoints();
                normals_in->getAddresses(&nu_in, &nv_in, &nw_in);
                if (nndata != n_pnt)
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
    else
    {
        no_normals = 1;
    }

    if (no_normals)
    {
        nu_in = NULL;
        nv_in = NULL;
        nw_in = NULL;
    }

    // aw: Null object handling
    if (n_pnt == 0)
    {
        DO_return = new coDistributedObject *[3];
        DO_return[0] = DO_return[1] = DO_return[2] = NULL;

        if (Mesh_out_name)
            DO_return[0] = new coDoPolygons(Mesh_out_name, 0, 0, 0);

        if (Data_out_name)
            if (dtype)
                if (strcmp(dtype, "USTSDT") == 0)
                    DO_return[1] = new coDoFloat(Data_out_name, 0);
                else if (strcmp(dtype, "USTVDT") == 0)
                    DO_return[1] = new coDoVec3(Data_out_name, 0);

        if (Normal_out_name)
            DO_return[2] = new coDoVec3(Normal_out_name, 0);

        return DO_return;
    }

    if (no_data == 0 && ndata == 0)
    {
        // make dummies, if we have dummy input data
        coDistributedObject **DO_Return = new coDistributedObject *[3];
        DO_Return[0] = new coDoPolygons(Mesh_out_name, 0, 0, 0);
        if (strcmp(dtype, "USTSDT") == 0)
            DO_Return[1] = new coDoFloat(Data_out_name, 0);
        else
            DO_Return[1] = new coDoVec3(Data_out_name, 0);
        DO_Return[2] = new coDoVec3(Normal_out_name, 0);
        return DO_Return;
    }

    if ( // (strategy==1 && no_data && no_normals) ||
        // using SurfaceEdgeCollapse as it is is not good, because
        // it does not create any normals, and this is not acceptable!!!
        (strategy == 2 && no_data))
    {
        switch (strategy)
        {
        case 0:
            gec = new SurfaceEdgeCollapse(n_pnt, n_vert, n_poly, gtype, pl_in, vl_in, x_in, y_in, z_in, nu_in, nv_in, nw_in);
            gec->Set_Percent(percent);
            gec->Set_FeatureAngle(feature_angle);
            gec->Set_VolumeBound(volume_bound);
            gec->Reduce(red_tri, red_pnt);
            DO_return = gec->createcoDistributedObjects(red_tri, red_pnt, Mesh_out_name, Data_out_name, Normal_out_name);
            delete gec;
            break;
        case 1:
            gvr = new SurfaceVertexRemoval(n_pnt, n_vert, n_poly, gtype, pl_in, vl_in, x_in, y_in, z_in, nu_in, nv_in, nw_in);
            gvr->Set_Percent(percent);
            gvr->Set_FeatureAngle(feature_angle);
            gvr->Set_VolumeBound(volume_bound);
            gvr->Reduce(red_tri, red_pnt);
            DO_return = gvr->createcoDistributedObjects(red_tri, red_pnt, Mesh_out_name, Data_out_name, Normal_out_name);
            delete gvr;
            break;
        default:
            Covise::sendError("ERROR: No decimation strategy indicated!");
            return (NULL);
        };
    }
    else // if no_data == 1, then we must needs have normals and strategy==1
    {
        if (no_data)
            dtype = "USTSDT";
        switch (strategy)
        {
        case 0:
            if (strcmp(dtype, "USTSDT") == 0)
            {
                sec = new ScalarDataEdgeCollapse(n_pnt, n_vert, n_poly, gtype, pl_in, vl_in, x_in, y_in, z_in, s_in, nu_in, nv_in, nw_in);
                sec->Set_Percent(percent);
                sec->Set_FeatureAngle(feature_angle);
                sec->Set_VolumeBound(volume_bound);
                sec->Reduce(red_tri, red_pnt);
                DO_return = sec->createcoDistributedObjects(red_tri, red_pnt, Mesh_out_name, Data_out_name, Normal_out_name);
                delete sec;
            }
            else
            {
                vec = new VectorDataEdgeCollapse(n_pnt, n_vert, n_poly, gtype, pl_in, vl_in, x_in, y_in, z_in, u_in, v_in, w_in, nu_in, nv_in, nw_in);
                vec->Set_Percent(percent);
                vec->Set_FeatureAngle(feature_angle);
                vec->Set_VolumeBound(volume_bound);
                vec->Reduce(red_tri, red_pnt);
                DO_return = vec->createcoDistributedObjects(red_tri, red_pnt, Mesh_out_name, Data_out_name, Normal_out_name);
                delete vec;
            }
            break;
        case 1:
            if (strcmp(dtype, "USTSDT") == 0)
            {
                svr = new ScalarDataVertexRemoval(n_pnt, n_vert, n_poly, gtype, pl_in, vl_in, x_in, y_in, z_in, s_in, nu_in, nv_in, nw_in);
                svr->Set_Percent(percent);
                svr->Set_FeatureAngle(feature_angle);
                svr->Set_VolumeBound(volume_bound);
                svr->Reduce(red_tri, red_pnt);
                DO_return = svr->createcoDistributedObjects(red_tri, red_pnt, Mesh_out_name, Data_out_name, Normal_out_name);
                delete svr;
            }
            else
            {
                vvr = new VectorDataVertexRemoval(n_pnt, n_vert, n_poly, gtype, pl_in, vl_in, x_in, y_in, z_in, u_in, v_in, w_in, nu_in, nv_in, nw_in);
                vvr->Set_Percent(percent);
                vvr->Set_FeatureAngle(feature_angle);
                vvr->Set_VolumeBound(volume_bound);
                vvr->Reduce(red_tri, red_pnt);
                DO_return = vvr->createcoDistributedObjects(red_tri, red_pnt, Mesh_out_name, Data_out_name, Normal_out_name);
                delete vvr;
            }
            break;
        default:
            Covise::sendError("ERROR: No decimation strategy indicated!");
            return (NULL);
        };
    }
    return (DO_return);
}

void SimplifySurface::copyAttributesToOutObj(coInputPort **input_ports,
                                             coOutputPort **output_ports, int n)
{
    int i, j;
    coDistributedObject *in_obj, *out_obj;
    int num_attr;
    const char **attr_n, **attr_v;

    if (n >= 3)
        j = 0;
    else
        j = n;
    if (input_ports[j] && output_ports[n])
    {
        in_obj = input_ports[j]->getCurrentObject();
        out_obj = output_ports[n]->getCurrentObject();

        if (in_obj != NULL && out_obj != NULL)
        {
            if (in_obj->getAttribute("Probe2D") == NULL)
            {
                copyAttributes(out_obj, in_obj);
            }
            else // update Probe2D attribute
            {
                num_attr = in_obj->getAllAttributes(&attr_n, &attr_v);
                for (i = 0; i < num_attr; i++)
                {
                    if (strcmp(attr_n[i], "Probe2D") != 0)
                    {
                        out_obj->addAttribute(attr_n[i], attr_v[i]);
                    }
                }
            }
            out_obj->addAttribute("Probe2D", output_ports[1]->getObjName());
        }
    }
}

MODULE_MAIN(Obsolete, SimplifySurface)
