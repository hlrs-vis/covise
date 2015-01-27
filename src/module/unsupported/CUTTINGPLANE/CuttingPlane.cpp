/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  COVISE ColorMap application module                       **
 **                                                                        **
 **                                                                        **
 **                             (C) 1994                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  R.Lang, D.Rantzau                                             **
 **                                                                        **
 **                                                                        **
 ** Date:  18.05.94  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "CuttingPlane.h"

void main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();
}

//  Local data
int npoint, plane, dim;
int i_fro_min, i_fro_max; //  Boundaries (frontier)
int j_fro_min, j_fro_max; //  Boundaries (frontier)
int k_fro_min, k_fro_max; //  Boundaries (frontier)
int i_index;
int j_index;
int k_index, isize, jsize, ksize;
int i_dim, j_dim, k_dim; //  Dim. of input object
float xpos, ypos, zpos;
float xmin, xmax, ymin, ymax, zmin, zmax;
float *x_in, *x_out;
float *y_in, *y_out;
float *z_in, *z_out;
float *s_in, *s_out;
float *u_in, *u_out;
float *v_in, *v_out;
float *w_in, *w_out;
char *dtype, *gtype;
char *GridIn, *GridOut, *DataIn, *DataOut;
char *COLOR = "COLOR";
char *color = "pink";

//  Shared memory data
coDoFloat *s_data_in;
coDoFloat *s_data_out;
coDoVec3 *v_data_in;
coDoVec3 *v_data_out;
coDoUniformGrid *u_grid_in;
coDoUniformGrid *u_grid_out;
coDoRectilinearGrid *r_grid_in;
coDoRectilinearGrid *r_grid_out;
coDoStructuredGrid *s_grid_in;
coDoStructuredGrid *s_grid_out;

//
// static stub callback functions calling the real class
// member functions
//

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

void Application::paramCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->param(callbackData);
}

//
//
//..........................................................................
//
//

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
// Computation routine (called when PARAM message arrrives)
//======================================================================
void Application::param(void *)
{
}

//======================================================================
// Computation routine (called when START message arrrives)
//======================================================================
void Application::compute(void *)
{

    coDistributedObject *mesh;
    coDistributedObject *data;
    char *gname;
    char *dname;

    //	get parameter
    Covise::get_slider_param("i_index", &i_fro_min, &i_fro_max, &i_index);
    Covise::get_slider_param("j_index", &j_fro_min, &j_fro_max, &j_index);
    Covise::get_slider_param("k_index", &k_fro_min, &k_fro_max, &k_index);
    Covise::get_choice_param("plane", &plane);

    //	get input data object names
    gname = Covise::get_object_name("meshIn");
    if (gname == 0L)
    {
        Covise::sendError("ERROR: Object name not correct for 'Mesh'");
        return;
    }
    //	retrieve object from shared memeory
    mesh = new coDistributedObject(gname);

    //	get input data object names
    dname = Covise::get_object_name("dataIn");
    if (dname == 0L)
    {
        Covise::sendError("ERROR: Object name not correct for 'dataIn'");
        return;
    }
    //	retrieve object from shared memeory
    data = new coDistributedObject(dname);

    //	get output grid object names
    gname = Covise::get_object_name("meshOut");

    //	get output data object names
    dname = Covise::get_object_name("dataOut");

    handle_objects(mesh->createUnknown(), data->createUnknown(), gname, dname);
}

void Application::handle_objects(coDistributedObject *mesh, coDistributedObject *data, char *Outgridname, char *Outdataname, coDistributedObject **mesh_set_out, coDistributedObject **data_set_out)
{
    coDoSet *D_set;
    coDoSet *G_set;
    coDistributedObject **grid_set_objs;
    coDistributedObject **data_set_objs;
    int i, set_num_elem;
    coDistributedObject **mesh_objs;
    coDistributedObject **data_objs;
    coDistributedObject *data_out;
    coDistributedObject *mesh_out;
    char buf[500];
    char buf2[500];
    char *dataType;

    if (mesh != 0L)
    {
        gtype = dataType = mesh->get_type();

        if (strcmp(gtype, "STRGRD") == 0)
        {
            s_grid_in = (coDoStructuredGrid *)mesh;
            s_grid_in->getGridSize(&i_dim, &j_dim, &k_dim);
            s_grid_in->getAddresses(&x_in, &y_in, &z_in);
        }

        else if (strcmp(gtype, "RCTGRD") == 0)
        {
            r_grid_in = (coDoRectilinearGrid *)mesh;
            r_grid_in->getGridSize(&i_dim, &j_dim, &k_dim);
            r_grid_in->getAddresses(&x_in, &y_in, &z_in);
        }

        else if (strcmp(gtype, "UNIGRD") == 0)
        {
            u_grid_in = (coDoUniformGrid *)mesh;
            u_grid_in->getGridSize(&i_dim, &j_dim, &k_dim);
        }
        dtype = data->get_type();
        if (strcmp(dtype, "STRSDT") == 0)
        {
            s_data_in = (coDoFloat *)data;
            s_data_in->getGridSize(&isize, &jsize, &ksize);
            s_data_in->getAddress(&s_in);
        }

        else if (strcmp(dtype, "STRVDT") == 0)
        {
            v_data_in = (coDoVec3 *)data;
            v_data_in->getGridSize(&isize, &jsize, &ksize);
            v_data_in->getAddresses(&u_in, &v_in, &w_in);
        }
        else if (strcmp(dataType, "SETELE") == 0)
        {
            mesh_objs = ((coDoSet *)mesh)->getAllElements(&set_num_elem);
            data_objs = ((coDoSet *)data)->getAllElements(&set_num_elem);
            grid_set_objs = new coDistributedObject *[set_num_elem];
            data_set_objs = new coDistributedObject *[set_num_elem];
            grid_set_objs[0] = NULL;
            data_set_objs[0] = NULL;
            for (i = 0; i < set_num_elem; i++)
            {
                sprintf(buf, "%s_%d", Outgridname, i);
                sprintf(buf2, "%s_%d", Outdataname, i);
                handle_objects(mesh_objs[i], data_objs[i], buf, buf2, grid_set_objs, data_set_objs);
            }
            G_set = new coDoSet(Outgridname, grid_set_objs);
            D_set = new coDoSet(Outdataname, data_set_objs);
            if (mesh->get_attribute("TIMESTEP"))
            {
                G_set->addAttribute("TIMESTEP", "1 16");
            }
            if (data->get_attribute("TIMESTEP"))
            {
                D_set->addAttribute("TIMESTEP", "1 16");
            }

            if (mesh_set_out)
            {
                for (i = 0; mesh_set_out[i]; i++)
                    ;

                mesh_set_out[i] = G_set;
                mesh_set_out[i + 1] = NULL;
            }
            else
                delete G_set;
            if (data_set_out)
            {
                for (i = 0; data_set_out[i]; i++)
                    ;

                data_set_out[i] = D_set;
                data_set_out[i + 1] = NULL;
            }
            else
                delete D_set;

            delete ((coDoSet *)mesh);
            delete ((coDoSet *)data);
            for (i = 0; data_set_objs[i]; i++)
                delete data_set_objs[i];
            delete[] data_set_objs;
            for (i = 0; grid_set_objs[i]; i++)
                delete grid_set_objs[i];
            delete[] grid_set_objs;
            return;
        }

        else
        {
            Covise::sendError("ERROR: Data object 'Data' has wrong data type");
            return;
        }
    }
    else
    {
#ifndef TOLERANT
        Covise::sendError("ERROR: Data object 'Data' can't be accessed in shared memory");
#endif
        return;
    }
    // check slider parameter and update if necessary
    if (i_index > i_dim)
    {
        i_index = i_dim / 2;
        Covise::sendWarning("WARNING: i-index out of range,  set new one");
        Covise::update_slider_param("i_index", i_fro_min, i_fro_max, i_index);
    }
    if (i_fro_max > i_dim) // set i max to i dimension
    {
        i_fro_max = i_dim;
        Covise::sendWarning("WARNING: i-max out of range,  set new one");
        Covise::update_slider_param("i_index", i_fro_min, i_fro_max, i_index);
    }
    if (i_fro_min < 1) // set i min to 1
    {
        i_fro_min = 1;
        Covise::sendWarning("WARNING: i-min out of range,  set new one");
        Covise::update_slider_param("i_index", i_fro_min, i_fro_max, i_index);
    }
    if (i_fro_max < i_fro_min)
    {
        i_fro_min = 1;
        i_fro_max = i_dim;
        Covise::sendWarning("WARNING: i-max > i_min,  set new one");
        Covise::update_slider_param("i_index", i_fro_min, i_fro_max, i_index);
    }

    if (j_index > j_dim)
    {
        j_index = j_dim / 2;
        Covise::sendWarning("WARNING: j-index out of range,  set new one");
        Covise::update_slider_param("j_index", j_fro_min, j_fro_max, j_index);
    }
    if (j_fro_max > j_dim) // set j max to j dimension
    {
        j_fro_max = j_dim;
        Covise::sendWarning("WARNING: j-max out of range,  set new one");
        Covise::update_slider_param("j_index", j_fro_min, j_fro_max, j_index);
    }
    if (j_fro_min < 1) // set j min to 1
    {
        j_fro_min = 1;
        Covise::sendWarning("WARNING: j-min out of range,  set new one");
        Covise::update_slider_param("j_index", j_fro_min, j_fro_max, j_index);
    }
    if (j_fro_max < j_fro_min)
    {
        j_fro_min = 1;
        j_fro_max = j_dim;
        Covise::sendWarning("WARNING: j-max > j_min,  set new one");
        Covise::update_slider_param("j_index", j_fro_min, j_fro_max, j_index);
    }

    if (k_index > k_dim)
    {
        k_index = k_dim / 2;
        Covise::sendWarning("WARNING: k-index out of range,  set new one");
        Covise::update_slider_param("k_index", k_fro_min, k_fro_max, k_index);
    }
    if (k_fro_max > k_dim) // set k max to k dimension
    {
        k_fro_max = k_dim;
        Covise::sendWarning("WARNING: k-max out of range,  set new one");
        Covise::update_slider_param("k_index", k_fro_min, k_fro_max, k_index);
    }
    if (k_fro_min < 1) // set k min to 1
    {
        k_fro_min = 1;
        Covise::sendWarning("WARNING: k-min out of range,  set new one");
        Covise::update_slider_param("k_index", k_fro_min, k_fro_max, k_index);
    }
    if (k_fro_max < k_fro_min)
    {
        k_fro_min = 1;
        k_fro_max = k_dim;
        Covise::sendWarning("WARNING: k-max > k_min,  set new one");
        Covise::update_slider_param("k_index", k_fro_min, k_fro_max, k_index);
    }

    //
    //      generate the output data objects
    //
    GridOut = Outgridname;
    DataOut = Outdataname;
    if (strcmp(gtype, "STRGRD") == 0)
    {
        Application::create_strgrid_plane();
        mesh_out = s_grid_out;
    }

    else if (strcmp(gtype, "RCTGRD") == 0)
    {
        Application::create_rectgrid_plane();
        mesh_out = r_grid_out;
    }

    else
    {
        Application::create_unigrid_plane();
        mesh_out = u_grid_out;
    }

    if (DataOut != NULL)
    {
        if (strcmp(dtype, "STRSDT") == 0)
        {
            Application::create_scalar_plane();
            data_out = s_data_out;
        }

        else
        {
            Application::create_vector_plane();
            data_out = v_data_out;
        }
    }

    //
    //      add objects to set
    //
    if (mesh_set_out)
    {
        for (i = 0; mesh_set_out[i]; i++)
            ;

        mesh_set_out[i] = mesh_out;
        mesh_set_out[i + 1] = NULL;
    }
    else
        delete mesh_out;
    if (data_set_out)
    {
        for (i = 0; data_set_out[i]; i++)
            ;

        data_set_out[i] = data_out;
        data_set_out[i + 1] = NULL;
    }
    else
        delete data_out;
}

//======================================================================
// create the cutting planes
//======================================================================
void Application::create_strgrid_plane()
{
    int i, j, k;

    if (plane == 1)
    {
        s_grid_out = new coDoStructuredGrid(GridOut, 1, j_dim, k_dim);
        if (s_grid_out->objectOk())
        {
            s_grid_out->getAddresses(&x_out, &y_out, &z_out);
            s_grid_out->addAttribute(COLOR, color);

            for (j = 0; j < j_dim; j++)
            {
                for (k = 0; k < k_dim; k++)
                {
                    *(x_out + j * k_dim + k) = *(x_in + (i_index - 1) * j_dim * k_dim + j * k_dim + k);
                    *(y_out + j * k_dim + k) = *(y_in + (i_index - 1) * j_dim * k_dim + j * k_dim + k);
                    *(z_out + j * k_dim + k) = *(z_in + (i_index - 1) * j_dim * k_dim + j * k_dim + k);
                }
            }
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'meshOut' failed");
            return;
        }
    }

    else if (plane == 2)
    {
        s_grid_out = new coDoStructuredGrid(GridOut, i_dim, 1, k_dim);
        if (s_grid_out->objectOk())
        {
            s_grid_out->getAddresses(&x_out, &y_out, &z_out);
            s_grid_out->addAttribute(COLOR, color);

            for (i = 0; i < i_dim; i++)
            {
                for (k = 0; k < k_dim; k++)
                {
                    *(x_out + i * k_dim + k) = *(x_in + i * j_dim * k_dim + (j_index - 1) * k_dim + k);
                    *(y_out + i * k_dim + k) = *(y_in + i * j_dim * k_dim + (j_index - 1) * k_dim + k);
                    *(z_out + i * k_dim + k) = *(z_in + i * j_dim * k_dim + (j_index - 1) * k_dim + k);
                }
            }
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'meshOut' failed");
            return;
        }
    }

    else if (plane == 3)
    {
        s_grid_out = new coDoStructuredGrid(GridOut, i_dim, j_dim, 1);
        if (s_grid_out->objectOk())
        {
            s_grid_out->getAddresses(&x_out, &y_out, &z_out);
            s_grid_out->addAttribute(COLOR, color);

            for (i = 0; i < i_dim; i++)
            {
                for (j = 0; j < j_dim; j++)
                {
                    *(x_out + i * j_dim + j) = *(x_in + i * j_dim * k_dim + j * k_dim + (k_index - 1));
                    *(y_out + i * j_dim + j) = *(y_in + i * j_dim * k_dim + j * k_dim + (k_index - 1));
                    *(z_out + i * j_dim + j) = *(z_in + i * j_dim * k_dim + j * k_dim + (k_index - 1));
                }
            }
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'meshOut' failed");
            return;
        }
    }
}

//======================================================================
// create the cutting planes
//======================================================================
void Application::create_rectgrid_plane()
{
    int i, j, k;

    if (plane == 1)
    {
        r_grid_out = new coDoRectilinearGrid(GridOut, 1, j_dim, k_dim);
        if (r_grid_out->objectOk())
        {
            r_grid_out->getAddresses(&x_out, &y_out, &z_out);
            r_grid_out->addAttribute(COLOR, color);

            x_out[0] = x_in[i_index - 1];
            for (j = 0; j < j_dim; j++)
                y_out[j] = y_in[j];
            for (k = 0; k < k_dim; k++)
                z_out[k] = z_in[k];
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'meshOut' failed");
            return;
        }
    }

    else if (plane == 2)
    {
        r_grid_out = new coDoRectilinearGrid(GridOut, i_dim, 1, k_dim);
        if (r_grid_out->objectOk())
        {
            r_grid_out->getAddresses(&x_out, &y_out, &z_out);
            r_grid_out->addAttribute(COLOR, color);

            y_out[0] = y_in[j_index - 1];
            for (i = 0; i < i_dim; i++)
                x_out[i] = x_in[i];
            for (k = 0; k < k_dim; k++)
                z_out[k] = z_in[k];
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'meshOut' failed");
            return;
        }
    }

    else if (plane == 3)
    {
        r_grid_out = new coDoRectilinearGrid(GridOut, i_dim, j_dim, 1);
        if (r_grid_out->objectOk())
        {
            r_grid_out->getAddresses(&x_out, &y_out, &z_out);
            r_grid_out->addAttribute(COLOR, color);

            z_out[0] = z_in[k_index - 1];
            for (i = 0; i < i_dim; i++)
                x_out[i] = x_in[i];
            for (j = 0; j < j_dim; j++)
                y_out[j] = y_in[j];
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'meshOut' failed");
            return;
        }
    }
}

//======================================================================
// create the cutting planes
//======================================================================
void Application::create_unigrid_plane()
{

    u_grid_in->get_point_coordinates(0, &xmin, 0, &ymin, 0, &zmin);
    u_grid_in->get_point_coordinates(i_dim - 1, &xmax, j_dim - 1, &ymax, k_dim - 1, &zmax);
    u_grid_in->get_point_coordinates(i_index - 1, &xpos, j_index - 1, &ypos, k_index - 1, &zpos);

    if (plane == 1)
    {
        u_grid_out = new coDoUniformGrid(GridOut, 1, j_dim, k_dim,
                                         xpos, xpos, ymin, ymax, zmin, zmax);
        u_grid_out->addAttribute(COLOR, color);
        if (!u_grid_out->objectOk())
        {
            Covise::sendError("ERROR: creation of data object 'meshOut' failed");
            return;
        }
    }

    else if (plane == 2)
    {
        u_grid_out = new coDoUniformGrid(GridOut, i_dim, 1, k_dim,
                                         xmin, xmax, ypos, ypos, zmin, zmax);
        u_grid_out->addAttribute(COLOR, color);
        if (!u_grid_out->objectOk())
        {
            Covise::sendError("ERROR: creation of data object 'meshOut' failed");
            return;
        }
    }

    else if (plane == 3)
    {
        u_grid_out = new coDoUniformGrid(GridOut, i_dim, j_dim, 1,
                                         xmin, xmax, ymin, ymax, zpos, zpos);
        u_grid_out->addAttribute(COLOR, color);
        if (!u_grid_out->objectOk())
        {
            Covise::sendError("ERROR: creation of data object 'meshOut' failed");
            return;
        }
    }
}

//======================================================================
// create the cutting planes
//======================================================================
void Application::create_scalar_plane()
{
    int i, j, k;

    if (plane == 1)
    {
        s_data_out = new coDoFloat(DataOut, 1, j_dim, k_dim);
        if (s_data_out->objectOk())
        {
            s_data_out->getAddress(&s_out);

            for (j = 0; j < j_dim; j++)
                for (k = 0; k < k_dim; k++)
                    *(s_out + j * k_dim + k) = *(s_in + (i_index - 1) * j_dim * k_dim + j * k_dim + k);
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
    }

    else if (plane == 2)
    {
        s_data_out = new coDoFloat(DataOut, i_dim, 1, k_dim);
        if (s_data_out->objectOk())
        {
            s_data_out->getAddress(&s_out);

            for (i = 0; i < i_dim; i++)
                for (k = 0; k < k_dim; k++)
                    *(s_out + i * k_dim + k) = *(s_in + i * j_dim * k_dim + (j_index - 1) * k_dim + k);
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
    }

    else if (plane == 3)
    {
        s_data_out = new coDoFloat(DataOut, i_dim, j_dim, 1);
        if (s_data_out->objectOk())
        {
            s_data_out->getAddress(&s_out);

            for (i = 0; i < i_dim; i++)
                for (j = 0; j < j_dim; j++)
                    *(s_out + i * j_dim + j) = *(s_in + i * j_dim * k_dim + j * k_dim + (k_index - 1));
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
    }
}

//======================================================================
// create the cutting planes
//======================================================================
void Application::create_vector_plane()
{
    int i, j, k;

    if (plane == 1)
    {
        v_data_out = new coDoVec3(DataOut, 1, j_dim, k_dim);
        if (v_data_out->objectOk())
        {
            v_data_out->getAddresses(&u_out, &v_out, &w_out);

            for (j = 0; j < j_dim; j++)
                for (k = 0; k < k_dim; k++)
                {
                    *(u_out + j * k_dim + k) = *(u_in + (i_index - 1) * j_dim * k_dim + j * k_dim + k);
                    *(v_out + j * k_dim + k) = *(v_in + (i_index - 1) * j_dim * k_dim + j * k_dim + k);
                    *(w_out + j * k_dim + k) = *(w_in + (i_index - 1) * j_dim * k_dim + j * k_dim + k);
                }
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
    }

    else if (plane == 2)
    {
        v_data_out = new coDoVec3(DataOut, i_dim, 1, k_dim);
        if (v_data_out->objectOk())
        {
            v_data_out->getAddresses(&u_out, &v_out, &w_out);

            for (i = 0; i < i_dim; i++)
                for (k = 0; k < k_dim; k++)
                {
                    *(u_out + i * k_dim + k) = *(u_in + i * j_dim * k_dim + (j_index - 1) * k_dim + k);
                    *(v_out + i * k_dim + k) = *(v_in + i * j_dim * k_dim + (j_index - 1) * k_dim + k);
                    *(w_out + i * k_dim + k) = *(w_in + i * j_dim * k_dim + (j_index - 1) * k_dim + k);
                }
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
    }

    else if (plane == 3)
    {
        v_data_out = new coDoVec3(DataOut, i_dim, j_dim, 1);
        if (v_data_out->objectOk())
        {
            v_data_out->getAddresses(&u_out, &v_out, &w_out);

            for (i = 0; i < i_dim; i++)
                for (j = 0; j < j_dim; j++)
                {
                    *(u_out + i * j_dim + j) = *(u_in + i * j_dim * k_dim + j * k_dim + (k_index - 1));
                    *(v_out + i * j_dim + j) = *(v_in + i * j_dim * k_dim + j * k_dim + (k_index - 1));
                    *(w_out + i * j_dim + j) = *(w_in + i * j_dim * k_dim + j * k_dim + (k_index - 1));
                }
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
    }
}
