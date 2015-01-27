/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
**                                                           (C)1995 RUS  **
**                                                                        **
** Description:  COVISE Isosurface application module                     **
**                                                                        **
**                                                                        **
**                             (C) 1995                                   **
**                Computer Center University of Stuttgart                 **
**                            Allmandring 30                              **
**                            70550 Stuttgart                             **
**                                                                        **
**                                                                        **
** Author:  Uwe Woessner                                                  **
**                                                                        **
**                                                                        **
** Date:  23.07.96  V1.0                                                  **
\**************************************************************************/

#include "ApplInterface.h"
#include "IsosurfaceKd.h"
#include "CuttingTables.h"
#include <math.h>

#define ADDVERTEX           \
    if (n1 < n2)            \
        add_vertex(n1, n2); \
    else                    \
        add_vertex(n2, n1);
#define ADDVERTEXX01                                                                                                                                                                                                  \
    if (n1 < n2)                                                                                                                                                                                                      \
        add_vertex(n1, n2, ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes], ii + x_add[*(polygon_nodes + 1)], jj + y_add[*(polygon_nodes + 1)], kk + z_add[*(polygon_nodes + 1)]); \
    else                                                                                                                                                                                                              \
        add_vertex(n2, n1, ii + x_add[*(polygon_nodes + 1)], jj + y_add[*(polygon_nodes + 1)], kk + z_add[*(polygon_nodes + 1)], ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes]);
#define ADDVERTEXX02                                                                                                                                                                                                  \
    if (n1 < n2)                                                                                                                                                                                                      \
        add_vertex(n1, n2, ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes], ii + x_add[*(polygon_nodes + 2)], jj + y_add[*(polygon_nodes + 2)], kk + z_add[*(polygon_nodes + 2)]); \
    else                                                                                                                                                                                                              \
        add_vertex(n2, n1, ii + x_add[*(polygon_nodes + 2)], jj + y_add[*(polygon_nodes + 2)], kk + z_add[*(polygon_nodes + 2)], ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes]);
#define ADDVERTEXX03                                                                                                                                                                                                  \
    if (n1 < n2)                                                                                                                                                                                                      \
        add_vertex(n1, n2, ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes], ii + x_add[*(polygon_nodes + 3)], jj + y_add[*(polygon_nodes + 3)], kk + z_add[*(polygon_nodes + 3)]); \
    else                                                                                                                                                                                                              \
        add_vertex(n2, n1, ii + x_add[*(polygon_nodes + 3)], jj + y_add[*(polygon_nodes + 3)], kk + z_add[*(polygon_nodes + 3)], ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes]);
#define ADDVERTEXX04                                                                                                                                                                                                  \
    if (n1 < n2)                                                                                                                                                                                                      \
        add_vertex(n1, n2, ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes], ii + x_add[*(polygon_nodes + 4)], jj + y_add[*(polygon_nodes + 4)], kk + z_add[*(polygon_nodes + 4)]); \
    else                                                                                                                                                                                                              \
        add_vertex(n2, n1, ii + x_add[*(polygon_nodes + 4)], jj + y_add[*(polygon_nodes + 4)], kk + z_add[*(polygon_nodes + 4)], ii + x_add[*polygon_nodes], jj + y_add[*polygon_nodes], kk + z_add[*polygon_nodes]);
#define ADDVERTEXX(a1, a2)                                                                                                                                                                                                                    \
    if (n1 < n2)                                                                                                                                                                                                                              \
        add_vertex(n1, n2, ii + x_add[*(polygon_nodes + a1)], jj + y_add[*(polygon_nodes + a1)], kk + z_add[*(polygon_nodes + a1)], ii + x_add[*(polygon_nodes + a2)], jj + y_add[*(polygon_nodes + a2)], kk + z_add[*(polygon_nodes + a2)]); \
    else                                                                                                                                                                                                                                      \
        add_vertex(n2, n1, ii + x_add[*(polygon_nodes + a2)], jj + y_add[*(polygon_nodes + a2)], kk + z_add[*(polygon_nodes + a2)], ii + x_add[*(polygon_nodes + a1)], jj + y_add[*(polygon_nodes + a1)], kk + z_add[*(polygon_nodes + a1)]);
#define ADDVERTEXXX(a1, a2)                                                                                                 \
    if (n1 < n2)                                                                                                            \
        add_vertex(n1, n2, ii + x_add[a1], jj + y_add[a1], kk + z_add[a1], ii + x_add[a2], jj + y_add[a2], kk + z_add[a2]); \
    else                                                                                                                    \
        add_vertex(n2, n1, ii + x_add[a2], jj + y_add[a2], kk + z_add[a2], ii + x_add[a1], jj + y_add[a1], kk + z_add[a1]);

int save_memory;
int *el, *cl, *tl;
int gennormals, genstrips;
int numiso, set_num_elem = 0, cur_elem, cur_line_elem;
float isovalue;
float planei, planej, planek;
float *x_in;
float *y_in;
float *z_in;
float *s_in;
float *i_in;
float *u_in;
float *v_in;
float *w_in;
float distance;
float x_min, x_max, y_min, y_max, z_min, z_max;
int x_size, y_size, z_size;
char *dtype, *gtype;
char *GridIn, *GridOut, *DataIn, *IsoDataIn,
    *DataOut, *NormalsOut, *KdIn;
char *colorn;
char *color[10] = { "yellow", "green", "blue", "red", "violet", "chocolat",
                    "linen", "pink", "crimson", "indigo" };

//  Shared memory data
coDoFloat *s_data_in = NULL;
coDoVec3 *v_data_in = NULL;
coDoFloat *i_data_in = NULL;
coDoVec3 *uv_data_in = NULL;
coDoFloat *us_data_in = NULL;
coDoFloat *ui_data_in = NULL;
coDoUnstructuredGrid *grid_in = NULL;
coDoStructuredGrid *sgrid_in = NULL;
coDoUniformGrid *ugrid_in = NULL;
coDoRectilinearGrid *rgrid_in = NULL;
coDoSet *polygons_set_out, *normals_set_out,
    *data_set_out;

void main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();
}

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
// Computation routine (called when START message arrrives)
//======================================================================
void Application::compute(void *)
{
    coDistributedObject *tmp_obj_1, *tmp_obj_2, *tmp_obj_3, *tmp_obj_4;

    //	get input data object names
    GridIn = Covise::get_object_name("meshIn");
    DataIn = Covise::get_object_name("dataIn");
    IsoDataIn = Covise::get_object_name("isoDataIn");
    KdIn = Covise::get_object_name("kdTreeIn");

    //	get output data object	names
    GridOut = Covise::get_object_name("meshOut");
    NormalsOut = Covise::get_object_name("normalsOut");
    DataOut = Covise::get_object_name("dataOut");

    //	get parameter
    Covise::get_boolean_param("gennormals", &gennormals);
    Covise::get_boolean_param("genstrips", &genstrips);
    Covise::get_boolean_param("save_memory", &save_memory);
    Covise::get_scalar_param("isovalue", &isovalue);

    //	retrieve grid object from shared memeory
    if (GridIn == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for 'meshIn'");
        return;
    }
    if (IsoDataIn == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for 'dataIn'");
        return;
    }
    set_num_elem = 0;
    s_data_in = NULL;
    v_data_in = NULL;
    i_data_in = NULL;
    us_data_in = NULL;
    uv_data_in = NULL;
    ui_data_in = NULL;
    grid_in = NULL;

    //	retrieve data object from shared memeory
    tmp_obj_1 = new coDistributedObject(GridIn);
    tmp_obj_2 = new coDistributedObject(DataIn);
    tmp_obj_3 = new coDistributedObject(IsoDataIn);
    tmp_obj_4 = new coDistributedObject(KdIn);

    HandleObjects(tmp_obj_1->createUnknown(), tmp_obj_2->createUnknown(), tmp_obj_3->createUnknown(), tmp_obj_4->createUnknown(),
                  GridOut, NormalsOut, DataOut);
}

void Application::HandleObjects(coDistributedObject *grid_object,
                                coDistributedObject *data_object,
                                coDistributedObject *i_data_object,
                                coDistributedObject *kd_data_object,
                                char *Triangle_out_name,
                                char *Normal_out_name,
                                char *Data_out_name,
                                coDoSet *Triangle_set_object,
                                coDoSet *Normal_set_object,
                                coDoSet *Data_set_object)
{
    coDoSet *grid_set_in, *data_set_in, *idata_set_in, *kd_set_in;
    coDoSet *T_set, *N_set, *D_set;
    coDistributedObject **grid_objs, **data_objs, **idata_objs, **kd_objs;
    int set_num_elem = 0, i;
    int numelem, numconn, numcoord, data_anz, DataType;
    int sx, sy, sz;
    Plane *plane;
    STR_Plane *splane;
    UNI_Plane *uplane;
    RECT_Plane *rplane;
    char buf1[500];
    char buf2[500];
    char buf3[500];

    s_data_in = NULL;
    v_data_in = NULL;
    i_data_in = NULL;
    us_data_in = NULL;
    uv_data_in = NULL;
    ui_data_in = NULL;
    grid_set_in = NULL;
    data_set_in = NULL;
    idata_set_in = NULL;
    if (grid_object != 0L)
    {
        gtype = grid_object->getType();
        if (strcmp(gtype, "UNSGRD") == 0)
        {
            grid_in = (coDoUnstructuredGrid *)grid_object;
            grid_in->getGridSize(&numelem, &numconn, &numcoord);
            grid_in->get_adresses(&el, &cl, &x_in, &y_in, &z_in);
            grid_in->getTypeList(&tl);
            if ((colorn = grid_in->getAttribute("COLOR")) == NULL)
            {
                colorn = new char[20];
                strcpy(colorn, "white");
            }
        }
        else if (strcmp(gtype, "UNIGRD") == 0)
        {
            ugrid_in = (coDoUniformGrid *)grid_object;
            ugrid_in->getGridSize(&x_size, &y_size, &z_size);
            ugrid_in->getMinMax(&x_min, &x_max, &y_min, &y_max,
                                &z_min, &z_max);
            numcoord = x_size * y_size * z_size;
            numelem = ((x_size - 1) * (y_size - 1) * (z_size - 1));
            if ((colorn = ugrid_in->getAttribute("COLOR")) == NULL)
            {
                colorn = new char[20];
                strcpy(colorn, "white");
            }
        }
        else if (strcmp(gtype, "RCTGRD") == 0)
        {
            rgrid_in = (coDoRectilinearGrid *)grid_object;
            rgrid_in->getGridSize(&x_size, &y_size, &z_size);
            rgrid_in->get_adresses(&x_in, &y_in, &z_in);
            numcoord = x_size * y_size * z_size;
            numelem = ((x_size - 1) * (y_size - 1) * (z_size - 1));
            if ((colorn = rgrid_in->getAttribute("COLOR")) == NULL)
            {
                colorn = new char[20];
                strcpy(colorn, "white");
            }
        }
        else if (strcmp(gtype, "STRGRD") == 0)
        {
            sgrid_in = (coDoStructuredGrid *)grid_object;
            sgrid_in->getGridSize(&x_size, &y_size, &z_size);
            sgrid_in->get_adresses(&x_in, &y_in, &z_in);
            numcoord = x_size * y_size * z_size;
            numelem = ((x_size - 1) * (y_size - 1) * (z_size - 1));
            if ((colorn = sgrid_in->getAttribute("COLOR")) == NULL)
            {
                colorn = new char[20];
                strcpy(colorn, "white");
            }
        }
        else if (strcmp(gtype, "SETELE") == 0)
        {
            grid_set_in = (coDoSet *)grid_object;
            grid_objs = grid_set_in->getAllElements(&set_num_elem);
        }
        else
        {
            Covise::sendError("ERROR: Data object 'meshIn' has wrong data type");
            return;
        }
    }
    else
    {
        Covise::sendError("ERROR: Data object 'meshIn' can't be accessed in shared memory");
        return;
    }
    if (numelem == 0 || numconn == 0 || numcoord == 0)
    {
        Covise::sendWarning("WARNING: Data object 'meshIn' isempty");
    }
    DataType = 0;
    if (data_object != 0L)
    {
        dtype = data_object->getType();
        if (strcmp(dtype, "STRSDT") == 0)
        {
            s_data_in = (coDoFloat *)data_object;
            s_data_in->getGridSize(&sx, &sy, &sz);
            data_anz = sx * sy * sz;
            s_data_in->get_adress(&s_in);
            DataType = 1;
        }
        else if (strcmp(dtype, "USTSDT") == 0)
        {
            us_data_in = (coDoFloat *)data_object;
            data_anz = us_data_in->getNumPoints();
            us_data_in->get_adress(&s_in);
            DataType = 1;
        }
        else if (strcmp(dtype, "STRVDT") == 0)
        {
            v_data_in = (coDoVec3 *)data_object;
            v_data_in->getGridSize(&sx, &sy, &sz);
            data_anz = sx * sy * sz;
            v_data_in->get_adresses(&u_in, &v_in, &w_in);
            DataType = 2;
        }
        else if (strcmp(dtype, "USTVDT") == 0)
        {
            uv_data_in = (coDoVec3 *)data_object;
            data_anz = uv_data_in->getNumPoints();
            uv_data_in->get_adresses(&u_in, &v_in, &w_in);
            DataType = 2;
        }
        else if (strcmp(gtype, "SETELE") == 0)
        {
            data_set_in = (coDoSet *)data_object;
            data_objs = data_set_in->getAllElements(&set_num_elem);
        }
        else
        {
            Covise::sendError("ERROR: Data object 'dataIn' has wrong data type");
            return;
        }
    }
    if (i_data_object != 0L)
    {
        dtype = i_data_object->getType();
        if (strcmp(dtype, "STRSDT") == 0)
        {
            i_data_in = (coDoFloat *)i_data_object;
            i_data_in->getGridSize(&sx, &sy, &sz);
            data_anz = sx * sy * sz;
            i_data_in->get_adress(&i_in);
        }
        else if (strcmp(dtype, "USTSDT") == 0)
        {
            ui_data_in = (coDoFloat *)i_data_object;
            data_anz = ui_data_in->getNumPoints();
            ui_data_in->get_adress(&i_in);
        }
        else if (strcmp(gtype, "SETELE") == 0)
        {
            idata_set_in = (coDoSet *)i_data_object;
            idata_objs = idata_set_in->getAllElements(&set_num_elem);
        }
        else
        {
            Covise::sendError("ERROR: Data object 'dataIn' has wrong data type");
            return;
        }
    }
    else
    {
        Covise::sendError("ERROR: Data object 'isoDataIn' can't be accessed in shared memory");
        return;
    }
    kdtree = NULL;
    kd_objs = NULL;
    if (kd_data_object != 0L)
    {
        dtype = kd_data_object->getType();
        if (strcmp(dtype, "KDTREE") == 0)
        {
            kdtree = (DO_KdTree *)kd_data_object;
            kdtree->set_data(cl, el, i_in);
        }
        else if (strcmp(gtype, "SETELE") == 0)
        {
            kd_set_in = (coDoSet *)kd_data_object;
            kd_objs = kd_set_in->getAllElements(&set_num_elem);
        }
        else
        {
            Covise::sendError("ERROR: Data object 'KdTreeIn' has wrong data type");
            return;
        }
    }

    if (set_num_elem)
    {
        T_set = new coDoSet(Triangle_out_name, SET_CREATE);
        N_set = new coDoSet(Normal_out_name, SET_CREATE);
        D_set = NULL;
        if (data_object)
            D_set = new coDoSet(Data_out_name, SET_CREATE);
        for (i = 0; i < set_num_elem; i++)
        {
            sprintf(buf1, "%s_%d", Triangle_out_name, i);
            sprintf(buf2, "%s_%d", Normal_out_name, i);
            sprintf(buf3, "%s_%d", Data_out_name, i);
            if (D_set)
            {
                if (kd_objs)
                    HandleObjects(grid_objs[i], data_objs[i], idata_objs[i], kd_objs[i],
                                  buf1, buf2, buf3, T_set, N_set, D_set);
                else
                    HandleObjects(grid_objs[i], data_objs[i], idata_objs[i], NULL,
                                  buf1, buf2, buf3, T_set, N_set, D_set);
            }
            else
            {
                if (kd_objs)
                    HandleObjects(grid_objs[i], NULL, idata_objs[i], kd_objs[i],
                                  buf1, buf2, buf3, T_set, N_set, D_set);
                else
                    HandleObjects(grid_objs[i], NULL, idata_objs[i], NULL,
                                  buf1, buf2, buf3, T_set, N_set, D_set);
            }
        }
        if (Triangle_set_object)
            Triangle_set_object->addElement(T_set);
        if (Normal_set_object)
            Normal_set_object->addElement(N_set);
        if (Data_set_object)
            Data_set_object->addElement(D_set);

        if (grid_set_in->getAttribute("TIMESTEP"))
        {
            T_set->addAttribute("TIMESTEP", "1 4");
        }
        delete T_set;
        delete D_set;
        delete N_set;
        if (idata_set_in)
            delete idata_set_in;
        if (data_set_in)
            delete data_set_in;
        if (grid_set_in)
            delete grid_set_in;
    }
    else
    {

        if (data_anz == 0)
        {
            Covise::sendWarning("WARNING: Data object 'isoDataIn' is empty");
        }

        // check dimensions
        if (data_anz != numcoord)
        {
            Covise::sendError("ERROR: Dataobject's dimension doesn't match Grid ones");
            return;
        }

        if (strcmp(gtype, "UNSGRD") == 0)
        {
            plane = new Plane(numelem, numcoord, DataType, kdtree);
            plane->createPlane();
            plane->createcoDistributedObjects(Data_out_name, Normal_out_name, Triangle_out_name, Data_set_object, Normal_set_object, Triangle_set_object);
            delete plane;
        }
        if (strcmp(gtype, "UNIGRD") == 0)
        {
            uplane = new UNI_Plane(numelem, numcoord, DataType, kdtree);
            uplane->createPlane();
            uplane->createcoDistributedObjects(Data_out_name, Normal_out_name, Triangle_out_name, Data_set_object, Normal_set_object, Triangle_set_object);
            delete uplane;
        }
        if (strcmp(gtype, "RCTGRD") == 0)
        {
            rplane = new RECT_Plane(numelem, numcoord, DataType, kdtree);
            rplane->createPlane();
            rplane->createcoDistributedObjects(Data_out_name, Normal_out_name, Triangle_out_name, Data_set_object, Normal_set_object, Triangle_set_object);
            delete rplane;
        }
        if (strcmp(gtype, "STRGRD") == 0)
        {
            splane = new STR_Plane(numelem, numcoord, DataType, kdtree);
            splane->createPlane();
            splane->createcoDistributedObjects(Data_out_name, Normal_out_name, Triangle_out_name, Data_set_object, Normal_set_object, Triangle_set_object);
            delete splane;
        }

        if (s_data_in)
            delete s_data_in;
        if (v_data_in)
            delete v_data_in;
        if (i_data_in)
            delete i_data_in;
        if (us_data_in)
            delete us_data_in;
        if (uv_data_in)
            delete uv_data_in;
        if (ui_data_in)
            delete ui_data_in;
        if (grid_in)
            delete grid_in;
    }
}

Plane::Plane(int n_elem, int n_nodes, int Type, DO_KdTree *kd)
{
    NodeInfo *node;
    int i;
    Datatype = Type;
    num_nodes = n_nodes;
    num_elem = n_elem;
    kdtree = kd;
    node_table = NULL;
    if (!save_memory)
    {
        node_table = (NodeInfo *)malloc(n_nodes * sizeof(NodeInfo));
        node = node_table;
        if (node)
            for (i = 0; i < n_nodes; i++)
            {
                node->targets[0] = 0;
                // Calculate the distance of each node
                // to the Isovalue
                //node->dist = (i_in[i] - isovalue);
                //node->side = (node->dist >= 0 ? 1 : 0);
                node++;
            }
    }
    num_triangles = num_vertices = num_coords = 0;
    vertice_list = new int[n_elem * 12];
    vertex = vertice_list;
    coords_x = new float[n_nodes * 3];
    coords_y = new float[n_nodes * 3];
    coords_z = new float[n_nodes * 3];
    coord_x = coords_x;
    coord_y = coords_y;
    coord_z = coords_z;
    S_Data = V_Data_U = NULL;
    if (Datatype == 1) // (Scalar Data)
    {
        S_Data_p = S_Data = new float[n_nodes * 3];
    }
    else if (Datatype == 2)
    {
        V_Data_U_p = V_Data_U = new float[n_nodes * 3];
        V_Data_V_p = V_Data_V = new float[n_nodes * 3];
        V_Data_W_p = V_Data_W = new float[n_nodes * 3];
    }
}

UNI_Plane::UNI_Plane(int n_elem, int n_nodes, int Type, DO_KdTree *kd)
{
    NodeInfo *node;
    int i;
    Datatype = Type;
    num_nodes = n_nodes;
    num_elem = n_elem;
    kdtree = kd;
    node_table = NULL;
    float xdisc, ydisc, zdisc;
    x_in = new float[x_size];
    y_in = new float[y_size];
    z_in = new float[z_size];
    xdisc = (x_max - x_min) / (x_size - 1);
    ydisc = (y_max - y_min) / (y_size - 1);
    zdisc = (z_max - z_min) / (z_size - 1);
    for (i = 0; i < x_size; i++)
        x_in[i] = x_min + xdisc * i;
    for (i = 0; i < y_size; i++)
        y_in[i] = y_min + ydisc * i;
    for (i = 0; i < z_size; i++)
        z_in[i] = z_min + zdisc * i;
    if (!save_memory)
    {
        node_table = (NodeInfo *)malloc(n_nodes * sizeof(NodeInfo));
        node = node_table;
        if (node)
            for (i = 0; i < n_nodes; i++)
            {
                node->targets[0] = 0;
                node++;
            }
        vertice_list = new int[n_elem * 12];
        coords_x = new float[n_nodes * 3];
        coords_y = new float[n_nodes * 3];
        coords_z = new float[n_nodes * 3];
    }
    else
    {
        vertice_list = new int[n_elem / 10];
        coords_x = new float[n_nodes / 10];
        coords_y = new float[n_nodes / 10];
        coords_z = new float[n_nodes / 10];
    }
    num_triangles = num_vertices = num_coords = 0;
    vertex = vertice_list;
    coord_x = coords_x;
    coord_y = coords_y;
    coord_z = coords_z;
    S_Data = V_Data_U = NULL;
    if (Datatype == 1) // (Scalar Data)
    {
        S_Data_p = S_Data = new float[n_nodes * 3];
    }
    else if (Datatype == 2)
    {
        V_Data_U_p = V_Data_U = new float[n_nodes * 3];
        V_Data_V_p = V_Data_V = new float[n_nodes * 3];
        V_Data_W_p = V_Data_W = new float[n_nodes * 3];
    }
}

RECT_Plane::RECT_Plane(int n_elem, int n_nodes, int Type, DO_KdTree *kd)
{
    NodeInfo *node;
    int i;
    Datatype = Type;
    num_nodes = n_nodes;
    num_elem = n_elem;
    kdtree = kd;
    node_table = NULL;
    if (!save_memory)
    {
        node_table = (NodeInfo *)malloc(n_nodes * sizeof(NodeInfo));
        node = node_table;
        if (node)
            for (i = 0; i < n_nodes; i++)
            {
                node->targets[0] = 0;
                node++;
            }
    }
    num_triangles = num_vertices = num_coords = 0;
    vertice_list = new int[n_elem * 12];
    vertex = vertice_list;
    coords_x = new float[n_nodes * 3];
    coords_y = new float[n_nodes * 3];
    coords_z = new float[n_nodes * 3];
    coord_x = coords_x;
    coord_y = coords_y;
    coord_z = coords_z;
    S_Data = V_Data_U = NULL;
    if (Datatype == 1) // (Scalar Data)
    {
        S_Data_p = S_Data = new float[n_nodes * 3];
    }
    else if (Datatype == 2)
    {
        V_Data_U_p = V_Data_U = new float[n_nodes * 3];
        V_Data_V_p = V_Data_V = new float[n_nodes * 3];
        V_Data_W_p = V_Data_W = new float[n_nodes * 3];
    }
}

void Plane::createPlane()
{
    int element;
    register int bitmap; // index in the MarchingCubes table
    // 1 = above; 0 = below
    register int i;
    int n;
    int *node_list;
    int *node;
    int elementtype;
    int *active_elements, num_active_elems;
    int numIntersections;
    int *polygon_nodes;
    int n1, n2, no1, no2, no3, no4, no5, no6;
    int *vertex1, *vertex2;
    cutting_info *C_Info;

// just for testing
#ifdef DEBUG
    int cases[256];
    for (i = 0; i < 256; i++)
        cases[i] = 0;
#endif
    if (kdtree)
    {
        active_elements = kdtree->search(isovalue);
        num_active_elems = kdtree->getNumActiveElements();
    }
    else
        num_active_elems = num_elem;

    if (!node_table)
    {
        /*Initialize hash table */
        hash_table_size = (num_active_elems * 3) + 3;

        hash_table = (hash_table_entry *)malloc(hash_table_size * sizeof(hash_table_entry));
        for (n = 0; n < hash_table_size; n++)
        {
            hash_table[n].key = -1;
        }
    }

    for (n = 0; n < num_active_elems; n++)
    {
        if (kdtree)
            element = active_elements[n];
        else
            element = n;
        elementtype = tl[element];
        bitmap = 0;
        i = UnstructuredGrid_Num_Nodes[elementtype];
        // number of nodes for current element
        node_list = cl + el[element];
        // pointer to nodes of current element
        node = node_list + i;
        // node = pointer to last node of current element

        while (i--)
            bitmap |= (i_in[*--node] >= isovalue ? 1 : 0) << i;
        // bitmap is now an index to the Cuttingtable
        C_Info = Cutting_Info[elementtype] + bitmap;
// just for testing
#ifdef DEBUG
        cases[bitmap]++;
#endif
        numIntersections = C_Info->nvert;
        if (numIntersections)
        {
            polygon_nodes = C_Info->node_pairs;
            switch (numIntersections)
            {
            case 1:
                num_triangles++;
                vertex1 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                break;
            case 2:
                num_triangles += 2;
                vertex1 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                vertex2 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                *vertex = *vertex1;
                vertex++;
                *vertex = *vertex2;
                vertex++;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                break;
            case 3:
                /*
 *      Something of a special case here:  If the average of the vertices
 *      is greater than the isovalue, we create two separated polygons 
 *      at the vertices.  If it is less, then we make a little valley
 *      shape.
 */
                no1 = node_list[*polygon_nodes++];
                no2 = node_list[*polygon_nodes++];
                no3 = node_list[*polygon_nodes++];
                no4 = node_list[*polygon_nodes++];
                no5 = node_list[*polygon_nodes++];
                no6 = node_list[*polygon_nodes++];
                //  if((node_table[no1].dist +
                //	node_table[no3].dist +
                //	node_table[no4].dist +
                //	node_table[no5].dist ) < 0)
                /*   {
			num_triangles +=2;
			n1 = no1;
			n2 = no2;
			ADDVERTEX;
			n2 = no3;
			ADDVERTEX;
			n2 = no4;
			ADDVERTEX;
			n1 = no5;
			n2 = no4;
			ADDVERTEX;
			n2 = no3;
			ADDVERTEX;
			n2 = no6;
			ADDVERTEX;
		    }*/
                //   else
                //    {
                num_triangles += 4;
                n1 = no1;
                n2 = no2;
                vertex1 = vertex;
                ADDVERTEX;
                n2 = no3;
                ADDVERTEX;
                n1 = no5;
                n2 = no3;
                vertex2 = vertex;
                ADDVERTEX;
                *vertex = *vertex1;
                vertex++;
                *vertex = *vertex2;
                vertex++;
                n2 = no6;
                ADDVERTEX;
                n1 = no1;
                n2 = no4;
                vertex1 = vertex;
                ADDVERTEX;
                n2 = no2;
                ADDVERTEX;
                n1 = no5;
                n2 = no6;
                vertex2 = vertex;
                ADDVERTEX;
                *vertex = *vertex1;
                vertex++;
                *vertex = *vertex2;
                vertex++;
                n2 = no4;
                ADDVERTEX;
                //   }
                break;
            case 4:
                /*
 *      Something of a special case here:  If the average of the vertices
 *      is smaller than the isovalue, we create two separated polygons 
 *      at the vertices.  If it is less, then we make a little valley
 *      shape.
 */
                no1 = node_list[*polygon_nodes++];
                no2 = node_list[*polygon_nodes++];
                no3 = node_list[*polygon_nodes++];
                no4 = node_list[*polygon_nodes++];
                no5 = node_list[*polygon_nodes++];
                no6 = node_list[*polygon_nodes++];
                //if((node_table[no1].dist +
                //node_table[no3].dist +
                //	node_table[no4].dist +
                //	node_table[no5].dist ) > 0)
                /* {
			num_triangles +=2;
			n1 = no1;
			n2 = no2;
			ADDVERTEX;
			n2 = no3;
			ADDVERTEX;
			n2 = no4;
			ADDVERTEX;
			n1 = no5;
			n2 = no4;
			ADDVERTEX;
			n2 = no3;
			ADDVERTEX;
			n2 = no6;
			ADDVERTEX;
		    } */
                //   else
                {
                    num_triangles += 4;
                    n1 = no1;
                    n2 = no2;
                    vertex1 = vertex;
                    ADDVERTEX;
                    n2 = no3;
                    ADDVERTEX;
                    n1 = no5;
                    n2 = no3;
                    vertex2 = vertex;
                    ADDVERTEX;
                    *vertex = *vertex1;
                    vertex++;
                    *vertex = *vertex2;
                    vertex++;
                    n2 = no6;
                    ADDVERTEX;
                    n1 = no1;
                    n2 = no4;
                    vertex1 = vertex;
                    ADDVERTEX;
                    n2 = no2;
                    ADDVERTEX;
                    n1 = no5;
                    n2 = no6;
                    vertex2 = vertex;
                    ADDVERTEX;
                    *vertex = *vertex1;
                    vertex++;
                    *vertex = *vertex2;
                    vertex++;
                    n2 = no4;
                    ADDVERTEX;
                }
                break;
            case 5:
                num_triangles += 3;
                vertex1 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                vertex2 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                *vertex = *vertex1;
                vertex++;
                *vertex = *vertex2;
                vertex++;
                vertex2 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                *vertex = *vertex1;
                vertex++;
                *vertex = *vertex2;
                vertex++;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                break;
            case 6:
                num_triangles += 2;
                vertex1 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                vertex2 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                *vertex = *vertex1;
                vertex++;
                *vertex = *vertex2;
                vertex++;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                break;
            case 7:
                num_triangles += 2;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                break;
            case 8:
                num_triangles += 3;
                vertex1 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                vertex2 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                *vertex = *vertex1;
                vertex++;
                *vertex = *vertex2;
                vertex++;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                break;
            case 9:
                num_triangles += 4;
                vertex1 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                vertex2 = vertex;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                *vertex = *vertex1;
                vertex++;
                *vertex = *vertex2;
                vertex++;
                vertex2 = vertex;
                n1 = node_list[*polygon_nodes++];
                ADDVERTEX;
                *vertex = *vertex1;
                vertex++;
                *vertex = *vertex2;
                vertex++;
                vertex2 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                *vertex = *vertex1;
                vertex++;
                *vertex = *vertex2;
                vertex++;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                break;
            case 10:
                num_triangles += 3;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                break;
            case 11:
                num_triangles += 4;
                vertex1 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                vertex2 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                *vertex = *vertex1;
                vertex++;
                *vertex = *vertex2;
                vertex++;
                vertex2 = vertex;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                *vertex = *vertex1;
                vertex++;
                *vertex = *vertex2;
                vertex++;
                vertex2 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                *vertex = *vertex1;
                vertex++;
                *vertex = *vertex2;
                vertex++;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                break;
            case 12:
                num_triangles += 4;
                vertex1 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                vertex2 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                *vertex = *vertex1;
                vertex++;
                *vertex = *vertex2;
                vertex++;
                vertex2 = vertex;
                n1 = node_list[*polygon_nodes++];
                ADDVERTEX;
                *vertex = *vertex1;
                vertex++;
                *vertex = *vertex2;
                vertex++;
                vertex2 = vertex;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                *vertex = *vertex1;
                vertex++;
                *vertex = *vertex2;
                vertex++;
                n2 = node_list[*polygon_nodes++];
                n1 = node_list[*polygon_nodes++];
                ADDVERTEX;
                break;
            case 13:
                num_triangles += 4;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                vertex1 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                vertex2 = vertex;
                n1 = node_list[*polygon_nodes++];
                ADDVERTEX;
                *vertex = *vertex1;
                vertex++;
                *vertex = *vertex2;
                vertex++;
                vertex2 = vertex;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                *vertex = *vertex1;
                vertex++;
                *vertex = *vertex2;
                vertex++;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                break;
            case 14:
                num_triangles += 4;
                vertex1 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                vertex2 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                *vertex = *vertex1;
                vertex++;
                *vertex = *vertex2;
                vertex++;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                vertex1 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                vertex2 = vertex;
                n1 = node_list[*polygon_nodes++];
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                *vertex = *vertex1;
                vertex++;
                *vertex = *vertex2;
                vertex++;
                n2 = node_list[*polygon_nodes++];
                ADDVERTEX;
                break;
            case 15:
                num_triangles += 4;
                if (*polygon_nodes)
                {
                    n1 = node_list[1];
                    n2 = node_list[0];
                    ADDVERTEX;
                    n2 = node_list[5];
                    ADDVERTEX;
                    n2 = node_list[2];
                    ADDVERTEX;
                    n1 = node_list[4];
                    n2 = node_list[5];
                    ADDVERTEX;
                    n2 = node_list[0];
                    ADDVERTEX;
                    n2 = node_list[7];
                    ADDVERTEX;
                    n1 = node_list[6];
                    n2 = node_list[2];
                    ADDVERTEX;
                    n2 = node_list[5];
                    ADDVERTEX;
                    n2 = node_list[7];
                    ADDVERTEX;
                    n1 = node_list[3];
                    n2 = node_list[0];
                    ADDVERTEX;
                    n2 = node_list[2];
                    ADDVERTEX;
                    n2 = node_list[7];
                    ADDVERTEX;
                }
                else
                {
                    n1 = node_list[0];
                    n2 = node_list[1];
                    ADDVERTEX;
                    n2 = node_list[3];
                    ADDVERTEX;
                    n2 = node_list[4];
                    ADDVERTEX;
                    n1 = node_list[5];
                    n2 = node_list[1];
                    ADDVERTEX;
                    n2 = node_list[4];
                    ADDVERTEX;
                    n2 = node_list[6];
                    ADDVERTEX;
                    n1 = node_list[2];
                    n2 = node_list[1];
                    ADDVERTEX;
                    n2 = node_list[6];
                    ADDVERTEX;
                    n2 = node_list[3];
                    ADDVERTEX;
                    n1 = node_list[7];
                    n2 = node_list[4];
                    ADDVERTEX;
                    n2 = node_list[3];
                    ADDVERTEX;
                    n2 = node_list[6];
                    ADDVERTEX;
                }
                break;
            }
        }
    }
// just for testing
#ifdef DEBUG
    fprintf(stderr, " Dreiecke: %d\n", num_triangles);
    for (i = 0; i < 256; i++)
        fprintf(stderr, " %d : %d\n", i, cases[i]);
#endif
}

void UNI_Plane::createPlane()
{
    register int bitmap; // index in the MarchingCubes table
    int n; // 1 = above; 0 = below
    int node_list[8];
    int x_add[] = { 0, 0, 1, 1, 0, 0, 1, 1 };
    int y_add[] = { 0, 1, 1, 0, 0, 1, 1, 0 };
    int z_add[] = { 0, 0, 0, 0, 1, 1, 1, 1 };
    int numIntersections;
    int *polygon_nodes;
    int n1, n2, ii, jj, kk;
    int no1, no2, no3, no4, no5, no6;
    int *vertex1, *vertex2;
    int *n_1 = node_list, *n_2 = node_list + 1, *n_3 = node_list + 2, *n_4 = node_list + 3, *n_5 = node_list + 4, *n_6 = node_list + 5, *n_7 = node_list + 6, *n_8 = node_list + 7;
    *n_1 = 0;
    *n_2 = z_size;
    *n_3 = z_size * (y_size + 1);
    *n_4 = y_size * z_size;
    *n_5 = (*n_1) + 1;
    *n_6 = (*n_2) + 1;
    *n_7 = (*n_3) + 1;
    *n_8 = (*n_4) + 1;
    cutting_info *C_Info;

    if (!node_table)
    {
        /*Initialize hash table */
        hash_table_size = x_size * y_size + y_size * z_size + x_size * z_size;
        if (hash_table_size > num_nodes)
            hash_table_size = num_nodes;

        small_hash_table = (small_hash_table_entry *)malloc(hash_table_size * sizeof(small_hash_table_entry));
        for (n = 0; n < hash_table_size; n++)
        {
            small_hash_table[n].key = -1;
        }
    }

    for (ii = 0; ii < x_size - 1; ii++)
    {
        for (jj = 0; jj < y_size - 1; jj++)
        {
            for (kk = 0; kk < z_size - 1; kk++)
            {

                bitmap = (i_in[*n_1] >= isovalue ? 1 : 0) | (i_in[*n_2] >= isovalue ? 1 : 0) << 1
                         | (i_in[*n_3] >= isovalue ? 1 : 0) << 2 | (i_in[*n_4] >= isovalue ? 1 : 0) << 3
                         | (i_in[*n_5] >= isovalue ? 1 : 0) << 4 | (i_in[*n_6] >= isovalue ? 1 : 0) << 5
                         | (i_in[*n_7] >= isovalue ? 1 : 0) << 6 | (i_in[*n_8] >= isovalue ? 1 : 0) << 7;

                // bitmap is now an index to the Cuttingtable
                C_Info = Cutting_Info[TYPE_HEXAGON] + bitmap;
                numIntersections = C_Info->nvert;
                if (numIntersections)
                {
                    polygon_nodes = C_Info->node_pairs;
                    switch (numIntersections)
                    {
                    case 1:
                        num_triangles++;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        n2 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX03;
                        break;
                    case 2:
                        num_triangles += 2;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        polygon_nodes += 3;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        break;
                    case 3:
                        /*
	 *      Something of a special case here:  If the average of the vertices
	 *      is greater than the isovalue, we create two separated polygons 
	 *      at the vertices.  If it is less, then we make a little valley
	 *      shape.
	 */
                        no1 = node_list[*(polygon_nodes)];
                        no2 = node_list[*(polygon_nodes + 1)];
                        no3 = node_list[*(polygon_nodes + 2)];
                        no4 = node_list[*(polygon_nodes + 3)];
                        no5 = node_list[*(polygon_nodes + 4)];
                        no6 = node_list[*(polygon_nodes + 5)];
                        //if((node_table[no1].dist +
                        //	node_table[no3].dist +
                        //	node_table[no4].dist +
                        //	node_table[no5].dist ) < 0)
                        /*    {
				num_triangles +=2;
				n1 = no1;
				n2 = no2;
				ADDVERTEXX01;
				n2 = no3;
				ADDVERTEXX02;
				n2 = no4;
				ADDVERTEXX03;
				n1 = no5;
				n2 = no4;
				ADDVERTEXX(4,3);
				n2 = no3;
				ADDVERTEXX(4,2);
				n2 = no6;
				ADDVERTEXX(4,5);
			    }
			    else*/
                        {
                            num_triangles += 4;
                            n1 = no1;
                            n2 = no2;
                            vertex1 = vertex;
                            ADDVERTEXX01;
                            n2 = no3;
                            ADDVERTEXX02;
                            n1 = no5;
                            n2 = no3;
                            vertex2 = vertex;
                            ADDVERTEXX(4, 2);
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = no6;
                            ADDVERTEXX(4, 6);
                            n1 = no1;
                            n2 = no4;
                            vertex1 = vertex;
                            ADDVERTEXX04;
                            n2 = no2;
                            ADDVERTEXX01;
                            n1 = no5;
                            n2 = no6;
                            vertex2 = vertex;
                            ADDVERTEXX(4, 5);
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = no4;
                            ADDVERTEXX(4, 3);
                        }
                        break;
                    case 4:
                        /*
	 *      Something of a special case here:  If the average of the vertices
	 *      is smaller than the isovalue, we create two separated polygons 
	 *      at the vertices.  If it is less, then we make a little valley
	 *      shape.
	 */
                        no1 = node_list[*(polygon_nodes)];
                        no2 = node_list[*(polygon_nodes + 1)];
                        no3 = node_list[*(polygon_nodes + 2)];
                        no4 = node_list[*(polygon_nodes + 3)];
                        no5 = node_list[*(polygon_nodes + 4)];
                        no6 = node_list[*(polygon_nodes + 5)];
                        /* if((node_table[no1].dist +
				node_table[no3].dist +
				node_table[no4].dist +
				node_table[no5].dist ) > 0)
			    {
				num_triangles +=2;
				n1 = no1;
				n2 = no2;
				ADDVERTEXX01;
				n2 = no3;
				ADDVERTEXX02;
				n2 = no4;
				ADDVERTEXX03;
				n1 = no5;
				n2 = no4;
				ADDVERTEXX(4,3);
				n2 = no3;
				ADDVERTEXX(4,2);
				n2 = no6;
				ADDVERTEXX(4,5);
			    }
			    else*/
                        {
                            num_triangles += 4;
                            n1 = no1;
                            n2 = no2;
                            vertex1 = vertex;
                            ADDVERTEXX01;
                            n2 = no3;
                            ADDVERTEXX02;
                            n1 = no5;
                            n2 = no3;
                            vertex2 = vertex;
                            ADDVERTEXX(4, 2);
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = no6;
                            ADDVERTEXX(4, 5);
                            n1 = no1;
                            n2 = no4;
                            vertex1 = vertex;
                            ADDVERTEXX03;
                            n2 = no2;
                            ADDVERTEXX01;
                            n1 = no5;
                            n2 = no6;
                            vertex2 = vertex;
                            ADDVERTEXX(4, 5);
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = no4;
                            ADDVERTEXX(4, 3);
                        }
                        break;
                    case 5:
                        num_triangles += 3;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        polygon_nodes += 3;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        polygon_nodes += 2;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        break;
                    case 6:
                        num_triangles += 2;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        polygon_nodes += 2;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        polygon_nodes += 2;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        polygon_nodes += 2;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        break;
                    case 7:
                        num_triangles += 2;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        n2 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX03;
                        polygon_nodes += 4;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        n2 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX03;
                        break;
                    case 8:
                        num_triangles += 3;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        polygon_nodes += 3;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        polygon_nodes += 3;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        n2 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX03;
                        break;
                    case 9:
                        num_triangles += 4;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        polygon_nodes += 2;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        vertex2 = vertex;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n1 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX(3, 2);
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        polygon_nodes += 4;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        break;
                    case 10:
                        num_triangles += 3;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        n2 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX03;
                        polygon_nodes += 4;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        n2 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX03;
                        polygon_nodes += 4;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        n2 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX03;
                        break;
                    case 11:
                        num_triangles += 4;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        polygon_nodes += 3;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        polygon_nodes += 3;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        break;
                    case 12:
                        num_triangles += 4;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        polygon_nodes += 3;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n1 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX(2, 1);
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n2 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX(2, 3);
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n1 = node_list[*(polygon_nodes + 5)];
                        n2 = node_list[*(polygon_nodes + 4)];
                        ADDVERTEXX(5, 4);
                        break;
                    case 13:
                        num_triangles += 4;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        n2 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX03;
                        polygon_nodes += 4;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        vertex2 = vertex;
                        n1 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX(3, 2);
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n2 = node_list[*(polygon_nodes + 4)];
                        ADDVERTEXX(3, 4);
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n1 = node_list[*(polygon_nodes + 5)];
                        n2 = node_list[*(polygon_nodes + 6)];
                        ADDVERTEXX(5, 6);
                        break;
                    case 14:
                        num_triangles += 4;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        polygon_nodes += 3;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        polygon_nodes += 3;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        polygon_nodes += 3;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        break;
                    case 15:
                        num_triangles += 4;
                        if (*polygon_nodes)
                        {
                            n1 = node_list[1];
                            n2 = node_list[0];
                            ADDVERTEXXX(1, 0);
                            n2 = node_list[5];
                            ADDVERTEXXX(1, 5);
                            n2 = node_list[2];
                            ADDVERTEXXX(1, 2);
                            n1 = node_list[4];
                            n2 = node_list[5];
                            ADDVERTEXXX(4, 5);
                            n2 = node_list[0];
                            ADDVERTEXXX(4, 0);
                            n2 = node_list[7];
                            ADDVERTEXXX(4, 7);
                            n1 = node_list[6];
                            n2 = node_list[2];
                            ADDVERTEXXX(6, 2);
                            n2 = node_list[5];
                            ADDVERTEXXX(6, 5);
                            n2 = node_list[7];
                            ADDVERTEXXX(6, 7);
                            n1 = node_list[3];
                            n2 = node_list[0];
                            ADDVERTEXXX(3, 0);
                            n2 = node_list[2];
                            ADDVERTEXXX(3, 2);
                            n2 = node_list[7];
                            ADDVERTEXXX(3, 7);
                        }
                        else
                        {
                            n1 = node_list[0];
                            n2 = node_list[1];
                            ADDVERTEXXX(0, 1);
                            n2 = node_list[3];
                            ADDVERTEXXX(0, 3);
                            n2 = node_list[4];
                            ADDVERTEXXX(0, 4);
                            n1 = node_list[5];
                            n2 = node_list[1];
                            ADDVERTEXXX(5, 1);
                            n2 = node_list[4];
                            ADDVERTEXXX(5, 4);
                            n2 = node_list[6];
                            ADDVERTEXXX(5, 6);
                            n1 = node_list[2];
                            n2 = node_list[1];
                            ADDVERTEXXX(2, 1);
                            n2 = node_list[6];
                            ADDVERTEXXX(2, 6);
                            n2 = node_list[3];
                            ADDVERTEXXX(2, 3);
                            n1 = node_list[7];
                            n2 = node_list[4];
                            ADDVERTEXXX(7, 4);
                            n2 = node_list[3];
                            ADDVERTEXXX(7, 3);
                            n2 = node_list[6];
                            ADDVERTEXXX(7, 6);
                        }
                        break;
                    }
                }
                (*n_1)++;
                (*n_2)++;
                (*n_3)++;
                (*n_4)++;
                (*n_5)++;
                (*n_6)++;
                (*n_7)++;
                (*n_8)++;
            }
            (*n_1)++;
            (*n_2)++;
            (*n_3)++;
            (*n_4)++;
            (*n_5)++;
            (*n_6)++;
            (*n_7)++;
            (*n_8)++;
        }
        (*n_1) += z_size;
        (*n_2) += z_size;
        (*n_3) += z_size;
        (*n_4) += z_size;
        (*n_5) += z_size;
        (*n_6) += z_size;
        (*n_7) += z_size;
        (*n_8) += z_size;
    }
}

void RECT_Plane::createPlane()
{
    register int bitmap; // index in the MarchingCubes table
    // 1 = above; 0 = below
    int node_list[8];
    int x_add[] = { 0, 0, 1, 1, 0, 0, 1, 1 };
    int y_add[] = { 0, 1, 1, 0, 0, 1, 1, 0 };
    int z_add[] = { 0, 0, 0, 0, 1, 1, 1, 1 };
    int numIntersections;
    int *polygon_nodes;
    int n1, n2, ii, jj, kk, n;
    int no1, no2, no3, no4, no5, no6;
    int *vertex1, *vertex2;
    int *n_1 = node_list, *n_2 = node_list + 1, *n_3 = node_list + 2, *n_4 = node_list + 3, *n_5 = node_list + 4, *n_6 = node_list + 5, *n_7 = node_list + 6, *n_8 = node_list + 7;
    *n_1 = 0;
    *n_2 = z_size;
    *n_3 = z_size * (y_size + 1);
    *n_4 = y_size * z_size;
    *n_5 = (*n_1) + 1;
    *n_6 = (*n_2) + 1;
    *n_7 = (*n_3) + 1;
    *n_8 = (*n_4) + 1;
    cutting_info *C_Info;

    if (!node_table)
    {
        /*Initialize hash table */
        hash_table_size = 2 * x_size * y_size + 2 * y_size * z_size + 2 * x_size * z_size;
        if (hash_table_size > num_nodes)
            hash_table_size = num_nodes;

        small_hash_table = (small_hash_table_entry *)malloc(hash_table_size * sizeof(small_hash_table_entry));
        for (n = 0; n < hash_table_size; n++)
        {
            small_hash_table[n].key = -1;
        }
    }

    for (ii = 0; ii < x_size - 1; ii++)
    {
        for (jj = 0; jj < y_size - 1; jj++)
        {
            for (kk = 0; kk < z_size - 1; kk++)
            {

                bitmap = (i_in[*n_1] >= isovalue ? 1 : 0) | (i_in[*n_2] >= isovalue ? 1 : 0) << 1
                         | (i_in[*n_3] >= isovalue ? 1 : 0) << 2 | (i_in[*n_4] >= isovalue ? 1 : 0) << 3
                         | (i_in[*n_5] >= isovalue ? 1 : 0) << 4 | (i_in[*n_6] >= isovalue ? 1 : 0) << 5
                         | (i_in[*n_7] >= isovalue ? 1 : 0) << 6 | (i_in[*n_8] >= isovalue ? 1 : 0) << 7;

                // bitmap is now an index to the Cuttingtable
                C_Info = Cutting_Info[TYPE_HEXAGON] + bitmap;
                numIntersections = C_Info->nvert;
                if (numIntersections)
                {
                    polygon_nodes = C_Info->node_pairs;
                    switch (numIntersections)
                    {
                    case 1:
                        num_triangles++;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        n2 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX03;
                        break;
                    case 2:
                        num_triangles += 2;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        polygon_nodes += 3;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        break;
                    case 3:
                        /*
	 *      Something of a special case here:  If the average of the vertices
	 *      is greater than the isovalue, we create two separated polygons 
	 *      at the vertices.  If it is less, then we make a little valley
	 *      shape.
	 */
                        no1 = node_list[*(polygon_nodes)];
                        no2 = node_list[*(polygon_nodes + 1)];
                        no3 = node_list[*(polygon_nodes + 2)];
                        no4 = node_list[*(polygon_nodes + 3)];
                        no5 = node_list[*(polygon_nodes + 4)];
                        no6 = node_list[*(polygon_nodes + 5)];
                        /* if((node_table[no1].dist +
				node_table[no3].dist +
				node_table[no4].dist +
				node_table[no5].dist ) < 0)
			    {
				num_triangles +=2;
				n1 = no1;
				n2 = no2;
				ADDVERTEXX01;
				n2 = no3;
				ADDVERTEXX02;
				n2 = no4;
				ADDVERTEXX03;
				n1 = no5;
				n2 = no4;
				ADDVERTEXX(4,3);
				n2 = no3;
				ADDVERTEXX(4,2);
				n2 = no6;
				ADDVERTEXX(4,5);
			    }
			    else*/
                        {
                            num_triangles += 4;
                            n1 = no1;
                            n2 = no2;
                            vertex1 = vertex;
                            ADDVERTEXX01;
                            n2 = no3;
                            ADDVERTEXX02;
                            n1 = no5;
                            n2 = no3;
                            vertex2 = vertex;
                            ADDVERTEXX(4, 2);
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = no6;
                            ADDVERTEXX(4, 6);
                            n1 = no1;
                            n2 = no4;
                            vertex1 = vertex;
                            ADDVERTEXX04;
                            n2 = no2;
                            ADDVERTEXX01;
                            n1 = no5;
                            n2 = no6;
                            vertex2 = vertex;
                            ADDVERTEXX(4, 5);
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = no4;
                            ADDVERTEXX(4, 3);
                        }
                        break;
                    case 4:
                        /*
	 *      Something of a special case here:  If the average of the vertices
	 *      is smaller than the isovalue, we create two separated polygons 
	 *      at the vertices.  If it is less, then we make a little valley
	 *      shape.
	 */
                        no1 = node_list[*(polygon_nodes)];
                        no2 = node_list[*(polygon_nodes + 1)];
                        no3 = node_list[*(polygon_nodes + 2)];
                        no4 = node_list[*(polygon_nodes + 3)];
                        no5 = node_list[*(polygon_nodes + 4)];
                        no6 = node_list[*(polygon_nodes + 5)];
                        /*if((node_table[no1].dist +
				node_table[no3].dist +
				node_table[no4].dist +
				node_table[no5].dist ) > 0)
			    {
				num_triangles +=2;
				n1 = no1;
				n2 = no2;
				ADDVERTEXX01;
				n2 = no3;
				ADDVERTEXX02;
				n2 = no4;
				ADDVERTEXX03;
				n1 = no5;
				n2 = no4;
				ADDVERTEXX(4,3);
				n2 = no3;
				ADDVERTEXX(4,2);
				n2 = no6;
				ADDVERTEXX(4,5);
			    }
			    else*/
                        {
                            num_triangles += 4;
                            n1 = no1;
                            n2 = no2;
                            vertex1 = vertex;
                            ADDVERTEXX01;
                            n2 = no3;
                            ADDVERTEXX02;
                            n1 = no5;
                            n2 = no3;
                            vertex2 = vertex;
                            ADDVERTEXX(4, 2);
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = no6;
                            ADDVERTEXX(4, 5);
                            n1 = no1;
                            n2 = no4;
                            vertex1 = vertex;
                            ADDVERTEXX03;
                            n2 = no2;
                            ADDVERTEXX01;
                            n1 = no5;
                            n2 = no6;
                            vertex2 = vertex;
                            ADDVERTEXX(4, 5);
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = no4;
                            ADDVERTEXX(4, 3);
                        }
                        break;
                    case 5:
                        num_triangles += 3;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        polygon_nodes += 3;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        polygon_nodes += 2;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        break;
                    case 6:
                        num_triangles += 2;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        polygon_nodes += 2;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        polygon_nodes += 2;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        polygon_nodes += 2;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        break;
                    case 7:
                        num_triangles += 2;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        n2 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX03;
                        polygon_nodes += 4;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        n2 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX03;
                        break;
                    case 8:
                        num_triangles += 3;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        polygon_nodes += 3;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        polygon_nodes += 3;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        n2 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX03;
                        break;
                    case 9:
                        num_triangles += 4;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        polygon_nodes += 2;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        vertex2 = vertex;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n1 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX(3, 2);
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        polygon_nodes += 4;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        break;
                    case 10:
                        num_triangles += 3;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        n2 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX03;
                        polygon_nodes += 4;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        n2 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX03;
                        polygon_nodes += 4;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        n2 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX03;
                        break;
                    case 11:
                        num_triangles += 4;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        polygon_nodes += 3;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        polygon_nodes += 3;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        break;
                    case 12:
                        num_triangles += 4;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        polygon_nodes += 3;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n1 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX(2, 1);
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n2 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX(2, 3);
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n1 = node_list[*(polygon_nodes + 5)];
                        n2 = node_list[*(polygon_nodes + 4)];
                        ADDVERTEXX(5, 4);
                        break;
                    case 13:
                        num_triangles += 4;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        n2 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX03;
                        polygon_nodes += 4;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        vertex2 = vertex;
                        n1 = node_list[*(polygon_nodes + 3)];
                        ADDVERTEXX(3, 2);
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n2 = node_list[*(polygon_nodes + 4)];
                        ADDVERTEXX(3, 4);
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n1 = node_list[*(polygon_nodes + 5)];
                        n2 = node_list[*(polygon_nodes + 6)];
                        ADDVERTEXX(5, 6);
                        break;
                    case 14:
                        num_triangles += 4;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        polygon_nodes += 3;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        polygon_nodes += 3;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        polygon_nodes += 3;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes];
                        n2 = node_list[*(polygon_nodes + 1)];
                        ADDVERTEXX01;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*(polygon_nodes + 2)];
                        ADDVERTEXX02;
                        break;
                    case 15:
                        num_triangles += 4;
                        if (*polygon_nodes)
                        {
                            n1 = node_list[1];
                            n2 = node_list[0];
                            ADDVERTEXXX(1, 0);
                            n2 = node_list[5];
                            ADDVERTEXXX(1, 5);
                            n2 = node_list[2];
                            ADDVERTEXXX(1, 2);
                            n1 = node_list[4];
                            n2 = node_list[5];
                            ADDVERTEXXX(4, 5);
                            n2 = node_list[0];
                            ADDVERTEXXX(4, 0);
                            n2 = node_list[7];
                            ADDVERTEXXX(4, 7);
                            n1 = node_list[6];
                            n2 = node_list[2];
                            ADDVERTEXXX(6, 2);
                            n2 = node_list[5];
                            ADDVERTEXXX(6, 5);
                            n2 = node_list[7];
                            ADDVERTEXXX(6, 7);
                            n1 = node_list[3];
                            n2 = node_list[0];
                            ADDVERTEXXX(3, 0);
                            n2 = node_list[2];
                            ADDVERTEXXX(3, 2);
                            n2 = node_list[7];
                            ADDVERTEXXX(3, 7);
                        }
                        else
                        {
                            n1 = node_list[0];
                            n2 = node_list[1];
                            ADDVERTEXXX(0, 1);
                            n2 = node_list[3];
                            ADDVERTEXXX(0, 3);
                            n2 = node_list[4];
                            ADDVERTEXXX(0, 4);
                            n1 = node_list[5];
                            n2 = node_list[1];
                            ADDVERTEXXX(5, 1);
                            n2 = node_list[4];
                            ADDVERTEXXX(5, 4);
                            n2 = node_list[6];
                            ADDVERTEXXX(5, 6);
                            n1 = node_list[2];
                            n2 = node_list[1];
                            ADDVERTEXXX(2, 1);
                            n2 = node_list[6];
                            ADDVERTEXXX(2, 6);
                            n2 = node_list[3];
                            ADDVERTEXXX(2, 3);
                            n1 = node_list[7];
                            n2 = node_list[4];
                            ADDVERTEXXX(7, 4);
                            n2 = node_list[3];
                            ADDVERTEXXX(7, 3);
                            n2 = node_list[6];
                            ADDVERTEXXX(7, 6);
                        }
                        break;
                    }
                }
                (*n_1)++;
                (*n_2)++;
                (*n_3)++;
                (*n_4)++;
                (*n_5)++;
                (*n_6)++;
                (*n_7)++;
                (*n_8)++;
            }
            (*n_1)++;
            (*n_2)++;
            (*n_3)++;
            (*n_4)++;
            (*n_5)++;
            (*n_6)++;
            (*n_7)++;
            (*n_8)++;
        }
        (*n_1) += z_size;
        (*n_2) += z_size;
        (*n_3) += z_size;
        (*n_4) += z_size;
        (*n_5) += z_size;
        (*n_6) += z_size;
        (*n_7) += z_size;
        (*n_8) += z_size;
    }
}

void STR_Plane::createPlane()
{
    register int bitmap; // index in the MarchingCubes table
    // 1 = above; 0 = below
    int node_list[8];
    int numIntersections;
    int *polygon_nodes;
    int n1, n2, ii, jj, kk, n;
    int no1, no2, no3, no4, no5, no6;
    int *vertex1, *vertex2;
    int *n_1 = node_list, *n_2 = node_list + 1, *n_3 = node_list + 2, *n_4 = node_list + 3, *n_5 = node_list + 4, *n_6 = node_list + 5, *n_7 = node_list + 6, *n_8 = node_list + 7;
    *n_1 = 0;
    *n_2 = z_size;
    *n_3 = z_size * (y_size + 1);
    *n_4 = y_size * z_size;
    *n_5 = (*n_1) + 1;
    *n_6 = (*n_2) + 1;
    *n_7 = (*n_3) + 1;
    *n_8 = (*n_4) + 1;
    cutting_info *C_Info;

    if (!node_table)
    {
        /*Initialize hash table */
        hash_table_size = 2 * x_size * y_size + 2 * y_size * z_size + 2 * x_size * z_size;
        if (hash_table_size > num_nodes)
            hash_table_size = num_nodes;

        small_hash_table = (small_hash_table_entry *)malloc(hash_table_size * sizeof(small_hash_table_entry));
        for (n = 0; n < hash_table_size; n++)
        {
            small_hash_table[n].key = -1;
        }
    }

    for (ii = 0; ii < x_size - 1; ii++)
    {
        for (jj = 0; jj < y_size - 1; jj++)
        {
            for (kk = 0; kk < z_size - 1; kk++)
            {

                bitmap = (i_in[*n_1] >= isovalue ? 1 : 0) | (i_in[*n_2] >= isovalue ? 1 : 0) << 1
                         | (i_in[*n_3] >= isovalue ? 1 : 0) << 2 | (i_in[*n_4] >= isovalue ? 1 : 0) << 3
                         | (i_in[*n_5] >= isovalue ? 1 : 0) << 4 | (i_in[*n_6] >= isovalue ? 1 : 0) << 5
                         | (i_in[*n_7] >= isovalue ? 1 : 0) << 6 | (i_in[*n_8] >= isovalue ? 1 : 0) << 7;

                // bitmap is now an index to the Cuttingtable
                C_Info = Cutting_Info[TYPE_HEXAGON] + bitmap;
                numIntersections = C_Info->nvert;

                if (numIntersections)
                {
                    polygon_nodes = C_Info->node_pairs;
                    switch (numIntersections)
                    {
                    case 1:
                        num_triangles++;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 2:
                        num_triangles += 2;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 3:
                        /*
	 *      Something of a special case here:  If the average of the vertices
	 *      is greater than the isovalue, we create two separated polygons 
	 *      at the vertices.  If it is less, then we make a little valley
	 *      shape.
	 */
                        no1 = node_list[*polygon_nodes++];
                        no2 = node_list[*polygon_nodes++];
                        no3 = node_list[*polygon_nodes++];
                        no4 = node_list[*polygon_nodes++];
                        no5 = node_list[*polygon_nodes++];
                        no6 = node_list[*polygon_nodes++];
                        /*if((node_table[no1].dist +
				node_table[no3].dist +
				node_table[no4].dist +
				node_table[no5].dist ) < 0)
			    {
				num_triangles +=2;
				n1 = no1;
				n2 = no2;
				ADDVERTEX;
				n2 = no3;
				ADDVERTEX;
				n2 = no4;
				ADDVERTEX;
				n1 = no5;
				n2 = no4;
				ADDVERTEX;
				n2 = no3;
				ADDVERTEX;
				n2 = no6;
				ADDVERTEX;
			    }
			    else*/
                        {
                            num_triangles += 4;
                            n1 = no1;
                            n2 = no2;
                            vertex1 = vertex;
                            ADDVERTEX;
                            n2 = no3;
                            ADDVERTEX;
                            n1 = no5;
                            n2 = no3;
                            vertex2 = vertex;
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = no6;
                            ADDVERTEX;
                            n1 = no1;
                            n2 = no4;
                            vertex1 = vertex;
                            ADDVERTEX;
                            n2 = no2;
                            ADDVERTEX;
                            n1 = no5;
                            n2 = no6;
                            vertex2 = vertex;
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = no4;
                            ADDVERTEX;
                        }
                        break;
                    case 4:
                        /*
	 *      Something of a special case here:  If the average of the vertices
	 *      is smaller than the isovalue, we create two separated polygons 
	 *      at the vertices.  If it is less, then we make a little valley
	 *      shape.
	 */
                        no1 = node_list[*polygon_nodes++];
                        no2 = node_list[*polygon_nodes++];
                        no3 = node_list[*polygon_nodes++];
                        no4 = node_list[*polygon_nodes++];
                        no5 = node_list[*polygon_nodes++];
                        no6 = node_list[*polygon_nodes++];
                        /*if((node_table[no1].dist +
				node_table[no3].dist +
				node_table[no4].dist +
				node_table[no5].dist ) > 0)
			    {
				num_triangles +=2;
				n1 = no1;
				n2 = no2;
				ADDVERTEX;
				n2 = no3;
				ADDVERTEX;
				n2 = no4;
				ADDVERTEX;
				n1 = no5;
				n2 = no4;
				ADDVERTEX;
				n2 = no3;
				ADDVERTEX;
				n2 = no6;
				ADDVERTEX;
			    }
			    else*/
                        {
                            num_triangles += 4;
                            n1 = no1;
                            n2 = no2;
                            vertex1 = vertex;
                            ADDVERTEX;
                            n2 = no3;
                            ADDVERTEX;
                            n1 = no5;
                            n2 = no3;
                            vertex2 = vertex;
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = no6;
                            ADDVERTEX;
                            n1 = no1;
                            n2 = no4;
                            vertex1 = vertex;
                            ADDVERTEX;
                            n2 = no2;
                            ADDVERTEX;
                            n1 = no5;
                            n2 = no6;
                            vertex2 = vertex;
                            ADDVERTEX;
                            *vertex = *vertex1;
                            vertex++;
                            *vertex = *vertex2;
                            vertex++;
                            n2 = no4;
                            ADDVERTEX;
                        }
                        break;
                    case 5:
                        num_triangles += 3;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 6:
                        num_triangles += 2;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 7:
                        num_triangles += 2;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 8:
                        num_triangles += 3;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 9:
                        num_triangles += 4;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex2 = vertex;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 10:
                        num_triangles += 3;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 11:
                        num_triangles += 4;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 12:
                        num_triangles += 4;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*polygon_nodes++];
                        n1 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 13:
                        num_triangles += 4;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        vertex2 = vertex;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 14:
                        num_triangles += 4;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex1 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        vertex2 = vertex;
                        n1 = node_list[*polygon_nodes++];
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        *vertex = *vertex1;
                        vertex++;
                        *vertex = *vertex2;
                        vertex++;
                        n2 = node_list[*polygon_nodes++];
                        ADDVERTEX;
                        break;
                    case 15:
                        num_triangles += 4;
                        if (*polygon_nodes)
                        {
                            n1 = node_list[1];
                            n2 = node_list[0];
                            ADDVERTEX;
                            n2 = node_list[5];
                            ADDVERTEX;
                            n2 = node_list[2];
                            ADDVERTEX;
                            n1 = node_list[4];
                            n2 = node_list[5];
                            ADDVERTEX;
                            n2 = node_list[0];
                            ADDVERTEX;
                            n2 = node_list[7];
                            ADDVERTEX;
                            n1 = node_list[6];
                            n2 = node_list[2];
                            ADDVERTEX;
                            n2 = node_list[5];
                            ADDVERTEX;
                            n2 = node_list[7];
                            ADDVERTEX;
                            n1 = node_list[3];
                            n2 = node_list[0];
                            ADDVERTEX;
                            n2 = node_list[2];
                            ADDVERTEX;
                            n2 = node_list[7];
                            ADDVERTEX;
                        }
                        else
                        {
                            n1 = node_list[0];
                            n2 = node_list[1];
                            ADDVERTEX;
                            n2 = node_list[3];
                            ADDVERTEX;
                            n2 = node_list[4];
                            ADDVERTEX;
                            n1 = node_list[5];
                            n2 = node_list[1];
                            ADDVERTEX;
                            n2 = node_list[4];
                            ADDVERTEX;
                            n2 = node_list[6];
                            ADDVERTEX;
                            n1 = node_list[2];
                            n2 = node_list[1];
                            ADDVERTEX;
                            n2 = node_list[6];
                            ADDVERTEX;
                            n2 = node_list[3];
                            ADDVERTEX;
                            n1 = node_list[7];
                            n2 = node_list[4];
                            ADDVERTEX;
                            n2 = node_list[3];
                            ADDVERTEX;
                            n2 = node_list[6];
                            ADDVERTEX;
                        }
                        break;
                    }
                }
                (*n_1)++;
                (*n_2)++;
                (*n_3)++;
                (*n_4)++;
                (*n_5)++;
                (*n_6)++;
                (*n_7)++;
                (*n_8)++;
            }
            (*n_1)++;
            (*n_2)++;
            (*n_3)++;
            (*n_4)++;
            (*n_5)++;
            (*n_6)++;
            (*n_7)++;
            (*n_8)++;
        }
        (*n_1) += z_size;
        (*n_2) += z_size;
        (*n_3) += z_size;
        (*n_4) += z_size;
        (*n_5) += z_size;
        (*n_6) += z_size;
        (*n_7) += z_size;
        (*n_8) += z_size;
    }
}
void Plane::add_vertex(int n1, int n2)
{

    int *targets, *indices; // Pointers into the node_info structure
    double w2, w1;
    double d1, d2;

    int n = 0, around_once;

    hash_table_entry *cur_hash_entry;
    int cur_key;

    if (!node_table)
    {
        cur_key = n1;

        /* In the hash table, find an entry with the current key (if there is one) */
        cur_hash_entry = &hash_table[FIND_HASH_INDEX(cur_key)];
        around_once = 0;
        while (cur_hash_entry->key != cur_key && cur_hash_entry->key != -1)
        {
            /* Keep looking, if this entry is empty, or is not the current key*/
            cur_hash_entry++;
            if (cur_hash_entry == hash_table + hash_table_size)
            {
                /* Make sure we don't loop infinitely */
                if (around_once)
                {
                    fprintf(stderr, "Hash Table full");
                }
                around_once = 1;
                cur_hash_entry = hash_table;
            }
        }
        if (cur_hash_entry->key == cur_key)
        {
            targets = cur_hash_entry->targets;
            indices = cur_hash_entry->vertice_list;
            while ((*targets) && (n < 12))
            {
                if (*targets == n2) // did we already calculate this vertex?
                {
                    *vertex++ = *indices; // great! just put in the right index.
                    return;
                }
                targets++;
                indices++;
                n++;
            }
        }
        else if (cur_hash_entry->key == -1)
        {
            cur_hash_entry->key = cur_key;
            cur_hash_entry->targets[0] = 0;
            targets = cur_hash_entry->targets;
            indices = cur_hash_entry->vertice_list;
        }
    }
    else
    {
        targets = node_table[n1].targets;
        indices = node_table[n1].vertice_list;

        int n = 0;
        while ((*targets) && (n < 12))
        {
            if (*targets == n2) // did we already calculate this vertex?
            {
                *vertex++ = *indices; // great! just put in the right index.
                return;
            }
            targets++;
            indices++;
            n++;
        }
    }
    // remember the target we will calculate now

    *targets++ = n2;
    *targets = 0;
    *indices = num_coords;
    *vertex++ = *indices;

    // Calculate the interpolation weights (linear interpolation)
    d1 = i_in[n1] - isovalue;
    d2 = i_in[n2] - isovalue;
    if (d1 == d2)
        w2 = 1.0;
    else
    {
        w2 = d1 / (d1 - d2);
        if (w2 > 1.0)
            w2 = 1.0;
        if (w2 < 0)
            w2 = 0.0;
    }
    w1 = 1.0 - w2;
    *coord_x++ = x_in[n1] * w1 + x_in[n2] * w2;
    *coord_y++ = y_in[n1] * w1 + y_in[n2] * w2;
    *coord_z++ = z_in[n1] * w1 + z_in[n2] * w2;
    if (Datatype == 1)
        *S_Data_p++ = s_in[n1] * w1 + s_in[n2] * w2;
    else if (Datatype == 2)
    {
        *V_Data_U_p++ = u_in[n1] * w1 + u_in[n2] * w2;
        *V_Data_V_p++ = v_in[n1] * w1 + v_in[n2] * w2;
        *V_Data_W_p++ = w_in[n1] * w1 + w_in[n2] * w2;
    }
    num_coords++;
}

void Plane::add_vertex(int n1, int n2, int x, int y, int z, int u, int v, int w)
{

    int *targets, *indices; // Pointers into the node_info structure
    double w2, w1;
    double d1, d2;

    int n = 0, around_once;

    small_hash_table_entry *cur_hash_entry;
    int cur_key;

    if (!node_table)
    {
        cur_key = n1;

        /* In the hash table, find an entry with the current key (if there is one) */
        cur_hash_entry = &small_hash_table[FIND_HASH_INDEX(cur_key)];
        around_once = 0;
        while (cur_hash_entry->key != cur_key && cur_hash_entry->key != -1)
        {
            /* Keep looking, if this entry is empty, or is not the current key*/
            cur_hash_entry++;
            if (cur_hash_entry == small_hash_table + hash_table_size)
            {
                /* Make sure we don't loop infinitely */
                if (around_once)
                {
                    fprintf(stderr, "Hash Table full");
                }
                around_once = 1;
                cur_hash_entry = small_hash_table;
            }
        }
        if (cur_hash_entry->key == cur_key)
        {
            targets = cur_hash_entry->targets;
            indices = cur_hash_entry->vertice_list;
            while ((*targets) && (n < 12))
            {
                if (*targets == n2) // did we already calculate this vertex?
                {
                    *vertex++ = *indices; // great! just put in the right index.
                    return;
                }
                targets++;
                indices++;
                n++;
            }
        }
        else if (cur_hash_entry->key == -1)
        {
            cur_hash_entry->key = cur_key;
            cur_hash_entry->targets[0] = 0;
            targets = cur_hash_entry->targets;
            indices = cur_hash_entry->vertice_list;
        }
    }
    else
    {
        targets = node_table[n1].targets;
        indices = node_table[n1].vertice_list;

        int n = 0;
        while ((*targets) && (n < 12))
        {
            if (*targets == n2) // did we already calculate this vertex?
            {
                *vertex++ = *indices; // great! just put in the right index.
                return;
            }
            targets++;
            indices++;
            n++;
        }
    }
    // remember the target we will calculate now
    // remember the target we will calculate now

    *targets++ = n2;
    *targets = 0;
    *indices = num_coords;
    *vertex++ = *indices;

    // Calculate the interpolation weights (linear interpolation)

    d1 = i_in[n1] - isovalue;
    d2 = i_in[n2] - isovalue;
    if (d1 == d2)
        w2 = 1.0;
    else
    {
        w2 = d1 / (d1 - d2);
        if (w2 > 1.0)
            w2 = 1.0;
        if (w2 < 0)
            w2 = 0.0;
    }
    w1 = 1.0 - w2;
    *coord_x++ = x_in[x] * w1 + x_in[u] * w2;
    *coord_y++ = y_in[y] * w1 + y_in[v] * w2;
    *coord_z++ = z_in[z] * w1 + z_in[w] * w2;
    if (Datatype == 1)
        *S_Data_p++ = s_in[n1] * w1 + s_in[n2] * w2;
    else if (Datatype == 2)
    {
        *V_Data_U_p++ = u_in[n1] * w1 + u_in[n2] * w2;
        *V_Data_V_p++ = v_in[n1] * w1 + v_in[n2] * w2;
        *V_Data_W_p++ = w_in[n1] * w1 + w_in[n2] * w2;
    }
    num_coords++;
}

void Plane::createcoDistributedObjects(char *Data_name, char *Normal_name,
                                       char *Triangle_name, coDoSet *Data_set,
                                       coDoSet *Normal_set, coDoSet *Triangle_set)
{
    float *u_out, *v_out, *w_out;
    int *vl, *pl, i;
    coDoFloat *s_data_out;
    coDoVec3 *v_data_out;
    coDoPolygons *polygons_out;
    coDoTriangleStrips *strips_out;
    coDoVec3 *normals_out;
    if (num_coords == 0)
        return;
    if (Datatype == 1) // (Scalar Data)
    {
        s_data_out = new coDoFloat(Data_name, num_coords, S_Data);
        if (!s_data_out->objectOk())
        {
            Covise::sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
        if (Data_set)
            Data_set->addElement(s_data_out);
        delete s_data_out;
    }
    else if (Datatype == 2)
    {
        v_data_out = new coDoVec3(Data_name, num_coords, V_Data_U, V_Data_V, V_Data_W);
        if (!v_data_out->objectOk())
        {
            Covise::sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
        if (Data_set)
            Data_set->addElement(v_data_out);
        delete v_data_out;
    }
    num_vertices = vertex - vertice_list;
    if (gennormals)
    {
        createNormals();
        normals_out = new coDoVec3(Normal_name, num_coords, Normals_U, Normals_V, Normals_W);
        if (!normals_out->objectOk())
        {
            Covise::sendError("ERROR: creation of data object 'normalsOut' failed");
            return;
        }
        if (Normal_set)
            Normal_set->addElement(normals_out);
        delete normals_out;
        delete[] Normals_U;
        delete[] Normals_V;
        delete[] Normals_W;
    }
    if (genstrips)
    {
        createStrips();
        strips_out = new coDoTriangleStrips(Triangle_name, num_coords, coords_x, coords_y, coords_z, num_triangles + 2 * num_strips, ts_vertice_list, num_strips, ts_line_list);
        delete[] ts_vertice_list;
        delete[] ts_line_list;
        if (strips_out->objectOk())
        {
            strips_out->addAttribute("vertexOrder", "2");
            strips_out->addAttribute("COLOR", colorn);
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
        if (Triangle_set)
            Triangle_set->addElement(strips_out);
        delete strips_out;
    }
    else
    {
        polygons_out = new coDoPolygons(Triangle_name, num_coords, num_vertices, num_triangles);
        if (polygons_out->objectOk())
        {
            polygons_out->get_adresses(&u_out, &v_out, &w_out, &vl, &pl);
            memcpy(u_out, coords_x, num_coords * sizeof(float));
            memcpy(v_out, coords_y, num_coords * sizeof(float));
            memcpy(w_out, coords_z, num_coords * sizeof(float));
            memcpy(vl, vertice_list, num_vertices * sizeof(int));
            for (i = 0; i < num_triangles; i++)
                pl[i] = i * 3;
            polygons_out->addAttribute("vertexOrder", "2");
            polygons_out->addAttribute("COLOR", colorn);
        }
        else
        {
            Covise::sendError("ERROR: creation of data object 'dataOut' failed");
            return;
        }
        if (Triangle_set)
            Triangle_set->addElement(polygons_out);
        delete polygons_out;
    }
}

void Plane::createNormals()
{
    int i, n0, n1, n2, n, *np, *np2;
    float *U, *V, *W, x1, y1, z1, x2, y2, z2;
    float *NU;
    float *NV;
    float *NW;
    float *F_Normals_U;
    float *F_Normals_V;
    float *F_Normals_W;
    NU = Normals_U = new float[num_coords];
    NV = Normals_V = new float[num_coords];
    NW = Normals_W = new float[num_coords];
    U = F_Normals_U = new float[num_triangles];
    V = F_Normals_V = new float[num_triangles];
    W = F_Normals_W = new float[num_triangles];
    np = neighbors = new int[num_coords * 17];
    for (i = 0; i < num_coords; i++)
    {
        *np = 0;
        np += 17;
    }
    n = 0;
    for (i = 0; i < num_vertices; i += 3)
    {
        n0 = vertice_list[i];
        n1 = vertice_list[i + 1];
        n2 = vertice_list[i + 2];
        x1 = coords_x[n1] - coords_x[n0];
        y1 = coords_y[n1] - coords_y[n0];
        z1 = coords_z[n1] - coords_z[n0];
        x2 = coords_x[n2] - coords_x[n0];
        y2 = coords_y[n2] - coords_y[n0];
        z2 = coords_z[n2] - coords_z[n0];
        *U = y1 * z2 - y2 * z1;
        *V = x2 * z1 - x1 * z2;
        *W = x1 * y2 - x2 * y1;
        /*	Normalize Face-Normals 
	l=sqrt(*U * *U+*V * *V+*W * *W);
	if(l!=0.0)
	{
	    *U/=l;
	    *V/=l;
	    *W/=l;
	}
	else
	{
	    *U=1.0;
	    *V=0;
	    *W=0;
	} */
        U++;
        V++;
        W++;
        np = neighbors + 17 * n0;
        (*np)++;
        *(np + (*np)) = n;
        np = neighbors + 17 * n1;
        (*np)++;
        *(np + (*np)) = n;
        np = neighbors + 17 * n2;
        (*np)++;
        *(np + (*np)) = n;
        n++;
    }
    np = neighbors;
    for (i = 0; i < num_coords; i++)
    {
        np2 = np;
        *NU = *NV = *NW = 0;
        // if(*np > 12) printf("np: %d\n",*np);
        for (n = 0; n < *np; n++)
        {
            // if(*np2>num_triangles)
            //	printf("np2: %d\n",*np2);
            np2++;
            *NU += F_Normals_U[*np2];
            *NV += F_Normals_V[*np2];
            *NW += F_Normals_W[*np2];
        }
        np += 17;
        /*	Normalize Normals
	l=sqrt(*NU * *NU+*NV * *NV+*NW * *NW);
	if(l!=0.0)
	{
	    *NU/=l;
	    *NV/=l;
	    *NW/=l;
	}
	else
	{
	    *NU=1.0;
	    *NV=0;
	    *NW=0;
	} */

        NU++;
        NV++;
        NW++;
    }
    delete[] F_Normals_U;
    delete[] F_Normals_V;
    delete[] F_Normals_W;
    if (!genstrips) // do not delete the neighborlist because we need it for the strips
        delete[] neighbors;
}

void Plane::createStrips()
{
    int i, n0, n1, n2, n, next_n, j, tn, el = 0, num_try;
    int *np, *ts_vl, *ts_ll, *td, *triangle_done;
    td = triangle_done = new int[num_triangles];
    ts_vl = ts_vertice_list = new int[num_vertices];
    ts_ll = ts_line_list = new int[num_triangles];
    if (!gennormals)
    {
        np = neighbors = new int[num_coords * 17];
        for (i = 0; i < num_coords; i++)
        {
            *np = 0;
            np += 17;
        }
        n = 0;
        for (i = 0; i < num_vertices; i += 3)
        {
            np = neighbors + 17 * vertice_list[i];
            (*np)++;
            *(np + (*np)) = n;
            np = neighbors + 17 * vertice_list[i + 1];
            (*np)++;
            *(np + (*np)) = n;
            np = neighbors + 17 * vertice_list[i + 2];
            (*np)++;
            *(np + (*np)) = n;
            *td++ = 0; // flag = TRUE if Triangle already done
            n++;
        }
    }
    else
    {
        for (i = 0; i < num_triangles; i++)
            *td++ = 0;
    }
    np = neighbors;
    num_strips = 0;
    el = 0;
    td = triangle_done;
    n0 = 0;
    n1 = 1;
    n2 = 2;
    num_try = 3;
    for (i = 0; i < num_vertices; i += 3)
    {
        if (!(*td)) // Skip Triangle if we already processed it
        {
            // First Triangle of strip
            //printf("%d\n",el);
            *td = 1;
            num_strips++;
            el = 0;
            num_try = 0;
            *ts_ll++ = ts_vl - ts_vertice_list; // line list points to beginning of strip
            *ts_vl++ = n0 = vertice_list[i]; // first and second vertex of strip
            *ts_vl++ = n1 = vertice_list[i + 1];
            next_n = n2 = vertice_list[i + 2];
            while ((el < 2) && (num_try < 3))
            {
                while (next_n != -1)
                {
                    el++;
                    *ts_vl++ = next_n; // next vertex of Strip
                    n2 = next_n;
                    next_n = -1;
                    // find the next vertex now
                    np = neighbors + 17 * n2; // look for neighbors at point 2
                    for (j = *np; j > 0; j--)
                    {
                        tn = np[j]; // this could be the next triangle
                        if (!triangle_done[np[j]]) // if the neighbortriangle is not already processed
                        {
                            tn *= 3; // tn is now an index to the verice_list
                            if (n2 == vertice_list[tn])
                            {
                                if (n1 == vertice_list[tn + 1])
                                {
                                    next_n = vertice_list[tn + 2];
                                    n0 = n1;
                                    n1 = n2;
                                    triangle_done[tn / 3] = 1;
                                    break;
                                }
                                else if (n1 == vertice_list[tn + 2])
                                {
                                    next_n = vertice_list[tn + 1];
                                    n0 = n1;
                                    n1 = n2;
                                    triangle_done[tn / 3] = 1;
                                    break;
                                }
                            }
                            else if (n2 == vertice_list[tn + 1])
                            {
                                if (n1 == vertice_list[tn])
                                {
                                    next_n = vertice_list[tn + 2];
                                    n0 = n1;
                                    n1 = n2;
                                    triangle_done[tn / 3] = 1;
                                    break;
                                }
                                else if (n1 == vertice_list[tn + 2])
                                {
                                    next_n = vertice_list[tn];
                                    n0 = n1;
                                    n1 = n2;
                                    triangle_done[tn / 3] = 1;
                                    break;
                                }
                            }
                            else if (n2 == vertice_list[tn + 2])
                            {
                                if (n1 == vertice_list[tn])
                                {
                                    next_n = vertice_list[tn + 1];
                                    n0 = n1;
                                    n1 = n2;
                                    triangle_done[tn / 3] = 1;
                                    break;
                                }
                                else if (n1 == vertice_list[tn + 1])
                                {
                                    next_n = vertice_list[tn];
                                    n0 = n1;
                                    n1 = n2;
                                    triangle_done[tn / 3] = 1;
                                    break;
                                }
                            }
                        }
                    }
                }
                num_try++;
                if ((el == 1) && (num_try < 2)) // Try the other two Sides if no neighbor found
                {
                    el = 0;
                    next_n = n0;
                    n0 = n1;
                    n1 = n2;
                    ts_vl--;
                    *(ts_vl - 1) = n1;
                    *(ts_vl - 2) = n0;
                }
            }
        }
        td++;
    }
    //printf("%d\n",el);
    delete[] neighbors; // we don´t need it anymore
    delete[] triangle_done;
    //printf("strips: %d\n",num_strips);
    // printf("triangles: %d\n",num_triangles);
}
