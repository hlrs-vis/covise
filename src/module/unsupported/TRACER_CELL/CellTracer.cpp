/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/**************************************************************************\ 
 **                                                           (C)1996 RUS  **
 **                                                                        **
 ** Description: COVISE Tracer3_USG create streamlines 			  **
 **		of unstructured grids using adjacency information 	  **
 **              (single block)                                            **
 **		Uses 2nd order and 4th order integration with adaptive    **
 **		stepsize control. Adjacency information is computed in	  **
 **		a pre process and stored in a list (memory expensive)	  **
 **		or computed on the fly(slower)				  **
 **		Contains possibility of point reduction on output and	  **
 **									  **
 **		Input: geometrie field, data field (vector)		  **
 **		Output: Streamlines 			 		  **
 **			Datafield (vorticity of line (scalar))		  **
 **			Datafield (vorticity of line (vector))		  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) 1996                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Oliver Heck                                                   **
 **                                                                        **
 **                                                                        **
 ** Date:  26.03.96  V1.0  						  **
 **                                                                        **
 ** 								          **
\**************************************************************************/

//
// Covise include stuff
//
#include <stdio.h>
#include <iostream.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include "CellTracer.h"
//
// functions
//
void rk2_tet(float px[], float v[], float dt, float phead[]);
void rk4_tet(float px[], float v[], float dt, float phead[]);
void tracer(int, float *, int *, float *, float *, float *, float *,
            float *, float *, float *, int *, int *);

void print_adjacency();

int isin_cell(float[], int);
int compare(const void *, const void *);
int find_startcell_fast(float *);
int find_actualcell(int, float *);

float stepsize(int m);
//   float p_ang(float[], float[], float[]);
float dout(float[]);
float Min_of(float[], int);
float Max_of(float[], int);

double p_ang(float[], float[], float[]);

inline float sqr(float val)
{
    val *= val;
    return val;
}

#ifndef __sgi
#define fabsf fabs
#endif

// global variables
//static int actual_cell_type;
float trace_eps = 0.00001;
float trace_len = 0.00001;
int *el, *cl, *tl;
float *xm, *ym, *zm;
int *gci_l, *gcsc_l, *cigc_l;
int xglob, yglob, zglob;
int intpol_dir, set_num_elem, option, stp_control; //get choice of interpolation method
int reduce, data_out_type;
int numelem, numelem_o, numconn, numcoord;
// CellGrid specific
int numcells, numedges, numfaces, numfaceneighbours, numfniil;
int numstart, numstart_min, numstart_max;
int trace_num;
int startStyle;
int box_flag, opt2, opt2_old;
char *color[10] = {
    "yellow", "green", "blue", "red", "violet", "chocolat",
    "linen", "pink", "crimson", "indigo"
};

const int nosteps = 15;

//  adjacency vars
int cuc_count, *gpt_nbr_count = NULL;
int *cuc = NULL, *cuc_pos = NULL;
int cnbr_count;
int *cnbr = NULL, *cnbr_pos = NULL;
; //cnbr -> cellneighbours

float *x_in, *y_in, *z_in; // positions data
float *u_in, *v_in, *w_in; // vorticity data
float startp[3], startpoint1[3], startpoint2[3]; // startpoint
float startNormal[3], startDirection[3]; // startpoint
float box_length = 0.;

char *dtype, *gtype, *gtype_test; // data type, grid type
char *GridIn, *DataIn;
char *colorn, *startpoint;

//  Shared memory data
coDoVec3 *v_data_in = NULL;
coDoVec3 *uv_data_in = NULL;
DO_CellGrid *grid_in = NULL;

void main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();
}

coDistributedObject **Application::compute(coDistributedObject **in, char **outNames)
{
    coDistributedObject **returnObject = NULL;

    returnObject = new coDistributedObject *[4];
    returnObject[0] = returnObject[1] = returnObject[2] = returnObject[3] = NULL;
    //      Distributed Objects
    coDistributedObject *data_obj;
    coDoLines *lineobj = NULL;
    //coDoFloat*   data_out = NULL;
    //coDoVec3*   vel_out  = NULL;
    coDistributedObject **in_set_elements = NULL;

    //      local variables
    int i, j, k, factor = 1;
    int sx, sy, sz, data_anz, arr_size;
    int no_points, no_vert;
    int t_no_points = 0, t_no_vert = 0, t_no_lines = 0;
    int *vert_list = NULL;
    int *t_vert_list, *t_line_list;
    int direction, *dummy;
    int startcell_count, *startcell, *startp_ind;

    float *x_out = NULL, *y_out = NULL, *z_out = NULL;
    float *t_x_out, *t_y_out, *t_z_out;
    float *s_out, *vx_out, *vy_out, *vz_out;
    float *t_s_out, *t_vx_out, *t_vy_out, *t_vz_out;

    //	double time;

    char *Lines, *DataOut, *VelOut;
    char buf[1000];

    //	struct timeval tp1, tp2;

    //      get input data object names
    Lines = outNames[0];
    DataOut = outNames[1];
    VelOut = outNames[2];

    //      get parameters
    //get choice of creation cnbr_list (1:as stored array; 2:on the fly)
    Covise::get_choice_param("nbrs_comp", &opt2);

    //get number of startpoints
    Covise::get_slider_param("no_startp", &numstart_min, &numstart_max, &numstart);

    //get startpoints
    for (i = 0; i < 3; i++)
    {
        // 1st start point
        Covise::get_vector_param("startpoint1", i, &startpoint1[i]);
        // 2nd start point
        Covise::get_vector_param("startpoint2", i, &startpoint2[i]);
        Covise::get_vector_param("normal", i, &startNormal[i]);
        Covise::get_vector_param("direction", i, &startDirection[i]);
    }
    Covise::get_choice_param("startStyle", &startStyle);
    if (startStyle > 2) // just plane&line supported at the moment
        startStyle = 1;

    //get choice of interpolation method
    Covise::get_choice_param("option", &option);

    //get choice of stepsize control
    Covise::get_choice_param("stp_control", &stp_control);

    //get direction of interpolation
    Covise::get_choice_param("tdirection", &direction);
    intpol_dir = direction;

    //get choice of point reduction on output geometry
    Covise::get_choice_param("reduce", &reduce);

    //get choice of output type
    Covise::get_choice_param("whatout", &data_out_type);

    //get choice of output type
    Covise::get_scalar_param("trace_eps", &trace_eps);
    //get choice of output type
    Covise::get_scalar_param("trace_len", &trace_len);

    //      retrieve grid object from shared memory

    if (DataOut == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for 'dataOut'");
        return (returnObject);
    }

    if (VelOut == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for 'velocOut'");
        return (returnObject);
    }

    grid_in = NULL;
    v_data_in = NULL; //structured data
    uv_data_in = NULL; //unstructured data

    data_obj = in[0];

    if (data_obj != 0L)
    {
        gtype = data_obj->getType(); // get the grid type

        if (strcmp(gtype, "SETELE") == 0)
        {
            in_set_elements = ((coDoSet *)data_obj)->getAllElements(&i);
            data_obj = in_set_elements[0];
            gtype = data_obj->getType();
        }

        if (strcmp(gtype, "CELGRD") == 0)
        {
            grid_in = (DO_CellGrid *)data_obj;
            grid_in->getGridSize(&numcoord, &numedges, &numfaces, &numfniil,
                                 &numfaceneighbours, &numcells, &xglob, &yglob, &zglob);
            grid_in->getAddresses(&x_in, &y_in, &z_in, &cl, &dummy,
                                  &dummy, &dummy, &el, &tl, &xm, &ym, &zm, &gci_l, &gcsc_l, &cigc_l);
            if ((colorn = grid_in->getAttribute("COLOR")) == NULL)
            {
                colorn = new char[20];
                strcpy(colorn, "white");
            }
        }
        else
        {
            Covise::sendError("ERROR: Data object 'meshIn' has wrong data type");
            return (returnObject);
        }
    }
    else
    {
        Covise::sendError("ERROR: Data object 'meshIn' can't be accessed in shared memory");
        return (returnObject);
    }
    if (numcells == 0)
    {
        Covise::sendWarning("WARNING: Data object 'meshIn' is empty");
    }

    data_obj = in[1];
    if (data_obj != 0L)
    {
        dtype = data_obj->getType();

        if (strcmp(dtype, "SETELE") == 0)
        {
            in_set_elements = ((coDoSet *)data_obj)->getAllElements(&i);
            data_obj = in_set_elements[0];
            dtype = data_obj->getType();
        }

        if (strcmp(dtype, "STRVDT") == 0)
        {
            v_data_in = (coDoVec3 *)data_obj;
            v_data_in->getGridSize(&sx, &sy, &sz);
            data_anz = sx * sy * sz;
            v_data_in->getAddresses(&u_in, &v_in, &w_in);
        }
        else if (strcmp(dtype, "USTVDT") == 0)
        {
            uv_data_in = (coDoVec3 *)data_obj;
            data_anz = uv_data_in->getNumPoints();
            uv_data_in->getAddresses(&u_in, &v_in, &w_in);
        }
        else
        {
            Covise::sendError("ERROR: Data object 'dataIn' has wrong data type");
            return (returnObject);
        }
    }
    else
    {
        Covise::sendError("ERROR: Data object 'dataIn' can't be accessed in shared memory");
        return (returnObject);
    }
    if (data_anz == 0)
    {
        Covise::sendWarning("WARNING: Data object 'dataIn' is empty");
    }

    // check dimensions
    if (data_anz != numcells)
    {
        Covise::sendError("ERROR: Data Object's dimension doesn't match Grid ones");
        cerr << "USTVDT: " << data_anz << "  CellGrid: " << numcells << endl;
        return NULL;
    }

    //	Begin tracing
    xStart = new float[numstart];
    yStart = new float[numstart];
    zStart = new float[numstart];
    startcell = new int[numstart];
    startp_ind = new int[numstart];
    startcell_count = 0;
    t_line_list = new int[(numstart * 4) + 1];
    t_line_list[0] = 0;

    if (numstart < 1)
    {
        Covise::sendError("ERROR:  Number of startpoints is < 1");
        return (returnObject);
    }

    computeStartPoints();

    //	find the startcells
    for (i = 0; i < numstart; i++)
    {
        /* this is donw in computeStartPoints();
        if(numstart==1){
         startp[0]=startpoint1[0];
         startp[1]=startpoint1[1];
         startp[2]=startpoint1[2];
      }
      else{
         startp[0]=startpoint1[0]+i*(startpoint2[0]-startpoint1[0])/float(numstart-1);
         startp[1]=startpoint1[1]+i*(startpoint2[1]-startpoint1[1])/float(numstart-1);
         startp[2]=startpoint1[2]+i*(startpoint2[2]-startpoint1[2])/float(numstart-1);
      } */
        startp[0] = xStart[i];
        startp[1] = yStart[i];
        startp[2] = zStart[i];
        //cerr << "Look for startpoint no." << i+1 << endl;
        //find the startcell

        //calling startcell functions
        j = find_startcell_fast(startp);
        if (j == -1)
        {
        }
        else if (j == -2)
        {

            return (returnObject);
        }
        else
        {
            startp_ind[startcell_count] = i;
            startcell[startcell_count++] = j;
        }
    }
    if (startcell_count == 0)
    {
        Covise::sendInfo("INFO:  No startpoints found");
        return (returnObject);
    }
    //cerr << endl << startcell_count << " startpoints found" << endl;

    if (numelem < 1000)
    {
        arr_size = 5 * numelem * nosteps;
    }
    else if (numelem < 5000)
    {
        arr_size = 3 * numelem * nosteps;
    }
    else if (numelem < 20000)
    {
        arr_size = numelem * nosteps;
    }
    else
    {
        arr_size = numelem;
    }

    fprintf(stderr, "arr_size: %d\n", arr_size);
    arr_size = 5000;

    if (direction == 3)
    {
        factor = 2;
    }
    else
    {
        factor = 1;
    }

    t_x_out = new float[arr_size * numstart * factor];
    t_y_out = new float[arr_size * numstart * factor];
    t_z_out = new float[arr_size * numstart * factor];
    t_s_out = new float[arr_size * numstart * factor];
    t_vx_out = new float[arr_size * numstart * factor];
    t_vy_out = new float[arr_size * numstart * factor];
    t_vz_out = new float[arr_size * numstart * factor];
    t_vert_list = new int[arr_size * numstart * factor];
    x_out = new float[arr_size];
    y_out = new float[arr_size];
    z_out = new float[arr_size];
    s_out = new float[arr_size];
    vx_out = new float[arr_size];
    vy_out = new float[arr_size];
    vz_out = new float[arr_size];
    vert_list = new int[arr_size];

    for (i = 0; i < startcell_count; i++)
    {
        if (direction == 3)
        {

            intpol_dir = 2;
            trace_num = i;

            startp[0] = xStart[startp_ind[i]];
            startp[1] = yStart[startp_ind[i]];
            startp[2] = zStart[startp_ind[i]];

            j = startcell[i];

            //compute the streamlines
            tracer(j, startp, &no_points, x_out, y_out, z_out, s_out,
                   vx_out, vy_out, vz_out, &no_vert, vert_list);
            for (k = 0; k < no_points; k++)
            {
                //positions-comp-x
                t_x_out[t_no_points + no_points - 1 - k] = x_out[k];
                //positions-comp-y
                t_y_out[t_no_points + no_points - 1 - k] = y_out[k];
                //positions-comp-z
                t_z_out[t_no_points + no_points - 1 - k] = z_out[k];
                //data-out (scalar)
                t_s_out[t_no_points + no_points - 1 - k] = s_out[k];
                //data-out (vector x-comp)
                t_vx_out[t_no_points + no_points - 1 - k] = vx_out[k];
                //data-out (vector y-comp)
                t_vy_out[t_no_points + no_points - 1 - k] = vy_out[k];
                //data-out (vector z-comp)
                t_vz_out[t_no_points + no_points - 1 - k] = vz_out[k];
            }

            for (k = 0; k < no_vert; k++)
            {
                t_vert_list[t_no_vert + k] = vert_list[k] + t_no_points;
            }
            t_line_list[t_no_lines] = t_no_vert;
            t_no_vert += no_vert;
            t_no_points += no_points;

            t_no_lines += 1;
        }
        trace_num = i;

        startp[0] = xStart[startp_ind[i]];
        startp[1] = yStart[startp_ind[i]];
        startp[2] = zStart[startp_ind[i]];

        j = startcell[i];

        //compute the streamlines
        if (direction == 3)
        {
            intpol_dir = 1;
        }
        tracer(j, startp, &no_points, x_out, y_out, z_out, s_out,
               vx_out, vy_out, vz_out, &no_vert, vert_list);
        if (intpol_dir == 1)
        {
            for (k = 0; k < no_points; k++)
            {
                t_x_out[t_no_points + k] = x_out[k]; //positions-comp-x
                t_y_out[t_no_points + k] = y_out[k]; //positions-comp-y
                t_z_out[t_no_points + k] = z_out[k]; //positions-comp-z
                t_s_out[t_no_points + k] = s_out[k]; //data-out (scalar)
                t_vx_out[t_no_points + k] = vx_out[k]; //data-out (vector x-comp)
                t_vy_out[t_no_points + k] = vy_out[k]; //data-out (vector y-comp)
                t_vz_out[t_no_points + k] = vz_out[k]; //data-out (vector z-comp)
            }
        }
        else
        {
            for (k = 0; k < no_points; k++)
            {
                //positions-comp-x
                t_x_out[t_no_points + no_points - 1 - k] = x_out[k];
                //positions-comp-y
                t_y_out[t_no_points + no_points - 1 - k] = y_out[k];
                //positions-comp-z
                t_z_out[t_no_points + no_points - 1 - k] = z_out[k];
                //data-out (scalar)
                t_s_out[t_no_points + no_points - 1 - k] = s_out[k];
                //data-out (vector x-comp)
                t_vx_out[t_no_points + no_points - 1 - k] = vx_out[k];
                //data-out (vector y-comp)
                t_vy_out[t_no_points + no_points - 1 - k] = vy_out[k];
                //data-out (vector z-comp)
                t_vz_out[t_no_points + no_points - 1 - k] = vz_out[k];
            }
        }

        for (k = 0; k < no_vert; k++)
        {
            t_vert_list[t_no_vert + k] = vert_list[k] + t_no_points;
        }
        if (direction != 3)
        {
            t_line_list[t_no_lines] = t_no_vert;
            t_no_lines += 1;
        }
        t_no_vert += no_vert;
        t_no_points += no_points;
    }

    // by Lars Frenzel
    if (t_no_lines == 1)
    {
        returnObject[0] = lineobj = new coDoLines(Lines, t_no_points, t_x_out, t_y_out, t_z_out, t_no_vert, t_vert_list, t_no_lines, t_line_list);
        returnObject[1] /*=data_out*/ = new coDoFloat(DataOut, t_no_points, t_s_out);
        returnObject[2] /*=vel_out*/ = new coDoVec3(VelOut, t_no_points, t_vx_out, t_vy_out, t_vz_out);

        lineobj->addAttribute("COLOR", "red");
        // and attribute for interaction
        // depending on style !!!
        switch (startStyle)
        {
        case 2: // plane
            sprintf(buf, "P%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
            break;

        default: // line
            sprintf(buf, "T%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
            break;
        }
        lineobj->addAttribute("FEEDBACK", buf);

        //delete	 lineobj;
        //delete	 data_out;
        //delete   	vel_out;
    }
    else
    {
        coDoSet *set_lines_out = NULL;
        //coDoSet *set_data_out = NULL;
        //coDoSet *set_vel_out = NULL;

        coDistributedObject **line_elements = NULL;
        coDistributedObject **data_elements = NULL;
        coDistributedObject **vel_elements = NULL;

        // lines
        float *this_x, *this_y, *this_z;
        int this_num_points, this_num_vert;
        int *this_vl, *this_ll;
        int c;

        // scalar-data
        float *this_scalar;

        // vector data
        float *this_vector_x, *this_vector_y, *this_vector_z;

        // malloc
        line_elements = new coDistributedObject *[t_no_lines + 1];
        data_elements = new coDistributedObject *[t_no_lines + 1];
        vel_elements = new coDistributedObject *[t_no_lines + 1];

        line_elements[t_no_lines] = NULL;
        data_elements[t_no_lines] = NULL;
        vel_elements[t_no_lines] = NULL;

        // we malloc far too much but it doesn't matter
        this_x = new float[t_no_points];
        this_y = new float[t_no_points];
        this_z = new float[t_no_points];

        this_vl = new int[t_no_vert];
        this_ll = new int[1];

        this_scalar = new float[t_no_points];
        this_vector_x = new float[t_no_points];
        this_vector_y = new float[t_no_points];
        this_vector_z = new float[t_no_points];

        // create set of lines instead of linestrips
        for (i = 0; i < t_no_lines; i++)
        {

            // compute current element
            j = t_line_list[i];
            if (i == t_no_lines - 1)
                k = t_no_vert;
            else
                k = t_line_list[i + 1];

            this_ll[0] = 0;
            this_num_vert = 0;
            this_num_points = 0;

            for (c = 0; j < k; j++, c++)
            {
                // line
                this_x[c] = t_x_out[t_vert_list[j]];
                this_y[c] = t_y_out[t_vert_list[j]];
                this_z[c] = t_z_out[t_vert_list[j]];

                this_vl[c] = c;

                // scalar
                this_scalar[c] = t_s_out[t_vert_list[j]];

                // vector
                this_vector_x[c] = t_vx_out[t_vert_list[j]];
                this_vector_y[c] = t_vy_out[t_vert_list[j]];
                this_vector_z[c] = t_vz_out[t_vert_list[j]];

                this_num_vert++;
                this_num_points++;
            }

            sprintf(buf, "%s_%d", Lines, i);
            line_elements[i] = new coDoLines(buf, this_num_points, this_x, this_y, this_z, this_num_vert, this_vl, 1, this_ll);
            line_elements[i]->addAttribute("COLOR", "red");

            sprintf(buf, "%s_%d", DataOut, i);
            data_elements[i] = new coDoFloat(buf, this_num_points, this_scalar);

            sprintf(buf, "%s_%d", VelOut, i);
            vel_elements[i] = new coDoVec3(buf, this_num_points, this_vector_x, this_vector_y, this_vector_z);
        }

        // create sets
        returnObject[0] = set_lines_out = new coDoSet(Lines, line_elements);
        returnObject[1] /*=set_data_out*/ = new coDoSet(DataOut, data_elements);
        returnObject[2] /*=set_vel_out */ = new coDoSet(VelOut, vel_elements);

        // attributes
        set_lines_out->addAttribute("COLOR", "red");
        // and attribute for interaction
        // depending on style !!!
        switch (startStyle)
        {
        case 2: // plane
            sprintf(buf, "P%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
            break;

        default: // line
            sprintf(buf, "T%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
            break;
        }
        set_lines_out->addAttribute("FEEDBACK", buf);

        // clean sets
        //delete set_lines_out;
        //delete set_data_out;
        //delete set_vel_out;

        // clean up
        for (i = 0; i < t_no_lines; i++)
        {
            delete line_elements[i];
            delete data_elements[i];
            delete vel_elements[i];
        }
        delete[] line_elements;
        delete[] data_elements;
        delete[] vel_elements;

        delete[] this_x;
        delete[] this_y;
        delete[] this_z;

        delete[] this_vl;
        delete[] this_ll;

        delete[] this_scalar;
        delete[] this_vector_x;
        delete[] this_vector_y;
        delete[] this_vector_z;
    }

    delete data_obj;

    delete[] vert_list;
    delete[] x_out;
    delete[] y_out;
    delete[] z_out;
    delete[] s_out;
    delete[] vx_out;
    delete[] vy_out;
    delete[] vz_out;

    delete[] t_vert_list;
    delete[] t_line_list;
    delete[] t_x_out;
    delete[] t_y_out;
    delete[] t_z_out;
    delete[] t_s_out;
    delete[] t_vx_out;
    delete[] t_vy_out;
    delete[] t_vz_out;
    delete[] xStart;
    delete[] yStart;
    delete[] zStart;
    delete[] startcell;
    delete[] startp_ind;
    return (returnObject);
}

//=====================================================================
// compare function for qsort library function
//=====================================================================
int compare(const void *v1, const void *v2)
{

    if (*((int *)v1) < *((int *)v2))
        return -1;
    else if (*((int *)v1) == *((int *)v2))
        return 0;
    else
        return 1;
}

//=====================================================================
// find the actualcell (sequentiell)
//=====================================================================
int find_actualcell(int base_cell, float *head)
{
    int i, j, k, l, neighbours[5000], nb_count; //no_of_neigh,
    //int tmp_el[5], tmp_cl[20];
    float xmin, xmax, ymin, ymax, zmin, zmax;
    int xval, yval, zval, rest;
    int global_base_cell_no; //, gx, gy, gz;
    //float p0[3], p1[3], p2[3], p3[3];
    //actual_cell_type=1;
    // look for cell
    if (opt2 != 2)
    {
        Covise::sendError("ERROR:  Wrong option");
        return -2;
        //	exit(0);
    }
    else
    {

        global_base_cell_no = cigc_l[base_cell];

        zval = global_base_cell_no / (xglob * yglob);
        rest = global_base_cell_no % (xglob * yglob);
        yval = rest / xglob;
        xval = rest % xglob;

        nb_count = 0;
        for (i = -1; i < 2; i++)
        {
            for (j = -1; j < 2; j++)
            {
                for (k = -1; k < 2; k++)
                {
                    int neighbour_no = (xval + i) + (yval + j) * xglob + (zval + k) * xglob * yglob;
                    int fn = gci_l[neighbour_no];
                    int ln = gci_l[neighbour_no + 1];
                    for (l = fn; l < ln; l++)
                    {
                        neighbours[nb_count] = gcsc_l[l];
                        nb_count++;
                    }
                }
            }
        }

        for (i = 0; i < nb_count; i++)
        {
            xmin = x_in[cl[el[neighbours[i]]]];
            xmax = x_in[cl[el[neighbours[i]]] + 1];
            ymin = y_in[cl[el[neighbours[i]]]];
            ymax = y_in[cl[el[neighbours[i]]] + 4];
            zmin = z_in[cl[el[neighbours[i]]]];
            zmax = z_in[cl[el[neighbours[i]]] + 2];
            if ((xmin < head[0]) && (head[0] < xmax) && (ymin < head[1]) && (head[1] < ymax) && (zmin < head[2]) && (head[2] < zmax))
                return neighbours[i];
        }
    }
    return -1;
}

//=====================================================================
//compute the stepsize at a point
//=====================================================================
float stepsize(int cell_no)
{
    float dt, vq, vol;
    //float v[3];
    float xd, yd, zd;

    vq = sqrt(sqr(u_in[cell_no]) + sqr(v_in[cell_no]) + sqr(w_in[cell_no]));

    xd = fabsf(x_in[cl[el[cell_no]] + 1] - x_in[cl[el[cell_no]]]);
    yd = fabsf(y_in[cl[el[cell_no]] + 4] - y_in[cl[el[cell_no]]]);
    zd = fabsf(z_in[cl[el[cell_no]] + 2] - z_in[cl[el[cell_no]]]);
    vol = xd * yd * zd;

#ifdef __sgi
    dt = 1. / vq * powf(vol, 1. / 3.) / 10.;
#else
    dt = 1. / vq * pow(vol, 1. / 3.) / 10.;
#endif

    return dt;
}

//=====================================================================
// is the point inside the tetrahedra cell
//=====================================================================
int isin_cell(float px[3], int cell_no)
{
    float xmin, xmax, ymin, ymax, zmin, zmax;

    xmin = x_in[cl[el[cell_no]]];
    xmax = x_in[cl[el[cell_no]] + 1];
    ymin = y_in[cl[el[cell_no]]];
    ymax = y_in[cl[el[cell_no]] + 4];
    zmin = z_in[cl[el[cell_no]]];
    zmax = z_in[cl[el[cell_no]] + 2];
    if ((xmin < px[0]) && (px[0] < xmax) && (ymin < px[1]) && (px[1] < ymax) && (zmin < px[2]) && (px[2] < zmax))
        return 1;
    else
        return 0;
}

//=====================================================================
// 2nd order Runge Kutta function (Heun's scheme)
//=====================================================================
void rk2_tet(float px[3], float v[3], float dt, float phead[3])
{
    //computes the new headposition for the streamline (phead[x|y|z])
    phead[0] = px[0] + dt * v[0];
    phead[1] = px[1] + dt * v[1];
    phead[2] = px[2] + dt * v[2];

    return;
}

//=====================================================================
// 4th order Runge Kutta function
//=====================================================================
void rk4_tet(float px[3], float v[3], float dt, float phead[3])
{
    //computes the new headposition for the streamline (phead[x|y|z])
    phead[0] = px[0] + dt * v[0];
    phead[1] = px[1] + dt * v[1];
    phead[2] = px[2] + dt * v[2];

    return;
}

//=====================================================================
// compute a particle trace
//=====================================================================
void tracer(int start, float startp[3], int *no_points, float *x_out, float *y_out, float *z_out,
            float *v_out, float *v_x_out, float *v_y_out, float *v_z_out, int *no_vert, int *vert_list)
{
    int j, is_in_domain, is_in_cell; //i, is_in_subdomain,
    //int tmp_el[5], tmp_cl[20], tmp_count;
    int actual_cell;
    int flag, arr_size;

    float px[3], phead[3], pn_1[3], p2n_1[3], p2x[3];
    float v[3], tmp_v[3];

    float dt, dt_min, dt_max;

    float curr_len = 0.0;
    if (numelem < 1000)
    {
        arr_size = 5 * numelem * nosteps;
    }
    else if (numelem < 5000)
    {
        arr_size = 3 * numelem * nosteps;
    }
    else if (numelem < 20000)
    {
        arr_size = numelem * nosteps;
    }
    else
    {
        arr_size = numelem;
    }
    arr_size = 5000;

    actual_cell = start;
    //actual_cell_type = 1;

    // add startpoint
    x_out[0] = px[0] = p2x[0] = startp[0];
    y_out[0] = px[1] = p2x[1] = startp[1];
    z_out[0] = px[2] = p2x[2] = startp[2];
    *no_points = 0;
    *no_vert = 0;

    //while head of trace is inside domain
    is_in_domain = 1;
    while ((is_in_domain == 1) && (curr_len < trace_len))
    {
        is_in_cell = 1;
        while ((is_in_cell == 1) && (curr_len < trace_len))
        {
            if (is_in_cell == 1)
            {
                v[0] = u_in[actual_cell];
                v[1] = v_in[actual_cell];
                v[2] = w_in[actual_cell];

                if (intpol_dir == 2)
                {
                    for (j = 0; j < 3; j++)
                    {
                        v[j] *= -1;
                    }
                }

                //compute the initial stepsize for integration
                if (*no_points == 0)
                {
                    dt = stepsize(actual_cell);
                }

                //compute the new position
                switch (option)
                {
                case 1:
                {
                    flag = 0;
                    while (flag < 1)
                    {
                        rk2_tet(px, v, dt, phead);
                        if ((px[0] == phead[0]) && (px[1] == phead[1]) && (px[2] == phead[2]))
                            return;

                        if (stp_control == 1 || *no_points <= 1)
                        {
                            dt = stepsize(actual_cell);
                            flag = 2;
                        }
                        else if (stp_control == 2 && *no_points > 1)
                        {
                            // < 5 degrees
                            if (p_ang(pn_1, px, phead) > .9961)
                            {
                                dt *= 1.8;
                                dt_max = 3. * stepsize(actual_cell);
                                dt = (dt > dt_max ? dt_max : dt);
                                flag = 2;
                            } // > 12 degrees
                            else if (p_ang(pn_1, px, phead) < .978)
                            {
                                dt /= 2.;
                                dt_min = .1 * stepsize(actual_cell);
                                if (dt < dt_min)
                                {
                                    dt = dt_min;
                                    flag = 2;
                                }
                                else
                                {
                                    flag = 0;
                                }
                            }
                            else
                                flag = 2;
                        }
                        else if (stp_control == 3 && *no_points > 1)
                        {
                            // < 2 degrees
                            if (p_ang(pn_1, px, phead) > .99939)
                            {
                                dt *= 1.8;
                                dt_max = 2. * stepsize(actual_cell);
                                dt = (dt > dt_max ? dt_max : dt);
                                flag = 2;
                            } // > 6 degrees
                            else if (p_ang(pn_1, px, phead) < .9945)
                            {
                                dt /= 2.;
                                dt_min = .01 * stepsize(actual_cell);
                                if (dt < dt_min)
                                {
                                    dt = dt_min;
                                    flag = 2;
                                }
                                else
                                {
                                    flag = 0;
                                }
                            }
                            else
                                flag = 2;
                        }
                    }
                }
                break;
                default:
                {
                    flag = 0;
                    while (flag < 1)
                    {
                        rk4_tet(px, v, dt, phead);
                        if ((px[0] == phead[0]) && (px[1] == phead[1]) && (px[2] == phead[2]))
                            return;
                        if (stp_control == 1 || *no_points <= 1)
                        {
                            dt = stepsize(actual_cell);
                            flag = 2;
                        }
                        else if (stp_control == 2 && *no_points > 1)
                        {
                            // < 5 degrees
                            if (p_ang(pn_1, px, phead) > .9961)
                            {
                                dt *= 1.8;
                                dt_max = 3. * stepsize(actual_cell);
                                dt = (dt > dt_max ? dt_max : dt);
                                flag = 2;
                            } // > 12 degrees
                            else if (p_ang(pn_1, px, phead) < .978)
                            {
                                dt /= 2.;
                                dt_min = .1 * stepsize(actual_cell);
                                if (dt < dt_min)
                                {
                                    dt = dt_min;
                                    flag = 2;
                                }
                                else
                                {
                                    flag = 0;
                                }
                            }
                            else
                                flag = 2;
                        }
                        else if (stp_control == 3 && *no_points > 1)
                        {
                            // < 2 degrees
                            if (p_ang(pn_1, px, phead) > .99939)
                            {
                                dt *= 1.8;
                                flag = 2;
                            } // > 6 degrees
                            else if (p_ang(pn_1, px, phead) < .9945)
                            {
                                dt /= 2.;
                                dt_min = .01 * stepsize(actual_cell);
                                if (dt < dt_min)
                                {
                                    dt = dt_min;
                                    flag = 2;
                                }
                                else
                                {
                                    flag = 0;
                                }
                            }
                            else
                                flag = 2;
                        }
                    }
                }
                break;
                }

                curr_len += sqrt((px[0] - phead[0]) * (px[0] - phead[0]) + (px[1] - phead[1]) * (px[1] - phead[1]) + (px[2] - phead[2]) * (px[2] - phead[2]));

                // write the new position to tmp arrays
                // reduce cellpoints for output
                if ((reduce == 2 || reduce == 3) && *no_points >= 1)
                {
                    if (reduce == 2)
                    {
                        if (p_ang(p2n_1, p2x, phead) < .9995)
                        {
                            pn_1[0] = px[0];
                            pn_1[1] = px[1];
                            pn_1[2] = px[2];

                            p2n_1[0] = p2x[0];
                            p2n_1[1] = p2x[1];
                            p2n_1[2] = p2x[2];

                            x_out[*no_points] = px[0];
                            y_out[*no_points] = px[1];
                            z_out[*no_points] = px[2];
                            tmp_v[0] = u_in[actual_cell];
                            tmp_v[1] = v_in[actual_cell];
                            tmp_v[2] = w_in[actual_cell];
                            v_out[*no_points] = dout(tmp_v);
                            v_x_out[*no_points] = tmp_v[0];
                            v_y_out[*no_points] = tmp_v[1];
                            v_z_out[*no_points] = tmp_v[2];
                            px[0] = p2x[0] = phead[0];
                            px[1] = p2x[1] = phead[1];
                            px[2] = p2x[2] = phead[2];
                            (*no_points)++;

                            vert_list[*no_vert] = *no_vert;
                            (*no_vert)++;
                            if (*no_points >= arr_size)
                            {
                                cerr << "arr_size: " << arr_size << " < " << *no_points << endl;
                                Covise::sendInfo("INFO: number of points exceed allocated array length (numelem*nosteps)");
                                return;
                            }
                        }
                        else
                        {
                            pn_1[0] = px[0];
                            pn_1[1] = px[1];
                            pn_1[2] = px[2];

                            px[0] = phead[0];
                            px[1] = phead[1];
                            px[2] = phead[2];
                        }
                    }
                    else
                    {
                        if (p_ang(p2n_1, p2x, phead) < .9986)
                        {
                            pn_1[0] = px[0];
                            pn_1[1] = px[1];
                            pn_1[2] = px[2];

                            p2n_1[0] = p2x[0];
                            p2n_1[1] = p2x[1];
                            p2n_1[2] = p2x[2];

                            x_out[*no_points] = px[0];
                            y_out[*no_points] = px[1];
                            z_out[*no_points] = px[2];
                            tmp_v[0] = u_in[actual_cell];
                            tmp_v[1] = v_in[actual_cell];
                            tmp_v[2] = w_in[actual_cell];
                            v_out[*no_points] = dout(tmp_v);
                            v_x_out[*no_points] = tmp_v[0];
                            v_y_out[*no_points] = tmp_v[1];
                            v_z_out[*no_points] = tmp_v[2];
                            px[0] = p2x[0] = phead[0];
                            px[1] = p2x[1] = phead[1];
                            px[2] = p2x[2] = phead[2];
                            (*no_points)++;

                            vert_list[*no_vert] = *no_vert;
                            (*no_vert)++;
                            if (*no_points >= arr_size)
                            {
                                cerr << "arr_size: " << arr_size << " < " << *no_points << endl;
                                Covise::sendInfo("INFO: number of points exceed allocated array length (numelem*nosteps)");
                                return;
                            }
                        }
                        else
                        {
                            pn_1[0] = px[0];
                            pn_1[1] = px[1];
                            pn_1[2] = px[2];

                            px[0] = phead[0];
                            px[1] = phead[1];
                            px[2] = phead[2];
                        }
                    }
                }
                else // write direct to output
                {
                    pn_1[0] = p2n_1[0] = px[0];
                    pn_1[1] = p2n_1[1] = px[1];
                    pn_1[2] = p2n_1[2] = px[2];

                    x_out[*no_points] = px[0];
                    y_out[*no_points] = px[1];
                    z_out[*no_points] = px[2];
                    tmp_v[0] = u_in[actual_cell];
                    tmp_v[1] = v_in[actual_cell];
                    tmp_v[2] = w_in[actual_cell];
                    v_out[*no_points] = dout(tmp_v);
                    v_x_out[*no_points] = tmp_v[0];
                    v_y_out[*no_points] = tmp_v[1];
                    v_z_out[*no_points] = tmp_v[2];
                    px[0] = p2x[0] = phead[0];
                    px[1] = p2x[1] = phead[1];
                    px[2] = p2x[2] = phead[2];
                    (*no_points)++;

                    vert_list[*no_vert] = *no_vert;
                    (*no_vert)++;
                }

                is_in_cell = isin_cell(phead, actual_cell);
                if (*no_points >= arr_size)
                {
                    cerr << "arr_size: " << arr_size << " < " << *no_points << endl;
                    Covise::sendInfo("INFO: number of points exceed allocated array length (numelem*nosteps)");
                    return;
                }
            }
        }
        int tryit = 0;
        int this_cell = actual_cell;
        // look for neighbors in cnbr_list
        while (((actual_cell = find_actualcell(this_cell, phead)) == -1) && (tryit < 5))
        {
            if (tryit == 0)
            {
                phead[0] = pn_1[0] + (px[0] - pn_1[0]) * 0.5;
                phead[1] = pn_1[1] + (px[1] - pn_1[1]) * 0.5;
                phead[2] = pn_1[2] + (px[2] - pn_1[2]) * 0.5;
            }
            if (tryit == 1)
            {
                phead[0] = pn_1[0] + (px[0] - pn_1[0]) * 0.25;
                phead[1] = pn_1[1] + (px[1] - pn_1[1]) * 0.25;
                phead[2] = pn_1[2] + (px[2] - pn_1[2]) * 0.25;
            }
            if (tryit == 2)
            {
                phead[0] = pn_1[0] + (px[0] - pn_1[0]) * 0.01;
                phead[1] = pn_1[1] + (px[1] - pn_1[1]) * 0.01;
                phead[2] = pn_1[2] + (px[2] - pn_1[2]) * 0.01;
            }
            if (tryit == 3)
            {
                phead[0] = pn_1[0] + (px[0] - pn_1[0]) * 1.5;
                phead[1] = pn_1[1] + (px[1] - pn_1[1]) * 1.5;
                phead[2] = pn_1[2] + (px[2] - pn_1[2]) * 1.5;
            }
            if (tryit == 4)
            {
                phead[0] = pn_1[0] + (px[0] - pn_1[0]) * 0.001;
                phead[1] = pn_1[1] + (px[1] - pn_1[1]) * 0.001;
                phead[2] = pn_1[2] + (px[2] - pn_1[2]) * 0.001;
            }
            //Covise::sendInfo("INFO: retried");
            tryit++;
        }
        if (actual_cell < 0)
        {
            phead[0] = px[0];
            phead[1] = px[1];
            phead[2] = px[2];
            actual_cell = find_startcell_fast(px);
            fprintf(stderr, "Restart Cell: %d\n", this_cell);
        }

        if (actual_cell < 0)
        {
            //if reduction was on an last point has not been written

            if ((x_out[*(no_points)-1] != px[0]) && (y_out[*(no_points)-1] != px[1]) && (z_out[*(no_points)-1] != px[2]))
            {
                x_out[*no_points] = px[0];
                y_out[*no_points] = px[1];
                z_out[*no_points] = px[2];
                tmp_v[0] = u_in[actual_cell];
                tmp_v[1] = v_in[actual_cell];
                tmp_v[2] = w_in[actual_cell];
                v_out[*no_points] = dout(tmp_v);
                v_x_out[*no_points] = tmp_v[0];
                v_y_out[*no_points] = tmp_v[1];
                v_z_out[*no_points] = tmp_v[2];
                px[0] = p2x[0] = phead[0];
                px[1] = p2x[1] = phead[1];
                px[2] = p2x[2] = phead[2];
                (*no_points)++;

                vert_list[*no_vert] = *no_vert;
                (*no_vert)++;
            }

            return;
        }
        //if(actual_cell == -2){
        //    Covise::sendError("ERROR: unsupported grid type detected, stop computation");
        //    return;
        //}
        else
        {
            is_in_domain = 1;
        }
    }
    return;
}

//=====================================================================
//print adjacency information
//=====================================================================
void print_adjacency()
{
    int i, j, k;
    FILE *file1 = fopen("adjacency.out", "w");

    fprintf(file1, "\nZellnachbarn:");
    for (i = 0; i < numelem; i++)
    {
        fprintf(file1, "\nZelle[%d]: ", i);
        k = 0;
        for (j = cnbr_pos[i]; j < cnbr_pos[i + 1]; j++)
        {
            fprintf(file1, "%d, ", cnbr[j]);
            k++;
        }
        fprintf(file1, "\nSumme: %d", k);
        fprintf(file1, "\n");
    }
    fclose(file1);
}

//=====================================================================
//returns the cos(angle) from the deviation of the pathline between 2 steps
//=====================================================================
double p_ang(float p0[3], float p1[3], float p2[3])
{
    double cos_tn;

    cos_tn = ((double)(p1[0] - p0[0]) * (double)(p2[0] - p1[0]) + (double)(p1[1] - p0[1]) * (double)(p2[1] - p1[1]) + (double)(p1[2] - p0[2]) * (double)(p2[2] - p1[2])) / (sqrt(sqr((double)(p1[0] - p0[0])) + sqr((double)(p1[1] - p0[1])) + sqr((double)(p1[2] - p0[2]))) * sqrt(sqr((double)(p2[0] - p1[0])) + sqr((double)(p2[1] - p1[1])) + sqr((double)(p2[2] - p1[2]))));

    return cos_tn;
}

//=====================================================================
//returns the scalar of chosen vorticity
//=====================================================================
float dout(float t[3])
{
    float s;

    switch (data_out_type)
    {

    case 1:
    {
        s = sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
    }
    break;

    case 2:
    {
        s = t[0];
    }
    break;

    case 3:
    {
        s = t[1];
    }
    break;

    case 4:
    {
        s = t[2];
    }
    break;

    case 5:
    {
        s = trace_num;
    }
    break;

    default:
    {
        Covise::sendInfo("Info: Choice of whatout incorrect -> DataOut = 0");
        s = 0;
    }
    }

    return s;
}

//=====================================================================
//return the minimum of a float array
//=====================================================================
float Min_of(float array[], int length)
{
    int i;
    float val = array[0];

    for (i = 1; i < length; i++)
    {
        val = (val <= array[i]) ? val : array[i];
    }
    return val;
}

//=====================================================================
//return the maximum of a float array
//=====================================================================
float Max_of(float array[], int length)
{
    int i;
    float val = array[0];

    for (i = 1; i < length; i++)
    {
        val = (val >= array[i]) ? val : array[i];
    }
    return val;
}

//=====================================================================
// find the startcell (fast)
//=====================================================================
int find_startcell_fast(float *p)
{
    int i, j, k, cell_no;
    //int tmp_el[5], tmp_cl[20];
    //int *tmp_inbox, tmp_inbox_count, *inbox, inbox_count;
    //int *pib, pib_count;

    //float p0[3], p1[3], p2[3], p3[3];
    //   float tp[3];
    //float xb[8], yb[8], zb[8];
    //float tmp_length[3];
    float xmin, xmax, ymin, ymax, zmin, zmax;

    //float factor = 2;

    i = 0;
    while ((p[0] > xm[i]) && (i < (xglob + 1)))
        i++;

    i--;
    if (i == (xglob + 1))
        return -1;

    j = 0;
    while ((p[1] > ym[j]) && (j < (yglob + 1)))
        j++;

    j--;
    if (j == (yglob + 1))
        return -1;

    k = 0;
    while ((p[2] > zm[k]) && (k < (zglob + 1)))
        k++;

    k--;
    if (k == (zglob + 1))
        return -1;

    // now we know that p is in the global cell (i,j,k)

    cell_no = i + j * xglob + k * xglob * yglob;

    int fn = gci_l[cell_no];
    int ln = gci_l[cell_no + 1];

    for (i = fn; i < ln; i++)
    {
        xmin = x_in[cl[el[gcsc_l[i]]]];
        xmax = x_in[cl[el[gcsc_l[i]]] + 1];
        ymin = y_in[cl[el[gcsc_l[i]]]];
        ymax = y_in[cl[el[gcsc_l[i]]] + 4];
        zmin = z_in[cl[el[gcsc_l[i]]]];
        zmax = z_in[cl[el[gcsc_l[i]]] + 2];
        if ((xmin < p[0]) && (p[0] < xmax) && (ymin < p[1]) && (p[1] < ymax) && (zmin < p[2]) && (p[2] < zmax))
            return gcsc_l[i];
    }

    return -1;
}

void Application::computeStartPoints()
{
    int i, j;
    int n0, n1;
    float v[3], o[3];
    float v0[3], v1[3];

    float myDir[3];

    float s, r;
    float s0, s1;
    // float sD2;

    o[0] = startpoint1[0];
    o[1] = startpoint1[1];
    o[2] = startpoint1[2];

    v[0] = startpoint2[0] - o[0];
    v[1] = startpoint2[1] - o[1];
    v[2] = startpoint2[2] - o[2];

    s = 1.0 / (float)(numstart - 1);

    // here we go
    switch (startStyle)
    {

    case 2: // plane
    {
        // given:  startpoint1, startpoint2, startDirection, startNormal
        //
        // we have to compute: s0, s1 first, then n0, n1
        //

        // normize all vectors first
        r = 1.0 / sqrt((startDirection[0] * startDirection[0]) + (startDirection[1] * startDirection[1]) + (startDirection[2] * startDirection[2]));
        startDirection[0] *= r;
        startDirection[1] *= r;
        startDirection[2] *= r;

        r = 1.0 / sqrt((startNormal[0] * startNormal[0]) + (startNormal[1] * startNormal[1]) + (startNormal[2] * startNormal[2]));
        startNormal[0] *= r;
        startNormal[1] *= r;
        startNormal[2] *= r;

        myDir[0] = startNormal[1] * startDirection[2] - startNormal[2] * startDirection[1];
        myDir[1] = startNormal[2] * startDirection[0] - startNormal[0] * startDirection[2];
        myDir[2] = startNormal[0] * startDirection[1] - startNormal[1] * startDirection[0];
        // that should be normized by definition, so we have nothing to do

        // the following algorithm is standard, though it is taken
        // from GraphicGems I, side 304
        //

        // v1 x v2 = v0  (for speed up)
        v0[0] = startDirection[1] * myDir[2] - startDirection[2] * myDir[1];
        v0[1] = startDirection[2] * myDir[0] - startDirection[0] * myDir[2];
        v0[2] = startDirection[0] * myDir[1] - startDirection[1] * myDir[0];

        // v1 = p2 - p1
        v1[0] = startpoint2[0] - startpoint1[0];
        v1[1] = startpoint2[1] - startpoint1[1];
        v1[2] = startpoint2[2] - startpoint1[2];

        // compute r = det(v1, myDir, v0)
        r = (v1[0] * myDir[1] * v0[2]) + (myDir[0] * v0[1] * v1[2]) + (v0[0] * v1[1] * myDir[2]) - (v0[0] * myDir[1] * v1[2]) - (v1[0] * v0[1] * myDir[2]) - (myDir[0] * v1[1] * v0[2]);

        // and divide by v0^2
        s0 = (v0[0] * v0[0]) + (v0[1] * v0[1]) + (v0[2] * v0[2]);
        if (s0 == 0.0)
            fprintf(stderr, "s0 = 0.0 !!!\n");

        r /= s0;

        // so upper left point is at
        s1 = -r; // ulp = startpoint1 + s1*myDir

        // and compute r = det(v1, startDirection, v0)
        r = (v1[0] * startDirection[1] * v0[2]) + (startDirection[0] * v0[1] * v1[2]) + (v0[0] * v1[1] * startDirection[2]) - (v0[0] * startDirection[1] * v1[2]) - (v1[0] * v0[1] * startDirection[2]) - (startDirection[0] * v1[1] * v0[2]);
        r /= s0;

        // so lower right point is at
        s0 = r; // lrp = startpoint1 + s0*startDirection

        // we flipped 'em somehow....damn
        r = s0;
        s0 = -s1;
        s1 = -r;

        fprintf(stderr, "s0: %19.16f   s1: %19.16f\n", s0, s1);

        // now compute "stepsize" and "stepcount"

        n1 = (int)sqrt(fabsf(((float)numstart) * (s1 / s0)));
        n0 = numstart / n1;

        s0 /= ((float)n0);
        s1 /= ((float)n1);

        v0[0] = startDirection[0];
        v0[1] = startDirection[1];
        v0[2] = startDirection[2];

        v1[0] = myDir[0];
        v1[1] = myDir[1];
        v1[2] = myDir[2];

        fprintf(stderr, "numStart will be %d\n", n1 * n0);

        numstart = n1 * n0;
        //numStartX = n0;
        //numStartY = n1;

        /*

         //s0 = ((

         s0 = s;
         s1 = s;

         n0 = (int)sqrt((float)numStart);
         n1 = n0;

         v0[0] = 0.0;
         v0[1] = 0.0;
         v0[2] = 1.0;
         v1[0] = 0.0;
         v1[1] = 1.0;
         v1[2] = 0.0;

         */

        for (j = 0; j < n1; j++)
            for (i = 0; i < n0; i++)
            {
                xStart[i + (j * n0)] = o[0] + v0[0] * s0 * (float)i + v1[0] * s1 * (float)j;
                yStart[i + (j * n0)] = o[1] + v0[1] * s0 * (float)i + v1[1] * s1 * (float)j;
                zStart[i + (j * n0)] = o[2] + v0[2] * s0 * (float)i + v1[2] * s1 * (float)j;
            }
    }
    break;

    default: // connecting line between both startpoints
    {
        for (i = 0; i < numstart; i++)
        {
            xStart[i] = o[0] + v[0] * s * (float)i;
            yStart[i] = o[1] + v[1] * s * (float)i;
            zStart[i] = o[2] + v[2] * s * (float)i;
        }
    }
    break;
    }

    // done
    return;
}
