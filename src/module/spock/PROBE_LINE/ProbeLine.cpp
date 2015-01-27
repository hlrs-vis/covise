/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: dieses spitzen Modul wertet eine Flaeche    aus und       **
 **              erzeugt interessante output-Objekte                       **
 **                                                                        **
 **              WICHTIG: Multiblock-Objekte mit TIMESTEPs werden          **
 **                       nicht unterstuetzt, koennen aber evtl.           **
 **                       kompliziert erweitert werden.                    **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Lars Frenzel                                **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  14.09.97                                                        **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include <iostream.h>
#include <stdlib.h>
#include "HandleSet.h"

////// our class
class Application : public Covise_Set_Handler
{
private:
    int to_remove_vertex(float, float, float);
    float scalar_product(float *, float *);
    float line_with_plane(float *, float *, float **);
    float vert_dist_sqr(float *, float *);

    // definitions
    struct line_element
    {
        float a[3];
        float b[3];
        float a_data, b_data;
        line_element *last;
    };
    struct value_element
    {
        float x1;
        float y1;
        float x2;
        float y2;
        int flag;
    };

    // required variables
    line_element *found_lines;
    int num_lines_found;
    value_element *values_2d;
    value_element *sorted_values;
    int num_sorted_values;

    // some tools
    void add_line_element(float *, float, float *, float);
    void sort_line(void);
    void linear_element_sort(int);
    int element_seek(float, float, int, int, float *, float *);
    int find_next_unhandled(float x, float y);

public:
    /// Input Parameters

    // computed Input Parameters
    float BASE_POINT[3];
    float CUT_NORMAL[3];
    float X_DIRECTION[3];
    float Y_DIRECTION[3];

    /// Methods
    void Reset(void);
    void Compute_No_Output(coDistributedObject **, int);
    void FinalCompute(char **);
};

////// COVISE Callback
void myCallback(void *, void *)
{
    Application App;
    int i;
    float len;
    float vertex[3], normal[3], direction[3];
    int mode_param;

    char *inportnames[] = { "poly_in", "data_in" };
    char *outportnames[] = { "data2d_out", "plane_out", "lines_out", "probe_line_out" };

    // get parameters first
    for (i = 0; i < 3; i++)
    {
        Covise::get_vector_param("position", i, &(vertex[i]));
        Covise::get_vector_param("normal", i, &(normal[i]));
        Covise::get_vector_param("normal2", i, &(direction[i]));
    }
    Covise::get_choice_param("mode", &mode_param);

    // compute input parameters
    switch (mode_param)
    { // x
    case 2:
        normal[0] = 0;
        normal[1] = 0;
        normal[2] = -1;
        direction[0] = 1;
        direction[1] = 0;
        direction[2] = 0;
        break;
    case 3: // y
        normal[0] = 0;
        normal[1] = 0;
        normal[2] = -1;
        direction[0] = 0;
        direction[1] = 1;
        direction[2] = 0;
        break;
    case 4: // z
        normal[0] = 0;
        normal[1] = -1;
        normal[2] = 0;
        direction[0] = 0;
        direction[1] = 0;
        direction[2] = 1;
        break;
    }
    // error check
    if ((normal[0] == 0) && (normal[1] == 0) && (normal[2] == 0))
    {
        Covise::sendError("ERROR: no normal given");
        return;
    }
    if ((direction[0] == 0) && (direction[1] == 0) && (direction[2] == 0))
    {
        Covise::sendError("ERROR: no x-direction given");
        return;
    }

    App.CUT_NORMAL[0] = normal[0];
    App.CUT_NORMAL[1] = normal[1];
    App.CUT_NORMAL[2] = normal[2];

    App.BASE_POINT[0] = vertex[0];
    App.BASE_POINT[1] = vertex[1];
    App.BASE_POINT[2] = vertex[2];

    App.X_DIRECTION[0] = direction[0];
    App.X_DIRECTION[1] = direction[1];
    App.X_DIRECTION[2] = direction[2];

    // normize required vectors
    len = App.CUT_NORMAL[0] * App.CUT_NORMAL[0] + App.CUT_NORMAL[1] * App.CUT_NORMAL[1] + App.CUT_NORMAL[2] * App.CUT_NORMAL[2];
    len = 1 / sqrt(len);
    App.CUT_NORMAL[0] *= len;
    App.CUT_NORMAL[1] *= len;
    App.CUT_NORMAL[2] *= len;

    len = App.X_DIRECTION[0] * App.X_DIRECTION[0] + App.X_DIRECTION[1] * App.X_DIRECTION[1] + App.X_DIRECTION[2] * App.X_DIRECTION[2];
    len = 1 / sqrt(len);
    App.X_DIRECTION[0] *= len;
    App.X_DIRECTION[1] *= len;
    App.X_DIRECTION[2] *= len;

    // project X_DIRECTION onto the plane (with normal CUT_NORMAL)
    len = App.CUT_NORMAL[0] * App.X_DIRECTION[0] + App.CUT_NORMAL[1] * App.X_DIRECTION[1] + App.CUT_NORMAL[2] * App.X_DIRECTION[2];
    App.X_DIRECTION[0] = App.X_DIRECTION[0] - len * App.CUT_NORMAL[0];
    App.X_DIRECTION[1] = App.X_DIRECTION[1] - len * App.CUT_NORMAL[1];
    App.X_DIRECTION[2] = App.X_DIRECTION[2] - len * App.CUT_NORMAL[2];

    // compute Y_DIRECTION as vector-product of X_DIRECTION and CUT_NORMAL
    App.Y_DIRECTION[0] = App.X_DIRECTION[1] * App.CUT_NORMAL[2] - App.CUT_NORMAL[1] * App.X_DIRECTION[2];
    App.Y_DIRECTION[1] = App.X_DIRECTION[2] * App.CUT_NORMAL[0] - App.CUT_NORMAL[2] * App.X_DIRECTION[0];
    App.Y_DIRECTION[2] = App.X_DIRECTION[0] * App.CUT_NORMAL[1] - App.CUT_NORMAL[0] * App.X_DIRECTION[1];

    // we might have to change x/y once again
    if (mode_param == 1)
    {
        App.X_DIRECTION[0] = App.Y_DIRECTION[0];
        App.X_DIRECTION[1] = App.Y_DIRECTION[1];
        App.X_DIRECTION[2] = App.Y_DIRECTION[2];

        // compute Y_DIRECTION as vector-product of X_DIRECTION and CUT_NORMAL
        App.Y_DIRECTION[0] = App.X_DIRECTION[1] * App.CUT_NORMAL[2] - App.CUT_NORMAL[1] * App.X_DIRECTION[2];
        App.Y_DIRECTION[1] = App.X_DIRECTION[2] * App.CUT_NORMAL[0] - App.CUT_NORMAL[2] * App.X_DIRECTION[0];
        App.Y_DIRECTION[2] = App.X_DIRECTION[0] * App.CUT_NORMAL[1] - App.CUT_NORMAL[0] * App.X_DIRECTION[1];

        // inverse directions

        App.X_DIRECTION[0] *= -1;
        App.X_DIRECTION[1] *= -1;
        App.X_DIRECTION[2] *= -1;

        App.Y_DIRECTION[0] *= -1;
        App.Y_DIRECTION[1] *= -1;
        App.Y_DIRECTION[2] *= -1;
    }

    // normize all vectors
    len = App.X_DIRECTION[0] * App.X_DIRECTION[0] + App.X_DIRECTION[1] * App.X_DIRECTION[1] + App.X_DIRECTION[2] * App.X_DIRECTION[2];
    len = 1 / sqrt(len);
    App.X_DIRECTION[0] *= len;
    App.X_DIRECTION[1] *= len;
    App.X_DIRECTION[2] *= len;

    len = App.Y_DIRECTION[0] * App.Y_DIRECTION[0] + App.Y_DIRECTION[1] * App.Y_DIRECTION[1] + App.Y_DIRECTION[2] * App.Y_DIRECTION[2];
    len = 1 / sqrt(len);
    App.Y_DIRECTION[0] *= len;
    App.Y_DIRECTION[1] *= len;
    App.Y_DIRECTION[2] *= len;

    // reset
    App.Reset();

    // now let's do the work
    App.Compute(2, inportnames, 0, outportnames);

    // finish our work
    App.FinalCompute(outportnames);

    // bye
}

////// main
void main(int argc, char *argv[])
{
    // initialize
    Covise::set_module_description("map data to 2d-data and 3d-graph");

    // input
    Covise::add_port(INPUT_PORT, "poly_in", "Set_Polygons", "surface input");
    Covise::add_port(INPUT_PORT, "data_in", "Set_Float", "data input");

    // parameters
    Covise::add_port(PARIN, "position", "Vector", "point on the plane and on the new x-axis");
    Covise::set_port_default("position", "2.0 0.0 0.0");
    Covise::add_port(PARIN, "normal", "Vector", "normal of the plane");
    Covise::set_port_default("normal", "1.0 0.0 0.0");
    Covise::add_port(PARIN, "normal2", "Vector", "direction of the new x-axis");
    Covise::set_port_default("normal2", "0.0 1.0 0.0");
    Covise::add_port(PARIN, "mode", "Choice", "if a non-VR-mode is chosen, then only the vertex has to be given");
    Covise::set_port_default("mode", "1 VR x y z");

    // output
    Covise::add_port(OUTPUT_PORT, "data2d_out", "coDoVec2", "computed data");
    Covise::add_port(OUTPUT_PORT, "plane_out", "coDoPolygons", "the background-plane");
    Covise::add_port(OUTPUT_PORT, "lines_out", "Set_Lines", "computed graph");
    Covise::add_port(OUTPUT_PORT, "probe_line_out", "coDoLines", "line through the points where we interpolated the data");

    // Covise
    Covise::init(argc, argv);

    // out callback
    Covise::set_start_callback(myCallback, NULL);

    // finished
    Covise::main_loop();
}

////// this is called whenever we have to do something
void Application::Compute_No_Output(coDistributedObject **in_objs, int)
{
    // input stuff
    coDoPolygons *poly_in = NULL;
    coDoFloat *data_in = NULL;
    float *data = NULL;
    int data_size;
    int num_poly, num_vert, num_coord;
    int *vl_in, *pl_in;
    float *x_in = NULL;
    float *y_in = NULL;
    float *z_in = NULL;

    // computed things
    int last;
    int k;
    float t;

    // counters
    int poly_counter;
    int j;

    // temporary stuff
    int last_vertex;
    float came_from_vert[3];
    float came_from_data;
    float temp_vect[3];
    float *temp_vect2;
    int temp_flag;
    float a_back[3], a_data_bak;

    // output stuff

    // get input
    poly_in = (coDoPolygons *)in_objs[0];
    num_vert = poly_in->getNumVertices();
    num_poly = poly_in->getNumPolygons();
    num_coord = poly_in->getNumPoints();
    poly_in->getAddresses(&x_in, &y_in, &z_in, &vl_in, &pl_in);

    data_in = (coDoFloat *)in_objs[1];
    data_size = data_in->getNumPoints();
    ;
    data_in->getAddress(&data);

    // error-check
    if (num_coord != data_size)
    {
        Covise::sendError("ERROR: data has to be PER_VERTEX !");
        return;
    }

    // allocate mem
    temp_vect2 = new float[3];

    // work through every polygon
    for (poly_counter = 0; poly_counter < num_poly; poly_counter++)
    { // work through current polygon
        last_vertex = (poly_counter == num_poly - 1) ? num_vert : pl_in[poly_counter + 1];

        // begin with the last vertex
        j = last_vertex - 1;
        last = to_remove_vertex(x_in[vl_in[j]], y_in[vl_in[j]], z_in[vl_in[j]]);

        // remember where we came from
        came_from_vert[0] = x_in[vl_in[j]];
        came_from_vert[1] = y_in[vl_in[j]];
        came_from_vert[2] = z_in[vl_in[j]];
        came_from_data = data[vl_in[j]];

        // add a first, then b
        temp_flag = 0;

        for (j = pl_in[poly_counter]; j < last_vertex; j++)
        {
            // work through every vertex
            temp_vect[0] = x_in[vl_in[j]];
            temp_vect[1] = y_in[vl_in[j]];
            temp_vect[2] = z_in[vl_in[j]];

            if ((k = to_remove_vertex(temp_vect[0], temp_vect[1], temp_vect[2])) != last)
            {
                // we have to interpolate coordinates and data
                t = line_with_plane(came_from_vert, temp_vect, &temp_vect2);

                // store the computed stuff
                if (temp_flag)
                    add_line_element(a_back, a_data_bak, temp_vect2, came_from_data + t * (data[vl_in[j]] - came_from_data));
                else
                {
                    a_back[0] = temp_vect2[0];
                    a_back[1] = temp_vect2[1];
                    a_back[2] = temp_vect2[2];
                    a_data_bak = came_from_data + t * (data[vl_in[j]] - came_from_data);
                }

                temp_flag = !temp_flag;

                // on to the next vertex
            }

            // remember what we did with the current vertex
            last = k;

            // remember where we come from
            came_from_vert[0] = x_in[vl_in[j]];
            came_from_vert[1] = y_in[vl_in[j]];
            came_from_vert[2] = z_in[vl_in[j]];
            came_from_data = data[vl_in[j]];
        }

        // we might have one single point
        if (temp_flag)
            add_line_element(a_back, a_data_bak, a_back, a_data_bak);

        // and the next polygon
    }

    // clean up
    delete[] temp_vect2;

    // up, up and away
    return;
}

void Application::Reset(void)
{
    // reset our 'linked list'
    found_lines = NULL;
    num_lines_found = 0;

    // ok
    return;
}

void Application::add_line_element(float *a, float da, float *b, float db)
{
    line_element *p;
    int i;

    if (found_lines)
    {
        p = found_lines;
        found_lines = new line_element;
        found_lines->last = p;
    }
    else
    {
        found_lines = new line_element;
        found_lines->last = NULL;
    }
    p = found_lines;

    // store vertices
    for (i = 0; i < 3; i++)
    {
        p->a[i] = a[i];
        p->b[i] = b[i];
    }

    // store data
    p->a_data = da;
    p->b_data = db;

    num_lines_found++;

    // done
    return;
}

int Application::to_remove_vertex(float x, float y, float z)
{
    int r = 0;
    float temp_vector[3];

    // this function should return 0 or !0 if the given vertex
    // has to be removed or not
    temp_vector[0] = x - BASE_POINT[0];
    temp_vector[1] = y - BASE_POINT[1];
    temp_vector[2] = z - BASE_POINT[2];

    if (scalar_product(temp_vector, CUT_NORMAL) > 0)
        r = 1;

    return (r);
}

float Application::vert_dist_sqr(float *a, float *b)
{
    float t[3];

    t[0] = b[0] - a[0];
    t[1] = b[1] - a[1];
    t[2] = b[2] - a[2];

    return (scalar_product(t, t));
}

float Application::scalar_product(float *a, float *b)
{
    float r;
    r = (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
    return (r);
}

float Application::line_with_plane(float *p1, float *p2, float **r)
{
    float line_vect[3], temp_vect[3], t;
    int i;

    for (i = 0; i < 3; i++)
    {
        line_vect[i] = p2[i] - p1[i];
        temp_vect[i] = BASE_POINT[i] - p1[i];
    }

    // compute t
    t = scalar_product(temp_vect, CUT_NORMAL) / scalar_product(line_vect, CUT_NORMAL);

    // compute the final coordinates
    for (i = 0; i < 3; i++)
        (*r)[i] = p1[i] + t * line_vect[i];

    // done
    return (t);
}

void Application::FinalCompute(char **out_names)
{

    // output stuff
    char *obj_name;
    int *ll_out, *vl_out, *pl_out;
    float *x_out, *y_out, *z_out;

    // counters
    int num_lines_out;
    int num_vert_out;
    int i;

    // computed stuff
    float O[3], lines_O[3];
    float min_t, t;
    float x_max, y_min, y_max;
    float x_value_max, y_value_min, y_value_max;
    float x, y;
    float y_scale, y_scale_size;
    float bb_center[3];
    float bb_corner[4][3];
    int values_2d_alloc;

    // temporary stuff
    line_element *p;
    float temp_vect[3];
    char bfr[500];

    //////
    ////// we might have no points
    //////

    if (!num_lines_found)
    {
        // clean up
        p = found_lines;
        for (i = 0; i < num_lines_found; i++)
        {
            p = found_lines->last;
            delete found_lines;
            found_lines = p;
        }

        // to enable VR-support (add the ProbeLine option to the Pinboard)
        // we have to give an empty plane_out with the propper attribute set
        obj_name = Covise::get_object_name(out_names[1]);
        if (obj_name)
        {
            coDoPolygons *temp_out;

            temp_out = new coDoPolygons(obj_name, 0, 0, 0);
            sprintf(bfr, "A%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
            temp_out->addAttribute("FEEDBACK", bfr);
        }

        // that's it
        return;
    }

    //////
    ////// compute O  (lower left corner of cut-bounding-box)
    //////

    temp_vect[0] = found_lines->a[0] - BASE_POINT[0];
    temp_vect[1] = found_lines->a[1] - BASE_POINT[1];
    temp_vect[2] = found_lines->a[2] - BASE_POINT[2];
    min_t = scalar_product(X_DIRECTION, temp_vect);

    p = found_lines;

    for (i = 0; i < num_lines_found; i++)
    {
        temp_vect[0] = p->a[0] - BASE_POINT[0];
        temp_vect[1] = p->a[1] - BASE_POINT[1];
        temp_vect[2] = p->a[2] - BASE_POINT[2];
        t = scalar_product(X_DIRECTION, temp_vect);
        if (t < min_t)
            min_t = t;

        temp_vect[0] = p->b[0] - BASE_POINT[0];
        temp_vect[1] = p->b[1] - BASE_POINT[1];
        temp_vect[2] = p->b[2] - BASE_POINT[2];
        t = scalar_product(X_DIRECTION, temp_vect);
        if (t < min_t)
            min_t = t;

        p = p->last;
    }

    O[0] = BASE_POINT[0] + min_t * X_DIRECTION[0];
    O[1] = BASE_POINT[1] + min_t * X_DIRECTION[1];
    O[2] = BASE_POINT[2] + min_t * X_DIRECTION[2];

    //////
    ////// compute x_max, y_min, y_max (bounding box of cut in plane)
    //////

    x_max = 0;
    temp_vect[0] = found_lines->a[0] - O[0];
    temp_vect[1] = found_lines->a[1] - O[1];
    temp_vect[2] = found_lines->a[2] - O[2];
    y_min = y_max = scalar_product(Y_DIRECTION, temp_vect);

    p = found_lines;

    for (i = 0; i < num_lines_found; i++)
    {
        temp_vect[0] = p->a[0] - O[0];
        temp_vect[1] = p->a[1] - O[1];
        temp_vect[2] = p->a[2] - O[2];
        x = scalar_product(X_DIRECTION, temp_vect);
        y = scalar_product(Y_DIRECTION, temp_vect);
        if (x > x_max)
            x_max = x;
        if (y < y_min)
            y_min = y;
        if (y > y_max)
            y_max = y;

        temp_vect[0] = p->b[0] - O[0];
        temp_vect[1] = p->b[1] - O[1];
        temp_vect[2] = p->b[2] - O[2];
        x = scalar_product(X_DIRECTION, temp_vect);
        y = scalar_product(Y_DIRECTION, temp_vect);
        if (x > x_max)
            x_max = x;
        if (y < y_min)
            y_min = y;
        if (y > y_max)
            y_max = y;

        p = p->last;
    }

    //////
    ////// compute x/y values (values_2d) for each line-element !
    //////

    // alloc
    values_2d_alloc = num_lines_found;
    values_2d = new value_element[values_2d_alloc];

    p = found_lines;

    for (i = 0; i < num_lines_found; i++)
    {
        // a
        temp_vect[0] = p->a[0] - O[0];
        temp_vect[1] = p->a[1] - O[1];
        temp_vect[2] = p->a[2] - O[2];

        x = scalar_product(X_DIRECTION, temp_vect);

        values_2d[i].x1 = x;
        values_2d[i].y1 = p->a_data;

        // b
        temp_vect[0] = p->b[0] - O[0];
        temp_vect[1] = p->b[1] - O[1];
        temp_vect[2] = p->b[2] - O[2];

        x = scalar_product(X_DIRECTION, temp_vect);

        values_2d[i].x2 = x;
        values_2d[i].y2 = p->b_data;

        // on to the next element
        p = p->last;
    }

    //////
    ////// we have to 'sort' the line-elements so they will form the
    ////// longest possible lines (for plot-output !)
    //////

    sort_line();

    //////
    ////// compute y_value_min, y_value_max, x_value_max
    //////

    x_value_max = values_2d[0].x1;
    y_value_min = y_value_max = values_2d[0].y1;

    for (i = 0; i < num_lines_found; i++)
    {
        if (values_2d[i].x1 > x_value_max)
            x_value_max = values_2d[i].x1;
        if (values_2d[i].y1 > y_value_max)
            y_value_max = values_2d[i].y1;
        if (values_2d[i].y1 < y_value_min)
            y_value_min = values_2d[i].y1;

        if (values_2d[i].x2 > x_value_max)
            x_value_max = values_2d[i].x2;
        if (values_2d[i].y2 > y_value_max)
            y_value_max = values_2d[i].y2;
        if (values_2d[i].y2 < y_value_min)
            y_value_min = values_2d[i].y2;
    }

    //////
    ////// compute how 'high' the output 3D-Plot should be
    ////// (therefor use the bb-size)
    //////

    if (y_value_max < 0)
        y_value_max = 0.08;
    if (y_value_min < 0)
        y_scale_size = (y_value_max - y_value_min);
    else
        y_scale_size = y_value_max;

    //////
    ////// and compute the factor to scale the graph with
    //////

    y_scale = (2 * (y_max - y_min)) / y_scale_size;
    y_scale_size *= y_scale;

    //////
    ////// plane_out (and compute lines_O !)
    //////

    coDoPolygons *plane_out;

    obj_name = Covise::get_object_name(out_names[1]);
    if (obj_name)
    {
        plane_out = new coDoPolygons(obj_name, 4, 4, 1);
        plane_out->getAddresses(&x_out, &y_out, &z_out, &vl_out, &pl_out);

        // compute bounding-box corners
        // upper left
        bb_corner[0][0] = O[0] + y_max * Y_DIRECTION[0];
        bb_corner[0][1] = O[1] + y_max * Y_DIRECTION[1];
        bb_corner[0][2] = O[2] + y_max * Y_DIRECTION[2];
        // upper right
        bb_corner[1][0] = O[0] + y_max * Y_DIRECTION[0] + x_max * X_DIRECTION[0];
        bb_corner[1][1] = O[1] + y_max * Y_DIRECTION[1] + x_max * X_DIRECTION[1];
        bb_corner[1][2] = O[2] + y_max * Y_DIRECTION[2] + x_max * X_DIRECTION[2];
        // lower right
        bb_corner[2][0] = O[0] + y_min * Y_DIRECTION[0] + x_max * X_DIRECTION[0];
        bb_corner[2][1] = O[1] + y_min * Y_DIRECTION[1] + x_max * X_DIRECTION[1];
        bb_corner[2][2] = O[2] + y_min * Y_DIRECTION[2] + x_max * X_DIRECTION[2];
        // lower left
        bb_corner[3][0] = O[0] + y_min * Y_DIRECTION[0];
        bb_corner[3][1] = O[1] + y_min * Y_DIRECTION[1];
        bb_corner[3][2] = O[2] + y_min * Y_DIRECTION[2];

        // a distance to the graph might be added here
        bb_corner[0][0] += 0.2 * y_scale_size * Y_DIRECTION[0];
        bb_corner[0][1] += 0.2 * y_scale_size * Y_DIRECTION[1];
        bb_corner[0][2] += 0.2 * y_scale_size * Y_DIRECTION[2];

        bb_corner[1][0] += 0.2 * y_scale_size * Y_DIRECTION[0];
        bb_corner[1][1] += 0.2 * y_scale_size * Y_DIRECTION[1];
        bb_corner[1][2] += 0.2 * y_scale_size * Y_DIRECTION[2];

        // get lines_O
        lines_O[0] = bb_corner[0][0];
        lines_O[1] = bb_corner[0][1];
        lines_O[2] = bb_corner[0][2];

        // resize the bounding-box, so our graph fits on it
        bb_corner[0][0] += y_scale_size * Y_DIRECTION[0];
        bb_corner[0][1] += y_scale_size * Y_DIRECTION[1];
        bb_corner[0][2] += y_scale_size * Y_DIRECTION[2];
        bb_corner[1][0] += y_scale_size * Y_DIRECTION[0];
        bb_corner[1][1] += y_scale_size * Y_DIRECTION[1];
        bb_corner[1][2] += y_scale_size * Y_DIRECTION[2];

        // compute bounding-box center
        bb_center[0] = (bb_corner[3][0] + bb_corner[1][0]) / 2;
        bb_center[1] = (bb_corner[3][1] + bb_corner[1][1]) / 2;
        bb_center[2] = (bb_corner[3][2] + bb_corner[1][2]) / 2;

        // scale the bounding-box with factor t (relative to its center, so
        // it won't move)
        t = 1.2;
        // upper left
        bb_corner[0][0] = t * bb_corner[0][0] + (1 - t) * bb_center[0];
        bb_corner[0][1] = t * bb_corner[0][1] + (1 - t) * bb_center[1];
        bb_corner[0][2] = t * bb_corner[0][2] + (1 - t) * bb_center[2];
        // upper right
        bb_corner[1][0] = t * bb_corner[1][0] + (1 - t) * bb_center[0];
        bb_corner[1][1] = t * bb_corner[1][1] + (1 - t) * bb_center[1];
        bb_corner[1][2] = t * bb_corner[1][2] + (1 - t) * bb_center[2];
        // lower right
        bb_corner[2][0] = t * bb_corner[2][0] + (1 - t) * bb_center[0];
        bb_corner[2][1] = t * bb_corner[2][1] + (1 - t) * bb_center[1];
        bb_corner[2][2] = t * bb_corner[2][2] + (1 - t) * bb_center[2];
        // lower left
        bb_corner[3][0] = t * bb_corner[3][0] + (1 - t) * bb_center[0];
        bb_corner[3][1] = t * bb_corner[3][1] + (1 - t) * bb_center[1];
        bb_corner[3][2] = t * bb_corner[3][2] + (1 - t) * bb_center[2];

        // add plane (remember to put it back a little bit, so the graph
        // will float above it)
        pl_out[0] = 0;
        for (i = 0; i < 4; i++)
        {
            vl_out[i] = i;
            x_out[i] = bb_corner[i][0] + 0.01 * CUT_NORMAL[0];
            y_out[i] = bb_corner[i][1] + 0.01 * CUT_NORMAL[1];
            z_out[i] = bb_corner[i][2] + 0.01 * CUT_NORMAL[2];
        }

        // make it transparent
        plane_out->addAttribute("vertexOrder", "2");
        plane_out->addAttribute("TRANSPARENCY", "0.5");
    }

    // we support VR here
    sprintf(bfr, "A%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
    plane_out->addAttribute("FEEDBACK", bfr);

    //////
    ////// data2d_out
    //////

    coDoVec2 *data2d_out;

    obj_name = Covise::get_object_name(out_names[0]);
    if (obj_name)
    {
        /* 
            data2d_out = new coDoVec2(obj_name, num_lines_found*2);
            data2d_out->getAddresses( &x_out, &y_out );

            for( i=0; i<num_lines_found; i++ )
            {
               if( x_value_max )
               {
                  x_out[i*2] = values_2d[i].x1 / x_value_max;
                  x_out[(i*2)+1] = values_2d[i].x2 / x_value_max;
               }
      else
      {
      x_out[i*2] = values_2d[i].x1;
      x_out[(i*2)+1] = values_2d[i].x2;
      }

      y_out[i*2] = values_2d[i].y1;
      y_out[(i*2)+1] = values_2d[i].y2;
      }
      */
        data2d_out = new coDoVec2(obj_name, num_sorted_values);
        data2d_out->getAddresses(&x_out, &y_out);

        for (i = 0; i < num_sorted_values; i++)
        {
            if (x_value_max)
            {
                x_out[i] = sorted_values[i].x1 / x_value_max;
            }
            else
            {
                x_out[i] = sorted_values[i].x1;
            }
            y_out[i] = sorted_values[i].y1;
        }
    }

    //////
    ////// lines_out
    //////

    char lines_out_names[2][500];
    coDistributedObject *lines_out = NULL;
    coDoLines *set_lines_out[3]; // a set of 3 elements :
    // 0: coord.axis
    // 1: graph

    if (y_value_min < 0)
    {
        lines_O[0] -= y_value_min * y_scale * Y_DIRECTION[0];
        lines_O[1] -= y_value_min * y_scale * Y_DIRECTION[1];
        lines_O[2] -= y_value_min * y_scale * Y_DIRECTION[2];
    }

    // compute new names
    obj_name = Covise::get_object_name(out_names[2]);
    sprintf(lines_out_names[0], "%s_0", obj_name);
    sprintf(lines_out_names[1], "%s_1", obj_name);

    // coord.axis
    set_lines_out[0] = new coDoLines(lines_out_names[0], 8, 10, 2);
    set_lines_out[0]->getAddresses(&x_out, &y_out, &z_out, &vl_out, &ll_out);
    // x
    ll_out[0] = 0;
    x_out[0] = lines_O[0];
    y_out[0] = lines_O[1];
    z_out[0] = lines_O[2];
    x_out[1] = lines_O[0] + x_value_max * X_DIRECTION[0];
    y_out[1] = lines_O[1] + x_value_max * X_DIRECTION[1];
    z_out[1] = lines_O[2] + x_value_max * X_DIRECTION[2];
    if (y_scale < 1)
    {
        x_out[2] = x_out[1] - 0.03 * X_DIRECTION[0] + y_scale * 0.01 * Y_DIRECTION[0];
        y_out[2] = y_out[1] - 0.03 * X_DIRECTION[1] + y_scale * 0.01 * Y_DIRECTION[1];
        z_out[2] = z_out[1] - 0.03 * X_DIRECTION[2] + y_scale * 0.01 * Y_DIRECTION[2];
        x_out[3] = x_out[1] - 0.03 * X_DIRECTION[0] - y_scale * 0.01 * Y_DIRECTION[0];
        y_out[3] = y_out[1] - 0.03 * X_DIRECTION[1] - y_scale * 0.01 * Y_DIRECTION[1];
        z_out[3] = z_out[1] - 0.03 * X_DIRECTION[2] - y_scale * 0.01 * Y_DIRECTION[2];
    }
    else
    {
        x_out[2] = x_out[1] - 0.03 * X_DIRECTION[0] + 0.01 * Y_DIRECTION[0];
        y_out[2] = y_out[1] - 0.03 * X_DIRECTION[1] + 0.01 * Y_DIRECTION[1];
        z_out[2] = z_out[1] - 0.03 * X_DIRECTION[2] + 0.01 * Y_DIRECTION[2];
        x_out[3] = x_out[1] - 0.03 * X_DIRECTION[0] - 0.01 * Y_DIRECTION[0];
        y_out[3] = y_out[1] - 0.03 * X_DIRECTION[1] - 0.01 * Y_DIRECTION[1];
        z_out[3] = z_out[1] - 0.03 * X_DIRECTION[2] - 0.01 * Y_DIRECTION[2];
    }

    vl_out[0] = 0;
    vl_out[1] = 1;
    vl_out[2] = 2;
    vl_out[3] = 3;
    vl_out[4] = 1;
    // y
    ll_out[1] = 5;
    if (y_value_min < 0)
    {
        x_out[4] = lines_O[0] + y_value_min * y_scale * Y_DIRECTION[0];
        y_out[4] = lines_O[1] + y_value_min * y_scale * Y_DIRECTION[1];
        z_out[4] = lines_O[2] + y_value_min * y_scale * Y_DIRECTION[2];
    }
    else
    {
        x_out[4] = x_out[0];
        y_out[4] = y_out[0];
        z_out[4] = z_out[0];
    }

    if (y_value_max <= 0)
        y_value_max = 0.08;

    x_out[5] = lines_O[0] + y_value_max * y_scale * Y_DIRECTION[0];
    y_out[5] = lines_O[1] + y_value_max * y_scale * Y_DIRECTION[1];
    z_out[5] = lines_O[2] + y_value_max * y_scale * Y_DIRECTION[2];

    //fprintf(stderr, "y_value_max = %f\ny_scale = %f\n\n", y_value_max, y_scale);

    if (y_scale < 1)
    {
        x_out[6] = x_out[5] - y_scale * 0.01 * X_DIRECTION[0] - 0.03 * Y_DIRECTION[0];
        y_out[6] = y_out[5] - y_scale * 0.01 * X_DIRECTION[1] - 0.03 * Y_DIRECTION[1];
        z_out[6] = z_out[5] - y_scale * 0.01 * X_DIRECTION[2] - 0.03 * Y_DIRECTION[2];
        x_out[7] = x_out[5] + y_scale * 0.01 * X_DIRECTION[0] - 0.03 * Y_DIRECTION[0];
        y_out[7] = y_out[5] + y_scale * 0.01 * X_DIRECTION[1] - 0.03 * Y_DIRECTION[1];
        z_out[7] = z_out[5] + y_scale * 0.01 * X_DIRECTION[2] - 0.03 * Y_DIRECTION[2];
    }
    else
    {
        x_out[6] = x_out[5] - 0.01 * X_DIRECTION[0] - 0.03 * Y_DIRECTION[0];
        y_out[6] = y_out[5] - 0.01 * X_DIRECTION[1] - 0.03 * Y_DIRECTION[1];
        z_out[6] = z_out[5] - 0.01 * X_DIRECTION[2] - 0.03 * Y_DIRECTION[2];
        x_out[7] = x_out[5] + 0.01 * X_DIRECTION[0] - 0.03 * Y_DIRECTION[0];
        y_out[7] = y_out[5] + 0.01 * X_DIRECTION[1] - 0.03 * Y_DIRECTION[1];
        z_out[7] = z_out[5] + 0.01 * X_DIRECTION[2] - 0.03 * Y_DIRECTION[2];
    }

    vl_out[5] = 4;
    vl_out[6] = 5;
    vl_out[7] = 6;
    vl_out[8] = 7;
    vl_out[9] = 5;
    // set color
    set_lines_out[0]->addAttribute("COLOR", "red");

    // graph
    set_lines_out[1] = new coDoLines(lines_out_names[1], num_lines_found * 2, num_lines_found * 2, num_lines_found);
    set_lines_out[1]->getAddresses(&x_out, &y_out, &z_out, &vl_out, &ll_out);

    num_lines_out = 0;
    num_vert_out = 0;

    for (i = 0; i < num_lines_found; i++)
    {
        ll_out[num_lines_out] = num_vert_out;
        num_lines_out++;

        vl_out[num_vert_out] = num_vert_out;
        x_out[num_vert_out] = lines_O[0] + values_2d[i].x1 * X_DIRECTION[0] + y_scale * values_2d[i].y1 * Y_DIRECTION[0];
        y_out[num_vert_out] = lines_O[1] + values_2d[i].x1 * X_DIRECTION[1] + y_scale * values_2d[i].y1 * Y_DIRECTION[1];
        z_out[num_vert_out] = lines_O[2] + values_2d[i].x1 * X_DIRECTION[2] + y_scale * values_2d[i].y1 * Y_DIRECTION[2];
        num_vert_out++;

        vl_out[num_vert_out] = num_vert_out;
        x_out[num_vert_out] = lines_O[0] + values_2d[i].x2 * X_DIRECTION[0] + y_scale * values_2d[i].y2 * Y_DIRECTION[0];
        y_out[num_vert_out] = lines_O[1] + values_2d[i].x2 * X_DIRECTION[1] + y_scale * values_2d[i].y2 * Y_DIRECTION[1];
        z_out[num_vert_out] = lines_O[2] + values_2d[i].x2 * X_DIRECTION[2] + y_scale * values_2d[i].y2 * Y_DIRECTION[2];
        num_vert_out++;
    }

    // set color
    set_lines_out[1]->addAttribute("COLOR", "green");

    // add all elements to the set
    set_lines_out[2] = NULL;
    lines_out = new coDoSet(obj_name, (coDistributedObject **)set_lines_out);
    delete set_lines_out[0];
    delete set_lines_out[1];
    delete lines_out;

    //////
    ////// probe_line_out + clean up (found_lines)
    //////

    coDoLines *probe_line_out;

    obj_name = Covise::get_object_name(out_names[3]);
    if (obj_name)
    {
        probe_line_out = new coDoLines(obj_name, num_lines_found * 2, num_lines_found * 2, num_lines_found);
        probe_line_out->getAddresses(&x_out, &y_out, &z_out, &vl_out, &ll_out);

        num_lines_out = 0;
        num_vert_out = 0;

        p = found_lines;
        for (i = 0; i < num_lines_found; i++)
        {
            ll_out[num_lines_out] = num_vert_out;
            num_lines_out++;

            vl_out[num_vert_out] = num_vert_out;
            x_out[num_vert_out] = p->a[0];
            y_out[num_vert_out] = p->a[1];
            z_out[num_vert_out] = p->a[2];
            num_vert_out++;

            vl_out[num_vert_out] = num_vert_out;
            x_out[num_vert_out] = p->b[0];
            y_out[num_vert_out] = p->b[1];
            z_out[num_vert_out] = p->b[2];
            num_vert_out++;

            p = found_lines->last;
            delete found_lines;
            found_lines = p;
        }

        // done
    }

    //////
    ////// clean up
    //////

    delete[] values_2d;
    delete[] sorted_values;

    // bye
    return;
}

void Application::sort_line(void)
{
    int i;
    float x, y;

    int n;
    char bfr[200];

    // reset
    for (i = 0; i < num_lines_found; i++)
        values_2d[i].flag = 0;
    num_sorted_values = 0;

    // alloc
    // +2 reicht wohl auch aber sicher ist sicher
    sorted_values = new value_element[num_lines_found * 2 + 10];

    n = 0;

    // work through
    x = 0.0;
    y = 0.0;
    while ((i = find_next_unhandled(x, y)) != -1)
    {
        linear_element_sort(i);
        x = sorted_values[num_sorted_values - 1].x1;
        y = sorted_values[num_sorted_values - 1].y1;

        n++;
    }

    sprintf(bfr, "number of lines: n=%d,  num_sorted_values=%d", n, num_sorted_values);
    Covise::sendInfo(bfr);

    // return
    return;
}

int Application::element_seek(float x, float y, int excl1, int excl2, float *next_x, float *next_y)
{
    // return id of element (or -1) in values_2d that contains the
    // point x/y (and is different from excl1 and excl2)

    int i, r;

    r = -1;
    for (i = 0; i < num_lines_found && r == -1; i++)
    {
        if (!values_2d[i].flag && i != excl1 && i != excl2)
        {
            if (values_2d[i].x1 == x && values_2d[i].y1 == y)
            {
                r = i;
                *next_x = values_2d[i].x2;
                *next_y = values_2d[i].y2;
            }
            else if (values_2d[i].x2 == x && values_2d[i].y2 == y)
            {
                r = i;
                *next_x = values_2d[i].x1;
                *next_y = values_2d[i].y1;
            }
        }
    }

    // done
    return (r);
}

int Application::find_next_unhandled(float x, float y)
{
    int i, r;
    float dist, min;

    r = -1;
    for (i = 0; i < num_lines_found && values_2d[i].flag; i++)
        ;

    if (i < num_lines_found)
    {
        min = (values_2d[i].x1 - x) * (values_2d[i].x1 - x) + (values_2d[i].y1 - y) * (values_2d[i].y1 - y);
        r = i;

        for (i = 0; i < num_lines_found; i++)
        {
            if (!values_2d[i].flag)
            {
                dist = (values_2d[i].x1 - x) * (values_2d[i].x1 - x) + (values_2d[i].y1 - y) * (values_2d[i].y1 - y);
                if (dist < min)
                {
                    min = dist;
                    r = i;
                }

                dist = (values_2d[i].x2 - x) * (values_2d[i].x2 - x) + (values_2d[i].y2 - y) * (values_2d[i].y2 - y);
                if (dist < min)
                {
                    min = dist;
                    r = i;
                }
            }
        }
    }

    // done
    return (r);
}

void Application::linear_element_sort(int i)
{
    float x1, y1; //, x2, y2;
    int j, n;
    float x, y;

    // store current values
    x1 = values_2d[i].x1;
    y1 = values_2d[i].y1;

    //x2 = values_2d[i].x2;
    //y2 = values_2d[i].y2;

    //
    // i: start element
    // n: next element
    // j: current element
    //

    // go to the left-most element
    x = x1;
    y = y1;
    j = i;
    n = i;
    int count = 0;
    do
    {
        j = n;
        n = element_seek(x, y, i, j, &x, &y);
        count++;
    } while (n != -1 && count < 2 * num_lines_found);

    // now build sorted_values by going to the right

    // add first point
    sorted_values[num_sorted_values].x1 = x;
    sorted_values[num_sorted_values].y1 = y;
    num_sorted_values++;

    if (x == values_2d[j].x1)
    {
        x = values_2d[j].x2;
        y = values_2d[j].y2;
    }
    else
    {
        x = values_2d[j].x1;
        y = values_2d[j].y1;
    }

    i = j;
    n = j;
    count = 0;
    do
    {
        values_2d[j].flag = 1;
        sorted_values[num_sorted_values].x1 = x;
        sorted_values[num_sorted_values].y1 = y;
        num_sorted_values++;

        j = n;
        n = element_seek(x, y, i, j, &x, &y);
        count++;
    } while (n != -1 && count < 2 * num_lines_found);

    // done
}
