/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: axe-murder geometry                                          **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Lars Frenzel                                **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  05.01.97  V0.1                                                  **
\**************************************************************************/

#include <api/coSimpleModule.h>
#include <util/coviseCompat.h>
#include <do/coDoData.h>

#include <float.h>

using namespace covise;

class Interpol
{
private:
    // pointers to the 'real' data
    int *num_vert_out;
    int **vl_out;
    int *num_coord_out;
    float **x_coord_out;
    float **y_coord_out;
    float **z_coord_out;

    // internal data
    struct record_struct
    {
        int to_vertex;
        int interpol_id;
        record_struct *next;
    } *record_data, **vertexID;

    int cur_heap_pos;
    int heap_blk_size;

    // internal functions
    void add_record(int v1, int v2, int id);

public:
    // reset everything
    void reset(int, int *, int **, int *, float **, float **, float **, int);

    // check if the connection from v1 to v2 has allready been
    // interpolated
    int interpolated(int v1, int v2);

    // add a new vertex to the output data
    int add_vertex(int, int, float, float, float);

    // clean up
    void finished(void);
};

////// our class
class IsoLines : public coSimpleModule, public Interpol
{
    COMODULE

private:
    coFloatParam *pFrom, *pTo, *pDistance;
    coIntScalarParam *pNLines;
    coInputPort *pGeoIn, *pDataIn;
    coOutputPort *pLinesOut, *pDataOut;

public:
    /// Methods
    IsoLines(int argc, char **argv);
    virtual ~IsoLines() {}
    virtual int compute(const char *port);
};

IsoLines::IsoLines(int argc, char **argv)
    : coSimpleModule(argc, argv, "compute ISO-lines")
{
    // input
    pGeoIn = addInputPort("GridIn0", "Polygons", "geometry");
    pDataIn = addInputPort("DataIn0", "Float", "data");

    // parameters
    pFrom = addFloatParam("from", "start at this value");
    pFrom->setValue(0.0);
    pTo = addFloatParam("to", "end at this value");
    pTo->setValue(1.0);
    //pStep = addFloatParam("step", "step");
    //pStep->setValue(1./3.);
    pNLines = addInt32Param("nlines", "nlines");
    pNLines->setValue(10);
    pDistance = addFloatParam("distance", "distance to surface");
    pDistance->setValue(0.5);

    // output
    pLinesOut = addOutputPort("GridOut0", "Lines", "the ISO-lines");
    pDataOut = addOutputPort("DataOut0", "Float", "data");
}

int IsoLines::compute(const char *)
{

    float ISO_from = pFrom->getValue();
    float ISO_to = pTo->getValue();
    //float ISO_step = pStep->getValue();
    int ISO_nlines = pNLines->getValue();
    float ISO_distance = pDistance->getValue();

    float ISO_step = (ISO_to - ISO_from) / (ISO_nlines - 1);

    // error check
    if (ISO_step == 0)
    {
        ISO_to = ISO_from;
        ISO_step = 1;
    }
    if ((ISO_from < ISO_to && ISO_step < 0) || (ISO_from > ISO_to && ISO_step > 0))
    {
        ISO_to = ISO_from;
        ISO_step = 1;
    }

    // objects
    coDistributedObject *data_return = NULL;
    coDistributedObject *lines_return = NULL;
    const coDoPolygons *poly_in = NULL;
    coDistributedObject **r = new coDistributedObject *[2];
    const coDoFloat *s_data = NULL;

    // variables for in/output-data
    float *in_data, *out_data;

    // coordinates
    float *x_in, *y_in, *z_in;
    float *x_out, *y_out, *z_out;
    float x_last, y_last, z_last;

    // polygon-input
    int *vl_in;
    int *pl_in;

    int num_poly, num_vert, num_coord;

    // line output
    int *ll_out;
    int *vl_out;

    int num_lines_out, num_vert_out, num_coord_out;

    // counters
    float ISO_value;
    int i, k, p, a, poly_count, n;
    int last_vl;

    // other stuff
    float last_data;
    int la_flag;
    float vect[3], normal[3] = { 0.f, 0.f, 0.f }, u[3], v[3];
    float t;

    // get the input geometry
    const coDistributedObject *geoIn = pGeoIn->getCurrentObject();
    if (!geoIn)
    {
        sendError("ERROR: no geometry object");
        return STOP_PIPELINE;
    }

    if ((poly_in = dynamic_cast<const coDoPolygons *>(geoIn)))
    {
        num_coord = poly_in->getNumPoints();
        num_vert = poly_in->getNumVertices();
        num_poly = poly_in->getNumPolygons();
        poly_in->getAddresses(&x_in, &y_in, &z_in, &vl_in, &pl_in);
    }
    else
    {
        sendError("ERROR: invalid geometry object");
        return STOP_PIPELINE;
    }

    const coDistributedObject *dataIn = pDataIn->getCurrentObject();
    if (!pDataIn)
    {
        sendError("ERROR: no data object");
        return STOP_PIPELINE;
    }

    if (!dataIn->isType("USTSDT"))
    {
        sendError("ERROR: invalid data object");
        return STOP_PIPELINE;
    }

    // get the input data
    s_data = (const coDoFloat *)dataIn;
    s_data->getAddress(&in_data);
    if (s_data->getNumPoints() == 0) // dummy input data
    {
        lines_return = new coDoLines(pLinesOut->getObjName(), 0, 0, 0);
        //   Covise_Set_Handler::copy_attributes( poly_in , lines_return);

        // and generate data-output
        data_return = new coDoFloat(pDataOut->getObjName(), 0);
        r[0] = lines_return;
        r[1] = data_return;

        pLinesOut->setCurrentObject(lines_return);
        pDataOut->setCurrentObject(data_return);

        return CONTINUE_PIPELINE;
    }

    if (s_data->getNumPoints() != num_coord)
    {
        sendError("ERROR: this module requires per-vertex data");
        return STOP_PIPELINE;
    }

    // allocate memory
    num_lines_out = 0;
    num_coord_out = 0;
    num_vert_out = 0;
    // compute how much mem we will need for temp data
    if (dynamic_cast<const coDoPolygons *>(geoIn))
    {
        for (poly_count = 0; poly_count < num_poly; poly_count++)
        {
            for (ISO_value = ISO_from; ISO_value <= ISO_to; ISO_value += ISO_step)
            { // now go through this polygon
                // first compute how many vertices it has
                if (poly_count == num_poly - 1)
                    p = num_vert - pl_in[poly_count];
                else
                    p = pl_in[poly_count + 1] - pl_in[poly_count];

                k = vl_in[pl_in[poly_count]];
                last_data = in_data[k];
                x_last = x_in[k];
                y_last = y_in[k];
                z_last = z_in[k];

                k = pl_in[poly_count];

                // we might have to add a new line-element
                la_flag = 1;

                for (i = 1; i < p + 1; i++)
                {
                    if (i == p)
                        // 'close the circle' to the first vertex
                        a = vl_in[k];
                    else
                        a = vl_in[k + i];
                    // check if our value is there
                    if (fabs(in_data[a] - last_data) > FLT_EPSILON && ((last_data <= ISO_value && ISO_value <= in_data[a])
                                                                       || (in_data[a] <= ISO_value && ISO_value <= last_data)))
                    { // our value is there
                        if (la_flag)
                        { // add a new line
                            num_lines_out++;
                        }

                        // add a new vertex
                        num_vert_out++;

                        // add the coordinates
                        num_coord_out++;
                    }

                    // remember current coordinates for later
                    x_last = x_in[a];
                    y_last = y_in[a];
                    z_last = z_in[a];

                    // if our value isn't there then just do nothing
                    last_data = in_data[a];
                }
            }
        }
    }

    // and allocate it
    x_out = new float[num_coord_out];
    y_out = new float[num_coord_out];
    z_out = new float[num_coord_out];
    vl_out = new int[num_vert_out];
    ll_out = new int[num_lines_out];
    out_data = new float[num_coord_out];

    // reset
    num_lines_out = 0;
    num_vert_out = 0;
    num_coord_out = 0;

    // work through the geometry
    if (dynamic_cast<const coDoPolygons *>(geoIn))
    {
        for (ISO_value = ISO_from; ISO_value <= ISO_to; ISO_value += ISO_step)
        { // reset interpol
            Interpol::reset(num_coord, &num_vert_out, &vl_out,
                            &num_coord_out, &x_out, &y_out, &z_out, 2000);

            // go throug all polygons
            for (poly_count = 0; poly_count < num_poly; poly_count++)
            { // now go through this polygon
                // first compute how many vertices it has
                if (poly_count == num_poly - 1)
                    p = num_vert - pl_in[poly_count];
                else
                    p = pl_in[poly_count + 1] - pl_in[poly_count];

                k = vl_in[pl_in[poly_count]];
                last_data = in_data[k];
                x_last = x_in[k];
                y_last = y_in[k];
                z_last = z_in[k];
                last_vl = k;

                k = pl_in[poly_count];

                // we might have to add a new line-element
                la_flag = 1;

                for (i = 1; i < p + 1; i++)
                {
                    if (i == p)
                        // 'close the circle' to the first vertex
                        a = vl_in[k];
                    else
                        a = vl_in[k + i];
                    // check if our value is there
                    if (fabs(in_data[a] - last_data) > FLT_EPSILON && ((last_data <= ISO_value && ISO_value <= in_data[a])
                                                                       || (in_data[a] <= ISO_value && ISO_value <= last_data)))
                    { // our value is there
                        if (la_flag)
                        { // add a new line
                            ll_out[num_lines_out] = num_vert_out;
                            num_lines_out++;
                            la_flag = 0;

                            // and compute normal, as we haven't done it before
                            u[0] = x_in[vl_in[k + 1]] - x_in[vl_in[k]];
                            u[1] = y_in[vl_in[k + 1]] - y_in[vl_in[k]];
                            u[2] = z_in[vl_in[k + 1]] - z_in[vl_in[k]];
                            v[0] = x_in[vl_in[k + 2]] - x_in[vl_in[k + 1]];
                            v[1] = y_in[vl_in[k + 2]] - y_in[vl_in[k + 1]];
                            v[2] = z_in[vl_in[k + 2]] - z_in[vl_in[k + 1]];

                            normal[0] = u[1] * v[2] - v[1] * u[2];
                            normal[1] = u[2] * v[0] - v[2] * u[0];
                            normal[2] = u[0] * v[1] - v[0] * u[1];
                            if (fabs(normal[0]) > FLT_EPSILON || fabs(normal[1]) > FLT_EPSILON || fabs(normal[2]) > FLT_EPSILON)
                            {
                                t = ISO_distance / sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);

                                normal[0] *= t;
                                normal[1] *= t;
                                normal[2] *= t;
                            }
                            else
                            {
                                num_lines_out--;
                                break;
                            }
                        }

                        if ((n = Interpol::interpolated(last_vl, a)))
                        {
                            // vertex allready interpolated, so just update the vertexlist
                            vl_out[num_vert_out] = n - 1;
                            num_vert_out++;

                            // and remember to let it float
                            // x_out[n-1] += normal[0];
                            // y_out[n-1] += normal[1];
                            // z_out[n-1] += normal[2];
                        }
                        else
                        {
                            // add a new vertex

                            vect[0] = x_in[a] - x_last;
                            vect[1] = y_in[a] - y_last;
                            vect[2] = z_in[a] - z_last;

                            // compute t
                            t = (ISO_value - last_data) / (in_data[a] - last_data);

                            vect[0] = x_last + t * vect[0] + 2 * normal[0];
                            vect[1] = y_last + t * vect[1] + 2 * normal[1];
                            vect[2] = z_last + t * vect[2] + 2 * normal[2];

                            // and the vertex
                            Interpol::add_vertex(last_vl, a, vect[0],
                                                 vect[1], vect[2]);
                        }

                        /* 

                  // add a new vertex
                  vl_out[num_vert_out] = num_coord_out;
                  num_vert_out++;

                  // add the coordinates
                  vect[0] = x_in[a] - x_last;
                  vect[1] = y_in[a] - y_last;
                  vect[2] = z_in[a] - z_last;

                  // compute t
                  t = (ISO_value - last_data) / (in_data[a] - last_data);

                  // finally add 'em
                  x_out[num_coord_out] = x_last + t*vect[0] + normal[0];
                  y_out[num_coord_out] = y_last + t*vect[1] + normal[1];
                  z_out[num_coord_out] = z_last + t*vect[2] + normal[2];
                  num_coord_out++;
                   */
                        // and the data
                        ///////// we might implement user-adjustable colours here
                        out_data[num_coord_out - 1] = ISO_value;
                    }

                    // remember current coordinates for later
                    x_last = x_in[a];
                    y_last = y_in[a];
                    z_last = z_in[a];
                    last_vl = a;

                    // if our value isn't there then just do nothing
                    last_data = in_data[a];
                }
            }
            // clean up
            Interpol::finished();
        }
    }
    else
    {
        sendError("ERROR: invalid input-object");
        return STOP_PIPELINE;
    }

    // we have multi vertices and lines containing only one single connectivity
    // so we have to do some work on this

    // generate output-lines
    lines_return = new coDoLines(pLinesOut->getObjName(), num_coord_out, x_out, y_out, z_out,
                                 num_vert_out, vl_out, num_lines_out, ll_out);
    //   Covise_Set_Handler::copy_attributes( poly_in , lines_return);

    // and generate data-output
    data_return = new coDoFloat(pDataOut->getObjName(), num_coord_out, out_data);
    //   Covise_Set_Handler::copy_attributes( s_data, data_return );

    // clean up
    delete[] x_out;
    delete[] y_out;
    delete[] z_out;
    delete[] vl_out;
    delete[] ll_out;
    delete[] out_data;

    // return something
    r[0] = lines_return;
    r[1] = data_return;

    lines_return->addAttribute("COLOR", "black");

    pLinesOut->setCurrentObject(lines_return);
    pDataOut->setCurrentObject(data_return);

    return CONTINUE_PIPELINE;
}

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

// data stored in the following format:
//
//    vertexID[ lower-vertex-id ] -> ptr to beginning of the record or NULL
//    record:    int higher_vertex_id;
//               int interpolated_vertex_id;
//               ptr to beginning of next record or NULL
//

void Interpol::reset(int total_num_vert, int *num_vert, int **vl, int *num_coord,
                     float **x, float **y, float **z, int hsize = 2000)
{
    int i;

    // get parameters and store them
    num_vert_out = num_vert;
    vl_out = vl;
    num_coord_out = num_coord;
    x_coord_out = x;
    y_coord_out = y;
    z_coord_out = z;
    heap_blk_size = hsize;

    // init our environment
    vertexID = new record_struct *[total_num_vert];
    record_data = new record_struct[heap_blk_size];
    record_data[0].next = NULL;
    cur_heap_pos = 1;

    // we have interpolated no vertices at all
    for (i = 0; i < total_num_vert; i++)
        vertexID[i] = NULL;

    // that's all
    return;
}

int Interpol::interpolated(int v1, int v2)
{
    int r;
    int f, t;
    record_struct *ptr;

    // v1 has to be <v2
    if (v1 < v2)
    {
        f = v1;
        t = v2;
    }
    else
    {
        f = v2;
        t = v1;
    }

    if (vertexID[f] == NULL)
        r = 0;
    else
    {
        ptr = vertexID[f];
        while (ptr->to_vertex != t && ptr->next != NULL)
            ptr = ptr->next;
        if (ptr->to_vertex == t)
            r = ptr->interpol_id + 1;
        else
            r = 0;
    }

    return (r);
}

void Interpol::finished(void)
{
    record_struct *ptr;
    // clean up
    delete[] vertexID;

    // record_data
    ptr = record_data[0].next;
    while (ptr != NULL)
    {
        delete[] record_data;
        record_data = ptr;
        ptr = ptr[0].next;
    }
    delete[] record_data;

    // bye
    return;
}

int Interpol::add_vertex(int v1, int v2, float x = 0, float y = 0, float z = 0)
{
    int r;

    // we allready might have the coordinates
    if ((r = interpolated(v1, v2)))
    { // yes we have 'em
        // so just update the vertex-list
        r--;
        (*vl_out)[(*num_vert_out)] = r - 1;
        (*num_vert_out)++;
    }
    else
    { // no - we have to add them to the vertex-list, coordinates and
        // record it
        r = (*num_coord_out);
        add_record(v1, v2, r);

        (*x_coord_out)[r] = x;
        (*y_coord_out)[r] = y;
        (*z_coord_out)[r] = z;

        (*vl_out)[(*num_vert_out)] = r;
        (*num_vert_out)++;
        (*num_coord_out)++;
    }

    // we return the # in the coordinates-list
    return (r);
}

void Interpol::add_record(int v1, int v2, int id)
{
    int f, t;
    record_struct *ptr;
    record_struct *store;

    // v1 has to be <v2
    if (v1 < v2)
    {
        f = v1;
        t = v2;
    }
    else
    {
        f = v2;
        t = v1;
    }

    // compute where to store this record
    if (cur_heap_pos >= heap_blk_size)
    { // we need a new block
        ptr = record_data;
        record_data = new record_struct[heap_blk_size];
        record_data[0].next = ptr;
        cur_heap_pos = 1;
    }
    store = &record_data[cur_heap_pos];
    cur_heap_pos++;

    // we might have to update the linked list
    if ((ptr = vertexID[f]) != NULL)
    { // we have to update the linked-list
        // go through it 'till the end
        while (ptr->next != NULL)
            ptr = ptr->next;
        // and point to the next record
        ptr->next = store;
    }
    else
        vertexID[f] = store;

    // now store the record
    store->to_vertex = t;
    store->interpol_id = id;
    store->next = NULL;

    // that's it
    return;
}

MODULE_MAIN(Mapper, IsoLines)
