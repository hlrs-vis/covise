/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                    (C) 2000 VirCinity  **
 ** Description: axe-murder geometry                                       **
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
 **        18.10.2000 V1.0 new API, several data ports, triangle strips    **
 **                                               ( converted to polygons )**
 **			Sven Kufer 					  **
\**************************************************************************/

#define NUM_DATA_IN_PORTS 4 // number of data ports

#include <util/coviseCompat.h>
#include <do/coDoText.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoData.h>
#include "CutGeometry.h"

#define sqr(x) ((x) * (x))

// interpolate two RGBA colors saved in float fields
inline float interRGBA(float &d1, float &d2, float t)
{
    unsigned char res[4];

    unsigned char *c1 = (unsigned char *)&d1;
    unsigned char *c2 = (unsigned char *)&d2;

    float v;
    float t1 = 1 - t;
    v = t1 * c1[0] + t * c2[0];
    res[0] = (v <= 0) ? 0 : ((v >= 255) ? 255 : ((unsigned char)v));
    v = t1 * c1[1] + t * c2[1];
    res[1] = (v <= 0) ? 0 : ((v >= 255) ? 255 : ((unsigned char)v));
    v = t1 * c1[2] + t * c2[2];
    res[2] = (v <= 0) ? 0 : ((v >= 255) ? 255 : ((unsigned char)v));
    v = t1 * c1[3] + t * c2[3];
    res[3] = (v <= 0) ? 0 : ((v >= 255) ? 255 : ((unsigned char)v));

    return *((float *)&res);
}

CutGeometry::CutGeometry(int argc, char *argv[])
    : coSimpleModule(argc, argv, "cut something out of an object")
{
    int i;
    char portname[32];

    const char *method_labels[] = { "GeoCut", "DataCut" };
    p_method = addChoiceParam("method", "cut geometry based on data or based on geometry");
    p_method->setValue(2, method_labels, 0);

    const char *geoMethod_labels[] = { "Plane", "Cylinder", "Sphere" };
    p_geoMethod = addChoiceParam("geoMethod", "cut with a plane, cylider or sphere");
    p_geoMethod->setValue(3, geoMethod_labels, 0);

    // create the parameter
    distance_of_plane = addFloatParam("distance", "distance of plane or cylinder radius - use negative radius to invert cylinder cut!");
    normal_of_plane = addFloatVectorParam("normal", "normal of plane or cylinder axis");
    cylinder_bottom = addFloatVectorParam("bottom", "point on cylinder axis or center of sphere");

    // set default values
    distance_of_plane->setValue(0.0);
    normal_of_plane->setValue(0.0, 0.0, 1.0);
    cylinder_bottom->setValue(0.0, 0.0, 0.0);

    // ports
    p_geo_in = addInputPort("GridIn0", "Polygons|TriangleStrips|Lines", "geometry");
    p_geo_out = addOutputPort("GridOut0", "Polygons|Lines", "geometry");

    for (i = 0; i < NUM_DATA_IN_PORTS; i++)
    {
        sprintf(portname, "DataIn%d", i);
        p_data_in[i] = addInputPort(portname, "Float|Vec3|RGBA", "data");
        p_data_in[i]->setRequired(0);

        sprintf(portname, "DataOut%d", i);
        p_data_out[i] = addOutputPort(portname, "Float|Vec3|RGBA", "data");
        if (p_data_out[i] == NULL)
            sendError("Error");

        p_data_out[i]->setDependencyPort(p_data_in[i]);
    }
    p_adjustParams_ = addInputPort("adjustParams", "Text", "override parameter values");
    p_adjustParams_->setRequired(0);

    p_data_min = addFloatParam("data_min", "smallest data value, polygons with smaller values will be removed");
    p_data_min->setValue(0.0);

    p_data_max = addFloatParam("data_max", "biggest data value, polygons with bigger values will be removed");
    p_data_max->setValue(1.0);

    p_invert_cut = addBooleanParam("invert_cut", "invert selected polygons?");
    p_invert_cut->setValue(false);

    p_strictSelection = addBooleanParam("strict_selection", "one vertex out of bound is enough to erase polygon");
    p_strictSelection->setValue(false);
}

void
CutGeometry::postInst()
{
    hparams_.push_back(h_distance_of_plane_ = new coHideParam(distance_of_plane));
    hparams_.push_back(h_normal_of_plane_ = new coHideParam(normal_of_plane));

    p_data_min->disable();
    p_data_max->disable();
    p_invert_cut->disable();
    p_strictSelection->disable();
}

void
CutGeometry::param(const char *paramname, bool /*in_map_loading*/)
{
    if (strcmp(paramname, p_method->getName()) == 0)
    {
        switch (p_method->getValue())
        {
        case 0:
            p_geoMethod->enable();
            distance_of_plane->enable();
            normal_of_plane->enable();
            if ((p_geoMethod->getValue() == 1) || (p_geoMethod->getValue() == 2))
                cylinder_bottom->enable();
            p_data_min->disable();
            p_data_max->disable();
            p_invert_cut->disable();
            p_strictSelection->disable();
            break;

        case 1:
            p_geoMethod->setValue(0);
            p_geoMethod->disable();
            distance_of_plane->disable();
            normal_of_plane->disable();
            cylinder_bottom->disable();
            p_data_min->enable();
            p_data_max->enable();
            p_invert_cut->enable();
            p_strictSelection->enable();
            break;
        }
    }

    if (strcmp(paramname, p_geoMethod->getName()) == 0)
    {
        switch (p_geoMethod->getValue())
        {
        case 0: // plane
            cylinder_bottom->disable();
            distance_of_plane->enable();
            normal_of_plane->enable();
            break;

        case 1: // cylinder
            cylinder_bottom->enable();
            distance_of_plane->enable();
            normal_of_plane->enable();
            if (p_method->getValue() == 1)
            {
                p_geoMethod->setValue(0);
            }

        case 2: // sphere
            cylinder_bottom->enable();
            normal_of_plane->disable();
            distance_of_plane->enable();
            if (p_method->getValue() == 1)
            {
                p_geoMethod->setValue(0);
            }
            break;
        }
    }
}

////// this is called whenever we have to do something

void
CutGeometry::preHandleObjects(coInputPort **in_ports)
{
    (void)in_ports;
    // first of all reset all hparams_
    int param;
    for (param = 0; param < hparams_.size(); ++param)
    {
        hparams_[param]->reset();
    }
    // and load if necessary...
    preOK_ = true;
    if (p_adjustParams_->getCurrentObject())
    {
        const coDoText *Text = dynamic_cast<const coDoText *>(p_adjustParams_->getCurrentObject());
        if (!Text)
        {
            preOK_ = false;
            sendError("only coDoText accepted at adjustParams");
            return;
        }
        char *werte;
        Text->getAddress(&werte);
        istringstream pvalues(werte);
        char *value = new char[strlen(werte) + 1];
        while (pvalues.getline(value, strlen(werte) + 1))
        {
            int param;
            for (param = 0; param < hparams_.size(); ++param)
            {
                hparams_[param]->load(value);
            }
        }
        delete[] value;
    }
}

int CutGeometry::compute(const char *)
{
    if (!preOK_)
    {
        return FAIL;
    }

    // in objects

    const coDistributedObject *in_objs[NUM_DATA_IN_PORTS + 2];

    // out objects
    coDoVec3 *v_data_in[NUM_DATA_IN_PORTS + 1];
    coDoFloat *s_data_in[NUM_DATA_IN_PORTS + 1];
    coDoRGBA *c_data_in[NUM_DATA_IN_PORTS + 1];
    coDistributedObject *data_return = NULL;

    // initialize
    for (int s = 1; s <= NUM_DATA_IN_PORTS; s++)
    {
        v_data_in[s] = NULL;
        s_data_in[s] = NULL;
        c_data_in[s] = NULL;
    }

    coDistributedObject *geo_return = NULL;
    coDoPolygons *poly_in = NULL;
    coDoLines *lines_in = NULL;
    coDoTriangleStrips *strips_in = NULL;
    // return object
    // variables for in/output-data & geometry
    float *in_data[NUM_DATA_IN_PORTS + 1][3];
    float *x_in = NULL, *y_in = NULL, *z_in = NULL;
    int *vl_in = NULL;
    int *pl_in = NULL;
    std::vector<float> x_out, y_out, z_out;
    std::vector<int> vl_out, pl_out;

    // speed optimized
    int *new_vert_id;
    float came_from_vert[3], came_from_data[NUM_DATA_IN_PORTS + 1][3], t;
    int element_to_add, last_vertex, came_from_vert_id;
    float temp_vect[3], *temp_vect2;

    // other variables
    int Comp;
    int geoSize = 0, num_poly = 0, num_vert = 0, num_coord = 0;
    int num_data_out[NUM_DATA_IN_PORTS + 1];

    int last;

    char bfr[300];

    // get the inport objects

    in_objs[0] = p_geo_in->getCurrentObject();

    for (int s = 1; s <= NUM_DATA_IN_PORTS; s++)
        in_objs[s] = p_data_in[s - 1]->getCurrentObject();

    const char *geoType = in_objs[0]->getType();

    int *vl, *tl;
    int next_strip;

    if (strcmp(geoType, "TRIANG") == 0) // convert triangle strips to polygons
    {
        int num_polygons = 0;
        strips_in = (coDoTriangleStrips *)in_objs[0];
        int num_vert_in = strips_in->getNumVertices();
        int num_strips = strips_in->getNumStrips();
        strips_in->getAddresses(&x_in, &y_in, &z_in, &vl, &tl);

        for (int i = 0; i < num_strips; i++) // get size of new arrays
            if (i != num_strips - 1)
                num_polygons += tl[i + 1] - tl[i] - 2;
            else
                num_polygons += num_vert_in - tl[i] - 2;
        int num_corners = num_polygons * 3; // all polygons are triangles

        vl_in = new int[num_corners];
        pl_in = new int[num_polygons];

        num_polygons = 0;
        num_corners = 0;

        for (int i = 0; i < num_strips; i++)
        {

            next_strip = (i == num_strips - 1) ? num_vert_in : tl[i + 1];
            for (int j = tl[i]; j <= next_strip - 3; j++)
            {
                pl_in[num_polygons++] = num_corners; // add polygon

                vl_in[num_corners++] = vl[j]; // copy corners of triangle
                if ((j - tl[i]) % 2)
                {
                    vl_in[num_corners++] = vl[j + 2];
                    vl_in[num_corners++] = vl[j + 1];
                }
                else
                {
                    vl_in[num_corners++] = vl[j + 1];
                    vl_in[num_corners++] = vl[j + 2];
                }
            }
        }

        geoSize = num_coord = strips_in->getNumPoints();
        num_vert = num_corners;
        num_poly = num_polygons;
    }

    if (strcmp(geoType, "POLYGN") == 0)
    {

        poly_in = (coDoPolygons *)in_objs[0];
        geoSize = num_coord = poly_in->getNumPoints();
        num_vert = poly_in->getNumVertices();
        num_poly = poly_in->getNumPolygons();
        poly_in->getAddresses(&x_in, &y_in, &z_in, &vl_in, &pl_in);
    }
    if (strcmp(geoType, "LINES") == 0)
    {

        lines_in = (coDoLines *)in_objs[0];
        geoSize = num_coord = lines_in->getNumPoints();
        num_vert = lines_in->getNumVertices();
        num_poly = lines_in->getNumLines();
        lines_in->getAddresses(&x_in, &y_in, &z_in, &vl_in, &pl_in);
    }

    if (strcmp(geoType, "POLYGN") != 0 && strcmp(geoType, "TRIANG") != 0
        && strcmp(geoType, "LINES") != 0)
    {

        sendError("ERROR: incompatible input - neither Lines, Polygons nor TriangleStrips");
        return (0);
    }

    ///// add different types here

    // get the input data
    const char *dataType[NUM_DATA_IN_PORTS];
    int dataSize[NUM_DATA_IN_PORTS];
    int numComp[NUM_DATA_IN_PORTS];

    // remember in_objs[0] contains the geometry
    for (int s = 0; s < NUM_DATA_IN_PORTS; s++)
    {

        if (in_objs[s + 1] != NULL)
        {
            dataType[s] = in_objs[s + 1]->getType();

            if (strcmp(dataType[s], "USTSDT") == 0)
            {
                numComp[s] = 1;
                s_data_in[s] = (coDoFloat *)in_objs[s + 1];
                s_data_in[s]->getAddress(&in_data[s][0]);
                dataSize[s] = s_data_in[s]->getNumPoints();
            }
            else if (strcmp(dataType[s], "USTVDT") == 0)
            {
                numComp[s] = 3;
                v_data_in[s] = (coDoVec3 *)in_objs[s + 1];
                v_data_in[s]->getAddresses(&in_data[s][0], &in_data[s][1], &in_data[s][2]);
                dataSize[s] = v_data_in[s]->getNumPoints();
            }
            else if (strcmp(dataType[s], "RGBADT") == 0)
            {
                numComp[s] = -1; // mark for RGBA interpolation
                c_data_in[s] = (coDoRGBA *)in_objs[s + 1];
                c_data_in[s]->getAddress((int **)(void *)&in_data[s][0]);
                dataSize[s] = c_data_in[s]->getNumPoints();
            }
            else
            {
                sendInfo("ERROR: data has to be S3D or V3D but is %s", dataType[s]);
                return (0); // return( NULL );
            }
        }
        else
        {
            dataSize[s] = 0;
            numComp[s] = 0;
        }
    }
    // Compute how much data we will need for output and temp-usage

    // we handle both...per vertex and per cell data
    // allocate enough memory to put the output-data & geo in
    new_vert_id = new int[num_coord];

    float **out_data[NUM_DATA_IN_PORTS];

    for (int s = 0; s < NUM_DATA_IN_PORTS; s++)
    {

        out_data[s] = new float *[3];
        for (int i = 0; i < 3; i++)
            out_data[s][i] = new float[dataSize[s] * 2];
    }

    temp_vect2 = new float[3];

    // reset
    for (int i = 0; i < num_coord; i++)
        new_vert_id[i] = -1;
    for (int s = 0; s < NUM_DATA_IN_PORTS; s++)
        num_data_out[s] = 0;

    if (p_method->getValue() == 0)
    {

        // get parameters and set standard parameter
        normal_A[0] = h_normal_of_plane_->getFValue(0); // normal_A can also be cylinder axis if geoMethod == 1
        normal_A[1] = h_normal_of_plane_->getFValue(1);
        normal_A[2] = h_normal_of_plane_->getFValue(2);

        cylBottom_A[0] = cylinder_bottom->getValue(0);
        cylBottom_A[1] = cylinder_bottom->getValue(1);
        cylBottom_A[2] = cylinder_bottom->getValue(2);

        // error check
        if ((normal_A[0] == 0) && (normal_A[1] == 0) && (normal_A[2] == 0))
        {
            sendError("ERROR: no normal for plane A given");
            return (0); // return NULL;
        }

        distance_A = h_distance_of_plane_->getFValue(); // distance_A can also be cylinder radius if geoMethod == 1
        float len;
        // get length
        len = sqrt(normal_A[0] * normal_A[0] + normal_A[1] * normal_A[1] + normal_A[2] * normal_A[2]);
        distance_A /= len;
        base_A[0] = normal_A[0] * distance_A;
        base_A[1] = normal_A[1] * distance_A;
        base_A[2] = normal_A[2] * distance_A;

        //  standard parameter
        algo_option = 1;
        smooth_option = 1;
        dir_option = 1;

        if (p_geoMethod->getValue() == 1)
        {
            algo_option = 2;
            sendInfo("using clylinder as geoMethod!");
        }
        else if (p_geoMethod->getValue() == 2)
        {
            algo_option = 3;
            sendInfo("using sphere as geoMethod!");
        }

        Interpol::reset(num_coord, &vl_out, &x_out, &y_out, &z_out, 2000);

        // work through the geometry
        // now you will need vitamin tablets!!!!
        if (strcmp(geoType, "POLYGN") == 0 || strcmp(geoType, "TRIANG") == 0 || strcmp(geoType, "LINES") == 0)
        {

            for (int poly_counter = 0; poly_counter < num_poly; poly_counter++)
            {

                int last_polygon = pl_out.size();
                // work through current polygon
                last_vertex = (poly_counter == num_poly - 1) ? num_vert : pl_in[poly_counter + 1];

                // begin with the last vertex
                int j = last_vertex - 1;
                last = to_remove_vertex(x_in[vl_in[j]], y_in[vl_in[j]], z_in[vl_in[j]]);

                // remember where we came from
                came_from_vert[0] = x_in[vl_in[j]];
                came_from_vert[1] = y_in[vl_in[j]];
                came_from_vert[2] = z_in[vl_in[j]];

                for (int s = 0; s < NUM_DATA_IN_PORTS; s++)
                {
                    if (dataSize[s] == geoSize)
                    {
                        // per-vert data so remember it
                        for (Comp = 0; Comp < abs(numComp[s]); Comp++)
                            came_from_data[s][Comp] = in_data[s][Comp][vl_in[j]];
                    }
                }
                came_from_vert_id = j;

                // we haven't yet added an element
                element_to_add = 1;

                int first = pl_in[poly_counter];

                bool wasInterrupted = false;
                for (int j = first; j < last_vertex; j++)
                {
                    // work through every vertex

                    temp_vect[0] = x_in[vl_in[j]];
                    temp_vect[1] = y_in[vl_in[j]];
                    temp_vect[2] = z_in[vl_in[j]];

                    int k = to_remove_vertex(temp_vect[0], temp_vect[1], temp_vect[2]);
                    if (k != last && (strcmp(geoType, "LINES") != 0 || j != first))
                    {
                        if (k)
                        {
                            wasInterrupted = true;
                        }
                        else if (wasInterrupted && strcmp(geoType, "LINES") == 0)
                        {
                            wasInterrupted = false;
                            element_to_add = 1;
                        }
                        // we might have to add a new element
                        if (element_to_add)
                        {
                            pl_out.push_back(vl_out.size());
                            element_to_add = 0;
                        }

                        // add new vertex
                        int n = Interpol::interpolated(vl_in[came_from_vert_id], vl_in[j]);
                        if (n)
                        {
                            vl_out.push_back(n - 1);
                        }

                        else
                        {
                            // we have to interpolate coordinates and data
                            t = line_with_plane(came_from_vert, temp_vect, &temp_vect2);

                            // add the data
                            for (int s = 0; s < NUM_DATA_IN_PORTS; s++)
                                if (dataSize[s] == geoSize)
                                {
                                    if (numComp[s] >= 0)
                                    {
                                        for (Comp = 0; Comp < numComp[s]; Comp++)
                                            out_data[s][Comp][num_data_out[s]] = came_from_data[s][Comp]
                                                                                 + t * (in_data[s][Comp][vl_in[j]] - came_from_data[s][Comp]);
                                    }
                                    else
                                    {
                                        out_data[s][0][num_data_out[s]] = interRGBA(came_from_data[s][0], in_data[s][0][vl_in[j]], t);
                                    }
                                    num_data_out[s]++;
                                }

                            // and the vertex
                            Interpol::add_vertex(vl_in[came_from_vert_id], vl_in[j], temp_vect2[0],
                                                 temp_vect2[1], temp_vect2[2]);
                        }
                        last = k;
                    }
                    else if (k != last)
                    {
                        last = k;
                    }

                    if (!k)
                    {
                        // now add the current vertex
                        k = new_vert_id[vl_in[j]];
                        if (k == -1)
                        {
                            k = x_out.size();
                            // num_coord_out++;
                            new_vert_id[vl_in[j]] = k;

                            // add the data
                            for (int s = 0; s < NUM_DATA_IN_PORTS; s++)
                                if (dataSize[s] == geoSize)
                                {
                                    for (Comp = 0; Comp < abs(numComp[s]); Comp++)
                                        out_data[s][Comp][num_data_out[s]] = in_data[s][Comp][vl_in[j]];
                                    num_data_out[s]++;
                                }

                            // store coordinates
                            x_out.push_back(temp_vect[0]);
                            y_out.push_back(temp_vect[1]);
                            z_out.push_back(temp_vect[2]);
                        }

                        // we might have to add a new element
                        if (element_to_add)
                        {
                            pl_out.push_back(vl_out.size());
                            element_to_add = 0;
                        }

                        // update vertex_list
                        vl_out.push_back(k);

                        // remember that the last one was kept
                        last = 0;
                    }

                    // remember where we come from
                    came_from_vert[0] = x_in[vl_in[j]];
                    came_from_vert[1] = y_in[vl_in[j]];
                    came_from_vert[2] = z_in[vl_in[j]];
                    for (int s = 0; s < NUM_DATA_IN_PORTS; s++)
                        if (dataSize[s] == geoSize)
                        {
                            // per-vert data so remember it
                            for (Comp = 0; Comp < abs(numComp[s]); Comp++)
                                came_from_data[s][Comp] = in_data[s][Comp][vl_in[j]];
                        }
                    came_from_vert_id = j;

                    // on to the next vertex
                }

                // keep per-cell data
                for (int s = 0; s < NUM_DATA_IN_PORTS; s++)
                    // if by chance dataSize[s] == geoSize and ==num_poly
                    // we assume data per vertex and not per cell
                    if (dataSize[s] != geoSize
                        && dataSize[s] == num_poly
                        && pl_out.size() != last_polygon)
                    {
                        for (Comp = 0; Comp < abs(numComp[s]); Comp++)
                            out_data[s][Comp][num_data_out[s]] = in_data[s][Comp][poly_counter];
                        num_data_out[s]++;
                    }

                // and the next polygon
            }
        }

        // generate the output objects
        if (strcmp(geoType, "POLYGN") == 0)
        {
            if (pl_out.size() == 0)
                geo_return = new coDoPolygons(p_geo_out->getObjName(), 0, 0, 0);
            else
                geo_return = new coDoPolygons(p_geo_out->getObjName(), x_out.size(),
                                              &x_out[0], &y_out[0], &z_out[0],
                                              vl_out.size(), &vl_out[0],
                                              pl_out.size(), &pl_out[0]);
            // copyAttributes( poly_in , geo_return);
            p_geo_out->setCurrentObject(geo_return);
            // we support VR here
            sprintf(bfr, "G%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
            // geo_return->addAttribute("FEEDBACK", bfr);
            setInteraction(bfr);
        }
        if (strcmp(geoType, "LINES") == 0)
        {
            if (pl_out.size() == 0)
                geo_return = new coDoLines(p_geo_out->getObjName(), 0, 0, 0);
            else
                geo_return = new coDoLines(p_geo_out->getObjName(), x_out.size(),
                                           &x_out[0], &y_out[0], &z_out[0],
                                           vl_out.size(), &vl_out[0], pl_out.size(), &pl_out[0]);
            // copyAttributes( poly_in , geo_return);
            p_geo_out->setCurrentObject(geo_return);
            // we support VR here
            sprintf(bfr, "G%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
            // geo_return->addAttribute("FEEDBACK", bfr);
            setInteraction(bfr);
        }

        if (strcmp(geoType, "TRIANG") == 0)
        {
            if (pl_out.size() == 0)
                geo_return = new coDoPolygons(p_geo_out->getObjName(), 0, 0, 0);
            else
                geo_return = new coDoPolygons(p_geo_out->getObjName(), x_out.size(),
                                              &x_out[0], &y_out[0], &z_out[0],
                                              vl_out.size(), &vl_out[0], pl_out.size(), &pl_out[0]);
            // copyAttributes( strips_in , geo_return);
            p_geo_out->setCurrentObject(geo_return);
            // we support VR here
            sprintf(bfr, "G%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
            // geo_return->addAttribute("FEEDBACK", bfr);
            setInteraction(bfr);
        }
        /// add other types here

        // data is objectType-independant
        for (int s = 0; s < NUM_DATA_IN_PORTS; s++)
        {
            if (in_objs[s + 1] != NULL) // remember in_objs[0] contains geometry
            {
                if (numComp[s] == 1)
                {
                    if (num_data_out[s] == 0)
                        data_return = new coDoFloat(p_data_out[s]->getObjName(), 0);
                    else
                        data_return = new coDoFloat(p_data_out[s]->getObjName(), num_data_out[s], out_data[s][0]);
                    // copyAttributes( s_data_in[s], data_return );
                    p_data_out[s]->setCurrentObject(data_return);
                }
                else if (numComp[s] == -1)
                {
                    if (num_data_out[s] == 0)
                        data_return = new coDoRGBA(p_data_out[s]->getObjName(), 0);
                    else
                        data_return = new coDoRGBA(p_data_out[s]->getObjName(), num_data_out[s], (int *)out_data[s][0]);
                    // copyAttributes( s_data_in[s], data_return );
                    p_data_out[s]->setCurrentObject(data_return);
                }
                else
                {
                    if (num_data_out[s] == 0)
                        data_return = new coDoVec3(p_data_out[s]->getObjName(), 0);
                    else
                        data_return = new coDoVec3(p_data_out[s]->getObjName(), num_data_out[s], out_data[s][0],
                                                   out_data[s][1], out_data[s][2]);
                    //copyAttributes(v_data_in[s], data_return );
                    p_data_out[s]->setCurrentObject(data_return);
                }
            }
        }
        // clean up
        Interpol::finished();
    }
    else // p_method == 1 (DataCut)
    {
        // work through the polygons / lines / triangles and see whether we need to remove them ...
        // in this case we just handle one data port to make sure that data and geometry fit to each other

        bool *elemtoremove = new bool[num_poly];
        memset(elemtoremove, 0, sizeof(bool));
        int remove;
        int n_verts;
        float data;
        float min = p_data_min->getValue();
        float max = p_data_max->getValue();

        int remove_counter = 0;

        if (numComp[0] == -1)
        {
            sendInfo("ERROR: we cannot handle color data with method=DataCut yet.");
            return (0);
        }

        // triangle strips have already been converted to polygons
        bool strictSelection = p_strictSelection->getValue();

        for (int i = 0; i < num_poly; i++)
        {
            remove = 0;
            n_verts = (i == num_poly - 1) ? num_vert - pl_in[i] : pl_in[i + 1] - pl_in[i];
            if (dataSize[0] == num_coord) // per vertex data
            {
                for (int j = 0; j < n_verts; j++)
                {
                    if (numComp[0] == 1) // scalar data
                    {
                        data = in_data[0][0][vl_in[pl_in[i] + j]];
                    }
                    else // vector data: we take the magnitude
                    {
                        data = sqrt(in_data[0][0][vl_in[pl_in[i] + j]] * in_data[0][0][vl_in[pl_in[i] + j]]
                                    + in_data[0][1][vl_in[pl_in[i] + j]] * in_data[0][1][vl_in[pl_in[i] + j]]
                                    + in_data[0][2][vl_in[pl_in[i] + j]] * in_data[0][2][vl_in[pl_in[i] + j]]);
                    }
                    if ((data < min) || (data > max))
                    {
                        remove++;
                    }
                }
                if (strictSelection)
                {
                    if (remove > 0)
                    {
                        elemtoremove[i] = 1;
                        remove_counter++;
                    }
                    else
                    {
                        elemtoremove[i] = 0;
                    }
                }
                else
                {
                    if (remove == n_verts)
                    {
                        elemtoremove[i] = 1;
                        remove_counter++;
                    }
                    else
                    {
                        elemtoremove[i] = 0;
                    }
                }
            }
            else // cell data
            {
                if (numComp[0] == 1)
                {
                    data = in_data[0][0][i]; // scalar data
                }
                else // vector data: we take the magnitude
                {
                    data = sqrt(in_data[0][0][i] * in_data[0][0][i]
                                + in_data[0][1][i] * in_data[0][1][i]
                                + in_data[0][2][i] * in_data[0][2][i]);
                }
                if ((data < min) || (data > max))
                {
                    elemtoremove[i] = 1;
                }
            }
        }
        fprintf(stderr, "polygons to remove: %d of %d\n", remove_counter, num_poly);

        if (p_invert_cut->getValue())
        {
            for (int i = 0; i < num_poly; i++)
            {
                elemtoremove[i] = !elemtoremove[i];
            }
        }

        // we want to know which vertexes are used
        int *nodeused = new int[num_coord];
        memset(nodeused, 0, num_coord * sizeof(int));
        for (int i = 0; i < num_poly; i++)
        {
            if (elemtoremove[i] == 0)
            {
                n_verts = (i == num_poly - 1) ? num_vert - pl_in[i] : pl_in[i + 1] - pl_in[i];
                for (int j = 0; j < n_verts; j++)
                {
                    nodeused[vl_in[pl_in[i] + j]]++;
                }
            }
        }

        // create coordinate list
        int n_removed = 0;
        for (int i = 0; i < num_coord; i++)
        {
            if (nodeused[i])
            {
                x_out.push_back(x_in[i]);
                y_out.push_back(y_in[i]);
                z_out.push_back(z_in[i]);
                nodeused[i] = i - n_removed;
                // nodeused can now be used to map old to new vertex IDs
            }
            else
            {
                n_removed++;
                nodeused[i] = -1; // node is not used any more ...
            }
        }

        int pos = 0;
        // now remove polygons / lines ...
        for (int i = 0; i < num_poly; i++)
        {
            if (elemtoremove[i] == 0)
            {
                // vl
                n_verts = (i == num_poly - 1) ? num_vert - pl_in[i] : pl_in[i + 1] - pl_in[i];
                for (int j = 0; j < n_verts; j++)
                {
                    if (nodeused[vl_in[pl_in[i] + j]] == -1)
                    {
                        sendInfo("ERROR in DataCut algorithm");
                        return (0);
                    }
                    // we need all vertices of elements that are not removed
                    vl_out.push_back(nodeused[vl_in[pl_in[i] + j]]);
                }
                // pl
                pl_out.push_back(pos);
                pos += n_verts;
            }
        }

        // copy data
        for (int s = 0; s < NUM_DATA_IN_PORTS; s++)
        {
            if (!p_data_in[s]->isConnected())
            {
                break;
            }

            if (numComp[s] == 1) // scalar data
            {
                for (int i = 0; i < num_coord; i++)
                {
                    if (nodeused[i] >= 0)
                    {
                        out_data[s][0][nodeused[i]] = in_data[s][0][i];
                        num_data_out[s]++;
                    }
                }
            }

            else // (numComp[0]==3) - vector data
            {
                for (int i = 0; i < num_coord; i++)
                {
                    if (nodeused[i] >= 0)
                    {
                        out_data[s][0][nodeused[i]] = in_data[s][0][i];
                        out_data[s][1][nodeused[i]] = in_data[s][1][i];
                        out_data[s][2][nodeused[i]] = in_data[s][2][i];
                        num_data_out[s]++;
                    }
                }
            }
        }

        if ((strcmp(geoType, "POLYGN") == 0) || (strcmp(geoType, "TRIANG") == 0))
        {
            if (pl_out.size() == 0)
                geo_return = new coDoPolygons(p_geo_out->getObjName(), 0, 0, 0);
            else
                geo_return = new coDoPolygons(p_geo_out->getObjName(), x_out.size(),
                                              &x_out[0], &y_out[0], &z_out[0],
                                              vl_out.size(), &vl_out[0],
                                              pl_out.size(), &pl_out[0]);
            // copyAttributes( poly_in , geo_return);
            p_geo_out->setCurrentObject(geo_return);
            // we support VR here
            sprintf(bfr, "G%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
            // geo_return->addAttribute("FEEDBACK", bfr);
            setInteraction(bfr);
        }

        // data is objectType-independant
        for (int s = 1; s < NUM_DATA_IN_PORTS; s++)
        {
            if (!p_data_in[s]->isConnected())
            {
                break;
            }

            if (in_objs[s + 1] != NULL) // remember in_objs[0] contains geometry
            {
                if (numComp[s] == 1)
                {
                    if (num_data_out[s] == 0)
                        data_return = new coDoFloat(p_data_out[s]->getObjName(), 0);
                    else
                        data_return = new coDoFloat(p_data_out[s]->getObjName(), num_data_out[s], out_data[s][0]);
                    // copyAttributes( s_data_in[s], data_return );
                    p_data_out[s]->setCurrentObject(data_return);
                }
                else if (numComp[s] == -1)
                {
                    if (num_data_out[s] == 0)
                        data_return = new coDoRGBA(p_data_out[s]->getObjName(), 0);
                    else
                        data_return = new coDoRGBA(p_data_out[s]->getObjName(), num_data_out[s], (int *)out_data[s][0]);
                    // copyAttributes( s_data_in[s], data_return );
                    p_data_out[s]->setCurrentObject(data_return);
                }
                else
                {
                    if (num_data_out[s] == 0)
                        data_return = new coDoVec3(p_data_out[s]->getObjName(), 0);
                    else
                        data_return = new coDoVec3(p_data_out[s]->getObjName(), num_data_out[s], out_data[s][0],
                                                   out_data[s][1], out_data[s][2]);
                    //copyAttributes(v_data_in[s], data_return );
                    p_data_out[s]->setCurrentObject(data_return);
                }
            }
        }
    }

    if (in_objs[1] != NULL) // remember in_objs[0] contains geometry
    {
        if (numComp[0] == 1)
        {
            if (num_data_out[0] == 0)
                data_return = new coDoFloat(p_data_out[0]->getObjName(), 0);
            else
                data_return = new coDoFloat(p_data_out[0]->getObjName(), num_data_out[0], out_data[0][0]);
            p_data_out[0]->setCurrentObject(data_return);
        }
        else
        {
            if (num_data_out[0] == 0)
                data_return = new coDoVec3(p_data_out[0]->getObjName(), 0);
            else
                data_return = new coDoVec3(p_data_out[0]->getObjName(), num_data_out[0], out_data[0][0],
                                           out_data[0][1], out_data[0][2]);
            p_data_out[0]->setCurrentObject(data_return);
        }
    }

    if (strcmp(geoType, "TRIANG") == 0)
    {
        delete[] vl_in;
        delete[] pl_in;
    }

    delete[] new_vert_id;

    for (int s = 0; s < NUM_DATA_IN_PORTS; s++)
    {
        for (int i = 0; i < 3; i++)
            delete[] out_data[s][i];
        delete[] out_data[s];
    }
    delete[] temp_vect2;
    return CONTINUE_PIPELINE;
}

int
CutGeometry::to_remove_vertex(float x, float y, float z)
{

    int r = 0;
    float temp_vector[3];
    float length;
    float d;

    // this function should return 0 or !0 if the given vertex
    // has to be removed or not

    switch (algo_option)
    { // single Plane
    case 1:

        temp_vector[0] = x - base_A[0];
        temp_vector[1] = y - base_A[1];
        temp_vector[2] = z - base_A[2];

        if (scalar_product(temp_vector, normal_A) > 0)
            r = 1;

        // inverse cutting
        r = !r;

        break;

    case 2: // cylinder

        // so far we support: cylinder axis through O(0|0|0)
        // d ... distance vertex - cylinder
        // a ... point on cylinder axis
        // u ... cylinder axis vector
        // d = |(x - a) x u|  / |u|

        temp_vector[0] = (y - cylBottom_A[1]) * normal_A[2] - (z - cylBottom_A[2]) * normal_A[1];
        temp_vector[1] = (z - cylBottom_A[2]) * normal_A[0] - (x - cylBottom_A[0]) * normal_A[2];
        temp_vector[2] = (x - cylBottom_A[0]) * normal_A[1] - (y - cylBottom_A[1]) * normal_A[0];
        length = sqrt(sqr(temp_vector[0]) + sqr(temp_vector[1]) + sqr(temp_vector[2]));
        d = length / (sqr(normal_A[0]) + sqr(normal_A[1]) + sqr(normal_A[2]));
        if (d < fabs(distance_A))
        {
            r = 1;
        }
        if (distance_A < 0.)
            r = !r;

        break;

    case 3: // sphere

        // d ... distance vertex - sphere center
        // a ... sphere center

        temp_vector[0] = (x - cylBottom_A[0]);
        temp_vector[1] = (y - cylBottom_A[1]);
        temp_vector[2] = (z - cylBottom_A[2]);
        d = sqrt(sqr(temp_vector[0]) + sqr(temp_vector[1]) + sqr(temp_vector[2]));
        if (d < fabs(distance_A))
        {
            r = 1;
        }
        if (distance_A < 0.)
            r = !r;

        break;

    default:

        sendInfo("ERROR: to_remove_vertex not yet implemented for this algorithm");
        return (0);
    }
    return (r);
}

float CutGeometry::scalar_product(float *a, float *b)
{
    float r;
    r = (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
    return (r);
}

float CutGeometry::line_with_plane(float *p1, float *p2, float **r)
{ // return intersection between line p1-p2 with planeA/B (dep. nr=1/2)
    float line_vect[3], temp_vect[3], t;
    int i;

    float dist[2]; // distance of p1/p2 to cylinder axis
    float length;

    if (algo_option == 1) // plane
    {
        for (i = 0; i < 3; i++)
        {
            line_vect[i] = p2[i] - p1[i];
            temp_vect[i] = base_A[i] - p1[i];
        }

        // compute t

        t = scalar_product(temp_vect, normal_A) / scalar_product(line_vect, normal_A);
    }
    else if (algo_option == 2) // algo_option == 2: cylinder
    {
        for (i = 0; i < 3; i++)
        {
            line_vect[i] = p2[i] - p1[i];
        }
        temp_vect[0] = (p1[1] - cylBottom_A[1]) * normal_A[2] - (p1[2] - cylBottom_A[2]) * normal_A[1];
        temp_vect[1] = (p1[2] - cylBottom_A[2]) * normal_A[0] - (p1[0] - cylBottom_A[0]) * normal_A[2];
        temp_vect[2] = (p1[0] - cylBottom_A[0]) * normal_A[1] - (p1[1] - cylBottom_A[1]) * normal_A[0];
        length = sqrt(sqr(temp_vect[0]) + sqr(temp_vect[1]) + sqr(temp_vect[2]));
        dist[0] = length / (sqr(normal_A[0]) + sqr(normal_A[1]) + sqr(normal_A[2]));
        temp_vect[0] = (p2[1] - cylBottom_A[1]) * normal_A[2] - (p2[2] - cylBottom_A[2]) * normal_A[1];
        temp_vect[1] = (p2[2] - cylBottom_A[2]) * normal_A[0] - (p2[0] - cylBottom_A[0]) * normal_A[2];
        temp_vect[2] = (p2[0] - cylBottom_A[0]) * normal_A[1] - (p2[1] - cylBottom_A[1]) * normal_A[0];
        length = sqrt(sqr(temp_vect[0]) + sqr(temp_vect[1]) + sqr(temp_vect[2]));
        dist[1] = length / (sqr(normal_A[0]) + sqr(normal_A[1]) + sqr(normal_A[2]));
        t = (fabs(distance_A) - dist[0]) / (dist[1] - dist[0]);
    }
    else // algo_option == 3: sphere
    {
        for (i = 0; i < 3; i++)
        {
            line_vect[i] = p2[i] - p1[i];
        }

        dist[0] = sqrt(sqr(p1[0] - cylBottom_A[0]) + sqr(p1[1] - cylBottom_A[1]) + sqr(p1[2] - cylBottom_A[2]));
        dist[1] = sqrt(sqr(p2[0] - cylBottom_A[0]) + sqr(p2[1] - cylBottom_A[1]) + sqr(p2[2] - cylBottom_A[2]));

        // compute t
        t = (dist[0] - fabs(distance_A)) / (dist[0] - dist[1]);
    }

    // compute the final coordinates
    for (i = 0; i < 3; i++)
        (*r)[i] = p1[i] + t * line_vect[i];

    // done
    return (t);
}

void CutGeometry::copyAttributesToOutObj(coInputPort **input_ports,
                                         coOutputPort **output_ports, int n)
{
    int i, j;
    const coDistributedObject *in_obj;
    coDistributedObject *out_obj;
    int num_attr;
    const char **attr_n, **attr_v;

    if (n >= NUM_DATA_IN_PORTS + 1)
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
            for (i = 0; i < NUM_DATA_IN_PORTS; i++)
            {
                if (input_ports[i + 1]->getCurrentObject() != NULL)
                {
                    out_obj->addAttribute("Probe2D", output_ports[i + 1]->getObjName());
                }
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
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

void Interpol::reset(int total_num_coord, std::vector<int> *vl_out, std::vector<float> *x_out, std::vector<float> *y_out, std::vector<float> *z_out,
                     int hsize = 2000)
{
    int i;

    // get parameters and store them
    vl_out_ = vl_out;
    x_coord_out_ = x_out;
    y_coord_out_ = y_out;
    z_coord_out_ = z_out;
    heap_blk_size = hsize;

    // init our environment
    vertexID = new record_struct *[total_num_coord];
    record_data = new record_struct[heap_blk_size];
    record_data[0].next = NULL;
    cur_heap_pos = 1;

    // we have interpolated no vertices at all
    for (i = 0; i < total_num_coord; i++)
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
        (*vl_out_).push_back(r - 1);
    }
    else
    { // no - we have to add them to the vertex-list, coordinates and
        // record it
        r = (*x_coord_out_).size(); // (*num_coord_out);
        add_record(v1, v2, r);

        (*x_coord_out_).push_back(x);
        (*y_coord_out_).push_back(y);
        (*z_coord_out_).push_back(z);

        (*vl_out_).push_back(r);
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

MODULE_MAIN(Filter, CutGeometry)
