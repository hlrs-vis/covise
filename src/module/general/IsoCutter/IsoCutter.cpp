/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2007      ++
// ++ Description:                                                        ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Siegfried Hodri                          ++
// ++                               VISENSO                               ++
// ++                           Nobelstrasse 15                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// this includes our own class's headers
#include "IsoCutter.h"

#include <alg/coColors.h>

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
IsoCutter::IsoCutter(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Crops polygons along iso lines")
{
    _min = FLT_MAX;
    _max = -FLT_MAX;

    // Ports
    p_inPolygons = addInputPort("inPolygons", "Polygons", "Polygon to cut");
    p_inData = addInputPort("inData", "Unstructured_S3D_Data", "Data of polygon vertices");

    p_outPolygons = addOutputPort("outPolygons", "Polygons", "Cropped polygon");
    p_outData = addOutputPort("outData", "Unstructured_S3D_Data", "Data of cropped polygon");

    p_isovalue = addFloatSliderParam("iso_value", "Iso value to cut along");
    p_isovalue->setValue(-1.0, 1.0f, 0.0);

    p_autominmax = addBooleanParam("auto_minmax", "Automatic minmax alignment to incoming data");
    p_autominmax->setValue(true);

    p_cutdown = addBooleanParam("cutoff_side", "Side to cut away");
    p_cutdown->setValue(true);
}

void IsoCutter::preHandleObjects(coInputPort **InPorts)
{
    (void)InPorts;

    if (p_inData->getCurrentObject() != NULL)
    {
        if (p_autominmax->getValue())
        {
            ScalarContainer scalarField;
            scalarField.Initialise(p_inData->getCurrentObject());
            scalarField.MinMax(_min, _max);
            if (_min <= _max)
            {
                float old_val = p_isovalue->getValue();
                if (_max < old_val)
                {
                    p_isovalue->setValue(_min, _max, _max);
                }
                else if (_min > old_val)
                {
                    p_isovalue->setValue(_min, _max, _min);
                }
                else
                {
                    p_isovalue->setValue(_min, _max, old_val);
                }
            }
        }
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int IsoCutter::compute(const char *)
{
    const coDistributedObject *obj;

    inPolygons = NULL;
    inData = NULL;
    outPolygons = NULL;
    outData = NULL;

    obj = p_inPolygons->getCurrentObject();
    if (!obj)
    {
        sendError("Did not receive object at port '%s'", p_inPolygons->getName());
        return FAIL;
    }

    // it should be the correct type
    if (obj->isType("POLYGN"))
    {
        inPolygons = (coDoPolygons *)p_inPolygons->getCurrentObject();
        num_in_coord_list = inPolygons->getNumPoints();
        num_in_corner_list = inPolygons->getNumVertices();
        num_in_poly_list = inPolygons->getNumPolygons();
        inPolygons->getAddresses(&in_x_coords, &in_y_coords, &in_z_coords,
                                 &in_corner_list, &in_poly_list);
    }
    else
    {
        sendError("Received illegal type at port '%s'", p_inPolygons->getName());
        return FAIL;
    }

    // check for unstructured s3d data
    obj = p_inData->getCurrentObject();
    if (!obj)
    {
        sendError("Did not receive object at port '%s'", p_inData->getName());
        return FAIL;
    }
    if (obj->isType("USTSDT"))
    {
        inData = (coDoFloat *)p_inData->getCurrentObject();
        num_in_data = inData->getNumPoints();
        inData->getAddress(&in_data);

        /*
      if (num_in_poly_list == num_in_data)
      {
         send_error("Incoming data must be per point and not per element.");
         return FAIL;
      }
      */
    }
    else
    {
        sendError("Received illegal type at port '%s'", p_inData->getName());
        return FAIL;
    }

    calc_cropped_polygons();

    outPolygons = new coDoPolygons(p_outPolygons->getObjName(), num_out_coord_list, out_x_coords, out_y_coords, out_z_coords,
                                   num_out_corner_list, out_corner_list, num_out_poly_list, out_poly_list);
    outData = new coDoFloat(p_outData->getObjName(), num_out_coord_list, out_data); // #Daten == #Polygonpunktkoordinaten

    p_outPolygons->setCurrentObject(outPolygons);
    p_outData->setCurrentObject(outData);

    return SUCCESS;
}

void IsoCutter::calc_cropped_polygons()
{
    float *values;
    int num_values;

    out_poly_list = new int[num_in_poly_list];
    // beim schneiden koennen mehr punkte entstehen, waehle obergrenze
    // Annahme: Polygone aus UnsGrd-Elementen sind 3- oder 4-Ecke
    out_corner_list = new int[(int)ceil(1.5f * num_in_corner_list)];
    out_x_coords = new float[(int)ceil(1.5f * num_in_coord_list)];
    out_y_coords = new float[(int)ceil(1.5f * num_in_coord_list)];
    out_z_coords = new float[(int)ceil(1.5f * num_in_coord_list)];
    out_data = new float[(int)ceil(1.5f * num_in_coord_list)];

    num_out_poly_list = 0;
    num_out_corner_list = 0;
    num_out_coord_list = 0;

    values = NULL;
    num_values = 0;

    // ueber alle reinkommenden polygonflaechen iterieren
    for (int i = 0; i < num_in_poly_list; i++)
    {
        // mein array fuer die werte der polygonecken ggf. anpassen und mit werten fuellen
        if (i == num_in_poly_list - 1)
        {
            num_values = num_in_corner_list - in_poly_list[i];
            delete[] values;
            values = new float[num_values];
        }
        else if (in_poly_list[i + 1] - in_poly_list[i] != num_values)
        {
            num_values = in_poly_list[i + 1] - in_poly_list[i];
            delete[] values;
            values = new float[num_values];
        }
        for (int j = 0; j < num_values; j++)
        {
            values[j] = in_data[in_corner_list[in_poly_list[i] + j]];
        }

        // returns true, if polygon could be cropped and was therefore added
        add_isocropped_polygon(i, num_values, values);
    }
    delete[] values;
}

/*
Returns true, if polygon could be cropped and was added to the output.
False, if polygon doesn't have an iso line it could be cut along.
*/
bool IsoCutter::add_isocropped_polygon(int poly, int num_values, float *values)
{
    // breche vorzeitig ab, wenn an den polygonkanten kein iso-uebergang existiert
    bool lower = false;
    bool higher = false;
    for (int j = 0; j < num_values; j++)
    {
        if (values[j] < p_isovalue->getValue())
            lower = true;
        if (values[j] > p_isovalue->getValue())
            higher = true;
    }
    if (!(lower && higher))
    {
        return false;
    }

    // neues polygon in liste aufnehmen
    out_poly_list[num_out_poly_list] = num_out_corner_list;
    num_out_poly_list++;

    // iteriere ueber alle kanten des polygons, aber behandle letzte kante gesondert
    for (unsigned int edge_begin = 0; edge_begin < num_values; edge_begin++)
    {
        // letzte kante gesondert behandeln, weil endpunkt wieder bei 0 anfaengt
        unsigned int edge_end;
        if (edge_begin == num_values - 1)
        {
            edge_end = 0;
        }
        else
        {
            edge_end = edge_begin + 1;
        }

        // startpunkt der kante hinzufuegen, wenn iso-wert passt
        bool eval;
        if (p_cutdown->getValue() == true)
        {
            eval = (values[edge_begin] >= p_isovalue->getValue());
        }
        else
        {
            eval = (values[edge_begin] <= p_isovalue->getValue());
        }
        if (eval)
        {
            out_corner_list[num_out_corner_list] = num_out_coord_list;
            num_out_corner_list++;

            out_data[num_out_coord_list] = values[edge_begin];
            //out_data[num_out_coord_list] = p_isovalue->getValue();

            out_x_coords[num_out_coord_list] = in_x_coords[in_corner_list[in_poly_list[poly] + edge_begin]];
            out_y_coords[num_out_coord_list] = in_y_coords[in_corner_list[in_poly_list[poly] + edge_begin]];
            out_z_coords[num_out_coord_list] = in_z_coords[in_corner_list[in_poly_list[poly] + edge_begin]];
            num_out_coord_list++;
        }

        // falls iso-uebergang zwischen start- und endpunkt der kante,
        // dann fuege zwischenliegenden punkt hinzu (lineare interpolation)
        if ((values[edge_begin] > p_isovalue->getValue() && values[edge_end] < p_isovalue->getValue())
            || (values[edge_begin] < p_isovalue->getValue() && values[edge_end] > p_isovalue->getValue()))
        {
            float alpha;
            alpha = fabs(p_isovalue->getValue() - values[edge_begin]) / fabs((values[edge_end] - values[edge_begin]));

            out_corner_list[num_out_corner_list] = num_out_coord_list;
            num_out_corner_list++;

            out_data[num_out_coord_list] = p_isovalue->getValue();

            out_x_coords[num_out_coord_list] = in_x_coords[in_corner_list[in_poly_list[poly] + edge_begin]]
                                               + alpha * (in_x_coords[in_corner_list[in_poly_list[poly] + edge_end]] - in_x_coords[in_corner_list[in_poly_list[poly] + edge_begin]]);
            out_y_coords[num_out_coord_list] = in_y_coords[in_corner_list[in_poly_list[poly] + edge_begin]]
                                               + alpha * (in_y_coords[in_corner_list[in_poly_list[poly] + edge_end]] - in_y_coords[in_corner_list[in_poly_list[poly] + edge_begin]]);
            out_z_coords[num_out_coord_list] = in_z_coords[in_corner_list[in_poly_list[poly] + edge_begin]]
                                               + alpha * (in_z_coords[in_corner_list[in_poly_list[poly] + edge_end]] - in_z_coords[in_corner_list[in_poly_list[poly] + edge_begin]]);
            num_out_coord_list++;
        }
    }

    return true;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  What's left to do for the Main program:
// ++++                                    create the module and start it
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int main(int argc, char *argv[])

{
    // create the module
    IsoCutter *application = new IsoCutter(argc, argv);

    // this call leaves with exit(), so we ...
    application->start(argc, argv);

    // ... never reach this point
    return 0;
}
