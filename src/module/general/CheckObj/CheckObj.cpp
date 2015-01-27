/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                      (C)2000 Vircinity **
 **                                                                        **
 ** Description:  COVISE CheckObj application module                       **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:  Sasha Cioringa                                                **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Date:  28.10.00  V1.0                                                  **
 ** Last:                                                                  **
\**************************************************************************/

#include "CheckObj.h"

CheckObj::CheckObj(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Check Covise objects")
{
    // Parameters
    p_report = addBooleanParam("ReportAllErrors", "Report all errors of the object");
    p_report->setValue(0);

    // Ports
    p_inPort = addInputPort("GridIn0", "UnstructuredGrid|Polygons|Lines|TriangleStrips", "Input type");
}

int CheckObj::compute(const char *)
{
    final_error = 0;
    all_errors = p_report->getValue();

    const coDistributedObject *obj = p_inPort->getCurrentObject();

    if (!obj)
    {
        sendError("Did not receive object at port '%s'", p_inPort->getName());
        return FAIL;
    }
    if (dynamic_cast<const coDoUnstructuredGrid *>(obj))
    {
        if (check_unstructuredgrid((coDoUnstructuredGrid *)obj))
            sendInfo("The unstructured grid from the input port is correct!");
    }
    else if (dynamic_cast<const coDoPolygons *>(obj))
    {
        if (check_polygons((const coDoPolygons *)obj))
            sendInfo("The polygons list from the input port is correct!");
    }
    else if (dynamic_cast<const coDoLines *>(obj))
    {
        if (check_lines((const coDoLines *)obj))
            sendInfo("The lines list from the input port is correct!");
    }
    else if (dynamic_cast<const coDoTriangleStrips *>(obj))
    {
        if (check_trianglestrips((const coDoTriangleStrips *)obj))
            sendInfo("The triangle strips list from the input port is correct!");
    }
    else
    {
        sendError("Received illegal type at port '%s'", p_inPort->getName());
        return FAIL;
    }

    if (final_error)
        sendError("ERROR: At the 'Info Messages' box you have the list of errors!");

    return SUCCESS;
}

int CheckObj::check_unstructuredgrid(const coDoUnstructuredGrid *object)
{
    int i, j, k, diff;
    int correct = 1;

    object->getAddresses(&el, &cl, &x_coord, &y_coord, &z_coord);
    object->getTypeList(&tl);
    object->getGridSize(&n_el, &n_corners, &n_coord);
    if (!check_all(el, cl, n_el, n_corners, n_coord))
    {
        correct = 0;
        if (!all_errors)
            return correct;
    }
    for (i = 0; i < n_el - 1; i++)
    {
        switch (tl[i])
        {
        case TYPE_PRISM:
            diff = el[i + 1] - el[i];
            if (diff != 6)
            {
                correct = 0;
                if (!all_errors)
                {
                    sendError("ERROR: The number of points of the %dth element does not fit with the type PRISM!", i);
                    return correct;
                }
                else
                    sendInfo("ERROR: The number of points of the %dth element does not fit with the type PRISM!", i);
            }
            break;
        case TYPE_TETRAHEDER:
            diff = el[i + 1] - el[i];
            if (diff != 4)
            {
                correct = 0;
                if (!all_errors)
                {
                    sendError("ERROR: The number of points of the %dth element does not fit with the type TETRAHEDRON!", i);
                    return correct;
                }
                else
                    sendInfo("ERROR: The number of points of the %dth element does not fit with the type TETRAHEDRON!", i);
            }
            break;
        case TYPE_PYRAMID:
            diff = el[i + 1] - el[i];
            if (diff != 5)
            {
                correct = 0;
                if (!all_errors)
                {
                    sendError("ERROR: The number of points of the %dth element does not fit with the type PYRAMID!", i);
                    return correct;
                }
                else
                    sendInfo("ERROR: The number of points of the %dth element does not fit with the type PYRAMID!", i);
            }
            break;
        case TYPE_HEXAEDER:
            diff = el[i + 1] - el[i];
            if (diff != 8)
            {
                correct = 0;
                if (!all_errors)
                {
                    sendError("ERROR: The number of points of the %dth element does not fit with the type HEXAHEDRON!", i);
                    return correct;
                }
                else
                    sendInfo("ERROR: The number of points of the %dth element does not fit with the type HEXAHEDRON!", i);
            }
            break;
        case TYPE_QUAD:
            diff = el[i + 1] - el[i];
            if (diff != 4)
            {
                correct = 0;
                if (!all_errors)
                {
                    sendError("ERROR: The number of points of the %dth element does not fit with the type QUAD!", i);
                    return correct;
                }
                else
                    sendInfo("ERROR: The number of points of the %dth element does not fit with the type QUAD!", i);
            }
            break;
        case TYPE_TRIANGLE:
            diff = el[i + 1] - el[i];
            if (diff != 3)
            {
                correct = 0;
                if (!all_errors)
                {
                    sendError("ERROR: The number of points of the %dth element does not fit with the type TRIANGLE!", i);
                    return correct;
                }
                else
                    sendInfo("ERROR: The number of points of the %dth element does not fit with the type TRIANGLE!", i);
            }
            break;
        case TYPE_BAR:
            diff = el[i + 1] - el[i];
            if (diff != 2)
            {
                correct = 0;
                if (!all_errors)
                {
                    sendError("ERROR: The number of points of the %dth element does not fit with the type BAR!", i);
                    return correct;
                }
                else
                    sendInfo("ERROR: The number of points of the %dth element does not fit with the type BAR!", i);
            }
            break;
        case TYPE_POINT:
            diff = el[i + 1] - el[i];
            if (diff != 1)
            {
                correct = 0;
                if (!all_errors)
                {
                    sendError("ERROR: The number of points of the %dth element does not fit with the type POINT!", i);
                    return correct;
                }
                else
                    sendInfo("ERROR: The number of points of the %dth element does not fit with the type POINT!", i);
            }
            break;
        default:
            correct = 0;
            if (!all_errors)
            {
                sendError("The unstructed grid contains an incorrect element type!");
                return correct;
            }
            else
                sendInfo("The unstructed grid contains an incorrect element type!");
        }
    }

    if (!correct && all_errors)
        final_error = 1;
    else
        // an error found before can generate many false warnings
        for (i = 0; i < n_el - 1; i++)
            for (j = el[i]; j < el[i + 1] - 1; j++)
                for (k = j + 1; k < el[i + 1]; k++)
                    if (cl[j] == cl[k])
                    {
                        sendWarning("WARNING: The connectivity list contains the same point more then once for the %dth element in list ( cl [ %d ] = cl [ %d ] )", i, j, k);
                    }

    return correct;
}

int CheckObj::check_polygons(const coDoPolygons *object)
{
    object->getAddresses(&x_coord, &y_coord, &z_coord, &cl, &el);
    n_coord = object->getNumPoints();
    n_corners = object->getNumVertices();
    n_el = object->getNumPolygons();
    if (!check_all(el, cl, n_el, n_corners, n_coord))
        return 0;
    return 1;
}

int CheckObj::check_lines(const coDoLines *object)
{
    object->getAddresses(&x_coord, &y_coord, &z_coord, &cl, &el);
    n_coord = object->getNumPoints();
    n_corners = object->getNumVertices();
    n_el = object->getNumLines();
    if (!check_all(el, cl, n_el, n_corners, n_coord))
        return 0;
    return 1;
}

int CheckObj::check_trianglestrips(const coDoTriangleStrips *object)
{
    object->getAddresses(&x_coord, &y_coord, &z_coord, &cl, &el);
    n_coord = object->getNumPoints();
    n_corners = object->getNumVertices();
    n_el = object->getNumStrips();
    if (!check_all(el, cl, n_el, n_corners, n_coord))
        return 0;
    return 1;
}

int CheckObj::check_all(int *el, int *cl, int n_el, int n_corners, int n_coord)
{
    int i, correct = 1;

    for (i = 0; i < n_el; i++)
        if (el[i] < 0 || el[i] > n_corners - 1)
        {
            correct = 0;
            if (!all_errors)
            {
                sendError("ERROR: The %dth element in the list points out of the boundaries of the connectivity list!", i);
                return correct;
            }
            else
                sendInfo("ERROR: The %dth element in the list points out of the boundaries of the connectivity list!", i);
        }
    for (i = 0; i < n_corners; i++)
        if (cl[i] < 0 || cl[i] > n_coord - 1)
        {
            correct = 0;
            if (!all_errors)
            {
                sendError("ERROR: The %dth corner in the list points out of the boundaries of the coordinate list!", i);
                return correct;
            }
            else
                sendInfo("ERROR: The %dth corner in the list points out of the boundaries of the coordinate list!", i);
        }
    if (!correct && all_errors)
        final_error = 1;

    return correct;
}

MODULE_MAIN(Tools, CheckObj)
