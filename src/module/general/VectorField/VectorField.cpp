/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1998 RUS  **
 **                                                                        **
 ** Description:  COVISE TubeNew     application module                    **
 **                                                                        **
 **                                                                        **
 **                             (C) Vircinity 2000                         **
 **                                                                        **
 **                                                                        **
 ** Author:  Uwe Woessner, Sasha Cioringa                                  **
 **                                                                        **
 **                                                                        **
 ** Date: oct  2002 no reference systems (Sergio Leseduarte)               **
 ** Date: july 2001 reference systems (Sergio Leseduarte)                  **
 ** Date: june 2001 arrow points for vectors (Sergio Leseduarte)           **
 ** Date:  27.09.94  V1.0                                                  **
 ** Date:  08.11.00                                                        **
\**************************************************************************/

#include "VectorField.h"
#include <util/coviseCompat.h>
#include <do/coDoTriangleStrips.h>
#include <alg/coVectField.h>

VectField::VectField(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Draw Vector arrows")
{
    const char *ChoiseVal1[] = { "1*scale", "length*scale", "according_to_data" };
    const char *ChoiseVal2[] = { "on_the_bottom", "on_the_middle" };

    //parameters
    p_scale = addFloatSliderParam("scale", "Scale factor");
    p_scale->setValue(0.0, 1.0, 1.0);
    p_length = addChoiceParam("length", "Length of vectors");
    p_length->setValue(3, ChoiseVal1, 0);
    p_fasten = addChoiceParam("fasten", "on_the_bottom or on_the_middle");
    p_fasten->setValue(2, ChoiseVal2, 0);
    p_num_sectors = addInt32Param("num_sectors", "number of lines for line tip");
    p_num_sectors->setValue(0);
    p_arrow_head_factor = addFloatParam("arrow_head_factor", "Relative length of arrow head");
    p_arrow_head_factor->setValue(0.2f);
    p_arrow_head_angle = addFloatParam("arrow_head_angle", "Opening angle of arrow head");
    p_arrow_head_angle->setValue(9.5f);

    //ports
    p_inPort1 = addInputPort("meshIn", "StructuredGrid|RectilinearGrid|UniformGrid|Polygons|Lines|UnstructuredGrid|TriangleStrips|Points", "input mesh");
    p_inPort2 = addInputPort("vdataIn", "Vec3|Mat3", "input vector data");
    p_inPort3 = addInputPort("sdataIn", "Float", "input scalar data");
    p_inPort3->setRequired(0);
    p_outPort1 = addOutputPort("linesOut", "Lines", "Vectors (Lines)");
    p_outPort2 = addOutputPort("dataOut", "Float", "Data on arrows");
}

void VectField::fillRefLines(coDoMat3 *ur_data_in,
                             coDistributedObject **linesList,
                             int nume, int numv, int /*nump*/, int *l_l, int *v_l)
{
    int i, j;
    float origin[3];
    std::string xName = p_outPort1->getObjName();
    xName += "_X";
    std::string yName = p_outPort1->getObjName();
    yName += "_Y";
    std::string zName = p_outPort1->getObjName();
    zName += "_Z";
    linesList[0] = new coDoLines(xName, 2 * nume, 2 * nume, nume);
    linesList[1] = new coDoLines(yName, 2 * nume, 2 * nume, nume);
    linesList[2] = new coDoLines(zName, 2 * nume, 2 * nume, nume);
    linesList[0]->addAttribute("COLOR", "red");
    linesList[1]->addAttribute("COLOR", "green");
    linesList[2]->addAttribute("COLOR", "blue");
    // get lists for filling
    float *x_start[3], *y_start[3], *z_start[3];
    int *corner_list[3], *line_list[3];
    for (i = 0; i < 3; ++i)
    {
        ((coDoLines *)(linesList[i]))->getAddresses(&x_start[i], &y_start[i], &z_start[i], &corner_list[i], &line_list[i]);
    }
    // corner_lists and line_lists may already be filled
    for (i = 0; i < nume; ++i)
    {
        line_list[0][i] = 2 * i;
    }
    for (i = 0; i < 2 * nume; ++i)
    {
        corner_list[0][i] = i;
    }
    memcpy(line_list[1], line_list[0], nume * sizeof(int));
    memcpy(line_list[2], line_list[0], nume * sizeof(int));
    memcpy(corner_list[1], corner_list[0], 2 * nume * sizeof(int));
    memcpy(corner_list[2], corner_list[0], 2 * nume * sizeof(int));

    for (i = 0; i < nume; ++i)
    {
        int numnodes;
        if (l_l)
        {
            if (i < nume - 1)
            {
                numnodes = l_l[i + 1] - l_l[i];
            }
            else
            {
                numnodes = numv - l_l[i];
            }
        } // for points
        else
        {
            numnodes = 1;
        }
        origin[0] = origin[1] = origin[2] = 0.0;
        if (l_l && v_l)
        {
            for (j = l_l[i]; j < l_l[i] + numnodes; ++j)
            {
                origin[0] += x_in[v_l[j]];
                origin[1] += y_in[v_l[j]];
                origin[2] += z_in[v_l[j]];
            }
        } // for points
        else
        {
            origin[0] += x_in[i];
            origin[1] += y_in[i];
            origin[2] += z_in[i];
        }
        origin[0] /= numnodes;
        origin[1] /= numnodes;
        origin[2] /= numnodes;
        x_start[0][2 * i] = x_start[1][2 * i] = x_start[2][2 * i] = origin[0];
        y_start[0][2 * i] = y_start[1][2 * i] = y_start[2][2 * i] = origin[1];
        z_start[0][2 * i] = z_start[1][2 * i] = z_start[2][2 * i] = origin[2];
    }

    // extract the matrices from ur_data_in
    float *matrices;
    int two_i;
    ur_data_in->getAddress(&matrices);
    if (length_param == S_U)
    {
        for (two_i = 0; two_i < 2 * nume; two_i += 2) // length_param == S_V
        {
            x_start[0][two_i + 1] = x_start[0][two_i] + scale * (*(matrices++));
            x_start[1][two_i + 1] = x_start[1][two_i] + scale * (*(matrices++));
            x_start[2][two_i + 1] = x_start[2][two_i] + scale * (*(matrices++));
            y_start[0][two_i + 1] = y_start[0][two_i] + scale * (*(matrices++));
            y_start[1][two_i + 1] = y_start[1][two_i] + scale * (*(matrices++));
            y_start[2][two_i + 1] = y_start[2][two_i] + scale * (*(matrices++));
            z_start[0][two_i + 1] = z_start[0][two_i] + scale * (*(matrices++));
            z_start[1][two_i + 1] = z_start[1][two_i] + scale * (*(matrices++));
            z_start[2][two_i + 1] = z_start[2][two_i] + scale * (*(matrices++));
        }
    } // length_param == S_DATA
    else
    {
        float local_scale;
        float *s_in_copy = s_in;
        for (two_i = 0; two_i < 2 * nume; two_i += 2)
        {
            local_scale = scale * (*(s_in_copy++));
            x_start[0][two_i + 1] = x_start[0][two_i] + local_scale * (*(matrices++));
            x_start[1][two_i + 1] = x_start[1][two_i] + local_scale * (*(matrices++));
            x_start[2][two_i + 1] = x_start[2][two_i] + local_scale * (*(matrices++));
            y_start[0][two_i + 1] = y_start[0][two_i] + local_scale * (*(matrices++));
            y_start[1][two_i + 1] = y_start[1][two_i] + local_scale * (*(matrices++));
            y_start[2][two_i + 1] = y_start[2][two_i] + local_scale * (*(matrices++));
            z_start[0][two_i + 1] = z_start[0][two_i] + local_scale * (*(matrices++));
            z_start[1][two_i + 1] = z_start[1][two_i] + local_scale * (*(matrices++));
            z_start[2][two_i + 1] = z_start[2][two_i] + local_scale * (*(matrices++));
        }
    }
}

int VectField::compute(const char *)
{
    //  Shared memory data
    coDoVec3 *uv_data_in = NULL;
    coDoMat3 *ur_data_in = NULL;
    coDoFloat *us_data_in = NULL;
    coDoUniformGrid *u_grid_in = NULL;
    coDoRectilinearGrid *r_grid_in = NULL;
    coDoStructuredGrid *s_grid_in = NULL;
    coDoUnstructuredGrid *uns_grid_in = NULL;
    coDoPolygons *poly_grid_in = NULL;
    coDoPoints *point_grid_in = NULL;
    coDoTriangleStrips *strips_grid_in = NULL;
    coDoLines *lines_grid_in = NULL;
    coDoSet *lines_out_ref = NULL;

    int snumc = 0, vnumc = 0;
    int *dummy;
    int numc;

    scale = p_scale->getValue();
    length_param = p_length->getValue() + 1;
    fasten_param = p_fasten->getValue() + 1;
    num_sectors_ = p_num_sectors->getValue();
    arrow_head_factor_ = p_arrow_head_factor->getValue();
    arrow_head_angle_ = p_arrow_head_angle->getValue();

    if (num_sectors_ < 0)
    {
        sendWarning("The number of sectors may not be negative: assuming 0");
        p_num_sectors->setValue(0);
        num_sectors_ = 0;
    }

    const coDistributedObject *obj1 = p_inPort1->getCurrentObject();
    const coDistributedObject *obj2 = p_inPort2->getCurrentObject();
    const coDistributedObject *obj3 = p_inPort3->getCurrentObject();

    coVectField *vectfield;

    if (!obj2)
    {
        sendError("Did not receive object at port '%s'", p_inPort2->getName());
        return FAIL;
    }
    if (obj2->isType("USTVDT"))
    {
        uv_data_in = (coDoVec3 *)obj2;
        vnumc = uv_data_in->getNumPoints();
        uv_data_in->getAddresses(&u_in, &v_in, &w_in);
    }
    else if (obj2->isType("USTREF"))
    {
        ur_data_in = (coDoMat3 *)obj2;
        vnumc = ur_data_in->getNumPoints();
        ur_data_in->getAddress(&u_in);
    }
    else
    {
        sendError("Received illegal type at port '%s'", p_inPort2->getName());
        return FAIL;
    }

    if (!obj1)
    {
        sendError("Did not receive object at port '%s'", p_inPort1->getName());
        return FAIL;
    }

    if (obj1->isType("STRGRD"))
    {
        s_grid_in = (coDoStructuredGrid *)obj1;
        s_grid_in->getGridSize(&i_dim, &j_dim, &k_dim);
        s_grid_in->getAddresses(&x_in, &y_in, &z_in);
        numc = i_dim * j_dim * k_dim;
        vectfield = new coVectField(STR_GRD, x_in, y_in, z_in, u_in, v_in, w_in, i_dim, j_dim, k_dim);
    }
    else if (obj1->isType("UNSGRD"))
    {
        uns_grid_in = (coDoUnstructuredGrid *)obj1;
        uns_grid_in->getGridSize(&i_dim, &j_dim, &numc);
        uns_grid_in->getAddresses(&dummy, &dummy, &x_in, &y_in, &z_in);
        vectfield = new coVectField(numc, x_in, y_in, z_in, u_in, v_in, w_in);
    }
    else if (obj1->isType("POLYGN"))
    {
        poly_grid_in = (coDoPolygons *)obj1;
        numc = poly_grid_in->getNumPoints();
        poly_grid_in->getAddresses(&x_in, &y_in, &z_in, &dummy, &dummy);
        vectfield = new coVectField(numc, x_in, y_in, z_in, u_in, v_in, w_in);
    }
    else if (obj1->isType("POINTS"))
    {
        point_grid_in = (coDoPoints *)obj1;
        numc = point_grid_in->getNumPoints();
        point_grid_in->getAddresses(&x_in, &y_in, &z_in);
        vectfield = new coVectField(numc, x_in, y_in, z_in, u_in, v_in, w_in);
    }
    else if (obj1->isType("TRIANG"))
    {
        strips_grid_in = (coDoTriangleStrips *)obj1;
        numc = strips_grid_in->getNumPoints();
        strips_grid_in->getAddresses(&x_in, &y_in, &z_in, &dummy, &dummy);
        vectfield = new coVectField(numc, x_in, y_in, z_in, u_in, v_in, w_in);
    }
    else if (obj1->isType("LINES"))
    {
        lines_grid_in = (coDoLines *)obj1;
        numc = lines_grid_in->getNumPoints();
        lines_grid_in->getAddresses(&x_in, &y_in, &z_in, &dummy, &dummy);
        vectfield = new coVectField(numc, x_in, y_in, z_in, u_in, v_in, w_in);
    }
    else if (obj1->isType("RCTGRD"))
    {
        r_grid_in = (coDoRectilinearGrid *)obj1;
        r_grid_in->getGridSize(&i_dim, &j_dim, &k_dim);
        r_grid_in->getAddresses(&x_in, &y_in, &z_in);
        numc = i_dim * j_dim * k_dim;
        vectfield = new coVectField(RCT_GRD, x_in, y_in, z_in, u_in, v_in, w_in, i_dim, j_dim, k_dim);
    }
    else if (obj1->isType("UNIGRD"))
    {
        float min_max[6];
        u_grid_in = (coDoUniformGrid *)obj1;
        u_grid_in->getGridSize(&i_dim, &j_dim, &k_dim);
        u_grid_in->getMinMax(&min_max[0], &min_max[1], &min_max[2], &min_max[3], &min_max[4], &min_max[5]);
        numc = i_dim * j_dim * k_dim;
        vectfield = new coVectField(x_in, y_in, z_in, u_in, v_in, w_in, i_dim, j_dim, k_dim, min_max);
    }
    else
    {
        sendError("Received illegal type at port '%s'", p_inPort1->getName());
        return FAIL;
    }

    if (obj3)
    {
        if (obj3->isType("USTSDT"))
        {
            us_data_in = (coDoFloat *)obj3;
            snumc = us_data_in->getNumPoints();
            us_data_in->getAddress(&s_in);
            vectfield->setScalarInField(s_in);
        }
        else
        {
            sendWarning("Received illegal type at port '%s'", p_inPort3->getName());
            return FAIL;
        }
    }

    // sl: if data length == 0, output a dummy object
    if (((obj3)
         && length_param == S_DATA
         && snumc == 0)
        || vnumc == 0)
    {
        coDistributedObject *dummyObj = new coDoLines(p_outPort1->getObjName(), 0, 0, 0);
        p_outPort1->setCurrentObject(dummyObj);
        dummyObj = new coDoFloat(p_outPort2->getObjName(), 0);
        p_outPort2->setCurrentObject(dummyObj);
        return SUCCESS;
    }
    // check dimensions
    if (!ur_data_in)
    {
        if ((obj3) && (snumc != numc))
        {
            sendError("ERROR: Objects have different dimensions");
            return FAIL;
        }
        if (vnumc != numc)
        {
            sendError("ERROR: Grid and Vector Data have different dimensions");
            return FAIL;
        }
    }
    else // for references we want data per element
    {
        if ((obj3) && (snumc != i_dim))
        {
            sendError("ERROR: Scalar data has to be cell-based for references");
            return FAIL;
        }
        if (vnumc != i_dim)
        {
            sendError("ERROR: References have to be cell-based");
            return FAIL;
        }
    }

    // check choice parameter
    if (length_param == S_DATA && (!obj3))
    {
        length_param = S_V;
        sendWarning("WARNING: no scalar data present");
    }

    if (length_param == S_V && ur_data_in)
    {
        length_param = S_U;
        sendInfo("Mapping references: assuming length=1*scale");
    }

    // create output objects

    if (!ur_data_in)
    {
        coObjInfo out1(p_outPort1->getObjName());
        coObjInfo out2(p_outPort2->getObjName());
        vectfield->compute_vectorfields(scale, length_param, fasten_param, num_sectors_, arrow_head_factor_, arrow_head_angle_,
                                        p_outPort1->getObjName() ? &out1 : NULL,
                                        p_outPort2->getObjName() ? &out2 : NULL);

        p_outPort1->setCurrentObject(vectfield->get_obj_lines());
        if (numc == 0)
        {
            coDistributedObject *dummyObj = new coDoFloat(p_outPort2->getObjName(), 0); // work around only, TODO: Create Vector data if previous data has been vertex based or don't create a grid nor data but this is dangerous für timestep animations
            p_outPort2->setCurrentObject(dummyObj);
        }
        else
            p_outPort2->setCurrentObject(vectfield->get_obj_scalar());
    }
    else
    {
        // Reference systems
        // for the moment accept only unstr. grids or polygons
        // or lines or points in the first port
        // Variable naming was a bit chaotic... Use now other names
        int nume, numv, nump;
        if (obj1->isType("UNSGRD"))
        {
            uns_grid_in->getGridSize(&nume, &numv, &nump);
            uns_grid_in->getAddresses(&l_l, &v_l, &x_in, &y_in, &z_in);
        }
        else if (obj1->isType("POLYGN"))
        {
            nume = poly_grid_in->getNumPolygons();
            numv = poly_grid_in->getNumVertices();
            nump = poly_grid_in->getNumPoints();
            poly_grid_in->getAddresses(&x_in, &y_in, &z_in, &v_l, &l_l);
        }
        else if (obj1->isType("LINES"))
        {
            nume = lines_grid_in->getNumLines();
            numv = lines_grid_in->getNumVertices();
            nump = lines_grid_in->getNumPoints();
            lines_grid_in->getAddresses(&x_in, &y_in, &z_in, &v_l, &l_l);
        }
        else if (obj1->isType("POINTS"))
        {
            nume = numv = nump = point_grid_in->getNumPoints();
            v_l = l_l = 0;
        }
        else
        {
            sendError("Only UNSGRD, POLYGN, LINES or POINTS  accepted for references");
            return FAIL;
        }
        coDistributedObject *linesList[4];
        linesList[3] = 0;
        fillRefLines(ur_data_in, linesList, nume, numv, nump, l_l, v_l);
        lines_out_ref = new coDoSet(p_outPort1->getObjName(), linesList);
        p_outPort1->setCurrentObject(lines_out_ref);
    }
    return SUCCESS;
}

MODULE_MAIN(Mapper, VectField)
