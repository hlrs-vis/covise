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
 ** Author:  R.Lang, Uwe Woessner, Sasha Cioringa                          **
 **                                                                        **
 **                                                                        **
 ** Date:  18.05.94  V1.0                                                  **
 ** Date:  05.09.98                                                        **
 ** Date:  08.11.00                                                        **
\**************************************************************************/

#include "Tube.h"
#include <util/coviseCompat.h>
#include <do/coDoData.h>

const int Tube::tri = 3;
const int Tube::open_cylinder = 0;
const int Tube::closed_cylinder = 1;
const int Tube::arrows = 2;

const float Tube::arr_radius_coef = 1.5f; //arrow radius = arr_radius_coef*radius
const float Tube::arr_angle = (float)(M_PI / 12); //halb angle of the arrow
const float Tube::arr_hmax = 0.9f; //the arrow can be <= arr_hmax*delta_length;

Tube::Tube(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Make tubes from lines")
{
    const char *ChoiseVal[] = { "open_cylinder", "closed_cylinder", "arrows" };

    //parameters
    p_radius = addFloatParam("Radius", "Radius of tubes");
    p_radius->setValue(0.1f);
    p_parts = addInt32Param("Parts", "No. of sides of the tubes");
    p_parts->setValue(6);
    p_option = addChoiceParam("Option", "open or closed cylinder");
    p_option->setValue(3, ChoiseVal, 1);

    p_limitradius = addBooleanParam("LimitRadius", "Limit Radius of Tubes?");
    p_limitradius->setValue(false);
    p_maxradius = addFloatParam("max_Radius", "Maximum allowed Radius of tubes");
    p_maxradius->setValue(1.0f);

    //ports
    p_inLines = addInputPort("Lines", "Lines", "lines");
    p_inData = addInputPort("Data", "Float|RGBA", "Data");
    p_inDiameter = addInputPort("Diameter", "Float", "BeamDiameter");
    p_inData->setRequired(0);
    p_inDiameter->setRequired(0);
    p_outPoly = addOutputPort("Tubes", "Polygons", "mantle of the tubes");
    p_outData = addOutputPort("DataOut", "Float|RGBA", "Data Out");
    p_outNormals = addOutputPort("Normals", "Vec3", "Normals");
    p_outData->setDependencyPort(p_inData);
}

int Tube::compute(const char *)
{
    int i, data_anz = 0, n;
    const char *COLOR = "COLOR";
    const char *colorn = "blue";

    int line_c, vertex_l;
    int n_vertex_l, vertex_00;
    int counter = 0;

    int n_total_point, n_total_vert, n_total_poly;

    coDoLines *line;
    coDoPolygons *tube_out;
    coDoVec3 *normals = NULL;

    coDoFloat *us_data_out = NULL;
    coDoFloat *us_data_in = NULL;
    s_out = NULL;
    s_in = NULL;

    coDoRGBA *color_data_out = NULL;
    coDoRGBA *color_data_in = NULL;
    colors_out = NULL;
    colors_in = NULL;

    radius = p_radius->getValue();
    maxradius = p_maxradius->getValue();
    limitradius = p_limitradius->getValue();
    ngon = p_parts->getValue();
    option_param = p_option->getValue();
    gennormals = p_outNormals->isConnected();
    /*
     if(gennormals && p_inDiameter->isConnected() ){
        gennormals = 0;
        sendWarning("normals are not generated if the beam diameter is modulated through the object at the diameter input port");
     }
   */

    if (ngon < 3)
    {
        sendInfo("less than 3 sides specified: using 3");
        ngon = 3;
    }

    if (ngon > 1024)
    {
        sendInfo("more than 1024 sides specified: using 1024");
        ngon = 1024;
    }

    for (i = 0; i < ngon; i++)
    {
        cos_array[i] = cos((float)(2.0f * M_PI * (float)i / (float)ngon));
        sin_array[i] = sin((float)(2.0f * M_PI * (float)i / (float)ngon));
    }

    const coDistributedObject *lineObj = p_inLines->getCurrentObject();
    const coDistributedObject *dataObj = p_inData->getCurrentObject();

    if (!lineObj)
    {
        sendError("Did not receive object at port '%s'", p_inLines->getName());
        return FAIL;
    }
    if (lineObj->isType("LINES"))
    {
        line = (coDoLines *)lineObj;
        npoint = line->getNumPoints();
        nvert = line->getNumVertices();
        nlines = line->getNumLines();
        if (npoint == 0 || nvert == 0)
            nlines = 0;
        line->getAddresses(&xl, &yl, &zl, &vl, &ll);
        nlinesWithVertices = 0;
        int unusedVertices = 0;
        for (int line_c = 0; line_c < nlines; line_c++)
        {
            if (line_c == (nlines - 1))
                n_vertex_l = nvert - *(ll + line_c);
            else
                n_vertex_l = *(ll + line_c + 1) - *(ll + line_c);
            if (n_vertex_l > 1)
            {
                nlinesWithVertices++;
            }
            else
            {
                unusedVertices += n_vertex_l;
            }
        }

        nt_poly = ngon * ((nvert - unusedVertices) - nlinesWithVertices);
        if (ngon == 2)
            nt_poly = (nvert - unusedVertices) - nlinesWithVertices;
        nt_vert = 4 * nt_poly;
        nt_point = (nvert - unusedVertices) * ngon;
        if ((colorn = line->getAttribute(COLOR)) == NULL)
        {
            colorn = "blue";
        }
    }
    else
    {
        sendError("Received illegal type at port '%s'", p_inLines->getName());
        return FAIL;
    }

    //lists length for Stoppers
    if (option_param == closed_cylinder)
    {
        na_point = 0;
        na_vert = 0;
        na_poly = 0;

        ns_point = ngon * nlinesWithVertices * 2;
        ns_vert = ngon * nlinesWithVertices * 2;
        ns_poly = nlinesWithVertices * 2;
    }

    else if (option_param == arrows)
    {
        na_point = ngon * nlinesWithVertices * 2;
        na_vert = nlinesWithVertices * ngon * tri;
        na_poly = nlinesWithVertices * ngon;

        ns_point = ngon * nlinesWithVertices * 2;
        ns_vert = ngon * nlinesWithVertices * 2;
        ns_poly = nlinesWithVertices * 2;
    }
    else
    {
        na_point = 0;
        na_vert = 0;
        na_poly = 0;

        ns_point = 0;
        ns_vert = 0;
        ns_poly = 0;
    }

    // mantle, stopper and arrows together
    n_total_point = nt_point + ns_point + na_point;
    n_total_vert = nt_vert + ns_vert + na_vert;
    n_total_poly = nt_poly + ns_poly + na_poly;

    s_in = NULL;
    s_in_changed = new float[nvert];

    colors_in = NULL;
    colors_in_changed = new int[nvert];

    if (dataObj)
    {
        if (dataObj->isType("USTSDT"))
        {
            us_data_in = (coDoFloat *)dataObj;
            data_anz = us_data_in->getNumPoints();
            us_data_in->getAddress(&s_in);
        }
        else if (dataObj->isType("RGBADT"))
        {
            color_data_in = (coDoRGBA *)dataObj;
            data_anz = color_data_in->getNumPoints();
            color_data_in->getAddress(&colors_in);
        }
        else
        {
            sendError("Received illegal type at port '%s'", p_inData->getName());
            return FAIL;
        }
    }

    if (lineObj->isType("LINES"))
    {
        if (data_anz == 0)
        {
            s_in = NULL;
            colors_in = NULL;
        }
        else if (data_anz == nlines)
        {
            // take the data of a line and copy it for every vertex of this line
            for (line_c = 0; line_c < nlines; line_c++)
            {
                if (line_c == (nlines - 1))
                    n_vertex_l = nvert - *(ll + line_c);
                else
                    n_vertex_l = *(ll + line_c + 1) - *(ll + line_c);
                // coping
                if (n_vertex_l > 1)
                    for (vertex_l = 0; vertex_l < n_vertex_l; vertex_l++)
                    {
                        vertex_00 = *(ll + line_c);
                        if (s_in != NULL)
                        {
                            *(s_in_changed + vertex_00 + vertex_l) = *(s_in + line_c);
                        }
                        else if (colors_in != NULL)
                        {
                            *(colors_in_changed + vertex_00 + vertex_l) = *(colors_in + line_c);
                        }
                        counter = counter + 1;
                    }
            }
        }
        else if (data_anz == nvert)
        {
            for (i = 0; i < nvert; i++)
            {
                if (s_in != NULL)
                {
                    *(s_in_changed + i) = *(s_in + i);
                }
                else if (colors_in != NULL)
                {
                    *(colors_in_changed + i) = *(colors_in + i);
                }
            }
        }
        else if (data_anz == npoint) // sl: do not forget the case
        {
            //     that the data is point-based
            for (i = 0; i < nvert; ++i)
            {
                if (s_in != NULL)
                {
                    s_in_changed[i] = s_in[vl[i]];
                }
                else if (colors_in != NULL)
                {
                    colors_in_changed[i] = colors_in[vl[i]];
                }
            }
        }
        else
        {
            s_in = NULL;
            colors_in = NULL;
            sendWarning("WARNING: Size of 'Data' does not match neither vertex nor lines");
        }

        //tube(mantle) out
        tube_out = new coDoPolygons(p_outPoly->getObjName(), n_total_point, n_total_vert, n_total_poly);
        if (!tube_out->objectOk())
        {
            sendError("Failed to create object '%s' for port '%s'", p_outPoly->getObjName(), p_outPoly->getName());
            return FAIL;
        }
        tube_out->getAddresses(&xt, &yt, &zt, &vt, &lt);

        if (s_in != NULL)
        {
            us_data_out = new coDoFloat(p_outData->getObjName(), n_total_point);
            if (!us_data_out->objectOk())
            {
                sendError("Failed to create object '%s' for port '%s'", p_outData->getObjName(), p_outData->getName());
                return FAIL;
            }
            us_data_out->getAddress(&s_out);
            //out data for the points of the mantle

            for (i = 0; i < nvert; i++)
            {
                for (n = 0; n < ngon; n++)
                    s_out[i * ngon + n] = s_in_changed[i];
            }
        }
        else if (colors_in != NULL)
        {
            color_data_out = new coDoRGBA(p_outData->getObjName(), n_total_point);
            if (!color_data_out->objectOk())
            {
                sendError("Failed to create object '%s' for port '%s'", p_outData->getObjName(), p_outData->getName());
                return FAIL;
            }
            color_data_out->getAddress(&colors_out);
            //out data for the points ot the mantle

            for (i = 0; i < nvert; i++)
            {
                for (n = 0; n < ngon; n++)
                    colors_out[i * ngon + n] = colors_in_changed[i];
            }
        }
        else if (dataObj) // output dummy data if there is an input object
        {
            if (dataObj->isType("USTSDT"))
            {
                us_data_out = new coDoFloat(p_outData->getObjName(), 0);
            }
            else if (dataObj->isType("RGBADT"))
            {
                color_data_out = new coDoRGBA(p_outData->getObjName(), 0);
            }
        }

        if (gennormals)
        {
            normals = new coDoVec3(p_outNormals->getObjName(), n_total_point);
            if (!normals->objectOk())
            {
                sendError("Failed to create object '%s' for port '%s'", p_outNormals->getObjName(), p_outNormals->getName());
                return FAIL;
            }
            normals->getAddresses(&xn, &yn, &zn);
        }

        create_tube();
        if (option_param == closed_cylinder)
            create_stopper();
        if (option_param == arrows)
        {
            create_arrow();
            create_stopper();
        }
        tube_out->addAttribute(COLOR, colorn);
        if (ngon == 2)
            tube_out->addAttribute("vertexOrder", "2");

        p_outPoly->setCurrentObject(tube_out);
        if (gennormals)
            p_outNormals->setCurrentObject(normals);

        if (us_data_out)
        {
            us_data_out->copyAllAttributes(dataObj);
            p_outData->setCurrentObject(us_data_out);
        }
        else if (color_data_out)
        {
            color_data_out->copyAllAttributes(dataObj);
            p_outData->setCurrentObject(color_data_out);
        }
    }

    delete[] s_in_changed;
    delete[] colors_in_changed;

    return SUCCESS;
}

void Tube::create_tube()
//============================================================================
// create the tube for one line patch
//============================================================================
{

    int i, li, nseg, ns, n, seg;
    // #define float double
    float l, ang, lmin, ang2 = 0, det;
    float xp, yp, zp;
    float xr1, yr1, zr1;
    float x_normal2, y_normal2, z_normal2;
    float xpd, ypd, zpd;
    float xpe, ype, zpe;
    float xpm, ypm, zpm;
    float tmpf;
    float x_last_normal, y_last_normal, z_last_normal;
    // #undef float
    ns = 0;

    const coDistributedObject *diameter = p_inDiameter->getCurrentObject();
    coDoFloat *s_diameter = NULL;
    float *diameters = NULL;
    float *Pdiameters = NULL;
    // check if diameter is OK and scalar and has the right length
    if (diameter == NULL)
    {
    }
    else if (!diameter->objectOk())
    {
        sendWarning("Data with diameters is not OK. It will be ignored");
    }
    else if (!diameter->isType("USTSDT"))
    {
        sendWarning("Data with diameters is not UNSSDT. It will be ignored");
    }
    else
    {
        s_diameter = (coDoFloat *)(diameter);
        if (s_diameter->getNumPoints()
            != nlines)
        {
            if (s_diameter->getNumPoints()
                != npoint)
            {
                sendWarning("Diameter data is expected per line or point. Diameter data is ignored.");
            }
            else
            {
                s_diameter->getAddress(&Pdiameters);
            }
        }
        else
        {
            s_diameter->getAddress(&diameters);
        }
    }

    for (li = 0; li < nlines; li++)
    {

        if (li == nlines - 1) // Last Line
            nseg = nvert - ll[li] - 1;
        else
            nseg = ll[li + 1] - ll[li] - 1;
        if (nseg > 0)
        {
            float *l_cos_array = new float[ngon];
            float *l_sin_array = new float[ngon];
            float l_radius = radius;
            memcpy(l_cos_array, cos_array, ngon * sizeof(float));
            memcpy(l_sin_array, sin_array, ngon * sizeof(float));
            if (diameters)
            {
                l_radius = radius * diameters[li];
            }
            if (Pdiameters)
            {
                l_radius = radius * 0.5f * Pdiameters[vl[ll[li]]];
            }
            if (limitradius)
            {
                if (l_radius > maxradius)
                    l_radius = maxradius;
            }
            for (i = 0; i < ngon; i++)
            {
                l_cos_array[i] *= l_radius;
                l_sin_array[i] *= l_radius;
            }

            ang2 = 0;

            // Begin of Line

            int iPos = vl[ll[li]];
            xp = xl[iPos];
            yp = yl[iPos];
            zp = zl[iPos];

            int iPos2 = vl[ll[li] + 1];
            xpd = xl[iPos2] - xp;
            ypd = yl[iPos2] - yp;
            zpd = zl[iPos2] - zp;

            l = sqrt(xpd * xpd + ypd * ypd + zpd * zpd);
            if (l > 0)
            {
                float il = 1.0f / l;
                xpd *= il;
                ypd *= il;
                zpd *= il;
            }
            else
            {
                xpd = 1.0;
                ypd = zpd = 0.0;
            }
            // xr1 yr1 zr1 are radial directions?
            if ((xpd != 0.0) && (ypd == 0.0) && (zpd == 0.0))
            {
                xr1 = 0.0;
                yr1 = -1.0;
                zr1 = 0.0;
            }
            else
            {
                xr1 = 0;
                yr1 = -zpd;
                zr1 = +ypd;
            }
            l = sqrt(xr1 * xr1 + yr1 * yr1 + zr1 * zr1);
            if (l > 0)
            {
                float il = 1.0f / l;
                xr1 *= il;
                yr1 *= il;
                zr1 *= il;
            }
            else
            {
                xr1 = 1.0;
                yr1 = zr1 = 0.0;
            }
            x_normal2 = (ypd * zr1) - (yr1 * zpd);
            y_normal2 = -(xpd * zr1) + (xr1 * zpd);
            z_normal2 = (xpd * yr1) - (xr1 * ypd);
            l = sqrt(x_normal2 * x_normal2 + y_normal2 * y_normal2 + z_normal2 * z_normal2);
            x_normal2 /= -l;
            y_normal2 /= -l;
            z_normal2 /= -l;

            if (gennormals)
            {
                for (i = 0; i < ngon; i++)
                {
                    xt[ns * ngon + i] = xp + l_cos_array[i] * xr1 + l_sin_array[i] * x_normal2;
                    yt[ns * ngon + i] = yp + l_cos_array[i] * yr1 + l_sin_array[i] * y_normal2;
                    zt[ns * ngon + i] = zp + l_cos_array[i] * zr1 + l_sin_array[i] * z_normal2;
                    xn[ns * ngon + i] = l_cos_array[i] * xr1 + l_sin_array[i] * x_normal2;
                    yn[ns * ngon + i] = l_cos_array[i] * yr1 + l_sin_array[i] * y_normal2;
                    zn[ns * ngon + i] = l_cos_array[i] * zr1 + l_sin_array[i] * z_normal2;
                }
            }
            else
            {
                for (i = 0; i < ngon; i++)
                {
                    xt[ns * ngon + i] = xp + l_cos_array[i] * xr1 + l_sin_array[i] * x_normal2;
                    yt[ns * ngon + i] = yp + l_cos_array[i] * yr1 + l_sin_array[i] * y_normal2;
                    zt[ns * ngon + i] = zp + l_cos_array[i] * zr1 + l_sin_array[i] * z_normal2;
                }
            }
            x_last_normal = x_normal2;
            y_last_normal = y_normal2;
            z_last_normal = z_normal2;

            // Middle Part
            for (seg = 1; seg < nseg; seg++)
            {

                if (Pdiameters)
                {
                    l_radius = radius * 0.5f * Pdiameters[vl[ll[li] + seg]];
                    if (limitradius)
                    {
                        if (l_radius > maxradius)
                            l_radius = maxradius;
                    }
                    for (i = 0; i < ngon; i++)
                    {
                        l_cos_array[i] *= l_radius;
                        l_sin_array[i] *= l_radius;
                    }
                }
                xp = xl[vl[ll[li] + seg]];
                yp = yl[vl[ll[li] + seg]];
                zp = zl[vl[ll[li] + seg]];
                xpd = xp - xl[vl[ll[li] + seg - 1]];
                ypd = yp - yl[vl[ll[li] + seg - 1]];
                zpd = zp - zl[vl[ll[li] + seg - 1]];
                xpe = xp - xl[vl[ll[li] + seg + 1]];
                ype = yp - yl[vl[ll[li] + seg + 1]];
                zpe = zp - zl[vl[ll[li] + seg + 1]];
                lmin = l = sqrt(xpd * xpd + ypd * ypd + zpd * zpd);
                if (l > 0)
                {
                    xpd /= l;
                    ypd /= l;
                    zpd /= l;
                }
                else
                {
                    xpd = 1;
                    ypd = zpd = 0;
                }
                l = sqrt(xpe * xpe + ype * ype + zpe * zpe);
                if (l < lmin)
                    lmin = l;
                if (l > 0)
                {
                    xpe /= l;
                    ype /= l;
                    zpe /= l;
                }
                else
                {
                    xpd = -1;
                    ypd = zpd = 0;
                }
                //if((l<0.01)||((xpd*xpe+ypd*ype+zpd*zpe)*l>lmin))

                xpm = xpd + xpe;
                ypm = ypd + ype;
                zpm = zpd + zpe;

                l = sqrt(xpm * xpm + ypm * ypm + zpm * zpm);
                tmpf = xpd * xpe + ypd * ype + zpd * zpe;
                if (tmpf > 1.0)
                    tmpf = 1.0;
                if (tmpf < -1.0)
                    tmpf = -1.0;
                ang = acos(tmpf) / 2;

                float eps = 1e-3f; // sl: restrict its use to the case when l is almost 0
                //     situation of parallel (not antiparallel lines)
                //     1e-5 may be too small if we compare
                //     angles in the sequel and calculate with floats...

                // sl: segments are not parallel
                if (eps < M_PI - 2.0 * ang && sin(ang) * lmin > 2.0 * l_radius)
                // nor too short (think in the
                // antiparallel case)
                // the first condition is
                // "effectively" equivalent to l>eps

                // if ((l != 0) && (l!=2)) // segments are not parallel
                // Comment RM: the original line above  does
                // not realy work because we are not doing paper and pencil
                // math
                {

                    xpm /= l;
                    ypm /= l;
                    zpm /= l;
                }
                else // sl:  segments are parallel or may be too short to covet a beautiful elbow
                //      (the latter could create a misshaped tube)
                {
                    xpm = x_last_normal;
                    ypm = y_last_normal;
                    zpm = z_last_normal;
                }

                x_normal2 = (-ypd * zpm) + (zpd * ypm);
                y_normal2 = (-zpd * xpm) + (xpd * zpm);
                z_normal2 = (-xpd * ypm) + (ypd * xpm);
                l = sqrt(x_normal2 * x_normal2 + y_normal2 * y_normal2 + z_normal2 * z_normal2);
                x_normal2 /= l;
                y_normal2 /= l;
                z_normal2 /= l;

                if (eps < M_PI - 2.0 * ang && sin(ang) * lmin > 2.0 * l_radius)
                {
                    // sl: this is now done subject to this condition,
                    //     otherwise the vector (xpm,ypm,zpm) might be too large;
                    //     this may happen when ang is very small.
                    //     If the condition is not satisfied, (xpm,ypm,zpm)
                    //     preserves its value given by the last normal.
                    l = 1 / sin(ang);
                    xpm *= l;
                    ypm *= l;
                    zpm *= l;
                }

                // Determinante berechnen
                det = (xpd * y_normal2 * z_last_normal) - (x_last_normal * y_normal2 * zpd) + (x_normal2 * y_last_normal * zpd) - (xpd * y_last_normal * z_normal2) + (x_last_normal * ypd * z_normal2) - (x_normal2 * ypd * z_last_normal);
                tmpf = x_last_normal * x_normal2 + y_last_normal * y_normal2 + z_last_normal * z_normal2;
                if (tmpf > 1.0)
                    tmpf = 1.0;
                if (tmpf < -1.0)
                    tmpf = -1.0;

                /*			if(tmpf<=1.0 && tmpf>=-1.0)
         {*/
                if (det < 0)
                    ang2 += acos(tmpf);
                else if (det >= 0)
                    ang2 += (float)((2 * M_PI) - acos(tmpf));
                //}

                if (ang2 != 0.0)
                {
                    if (gennormals)
                    {
                        for (i = 0; i < ngon; i++)
                        {
                            xn[i + (seg + ns) * ngon] = (float)((l_radius * cos((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * xpm + (l_radius * sin((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * x_normal2);
                            yn[i + (seg + ns) * ngon] = (float)((l_radius * cos((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * ypm + (l_radius * sin((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * y_normal2);
                            zn[i + (seg + ns) * ngon] = (float)((l_radius * cos((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * zpm + (l_radius * sin((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * z_normal2);
                            xt[i + (seg + ns) * ngon] = xp + xn[i + (seg + ns) * ngon];
                            yt[i + (seg + ns) * ngon] = yp + yn[i + (seg + ns) * ngon];
                            zt[i + (seg + ns) * ngon] = zp + zn[i + (seg + ns) * ngon];
                        }
                    }
                    else
                    {
                        for (i = 0; i < ngon; i++)
                        {
                            xt[i + (seg + ns) * ngon] = (float)(xp + (l_radius * cos((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * xpm + (l_radius * sin((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * x_normal2);
                            yt[i + (seg + ns) * ngon] = (float)(yp + (l_radius * cos((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * ypm + (l_radius * sin((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * y_normal2);
                            zt[i + (seg + ns) * ngon] = (float)(zp + (l_radius * cos((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * zpm + (l_radius * sin((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * z_normal2);
                        }
                    }
                }
                else // ang == 0
                {
                    if (gennormals)
                    {
                        for (i = 0; i < ngon; i++)
                        {
                            xt[i + (seg + ns) * ngon] = xp + l_cos_array[i] * xpm + l_sin_array[i] * x_normal2;
                            yt[i + (seg + ns) * ngon] = yp + l_cos_array[i] * ypm + l_sin_array[i] * y_normal2;
                            zt[i + (seg + ns) * ngon] = zp + l_cos_array[i] * zpm + l_sin_array[i] * z_normal2;
                            xn[i + (seg + ns) * ngon] = l_cos_array[i] * xpm + l_sin_array[i] * x_normal2;
                            yn[i + (seg + ns) * ngon] = l_cos_array[i] * ypm + l_sin_array[i] * y_normal2;
                            zn[i + (seg + ns) * ngon] = l_cos_array[i] * zpm + l_sin_array[i] * z_normal2;
                        }
                    }
                    else
                    {
                        for (i = 0; i < ngon; i++)
                        {
                            xt[i + (seg + ns) * ngon] = xp + l_cos_array[i] * xpm + l_sin_array[i] * x_normal2;
                            yt[i + (seg + ns) * ngon] = yp + l_cos_array[i] * ypm + l_sin_array[i] * y_normal2;
                            zt[i + (seg + ns) * ngon] = zp + l_cos_array[i] * zpm + l_sin_array[i] * z_normal2;
                        }
                    }
                }
                x_last_normal = x_normal2;
                y_last_normal = y_normal2;
                z_last_normal = z_normal2;
            }

            // End of Line

            if (Pdiameters)
            {
                l_radius = radius * 0.5f * Pdiameters[vl[ll[li] + nseg]];
                if (limitradius)
                {
                    if (l_radius > maxradius)
                        l_radius = maxradius;
                }
                for (i = 0; i < ngon; i++)
                {
                    l_cos_array[i] *= l_radius;
                    l_sin_array[i] *= l_radius;
                }
            }
            xp = xl[vl[ll[li] + nseg]];
            yp = yl[vl[ll[li] + nseg]];
            zp = zl[vl[ll[li] + nseg]];
            xpd = xp - xl[vl[ll[li] + nseg - 1]];
            ypd = yp - yl[vl[ll[li] + nseg - 1]];
            zpd = zp - zl[vl[ll[li] + nseg - 1]];
            l = sqrt(xpd * xpd + ypd * ypd + zpd * zpd);
            if (l > 0)
            {
                xpd /= l;
                ypd /= l;
                zpd /= l;
            }
            else
            {
                xpd = 1;
                ypd = zpd = 0;
            }
            if ((xpd != 0.0) && (ypd == 0.0) && (zpd == 0.0))
            {
                xr1 = 0.0;
                if (nseg == 1)
                    yr1 = -1.0;
                else
                    yr1 = 1.0;
                zr1 = 0.0;
            }
            else
            {
                if (nseg == 1)
                {
                    xr1 = 0;
                    yr1 = -zpd;
                    zr1 = ypd;
                }
                else
                {
                    xr1 = 0;
                    yr1 = zpd;
                    zr1 = -ypd;
                }
                l = sqrt(xr1 * xr1 + yr1 * yr1 + zr1 * zr1);
                if (l > 0)
                {
                    xr1 /= l;
                    yr1 /= l;
                    zr1 /= l;
                }
                else
                {
                    xr1 = 1.0;
                    yr1 = zr1 = 0.0;
                }
            }
            x_normal2 = (ypd * zr1) - (yr1 * zpd);
            y_normal2 = -(xpd * zr1) + (xr1 * zpd);
            z_normal2 = (xpd * yr1) - (xr1 * ypd);

            l = sqrt(x_normal2 * x_normal2 + y_normal2 * y_normal2 + z_normal2 * z_normal2);
            x_normal2 /= -l;
            y_normal2 /= -l;
            z_normal2 /= -l;
            // Determinante berechnen
            det = (xpd * y_normal2 * z_last_normal) - (x_last_normal * y_normal2 * zpd) + (x_normal2 * y_last_normal * zpd) - (xpd * y_last_normal * z_normal2) + (x_last_normal * ypd * z_normal2) - (x_normal2 * ypd * z_last_normal);
            tmpf = (x_last_normal * x_normal2 + y_last_normal * y_normal2 + z_last_normal * z_normal2);
            if (tmpf > 1.0)
                tmpf = 1.0;
            if (tmpf < -1.0)
                tmpf = -1.0;
            //if(tmpf<=1.0 && tmpf>=-1.0)
            //{
            if (det < 0)
                ang2 += acos(tmpf);
            else if (det >= 0)
                ang2 += (float)((2 * M_PI) - acos(tmpf));
            //}

            if (ang2 != 0.0)
            {
                if (gennormals)
                {
                    for (i = 0; i < ngon; i++)
                    {
                        xn[i + (nseg + ns) * ngon] = (float)((l_radius * (float)cos((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * xr1 + (l_radius * sin((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * x_normal2);
                        yn[i + (nseg + ns) * ngon] = (float)((l_radius * (float)cos((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * yr1 + (l_radius * sin((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * y_normal2);
                        zn[i + (nseg + ns) * ngon] = (float)((l_radius * (float)cos((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * zr1 + (l_radius * sin((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * z_normal2);
                        xt[i + (nseg + ns) * ngon] = xp + xn[i + (nseg + ns) * ngon];
                        yt[i + (nseg + ns) * ngon] = yp + yn[i + (nseg + ns) * ngon];
                        zt[i + (nseg + ns) * ngon] = zp + zn[i + (nseg + ns) * ngon];
                    }
                }
                else
                {
                    for (i = 0; i < ngon; i++)
                    {
                        xt[i + (nseg + ns) * ngon] = (float)(xp + (l_radius * (float)cos((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * xr1 + (l_radius * sin((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * x_normal2);
                        yt[i + (nseg + ns) * ngon] = (float)(yp + (l_radius * (float)cos((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * yr1 + (l_radius * sin((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * y_normal2);
                        zt[i + (nseg + ns) * ngon] = (float)(zp + (l_radius * (float)cos((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * zr1 + (l_radius * sin((2.0 * M_PI * (float)i / (float)ngon) + ang2)) * z_normal2);
                    }
                }
            }
            else
            {
                if (gennormals)
                {
                    for (i = 0; i < ngon; i++)
                    {
                        xt[i + (nseg + ns) * ngon] = xp + l_cos_array[i] * xr1 + l_sin_array[i] * x_normal2;
                        yt[i + (nseg + ns) * ngon] = yp + l_cos_array[i] * yr1 + l_sin_array[i] * y_normal2;
                        zt[i + (nseg + ns) * ngon] = zp + l_cos_array[i] * zr1 + l_sin_array[i] * z_normal2;
                        xn[i + (nseg + ns) * ngon] = l_cos_array[i] * xr1 + l_sin_array[i] * x_normal2;
                        yn[i + (nseg + ns) * ngon] = l_cos_array[i] * yr1 + l_sin_array[i] * y_normal2;
                        zn[i + (nseg + ns) * ngon] = l_cos_array[i] * zr1 + l_sin_array[i] * z_normal2;
                    }
                }
                else
                {
                    for (i = 0; i < ngon; i++)
                    {
                        xt[i + (nseg + ns) * ngon] = xp + l_cos_array[i] * xr1 + l_sin_array[i] * x_normal2;
                        yt[i + (nseg + ns) * ngon] = yp + l_cos_array[i] * yr1 + l_sin_array[i] * y_normal2;
                        zt[i + (nseg + ns) * ngon] = zp + l_cos_array[i] * zr1 + l_sin_array[i] * z_normal2;
                    }
                }
            }
            ns += nseg + 1;

            delete[] l_cos_array;
            delete[] l_sin_array;
        }
    }

    if (ngon == 2)
    {
        // create vertex list
        ns = 0;
        int line = 0;
        for (li = 0; li < nlines; li++)
        {
            if (li == nlines - 1) // Last Line
                nseg = nvert - ll[li] - 1;
            else
                nseg = ll[li + 1] - ll[li] - 1;
            for (i = 0; i < nseg; i++)
            {
                n = 0;
                *(vt + (i + ns) * 4 + (n * 4) + 3) = (ns + line + i) * ngon + n;
                *(vt + (i + ns) * 4 + (n * 4) + 2) = (ns + line + i) * ngon + 1 + n;
                *(vt + (i + ns) * 4 + (n * 4) + 1) = (ns + line + i) * ngon + ngon + n + 1;
                *(vt + (i + ns) * 4 + (n * 4)) = (ns + line + i) * ngon + ngon + n;
            }
            if (nseg > 0)
            {
                line++;
                ns += nseg;
            }
        }
    }
    else
    {
        // create vertex list
        ns = 0;
        int line = 0;
        for (li = 0; li < nlines; li++)
        {
            if (li == nlines - 1) // Last Line
                nseg = nvert - ll[li] - 1;
            else
                nseg = ll[li + 1] - ll[li] - 1;
            for (i = 0; i < nseg; i++)
            {
                for (n = 0; n < (ngon - 1); n++)
                {
                    *(vt + (i + ns) * ngon * 4 + (n * 4) + 3) = (ns + line + i) * ngon + n;
                    *(vt + (i + ns) * ngon * 4 + (n * 4) + 2) = (ns + line + i) * ngon + 1 + n;
                    *(vt + (i + ns) * ngon * 4 + (n * 4) + 1) = (ns + line + i) * ngon + ngon + n + 1;
                    *(vt + (i + ns) * ngon * 4 + (n * 4)) = (ns + line + i) * ngon + ngon + n;
                }
                *(vt + (i + ns) * ngon * 4 + (n * 4) + 3) = (ns + line + i) * ngon + ngon - 1;
                *(vt + (i + ns) * ngon * 4 + (n * 4) + 2) = (ns + line + i) * ngon;
                *(vt + (i + ns) * ngon * 4 + (n * 4) + 1) = (ns + line + i) * ngon + ngon;
                *(vt + (i + ns) * ngon * 4 + (n * 4)) = (ns + line + i) * ngon + ngon + ngon - 1;
            }

            if (nseg > 0)
            {
                line++;
                ns += nseg;
            }
        }
    };

    // create polygon list for the mantlles
    for (i = 0; i < nt_poly; i++)
    {
        *(lt + i) = i * 4;
    };

    //	Create vertex list for the polygons to close the cylinder (stoppers)
}

void Tube::create_stopper()
//============================================================================
// create the 2 stoppers for every line
//============================================================================
{
    const float arr_radius = arr_radius_coef * radius;
    int line_c, counter;
    int point_source1, point_source2;
    int point_goal1, point_goal2 = 0;

    int vertice_source1, vertice_source2; //vertices of the input lines
    float xsn1, ysn1, zsn1, l_xyzn;
    float xsn2, ysn2, zsn2;
    float xsnu1, ysnu1, zsnu1;
    float xsnu2, ysnu2, zsnu2;
    float arr_high = 0.0;
    float x_temp, y_temp, z_temp;
    int line = 0;

    for (line_c = 0; line_c < nlines; line_c++)
    {
        int nseg = 0;
        if (line_c == nlines - 1) // Last Line
            nseg = nvert - ll[line_c] - 1;
        else
            nseg = ll[line_c + 1] - ll[line_c] - 1;
        if (nseg > 0)
        {

            vertice_source1 = ll[line_c];
            if (line_c < (nlines - 1))
            {
                vertice_source2 = ll[line_c + 1] - 1;
            }
            else // if it is the last line
            {
                vertice_source2 = nvert - 1;
            }

            //vector parallel to line
            xsn1 = xl[vertice_source1] - xl[vertice_source1 + 1];
            ysn1 = yl[vertice_source1] - yl[vertice_source1 + 1];
            zsn1 = zl[vertice_source1] - zl[vertice_source1 + 1];
            l_xyzn = sqrt(xsn1 * xsn1 + ysn1 * ysn1 + zsn1 * zsn1);
            if (l_xyzn > 0)
            {
                xsnu1 = xsn1 / l_xyzn;
                ysnu1 = ysn1 / l_xyzn;
                zsnu1 = zsn1 / l_xyzn;
            }
            else
            {
                //Haeh?????????
                //	xsnu1=1/sqrt(1*1+2*2+3*3);
                //	ysnu1=2/sqrt(1*1+2*2+3*3);
                //	zsnu1=3/sqrt(1*1+2*2+3*3);
                xsnu1 = 1 / sqrt(15.0f);
                ysnu1 = 2 / sqrt(15.0f);
                zsnu1 = 3 / sqrt(15.0f);
            }

            //vector parallel to line
            xsn2 = xl[vertice_source2] - xl[vertice_source2 - 1];
            ysn2 = yl[vertice_source2] - yl[vertice_source2 - 1];
            zsn2 = zl[vertice_source2] - zl[vertice_source2 - 1];
            l_xyzn = sqrt(xsn2 * xsn2 + ysn2 * ysn2 + zsn2 * zsn2);
            if (l_xyzn > 0)
            {
                xsnu2 = xsn2 / l_xyzn;
                ysnu2 = ysn2 / l_xyzn;
                zsnu2 = zsn2 / l_xyzn;
            }
            else
            {
                xsnu2 = 1 / sqrt(15.0f);
                ysnu2 = 2 / sqrt(15.0f);
                zsnu2 = 3 / sqrt(15.0f);
                //	xsnu2=1/sqrt(1*1+2*2+3*3);
                //	ysnu2=2/sqrt(1*1+2*2+3*3);
                //	zsnu2=3/sqrt(1*1+2*2+3*3);
            }

            if (option_param == arrows)
            {
                arr_high = arr_radius / tan(arr_angle);
                if (arr_high > l_xyzn * arr_hmax)
                {
                    arr_high = l_xyzn * arr_hmax;
                }
                xsnu2 = -xsnu2;
                ysnu2 = -ysnu2;
                zsnu2 = -zsnu2;
            }

            for (counter = 0; counter < ngon; counter++)
            {
                point_source1 = vertice_source1 * ngon + counter;
                point_goal1 = nt_point + na_point + line * 2 * ngon + counter;

                point_source2 = vertice_source2 * ngon + counter;

                xt[point_goal1] = xt[point_source1];
                yt[point_goal1] = yt[point_source1];
                zt[point_goal1] = zt[point_source1];

                if (option_param == closed_cylinder)
                {
                    point_goal2 = nt_point + na_point + line * 2 * ngon + ngon + (ngon - counter - 1);
                    xt[point_goal2] = xt[point_source2];
                    yt[point_goal2] = yt[point_source2];
                    zt[point_goal2] = zt[point_source2];
                }

                if (option_param == arrows)
                {
                    point_goal2 = nt_point + na_point + line * 2 * ngon + ngon + counter;
                    calc_cone_base(vertice_source2, point_source2,
                                   arr_radius, arr_high, &x_temp, &y_temp, &z_temp);

                    xt[point_goal2] = x_temp;
                    yt[point_goal2] = y_temp;
                    zt[point_goal2] = z_temp;

                    //changing the positions of the tube points behind the cone
                    calc_cone_base(vertice_source2, point_source2,
                                   radius, arr_high, &x_temp, &y_temp, &z_temp);
                    xt[point_source2] = x_temp;
                    yt[point_source2] = y_temp;
                    zt[point_source2] = z_temp;
                }

                if (s_in != NULL)
                {
                    s_out[point_goal1] = s_out[point_source1];
                    s_out[point_goal2] = s_out[point_source2];
                }

                if (gennormals)
                {
                    xn[point_goal1] = xsnu1;
                    yn[point_goal1] = ysnu1;
                    zn[point_goal1] = zsnu1;

                    xn[point_goal2] = xsnu2;
                    yn[point_goal2] = ysnu2;
                    zn[point_goal2] = zsnu2;
                }
            }
            line++;
        }
    }

    //vertex list
    for (counter = 0; counter < ns_vert; counter++)
    {
        vt[nt_vert + na_vert + counter] = nt_point + na_point + counter;
    }

    //polygon list
    for (counter = 0; counter < ns_poly; counter++)
    {
        lt[nt_poly + na_poly + counter] = nt_vert + na_vert + counter * ngon;
    }
}

void Tube::create_arrow()
//============================================================================
// create the 1 arrow for every line
//============================================================================
{
    const float arr_radius = arr_radius_coef * radius;
    float arr_high;
    int line_c, counter, jjj;
    int vertice_source2; //vertices of the input lines
    int point_source2;
    int point_goal2a; //points for the end of the cone
    int point_goal2b; //points for the base of the cone
    float x_temp, y_temp, z_temp;
    float delta_x, delta_y, delta_z; //unitary vector
    float delta_length; //length of the las segment of a line
    int line = 0;

    for (line_c = 0; line_c < nlines; line_c++)
    {
        int nseg = 0;
        if (line_c == nlines - 1) // Last Line
            nseg = nvert - ll[line_c] - 1;
        else
            nseg = ll[line_c + 1] - ll[line_c] - 1;
        if (nseg > 0)
        {
            // getting info of the last segment of the line
            if (line_c < (nlines - 1))
            {
                vertice_source2 = ll[line_c + 1] - 1;
            }
            else // if it is the last line
            {
                vertice_source2 = nvert - 1;
            }

            //vector parallel to last segment
            delta_x = xl[vertice_source2] - xl[vertice_source2 - 1];
            delta_y = yl[vertice_source2] - yl[vertice_source2 - 1];
            delta_z = zl[vertice_source2] - zl[vertice_source2 - 1];
            delta_length = sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);

            arr_high = arr_radius / tan(arr_angle);
            if (arr_high > delta_length * arr_hmax)
            {
                arr_high = delta_length * arr_hmax;
            }

            //getting tube_points around the last point of the line
            for (counter = 0; counter < ngon; counter++)
            {
                point_source2 = vertice_source2 * ngon + counter;
                point_goal2b = nt_point + line * ngon * 2 + counter;
                point_goal2a = nt_point + line * ngon * 2 + ngon + counter;

                //the end of the cone
                xt[point_goal2a] = xl[vertice_source2];
                yt[point_goal2a] = yl[vertice_source2];
                zt[point_goal2a] = zl[vertice_source2];

                //the base of the cone
                calc_cone_base(vertice_source2, point_source2,
                               arr_radius, arr_high, &x_temp, &y_temp, &z_temp);

                xt[point_goal2b] = x_temp;
                yt[point_goal2b] = y_temp;
                zt[point_goal2b] = z_temp;

                if (s_in != NULL)
                {
                    s_out[point_goal2a] = s_out[point_source2];
                    s_out[point_goal2b] = s_out[point_source2];
                }
            }

            if (gennormals)
            {

                for (counter = 0; counter < ngon; counter++)
                {
                    point_source2 = vertice_source2 * ngon + counter;
                    point_goal2b = nt_point + line * ngon * 2 + counter;
                    point_goal2a = nt_point + line * ngon * 2 + ngon + counter;

                    float ax, ay, az, l_a;
                    float bx, by, bz;
                    float cx, cy, cz, l_c;

                    //vector parallel to last segment
                    bx = xl[vertice_source2 - 1] - xl[vertice_source2];
                    by = yl[vertice_source2 - 1] - yl[vertice_source2];
                    bz = zl[vertice_source2 - 1] - zl[vertice_source2];

                    ax = xt[point_goal2b] - xt[point_goal2a]; //vector tangent to the cone
                    ay = yt[point_goal2b] - yt[point_goal2a];
                    az = zt[point_goal2b] - zt[point_goal2a];
                    l_a = sqrt(ax * ax + ay * ay + az * az);

                    if (counter < ngon - 1)
                    {

                        //next vector tangent to the cone
                        cx = xt[point_goal2b + 1] - xt[point_goal2a + 1];
                        cy = yt[point_goal2b + 1] - yt[point_goal2a + 1];
                        cz = zt[point_goal2b + 1] - zt[point_goal2a + 1];
                    }
                    else
                    {
                        //next vector tangent to the cone
                        cx = xt[point_goal2b + 1 - ngon] - xt[point_goal2a + 1 - ngon];
                        cy = yt[point_goal2b + 1 - ngon] - yt[point_goal2a + 1 - ngon];
                        cz = zt[point_goal2b + 1 - ngon] - zt[point_goal2a + 1 - ngon];
                    }
                    l_c = sqrt(cx * cx + cy * cy + cz * cz);

                    if (l_a > 0)
                    {
                        //vector normal to the cone
                        xn[point_goal2b] = (ax * bx + ay * by + az * bz) / (l_a * l_a) * ax - bx;
                        yn[point_goal2b] = (ax * bx + ay * by + az * bz) / (l_a * l_a) * ay - by;
                        zn[point_goal2b] = (ax * bx + ay * by + az * bz) / (l_a * l_a) * az - bz;

                        xn[point_goal2a] = 0.5f * (xn[point_goal2b] + (cx * bx + cy * by + cz * bz) / (l_c * l_c) * cx - bx);
                        yn[point_goal2a] = 0.5f * (yn[point_goal2b] + (cx * bx + cy * by + cz * bz) / (l_c * l_c) * cy - by);
                        zn[point_goal2a] = 0.5f * (zn[point_goal2b] + (cx * bx + cy * by + cz * bz) / (l_c * l_c) * cz - bz);
                    }

                    else
                    {

                        xn[point_goal2b] = 1; //vector normal to the cone
                        yn[point_goal2b] = 0;
                        zn[point_goal2b] = 0;
                        xn[point_goal2a] = 1;
                        yn[point_goal2a] = 0;
                        zn[point_goal2a] = 0;
                    }
                }
            }

            //vertex list
            for (counter = 0; counter < nlines; counter++)
            {
                for (jjj = 0; jjj < ngon; jjj++)
                {
                    vt[nt_vert + counter * ngon * tri + jjj * tri + 0] = nt_point + 2 * counter * ngon + jjj + 0;
                    vt[nt_vert + counter * ngon * tri + jjj * tri + 1] = nt_point + 2 * counter * ngon + jjj + ngon;
                    if (jjj < (ngon - 1))
                    {
                        vt[nt_vert + counter * ngon * tri + jjj * tri + 2] = nt_point + 2 * counter * ngon + jjj + 1;
                    }
                    else
                    {
                        vt[nt_vert + counter * ngon * tri + jjj * tri + 2] = nt_point + 2 * counter * ngon + jjj + 1 - ngon;
                    }
                }
            }
            line++;
        }
    }

    //polygon list

    for (counter = 0; counter < na_poly; counter++)
    {
        lt[nt_poly + counter] = nt_vert + counter * tri;
    }
}

void Tube::calc_cone_base(int line_point, int tube_point,
                          float base_radius, float cone_high, float *x_v, float *y_v, float *z_v)
//============================================================================
// create the 2 stoppers for every line
//============================================================================
{
    float delta_x, delta_y, delta_z;
    float deltau_x, deltau_y, deltau_z; //unitary vector
    float delta_length; //length of the las segment of a line
    float x_vr, y_vr, z_vr, xyz_vr_length; //radial vector from the line to the tube_points
    float x_ur, y_ur, z_ur; //

    delta_x = xl[line_point] - xl[line_point - 1]; //vector parallel to last segment
    delta_y = yl[line_point] - yl[line_point - 1];
    delta_z = zl[line_point] - zl[line_point - 1];
    delta_length = sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);

    if (delta_length > 0)
    {
        deltau_x = delta_x / delta_length;
        deltau_y = delta_y / delta_length;
        deltau_z = delta_z / delta_length;
    }
    else
    {
        deltau_x = 0;
        deltau_y = 0;
        deltau_z = 0;
    }

    x_vr = xt[tube_point] - xl[line_point]; // radial vector -
    y_vr = yt[tube_point] - yl[line_point]; // from the line to the tube
    z_vr = zt[tube_point] - zl[line_point];
    xyz_vr_length = sqrt(x_vr * x_vr + y_vr * y_vr + z_vr * z_vr);

    if (xyz_vr_length > 0)
    {
        x_ur = x_vr / xyz_vr_length;
        y_ur = y_vr / xyz_vr_length;
        z_ur = z_vr / xyz_vr_length;
    }
    else
    {
        x_ur = 0;
        y_ur = 0;
        z_ur = 0;
    }

    *x_v = xl[line_point] + base_radius * x_ur - cone_high * deltau_x;
    *y_v = yl[line_point] + base_radius * y_ur - cone_high * deltau_y;
    *z_v = zl[line_point] + base_radius * z_ur - cone_high * deltau_z;
}

MODULE_MAIN(Tools, Tube)
