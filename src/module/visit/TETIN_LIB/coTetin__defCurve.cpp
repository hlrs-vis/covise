/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__defCurve.h"
#include <string.h>
#include <iostream.h>
#include <stdio.h>

const int LINE_SIZE = 4096;
static coTetin__defCurve *current_defCurve = 0;

coTetin__defCurve::coTetin__defCurve(istream &str, int binary, ostream &ostr)
    : coTetinCommand(coTetin::DEFINE_CURVE)
    , cr(0)
{
    char line[LINE_SIZE];
    char *linev[500];
    char *tmp_name = 0;
    char *name = 0;
    char *tmp_family = 0;
    char *family = 0;
    char *tmp_end_names[2];
    tmp_end_names[0] = tmp_end_names[1] = 0;
    char *end_names[2];
    end_names[0] = end_names[1] = 0;

    getLine(line, LINE_SIZE, str);
    int linec;
    linec = coTetin__break_line(line, linev);
    if (linec == 0)
    {
        ostr << "define a curve";
    }
    else if (linec < 0)
    {
        goto usage;
    }

    int il;
    il = 0;

    float maxsize;
    maxsize = 1.0e10;

    int width;
    width = 0;

    float ratio;
    ratio = 0.0;

    float minlimit;
    minlimit = 0.0;

    float dev;
    dev = 0.0;

    float height;
    height = 0.0;
    int dormant;
    dormant = 0;

    while (il < linec)
    {
        if (strcmp("dormant", linev[il]) == 0)
        {
            dormant = 1;
            il++;
        }
        else if (strncmp("vertex", linev[il], 6) == 0)
        {
            if (linev[il][6] == '1')
            {
                il++;
                tmp_end_names[0] = linev[il];
                il++;
            }
            else if (linev[il][6] == '2')
            {
                il++;
                tmp_end_names[1] = linev[il];
                il++;
            }
            else
            {
                goto usage;
            }
        }
        else if (strcmp("height", linev[il]) == 0)
        {
            il++;
            if (il >= linec)
                goto usage;
            if (sscanf(linev[il], "%f", &height) != 1)
            {
                goto usage;
            }
            il++;
        }
        else if (strcmp("dev", linev[il]) == 0)
        {
            il++;
            if (il >= linec)
                goto usage;
            if (sscanf(linev[il], "%f", &dev) != 1)
            {
                goto usage;
            }
            il++;
        }
        else if (strcmp("min", linev[il]) == 0)
        {
            il++;
            if (il >= linec)
                goto usage;
            if (sscanf(linev[il], "%f", &minlimit) != 1)
            {
                goto usage;
            }
            il++;
        }
        else if (strcmp("ratio", linev[il]) == 0)
        {
            il++;
            if (il >= linec)
                goto usage;
            if (sscanf(linev[il], "%f", &ratio) != 1)
            {
                goto usage;
            }
            il++;
        }
        else if (strcmp("width", linev[il]) == 0)
        {
            il++;
            if (il >= linec)
                goto usage;
            if (sscanf(linev[il], "%d", &width) != 1)
            {
                goto usage;
            }
            il++;
        }
        else if (strcmp("family", linev[il]) == 0)
        {
            il++;
            if (il >= linec)
                goto usage;
            tmp_family = linev[il];
            il++;
        }
        else if (strcmp("tetra_size", linev[il]) == 0)
        {
            il++;
            if (il >= linec)
                goto usage;
            if (sscanf(linev[il], "%f", &maxsize) != 1)
            {
                goto usage;
            }
            if (maxsize <= 0.0)
            {
                maxsize = 1.0e10;
            }
            il++;
        }
        else if (strcmp("name", linev[il]) == 0)
        {
            il++;
            if (il >= linec)
                goto usage;
            tmp_name = linev[il];
            il++;
        }
        else
        {
            goto usage;
        }
    }
    if (tmp_family)
    {
        family = (char *)new char[strlen(tmp_family) + 1];
        strcpy(family, tmp_family);
    }
    if (tmp_name)
    {
        name = (char *)new char[strlen(tmp_name) + 1];
        strcpy(name, tmp_name);
    }
    int i;
    for (i = 0; i < 2; i++)
    {
        if (tmp_end_names[i])
        {
            end_names[i] = (char *)new char[strlen(tmp_end_names[i]) + 1];
            strcpy(end_names[i], tmp_end_names[i]);
        }
    }

    char type[100];
    // get the line and the first word in the line
    getLine(line, LINE_SIZE, str);
    if (str.eof() || strlen(line) == 0 || sscanf(line, "%s", type) != 1)
    {
        ostr << "error reading curve type";
        goto usage;
    }
    if (strcmp(type, "bspline") == 0)
    {
        // read and define a new bspline curve
        coTetin__BSpline *bspl;
        bspl = read_spline_curve(str, ostr);
        if (bspl == 0)
        {
            ostr << "error defining spline curve ";
            if (name)
                ostr << name;

            goto usage;
        }

        cr = new coTetin__param_curve(bspl);
    }
    else if (strcmp(type, "unstruct_curve") == 0)
    {
        char *sub_linev[10];
        int sub_linec = coTetin__break_line(line, sub_linev);
        if (sub_linec == 2)
        {
            // in domain file
            char *path = new char[strlen(sub_linev[1]) + 1];
            strcpy(path, sub_linev[1]);
            cr = (coTetin__curve_record *)new coTetin__unstruct_curve(path);
        }
        else
        {
            int npnts = 0;
            int nedges = 0;
            il = 1;
            while (il < sub_linec)
            {
                if (strcmp(sub_linev[il], "n_points") == 0)
                {
                    il++;
                    if (il >= sub_linec)
                    {
                        ostr << "missing arg to n_points";
                        goto usage;
                    }
                    if (sscanf(sub_linev[il], "%d", &npnts) != 1)
                    {
                        ostr << "error scanning number of points\n";
                        goto usage;
                    }
                    il++;
                }
                else if (strcmp(sub_linev[il], "n_edges") == 0)
                {
                    il++;
                    if (il >= sub_linec)
                    {
                        ostr << "missing arg to n_edges";
                        goto usage;
                    }
                    if (sscanf(sub_linev[il], "%d", &nedges) != 1)
                    {
                        ostr << "error scanning number of edges\n";
                        goto usage;
                    }
                    il++;
                }
                else
                {
                    ostr << "unrecognized arg " << sub_linev[il] << " to unstruct_curve\n";
                    goto usage;
                }
            }
            cr = pj_new_mesh_curve(str, ostr, npnts, nedges);
        }
    }
    else
    {
        ostr << "unrecognized curve type: " << type;
        goto usage;
    }
    if (cr == 0)
    {
        ostr << "error defining curve";
        goto usage;
    }
    for (i = 0; i < 2; i++)
    {
        cr->end_names[i] = end_names[i];
    }
    cr->name = name;
    cr->family = family;
    cr->maxsize = maxsize;
    cr->width = width;
    cr->ratio = ratio;
    cr->height = height;
    cr->minlimit = minlimit;
    cr->deviation = dev;
    cr->dormant = dormant;

    return;

usage:
    if (cr)
    {
        delete cr;
        cr = 0;
    }
    else
    {
        if (name)
        {
            delete[] name;
            name = 0;
        }
        if (family)
        {
            delete[] family;
            family = 0;
        }
        for (i = 0; i < 2; i++)
        {
            if (end_names[i])
            {
                delete[] end_names[i];
                end_names[i] = 0;
            }
        }
    }
    ostr << "Usage: " << linev[0] << " [tetra_size %%d] [name %%s] [family %%d] ";
    ostr << "[width %%d] [ratio %%f] [min %%f] [dev %%f] (data on following lines)";
    return;
}

coTetin__defCurve::coTetin__defCurve(float *points_x, float *points_y,
                                     float *points_z, int n_points,
                                     char *crv_name)
    : coTetinCommand(coTetin::DEFINE_CURVE)
    , cr(0)
{
    cr = pj_new_mesh_curve(points_x, points_y, points_z, n_points);
    if (cr)
    {
        char *name = 0;
        for (int i = 0; i < 2; i++)
        {
            cr->end_names[i] = 0;
        }
        if (crv_name)
        {
            name = (char *)new char[strlen(crv_name) + 1];
            strcpy(name, crv_name);
        }
        cr->name = name;
        cr->family = 0;
        cr->maxsize = -1.0;
        cr->width = -1;
        cr->ratio = -1.0;
        cr->height = -1.0;
        cr->minlimit = -1.0;
        cr->deviation = -1.0;
        cr->dormant = 0;
    }

    return;
}

coTetin__mesh_curve *
coTetin__defCurve::pj_new_mesh_curve(float *points_x, float *points_y,
                                     float *points_z, int n_nodes)
{
    current_defCurve = this;
    int n_edges = n_nodes - 1;
    coTetin__mesh_curve *cr;
    cr = new coTetin__mesh_curve(n_nodes, n_edges);
    if (n_edges > 0)
    {
        coTetin__mesh_curve(n_nodes, n_edges);

        point *pnts;
        pnts = cr->pnts;
        curve_edge *edges = cr->edges;
        int i;
        // read the nodes
        for (i = 0; i < n_nodes; i++)
        {
            pnts[i][0] = points_x[i];
            pnts[i][1] = points_y[i];
            pnts[i][2] = points_z[i];
        }
        for (i = 0; i < n_edges; i++)
        {
            edges[i][0] = i;
            edges[i][1] = i + 1;
        }
    }
    return cr;
}

coTetin__mesh_curve *
coTetin__defCurve::pj_new_mesh_curve(istream &str, ostream &ostr,
                                     int n_nodes, int n_edges)
{
    current_defCurve = this;
    coTetin__mesh_curve *cr;
    cr = new coTetin__mesh_curve(n_nodes, n_edges);

    point *pnts;
    pnts = cr->pnts;
    curve_edge *edges = cr->edges;
    int i;
    // read the nodes
    int j;
    float loc[3];
    char line[LINE_SIZE];
    for (i = 0; i < n_nodes; i++)
    {
        getLine(line, LINE_SIZE, str);
        if (str.eof() || strlen(line) == 0)
        {
            ostr << "error obtaining line for coordinates\n";
            return 0;
        }
        if (sscanf(line, "%f %f %f", loc, loc + 1, loc + 2) != 3)
        {
            ostr << "error scaning coordinates\n";
            return 0;
        }
        for (j = 0; j < 3; j++)
        {
            pnts[i][j] = loc[j];
        }
    }
    // read the elements
    for (i = 0; i < n_edges; i++)
    {
        getLine(line, LINE_SIZE, str);
        if (str.eof() || strlen(line) == 0)
        {
            ostr << "error obtaining line for segment indices\n";
            return 0;
        }
        if (sscanf(line, "%d %d",
                   edges[i], edges[i] + 1) != 2)
        {
            ostr << "error scaning segment indices\n";
            return 0;
        }
    }

    return cr;
}

coTetin__curve_record::coTetin__curve_record(int typ)
{
    dormant = 0;
    family = 0;
    maxsize = 0.0;
    width = 0;
    ratio = 0.0;
    name = 0;
    minlimit = 0.0;
    deviation = 0.0;
    height = 0.0;
    curve_type = typ;
    int i;
    for (i = 0; i < 2; i++)
    {
        end_names[i] = 0;
    }
}

coTetin__curve_record::~coTetin__curve_record()
{
    if (name)
        delete[] name;
    name = 0;
    if (family)
        delete[] family;
    family = 0;
    dormant = 0;
    maxsize = 0.0;
    width = 0;
    ratio = 0.0;
    minlimit = 0.0;
    deviation = 0.0;
    height = 0.0;
    curve_type = INVALID_CURVE;
    int i;
    for (i = 0; i < 2; i++)
    {
        if (end_names[i])
            delete[] end_names[i];
        end_names[i] = 0;
    }
}

coTetin__mesh_curve::coTetin__mesh_curve()
    : coTetin__curve_record(MESH_CURVE)
{
    n_pnts = 0;
    pnts = 0;
    n_edges = 0;
    edges = 0;
}

coTetin__mesh_curve::coTetin__mesh_curve(int npnts, int nedges)
    : coTetin__curve_record(MESH_CURVE)
{
    n_pnts = npnts;
    pnts = (point *)new point[npnts];
    n_edges = nedges;
    edges = (curve_edge *)new curve_edge[nedges];
}

coTetin__mesh_curve::~coTetin__mesh_curve()
{
    n_pnts = 0;
    if (pnts)
        delete[] pnts;
    pnts = 0;
    n_edges = 0;
    if (edges)
        delete[] edges;
    edges = 0;
}

coTetin__param_curve::coTetin__param_curve()
    : coTetin__curve_record(PARAMETRIC_CURVE)
{
    spl = 0;
}

coTetin__param_curve::coTetin__param_curve(coTetin__BSpline *bspl)
    : coTetin__curve_record(PARAMETRIC_CURVE)
{
    spl = bspl;
}

coTetin__param_curve::~coTetin__param_curve()
{
    if (spl)
        delete spl;
    spl = 0;
}

coTetin__unstruct_curve::coTetin__unstruct_curve()
    : coTetin__curve_record(UNSTRUCT_CURVE)
{
    path = 0;
}

coTetin__unstruct_curve::coTetin__unstruct_curve(char *name)
    : coTetin__curve_record(UNSTRUCT_CURVE)
{
    path = name;
}

coTetin__unstruct_curve::~coTetin__unstruct_curve()
{
    if (path)
        delete[] path;
    path = 0;
}

/// read from memory
coTetin__defCurve::coTetin__defCurve(int *&intDat, float *&floatDat, char *&charDat)
    : coTetinCommand(coTetin::DEFINE_CURVE)
{
    current_defCurve = this;

    coTetin__param_curve *pc = 0;
    coTetin__mesh_curve *mc = 0;
    coTetin__unstruct_curve *uc = 0;
    short curve_type;
    curve_type = *intDat++;
    switch (curve_type)
    {
    case PARAMETRIC_CURVE:
        pc = new coTetin__param_curve();
        cr = (coTetin__curve_record *)pc;
        break;
    case MESH_CURVE:
        mc = new coTetin__mesh_curve();
        cr = (coTetin__curve_record *)mc;
        break;
    case UNSTRUCT_CURVE:
        uc = new coTetin__unstruct_curve();
        cr = (coTetin__curve_record *)uc;
        break;
    }
    cr->family = getString(charDat);
    cr->maxsize = *floatDat++;
    cr->width = *intDat++;
    cr->ratio = *floatDat++;
    cr->name = getString(charDat);
    cr->minlimit = *floatDat++;
    cr->deviation = *floatDat++;
    cr->height = *floatDat++;
    cr->dormant = *charDat++;
    cr->end_names[0] = getString(charDat);
    cr->end_names[1] = getString(charDat);

    switch (curve_type)
    {
    case PARAMETRIC_CURVE:
        pc->putBinary(intDat, floatDat, charDat);
        break;
    case MESH_CURVE:
        mc->putBinary(intDat, floatDat, charDat);
        break;
    case UNSTRUCT_CURVE:
        uc->putBinary(intDat, floatDat, charDat);
        break;
    }
}

void coTetin__param_curve::putBinary(int *&intDat, float *&floatDat, char *&charDat)
{
    spl = new coTetin__BSpline();
    spl->putBinary(intDat, floatDat, charDat);
}

void coTetin__unstruct_curve::putBinary(int *&intDat, float *&floatDat, char *&charDat)
{
    path = current_defCurve->getNextString(charDat);
}

void coTetin__mesh_curve::putBinary(int *&intDat, float *&floatDat, char *&charDat)
{
    n_pnts = *intDat++;
    n_edges = *intDat++;
    pnts = 0;
    edges = 0;
    if (n_pnts > 0)
    {
        pnts = new point[n_pnts];
        memcpy((void *)pnts, (void *)floatDat, 3 * n_pnts * sizeof(float));
        floatDat += 3 * n_pnts;
    }
    if (n_edges > 0)
    {
        edges = new curve_edge[n_edges];
        memcpy((void *)edges, (void *)intDat, 2 * n_edges * sizeof(int));
        intDat += 2 * n_edges;
    }
}

/// Destructor
coTetin__defCurve::~coTetin__defCurve()
{
    current_defCurve = 0;
    if (cr)
    {
        delete cr;
        cr = 0;
    }
}

/// check whether Object is valid
int coTetin__defCurve::isValid() const
{
    if (!cr || (cr->curve_type != PARAMETRIC_CURVE && cr->curve_type != MESH_CURVE && cr->curve_type != UNSTRUCT_CURVE))
        return 0;
    else
        return 1;
}

/// count size required in fields
void coTetin__defCurve::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    if (!isValid())
        return;
    numInt += 3; // comm,curve_type,width
    numFloat += 5; // height,deviation,minlimit,ratio,maxsize
    numChar += 1; // dormant
    numChar += (cr->family) ? (strlen(cr->family) + 1) : 1;
    numChar += (cr->name) ? (strlen(cr->name) + 1) : 1;
    numChar += (cr->end_names[0]) ? (strlen(cr->end_names[0]) + 1) : 1;
    numChar += (cr->end_names[1]) ? (strlen(cr->end_names[1]) + 1) : 1;

    coTetin__param_curve *pc = 0;
    coTetin__mesh_curve *mc = 0;
    coTetin__unstruct_curve *uc = 0;
    switch (cr->curve_type)
    {
    case PARAMETRIC_CURVE:
        pc = (coTetin__param_curve *)cr;
        pc->addSizes(numInt, numFloat, numChar);
        break;
    case MESH_CURVE:
        mc = (coTetin__mesh_curve *)cr;
        mc->addSizes(numInt, numFloat, numChar);
        break;
    case UNSTRUCT_CURVE:
        uc = (coTetin__unstruct_curve *)cr;
        uc->addSizes(numInt, numFloat, numChar);
        break;
    }
}

static char cov_tmp_array[2000];
void coTetin__param_curve::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    if (spl)
    {
        sprintf(cov_tmp_array, "bspline");
        numChar += strlen(cov_tmp_array) + 1;
        numInt += 3; // ncps, ord, rat
        int rat = spl->rat;
        int ncps = spl->ncps[0];
        int ord = spl->ord[0];
        if (spl->knot[0])
        {
            numFloat += ncps + ord;
        }
        if (spl->cps)
        {
            if (rat)
            {
                numFloat += 4 * ncps;
            }
            else
            {
                numFloat += 3 * ncps;
            }
        }
    }
}

void coTetin__unstruct_curve::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    if (path)
    {
        sprintf(cov_tmp_array, "unstruct_curve %s", path);
        numChar += strlen(cov_tmp_array) + 1;
    }
}

void coTetin__mesh_curve::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    sprintf(cov_tmp_array, "unstruct_curve n_points %d n_edges %d",
            n_pnts, n_edges);
    numChar += strlen(cov_tmp_array) + 1;
    if (pnts)
        numFloat += 3 * n_pnts;
    if (edges)
        numInt += 2 * n_edges;
}

void coTetin__param_curve::addBSizes(int &numInt, int &numFloat, int &numChar)
    const
{
    if (spl)
    {
        spl->addSizes(numInt, numFloat, numChar);
    }
}

void coTetin__unstruct_curve::addBSizes(int &numInt, int &numFloat, int &numChar) const
{
    numChar += (path) ? (strlen(path) + 1) : 1;
}

void coTetin__mesh_curve::addBSizes(int &numInt, int &numFloat, int &numChar) const
{
    numInt += 2; // n_pnts, n_edges
    if (pnts)
        numFloat += 3 * n_pnts; // pnts
    if (edges)
        numInt += 2 * n_edges; // edges
}

/// put my data to a given set of pointers
void coTetin__defCurve::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    if (!isValid())
        return;

    // copy the command's name
    *intDat++ = d_comm;

    *intDat++ = cr->curve_type;
    // copy the data
    if (cr->family)
    {
        strcpy(charDat, cr->family);
        charDat += strlen(cr->family) + 1;
    }
    else
    {
        *charDat++ = '\0';
    }
    *floatDat++ = cr->maxsize;
    *intDat++ = cr->width;
    *floatDat++ = cr->ratio;
    if (cr->name)
    {
        strcpy(charDat, cr->name);
        charDat += strlen(cr->name) + 1;
    }
    else
    {
        *charDat++ = '\0';
    }
    *floatDat++ = cr->minlimit;
    *floatDat++ = cr->deviation;
    *floatDat++ = cr->height;
    *charDat++ = cr->dormant;
    if (cr->end_names[0])
    {
        strcpy(charDat, cr->end_names[0]);
        charDat += strlen(cr->end_names[0]) + 1;
    }
    else
    {
        *charDat++ = '\0';
    }
    if (cr->end_names[1])
    {
        strcpy(charDat, cr->end_names[1]);
        charDat += strlen(cr->end_names[1]) + 1;
    }
    else
    {
        *charDat++ = '\0';
    }

    coTetin__param_curve *pc = 0;
    coTetin__mesh_curve *mc = 0;
    coTetin__unstruct_curve *uc = 0;
    switch (cr->curve_type)
    {
    case PARAMETRIC_CURVE:
        pc = (coTetin__param_curve *)cr;
        pc->internal_write(cout, intDat, floatDat, charDat);
        break;
    case MESH_CURVE:
        mc = (coTetin__mesh_curve *)cr;
        mc->internal_write(cout, intDat, floatDat, charDat);
        break;
    case UNSTRUCT_CURVE:
        uc = (coTetin__unstruct_curve *)cr;
        uc->internal_write(cout, intDat, floatDat, charDat);
        break;
    }
}

void coTetin__param_curve::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    if (spl)
    {
        spl->getBinary(intDat, floatDat, charDat);
    }
}

void coTetin__unstruct_curve::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    // copy the data
    if (path)
    {
        strcpy(charDat, path);
        charDat += strlen(path) + 1;
    }
    else
    {
        *charDat++ = '\0';
    }
}

void coTetin__mesh_curve::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    *intDat++ = n_pnts;
    *intDat++ = n_edges;
    if (pnts && n_pnts > 0)
    {
        memcpy((void *)floatDat, (void *)pnts, 3 * n_pnts * sizeof(float));
        floatDat += 3 * n_pnts;
    }
    if (edges && n_edges > 0)
    {
        memcpy((void *)intDat, (void *)edges, 2 * n_edges * sizeof(int));
        intDat += 2 * n_edges;
    }
}

/// print to a stream in Tetin format
void coTetin__defCurve::print(ostream &str) const
{
    if (isValid())
    {
        str << "define_curve";
        if (cr->height > 0.0)
        {
            str << " height " << cr->height;
        }
        if (cr->deviation > 0.0)
        {
            str << " dev " << cr->deviation;
        }
        if (cr->minlimit > 0.0)
        {
            str << " min " << cr->minlimit;
        }
        if (cr->ratio > 0.0)
        {
            str << " ratio " << cr->ratio;
        }
        if (cr->width > 0)
        {
            str << " width " << cr->width;
        }
        if (cr->family && cr->family[0])
        {
            str << " family " << cr->family;
        }
        if (cr->maxsize != 1.0e10)
        {
            str << " tetra_size " << cr->maxsize;
        }
        if (cr->name && cr->name[0])
        {
            str << " name " << cr->name;
        }
        if (cr->end_names[0] && cr->end_names[0][0])
        {
            str << " vertex1 " << cr->end_names[0];
        }
        if (cr->end_names[1] && cr->end_names[1][0])
        {
            str << " vertex2 " << cr->end_names[1];
        }
        if (cr->dormant)
        {
            str << " dormant";
        }
        str << endl;

        int *iDat = 0;
        float *fDat = 0;
        char *cDat = 0;
        cr->internal_write(str, iDat, fDat, cDat);
    } // if (isValid())
}

void coTetin__curve_record::internal_write(ostream &str,
                                           int *&intDat, float *&floatDat, char *&charDat) const
{
    if (curve_type == PARAMETRIC_CURVE)
    {
        coTetin__param_curve *pc = (coTetin__param_curve *)this;
        pc->internal_write(str, intDat, floatDat, charDat);
    }
    else if (curve_type == MESH_CURVE)
    {
        coTetin__mesh_curve *mc = (coTetin__mesh_curve *)this;
        mc->internal_write(str, intDat, floatDat, charDat);
    }
    else if (curve_type == UNSTRUCT_CURVE)
    {
        coTetin__unstruct_curve *uc = (coTetin__unstruct_curve *)this;
        uc->internal_write(str, intDat, floatDat, charDat);
    }
    else
    {
        printf(
            "curve_record::internal_write not impelemented for this type\n");
    }
}

void coTetin__mesh_curve::internal_write(ostream &str,
                                         int *&intDat, float *&floatDat, char *&charDat) const
{
    int write_bin;
    if (intDat && floatDat && charDat)
    {
        write_bin = 1;
    }
    else
    {
        write_bin = 0;
    }
    if (write_bin)
    {
        sprintf(cov_tmp_array, "unstruct_curve n_points %d n_edges %d",
                n_pnts, n_edges);
        strcpy(charDat, cov_tmp_array);
        charDat += strlen(cov_tmp_array) + 1;
    }
    else
    {
        str << "unstruct_curve n_points " << n_pnts << " n_edges " << n_edges << endl;
    }
    int i;
    for (i = 0; i < n_pnts; i++)
    {
        if (write_bin)
        {
            *floatDat++ = pnts[i][0];
            *floatDat++ = pnts[i][1];
            *floatDat++ = pnts[i][2];
        }
        else
        {
            str << pnts[i][0] << ' ' << pnts[i][1] << ' ' << pnts[i][2] << endl;
        }
    }
    for (i = 0; i < n_edges; i++)
    {
        if (edges[i][0] >= n_pnts || edges[i][1] >= n_pnts)
        {
            printf("point index out of range\n");
            continue;
        }
        if (write_bin)
        {
            *intDat++ = edges[i][0];
            *intDat++ = edges[i][1];
        }
        else
        {
            str << edges[i][0] << ' ' << edges[i][1] << endl;
        }
    }
}

void coTetin__param_curve::internal_write(ostream &str,
                                          int *&intDat, float *&floatDat, char *&charDat) const
{
    if (spl)
    {
        int write_bin;
        if (intDat && floatDat && charDat)
        {
            write_bin = 1;
        }
        else
        {
            write_bin = 0;
        }
        if (write_bin)
        {
            sprintf(cov_tmp_array, "bspline");
            strcpy(charDat, cov_tmp_array);
            charDat += strlen(cov_tmp_array) + 1;
        }
        else
        {
            str << "bspline" << endl;
        }
        int rat = spl->rat;
        int ncps = spl->ncps[0];
        int ord = spl->ord[0];
        if (write_bin)
        {
            *intDat++ = ncps;
            *intDat++ = ord;
            *intDat++ = rat;
        }
        else
        {
            str << ncps << ',' << ord << ',' << rat << endl;
        }
        // print the knot vector
        int nknt = ncps + ord;
        int i;
        if (spl->knot[0])
        {
            float *knot = spl->knot[0];
            // only write 5 reals per line
            for (i = 0; i < nknt; i++)
            {
                if (write_bin)
                {
                    *floatDat++ = knot[i];
                }
                else
                {
                    str << knot[i];
                    if ((i + 1) % 5 == 0 || i == nknt - 1)
                    {
                        str << endl;
                    }
                    else
                    {
                        str << ',';
                    }
                }
            }
        }
        // print the control points
        if (spl->cps)
        {
            float *cps = spl->cps;
            for (i = 0; i < ncps; i++)
            {
                if (rat)
                {
                    if (write_bin)
                    {
                        *floatDat++ = cps[0];
                        *floatDat++ = cps[1];
                        *floatDat++ = cps[2];
                        *floatDat++ = cps[3];
                    }
                    else
                    {
                        str << cps[0] << ',' << cps[1] << ',' << cps[2] << ',' << cps[3] << endl;
                    }
                    cps += 4;
                }
                else
                {
                    if (write_bin)
                    {
                        *floatDat++ = cps[0];
                        *floatDat++ = cps[1];
                        *floatDat++ = cps[2];
                    }
                    else
                    {
                        str << cps[0] << ',' << cps[1] << ',' << cps[2] << endl;
                    }
                    cps += 3;
                }
            }
        }
    }
}

void coTetin__unstruct_curve::internal_write(ostream &str,
                                             int *&intDat, float *&floatDat, char *&charDat) const
{
    if (path)
    {
        int write_bin;
        if (intDat && floatDat && charDat)
        {
            write_bin = 1;
        }
        else
        {
            write_bin = 0;
        }
        if (write_bin)
        {
            sprintf(cov_tmp_array, "unstruct_curve %s", path);
            strcpy(charDat, cov_tmp_array);
            charDat += strlen(cov_tmp_array) + 1;
        }
        else
        {
            str << "unstruct_curve " << path << endl;
        }
    }
}

char *coTetin__defCurve::getNextString(char *&chPtr)
{
    return getString(chPtr);
}

// ===================== command-specific functions =====================
