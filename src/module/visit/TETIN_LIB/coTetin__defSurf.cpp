/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__defSurf.h"
#include <string.h>
#include <math.h>
#include <ctype.h>

const int LINE_SIZE = 4096;

static coTetin__defSurf *current_defSurf = 0;

coTetin__defSurf::coTetin__defSurf(istream &str, int binary, ostream &ostr)
    : coTetinCommand(coTetin::DEFINE_SURFACE)
    , sr(0)
{
    current_defSurf = this;

    char line[LINE_SIZE];
    char *linev[500];
    char *tmp_name = 0;
    char *name = 0;
    char *tmp_family = 0;
    char *family = 0;

    getLine(line, LINE_SIZE, str);
    int linec;
    linec = coTetin__break_line(line, linev);
    if (linec == 0)
    {
        ostr << "read a surface";
        goto usage;
    }
    else if (linec < 0)
    {
        goto usage;
    }

    float maxsize;
    maxsize = 1.0e10;

    int il;
    il = 0;
    float height;
    height = 0.0;
    float width;
    width = 0.0;
    float ratio;
    ratio = 0.;
    float minsize;
    minsize = 0.;
    float dev;
    dev = 0.;
    int level;
    level = 0;
    int number;
    number = 0;
    // -1 all into one surface
    // -2 divide into several surfaces
    // >= extract specified pid
    int by_ids;
    by_ids = -1;
    int reverse_normal;
    reverse_normal = 0;
    while (il < linec)
    {
        if (strcmp("reverse_normal", linev[il]) == 0)
        {
            il++;
            reverse_normal = 1;
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
            if (sscanf(linev[il], "%f", &minsize) != 1)
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
            if (ratio != 0.0 && ratio <= 1.0)
            {
                ostr << "ratio must be 0.0 or > 1.0";
                if (tmp_name)
                {
                    ostr << " for surface " << tmp_name;
                }
                ostr << endl;
                ratio = 2.0;
            }
            il++;
        }
        else if (strcmp("width", linev[il]) == 0)
        {
            il++;
            if (il >= linec)
                goto usage;
            if (sscanf(linev[il], "%f", &width) != 1)
            {
                goto usage;
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
        else if (strcmp("family", linev[il]) == 0)
        {
            il++;
            if (il >= linec)
                goto usage;
            tmp_family = linev[il];
            il++;
        }
        else if (strcmp("level", linev[il]) == 0)
        {
            il++;
            if (il >= linec)
                goto usage;
            if (sscanf(linev[il], "%d", &level) != 1)
                goto usage;
            il++;
        }
        else if (strcmp("number", linev[il]) == 0)
        {
            il++;
            if (il >= linec)
                goto usage;
            if (sscanf(linev[il], "%d", &number) != 1)
                goto usage;
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
                ostr << "maxsize must be > 0.0" << endl;
                maxsize = 1.0e10;
            }
            il++;
        }
        else if (strcmp("pid", linev[il]) == 0)
        {
            il++;
            if (il >= linec)
                goto usage;
            if (sscanf(linev[il], "%d", &by_ids) != 1)
            {
                goto usage;
            }
            il++;
        }
        else if (strcmp("by_pids", linev[il]) == 0)
        {
            il++;
            by_ids = -2;
        }
        else
        {
            ostr << "skipping parameter " << linev[il] << endl;
            il++;
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
    // get the line and the first word in the line
    getLine(line, LINE_SIZE, str);
    if (str.eof() || strlen(line) == 0)
    {
        ostr << "error reading surface type";
        goto usage;
    }
    char *sublinev[100];
    int sublinec;
    sublinec = coTetin__break_line(line, sublinev);
    if (sublinec < 1)
    {
        ostr << "error reading surface type";
        goto usage;
    }
    if (strcmp(sublinev[0], "bspline") == 0
        || strcmp(sublinev[0], "trim_surface") == 0)
    {
        sr = pj_new_param_surface(sublinec,
                                  sublinev, name, str, ostr);
        if (!sr)
        {
            ostr << "error defining surface " << (name ? name : "(no name)") << endl;
            goto usage;
        }
    }
    else if (strcmp(sublinev[0], "unstruct_mesh") == 0)
    {
        //
        // unstructured mesh surface
        //
        if (sublinec == 2)
        {
            // in domain file
            char *path = new char[strlen(sublinev[1]) + 1];
            strcpy(path, sublinev[1]);
            sr = (coTetin__surface_record *)new coTetin__unstruct_surface(path);
        }
        else
        {
            // in the tetin file
            il = 1;
            int n_points = 0;
            int n_tris = 0;
            while (il < sublinec)
            {
                if (strcmp(sublinev[il], "n_points") == 0)
                {
                    il++;
                    if (il >= sublinec)
                    {
                        ostr << "missing arg n_points";
                        goto usage;
                    }
                    if (sscanf(sublinev[il], "%d", &n_points) != 1)
                    {
                        ostr << "error scanning n_points";
                        goto usage;
                    }
                    il++;
                }
                else if (strcmp(sublinev[il], "n_triangles") == 0)
                {
                    il++;
                    if (il >= sublinec)
                    {
                        ostr << "missing arg n_triangles";
                        goto usage;
                    }
                    if (sscanf(sublinev[il], "%d", &n_tris) != 1)
                    {
                        ostr << "error scanning n_triangles";
                        goto usage;
                    }
                    il++;
                }
                else
                {
                    ostr << "unrecognized arg " << sublinev[il] << " to unstruct_mesh";
                    goto usage;
                }
            }
            sr = pj_new_mesh_surface(str, ostr,
                                     n_points, n_tris);
            if (!sr)
            {
                ostr << "error defining surface " << (name ? name : "(no name)") << endl;
                goto usage;
            }
        }
    }
    else if (strcmp(sublinev[0], "face_surface") == 0)
    {
        sr = pj_new_face_surface(sublinec,
                                 sublinev, name, str, ostr);
        if (!sr)
        {
            ostr << "error defining surface " << (name ? name : "(no name)") << endl;
            goto usage;
        }
    }
    else
    {
        ostr << "unrecognized type " << sublinev[0];
        goto usage;
    }
    sr->family = family;
    sr->name = name;
    sr->width = width;
    sr->maxsize = maxsize;
    sr->ratio = ratio;
    sr->minsize = minsize;
    sr->dev = dev;
    sr->height = height;
    sr->reverse_normal = reverse_normal;
    sr->level = level;
    sr->number = number;
    sr->by_ids = by_ids;
    return;
usage:
    if (sr)
    {
        delete sr;
        sr = 0;
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
    }

    ostr << "Usage: " << linev[0] << " [family %%d] ";
    ostr << "[level %%d] [number %%d] [tetra_size %%d] ";
    ostr << "[width %%f] [ratio %%f] [dev %%f] [name %%s] ";
    ostr << "[height %%f] ";
    ostr << "(surface data on following lines)";
    return;
}

coTetin__param_surface *
coTetin__defSurf::pj_new_param_surface(int linec, char *linev[], char *name,
                                       istream &str, ostream &ostr)
{

    coTetin__param_surface *sr = new coTetin__param_surface();

    char type[100];
    strcpy(type, linev[0]);
    int il = 1;
    int nloops = -1;
    while (il < linec)
    {
        if (strcmp(linev[il], "n_loops") == 0)
        {
            il++;
            sr->new_format = 1;
            if (sscanf(linev[il], "%d", &nloops) != 1)
            {
                ostr << "error scanning number of loops" << endl;
                return 0;
            }
            il++;
        }
        else
        {
            ostr << "skipping trim_surface argument " << linev[il] << endl;
            il++;
        }
    }
    int i, j;
    if (strcmp(type, "trim_surface") == 0)
    {
        char line[LINE_SIZE];
        if (nloops == -1)
        {
            getLine(line, LINE_SIZE, str);
            if (str.eof() || strlen(line) == 0)
            {
                return 0;
            }
            if (sscanf(line, "%d", &nloops) != 1)
            {
                return 0;
            }
        }

        sr->n_loops = nloops;
        sr->loops = (coTetin__Loop **)new coTetin__Loop *[nloops];
        // read into Loop database

        coTetin__Loop *Lp;
        for (i = 0; i < nloops; i++)
        {
            getLine(line, LINE_SIZE, str);
            if (str.eof() || strlen(line) == 0)
            {
                return 0;
            }
            char *sublinev[100];
            int sublinec = coTetin__break_line(line, sublinev);
            if (sublinec >= 1 && strcmp(sublinev[0], "loop") == 0)
            {
                sr->loops[i] = read_new_loop(sublinec,
                                             sublinev, str, ostr);
                Lp = sr->loops[i];
                for (j = 0; j < Lp->ncoedges; j++)
                {
                    if (Lp->coedges[j]->p_curve == 0)
                    {
                        ostr << "null_pcurves not allowed for trim surfaces" << endl;
                        return 0;
                    }
                }
            }
            else if (sublinec == 1)
            {
                int npnts;
                if (sscanf(sublinev[0], "%d", &npnts) != 1)
                {
                    return 0;
                }
                sr->loops[i] = read_old_loop(npnts, str, ostr);
            }
            else
            {
                ostr << "error scanning number of points" << endl;
                return 0;
            }
            if (sr->loops[i] == 0)
            {
                return 0;
            }
        }

        str >> type;
    }
    if (strcmp(type, "bspline") != 0)
    {
        ostr << "unrecognized surface type " << type << endl;
        return 0;
    }

    // read and define a new bspline surface
    sr->surf = read_spline(str, ostr);

    if (sr->surf == 0)
    {
        ostr << "error defining surface " << (name ? name : "(no name)") << endl;
        return 0;
    }

    return sr;
}

coTetin__mesh_surface *
coTetin__defSurf::pj_new_mesh_surface(istream &str, ostream &ostr,
                                      int n_points, int n_tris)
{
    coTetin__mesh_surface *surf = new coTetin__mesh_surface();
    surf->npnts = n_points;
    surf->pnts = (point *)new point[n_points];
    char line[LINE_SIZE];
    int i;
    int j;
    for (i = 0; i < n_points; i++)
    {
        getLine(line, LINE_SIZE, str);
        if (str.eof() || strlen(line) == 0)
        {
            ostr << "error getting line" << endl;
            return 0;
        }
        if (sscanf(line, "%f %f %f",
                   surf->pnts[i],
                   surf->pnts[i] + 1,
                   surf->pnts[i] + 2) != 3)
        {
            ostr << "error scanning point coordinates" << endl;
            return 0;
        }
    }
    surf->subsurf.n_tri = n_tris;
    Triangle *tris = surf->subsurf.tris = (Triangle *)new Triangle[n_tris];
    for (i = 0; i < n_tris; i++)
    {
        getLine(line, LINE_SIZE, str);
        if (str.eof() || strlen(line) == 0)
        {
            ostr << "error getting line" << endl;
            return 0;
        }
        if (sscanf(line, "%d %d %d", tris[i],
                   tris[i] + 1, tris[i] + 2) != 3)
        {
            ostr << "error scanning triangle indices" << endl;
            return 0;
        }
        for (j = 0; j < 3; j++)
        {
            if (tris[i][j] < 0 || tris[i][j] >= n_points)
            {
                ostr << "point number " << tris[i][j] << " is out of range 0 to " << n_points - 1 << endl;
                return 0;
            }
        }
    }
    return surf;
}

coTetin__face_surface *
coTetin__defSurf::pj_new_face_surface(int linec, char *linev[], char *name,
                                      istream &str, ostream &ostr)
{
    coTetin__face_surface *sr = new coTetin__face_surface();
    int il = 1;
    sr->n_loops = 0;
    while (il < linec)
    {
        if (strcmp(linev[il], "n_loops") == 0)
        {
            il++;
            sr->new_format = 1;
            if (sscanf(linev[il], "%d", &(sr->n_loops)) != 1)
            {
                ostr << "error scanning number of loops" << endl;
                return 0;
            }
            il++;
        }
        else
        {
            ostr << "skipping face_surface argument " << linev[il] << endl;
            il++;
        }
    }
    if (sr->n_loops <= 0)
    {
        ostr << "illegal number of loops in surface: " << name << endl;
        return 0;
    }
    sr->loops = (coTetin__Loop **)new coTetin__Loop *[sr->n_loops];
    int i;
    for (i = 0; i < sr->n_loops; i++)
    {
        char line[LINE_SIZE];
        getLine(line, LINE_SIZE, str);
        if (str.eof() || strlen(line) == 0)
        {
            return 0;
        }
        char *sublinev[100];
        int sublinec = coTetin__break_line(line, sublinev);
        if (sublinec >= 1 && strcmp(sublinev[0], "loop") == 0)
        {
            sr->loops[i] = read_new_loop(sublinec,
                                         sublinev, str, ostr);
            if (sr->loops[i] == 0)
            {
                return 0;
            }
        }
    }
    return sr;
}

coTetin__Loop *coTetin__defSurf::read_new_loop(int linec, char *linev[],
                                               istream &str, ostream &ostr)
{
    int il = 1;
    int ncrvs = -1;
    while (il < linec)
    {
        if (strcmp(linev[il], "n_curves") == 0)
        {
            il++;
            if (sscanf(linev[il], "%d", &ncrvs) != 1)
            {
                return 0;
            }
            il++;
        }
        else
        {
            ostr << "skipping loop argument " << linev[il];
            il++;
        }
    }
    if (ncrvs == -1)
    {
        ostr << "mandatory agrument n_curves missing" << endl;
        return 0;
    }
    coTetin__Loop *lp = new coTetin__Loop();
    lp->ncoedges = ncrvs;
    lp->coedges = (coTetin__coedge **)new coTetin__coedge *[ncrvs];
    int i;
    for (i = 0; i < ncrvs; i++)
    {
        lp->coedges[i] = read_coedge(str, ostr);
        if (lp->coedges[i] == 0)
        {
            delete lp;
            return 0;
        }
    }
    return lp;
}

coTetin__Loop *coTetin__defSurf::read_old_loop(int npnts, istream &str, ostream &ostr)
{
    coTetin__Loop *lp = new coTetin__Loop();
    lp->ncoedges = 1;
    lp->coedges = (coTetin__coedge **)new coTetin__coedge *();
    lp->coedges[0] = new coTetin__coedge;
    coTetin__coedge *coe = lp->coedges[0];
    coTetin__pcurve *pc = coe->p_curve = new coTetin__pcurve;
    pc->type = STD_PCURVE;
    pc->npnts = npnts;
    twoD *pnts = pc->pnts = (twoD *)new twoD[npnts];

    int k, m;
    float ftemp;
    for (k = 0; k < npnts; k++)
    {
        for (m = 0; m < 2; m++)
        {
            if (getgplfloat(str, ostr, &ftemp))
            {
                delete lp;
                return 0;
            }
            pnts[k][m] = ftemp;
        }
    }
    return lp;
}

coTetin__coedge *coTetin__defSurf::read_coedge(istream &str, ostream &ostr)
{
    char line[LINE_SIZE];
    getLine(line, LINE_SIZE, str);
    if (str.eof() || strlen(line) == 0)
    {
        return 0;
    }
    char *linev[100];
    coTetin__coedge *coe = new coTetin__coedge;

    int linec = coTetin__break_line(line, linev);
    if (strcmp(linev[0], "coedge") == 0)
    {
        int il;
        il = 1;
        while (il < linec)
        {
            if (strcmp(linev[il], "3dcurve") == 0)
            {
                il++;
                coe->rev = 0;
                if (linev[il][0] == '-' && linev[il][1] == 0)
                {
                    coe->rev = 1;
                    il++;
                }
                coe->curve_name = (char *)new char[strlen(linev[il]) + 1];
                strcpy(coe->curve_name, linev[il]);
                il++;
            }
            else
            {
                ostr << "skipping coedge parameter " << linev[il] << endl;
                il++;
            }
        }
        getLine(line, LINE_SIZE, str);
        if (str.eof() || strlen(line) == 0)
        {
            return coe;
        }
        linec = coTetin__break_line(line, linev);
    }
    if (strcmp(linev[0], "polyline") == 0)
    {
        coe->p_curve = read_polyline(linec, linev, str, ostr);
    }
    else if (strcmp(linev[0], "null_pcurve") == 0)
    {
        coe->p_curve = 0;
    }
    else
    {
        ostr << "unknown 2d curve type " << linev[0] << endl;
    }
    return coe;
}

coTetin__pcurve *coTetin__defSurf::read_polyline(int linec, char *linev[],
                                                 istream &str, ostream &ostr)
{
    int il = 1;
    int npnts = -1;
    while (il < linec)
    {
        if (strcmp(linev[il], "n_points") == 0)
        {
            il++;
            if (sscanf(linev[il], "%d", &npnts) != 1)
            {
                ostr << "error scanning number of points" << endl;
                return 0;
            }
            il++;
        }
        else
        {
            ostr << "skipping polyline argument " << linev[il]
                 << endl;
            il++;
        }
    }
    if (npnts == -1)
    {
        ostr << "mandatory arguement n_points missing" << endl;
        return 0;
    }
    coTetin__pcurve *pc = new coTetin__pcurve;

    pc->type = STD_PCURVE;
    pc->pnts = (twoD *)new twoD[npnts];
    pc->npnts = npnts;
    int i, j;
    for (i = 0; i < npnts; i++)
    {
        for (j = 0; j < 2; j++)
        {
            if (getgplfloat(str, ostr, pc->pnts[i] + j))
            {
                return 0;
            }
        }
    }
    return pc;
}

coTetin__surface_record::coTetin__surface_record(int typ)
{
    family = 0;
    maxsize = 0.0;
    width = 0.0;
    height = 0.0;
    ratio = 0.0;
    minsize = 0.0;
    dev = 0.0;
    name = 0;
    level = 0;
    number = 0;
    name = 0;
    surface_type = typ;
    new_format = 0;
}

coTetin__surface_record::~coTetin__surface_record()
{
    //	if(surface_type == INVALID_SURFACE) {
    //		fprintf(stderr, "surface has already been deleted!\n");
    //	}
    family = 0;
    reverse_normal = 0;
    width = 0.0;
    maxsize = 0.0;
    ratio = 0.0;
    minsize = 0.0;
    dev = 0.0;
    height = 0.0;
    if (name)
        delete[] name;
    name = 0;
    level = 0;
    number = 0;
    surface_type = INVALID_SURFACE;
    new_format = 0;
}

coTetin__pcurve::coTetin__pcurve()
{
    type = STD_PCURVE;
    npnts = 0;
    pnts = 0;
}

coTetin__pcurve::~coTetin__pcurve()
{
    npnts = 0;
    if (pnts)
        delete[] pnts;
    pnts = 0;
}

coTetin__Loop::coTetin__Loop()
{
    ncoedges = 0;
    coedges = 0;
}

coTetin__Loop::~coTetin__Loop()
{
    int i;
    for (i = 0; i < ncoedges; i++)
    {
        if (coedges[i])
            delete coedges[i];
    }
    if (coedges)
        delete[] coedges;
    coedges = 0;
    ncoedges = 0;
}

coTetin__mesh_subsurface::coTetin__mesh_subsurface()
{
    n_tri = 0;
    tris = 0;
}

coTetin__mesh_subsurface::~coTetin__mesh_subsurface()
{
    n_tri = 0;
    if (tris)
        delete[] tris;
    tris = 0;
}

coTetin__mesh_surface::coTetin__mesh_surface()
    : coTetin__surface_record(MESH_SURFACE)
{
    npnts = 0;
    pnts = 0;
}

coTetin__mesh_surface::~coTetin__mesh_surface()
{
    npnts = 0;
    if (pnts)
        delete[] pnts;
    pnts = 0;
}

coTetin__unstruct_surface::coTetin__unstruct_surface()
    : coTetin__surface_record(UNSTRUCT_SURFACE)
{
    path = 0;
}

coTetin__unstruct_surface::coTetin__unstruct_surface(char *name)
    : coTetin__surface_record(UNSTRUCT_SURFACE)
{
    path = name;
}

coTetin__unstruct_surface::~coTetin__unstruct_surface()
{
    if (path)
        delete[] path;
    path = 0;
}

coTetin__param_surface::coTetin__param_surface()
    : coTetin__surface_record(PARAMETRIC_SURFACE)
{
    surf = 0;
    n_loops = 0;
    loops = 0;
}

coTetin__param_surface::~coTetin__param_surface()
{
    if (surf)
        delete surf;
    surf = 0;
    int i;
    for (i = 0; i < n_loops; i++)
    {
        delete loops[i];
        loops[i] = 0;
    }
    if (loops)
        delete[] loops;
    loops = 0;
    n_loops = 0;
}

coTetin__face_surface::coTetin__face_surface()
    : coTetin__surface_record(FACE_SURFACE)
{
    n_loops = 0;
    loops = 0;
}

coTetin__face_surface::~coTetin__face_surface()
{
    int i;
    for (i = 0; i < n_loops; i++)
    {
        delete loops[i];
        loops[i] = 0;
    }
    if (loops)
        delete[] loops;
    loops = 0;
    n_loops = 0;
}

coTetin__coedge::coTetin__coedge()
{
    curve_name = 0;
    p_curve = 0;
    rev = 0;
}

coTetin__coedge::~coTetin__coedge()
{
    if (curve_name)
        delete[] curve_name;
    curve_name = 0;
    if (p_curve)
        delete p_curve;
    p_curve = 0;
    rev = 0;
}

/// read from memory
coTetin__defSurf::coTetin__defSurf(int *&intDat, float *&floatDat, char *&charDat)
    : coTetinCommand(coTetin::DEFINE_SURFACE)
{
    current_defSurf = this;

    coTetin__param_surface *ps = 0;
    coTetin__mesh_surface *ms = 0;
    coTetin__face_surface *fs = 0;
    coTetin__unstruct_surface *us = 0;
    char surface_type;
    surface_type = *charDat++;
    switch (surface_type)
    {
    case PARAMETRIC_SURFACE:
        ps = new coTetin__param_surface();
        sr = (coTetin__surface_record *)ps;
        break;
    case MESH_SURFACE:
        ms = new coTetin__mesh_surface();
        sr = (coTetin__surface_record *)ms;
        break;
    case FACE_SURFACE:
        fs = new coTetin__face_surface();
        sr = (coTetin__surface_record *)fs;
        break;
    case UNSTRUCT_SURFACE:
        us = new coTetin__unstruct_surface();
        sr = (coTetin__surface_record *)us;
        break;
    }

    sr->new_format = *charDat++;
    sr->family = getString(charDat);
    sr->reverse_normal = *intDat++;
    sr->width = *floatDat++;
    sr->maxsize = *floatDat++;
    sr->ratio = *floatDat++;
    sr->minsize = *floatDat++;
    sr->dev = *floatDat++;
    sr->height = *floatDat++;
    sr->name = getString(charDat);
    sr->level = *intDat++;
    sr->number = *intDat++;
    sr->by_ids = *intDat++;

    switch (surface_type)
    {
    case PARAMETRIC_SURFACE:
        ps->putBinary(intDat, floatDat, charDat);
        break;
    case MESH_SURFACE:
        ms->putBinary(intDat, floatDat, charDat);
        break;
    case FACE_SURFACE:
        fs->putBinary(intDat, floatDat, charDat);
        break;
    case UNSTRUCT_SURFACE:
        us->putBinary(intDat, floatDat, charDat);
        break;
    }
}

void coTetin__param_surface::putBinary(int *&intDat, float *&floatDat, char *&charDat)
{
    surf = new coTetin__BSpline();
    surf->putBinary(intDat, floatDat, charDat);
    n_loops = *intDat++;
    loops = 0;
    if (n_loops > 0)
    {
        loops = new coTetin__Loop *[n_loops];
        int i;
        for (i = 0; i < n_loops; i++)
        {
            loops[i] = new coTetin__Loop();
            loops[i]->putBinary(intDat, floatDat, charDat);
        }
    }
}

void coTetin__face_surface::putBinary(int *&intDat, float *&floatDat, char *&charDat)
{
    n_loops = *intDat++;
    loops = 0;
    if (n_loops > 0)
    {
        loops = new coTetin__Loop *[n_loops];
        int i;
        for (i = 0; i < n_loops; i++)
        {
            loops[i] = new coTetin__Loop();
            loops[i]->putBinary(intDat, floatDat, charDat);
        }
    }
}

void coTetin__unstruct_surface::putBinary(int *&intDat, float *&floatDat, char *&charDat)
{
    path = current_defSurf->getNextString(charDat);
}

void coTetin__mesh_surface::putBinary(int *&intDat, float *&floatDat, char *&charDat)
{
    npnts = *intDat++;
    pnts = 0;
    if (npnts > 0)
    {
        pnts = new point[npnts];
        memcpy((void *)pnts, (void *)floatDat, 3 * npnts * sizeof(float));
        floatDat += 3 * npnts;
    }
    subsurf.n_tri = *intDat++;
    subsurf.tris = 0;
    if (subsurf.n_tri > 0)
    {
        subsurf.tris = new Triangle[subsurf.n_tri];
        memcpy((void *)subsurf.tris, (void *)intDat, 3 * subsurf.n_tri * sizeof(int));
        intDat += 3 * subsurf.n_tri;
    }
}

void coTetin__Loop::putBinary(int *&intDat, float *&floatDat, char *&charDat)
{
    ncoedges = *intDat++;
    coedges = 0;
    if (ncoedges > 0)
    {
        coedges = new coTetin__coedge *[ncoedges];
        int i;
        for (i = 0; i < ncoedges; i++)
        {
            coedges[i] = new coTetin__coedge();
            coedges[i]->putBinary(intDat, floatDat, charDat);
        }
    }
}

void coTetin__coedge::putBinary(int *&intDat, float *&floatDat, char *&charDat)
{
    curve_name = current_defSurf->getNextString(charDat);
    rev = *intDat++;
    int pcurve_valid = *intDat++;
    p_curve = 0;
    if (pcurve_valid)
    {
        p_curve = new coTetin__pcurve();
        p_curve->type = *intDat++;
        p_curve->npnts = *intDat++;
        if (p_curve->npnts > 0)
        {
            p_curve->pnts = new twoD[p_curve->npnts];
            memcpy((void *)p_curve->pnts, (void *)floatDat,
                   2 * p_curve->npnts * sizeof(float));
            floatDat += 2 * p_curve->npnts;
        }
    }
}

/// Destructor
coTetin__defSurf::~coTetin__defSurf()
{
    current_defSurf = 0;
    if (sr)
    {
        delete sr;
        sr = 0;
    }
}

/// check whether Object is valid
int coTetin__defSurf::isValid() const
{
    if (!sr || (sr->surface_type != PARAMETRIC_SURFACE && sr->surface_type != MESH_SURFACE && sr->surface_type != FACE_SURFACE && sr->surface_type != UNSTRUCT_SURFACE))
        return 0;
    else
        return 1;
}

/// count size required in fields
void coTetin__defSurf::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    if (!isValid())
        return;
    numInt += 5; // comm,reverse_normal,level,number,by_ids
    numFloat += 6; // height,dev,minsize,ratio,width,maxsize
    numChar += 2; // new_format,surface_type
    numChar += (sr->family) ? (strlen(sr->family) + 1) : 1;
    numChar += (sr->name) ? (strlen(sr->name) + 1) : 1;

    coTetin__param_surface *ps = 0;
    coTetin__mesh_surface *ms = 0;
    coTetin__face_surface *fs = 0;
    coTetin__unstruct_surface *us = 0;
    switch (sr->surface_type)
    {
    case PARAMETRIC_SURFACE:
        ps = (coTetin__param_surface *)sr;
        ps->addSizes(numInt, numFloat, numChar);
        break;
    case MESH_SURFACE:
        ms = (coTetin__mesh_surface *)sr;
        ms->addSizes(numInt, numFloat, numChar);
        break;
    case FACE_SURFACE:
        fs = (coTetin__face_surface *)sr;
        fs->addSizes(numInt, numFloat, numChar);
        break;
    case UNSTRUCT_SURFACE:
        us = (coTetin__unstruct_surface *)sr;
        us->addSizes(numInt, numFloat, numChar);
        break;
    }
}

void coTetin__param_surface::addBSizes(int &numInt, int &numFloat, int &numChar) const
{
    if (surf)
    {
        surf->addSizes(numInt, numFloat, numChar);
        numInt += 1; // n_loops
        if (n_loops > 0 && loops)
        {
            int i;
            for (i = 0; i < n_loops; i++)
            {
                loops[i]->addBSizes(numInt, numFloat, numChar);
            }
        }
    }
}

void coTetin__face_surface::addBSizes(int &numInt, int &numFloat, int &numChar) const
{
    numInt += 1; // n_loops
    if (n_loops > 0 && loops)
    {
        int i;
        for (i = 0; i < n_loops; i++)
        {
            loops[i]->addBSizes(numInt, numFloat, numChar);
        }
    }
}

void coTetin__unstruct_surface::addBSizes(int &numInt, int &numFloat, int &numChar) const
{
    numChar += (path) ? (strlen(path) + 1) : 1;
}

void coTetin__mesh_surface::addBSizes(int &numInt, int &numFloat, int &numChar) const
{
    numInt += 1; // npnts
    if (pnts)
        numFloat += 3 * npnts; // pnts
    numInt += 1; // subsurf.n_tri
    if (subsurf.tris)
        numInt += 3 * subsurf.n_tri; // subsurf.tris
}

void coTetin__Loop::addBSizes(int &numInt, int &numFloat, int &numChar) const
{
    numInt += 1; // ncoedges
    if (ncoedges > 0 && coedges)
    {
        int i;
        for (i = 0; i < ncoedges; i++)
        {
            coedges[i]->addBSizes(numInt, numFloat, numChar);
        }
    }
}

void coTetin__coedge::addBSizes(int &numInt, int &numFloat, int &numChar) const
{
    numChar += (curve_name) ? (strlen(curve_name) + 1) : 1;
    numInt += 2; // rev, p_curve_valid
    if (p_curve)
    {
        numInt += 2; // p_curve->type,p_curve->npnts
        // p_curve->pnts
        if (p_curve->pnts)
            numFloat += 2 * p_curve->npnts;
    }
}

/// put my data to a given set of pointers
void coTetin__defSurf::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    if (!isValid())
        return;

    // copy the command's name
    *intDat++ = d_comm;

    *charDat++ = sr->surface_type;
    *charDat++ = sr->new_format;
    // copy the data
    if (sr->family)
    {
        strcpy(charDat, sr->family);
        charDat += strlen(sr->family) + 1;
    }
    else
    {
        *charDat++ = '\0';
    }
    *intDat++ = sr->reverse_normal;
    *floatDat++ = sr->width;
    *floatDat++ = sr->maxsize;
    *floatDat++ = sr->ratio;
    *floatDat++ = sr->minsize;
    *floatDat++ = sr->dev;
    *floatDat++ = sr->height;
    if (sr->name)
    {
        strcpy(charDat, sr->name);
        charDat += strlen(sr->name) + 1;
    }
    else
    {
        *charDat++ = '\0';
    }
    *intDat++ = sr->level;
    *intDat++ = sr->number;
    *intDat++ = sr->by_ids;

    coTetin__param_surface *ps = 0;
    coTetin__mesh_surface *ms = 0;
    coTetin__face_surface *fs = 0;
    coTetin__unstruct_surface *us = 0;
    switch (sr->surface_type)
    {
    case PARAMETRIC_SURFACE:
        ps = (coTetin__param_surface *)sr;
        ps->internal_write(cout, intDat, floatDat, charDat);
        break;
    case MESH_SURFACE:
        ms = (coTetin__mesh_surface *)sr;
        ms->internal_write(cout, intDat, floatDat, charDat);
        break;
    case FACE_SURFACE:
        fs = (coTetin__face_surface *)sr;
        fs->internal_write(cout, intDat, floatDat, charDat);
        break;
    case UNSTRUCT_SURFACE:
        us = (coTetin__unstruct_surface *)sr;
        us->internal_write(cout, intDat, floatDat, charDat);
        break;
    }
}

void coTetin__param_surface::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    if (surf)
    {
        surf->getBinary(intDat, floatDat, charDat);
        *intDat++ = n_loops;
        if (n_loops > 0 && loops)
        {
            int i;
            for (i = 0; i < n_loops; i++)
            {
                loops[i]->getBinary(intDat, floatDat, charDat);
            }
        }
    }
}

void coTetin__face_surface::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    *intDat++ = n_loops;
    if (n_loops > 0 && loops)
    {
        int i;
        for (i = 0; i < n_loops; i++)
        {
            loops[i]->getBinary(intDat, floatDat, charDat);
        }
    }
}

void coTetin__unstruct_surface::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
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

void coTetin__mesh_surface::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    *intDat++ = npnts;
    if (pnts && npnts > 0)
    {
        memcpy((void *)floatDat, (void *)pnts, 3 * npnts * sizeof(float));
        floatDat += 3 * npnts;
    }
    *intDat++ = subsurf.n_tri;
    if (subsurf.tris && subsurf.n_tri > 0)
    {
        memcpy((void *)intDat, (void *)subsurf.tris, 3 * subsurf.n_tri * sizeof(int));
        intDat += 3 * subsurf.n_tri;
    }
}

void coTetin__Loop::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    *intDat++ = ncoedges;
    if (ncoedges > 0 && coedges)
    {
        int i;
        for (i = 0; i < ncoedges; i++)
        {
            coedges[i]->getBinary(intDat, floatDat, charDat);
        }
    }
}

void coTetin__coedge::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    if (curve_name)
    {
        strcpy(charDat, curve_name);
        charDat += strlen(curve_name) + 1;
    }
    else
    {
        *charDat++ = '\0';
    }
    *intDat++ = rev;
    if (p_curve)
    {
        *intDat++ = 1;
        *intDat++ = p_curve->type;
        *intDat++ = p_curve->npnts;
        if (p_curve->pnts && p_curve->npnts > 0)
        {
            memcpy((void *)floatDat, (void *)p_curve->pnts,
                   2 * p_curve->npnts * sizeof(float));
            floatDat += 2 * p_curve->npnts;
        }
    }
    else
    {
        *intDat++ = 0;
    }
}

/// print to a stream in Tetin format
void coTetin__defSurf::print(ostream &str) const
{
    if (isValid())
    {
        str << "define_surface";
        if (sr->reverse_normal)
        {
            str << " reverse_normal";
        }
        if (sr->height > 0.0)
        {
            str << " height " << sr->height;
        }
        if (sr->dev > 0.0)
        {
            str << " dev " << sr->dev;
        }
        if (sr->minsize > 0.0)
        {
            str << " min " << sr->minsize;
        }
        if (sr->ratio > 0.0)
        {
            str << " ratio " << sr->ratio;
        }
        if (sr->width > 0.0)
        {
            str << " width " << sr->width;
        }
        if (sr->name && sr->name[0])
        {
            str << " name " << sr->name;
        }
        if (sr->family && sr->family[0])
        {
            str << " family " << sr->family;
        }
        if (sr->level)
        {
            str << " level " << sr->level;
        }
        if (sr->number)
        {
            str << " number " << sr->number;
        }
        if (sr->maxsize > 0.0)
        {
            str << " tetra_size " << sr->maxsize;
        }
        str << endl;
        int *iDat = 0;
        float *fDat = 0;
        char *cDat = 0;
        sr->internal_write(str, iDat, fDat, cDat);
    } // if (isValid())
}

void coTetin__surface_record::internal_write(ostream &str,
                                             int *&intDat, float *&floatDat, char *&charDat) const
{
    if (surface_type == PARAMETRIC_SURFACE)
    {
        coTetin__param_surface *ps = (coTetin__param_surface *)this;
        ps->internal_write(str, intDat, floatDat, charDat);
    }
    else if (surface_type == MESH_SURFACE)
    {
        coTetin__mesh_surface *us = (coTetin__mesh_surface *)this;
        us->internal_write(str, intDat, floatDat, charDat);
    }
    else if (surface_type == FACE_SURFACE)
    {
        coTetin__face_surface *fs = (coTetin__face_surface *)this;
        fs->internal_write(str, intDat, floatDat, charDat);
    }
    else if (surface_type == UNSTRUCT_SURFACE)
    {
        coTetin__unstruct_surface *us = (coTetin__unstruct_surface *)this;
        us->internal_write(str, intDat, floatDat, charDat);
    }
    else
    {
        printf("internal_write not implemented for this type\n");
        exit(1);
    }
}

static char cov_tmp_array[2000];
void coTetin__param_surface::internal_write(ostream &str,
                                            int *&intDat, float *&floatDat, char *&charDat) const
{
    int i, j;
    if (!surf)
        return;

    int write_bin;
    if (intDat && floatDat && charDat)
    {
        write_bin = 1;
    }
    else
    {
        write_bin = 0;
    }
    if (n_loops)
    {
        if (write_bin)
        {
            sprintf(cov_tmp_array, "trim_surface n_loops %d", n_loops);
            strcpy(charDat, cov_tmp_array);
            charDat += strlen(cov_tmp_array) + 1;
        }
        else
        {
            str << "trim_surface n_loops " << n_loops << endl;
        }
        for (i = 0; i < n_loops; i++)
        {
            loops[i]->write(str, intDat, floatDat, charDat);
        }
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
    int ncps[2], ord[2];
    int rat = surf->rat;
    int nt = 1;
    for (i = 0; i < 2; i++)
    {
        ncps[i] = surf->ncps[i];
        ord[i] = surf->ord[i];
        nt *= ncps[i];
    }

    if (write_bin)
    {
        *intDat++ = ncps[0];
        *intDat++ = ncps[1];
        *intDat++ = ord[0];
        *intDat++ = ord[1];
        *intDat++ = rat;
    }
    else
    {
        str << ncps[0] << ',' << ncps[1] << ',' << ord[0] << ',' << ord[1] << ',' << rat << endl;
    }
    // write out the knot vectors
    for (i = 0; i < 2; i++)
    {
        int nkt = ncps[i] + ord[i];
        for (j = 0; j < nkt; j++)
        {
            if (write_bin)
            {
                *floatDat++ = surf->knot[i][j];
            }
            else
            {
                str << surf->knot[i][j];
                // only write 5 reals per line
                if ((j + 1) % 5 == 0 || j == nkt - 1)
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
    // write out the control points
    float *cps = surf->cps;
    if (rat)
    {
        for (i = 0; i < nt; i++)
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
    }
    else
    {
        for (i = 0; i < nt; i++)
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

void coTetin__face_surface::internal_write(ostream &str,
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
        sprintf(cov_tmp_array, "face_surface n_loops %d", n_loops);
        strcpy(charDat, cov_tmp_array);
        charDat += strlen(cov_tmp_array) + 1;
    }
    else
    {
        str << "face_surface n_loops " << n_loops << endl;
    }
    int i;
    for (i = 0; i < n_loops; i++)
    {
        loops[i]->write(str, intDat, floatDat, charDat);
    }
}

void coTetin__mesh_surface::internal_write(ostream &str,
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
        sprintf(cov_tmp_array, "unstruct_mesh n_points %d n_triangles %d", npnts,
                subsurf.n_tri);
        strcpy(charDat, cov_tmp_array);
        charDat += strlen(cov_tmp_array) + 1;
    }
    else
    {
        str << "unstruct_mesh n_points " << npnts << " n_triangles " << subsurf.n_tri << endl;
    }
    int i;
    for (i = 0; i < npnts; i++)
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
    for (i = 0; i < subsurf.n_tri; i++)
    {
        if (write_bin)
        {
            *intDat++ = subsurf.tris[i][0];
            *intDat++ = subsurf.tris[i][1];
            *intDat++ = subsurf.tris[i][2];
        }
        else
        {
            str << subsurf.tris[i][0] << ' ' << subsurf.tris[i][1] << ' ' << subsurf.tris[i][2] << endl;
        }
    }
}

void coTetin__unstruct_surface::internal_write(ostream &str,
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
    if (path)
    {
        if (write_bin)
        {
            sprintf(cov_tmp_array, "unstruct_mesh %s", path);
            strcpy(charDat, cov_tmp_array);
            charDat += strlen(cov_tmp_array) + 1;
        }
        else
        {
            str << "unstruct_mesh " << path << endl;
        }
    }
}

void coTetin__Loop::write(ostream &str,
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
        sprintf(cov_tmp_array, "loop n_curves %d", ncoedges);
        strcpy(charDat, cov_tmp_array);
        charDat += strlen(cov_tmp_array) + 1;
    }
    else
    {
        str << "loop n_curves " << ncoedges << endl;
    }
    int i;
    for (i = 0; i < ncoedges; i++)
    {
        coedges[i]->write(str, intDat, floatDat, charDat);
    }
}

void coTetin__coedge::write(ostream &str,
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
        if (curve_name)
        {
            if (rev)
            {
                sprintf(cov_tmp_array, "coedge 3dcurve - %s", curve_name);
            }
            else
            {
                sprintf(cov_tmp_array, "coedge 3dcurve %s", curve_name);
            }
        }
        else
        {
            sprintf(cov_tmp_array, "coedge");
        }
        strcpy(charDat, cov_tmp_array);
        charDat += strlen(cov_tmp_array) + 1;
    }
    else
    {
        str << "coedge";
        // write the 3-d curve with a - in front if it is reversed
        if (curve_name)
        {
            str << " 3dcurve " << (rev ? "- " : " ") << curve_name;
        }
        str << endl;
    }
    if (p_curve)
    {
        p_curve->write(str, intDat, floatDat, charDat);
    }
    else
    {
        if (write_bin)
        {
            sprintf(cov_tmp_array, "null_pcurve");
            strcpy(charDat, cov_tmp_array);
            charDat += strlen(cov_tmp_array) + 1;
        }
        else
        {
            str << "null_pcurve" << endl;
        }
    }
}

void coTetin__pcurve::write(ostream &str,
                            int *&intDat, float *&floatDat, char *&charDat) const
{
    int i;
    int write_bin;
    if (intDat && floatDat && charDat)
    {
        write_bin = 1;
    }
    else
    {
        write_bin = 0;
    }
    if (type == STD_PCURVE)
    {
        if (write_bin)
        {
            sprintf(cov_tmp_array, "polyline n_points %d", npnts);
            strcpy(charDat, cov_tmp_array);
            charDat += strlen(cov_tmp_array) + 1;
        }
        else
        {
            str << "polyline n_points " << npnts << endl;
        }
        for (i = 0; i < npnts; i++)
        {
            if (write_bin)
            {
                *floatDat++ = pnts[i][0];
                *floatDat++ = pnts[i][1];
            }
            else
            {
                str << pnts[i][0] << ',' << pnts[i][1] << endl;
            }
        }
    }
    else
    {
        fprintf(stdout, "unknown pcurve type\n");
    }
}

void coTetin__param_surface::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    int i;
    if (surf)
    {
        if (n_loops)
        {
            sprintf(cov_tmp_array, "trim_surface n_loops %d", n_loops);
            numChar += strlen(cov_tmp_array) + 1;
            for (i = 0; i < n_loops; i++)
            {
                loops[i]->addSizes(numInt, numFloat, numChar);
            }
        }
        sprintf(cov_tmp_array, "bspline");
        numChar += strlen(cov_tmp_array) + 1;
        surf->addSizes(numInt, numFloat, numChar);
    }
}

void coTetin__face_surface::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    int i;
    sprintf(cov_tmp_array, "face_surface n_loops %d", n_loops);
    numChar += strlen(cov_tmp_array) + 1;
    for (i = 0; i < n_loops; i++)
    {
        loops[i]->addSizes(numInt, numFloat, numChar);
    }
}

void coTetin__mesh_surface::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    sprintf(cov_tmp_array, "unstruct_mesh n_points %d n_triangles %d", npnts,
            subsurf.n_tri);
    numChar += strlen(cov_tmp_array) + 1;
    if (pnts)
        numFloat += 3 * npnts; // pnts
    if (subsurf.tris)
        numInt += 3 * subsurf.n_tri; // subsurf.tris
}

void coTetin__unstruct_surface::addSizes(int &numInt, int &numFloat,
                                         int &numChar) const
{
    if (path)
    {
        sprintf(cov_tmp_array, "unstruct_mesh %s", path);
        numChar += strlen(cov_tmp_array) + 1;
    }
}

void coTetin__Loop::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    sprintf(cov_tmp_array, "loop n_curves %d", ncoedges);
    numChar += strlen(cov_tmp_array) + 1;
    int i;
    for (i = 0; i < ncoedges; i++)
    {
        coedges[i]->addSizes(numInt, numFloat, numChar);
    }
}

void coTetin__coedge::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    if (curve_name)
    {
        if (rev)
        {
            sprintf(cov_tmp_array, "coedge 3dcurve - %s", curve_name);
        }
        else
        {
            sprintf(cov_tmp_array, "coedge 3dcurve %s", curve_name);
        }
    }
    else
    {
        sprintf(cov_tmp_array, "coedge");
    }
    numChar += strlen(cov_tmp_array) + 1;
    if (p_curve)
    {
        p_curve->addSizes(numInt, numFloat, numChar);
    }
    else
    {
        sprintf(cov_tmp_array, "null_pcurve");
        numChar += strlen(cov_tmp_array) + 1;
    }
}

void coTetin__pcurve::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    if (type == STD_PCURVE)
    {
        sprintf(cov_tmp_array, "polyline n_points %d", npnts);
        numChar += strlen(cov_tmp_array) + 1;
        if (pnts)
            numFloat += 2 * npnts;
    }
}

char *coTetin__defSurf::getNextString(char *&chPtr)
{
    return getString(chPtr);
}

// ===================== command-specific functions =====================
