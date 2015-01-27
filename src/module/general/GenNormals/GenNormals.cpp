/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Interpolation from Cell Data to Vertex Data               **
 **                                                                        **
 **                                                                        **
\**************************************************************************/
#include <util/coviseCompat.h>
#include "GenNormals.h"
#include <iostream>
#include <do/coDoData.h>

void Negat(float *v, float *u, float *l);
void Normale(float *c, float euklc, int corner, int totalcorners, float *NU, float *NV, float *NW, float *g);
void Norm(float *v, float *u, float lva, float lvb, float *n, float *f);
float Length(float a, float b, float c);
void Vec(int *vl, int *ll, int i, int corner, float *x, float *y, float *z, float *v, float *u);

GenNormals::GenNormals(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Work out normals for lines")
{
    //select normaltype
    p_normalstyle = addChoiceParam("LineNormalStyle", "Which Normal Style For Lines");
    const char *choLabels[] = { "BisectLargeAngle", "BisectSmallAngle", "Orthogonal" };

    p_normalstyle->setValue(3, choLabels, 0);
    // Ports
    p_inPort = addInputPort("GridIn0", "Polygons|Lines", "Grid, polygonal or linear input");
    p_outPort = addOutputPort("DataOut0", "Vec3", "Normals");
}

int GenNormals::compute(const char *)
{
    // now let's do the work
    const coDistributedObject *in_obj = p_inPort->getCurrentObject();
    // we should have an object
    if (!in_obj)
    {
        sendError("Did not receive object at port '%s'", p_inPort->getName());
        return FAIL;
    }

    // it should be the correct type
    if (!in_obj->isType("POLYGN") && !in_obj->isType("LINES"))
    {
        sendError("Received illegal type '%s' at port '%s'", in_obj->getType(), p_inPort->getName());
        return FAIL;
    }

    coDoVec3 *out_obj = (coDoVec3 *)
        gen_normals(in_obj, p_outPort->getObjName());
    p_outPort->setCurrentObject(out_obj);

    normalstyle = (NormalSelectMap)p_normalstyle->getValue();

    // bye
    return SUCCESS;
}

////// workin' routines
coDistributedObject *GenNormals::gen_normals(const coDistributedObject *mesh_in,
                                             const char *out_name)
{
    const coDoLines *lines;
    const coDoPolygons *polygons;
    coDoVec3 *normals = NULL;
    float *x, *y, *z, *NU, *NV, *NW, *U, *V, *W, *F_Normals_U, *F_Normals_V, *F_Normals_W;
    int *vl, *pl;
    int num_n, *nl, *nli, numpoly, numlines, numcoord;
    int n0, n1, n2, n, i;
    float x1, x2, y1, y2, z1, z2, l; //,ang;
    const char *ptype = mesh_in->getType();

    if (strcmp(ptype, "POLYGN") == 0 || strcmp(ptype, "LINES") == 0)
    {
        if (strcmp(ptype, "POLYGN") == 0)
        {
            polygons = (const coDoPolygons *)mesh_in;
            numpoly = polygons->getNumPolygons();
            int numvert = polygons->getNumVertices();
            numcoord = polygons->getNumPoints();
            polygons->getAddresses(&x, &y, &z, &vl, &pl);
            polygons->getNeighborList(&num_n, &nl, &nli);
            normals = new coDoVec3(out_name, numcoord);
            normals->getAddresses(&NU, &NV, &NW);

            U = F_Normals_U = new float[numpoly];
            V = F_Normals_V = new float[numpoly];
            W = F_Normals_W = new float[numpoly];

            for (i = 0; i < numpoly; i++)
            {
                // find out number of corners
                int no_corners;
                if (i < numpoly - 1)
                {
                    no_corners = pl[i + 1] - pl[i];
                }
                else
                {
                    no_corners = numvert - pl[i];
                }
                int triangle;
                l = 0.0;
                for (triangle = 0; triangle < no_corners - 2; ++triangle)
                {
                    n0 = vl[pl[i] + triangle];
                    n1 = vl[pl[i] + 1 + triangle];
                    n2 = vl[pl[i] + 2 + triangle];
                    x1 = x[n1] - x[n0];
                    y1 = y[n1] - y[n0];
                    z1 = z[n1] - z[n0];
                    x2 = x[n2] - x[n0];
                    y2 = y[n2] - y[n0];
                    z2 = z[n2] - z[n0];
                    *U = y1 * z2 - y2 * z1;
                    *V = x2 * z1 - x1 * z2;
                    *W = x1 * y2 - x2 * y1;
                    l = sqrt(*U * *U + *V * *V + *W * *W);
                    if (l != 0.0)
                    {
                        break;
                    }
                }
                if (l != 0.0)
                {
                    *U /= l;
                    *V /= l;
                    *W /= l;
                }
                else
                {
                    *U = 0.0;
                    *V = 0.0;
                    *W = 0.0;
                }
                U++;
                V++;
                W++;
            }

            for (i = 0; i < numcoord; i++)
            {
                *NU = *NV = *NW = 0;
                for (n = nli[i]; n < nli[i + 1]; n++)
                {
                    *NU += F_Normals_U[nl[n]];
                    *NV += F_Normals_V[nl[n]];
                    *NW += F_Normals_W[nl[n]];
                }
                float l = sqrt(*NU * *NU + *NV * *NV + *NW * *NW);
                if (l > 0.0)
                {
                    *NU /= l;
                    *NV /= l;
                    *NW /= l;
                }
                NU++;
                NV++;
                NW++;
            }
            delete[] F_Normals_U;
            delete[] F_Normals_V;
            delete[] F_Normals_W;
        }
        else
        {
            //three types of normals for lines:
            //bicection of the larger angle between the two vectors va and  vb
            //bicection of the smaller angle between the two vectors va and  vb
            //orthogonal normals
            if (strcmp(ptype, "LINES") == 0)
            {
                int *vl, *ll, no_corners, corner, totalcorners = 0;
                lines = (const coDoLines *)mesh_in;
                numlines = lines->getNumLines();
                int numvert = lines->getNumVertices();
                lines->getAddresses(&x, &y, &z, &vl, &ll);
                normals = new coDoVec3(out_name, numvert);
                normals->getAddresses(&NU, &NV, &NW);

                for (i = 0; i < numlines; i++)
                {
                    if (i < numlines - 1)
                    {
                        no_corners = ll[i + 1] - ll[i];
                    }
                    else
                    {
                        no_corners = numvert - ll[i];
                    }

                    //Case: normal for one point
                    if (no_corners == 1 || no_corners == 2)
                    {
                        corner = 0;
                        NU[corner + totalcorners] = 0.0;
                        NV[corner + totalcorners] = 0.0;
                        NW[corner + totalcorners] = 0.0;
                    }
                    else
                    {
                        //Case: points>=2
                        //normal for the first point
                        for (corner = 0; corner < 1; corner++)
                        {
                            float mu, f[3], va[3], vb[3], lva, lvb;
                            float nva[3], nvb[3], p[3], g[3], euklp, Leng, d[3];
                            float euklf;
                            int n0, n1, n2;
                            n0 = vl[ll[i] + 1 + corner];
                            n1 = vl[ll[i] + corner];
                            n2 = vl[ll[i] + 2 + corner];

                            va[0] = x[n1] - x[n0];
                            va[1] = y[n1] - y[n0];
                            va[2] = z[n1] - z[n0];
                            vb[0] = x[n2] - x[n0];
                            vb[1] = y[n2] - y[n0];
                            vb[2] = z[n2] - z[n0];
                            lva = Length(va[0], va[1], va[2]);
                            lvb = Length(vb[0], vb[1], vb[2]);
                            if ((lva == 0.0) || (lvb == 0.0))
                            {
                                break;
                            }
                            else
                            {
                                Norm(va, vb, lva, lvb, nva, nvb);
                                Leng = Length((nva[0] + nvb[0]), (nva[1] + nvb[1]), (nva[2] + nvb[2]));

                                // vectors va and vb are not colinear
                                if (Leng > 0.001 && corner < no_corners - 1)
                                {
                                    corner = corner + 1;
                                    if (normalstyle == BisecLargeAngle)
                                    {
                                        Negat(nva, nvb, f);
                                    }
                                    if (normalstyle == BisecSmallAngle)
                                    {
                                        for (int i = 0; i < 3; i++)
                                        {
                                            f[i] = (nva[i] + nvb[i]);
                                        }
                                    }
                                    if (normalstyle == Orthogonal)
                                    {
                                        Negat(nva, nvb, f);
                                    }

                                    if ((va[0] * va[0] + va[1] * va[1] + va[2] * va[2]) == 0)
                                    {
                                        break;
                                    }
                                    else
                                    {
                                        if (normalstyle == BisecLargeAngle)
                                        {
                                            Negat(nva, nvb, f);
                                            euklf = Length(f[0], f[1], f[2]);
                                            Normale(f, euklf, 0, totalcorners, NU, NV, NW, d);
                                        }
                                        if (normalstyle == BisecSmallAngle)
                                        {
                                            for (int i = 0; i < 3; i++)
                                            {
                                                f[i] = (nva[i] + nvb[i]);
                                            }
                                            euklf = Length(f[0], f[1], f[2]);
                                            Normale(f, euklf, 0, totalcorners, NU, NV, NW, d);
                                        }
                                        if (normalstyle == Orthogonal)
                                        {
                                            float normale[3], euklnormale;
                                            normale[0] = va[1] * vb[2] - va[2] * vb[1];
                                            normale[1] = va[2] * vb[0] - va[0] * vb[2];
                                            normale[2] = va[0] * vb[1] - va[1] * vb[0];

                                            euklnormale = Length(normale[0], normale[1], normale[2]);
                                            Normale(normale, euklnormale, 0, totalcorners, NU, NV, NW, d);
                                        }
                                    }
                                }

                                // vectors va and vb are colinear
                                if (Leng <= 0.001 && corner < no_corners - 1)
                                {
                                    float endx = 0, endy = 0, endz = 0;
                                    float anfangx = 0, anfangy = 0, anfangz = 0;
                                    float euklf, llganz_x = 0, llganz_y = 0, llganz_z = 0, llganz = 0;
                                    float euklkoll, normf[3], a[3];
                                    float va[3], nva[3], nvb[3], lva, lvb = 1;
                                    float u0, u1, u2, llu;
                                    float t;

                                    n0 = vl[ll[i] + corner];
                                    anfangx = x[n0];
                                    anfangy = y[n0];
                                    anfangz = z[n0];
                                    corner = corner + 1;
                                    for (corner = 1; Leng <= 0.001 && corner < no_corners - 1; corner++)
                                    {
                                        Vec(vl, ll, i, corner, x, y, z, va, vb);

                                        lva = Length(va[0], va[1], va[2]);
                                        lvb = Length(vb[0], vb[1], vb[2]);
                                        if ((lva == 0.0) || (lvb == 0.0))
                                        {
                                            break;
                                        }
                                        else
                                        {
                                            Norm(va, vb, lva, lvb, nva, nvb);
                                            Leng = Length((nva[0] + nvb[0]), (nva[1] + nvb[1]), (nva[2] + nvb[2]));
                                        }
                                    }

                                    // normal for the last point
                                    if (corner == no_corners - 1)
                                    {
                                        for (corner = no_corners - 1; corner >= 0; corner--)
                                        {
                                            if (vb[0] >= 0.001 || vb[1] >= 0.001)
                                            {
                                                NU[corner + totalcorners] = -vb[1] / lvb;
                                                NV[corner + totalcorners] = vb[0] / lvb;
                                                NW[corner + totalcorners] = 0.0;
                                            }
                                            else
                                            {
                                                NU[corner + totalcorners] = 0.0;
                                                NV[corner + totalcorners] = -vb[2] / lvb;
                                                NW[corner + totalcorners] = vb[1] / lvb;
                                            }
                                        }
                                        corner = no_corners;
                                    }
                                    else
                                    {
                                        corner = corner - 1;
                                        Vec(vl, ll, i, corner, x, y, z, va, vb);
                                        if (normalstyle == BisecLargeAngle)
                                        {
                                            Negat(nva, nvb, f);
                                        }
                                        if (normalstyle == BisecSmallAngle)
                                        {
                                            for (int i = 0; i < 3; i++)
                                            {
                                                f[i] = (nva[i] + nvb[i]);
                                            }
                                        }
                                        if (normalstyle == Orthogonal)
                                        {
                                            Negat(nva, nvb, f);
                                        }

                                        if ((va[0] * va[0] + va[1] * va[1] + va[2] * va[2]) == 0)
                                        {
                                            break;
                                        }
                                        else
                                        {
                                            if (normalstyle == BisecLargeAngle)
                                            {
                                                Negat(nva, nvb, f);
                                                euklf = Length(f[0], f[1], f[2]);
                                                Normale(f, euklf, corner, totalcorners, NU, NV, NW, d);
                                            }
                                            if (normalstyle == BisecSmallAngle)
                                            {
                                                for (int i = 0; i < 3; i++)
                                                {
                                                    f[i] = (nva[i] + nvb[i]);
                                                }
                                                euklf = Length(f[0], f[1], f[2]);
                                                Normale(f, euklf, corner, totalcorners, NU, NV, NW, d);
                                            }
                                            if (normalstyle == Orthogonal)
                                            {
                                                float normale[3], euklnormale;
                                                normale[0] = va[1] * vb[2] - va[2] * vb[1];
                                                normale[1] = va[2] * vb[0] - va[0] * vb[2];
                                                normale[2] = va[0] * vb[1] - va[1] * vb[0];

                                                euklnormale = Length(normale[0], normale[1], normale[2]);
                                                Normale(normale, euklnormale, corner, totalcorners, NU, NV, NW, d);
                                            }
                                            mu = (va[0] * d[0] + va[1] * d[1] + va[2] * d[2]) / (va[0] * va[0] + va[1] * va[1] + va[2] * va[2]);
                                            p[0] = -mu * va[0] + d[0];
                                            p[1] = -mu * va[1] + d[1];
                                            p[2] = -mu * va[2] + d[2];

                                            euklp = Length(p[0], p[1], p[2]);
                                            Normale(p, euklp, 0, totalcorners, NU, NV, NW, g);
                                        }
                                        corner = corner - 1;
                                        Vec(vl, ll, i, corner, x, y, z, va, vb);
                                        lva = Length(va[0], va[1], va[2]);
                                        lvb = Length(vb[0], vb[1], vb[2]);
                                        if ((lva == 0.0) || (lvb == 0.0))
                                        {
                                            break;
                                        }
                                        else
                                        {
                                            Norm(va, vb, lva, lvb, nva, nvb);
                                            Leng = Length((nva[0] + nvb[0]), (nva[1] + nvb[1]), (nva[2] + nvb[2]));
                                        }

                                        //case:vectors va and vb are colinear
                                        for (corner = corner; Leng <= 0.001 && corner < no_corners - 1; corner++)
                                        {
                                            n = vl[ll[i] + corner];
                                            endx = x[n];
                                            endy = y[n];
                                            endz = z[n];
                                            Vec(vl, ll, i, corner, x, y, z, va, vb);
                                            lva = Length(va[0], va[1], va[2]);
                                            lvb = Length(vb[0], vb[1], vb[2]);
                                            if ((lva == 0.0) || (lvb == 0.0))
                                            {
                                                break;
                                            }
                                            else
                                            {
                                                Norm(va, vb, lva, lvb, nva, nvb);
                                                if (normalstyle == BisecLargeAngle)
                                                {
                                                    Negat(nva, nvb, f);
                                                    euklf = Length(f[0], f[1], f[2]);
                                                    Normale(f, euklf, corner, totalcorners, NU, NV, NW, g);
                                                }
                                                if (normalstyle == BisecSmallAngle)
                                                {
                                                    for (int i = 0; i < 3; i++)
                                                    {
                                                        f[i] = (nva[i] + nvb[i]);
                                                    }
                                                    euklf = Length(f[0], f[1], f[2]);
                                                    Normale(f, euklf, corner, totalcorners, NU, NV, NW, g);
                                                }
                                                if (normalstyle == Orthogonal)
                                                {
                                                    float normale[3], euklnormale;
                                                    normale[0] = va[1] * vb[2] - va[2] * vb[1];
                                                    normale[1] = va[2] * vb[0] - va[0] * vb[2];
                                                    normale[2] = va[0] * vb[1] - va[1] * vb[0];

                                                    euklnormale = Length(normale[0], normale[1], normale[2]);
                                                    Normale(normale, euklnormale, corner, totalcorners, NU, NV, NW, g);
                                                }
                                                llganz_x = endx - anfangx;
                                                llganz_y = endy - anfangy;
                                                llganz_z = endz - anfangz;
                                                llganz = Length(llganz_x, llganz_y, llganz_z);
                                                Leng = Length((nva[0] + nvb[0]), (nva[1] + nvb[1]), (nva[2] + nvb[2]));
                                            }
                                        }
                                        corner = corner - 2;
                                        Vec(vl, ll, i, corner, x, y, z, va, vb);
                                        lva = Length(va[0], va[1], va[2]);
                                        lvb = Length(vb[0], vb[1], vb[2]);
                                        if ((lva == 0.0) || (lvb == 0.0))
                                        {
                                            break;
                                        }
                                        else
                                        {
                                            Norm(va, vb, lva, lvb, nva, nvb);
                                            Leng = Length((nva[0] + nvb[0]), (nva[1] + nvb[1]), (nva[2] + nvb[2]));
                                        }
                                        corner = corner + 1;

                                        for (; Leng <= 0.001 && corner > 1;)
                                        {
                                            n0 = vl[ll[i] + corner - 1];
                                            u0 = endx - x[n0];
                                            u1 = endy - y[n0];
                                            u2 = endz - z[n0];

                                            if (llganz == 0.0)
                                            {
                                                break;
                                            }
                                            else
                                            {

                                                llu = Length(u0, u1, u2);
                                                t = llu / llganz;
                                            }
                                            normf[0] = g[0] + t * (p[0] - g[0]);
                                            normf[1] = g[1] + t * (p[1] - g[1]);
                                            normf[2] = g[2] + t * (p[2] - g[2]);

                                            euklkoll = Length(normf[0], normf[1], normf[2]);
                                            Normale(normf, euklkoll, corner - 1, totalcorners, NU, NV, NW, a);

                                            Vec(vl, ll, i, corner - 1, x, y, z, va, vb);
                                            corner = corner - 1;
                                            lva = Length(va[0], va[1], va[2]);
                                            lvb = Length(vb[0], vb[1], vb[2]);
                                            if ((lva == 0.0) || (lvb == 0.0))
                                            {
                                                break;
                                            }
                                            else
                                            {
                                                Norm(va, vb, lva, lvb, nva, nvb);
                                                Leng = Length((nva[0] + nvb[0]), (nva[1] + nvb[1]), (nva[2] + nvb[2]));
                                            }
                                        }
                                        corner = corner + 1;

                                        for (; Leng <= 0.001 && corner < no_corners - 1; corner++)
                                        {
                                            float va[3], vb[3], nva[3], nvb[3], lva, lvb;
                                            Vec(vl, ll, i, corner, x, y, z, va, vb);
                                            lva = Length(va[0], va[1], va[2]);
                                            lvb = Length(vb[0], vb[1], vb[2]);
                                            if ((lva == 0.0) || (lvb == 0.0))
                                            {
                                                break;
                                            }
                                            else
                                            {
                                                Norm(va, vb, lva, lvb, nva, nvb);
                                                Leng = Length((nva[0] + nvb[0]), (nva[1] + nvb[1]), (nva[2] + nvb[2]));
                                            }
                                        }
                                    }
                                }
                            }
                        } //else
                        corner = corner - 1;

                        //normals for the points 2 to number of corners-1
                        for (corner = corner; corner < no_corners - 1;)
                        {
                            float va[3], vb[3], nva[3], nvb[3], lva, lvb, anfangx, anfangy, anfangz;
                            Vec(vl, ll, i, corner, x, y, z, va, vb);
                            lva = Length(va[0], va[1], va[2]);
                            lvb = Length(vb[0], vb[1], vb[2]);
                            if ((lva == 0.0) || (lvb == 0.0))
                            {
                                break;
                            }
                            else
                            {
                                Norm(va, vb, lva, lvb, nva, nvb);
                                float euklc, Leng, g[3], c[3];
                                Leng = Length((nva[0] + nvb[0]), (nva[1] + nvb[1]), (nva[2] + nvb[2]));

                                //case: vectors va and vb are colinear
                                if (Leng > 0.001 && corner < no_corners - 1)
                                {
                                    n = vl[ll[i] + corner];
                                    if (normalstyle == BisecLargeAngle)
                                    {
                                        Negat(nva, nvb, c);
                                        euklc = Length(c[0], c[1], c[2]);
                                        Normale(c, euklc, corner, totalcorners, NU, NV, NW, g);
                                    }
                                    if (normalstyle == BisecSmallAngle)
                                    {
                                        for (int i = 0; i < 3; i++)
                                        {
                                            c[i] = (nva[i] + nvb[i]);
                                        }
                                        euklc = Length(c[0], c[1], c[2]);
                                        Normale(c, euklc, corner, totalcorners, NU, NV, NW, g);
                                    }
                                    if (normalstyle == Orthogonal)
                                    {
                                        float normale[3], euklnormale;
                                        normale[0] = va[1] * vb[2] - va[2] * vb[1];
                                        normale[1] = va[2] * vb[0] - va[0] * vb[2];
                                        normale[2] = va[0] * vb[1] - va[1] * vb[0];
                                        euklnormale = Length(normale[0], normale[1], normale[2]);
                                        Normale(normale, euklnormale, corner, totalcorners, NU, NV, NW, g);
                                    }
                                    corner = corner + 1;
                                }
                                float endx, endy, endz, t, d[3], llganz;
                                endx = endy = endz = t = d[0] = d[1] = d[2] = llganz = 0;

                                //case:vectors va and vb are colinear
                                if (Leng <= 0.001 && corner < no_corners - 1)
                                {
                                    int n1;
                                    float Leng;
                                    n1 = vl[ll[i] + corner - 1];
                                    anfangx = x[n1];
                                    anfangy = y[n1];
                                    anfangz = z[n1];
                                    Leng = Length((nva[0] + nvb[0]), (nva[1] + nvb[1]), (nva[2] + nvb[2]));

                                    for (corner = corner; Leng <= 0.001 && corner < no_corners - 1; corner++)
                                    {
                                        float va[3], vb[3], nva[3], nvb[3];
                                        float llganz_x, llganz_y, llganz_z, lva, lvb;
                                        int n;
                                        Vec(vl, ll, i, corner, x, y, z, va, vb);
                                        n = vl[ll[i] + corner];
                                        endx = x[n];
                                        endy = y[n];
                                        endz = z[n];

                                        lva = Length(va[0], va[1], va[2]);
                                        lvb = Length(vb[0], vb[1], vb[2]);
                                        if ((lva == 0.0) || (lvb == 0.0))
                                        {
                                            break;
                                        }
                                        else
                                        {
                                            float euklf, f[3];
                                            Norm(va, vb, lva, lvb, nva, nvb);
                                            if (normalstyle == BisecLargeAngle)
                                            {
                                                Negat(nva, nvb, f);
                                                euklf = Length(f[0], f[1], f[2]);
                                                Normale(f, euklf, corner, totalcorners, NU, NV, NW, d);
                                            }
                                            if (normalstyle == BisecSmallAngle)
                                            {
                                                for (int i = 0; i < 3; i++)
                                                {
                                                    f[i] = (nva[i] + nvb[i]);
                                                }
                                                euklf = Length(f[0], f[1], f[2]);
                                                Normale(f, euklf, corner, totalcorners, NU, NV, NW, d);
                                            }
                                            if (normalstyle == Orthogonal)
                                            {
                                                float normale[3], euklnormale;
                                                normale[0] = va[1] * vb[2] - va[2] * vb[1];
                                                normale[1] = va[2] * vb[0] - va[0] * vb[2];
                                                normale[2] = va[0] * vb[1] - va[1] * vb[0];
                                                euklnormale = Length(normale[0], normale[1], normale[2]);
                                                Normale(normale, euklnormale, corner, totalcorners, NU, NV, NW, d);
                                            }

                                            llganz_x = endx - anfangx;
                                            llganz_y = endy - anfangy;
                                            llganz_z = endz - anfangz;
                                            llganz = Length(llganz_x, llganz_y, llganz_z);
                                        }
                                        Leng = Length((nva[0] + nvb[0]), (nva[1] + nvb[1]), (nva[2] + nvb[2]));
                                    }
                                    corner = corner - 2;
                                    Vec(vl, ll, i, corner, x, y, z, va, vb);
                                    lva = Length(va[0], va[1], va[2]);
                                    lvb = Length(vb[0], vb[1], vb[2]);
                                    if ((lva == 0.0) || (lvb == 0.0))
                                    {
                                        break;
                                    }
                                    else
                                    {
                                        Norm(va, vb, lva, lvb, nva, nvb);
                                        Leng = Length((nva[0] + nvb[0]), (nva[1] + nvb[1]), (nva[2] + nvb[2]));
                                    }
                                    corner = corner + 1;

                                    for (; Leng <= 0.001 && corner >= 1 && corner < no_corners - 1;)
                                    {
                                        float nva[3], nvb[3], u0, u1, u2, llu;
                                        n0 = vl[ll[i] + corner];
                                        u0 = endx - x[n0];
                                        u1 = endy - y[n0];
                                        u2 = endz - z[n0];

                                        if (llganz == 0.0)
                                        {
                                            break;
                                        }
                                        else
                                        {
                                            llu = Length(u0, u1, u2);
                                            t = llu / llganz;
                                        }
                                        float euklkoll, normf[3], a[3];
                                        normf[0] = d[0] + t * (g[0] - d[0]);
                                        normf[1] = d[1] + t * (g[1] - d[1]);
                                        normf[2] = d[2] + t * (g[2] - d[2]);
                                        euklkoll = Length(normf[0], normf[1], normf[2]);
                                        Normale(normf, euklkoll, corner, totalcorners, NU, NV, NW, a);
                                        Vec(vl, ll, i, corner - 1, x, y, z, va, vb);
                                        corner = corner - 1;
                                        lva = Length(va[0], va[1], va[2]);
                                        lvb = Length(vb[0], vb[1], vb[2]);
                                        if ((lva == 0.0) || (lvb == 0.0))
                                        {
                                            break;
                                        }
                                        else
                                        {
                                            Norm(va, vb, lva, lvb, nva, nvb);
                                            Leng = Length((nva[0] + nvb[0]), (nva[1] + nvb[1]), (nva[2] + nvb[2]));
                                        }
                                    }
                                    corner = corner + 1;
                                    if (corner < no_corners - 1)
                                    {
                                        Vec(vl, ll, i, corner, x, y, z, va, vb);
                                        lva = Length(va[0], va[1], va[2]);
                                        lvb = Length(vb[0], vb[1], vb[2]);
                                        if ((lva == 0.0) || (lvb == 0.0))
                                        {
                                            break;
                                        }
                                        else
                                        {
                                            Norm(va, vb, lva, lvb, nva, nvb);
                                            Leng = Length((nva[0] + nvb[0]), (nva[1] + nvb[1]), (nva[2] + nvb[2]));
                                        }
                                    }

                                    for (; Leng <= 0.001 && corner < no_corners - 1; corner++)
                                    {
                                        float va[3], vb[3], nva[3], nvb[3], lva, lvb;
                                        Vec(vl, ll, i, corner, x, y, z, va, vb);
                                        lva = Length(va[0], va[1], va[2]);
                                        lvb = Length(vb[0], vb[1], vb[2]);
                                        if ((lva == 0.0) || (lvb == 0.0))
                                        {
                                            break;
                                        }
                                        else
                                        {
                                            Norm(va, vb, lva, lvb, nva, nvb);
                                            Leng = Length((nva[0] + nvb[0]), (nva[1] + nvb[1]), (nva[2] + nvb[2]));
                                        }
                                    }
                                }
                            }
                        }
                        //normal for the last point
                        if (corner == no_corners - 1)
                        {
                            float f[3], va[3], vb[3], d[3], lva, lvb;
                            float euklf, nva[3], nvb[3], Leng;
                            int n0, n1, n2;
                            n0 = vl[ll[i] - 1 + corner];
                            n1 = vl[ll[i] - 2 + corner];
                            n2 = vl[ll[i] + corner];
                            va[0] = x[n1] - x[n0];
                            va[1] = y[n1] - y[n0];
                            va[2] = z[n1] - z[n0];
                            vb[0] = x[n2] - x[n0];
                            vb[1] = y[n2] - y[n0];
                            vb[2] = z[n2] - z[n0];
                            lva = Length(va[0], va[1], va[2]);
                            lvb = Length(vb[0], vb[1], vb[2]);

                            if ((lva == 0.0) || (lvb == 0.0))
                            {
                                break;
                            }
                            else
                            {
                                Norm(va, vb, lva, lvb, nva, nvb);
                                Leng = Length((nva[0] + nvb[0]), (nva[1] + nvb[1]), (nva[2] + nvb[2]));

                                //case:the vectors va and vb from the point no_corners-2 are not colinear
                                if (Leng > 0.001 && corner == no_corners - 1)
                                {
                                    corner = corner - 1;
                                    if (normalstyle == BisecLargeAngle)
                                    {
                                        Negat(nva, nvb, f);
                                    }
                                    if (normalstyle == BisecSmallAngle)
                                    {
                                        for (int i = 0; i < 3; i++)
                                        {
                                            f[i] = (nva[i] + nvb[i]);
                                        }
                                    }
                                    if (normalstyle == Orthogonal)
                                    {
                                        Negat(nva, nvb, f);
                                    }

                                    if ((va[0] * va[0] + va[1] * va[1] + va[2] * va[2]) == 0)
                                    {
                                        break;
                                    }
                                    else
                                    {
                                        if (normalstyle == BisecLargeAngle)
                                        {
                                            Negat(nva, nvb, f);
                                            euklf = Length(f[0], f[1], f[2]);
                                            Normale(f, euklf, no_corners - 1, totalcorners, NU, NV, NW, d);
                                        }
                                        if (normalstyle == BisecSmallAngle)
                                        {
                                            for (int i = 0; i < 3; i++)
                                            {
                                                f[i] = (nva[i] + nvb[i]);
                                            }
                                            euklf = Length(f[0], f[1], f[2]);
                                            Normale(f, euklf, no_corners - 1, totalcorners, NU, NV, NW, d);
                                        }
                                        if (normalstyle == Orthogonal)
                                        {
                                            float normale[3], euklnormale;
                                            normale[0] = va[1] * vb[2] - va[2] * vb[1];
                                            normale[1] = va[2] * vb[0] - va[0] * vb[2];
                                            normale[2] = va[0] * vb[1] - va[1] * vb[0];
                                            euklnormale = Length(normale[0], normale[1], normale[2]);
                                            Normale(normale, euklnormale, no_corners - 1, totalcorners, NU, NV, NW, d);
                                        }
                                    }
                                }
                                //case:the vectors va and vb from the point no_corners-2 are colinear
                                if (Leng <= 0.001 && corner == no_corners - 1)
                                {
                                    float endx = 0, endy = 0, endz = 0;
                                    float anfangx = 0, anfangy = 0, anfangz = 0;
                                    float euklf, llganz_x = 0, llganz_y = 0, llganz_z = 0, llganz = 0;
                                    float euklkoll, normf[3], a[3], mu, euklp, p[3], g[3];
                                    float va[3], nva[3], nvb[3], lva, lvb;
                                    float u0, u1, u2, llu, t;
                                    n0 = vl[ll[i] + corner];
                                    endx = x[n0];
                                    endy = y[n0];
                                    endz = z[n0];

                                    for (corner = no_corners - 1; Leng <= 0.001 && corner > 0; corner--)
                                    {
                                        int n0, n1, n2;
                                        n0 = vl[ll[i] - 1 + corner];
                                        n1 = vl[ll[i] + corner];
                                        n2 = vl[ll[i] - 2 + corner];
                                        va[0] = x[n1] - x[n0];
                                        va[1] = y[n1] - y[n0];
                                        va[2] = z[n1] - z[n0];
                                        vb[0] = x[n2] - x[n0];
                                        vb[1] = y[n2] - y[n0];
                                        vb[2] = z[n2] - z[n0];

                                        lva = Length(va[0], va[1], va[2]);
                                        lvb = Length(vb[0], vb[1], vb[2]);
                                        if ((lva == 0.0) || (lvb == 0.0))
                                        {
                                            break;
                                        }
                                        else
                                        {
                                            Norm(va, vb, lva, lvb, nva, nvb);
                                            Leng = Length((nva[0] + nvb[0]), (nva[1] + nvb[1]), (nva[2] + nvb[2]));
                                        }
                                    }
                                    if (normalstyle == BisecLargeAngle)
                                    {
                                        Negat(nva, nvb, f);
                                    }
                                    if (normalstyle == BisecSmallAngle)
                                    {
                                        for (int i = 0; i < 3; i++)
                                        {
                                            f[i] = (nva[i] + nvb[i]);
                                        }
                                    }
                                    if (normalstyle == Orthogonal)
                                    {
                                        Negat(nva, nvb, f);
                                    }

                                    if ((va[0] * va[0] + va[1] * va[1] + va[2] * va[2]) == 0)
                                    {
                                        break;
                                    }
                                    else
                                    {
                                        if (normalstyle == BisecLargeAngle)
                                        {
                                            Negat(nva, nvb, f);
                                            euklf = Length(f[0], f[1], f[2]);
                                            Normale(f, euklf, corner, totalcorners, NU, NV, NW, d);
                                        }
                                        if (normalstyle == BisecSmallAngle)
                                        {
                                            for (int i = 0; i < 3; i++)
                                            {
                                                f[i] = (nva[i] + nvb[i]);
                                            }
                                            euklf = Length(f[0], f[1], f[2]);
                                            Normale(f, euklf, corner, totalcorners, NU, NV, NW, d);
                                        }
                                        if (normalstyle == Orthogonal)
                                        {
                                            float normale[3], euklnormale;
                                            normale[0] = va[1] * vb[2] - va[2] * vb[1];
                                            normale[1] = va[2] * vb[0] - va[0] * vb[2];
                                            normale[2] = va[0] * vb[1] - va[1] * vb[0];
                                            euklnormale = Length(normale[0], normale[1], normale[2]);
                                            Normale(normale, euklnormale, corner, totalcorners, NU, NV, NW, d);
                                        }

                                        mu = (va[0] * d[0] + va[1] * d[1] + va[2] * d[2]) / (va[0] * va[0] + va[1] * va[1] + va[2] * va[2]);
                                        p[0] = -mu * va[0] + d[0];
                                        p[1] = -mu * va[1] + d[1];
                                        p[2] = -mu * va[2] + d[2];
                                        euklp = Length(p[0], p[1], p[2]);
                                        Normale(p, euklp, no_corners - 1, totalcorners, NU, NV, NW, g);
                                    }
                                    corner = corner + 1;
                                    Vec(vl, ll, i, corner, x, y, z, va, vb);
                                    lva = Length(va[0], va[1], va[2]);
                                    lvb = Length(vb[0], vb[1], vb[2]);
                                    if ((lva == 0.0) || (lvb == 0.0))
                                    {
                                        break;
                                    }
                                    else
                                    {
                                        Norm(va, vb, lva, lvb, nva, nvb);
                                        Leng = Length((nva[0] + nvb[0]), (nva[1] + nvb[1]), (nva[2] + nvb[2]));
                                    }
                                    for (corner = corner; Leng <= 0.001 && corner > 0; corner--)
                                    {
                                        n = vl[ll[i] + corner];
                                        anfangx = x[n];
                                        anfangy = y[n];
                                        anfangz = z[n];
                                        Vec(vl, ll, i, corner, x, y, z, va, vb);
                                        lva = Length(va[0], va[1], va[2]);
                                        lvb = Length(vb[0], vb[1], vb[2]);
                                        if ((lva == 0.0) || (lvb == 0.0))
                                        {
                                            break;
                                        }
                                        else
                                        {
                                            Norm(va, vb, lva, lvb, nva, nvb);
                                            if (normalstyle == BisecLargeAngle)
                                            {
                                                Negat(nva, nvb, f);
                                                euklf = Length(f[0], f[1], f[2]);
                                                Normale(f, euklf, corner, totalcorners, NU, NV, NW, g);
                                            }
                                            if (normalstyle == BisecSmallAngle)
                                            {
                                                for (int i = 0; i < 3; i++)
                                                {
                                                    f[i] = (nva[i] + nvb[i]);
                                                }
                                                euklf = Length(f[0], f[1], f[2]);
                                                Normale(f, euklf, corner, totalcorners, NU, NV, NW, g);
                                            }
                                            if (normalstyle == Orthogonal)
                                            {
                                                float normale[3], euklnormale;
                                                normale[0] = va[1] * vb[2] - va[2] * vb[1];
                                                normale[1] = va[2] * vb[0] - va[0] * vb[2];
                                                normale[2] = va[0] * vb[1] - va[1] * vb[0];
                                                euklnormale = Length(normale[0], normale[1], normale[2]);
                                                Normale(normale, euklnormale, corner, totalcorners, NU, NV, NW, g);
                                            }

                                            llganz_x = endx - anfangx;
                                            llganz_y = endy - anfangy;
                                            llganz_z = endz - anfangz;
                                            llganz = Length(llganz_x, llganz_y, llganz_z);
                                            Leng = Length((nva[0] + nvb[0]), (nva[1] + nvb[1]), (nva[2] + nvb[2]));
                                        }
                                    }
                                    corner = corner + 2;
                                    Vec(vl, ll, i, corner, x, y, z, va, vb);
                                    lva = Length(va[0], va[1], va[2]);
                                    lvb = Length(vb[0], vb[1], vb[2]);
                                    if ((lva == 0.0) || (lvb == 0.0))
                                    {
                                        break;
                                    }
                                    else
                                    {
                                        Norm(va, vb, lva, lvb, nva, nvb);
                                        Leng = Length((nva[0] + nvb[0]), (nva[1] + nvb[1]), (nva[2] + nvb[2]));
                                    }
                                    for (; Leng <= 0.001 && corner < no_corners - 1; corner++)
                                    {
                                        n0 = vl[ll[i] + corner];
                                        u0 = -anfangx + x[n0];
                                        u1 = -anfangy + y[n0];
                                        u2 = -anfangz + z[n0];

                                        if (llganz == 0.0)
                                        {
                                            break;
                                        }
                                        else
                                        {

                                            llu = Length(u0, u1, u2);
                                            t = llu / llganz;
                                        }
                                        normf[0] = g[0] + t * (p[0] - g[0]);
                                        normf[1] = g[1] + t * (p[1] - g[1]);
                                        normf[2] = g[2] + t * (p[2] - g[2]);
                                        euklkoll = Length(normf[0], normf[1], normf[2]);
                                        Normale(normf, euklkoll, corner, totalcorners, NU, NV, NW, a);
                                        Vec(vl, ll, i, corner, x, y, z, va, vb);
                                        lva = Length(va[0], va[1], va[2]);
                                        lvb = Length(vb[0], vb[1], vb[2]);
                                        if ((lva == 0.0) || (lvb == 0.0))
                                        {
                                            break;
                                        }
                                        else
                                        {
                                            Norm(va, vb, lva, lvb, nva, nvb);
                                            Leng = Length((nva[0] + nvb[0]), (nva[1] + nvb[1]), (nva[2] + nvb[2]));
                                        }
                                    }
                                } //hier endet Leng<=0.001
                            }
                        }
                        totalcorners += no_corners;
                    } //else
                }
            }
        }
    }
    else
    {
        Covise::sendError("Sorry, only polygons and lines are supportet at the moment");
        return (NULL);
    }
    // that's it
    return normals;
}

void Vec(int *vl, int *ll, int i, int corner, float *x, float *y, float *z, float *v, float *u)
{
    int n0, n1, n2;
    n0 = vl[ll[i] + corner];
    n1 = vl[ll[i] - 1 + corner];
    n2 = vl[ll[i] + 1 + corner];

    v[0] = x[n1] - x[n0];
    v[1] = y[n1] - y[n0];
    v[2] = z[n1] - z[n0];
    u[0] = x[n2] - x[n0];
    u[1] = y[n2] - y[n0];
    u[2] = z[n2] - z[n0];
}

float Length(float a, float b, float c)
{
    return (sqrt(a * a + b * b + c * c));
}

void Norm(float *v, float *u, float lva, float lvb, float *n, float *f)
{
    for (int i = 0; i < 3; i++)
    {
        n[i] = v[i] / lva;
        f[i] = u[i] / lvb;
    }
}

void Normale(float *c, float euklc, int corner, int totalcorners, float *NU, float *NV, float *NW, float *g)
{
    NU[corner + totalcorners] = g[0] = c[0] / euklc;
    NV[corner + totalcorners] = g[1] = c[1] / euklc;
    NW[corner + totalcorners] = g[2] = c[2] / euklc;
}

void Negat(float *v, float *u, float *l)
{
    for (int i = 0; i < 3; i++)
    {
        l[i] = -(v[i] + u[i]);
    }
}

void Vect(int *vl, int *ll, int i, int corner, float *x, float *y, float *z, float *v, float *u)
{
    int n0, n1, n2;
    n0 = vl[ll[i] - 1 + corner];
    n1 = vl[ll[i] - 2 + corner];
    n2 = vl[ll[i] + corner];
    v[0] = x[n1] - x[n0];
    v[1] = y[n1] - y[n0];
    v[2] = z[n1] - z[n0];
    u[0] = x[n2] - x[n0];
    u[1] = y[n2] - y[n0];
    u[2] = z[n2] - z[n0];
}

MODULE_MAIN(Tools, GenNormals)
