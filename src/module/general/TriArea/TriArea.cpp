/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Calculate triangle area or ratio of edge lengths          **
 **                                                                        **
 **                                                                        **
\**************************************************************************/
#include <util/coviseCompat.h>
#include "TriArea.h"
#include <iostream>
#include <do/coDoData.h>

void Negat(float *v, float *u, float *l);
void Normale(float *c, float euklc, int corner, int totalcorners, float *NU, float *NV, float *NW, float *g);
void Norm(float *v, float *u, float lva, float lvb, float *n, float *f);
float Length(float a, float b, float c);
void Vec(int *vl, int *ll, int i, int corner, float *x, float *y, float *z, float *v, float *u);
float vnorm(float* v);
void xprod(float* a, float* b, float* c);

TriArea::TriArea(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Work out normals for lines")
{
    //select normaltype
    p_normalstyle = addChoiceParam("LineNormalStyle", "Which Normal Style For Lines");
    const char *choLabels[] = { "BisectLargeAngle", "BisectSmallAngle", "Orthogonal" };

    p_normalstyle->setValue(3, choLabels, 0);
    // Ports
    p_inPort = addInputPort("GridIn0", "Polygons|Lines", "Grid, polygonal or linear input");
    p_outPort = addOutputPort("DataOut0", "Float", "TriArea");
}

int TriArea::compute(const char *)
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

    coDoFloat *out_obj = (coDoFloat *)
        tri_area(in_obj, p_outPort->getObjName());
    p_outPort->setCurrentObject(out_obj);

    normalstyle = (NormalSelectMap)p_normalstyle->getValue();

    // bye
    return SUCCESS;
}

////// workin' routines
coDistributedObject *TriArea::tri_area(const coDistributedObject *mesh_in,
                                             const char *out_name)
{
    const coDoLines *lines;
    const coDoPolygons *polygons;
    coDoFloat *triArea = NULL;
    float *x, *y, *z, *TA, *NTA, *NU, *NV, *NW, *U, *V, *W, *F_Normals_U, *F_Normals_V, *F_Normals_W, *F_TA;
    int *vl, *pl;
    int num_n, *nl, *nli, numpoly, numlines, numcoord;
    int n0, n1, n2, n, i;
    float v[3][3], xn[3],yn[3],zn[3], xpr[3], x0, x1, x2, y0, y1, y2, z0, z1, z2, l[3], l2; //,ang;
    const char *ptype = mesh_in->getType();

    if (strcmp(ptype, "POLYGN") == 0 )
    {
        if (strcmp(ptype, "POLYGN") == 0)
        {
            polygons = (const coDoPolygons *)mesh_in;
            numpoly = polygons->getNumPolygons();
            int numvert = polygons->getNumVertices();
            numcoord = polygons->getNumPoints();
            polygons->getAddresses(&x, &y, &z, &vl, &pl);
            polygons->getNeighborList(&num_n, &nl, &nli);
            triArea = new coDoFloat(out_name, numpoly);
            triArea->getAddress(&TA);

            //TA= F_TA = new float[numpoly];
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
                l2 = 0.0;
                for (triangle = 0; triangle < no_corners - 2; ++triangle)
                {
                    n0 = vl[pl[i] + triangle];
                    n1 = vl[pl[i] + 1 + triangle];
                    n2 = vl[pl[i] + 2 + triangle];
					v[0][0] = x[n1] - x[n0];
                    v[0][1] = y[n1] - y[n0];
                    v[0][2] = z[n1] - z[n0];
                    v[1][0] = x[n2] - x[n0];
                    v[1][1] = y[n2] - y[n0];
                    v[1][2] = z[n2] - z[n0];
                    v[2][0] = x[n2] - x[n1];
                    v[2][1] = y[n2] - y[n1];
                    v[2][2] = z[n2] - z[n1];
					for (int j=0;j<3;j++) {
						l[j] = vnorm(v[j]);
					}
					xprod(v[0],v[1],xpr);
                    l2 = 0.5 * vnorm(xpr);
                    *TA = l2 ;
					if (l2 != 0.0)
					{
						break;
					}
                }
				TA++;
            }
/*
            for (i = 0; i < numcoord; i++)
            {
                //*NU = *NV = *NW = 0;
                *NTA =0.0;
                for (n = nli[i]; n < nli[i + 1]; n++)
                {
                   // *NU += F_Normals_U[nl[n]];
                   // *NV += F_Normals_V[nl[n]];
                   // *NW += F_Normals_W[nl[n]];
                   *NTA = F_TA[nl[n]];
                }
                float l = 0.0;//sqrt(*NU * *NU + *NV * *NV + *NW * *NW);
                if (l > 0.0)
                {
                    // *NU /= l;
                    // *NV /= l;
                    // *NW /= l;
                }
                //NU++;
                //NV++;
                //NW++;
                NTA++;
            }*/
            delete[] F_Normals_U;
            delete[] F_Normals_V;
            delete[] F_Normals_W;
			//delete[] F_TA;
        }
    }
    else
    {
        Covise::sendError("Sorry, only polygons are supportet at the moment");
        return (NULL);
    }
    // that's it
    return triArea;
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

float vnorm(float* v)
{
    float x;
    x = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    return x;
}

void xprod(float* a, float* b, float* c)
{
    c[0] = a[1]*b[2]-a[2]*b[1];
    c[1] = a[2]*b[0]-a[0]*b[2];
    c[2] = a[0]*b[1]-a[1]*b[0];
}

MODULE_MAIN(Tools, TriArea)
