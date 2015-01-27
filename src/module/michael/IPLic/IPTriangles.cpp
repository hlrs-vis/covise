/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "IPTriangles.h"

void Triangles::setNodes(ivec n)
{
    int size = n.size();
    nodes = ivec(size);

    for (int i = 0; i < size; i++)
    {
        nodes[i] = n[i];
    }

    return;
}

void Triangles::setCoord(f2ten c)
{
    coord.resize(3);

    for (int i = 0; i < 3; i++)
    {
        (coord[i]).resize(4, 0.0);
        for (int j = 0; j < 3; j++)
        {
            coord[i][j] = c[i][j];
        }
    }

    /////////////////////////////////////////////////////////////////////

    float ab = 0.0;
    float ac = 0.0;
    float bc = 0.0;

    fvec AB = fvec(3, 0);
    fvec AC = fvec(3, 0);
    fvec BC = fvec(3, 0);

    /////////////////////////////////////////////////////////////////////

    AB[0] = coord[0][1] - coord[0][0];
    AB[1] = coord[1][1] - coord[1][0];
    AB[2] = coord[2][1] - coord[2][0];

    AC[0] = coord[0][2] - coord[0][0];
    AC[1] = coord[1][2] - coord[1][0];
    AC[2] = coord[2][2] - coord[2][0];

    BC[0] = coord[0][2] - coord[0][1];
    BC[1] = coord[1][2] - coord[1][1];
    BC[2] = coord[2][2] - coord[2][1];

    ab = abs(AB);
    ac = abs(AC);
    bc = abs(BC);

    /////////////////////////////////////////////////////////////////////

    ivec edge_order(3, 0);
    edge_order = edgeOrder(edge_order, ab, ac, bc);

    /////////////////////////////////////////////////////////////////////

    //AB is the longest edge, e.g. edge_order[0]=1 !
    if (edge_order[0] == 1)
    {
        //cout << "\nlongest edge: 1, " << flush;
        //AC shortest ?
        if ((edge_order[1] == 2) && (edge_order[2] == 3))
        {
            //cout << "the second is 2\n" << flush;
            //ABC
            coord[0][3] = 1.0;
            coord[1][3] = 1.0;
        }
        else if ((edge_order[1] == 3) && (edge_order[2] == 2))
        {
            //cout << "the second is 3\n" << flush;
            //BAC
            coord[0][3] = 2.0;
            coord[1][3] = -1.0;
        }
        else
        {
            cout << "\n... unexpected situation occured in" << flush;
            cout << "\nsetting edge order of triangles ...\n" << flush;
        }
    }

    //BC ist the longest edge
    else if (edge_order[0] == 2)
    {
        //AB shortest ?
        if ((edge_order[1] == 3) && (edge_order[2] == 1))
        {
            //cout << "the second is 3\n" << flush;//BAC
            //BCA
            coord[0][3] = 2.0;
            coord[1][3] = 1.0;
        }
        else if ((edge_order[1] == 1) && (edge_order[2] == 3))
        {
            //cout << "the second is 1\n" << flush;//BAC
            //CBA
            coord[0][3] = 3.0;
            coord[1][3] = -1.0;
        }
        else
        {
            cout << "\n... unexpected situation occured in" << flush;
            cout << "\nsetting edge order of triangles ...\n" << flush;
        }
    }

    //AC or CA ist the longest edge
    else if (edge_order[0] == 3)
    {
        //BC shortest ?
        if ((edge_order[1] == 1) && (edge_order[2] == 2))
        {
            //cout << "the second is 1\n" << flush;
            //CAB
            coord[0][3] = 3.0;
            coord[1][3] = 1.0;
        }
        else if ((edge_order[1] == 2) && (edge_order[2] == 1))
        {
            //cout << "the second is 2\n" << flush;
            //ACB
            coord[0][3] = 1.0;
            coord[1][3] = -1.0;
        }
        else
        {
            cout << "\n... unexpected situation occured in" << flush;
            cout << "\nsetting edge order of triangles ...\n" << flush;
        }
    }
    else
    {
        cout << "\n... unexpected situation occured in" << flush;
        cout << "\nsetting edge order of triangles ...\n" << flush;
    }

    /////////////////////////////////////////////////////////////////////

    return;
}

//v contains the complete vector data
//index[0] contains the index of "A", ... , index[2] the index of "C"
void Triangles::setVdata(const f2ten &v, ivec index)
{
    /////////////////////////////////////////////////////////////////////

    vdata.resize(2);

    int i;
    for (i = 0; i < 2; i++)
    {
        (vdata[i]).resize(3, 0.0);
    }

    /////////////////////////////////////////////////////////////////////

    //for 2*2 normal equations: matrix = E'*E;
    //E = (e1, e2), E' transposed matrix
    f2ten matrix = f2ten(2);
    (matrix[0]).resize(2, 0.0);
    (matrix[1]).resize(2, 0.0);

    /////////////////////////////////////////////////////////////////////

    //vectors -> transposed v-matrix
    f2ten vt = f2ten(3);
    (vt[0]).resize(2, 0.0);
    (vt[1]).resize(2, 0.0);
    (vt[2]).resize(2, 0.0);

    fvec e1 = fvec(3, 0.0);
    fvec e2 = fvec(3, 0.0);
    float le1 = 0.0;
    float le2 = 0.0;

    e1[0] = coord[0][1] - coord[0][0];
    e1[1] = coord[1][1] - coord[1][0];
    e1[2] = coord[2][1] - coord[2][0];
    le1 = scalar_product(e1, e1);
    //le1 = sqrt(le1);

    e2[0] = coord[0][2] - coord[0][0];
    e2[1] = coord[1][2] - coord[1][0];
    e2[2] = coord[2][2] - coord[2][0];
    le2 = scalar_product(e2, e2);
    //le2 = sqrt(le2);

    /////////////////////////////////////////////////////////////////////

    //initial matrix
    if ((fabs(le1) > FLT_MIN) && (fabs(le2) > FLT_MIN))
    {
        matrix[0][0] = 1.0;
        matrix[0][1] = scalar_product(e1, e2);
        matrix[0][1] /= le1;

        matrix[1][1] = 1.0;
        matrix[1][0] = scalar_product(e2, e1);
        matrix[0][1] /= le2;

        //////////////////////////////////////////////////////////////////

        vt = createVt(v, index, e1, le1, e2, le2);

        fvec tmp(2, 0.0);

        {
            int i;
            for (i = 0; i < 3; i++)
            {
                tmp = solveNormalEq(matrix, le1, le2, vt[i]);
                //tmp = vt[i];
                vdata[0][i] = tmp[0];
                vdata[1][i] = tmp[1];
            }
        }
    }
    else
        ;

    /////////////////////////////////////////////////////////////////////

    return;
}

void Triangles::setVdata(fvec v, int ii)
{
    vdata[0][ii] = v[0];
    vdata[1][ii] = v[1];
    vdata[2][ii] = v[2];
}

void Triangles::setVTex(const f2ten &v)
{
    vTex.resize(2);

    int i;
    int j;
    for (i = 0; i < 2; i++)
    {
        (vTex[i]).resize(3, 0.0);
    }

    for (i = 0; i < 2; i++)
    {
        for (j = 0; j < 3; j++)
        {
            vTex[i][j] = v[i][j];
        }
    }

    return;
}

//nonsense !!!  why ???
void Triangles::setVTex()
{
    vTex.resize(2);
    (vTex[0]).resize(3, 0.0);
    (vTex[1]).resize(3, 0.0);

    fvec e1 = fvec(2, 0.0);
    e1[0] = c2dTex[0][1] - c2dTex[0][0];
    e1[1] = c2dTex[1][1] - c2dTex[1][0];

    fvec e2 = fvec(2, 0.0);
    e2[0] = c2dTex[0][2] - c2dTex[0][0];
    e2[1] = c2dTex[1][2] - c2dTex[1][0];

    fvec tmp = fvec(2, 0.0);

    int i;
    for (i = 0; i < 3; i++)
    {
        tmp = vdata[0][i] * e1 + vdata[1][i] * e2;
        vTex[0][i] = tmp[0];
        vTex[1][i] = tmp[1];
    }

    return;
}

void Triangles::initC2dTex()
{
    c2dTex.resize(2);
    (c2dTex[0]).resize(3, 0.0);
    (c2dTex[1]).resize(3, 0.0);
}

fvec Triangles::getCoord(int j)
{
    fvec c = fvec(3, 0.0);

    c[0] = coord[0][j];
    c[1] = coord[1][j];
    c[2] = coord[2][j];

    return c;
}

/*
f2ten Triangles::getVdata()
{
   f2ten v = f2ten(3);
   (v[0]).resize(3, 0.0);
   (v[1]).resize(3, 0.0);
   (v[2]).resize(3, 0.0);

   fvec e1 = fvec(3, 0.0);
   fvec e2 = fvec(3, 0.0);

/////////////////////////////////////////////////////////////////////

e1[0] = coord[0][1] - coord[0][0];
e1[1] = coord[1][1] - coord[1][0];
e1[2] = coord[2][1] - coord[2][0];

e2[0] = coord[0][2] - coord[0][0];
e2[1] = coord[1][2] - coord[1][0];
e2[2] = coord[2][2] - coord[2][0];

v[0][0] = vdata[0][0]*e1[0] + vdata[1][0]*e2[0];
v[1][0] = vdata[0][0]*e1[1] + vdata[1][0]*e2[1];
v[2][0] = vdata[0][0]*e1[2] + vdata[1][0]*e2[2];

v[0][1] = vdata[0][1]*e1[0] + vdata[1][1]*e2[0];
v[1][1] = vdata[0][1]*e1[1] + vdata[1][1]*e2[1];
v[2][1] = vdata[0][1]*e1[2] + vdata[1][1]*e2[2];

v[0][2] = vdata[0][2]*e1[0] + vdata[1][2]*e2[0];
v[1][2] = vdata[0][2]*e1[1] + vdata[1][2]*e2[1];
v[2][2] = vdata[0][2]*e1[2] + vdata[1][2]*e2[2];

/////////////////////////////////////////////////////////////////////

return v;
}
*/

fvec Triangles::getVdata(int j)
{
    fvec v = fvec(3, 0.0);

    fvec e1 = fvec(3, 0.0);
    fvec e2 = fvec(3, 0.0);

    /////////////////////////////////////////////////////////////////////

    e1[0] = coord[0][1] - coord[0][0];
    e1[1] = coord[1][1] - coord[1][0];
    e1[2] = coord[2][1] - coord[2][0];

    e2[0] = coord[0][2] - coord[0][0];
    e2[1] = coord[1][2] - coord[1][0];
    e2[2] = coord[2][2] - coord[2][0];

    v[0] = vdata[0][j] * e1[0] + vdata[1][j] * e2[0];
    v[1] = vdata[0][j] * e1[1] + vdata[1][j] * e2[1];
    v[2] = vdata[0][j] * e1[2] + vdata[1][j] * e2[2];

    /////////////////////////////////////////////////////////////////////

    return v;
}

fvec Triangles::getVTex(int j)
{
    fvec v = fvec(2, 0.0);

    v[0] = vTex[0][j];
    v[1] = vTex[1][j];

    return v;
}

fvec Triangles::getC2dTex(int j)
{
    fvec v = fvec(2, 0.0);

    v[0] = c2dTex[0][j];
    v[1] = c2dTex[1][j];

    return v;
}

int Triangles::getCoordIndex(int which)
{
    int index = -1;
    int start_node = -1;
    int ori = 0;

    if (fabs(coord[0][3] - 1.0) < 0.1)
    {
        start_node = 0;
    }

    else if (fabs(coord[0][3] - 2.0) < 0.1)
    {
        start_node = 1;
    }

    else if (fabs(coord[0][3] - 3.0) < 0.1)
    {
        start_node = 2;
    }

    else
        ;

    index = start_node;

    if (coord[1][3] > 0.1)
    {
        ori = 1;
    }
    else if (coord[1][3] < 0.1)
    {
        ori = 2;
    }
    else
        ;

    if (which == 1)
    {
        index = start_node;
    }
    else if (which == 2)
    {
        index = start_node;
        index += ori;
        index %= 3;
    }
    else if (which == 3)
    {
        index = start_node;
        index += ori;
        index %= 3;

        index += ori;
        index %= 3;
    }
    else
        ;

    return index;
}

Triangles::Triangles()
{
    nodes = ivec();
    coord = f2ten();
    vdata = f2ten();
    vTex = f2ten();
    c2dTex = f2ten();
    //trafo = f2ten();
}

////////////////////////////////////////////////////////////////////////

f2ten createVt(const f2ten &v, const ivec &index,
               const fvec &e1, int le1,
               const fvec &e2, int le2)
{
    f2ten rval = f2ten(3);
    (rval[0]).resize(2, 0.0);
    (rval[1]).resize(2, 0.0);
    (rval[2]).resize(2, 0.0);

    rval[0][0] = v[0][index[0]] * e1[0] + v[1][index[0]] * e2[0];
    rval[0][1] = v[0][index[0]] * e1[1] + v[1][index[0]] * e2[1];

    rval[1][0] = v[0][index[1]] * e1[0] + v[1][index[1]] * e2[0];
    rval[1][1] = v[0][index[1]] * e1[1] + v[1][index[1]] * e2[1];

    rval[2][0] = v[0][index[2]] * e1[0] + v[1][index[2]] * e2[0];
    rval[2][1] = v[0][index[2]] * e1[1] + v[1][index[2]] * e2[1];

    return rval;
}

fvec solveNormalEq(const f2ten &matrix, int le1, int le2,
                   const fvec &v)
{
    fvec rval = fvec(2, 0.0);

    if (!((matrix.size() == 2) && ((matrix[0]).size() == 2) && ((matrix[1]).size() == 2)))
    {
        return rval;
    }
    else
        ;

    const float det = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1];

    rval[0] = v[0] * matrix[1][1] - v[1] * matrix[0][1];
    rval[0] /= det;

    rval[1] = v[1] * matrix[0][0] - v[0] * matrix[1][0];
    rval[1] /= det;

    return rval;
}

////////////////////////////////////////////////////////////////////////

/*
//l corresponds to vector with 3 elements
void lsort(float* c, float* a, float* b, int* l)
{
   int tmp = 0;
   float arg1 = 0.0;
   float arg2 = 0.0;

   //initialise
   *l = 0;
   *(l+1) = 1;
*(l+2) = 2;

if(c < a)
{
tmp = *l;
*l = *(l+1);
*(l+1) = tmp;

arg1 = *c;
arg2 = *a;
*c = FMAX(arg1, arg2);
*a = FMIN(arg1, arg2);
}
else;

if(c < b)
{
tmp = *l;
*l = *(l+2);
*(l+2) = tmp;

arg1 = *c;
arg2 = *b;
*c = FMAX(arg1, arg2);
*b = FMIN(arg1, arg2);
}
else;

if(a < b)
{
tmp = *(l+1);
*(l+1) = *(l+2);
*(l+2) = tmp;

arg1 = *a;
arg2 = *b;
*a = FMAX(arg1, arg2);
*b = FMIN(arg1, arg2);
}
else;
}
*/

ivec edgeOrder(ivec edge_order, float ab, float ac, float bc)
{
    //set edge order with respect to the edge lengthes
    //ab := 1, bc := 2, ca = ac := 3;
    //edge_order[0] := the longest edge, ...
    {
        if ((ab >= bc) && (ab >= ac) && (bc >= ac))
        {
            edge_order[0] = 1;
            edge_order[1] = 2;
            edge_order[2] = 3;
        }
        else if ((ab >= bc) && (ab >= ac) && (bc < ac))
        {
            edge_order[0] = 1;
            edge_order[1] = 3;
            edge_order[2] = 2;
        }
        else if ((ab < bc) && (ab >= ac) && (bc >= ac))
        {
            edge_order[0] = 2;
            edge_order[1] = 1;
            edge_order[2] = 3;
        }
        else if ((ab < bc) && (ab < ac) && (bc >= ac))
        {
            edge_order[0] = 2;
            edge_order[1] = 3;
            edge_order[2] = 1;
        }
        else if ((ab >= bc) && (ab < ac) && (bc < ac))
        {
            edge_order[0] = 3;
            edge_order[1] = 1;
            edge_order[2] = 2;
        }
        else if ((ab < bc) && (ab < ac) && (bc < ac))
        {
            edge_order[0] = 3;
            edge_order[1] = 2;
            edge_order[2] = 1;
        }
        else
        {
            cout << "\n... unexpected situation occured in" << flush;
            cout << "\nsetting edge order of triangles ...\n" << flush;
        }
    }

    return edge_order;
}

//returns 2*area( triangle(a, b ,c) )
float doubleArea(const fvec &a, const fvec &b, const fvec &c)
{
    int asize = a.size();
    int bsize = b.size();
    int csize = c.size();

    float da = 0;

    if ((asize == 3) && (asize == bsize) && (asize == csize))
    {
        float cyclic = (a[0] * b[1] * c[2] + a[1] * b[2] * c[0] + a[2] * b[0] * c[1]);
        float anti_cyclic = (a[0] * b[2] * c[1] + a[1] * b[0] * c[2] + a[2] * b[1] * c[0]);

        da = (cyclic - anti_cyclic);
    }
    else if ((asize == 2) && (asize == bsize) && (asize == csize))
    {
        da = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    }
    else
        ;

    return da;
}

//returns area( triangle(a, b ,c) )
float area(const fvec &a, const fvec &b, const fvec &c)
{
    float area = doubleArea(a, b, c);
    area /= 2.0;

    return area;
}

/*
//rotate an euclidean vector counter-clockwise
fvec rotate(fvec a, float alpha)
{
   fvec tmp = fvec(2, 0.0);

   tmp[0] = cos(alpha)*a[0] - sin(alpha)*a[1];
   tmp[1] = sin(alpha)*a[0] + cos(alpha)*a[1];

   return tmp;
}
*/
