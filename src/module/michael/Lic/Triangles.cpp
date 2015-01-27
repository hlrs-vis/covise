/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Triangles.h"

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

void Triangles::setEdges(i2ten e)
{
    int isize = e.size();
    edges = i2ten(isize);

    for (int i = 0; i < isize; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            (edges[i]).resize(2);
            edges[i][j] = e[i][j];
        }
    }

    return;
}

void Triangles::setCoord(f2ten c)
{
    coord.resize(3);

    for (int i = 0; i < 3; i++)
    {
        (coord[i]).resize(3, 0);
        for (int j = 0; j < 3; j++)
        {
            coord[i][j] = c[i][j];
        }
    }

    return;
}

void Triangles::setVdata(fvec v, int index)
{
    vdata.resize(3);

    for (int i = 0; i < 3; i++)
    {
        (vdata[i]).resize(3, 0);
    }

    for (i = 0; i < 3; i++)
    {
        (vdata[index][i]) = v[i];
    }

    return;
}

void Triangles::setC2d(const f2ten &c)
{
    float temp = 0.0;
    float alpha = 0.0;

    float ab = 0.0;
    float ac = 0.0;
    //float bc = 0.0;

    fvec AB = fvec(3, 0);
    fvec AC = fvec(3, 0);
    //fvec BC = fvec(3, 0);

    c2d.resize(2);
    (c2d[0]).resize(4, 0.0);
    (c2d[1]).resize(4, 0.0);

    AB[0] = coord[0][1] - coord[0][0];
    AB[1] = coord[1][1] - coord[1][0];
    AB[2] = coord[2][1] - coord[2][0];

    AC[0] = coord[0][2] - coord[0][0];
    AC[1] = coord[1][2] - coord[1][0];
    AC[2] = coord[2][2] - coord[2][0];

    //BC[0] = coord[0][2] - coord[0][1];
    //BC[1] = coord[1][2] - coord[1][1];
    //BC[2] = coord[2][2] - coord[2][1];

    ab = abs(AB);
    ac = abs(AC);
    //bc = abs(BC);

    temp = scalar_product(AB, AC);
    temp /= ab;
    temp /= ac; //now temp == cos(aplha)

    alpha = acos(temp);

    c2d[0][1] = ab; //c2d[1][1] = 0;
    c2d[0][2] = ac * temp; //temp = cos(alpha)
    c2d[1][2] = ac * sin(alpha);

    if (c2d[1][2] < 0.0)
    {
        c2d[1][2] *= (-1);
    }
    else
        ;

    /*
      {
         cout << "\nnew triangle:\n" << flush;
         cout << "-------------\n" << flush;

         for(int j = 0; j < 4; j++)
         {
            for(int i = 0; i < 2; i++)
            {
               cout << c2d[i][j] << flush << "  " << flush;
            }
   cout << '\n';
   }
   cout << "\n____________________\n" << flush;
   }
   */
}

void Triangles::shiftC2d(const fvec &shift)
{
    c2d[0][0] += shift[0];
    c2d[1][0] += shift[1];

    c2d[0][1] += shift[0];
    c2d[1][1] += shift[1];

    c2d[0][2] += shift[0];
    c2d[1][2] += shift[1];
}

void Triangles::normaliseC2d()
{
    /////////////////////////////////////////////////////////////////////

    float temp = 0.0;
    float alpha = 0.0;
    float minus_alpha = 0.0; //for clockwise rotations

    float ab = 0.0;
    float ac = 0.0;
    float bc = 0.0;

    fvec AB = fvec(2, 0);
    fvec AC = fvec(2, 0);
    fvec BC = fvec(2, 0);

    f2ten cnew = f2ten(2);
    (cnew[0]).resize(4, 0);
    (cnew[1]).resize(4, 0);

    fvec old = fvec(2, 0);
    ivec edge_order(3, 0);

    /////////////////////////////////////////////////////////////////////

    AB[0] = c2d[0][1] - c2d[0][0];
    AB[1] = c2d[1][1] - c2d[1][0];

    AC[0] = c2d[0][2] - c2d[0][0];
    AC[1] = c2d[1][2] - c2d[1][0];

    BC[0] = c2d[0][2] - c2d[0][1];
    BC[1] = c2d[1][2] - c2d[1][1];

    ab = abs(AB);
    ac = abs(AC);
    bc = abs(BC);

    edge_order = edgeOrder(edge_order, ab, ac, bc);

    cnew = c2d;

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
            c2d[0][3] = 1.0;
            c2d[1][3] = 1.0;
        }
        else if ((edge_order[1] == 3) && (edge_order[2] == 2))
        {
            //cout << "the second is 3\n" << flush;
            //BAC
            c2d = triXflect(c2d, 1, 1);
            c2d[0][3] = 2.0;
            c2d[1][3] = -1.0;
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
        //cout << "\nlongest edge: 2, " << flush;
        old[0] = 1.0;
        old[1] = 0.0;
        temp = scalar_product(old, BC);
        temp /= bc;
        alpha = acos(temp);
        minus_alpha = (-1) * alpha;

        c2d = triXshift(c2d, c2d[0][1]);

        old[0] = c2d[0][0];
        old[1] = c2d[1][0];
        old = rotate(old, minus_alpha);

        c2d[0][0] = old[0];
        c2d[1][0] = old[0];
        c2d[0][1] = 0.0;
        c2d[1][1] = 0.0;
        c2d[0][2] = bc;
        c2d[1][2] = 0.0;

        //AB shortest ?
        if ((edge_order[1] == 3) && (edge_order[2] == 1))
        {
            //cout << "the second is 3\n" << flush;//BAC
            //BCA
            c2d[0][3] = 2.0;
            c2d[1][3] = 1.0;
        }
        else if ((edge_order[1] == 1) && (edge_order[2] == 3))
        {
            //cout << "the second is 1\n" << flush;//BAC
            //CBA
            c2d = triXflect(c2d, 2, 1);
            c2d[0][3] = 3.0;
            c2d[1][3] = -1.0;
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
        //cout << "\nlongest edge: 3, " << flush;
        old[0] = 1.0;
        old[1] = 0.0;
        fvec tmp = fvec(2, 0.0);
        tmp[0] = (-1) * AC[0];
        tmp[1] = (-1) * AC[1]; //tmp = CA wanted!!
        temp = scalar_product(tmp, old);
        temp /= ac;
        alpha = acos(temp);

        c2d = triShift(c2d, AC);
        c2d[0][2] = 0.0;
        c2d[1][2] = 0.0;
        c2d[0][0] = ac;
        c2d[1][0] = 0.0;

        old[0] = c2d[0][1];
        old[1] = c2d[1][1];
        old = rotate(old, alpha);
        c2d[0][1] = old[0];
        c2d[1][1] = old[1];

        //BC shortest ?
        if ((edge_order[1] == 1) && (edge_order[2] == 2))
        {
            //cout << "the second is 1\n" << flush;
            //CAB
            c2d[0][3] = 3.0;
            c2d[1][3] = 1.0;
        }
        else if ((edge_order[1] == 2) && (edge_order[2] == 1))
        {
            //cout << "the second is 2\n" << flush;
            //ACB
            c2d = triXflect(c2d, 3, 1);
            c2d[0][3] = 1.0;
            c2d[1][3] = -1.0;
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
}

float Triangles::getLength()
{
    int j = 0;
    float length = 0.0;

    for (j = 0; j < 3; j++)
    {
        if (fabs(c2d[0][j]) > length)
        {
            length = fabs(c2d[0][j]);
        }
    }

    return length;
}

float Triangles::getHeight()
{
    int j = 0;
    float height = 0.0;

    for (j = 0; j < 3; j++)
    {
        if (fabs(c2d[1][j]) > height)
        {
            height = fabs(c2d[1][j]);
        }
    }

    return height;
}

void Triangles::setNeighbour(int triangle, int neighbour_type)
{
    int size = 0;
    if (neighbours.empty())
    {
        size = 0;
    }
    else
    {
        size = neighbours.size();
    }

    if (size == 0)
    {
        neighbours.resize(1);
        (neighbours[0]).resize(2);
        neighbours[0][0] = triangle;
        neighbours[0][1] = neighbour_type;
        return;
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            if (neighbours[i][0] == triangle)
            {
                return;
            }
            else
            {
            }
        }
        ++size;
        neighbours.resize(size);
        (neighbours[size - 1]).resize(2);
        neighbours[size - 1][0] = triangle;
        neighbours[size - 1][1] = neighbour_type;
        return;
    }
    return;
}

//returns 1, if (0,0) = A
//returns 2, if (0,0) = B
//returns 3, if (0,0) = C
int Triangles::get2dStart()
{
    int start = -1;

    //cout << "\nstart value is: " << flush << c2d[0][3] << flush;

    if (fabs(c2d[0][3] - 1.0) < 0.1)
    {
        start = 1;
    }

    else if (fabs(c2d[0][3] - 2.0) < 0.1)
    {
        start = 2;
    }

    else if (fabs(c2d[0][3] - 3.0) < 0.1)
    {
        start = 3;
    }

    else
    {
    }

    return start;
}

//returns 1 if counter-clockwise
//returns -1 if clockwise
int Triangles::get2dOrientation()
{
    if (c2d[1][3] > 0.1)
    {
        return 1;
    }
    else if (c2d[1][3] < 0.1)
    {
        return -1;
    }
    else
    {
        return 0;
    }
}

int Triangles::getC2dIndex(int which)
{
    int index = -1;
    int start_node = -1;
    int ori = 0;

    if (fabs(c2d[0][3] - 1.0) < 0.1)
    {
        start_node = 0;
    }

    else if (fabs(c2d[0][3] - 2.0) < 0.1)
    {
        start_node = 1;
    }

    else if (fabs(c2d[0][3] - 3.0) < 0.1)
    {
        start_node = 2;
    }

    else
        ;

    index = start_node;

    if (c2d[1][3] > 0.1)
    {
        ori = 1;
    }
    else if (c2d[1][3] < 0.1)
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

//test printout of coordinates
void Triangles::prCoord()
{
    char buffer[256];

    cout << "\n  triangle " << flush << id << flush;
    cout << "\n--------------\n" << flush;

    int j = 0;
    for (; j < 3; j++)
    {
        sprintf(buffer, "%6.3f %6.3f\n", coord[0][j], coord[1][j]);
        cout << buffer << flush;
    }
    cout << '\n' << flush;
    cout << '\n' << flush;
}

//test printout of coordinates
void Triangles::prC2d()
{
    char buffer[256];

    cout << "\ntriangle " << flush << id << flush;
    cout << "\n--------------\n" << flush;

    int j = 0;
    for (; j < 4; j++)
    {
        sprintf(buffer, "%6.3f %6.3f\n", c2d[0][j], c2d[1][j]);
        cout << buffer << flush;
    }
    cout << '\n' << flush;
    cout << '\n' << flush;
}

Triangles::Triangles()
{
    id = 0;
    packed = false;

    nodes = ivec();
    edges = i2ten();
    coord = f2ten();
    neighbours = i2ten();
    vdata = f2ten();
    c2d = f2ten();
}

/*******************************\ 
 * find neighbouring triangles *
\*******************************/

bool commonEdge(Triangles *first, Triangles *second)
{
    bool hce = false;

    i2ten e1 = i2ten(3);
    i2ten e2 = i2ten(3);
    i2ten e3 = i2ten(3);

    int i = 0;
    while (i < 3)
    {
        (e1[i]).resize(2);
        (e2[i]).resize(2);
        (e3[i]).resize(2);
        ++i;
    }
    i = 0;

    i2tCopy(e1, (*first).getEdges());
    i2tCopy(e2, (*second).getEdges());

    while (i < 3)
    {
        e3[i][0] = e2[i][1];
        e3[i][1] = e2[i][0];
        ++i;
    }
    i = 0;

    while (i < 3)
    {
        if ((e1[0] == e2[i]) || (e1[0] == e3[i]))
        {
            hce = true;
            return hce;
        }
        else if ((e1[1] == e2[i]) || (e1[1] == e3[i]))
        {
            hce = true;
            return hce;
        }
        else if ((e1[2] == e2[i]) || (e1[2] == e3[i]))
        {
            hce = true;
            return hce;
        }
        else
        {
        }
        ++i;
    }
    i = 0;

    return hce;
}

bool commonNode(Triangles *first, Triangles *second)
{
    bool hcn = false;

    ivec n1 = ivec(3);
    ivec n2 = ivec(3);

    ivecCopy(n1, (*first).getNodes());
    ivecCopy(n2, (*second).getNodes());

    int i = 0;
    while (i < 3)
    {
        if (n1[0] == n2[i])
        {
            hcn = true;
            return hcn;
        }
        else if (n1[1] == n2[i])
        {
            hcn = true;
            return hcn;
        }
        else if (n1[2] == n2[i])
        {
            hcn = true;
            return hcn;
        }
        else
        {
        }
        ++i;
    }
    i = 0;

    return hcn;
}

//l corresponds to vector with 3 elements
void lsort(float *c, float *a, float *b, int *l)
{
    int tmp = 0;
    float arg1 = 0.0;
    float arg2 = 0.0;

    //initialise
    *l = 0;
    *(l + 1) = 1;
    *(l + 2) = 2;

    if (c < a)
    {
        tmp = *l;
        *l = *(l + 1);
        *(l + 1) = tmp;

        arg1 = *c;
        arg2 = *a;
        *c = FMAX(arg1, arg2);
        *a = FMIN(arg1, arg2);
    }
    else
        ;

    if (c < b)
    {
        tmp = *l;
        *l = *(l + 2);
        *(l + 2) = tmp;

        arg1 = *c;
        arg2 = *b;
        *c = FMAX(arg1, arg2);
        *b = FMIN(arg1, arg2);
    }
    else
        ;

    if (a < b)
    {
        tmp = *(l + 1);
        *(l + 1) = *(l + 2);
        *(l + 2) = tmp;

        arg1 = *a;
        arg2 = *b;
        *a = FMAX(arg1, arg2);
        *b = FMIN(arg1, arg2);
    }
    else
        ;
}

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

//rotate an euclidean vector counter-clockwise
fvec rotate(fvec a, float alpha)
{
    fvec tmp = fvec(2, 0.0);

    tmp[0] = cos(alpha) * a[0] - sin(alpha) * a[1];
    tmp[1] = sin(alpha) * a[0] + cos(alpha) * a[1];

    return tmp;
}

f2ten triXflect(f2ten coord, int start, int orientation)
{
    float tmp1 = 0.0;

    int first = 0;
    int second = 0;
    int third = 0;

    start -= 1; //start now 0, 1, 2
    orientation *= (-1); //-1 -> +1 & vice versa
    orientation += 1; //orientation now 0, 2
    orientation /= 2; //orientation now 0, 1
    orientation += 1; //orientation now 1, 2

    first = start;
    second = (first + orientation) % 3;
    third = (second + orientation) % 3;

    tmp1 = coord[0][first];
    coord[0][first] = coord[0][second];
    coord[0][second] = tmp1;

    //tmp1 = coord[0][first] + coord[0][second];
    //tmp1 /= 2.0;
    //coord[0][third] = 2*tmp1 - coord[0][third];
    tmp1 = coord[0][first] + coord[0][second];
    tmp1 -= coord[0][third];
    coord[0][third] = tmp1;

    return coord;
}

/*
f2ten triYflect(f2ten coord, int start, int orientation)
{
   float tmp1 = 0.0;

   f2ten ctri = f2ten(2);
   ctri[0] = fvec(3, 0.0);
   ctri[1] = fvec(3, 0.0);
   ctri[2] = fvec(3, 0.0);

   ctri = coord;

int first = 0;
int second = 0;
int third = 0;

start -= 1;  //start now 0, 1, 2
orientation *= (-1);  //-1 -> +1 & vice versa
orientation += 1;  //orientation now 0, 2
orientation /= 2;  //orientation now 0, 1
orientation += 1;  //orientation now 1, 2

first = start;
second = (first+orientation)%3;
third = (second+orientation)%3;

tmp1 = coord[1][first] + coord[1][second];
tmp1 /= 2.0;

float tmp2 = coord[1][third] + tmp1;

ctri[1][third] = tmp2 - coord[1][third];
ctri[1][first] = tmp2 - coord[1][first];
ctri[1][second] = tmp2 - coord[1][first];

return ctri;
}
*/

f2ten triXshift(f2ten coord, float val)
{
    coord[0][0] -= val;
    coord[0][1] -= val;
    coord[0][2] -= val;

    return coord;
}

f2ten triShift(f2ten coord, fvec val)
{
    coord[0][0] -= val[0];
    coord[0][1] -= val[0];
    coord[0][2] -= val[0];

    coord[1][0] -= val[1];
    coord[1][1] -= val[1];
    coord[1][2] -= val[1];

    return coord;
}

//provides the index of the highest triangle of a set
//criterion are the normalised 2D-euclidean coordinates
int getHighest(trivec &triangles, ivec &index, IntList &list, bool sorted)
{
    /* 
      int i = 0;
      int ind = 0;

      for(i = 0 ; i < index.size(); i++)
      {
         ind = index[i];
         if( sorted == true )
         {
            if( (triangles[ind]).getPacked() == true )
            {
   continue;
   }
   else
   {
   break;
   }
   }
   else
   {
   cout << "\n... i don't like to run after special\n\ 
   elements in non-sorted lists ...\n" << flush;
   }
   }
   */
    int j = list.getStart();
    int el = list.getElement(j);
    //cout << "\nlist index  : " << flush << j << flush;
    //cout << "\nlist element: " << flush << el << flush;
    //cout << '\n' << flush;
    list.remove(j);

    return el;
}
