/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Geometry.h"

////////////////////////////////////////////////////////////////////////

void Patch::appendRow()
{
    ++num_rows;

    rows.resize(num_rows);
    rows[num_rows - 1] = ivec();

    (xbounds[0]).resize(num_rows);
    (xbounds[1]).resize(num_rows);
    (ybounds[0]).resize(num_rows);
    (ybounds[1]).resize(num_rows);

    if (num_rows == 1)
    {
        xbounds[0][0] = 0.0;
        xbounds[1][0] = 0.0;
        ybounds[0][0] = 0.0;
        ybounds[1][0] = 0.0;
    }
    else if (num_rows > 1)
    {
        xbounds[0][num_rows - 1] = 0.0;
        xbounds[1][num_rows - 1] = 0.0;
        ybounds[0][num_rows - 1] = ybounds[1][num_rows - 2];
        ybounds[1][num_rows - 1] = ybounds[1][num_rows - 2];
    }
    else
        ;
}

void Patch::insertTriangle(int row_index, int triangle_index,
                           Triangles *triangle, f2ten bdry)
{
    const int X = 0;
    const int Y = 1;
    const int LOWER = 0;
    const int UPPER = 1;
    const int first = triangle->getC2dIndex(1);
    const int second = triangle->getC2dIndex(2);
    const int third = triangle->getC2dIndex(3);

    (rows[row_index]).push_back(triangle_index);
    triangle->setPacked(true);

    xbounds[LOWER][row_index] = bdry[X][LOWER];
    xbounds[UPPER][row_index] = bdry[X][UPPER];
    ybounds[LOWER][row_index] = bdry[Y][LOWER];
    ybounds[UPPER][row_index] = bdry[Y][UPPER];

    //cout << "\nrow_index: " << flush << row_index << flush;
    //triangle->prC2d();
}

fvec Patch::getXbounds(int row_index)
{
    fvec temp(2, 0);
    temp[0] = xbounds[0][row_index];
    temp[1] = xbounds[1][row_index];

    return temp;
}

fvec Patch::getYbounds(int row_index)
{
    fvec temp(2, 0);
    temp[0] = ybounds[0][row_index];
    temp[1] = ybounds[1][row_index];

    return temp;
}

void Patch::pack(trivec &triangles)
{
    //char buffer[256];
    //int first = 0;
    int second = 0;
    int third = 0;
    int upper = 0;
    int lower = 0;
    const int X = 0;
    const int Y = 1;

    /////////////////////////////////////////////////////////////////////

    const int tsize = triangles.size();
    float quad_length = quadLength(triangles);
    quad_length *= Q;
    this->setSquareLength(quad_length);
    //float quad_height = quad_length;

    /////////////////////////////////////////////////////////////////////

    //try to pack the triangles in a square(squareEdge*squareEdge)
    int npt = 0; //number of triangles already packed
    int ti = -1; //triangle index
    int ri = -1; //row_index index

    /////////////////////////////////////////////////////////////////////

    f2ten bdry = f2ten(2);
    bdry[X] = fvec(2, 0.0); //xlower, xupper
    bdry[Y] = fvec(2, 0.0); //ylower, yupper

    fvec shift = fvec(2, 0.0);

    /////////////////////////////////////////////////////////////////////

    ivec index = ivec(tsize, 0);
    fvec height = fvec(tsize, 0.0);

    {
        int i = 0;
        for (; i < tsize; i++)
        {
            index[i] = i;
            height[i] = (triangles[i]).getHeight();
        }
    }

    heapSort(height, index, tsize);
    reverse(height, index, tsize);
    IntList ilist = IntList(index);
    /*	
      {
         int j = 0;
         int size = index.size();
         for( ; j < size; j++)
         {
            cout << "\nindex  = " << flush << index[j]  << flush;
            cout << "   height = " << flush << height[j] << flush;
         }
         cout << "\n\n" << flush;
      }
   */
    bool sorted = true;

    /////////////////////////////////////////////////////////////////////

    while (npt < tsize)
    {
        if (ri >= 0)
        {
            bdry[X] = this->getXbounds(ri);
            bdry[Y] = this->getYbounds(ri);

            ti = nextTriangle(triangles, quad_length, ri,
                              index, ilist, bdry[X], bdry[Y]);
        }
        else
            ;

        if (ti < 0)
        {
            ++ri;

            if (ri < 1)
            {
                shift[1] = 0.0;
            }
            else
            {
                shift[1] = this->getYbounds(ri - 1, 1);
            }

            this->appendRow();

            ti = getHighest(triangles, index, ilist, sorted);
            //float security = (triangles[ti]).getHeight();
            //security *= 0.01;
            //shift[1] += security;

            second = (triangles[ti]).getC2dIndex(2);
            third = (triangles[ti]).getC2dIndex(3);
            lower = second;
            upper = third;

            (triangles[ti]).shiftC2d(shift);

            bdry[X][0] = (triangles[ti]).getC2d(X, lower);
            bdry[X][1] = (triangles[ti]).getC2d(X, upper);
            bdry[Y][0] = (triangles[ti]).getC2d(Y, lower);
            bdry[Y][1] = (triangles[ti]).getC2d(Y, upper);

            this->insertTriangle(ri, ti, &triangles[ti], bdry);
        }

        else
        {
            this->insertTriangle(ri, ti, &triangles[ti], bdry);
        }

        ++npt;
    }
}

int Patch::nextTriangle(trivec &triangles, float quad_length, int row_index,
                        const ivec &index, IntList &list,
                        fvec &xbdry, fvec &ybdry)
{
    /////////////////////////////////////////////////////////////////////

    const int X = 0;
    const int Y = 1;
    int first = 0;
    int second = 0;
    int third = 0;

    int triangle_index = -1;
    int tsize = triangles.size();

    /////////////////////////////////////////////////////////////////////

    f2ten nodes = f2ten(2);
    nodes[0] = fvec(3, 0.0);
    nodes[1] = fvec(3, 0.0);

    fvec p1 = fvec(2, 0.0);
    fvec p2 = fvec(2, 0.0);
    fvec p3 = fvec(2, 0.0);

    //needed for new bounds computation
    fvec tmp1 = fvec(2, 0.0);
    fvec tmp2 = fvec(2, 0.0);
    fvec tmp3 = fvec(2, 0.0);

    /////////////////////////////////////////////////////////////////////

    fvec xbOld = fvec(2, 0.0);
    fvec ybOld = fvec(2, 0.0);

    xbOld = this->getXbounds(row_index);
    ybOld = this->getYbounds(row_index);

    fvec xbNew = fvec(2, 0.0);
    fvec ybNew = fvec(2, 0.0);

    xbNew = this->getXbounds(row_index);
    ybNew = this->getYbounds(row_index);

    float wasteOld = -1.0;
    float wasteNew = -1.0;
    float wasteRatio = -1.0;

    /////////////////////////////////////////////////////////////////////

    float bound_length = 0.0;
    float row_area = 0.0;
    float rest_area = 0.0;

    float rl = 0.0;
    float rh = 0.0;

    /////////////////////////////////////////////////////////////////////

    int i = 0;
    int rj = 0;

    rj = list.getStart();
    while (rj > 0)
    {
        i = list.getElement(rj);

        //////////////////////////////////////////////////////////////////

        bound_length = FMIN(xbNew[0], xbNew[1]);
        bound_length += (triangles[i]).getLength();

        if (bound_length < quad_length)
        {

            ///////////////////////////////////////////////////////////////

            nodes = (triangles[i]).getC2d();
            first = (triangles[i]).getC2dIndex(1);
            second = (triangles[i]).getC2dIndex(2);
            third = (triangles[i]).getC2dIndex(3);
            points(tmp1, tmp2, tmp3, nodes,
                   first, second, third);

            ///////////////////////////////////////////////////////////////

            tryPoints(tmp1, tmp2, tmp3, xbdry,
                      ybdry, xbNew, ybNew);
            wasteNew = waste(tmp1, tmp2, tmp3, xbdry, ybdry, xbNew, ybNew);

            //if( (wasteOld < -0.1) && (FMAX(xbNew[0], xbNew[1]) <= quad_length ) )
            if ((wasteOld < -0.1))
            {
                p1 = tmp1;
                p2 = tmp2;
                p3 = tmp3;

                wasteOld = wasteNew;
                xbOld = xbNew;
                ybOld = ybNew;

                triangle_index = i;

                wasteRatio = wasteNew / (wasteNew + area(p1, p2, p3));

                if (wasteRatio < 0.1)
                {
                    break;
                }
                else
                    ;
            }

            //else if( (FMAX(xbNew[0], xbNew[1]) <= quad_length ) &&\ 
         //(wasteNew > -0.1)  && (wasteNew < wasteOld*0.95) )
            else if ((wasteNew > -0.1) && (wasteNew < wasteOld * 0.95))
            {
                p1 = tmp1;
                p2 = tmp2;
                p3 = tmp3;

                wasteOld = wasteNew;
                xbOld = xbNew;
                ybOld = ybNew;

                triangle_index = i;

                wasteRatio = wasteNew / (wasteNew + area(p1, p2, p3));

                if (wasteRatio < 0.1)
                {
                    break;
                }
                else
                    ;
            }
            else
            {
                rj = list.getNext(rj);
                continue;
            }

            ///////////////////////////////////////////////////////////////

            rj = list.getNext(rj);
        }
        else
        {
            rl = quad_length;
            rl -= ((xbNew[1] + xbNew[0]) / 2.0);
            rh = ybNew[1] - ybNew[0];

            rest_area = rl * rh;
            row_area = quad_length * rh;
            float area_ratio = rest_area / row_area;

            if (area_ratio < 0.05)
            {
                break;
            }
            else
            {
                rj = list.getNext(rj);
                continue;
            }
        }
    }

    if (triangle_index >= 0)
    {
        list.remove(rj);

        xbdry[0] = xbOld[0];
        xbdry[1] = xbOld[1];
        ybdry[0] = ybOld[0];
        ybdry[1] = ybOld[1];

        first = (triangles[triangle_index]).getC2dIndex(1);
        second = (triangles[triangle_index]).getC2dIndex(2);
        third = (triangles[triangle_index]).getC2dIndex(3);

        (triangles[triangle_index]).setC2d(p1[0], X, first);
        (triangles[triangle_index]).setC2d(p1[1] + ybdry[0], Y, first);
        (triangles[triangle_index]).setC2d(p2[0], X, second);
        (triangles[triangle_index]).setC2d(p2[1] + ybdry[0], Y, second);
        (triangles[triangle_index]).setC2d(p3[0], X, third);
        (triangles[triangle_index]).setC2d(p3[1] + ybdry[0], Y, third);
    }
    else
        ;

    return triangle_index;
}

Patch::Patch()
{
    Q = 1.1;
    num_rows = 0;
    rows = i2ten();
    xbounds = f2ten(2);
    ybounds = f2ten(2);
    square_length = 0;
}

////////////////////////////////////////////////////////////////////////

void points(fvec &p1, fvec &p2, fvec &p3, const f2ten &nodes,
            int first, int second, int third)
{
    //now start = 0  if A, 1 if B, 2 if C
    //orientation = 2 for clockwise, 1 for couter-clockwise

    int kk = -1;

    kk = first;
    p1[0] = nodes[0][kk];
    p1[1] = nodes[1][kk];

    kk = second;
    p2[0] = nodes[0][kk];
    p2[1] = nodes[1][kk];

    kk = third;
    p3[0] = nodes[0][kk];
    p3[1] = nodes[1][kk];
}

void tryPoints(fvec &p1, fvec &p2, fvec &p3, fvec xbdry,
               fvec ybdry, fvec &xbNew, fvec &ybNew)
{
    const float dy = ybdry[1] - ybdry[0];

    fvec shift = fvec(2, 0.0);
    float xs = 0.0;
    float lambda = 0.0;
    float xTemp = 0.0;

    if (xbdry[0] <= xbdry[1])
    {
        //no rotation of the triangle necessary

        shift[0] = xbdry[0];
        shift[1] = ybdry[0];

        //////////////////////////////////////////////////////////////////

        p1 += shift;
        p2 += shift;
        p3 += shift;

        //////////////////////////////////////////////////////////////////

        lambda = (ybdry[1] - ybdry[0]) / (p3[1] - p2[1]);
        xTemp = lambda * (p3[0] - p2[0]);
        xTemp += p2[0];

        //////////////////////////////////////////////////////////////////

        if (xTemp < xbdry[1])
        {
            xs = xbdry[1] - xTemp;
            p1[0] += xs;
            p2[0] += xs;
            p3[0] += xs;
            xbNew[0] = p2[0];
            xbNew[1] = xbdry[1];
            ybNew = ybdry;
        }
        else
        {
            xbNew[1] = xTemp;
            xbNew[0] = p2[0];
            ybNew = ybdry;
        }
    }
    else
    {
        //rotation of the triangle necessary
        rotate180(p1, p2, p3);

        //////////////////////////////////////////////////////////////////

        shift[0] = xbdry[1] - p2[0];
        shift[1] = ybdry[1] - p1[1];
        p1 += shift;
        p2 += shift;
        p3 += shift;

        //////////////////////////////////////////////////////////////////

        lambda = (ybdry[1] - ybdry[0]) / (p1[1] - p3[1]);
        xTemp = lambda * (p3[0] - p1[0]);
        xTemp += p1[0];

        //////////////////////////////////////////////////////////////////

        if (xTemp < xbdry[0])
        {
            xs = xbdry[0] - xTemp;
            p1[0] += xs;
            p2[0] += xs;
            p3[0] += xs;
            xbNew[0] = xbdry[0];
            xbNew[1] = p1[0];
            ybNew = ybdry;
        }
        else
        {
            xbNew[0] = xTemp;
            xbNew[1] = p1[0];
            ybNew = ybdry;
        }
    }
}

float waste(fvec p1, fvec p2, fvec p3, fvec xbdry, fvec ybdry,
            fvec xbNew, fvec ybNew)
{
    float surrArea = 0.0;
    float triangleArea = area(p1, p2, p3);

    //cout << "\narea P1-P2-P3 = " << flush << triangleArea << flush;

    float dx1 = xbNew[0] - xbdry[0];
    float dx2 = xbNew[1] - xbdry[1];
    float dx = (dx1 + dx2) / 2.0;
    float h1 = ybNew[1] - ybNew[0];
    float h2 = ybdry[1] - ybdry[0];
    float h = (h1 + h2) / 2.0;

    if (fabs(h1 - h2) > (1e-03 * fabs(h)))
    {
        cout << "\nerror while creating triangle patch ...\n" << flush;
        cout << "\n... problem with height of row\n" << flush;
        return -1.0;
    }
    else
    {
        surrArea = dx * h;
    }
    return (surrArea - triangleArea);
}

void rotate180(fvec &p1, fvec &p2, fvec &p3)
{
    //inversion with respect to the centre of mass
    float shift = 0.0;
    const float xCentre = (p1[0] + p2[0] + p3[0]) / 3.0;
    const float yCentre = (p1[1] + p2[1] + p3[1]) / 3.0;

    p1[0] -= xCentre;
    p2[0] -= xCentre;
    p3[0] -= xCentre;

    p1[1] -= yCentre;
    p2[1] -= yCentre;
    p3[1] -= yCentre;

    p1[0] *= (-1);
    p2[0] *= (-1);
    p3[0] *= (-1);

    p1[1] *= (-1);
    p2[1] *= (-1);
    p3[1] *= (-1);

    p1[0] += xCentre;
    p2[0] += xCentre;
    p3[0] += xCentre;

    p1[1] += yCentre;
    p2[1] += yCentre;
    p3[1] += yCentre;

    shift = p3[1];
    if (shift != 0.0)
    {
        p1[1] -= shift;
        p2[1] -= shift;
        p3[1] -= shift;
    }
    else
        ;

    shift = p2[0];
    if (shift != 0.0)
    {
        p1[0] -= shift;
        p2[0] -= shift;
        p3[0] -= shift;
    }
    else
        ;
}

////////////////////////////////////////////////////////////////////////

//computes barycentric coordinates of a triangle in the moment only
//anybody is free to extend it ;-)
fvec bary(const fvec &point, const f2ten &coord)
{
    int nvert = (coord[0]).size();
    int size = point.size();
    fvec lambda = fvec(nvert, 0);

    float delta = 0;
    fvec tmp = fvec();

    if (nvert != 3)
    {
        cout << "\njust triangles implemented," << flush;
        cout << "\nreturning lambda = (0,0,0,)\n" << flush;

        return lambda;
    }
    else
    {
    }

    if (size == 3)
    {
        tmp.resize(3, 0);
        lambda.resize(3, 0);

        fvec vec1 = fvec(3);
        fvec vec2 = fvec(3);
        fvec vec3 = fvec(3);

        {
            vec1[0] = coord[0][0];
            vec1[1] = coord[1][0];
            vec1[2] = coord[2][0];

            vec2[0] = coord[0][1];
            vec2[1] = coord[1][1];
            vec2[2] = coord[2][1];

            vec3[0] = coord[0][2];
            vec3[1] = coord[1][2];
            vec3[2] = coord[2][2];
        }

        delta = doubleArea(vec1, vec2, vec3);

        lambda[0] = doubleArea(point, vec2, vec3) / delta;
        lambda[1] = doubleArea(vec1, point, vec3) / delta;
        lambda[2] = doubleArea(vec1, vec2, point) / delta;
    }

    else if (size == 2)
    {
        lambda.resize(2, 0);

        fvec vec1 = fvec(2);
        fvec vec2 = fvec(2);
        fvec vec3 = fvec(2);

        {
            vec1[0] = coord[0][0];
            vec1[1] = coord[1][0];

            vec2[0] = coord[0][1];
            vec2[1] = coord[1][1];

            vec3[0] = coord[0][2];
            vec3[1] = coord[1][2];
        }
    }

    else
    {
    }

    return lambda;
}

/********************************************************\ 
 * 3D vector data ==>> tangential components as 2D data *
\********************************************************/

void project2triangle(Triangles *tri, const f2ten &v)
{
    f2ten c = f2ten();
    fvec cross = fvec(3, 0);
    fvec vnormal = fvec(3, 0);
    fvec vtang = fvec(3, 0);

    float scal = 0;
    float fact = 0;

    f2tCopy(c, (*tri).getCoord());

    cross = cross_product((c[1] - c[0]), (c[2] - c[0]));

    {
        int i = 0;
        int j = 0;
        scal = abs(cross);

        //unit normal to triangle
        for (i = 0; i < 3; i++)
        {
            cross[i] /= scal;
        }

        //for all three v-vectors
        for (i = 0; i < 3; i++)
        {
            //compute projection of v on the triangle normal
            for (j = 0; j < 3; j++)
            {
                fact += v[i][j] * cross[j];
            }
            fact /= scal;
            //now we have the length of the projection of v[i] on cross

            vnormal = cross;
            for (j = 0; j < 3; j++)
            {
                vnormal[i] *= fact;
            }
            vtang = v[i] - vnormal;

            (*tri).setVdata(vtang, i);
        }
    }
}

float quadLength(trivec &triangles)
{
    int tsize = triangles.size();
    float triangles_area = 0.0;
    float ql = 0.0;

    int start = 0;
    int orientation = 0;
    int jj = 0;

    fvec tmp1 = fvec(2, 0.0);
    fvec tmp2 = fvec(2, 0.0);
    fvec tmp3 = fvec(2, 0.0);

    for (int i = 0; i < tsize; i++)
    {
        start = (triangles[i]).get2dStart();
        orientation = (triangles[i]).get2dOrientation();
        modify(&start, &orientation);

        jj = start;
        tmp1[0] = (triangles[i]).getC2d(0, jj);
        tmp1[1] = (triangles[i]).getC2d(1, jj);

        jj += orientation;
        jj %= 3;
        tmp2[0] = (triangles[i]).getC2d(0, jj);
        tmp2[1] = (triangles[i]).getC2d(1, jj);

        jj += orientation;
        jj %= 3;
        tmp3[0] = (triangles[i]).getC2d(0, jj);
        tmp3[1] = (triangles[i]).getC2d(1, jj);

        triangles_area += area(tmp1, tmp2, tmp3);
    }
    ql = sqrt(triangles_area);
    //ql *= Q;

    //cout << "Laenge des Packens: " << flush << ql << flush;

    return ql;
}

void modify(int *start, int *orientation)
{
    *start -= 1;

    *orientation *= -1;
    *orientation += 1;
    *orientation /= 2;
    *orientation += 1;
}
