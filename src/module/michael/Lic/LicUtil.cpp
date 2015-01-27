/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "LicUtil.h"

//returns pseudo random number 0 < ran-num < 1
float random2(long *idum)
{
    const int IM1 = 2147483563;
    const int IM2 = 2147483339;
    const float AM = (1.0 / IM1);
    const int IMM1 = (IM1 - 1);
    const int IA1 = 40014;
    const int IA2 = 40692;
    const int IQ1 = 53668;
    const int IQ2 = 52774;
    const int IR1 = 12211;
    const int IR2 = 3791;
    const int NTAB = 32;
    const int NDIV = (1 + (IMM1 / NTAB));
    const float EPS = 1.2e-7;
    const float RNMX = (1.0 - EPS);

    //random number generator of L'Ecuyer with Bays-Durham shuffle
    //and added safeguards. period > 2x10^8; call with idum < 0.

    int j = 0;
    long k = 0;
    static long idum2 = 123456789;
    static long iy = 0;
    static long iv[NTAB];
    float temp = 0.0;

    if (*idum <= 0)
    {
        if (-(*idum) < 1)
        {
            *idum = 1;
        }
        else
        {
            *idum = -(*idum);
        }
        idum2 = (*idum);

        for (j = NTAB + 7; j >= 0; j--) //Load the shuffle table(after 8 warm-ups) ??
        {
            k = (*idum) / IQ1;
            (*idum) = ((IA1 * ((*idum) - (k * IQ1))) - (k * IR1));
            if ((*idum) < 0)
            {
                (*idum) += IM1;
            }
            else
            {
            }
            if (j < NTAB)
            {
                iv[j] = (*idum);
            }
            else
            {
            }
        }

        iy = iv[0];
    }
    else
    {
    }

    k = (*idum) / IQ1; //starting point when not initializing
    (*idum) = ((IA1 * ((*idum) - (k * IQ1))) - (k * IR1));

    if ((*idum) < 0)
    {
        (*idum) += IM1;
    }
    else
    {
    }

    k = idum2 / IQ2;
    idum2 = ((IA2 * (idum2 - (k * IQ2))) - (k * IR2));

    if (idum2 < 0)
    {
        idum2 += IM2;
    }
    else
    {
    }

    j = iy / NDIV;
    iy = iv[j] - idum2;
    iv[j] = (*idum);

    if (iy < 1)
    {
        iy += IMM1;
    }
    else
    {
    }
    if ((temp = AM * iy) > RNMX)
    {
        return RNMX;
    }
    else
    {
        return temp;
    }

    return temp;
}

//0 < number < 1, psize: number of bytes per pixel
int ran2int(float number, short psize)
{
    int numGreyLevels = psize * 256;

    float value = number * numGreyLevels;

    for (int i = 0; i < numGreyLevels; i++)
    {
        if (value > i)
        {
            return (i - 1);
        }
        else
        {
            continue;
        }
    }
    return (numGreyLevels - 1);
}

void triPack2polygons(coOutputPort **packageOutPort, coDoPolygons **package,
                      trivec &triangles)
{
    int tsize = triangles.size();

    int num_coord = 3 * tsize;
    f2ten coord = f2ten(3);
    coord[0] = fvec(num_coord, 0.0);
    coord[1] = fvec(num_coord, 0.0);
    coord[2] = fvec(num_coord, 0.0);

    int num_conn = num_coord;
    ivec conn = ivec(num_conn);

    int num_poly = tsize;
    ivec poly = ivec(num_poly);

    int i = 0;
    for (; i < num_poly; i++)
    {
        poly[i] = 3 * i;
    }

    for (i = 0; i < num_conn; i++)
    {
        conn[i] = i;
    }

    for (i = 0; i < tsize; i++)
    {
        f2ten points = f2ten(2);
        points[0] = fvec(3, 0.0);
        points[1] = fvec(3, 0.0);

        points = (triangles[i]).getC2d();

        int j = 0;
        for (; j < 3; j++)
        {
            coord[0][3 * i + j] = points[0][j];
            coord[1][3 * i + j] = points[1][j];
        }
    }

    //(*package)
    (*package) = new coDoPolygons((*packageOutPort)->getObjName(), num_coord,
                                  &coord[0][0], &coord[1][0], &coord[2][0],
                                  num_conn, &conn[0],
                                  num_poly, &poly[0]);

    (*packageOutPort)->setCurrentObject(*package);
    (*package)->addAttribute("vertexOrder", "2");
}

void heapSort(fvec &height, ivec &index, int tsize)
{
    int i = 0;
    float ftemp = 0.0;
    int itemp = 0;
    int size = index.size();

    if (tsize != height.size())
    {
        tsize = height.size();
    }
    else
        ;

    if (tsize != size)
    {
        cout << "\nfuck you\n" << flush;
    }
    else
        ;

    i = tsize;
    i /= 2;
    i -= 1;
    for (; i >= 0; i--)
    {
        siftDown(height, index, i, tsize);
    }

    i = tsize - 1;
    for (; i >= 1; i--)
    {
        ftemp = height[0];
        height[0] = height[i];
        height[i] = ftemp;

        itemp = index[0];
        index[0] = index[i];
        index[i] = itemp;

        siftDown(height, index, 0, i - 1);
    }
}

void siftDown(fvec &height, ivec &index, int root, int bottom)
{
    bool done = false;
    int maxChild = 0;
    float ftmp = 0.0;
    int itmp = 0;

    done = 0;
    while ((root * 2 <= bottom) && (!done))
    {
        if ((root * 2) == bottom)
        {
            maxChild = root * 2;
        }
        else if (height[root * 2] > height[root * 2 + 1])
        {
            maxChild = root * 2;
        }
        else
        {
            maxChild = root * 2 + 1;
        }

        if (height[root] < height[maxChild])
        {
            ftmp = height[root];
            height[root] = height[maxChild];
            height[maxChild] = ftmp;

            itmp = index[root];
            index[root] = index[maxChild];
            index[maxChild] = itmp;

            root = maxChild;
        }
        else
        {
            done = true;
        }
    }
}

void reverse(fvec &height, ivec &index, int tsize)
{
    float tmp_height = 0.0;
    int tmp_index = 0;

    int i = 0;
    int end = tsize / 2;
    for (; i < end; i++)
    {
        tmp_index = index[i];
        tmp_height = height[i];

        index[i] = index[tsize - i - 1];
        index[tsize - i - 1] = tmp_index;

        height[i] = height[tsize - i - 1];
        height[tsize - i - 1] = tmp_height;
    }
}
