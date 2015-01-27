/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "nrutil.h"

/**********************************************************\ 
 *                                                        *
 * vector & tensor operators for addition and subtraction *
 *                                                        *
\**********************************************************/

ivec operator+(ivec a, ivec b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }

    for (int i = 0; i < size; i++)
    {
        a[i] += b[i];
    }

    return a;
}

lvec operator+(lvec a, lvec b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }

    for (int i = 0; i < size; i++)
    {
        a[i] += b[i];
    }

    return a;
}

fvec operator+(fvec a, fvec b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }

    for (int i = 0; i < size; i++)
    {
        a[i] += b[i];
    }

    return a;
}

dvec operator+(dvec a, dvec b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }

    for (int i = 0; i < size; i++)
    {
        a[i] += b[i];
    }

    return a;
}

ivec operator+=(ivec a, ivec b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }

    for (int i = 0; i < size; i++)
    {
        a[i] += b[i];
    }

    return a;
}

lvec operator+=(lvec a, lvec b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }

    for (int i = 0; i < size; i++)
    {
        a[i] += b[i];
    }

    return a;
}

fvec operator+=(fvec a, fvec b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }

    for (int i = 0; i < size; i++)
    {
        a[i] += b[i];
    }

    return a;
}

dvec operator+=(dvec a, dvec b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }

    for (int i = 0; i < size; i++)
    {
        a[i] += b[i];
    }

    return a;
}

ivec operator-(ivec a, ivec b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }

    for (int i = 0; i < size; i++)
    {
        a[i] -= b[i];
    }

    return a;
}

lvec operator-(lvec a, lvec b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }

    for (int i = 0; i < size; i++)
    {
        a[i] -= b[i];
    }

    return a;
}

fvec operator-(fvec a, fvec b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        //return a;
        exit(1);
    }
    else
    {
        size = a.size();
    }

    for (int i = 0; i < size; i++)
    {
        a[i] -= b[i];
    }

    return a;
}

dvec operator-(dvec a, dvec b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }

    for (int i = 0; i < size; i++)
    {
        a[i] -= b[i];
    }

    return a;
}

ivec operator-=(ivec a, ivec b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }

    for (int i = 0; i < size; i++)
    {
        a[i] -= b[i];
    }

    return a;
}

lvec operator-=(lvec a, lvec b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }

    for (int i = 0; i < size; i++)
    {
        a[i] -= b[i];
    }

    return a;
}

fvec operator-=(fvec a, fvec b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }

    for (int i = 0; i < size; i++)
    {
        a[i] -= b[i];
    }

    return a;
}

dvec operator-=(dvec a, dvec b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }

    for (int i = 0; i < size; i++)
    {
        a[i] -= b[i];
    }

    return a;
}

i2ten operator+(i2ten a, i2ten b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }
    for (int i = 0; i < size; i++)
    {
        a[i] += b[i];
    }

    return a;
}

l2ten operator+(l2ten a, l2ten b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }
    for (int i = 0; i < size; i++)
    {
        a[i] += b[i];
    }

    return a;
}

f2ten operator+(f2ten a, f2ten b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }
    for (int i = 0; i < size; i++)
    {
        a[i] += b[i];
    }

    return a;
}

d2ten operator+(d2ten a, d2ten b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }
    for (int i = 0; i < size; i++)
    {
        a[i] += b[i];
    }

    return a;
}

i2ten operator-(i2ten a, i2ten b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }
    for (int i = 0; i < size; i++)
    {
        a[i] -= b[i];
    }

    return a;
}

l2ten operator-(l2ten a, l2ten b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }
    for (int i = 0; i < size; i++)
    {
        a[i] -= b[i];
    }

    return a;
}

f2ten operator-(f2ten a, f2ten b)
{
    int size = 0;
    for (int i = 0; i < size; i++)
    {
        a[i] -= b[i];
    }

    return a;
}

d2ten operator-(d2ten a, d2ten b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }
    for (int i = 0; i < size; i++)
    {
        a[i] -= b[i];
    }

    return a;
}

i2ten operator*(i2ten a, i2ten b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }

    i2ten tmp = i2ten(size);
    for (int i = 0; i < size; i++)
    {
        (tmp[i]).resize(size, 0);
    }

    {
        int i;
        int j;
        int k;

        for (i = 0; i < size; i++)
        {
            for (j = 0; j < size; j++)
            {
                for (k = 0; k < size; k++)
                {
                    tmp[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }

    return tmp;
}

l2ten operator*(l2ten a, l2ten b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }

    l2ten tmp = l2ten(size);
    for (int i = 0; i < size; i++)
    {
        (tmp[i]).resize(size, 0);
    }

    {
        int i;
        int j;
        int k;

        for (i = 0; i < size; i++)
        {
            for (j = 0; j < size; j++)
            {
                for (k = 0; k < size; k++)
                {
                    tmp[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }

    return tmp;
}

f2ten operator*(f2ten a, f2ten b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }

    f2ten tmp = f2ten(size);
    for (int i = 0; i < size; i++)
    {
        (tmp[i]).resize(size, 0.0);
    }

    {
        int i;
        int j;
        int k;

        for (i = 0; i < size; i++)
        {
            for (j = 0; j < size; j++)
            {
                for (k = 0; k < size; k++)
                {
                    tmp[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }

    return tmp;
}

d2ten operator*(d2ten a, d2ten b)
{
    int size = 0;
    if (a.size() != b.size())
    {
        exit(1);
    }
    else
    {
        size = a.size();
    }

    d2ten tmp = d2ten(size);
    for (int i = 0; i < size; i++)
    {
        (tmp[i]).resize(size, 0.0);
    }

    {
        int i;
        int j;
        int k;

        for (i = 0; i < size; i++)
        {
            for (j = 0; j < size; j++)
            {
                for (k = 0; k < size; k++)
                {
                    tmp[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }

    return tmp;
}

////////////////////////////////////////////////////////////////////////

int maxIndex(fvec v)
{
    int k = 0;
    float max = 0.0;
    int size = v.size();

    int i;
    for (i = 0; i < size; i++)
    {
        if (max < fabs(v[i]))
        {
            max = fabs(v[i]);
            k = i;
        }
        else
            ;
    }

    return k;
}

fvec scaleVector(fvec vec, float val)
{
    int size = vec.size();

    int i;
    for (i = 0; i < size; i++)
    {
        vec[i] *= val;
    }
    return vec;
}

fvec operator*(fvec vec, float val)
{
    int size = vec.size();

    int i;
    for (i = 0; i < size; i++)
    {
        vec[i] *= val;
    }
    return vec;
}

fvec operator*(float val, fvec vec)
{
    int size = vec.size();

    int i;
    for (i = 0; i < size; i++)
    {
        vec[i] *= val;
    }
    return vec;
}

fvec operator/(fvec vec, float val)
{
    int size = vec.size();

    int i;
    for (i = 0; i < size; i++)
    {
        vec[i] /= val;
    }
    return vec;
}

void swap(float &a, float &b)
{
    float temp = b;
    b = a;
    a = temp;
}

void swap(fvec &a, fvec &b)
{
    int asize = a.size();
    int bsize = b.size();

    if (asize != bsize)
    {
        exit(99);
    }
    else
        ;

    fvec tmp = fvec(asize, 0.0);
    tmp = b;
    b = a;
    a = tmp;
}

float abs(fvec vec)
{
    float absVec = 0;

    for (int i = 0; i < vec.size(); i++)
    {
        absVec += ((vec[i]) * (vec[i]));
    }
    absVec = sqrt(absVec);

    return absVec;
}

//vector product in 3D space
fvec cross_product(const fvec &a, const fvec &b)
{
    fvec c = fvec(3, 0);

    if ((a.size() != 3) || (b.size() != 3))
    {
        cout << "\nsize error of input vectors in cross_product\n" << flush;
    }
    else
    {
        c[0] = a[1] * b[2] - a[2] * b[1];
        c[1] = a[2] * b[0] - a[0] * b[2];
        c[2] = a[0] * b[1] - a[1] * b[0];
    }

    return c;
}

//scalar product of two vectors
float scalar_product(const fvec &a, const fvec &b)
{
    int size = 0;
    float temp = 0;

    size = a.size();
    if (size != b.size())
    {
        cout << "\nWARNING: Scalar Product of Vectors with" << flush;
        cout << "Unequal Size Occured\n" << flush;
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            temp += (a[i] * b[i]);
        }
    }

    return temp;
}

float operator*(const fvec &a, const fvec &b)
{
    int size = 0;
    float temp = 0;

    size = a.size();
    if (size != b.size())
    {
        cout << "\nWARNING: Scalar Product of Vectors with" << flush;
        cout << "Unequal Size Occured\n" << flush;
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            temp += (a[i] * b[i]);
        }
    }

    return temp;
}

/*
void polar_coordinates(const fvec& a, float* phi, float* theta)
{
   float zwopi = 2*arccos(1.0);

   int size = 0;
   float aa = 0.0;
   float tmp = 0.0;

   fvec ex = fvec(3,0);  ex[0] = 1.0;
   fvec ey = fvec(3,0);  ey[1] = 1.0;
fvec ez = fvec(3,0);  ez[2] = 1.0;

size = a.size();
aa = abs(a);

tmp = scalar_product(a, ez);
tmp /= aa;
*theta = acos(tmp);

if( fabs(1.0 - tmp) < FEPS )
{
*phi = 0.0;
}
else
{
tmp = scalar_product(a, ex);
tmp /= aa
if( (a[0] >= 0) || (a[1] >= 0) )
{
*phi = acos(tmp);
}
else if( (a[0] >= 0) || (a[1] < 0) )
{
*phi = zwopi - acos(tmp);
}
else if( (a[0] < 0) || (a[1] >= 0) )
{
*phi = arccos(tmp);
}
else if( (a[0] < 0) || (a[1] < 0) )
{
*phi = zwopi - acos(tmp);
}
else  //haeh ???
{
}
}
}
*/

/******************************\ 
 *      print functions       *
\******************************/

void prIvec(ivec iv)
{
    int size = iv.size();

    for (int i = 0; i < size; i++)
    {
        cout << iv[i] << flush << "  " << flush;
    }

    return;
}

void prI2ten(i2ten i2t)
{
    int row_size = i2t.size();
    int col_size = 0;

    for (int i = 0; i < row_size; i++)
    {
        col_size = (i2t[i]).size();
        for (int j = 0; j < col_size; j++)
        {
            cout << i2t[i][j] << flush << "  " << flush;
        }
        if (i != (row_size - 1))
        {
            cout << '\n' << flush;
        }
        else
        {
        }
    }

    return;
}

void prFvec(fvec fv)
{
    int size = fv.size();

    cout << "\nprinting vector ...\n\n" << flush;

    for (int i = 0; i < size; i++)
    {
        cout << fv[i] << flush << "  " << flush;
    }
    cout << '\n' << flush;

    return;
}

void prF2ten(f2ten f2t)
{
    int row_size = f2t.size();
    int col_size = 0;

    cout << "\nnew object:\n" << flush;
    cout << "-----------\n" << flush;

    for (int i = 0; i < row_size; i++)
    {
        col_size = (f2t[i]).size();
        for (int j = 0; j < col_size; j++)
        {
            cout << f2t[i][j] << flush << "  " << flush;
        }
        cout << '\n';
    }
    cout << "\n____________________\n" << flush;

    return;
}

/******************************\ 
 *      copy functions        *
\******************************/

void ivecCopy(ivec &copy, ivec original)
{
    int size = original.size();
    copy.resize(size);
    for (int i = 0; i < size; i++)
    {
        copy[i] = original[i];
    }

    return;
}

void i2tCopy(i2ten &copy, i2ten original)
{
    int row_size = original.size();
    int col_size = 0;

    copy.resize(row_size);

    int i = 0;
    while (i < row_size)
    {
        col_size = (original[i]).size();
        (copy[i]).resize(col_size);

        for (int j = 0; j < col_size; j++)
        {
            copy[i][j] = original[i][j];
        }

        ++i;
    }

    return;
}

void fvecCopy(fvec &copy, fvec original)
{
    int size = original.size();
    copy.resize(size);
    for (int i = 0; i < size; i++)
    {
        copy[i] = original[i];
    }

    return;
}

void f2tCopy(f2ten &copy, f2ten original)
{
    int row_size = original.size();
    int col_size = 0;

    copy.resize(row_size);

    int i = 0;
    while (i < row_size)
    {
        col_size = (original[i]).size();
        (copy[i]).resize(col_size);

        for (int j = 0; j < col_size; j++)
        {
            copy[i][j] = original[i][j];
        }

        ++i;
    }

    return;
}

////////////////////////////////////////////////////////////////////////

void IntList::remove(int pos)
{
    int i = 0;
    int j = 0;

    if (pos == start)
    {
        start = next[start];
        j = next[pos];

        previous[j] = -1;
        previous[pos] = -2;
        next[pos] = -2;
    }
    else if (pos == (end - 1))
    {
        end = previous[end - 1];
        i = previous[pos];

        next[i] = -1;
        previous[pos] = -2;
        next[pos] = -2;
    }
    else
    {
        i = previous[pos];
        j = next[pos];

        previous[j] = previous[pos];
        next[i] = next[pos];
        previous[pos] = -2;
        next[pos] = -2;
    }
}

IntList::IntList(const ivec &il)
{
    start = 0;
    end = il.size();
    list.resize(end);
    next.resize(end);
    previous.resize(end);

    list = il;

    {
        int i = 0;
        for (; i < end; i++)
        {
            next[i] = i + 1;
            previous[i] = i - 1;
        }
        next[end - 1] = -1;
    }
}

////////////////////////////////////////////////////////////////////////

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

float sgn(float x)
{
    if (x > 0)
    {
        return 1.0;
    }
    else
    {
        return -1.0;
    }
}
