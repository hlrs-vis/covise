/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _NR_UTIL_H
#define _NR_UTIL_H

#include <util/coviseCompat.h>

const int NR_END = 1;
const int SECURITY = 100;
//const float FEPS = 1e-06;

////////////////////////////////////////////////////////////////////////

//vectors
typedef vector<int, allocator<int> > ivec;
typedef vector<long, allocator<long> > lvec;
typedef vector<float, allocator<float> > fvec;
typedef vector<double, allocator<double> > dvec;

ivec operator+(ivec a, ivec b);
lvec operator+(lvec a, lvec b);
fvec operator+(fvec a, fvec b);
dvec operator+(dvec a, dvec b);

ivec operator+=(ivec a, ivec b);
lvec operator+=(lvec a, lvec b);
fvec operator+=(fvec a, fvec b);
dvec operator+=(dvec a, dvec b);

ivec operator-(ivec a, ivec b);
lvec operator-(lvec a, lvec b);
fvec operator-(fvec a, fvec b);
dvec operator-(dvec a, dvec b);

ivec operator-=(ivec a, ivec b);
lvec operator-=(lvec a, lvec b);
fvec operator-=(fvec a, fvec b);
dvec operator-=(dvec a, dvec b);

////////////////////////////////////////////////////////////////////////

//tensors of rank 2
typedef vector<ivec, allocator<ivec> > i2ten;
typedef vector<lvec, allocator<lvec> > l2ten;
typedef vector<fvec, allocator<fvec> > f2ten;
typedef vector<dvec, allocator<dvec> > d2ten;

i2ten operator+(i2ten a, i2ten b);
l2ten operator+(l2ten a, l2ten b);
f2ten operator+(f2ten a, f2ten b);
d2ten operator+(d2ten a, d2ten b);

i2ten operator-(i2ten a, i2ten b);
l2ten operator-(l2ten a, l2ten b);
f2ten operator-(f2ten a, f2ten b);
d2ten operator-(d2ten a, d2ten b);

i2ten operator*(i2ten a, i2ten b);
l2ten operator*(l2ten a, l2ten b);
f2ten operator*(f2ten a, f2ten b);
d2ten operator*(d2ten a, d2ten b);

//obsolete but unfortunately used
//typedef vector<float, allocator<float> > float_vector;

////////////////////////////////////////////////////////////////////////

//tensors of rank 2
typedef vector<i2ten, allocator<i2ten> > i3ten;
//typedef vector<l2ten, allocator<l2ten> > l3ten;
typedef vector<f2ten, allocator<f2ten> > f3ten;
//typedef vector<d2ten, allocator<d2ten> > d3ten;

////////////////////////////////////////////////////////////////////////

inline int IMAX(int maxarg1, int maxarg2)
{
    return (maxarg1 > maxarg2 ? maxarg1 : maxarg2);
};

inline int IMIN(int minarg1, int minarg2)
{
    return (minarg1 < minarg2 ? minarg1 : minarg2);
};

inline long LMAX(long maxarg1, long maxarg2)
{
    return (maxarg1 > maxarg2 ? maxarg1 : maxarg2);
};

inline long LMIN(long minarg1, long minarg2)
{
    return (minarg1 < minarg2 ? minarg1 : minarg2);
};

inline float FMAX(float maxarg1, float maxarg2)
{
    return (maxarg1 > maxarg2 ? maxarg1 : maxarg2);
};

inline float FMIN(float minarg1, float minarg2)
{
    return (minarg1 < minarg2 ? minarg1 : minarg2);
};

inline double DMAX(double maxarg1, double maxarg2)
{
    return (maxarg1 > maxarg2 ? maxarg1 : maxarg2);
};

inline double DMIN(double minarg1, double minarg2)
{
    return (minarg1 < minarg2 ? minarg1 : minarg2);
};

////////////////////////////////////////////////////////////////////////

int maxIndex(fvec v);

fvec scaleVector(fvec vec, float val);
fvec operator*(fvec vec, float val);
fvec operator*(float val, fvec vec);
fvec operator/(fvec vec, float val);

void swap(float &a, float &b);
void swap(fvec &a, fvec &b);

float abs(fvec vec);

////////////////////////////////////////////////////////////////////////

fvec cross_product(const fvec &a, const fvec &b);
float scalar_product(const fvec &a, const fvec &b);
float operator*(const fvec &a, const fvec &b);

//polar coordinates:a[0] = cos(phi)*sin(theta)
//                  a[1] = sin(phi)*sin(theta)
//                  a[2] = cos(theta)
//theta: angle with positive z-axis
//void polar_coordinates(const fvec& a, float* phi, float* theta);

////////////////////////////////////////////////////////////////////////

/*****************************\ 
 *         print             *
\*****************************/
void prIvec(ivec iv);
void prI2ten(i2ten i2t);
void prFvec(fvec fv);
void prF2ten(f2ten f2t);

/*****************************\ 
 *          copy             *
\*****************************/
void ivecCopy(ivec &copy, ivec original);
void i2tCopy(i2ten &copy, i2ten original);
void fvecCopy(fvec &copy, fvec original);
void f2tCopy(f2ten &copy, f2ten original);

////////////////////////////////////////////////////////////////////////

class IntList
{

private:
    int start;
    int end;
    ivec list;
    ivec previous;
    ivec next;

public:
    inline int getStart()
    {
        return start;
    };

    int getEnd()
    {
        return end;
    };

    int getElement(int n)
    {
        return list[n];
    };

    int getNext(int n)
    {
        return next[n];
    };

    int getPrevious(int n)
    {
        return previous[n];
    };

    void remove(int pos);

    IntList(const ivec &il);
    //~IntList();
};

////////////////////////////////////////////////////////////////////////

//see W.H. Press et al.: Numerical Recipes in C, 2nd edition, pg. 282
//period > 2*10^8 (or 2*10**8 for "FORTRANers)
float random2(long *idum);

inline int ran2int(float number, int steps)
{
    int value = static_cast<int>(((steps - 1) * number) + 0.5);
    return value;
};

inline int f2int(float number)
{
    int value = static_cast<int>(number);
    return value;
};

//function unfortunately used though obsolte, see "nrutil.h"
inline float max(float a, float b)
{
    return ((a > b) ? a : b);
};

float sgn(float x);
#endif
