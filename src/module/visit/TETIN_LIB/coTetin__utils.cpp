/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coTetin__utils.h"
#include <string.h>
#include <math.h>
#include <ctype.h>

#define issep(c) ((c) == ',' || isspace(c))

typedef float four_vec[4];
typedef float triple[3];
coTetin__BSpline *read_spline(istream &str, ostream &ostr)
{
    // read the number of control points and orders
    float ftemp;
    if (getgplfloat(str, ostr, &ftemp))
        return 0;
    int ncps[2];
    ncps[0] = (int)ftemp;

    if (getgplfloat(str, ostr, &ftemp))
        return 0;
    ncps[1] = (int)ftemp;

    if (getgplfloat(str, ostr, &ftemp))
        return 0;
    int ord[2];
    ord[0] = (int)ftemp;

    if (getgplfloat(str, ostr, &ftemp))
        return 0;
    ord[1] = (int)ftemp;

    if (getgplfloat(str, ostr, &ftemp))
        return 0;
    int rat = (int)ftemp;

    coTetin__BSpline *spl = new coTetin__BSpline();
    // sanity check
    if (ord[0] <= 0 || ord[0] > 30 || ord[1] <= 0 || ord[1] > 30 || ncps[0] <= 0 || ncps[0] > 10000 || ncps[1] <= 0 || ncps[1] > 10000)
    {
        ostr << "error reading spline surface\n";
        return 0;
    }
    spl->ncps[0] = ncps[0];
    spl->ncps[1] = ncps[1];
    spl->ord[0] = ord[0];
    spl->ord[1] = ord[1];
    spl->rat = rat;

    // read the knot vectors
    int i, j;
    float **knot = spl->knot;
    for (i = 0; i < 2; i++)
    {
        knot[i] = new float[ncps[i] + ord[i]];
        for (j = 0; j < ncps[i] + ord[i]; j++)
        {
            if (getgplfloat(str, ostr, knot[i] + j))
                return 0;
            if (j && knot[i][j] < knot[i][j - 1])
            {
                ostr << "knot[i][" << j << "d] < knot[i][" << j - 1 << "] (" << knot[i][j] << " < " << knot[i][j - 1] << ")\n";
                return 0;
            }
            if (j >= ord[i] && knot[i][j] <= knot[i][j - ord[i]])
            {
                ostr << "knot multiplcity too high\n";
                ostr << "knot[" << i << "][" << j << "] <= knot[" << i << "][" << j - ord[i] << "] (" << knot[i][j] << knot[i][j - ord[i]] << ")\n";
                return 0;
            }
        }
    }
    // total number of points
    int nt = ncps[0] * ncps[1];
    if (rat)
    {

        spl->cps = new float[nt * 4];
        four_vec *cps = (four_vec *)spl->cps;
        // read the control points
        for (j = 0; j < nt; j++)
        {
            for (i = 0; i < 4; i++)
            {
                if (getgplfloat(str, ostr, cps[j] + i))
                    return 0;
            }
        }
    }
    else
    {
        spl->cps = new float[nt * 3];
        triple *cps = (triple *)spl->cps;
        // read the control points
        for (j = 0; j < nt; j++)
        {
            for (i = 0; i < 3; i++)
            {
                if (getgplfloat(str, ostr, cps[j] + i))
                    return 0;
            }
        }
    }
    return spl;
}

coTetin__BSpline *read_spline_curve(istream &str, ostream &ostr)
{
    // read the number of control points and orders
    float ftemp;
    if (getgplfloat(str, ostr, &ftemp))
        return 0;
    int m = (int)ftemp;

    if (getgplfloat(str, ostr, &ftemp))
        return 0;
    int ord = (int)ftemp;

    if (getgplfloat(str, ostr, &ftemp))
        return 0;
    int rat = (int)ftemp;

    coTetin__BSpline *spl = new coTetin__BSpline();
    // sanity check
    if (ord <= 0 || ord > 30 || m <= 0 || m > 10000)
    {
        ostr << "error in input file";
        return 0;
    }
    spl->ncps[0] = m;
    spl->ncps[1] = 0;
    spl->ord[0] = ord;
    spl->ord[1] = 0;
    spl->rat = rat;
    // read the knot vectors
    float *knot = new float[m + ord];
    spl->knot[0] = knot;
    spl->knot[1] = 0;
    int j;
    for (j = 0; j < m + ord; j++)
    {
        if (getgplfloat(str, ostr, knot + j))
            return 0;
    }
    int i;
    if (rat)
    {
        spl->cps = new float[m * 4];
        four_vec *cps = (four_vec *)spl->cps;
        // read the control points
        for (j = 0; j < m; j++)
        {
            for (i = 0; i < 4; i++)
            {
                if (getgplfloat(str, ostr,
                                cps[j] + i))
                    return 0;
            }
        }
    }
    else
    {
        spl->cps = new float[m * 3];
        triple *cps = (triple *)spl->cps;
        // read the control points
        for (j = 0; j < m; j++)
        {
            for (i = 0; i < 3; i++)
            {
                if (getgplfloat(str, ostr,
                                cps[j] + i))
                    return 0;
            }
        }
    }
    return spl;
}

int comment_getc(istream &str, ostream &ostr)
{
    while (1)
    {
        char c;
        str.get(c);
        // skip comment line
        if (c == '/')
        {
            char nextcar;
            if (str.get(nextcar))
            {
                if (nextcar == '/')
                {
                    while (c != '\n' && (int)c != EOF && !str.eof())
                    {
                        str.get(c);
                    }
                }
                else
                {
                    str.putback(nextcar);
                    return c;
                }
            }
        }
        else
        {
            return c;
        }
    }
}

int getgplfloat(istream &str, ostream &ostr, float *val)
{
#define BUFFER_SIZE 200
    char buffer[BUFFER_SIZE];

    int count = 0;
    // read up to the next dark space
    int c = ' ';
#define issep(c) ((c) == ',' || isspace(c))
    while (issep(c))
    {
        c = comment_getc(str, ostr);
    }
    buffer[count] = c;
    count++;
    // read the number
    while (count < BUFFER_SIZE)
    {
        c = comment_getc(str, ostr);
        if (c == EOF)
            break;
        buffer[count] = c;
        count++;
        if (issep(c))
            break;
    }
#if 1
    // need this to handle nt-files
    // can't do this if input will hang rather than produce EOF
    // read up to but not including the next dark space
    // or up to and including a new line
    while (c != '\n' && c != EOF)
    {
        c = comment_getc(str, ostr);
        if (issep(c) == 0)
        {
            str.putback((char)c);
            break;
        }
    }
#endif
    buffer[count] = 0;
    if (sscanf(buffer, "%f", val) != 1)
    {
        ostr << "error reading float from surface file!\n";
        return 1;
    }
    else
    {
        return 0;
    }
}

int coTetin__break_line(char *temp, char *linev[])
{
    char left = '{';
    char right = '}';

    int linec = 0;

    while (1)
    {
        // skip over white space
        while (isspace(*temp))
            temp++;
        if (*temp == 0)
            break;
        if (*temp == left)
        {
            // check for stuff in brackets
            temp++;
            linev[linec] = temp;
            linec++;
            while (*temp && *temp != right)
            {
                temp++;
            }
            if (*temp == 0)
            {
                return linec;
            }
            *temp = 0;
            temp++;
        }
        else
        {
            // unbracketed stuff
            linev[linec] = temp;
            linec++;
            // skip over dark space
            while (isspace(*temp) == 0 && *temp != 0)
                temp++;
            if (*temp == 0)
                break;
            *temp = 0;
            temp++;
        }
    }
    return linec;
}

coTetin__BSpline::coTetin__BSpline()
{
    ncps[0] = ncps[1] = 0;
    ord[0] = ord[1] = 0;
    rat = 0;
    knot[0] = knot[1] = 0;
    cps = 0;
}

coTetin__BSpline::~coTetin__BSpline()
{
    ncps[0] = ncps[1] = 0;
    ord[0] = ord[1] = 0;
    rat = 0;
    if (knot[0])
        delete[] knot[0];
    knot[0] = 0;
    if (knot[1])
        delete[] knot[1];
    knot[1] = 0;
    if (cps)
        delete[] cps;
    cps = 0;
}

void coTetin__BSpline::putBinary(int *&intDat, float *&floatDat, char *&charDat)
{
    ncps[0] = *intDat++;
    ncps[1] = *intDat++;
    ord[0] = *intDat++;
    ord[1] = *intDat++;
    rat = *intDat++;
    int i;
    for (i = 0; i < 2; i++)
    {
        knot[i] = 0;
        if (ncps[i] + ord[i] > 0)
        {
            knot[i] = new float[ncps[i] + ord[i]];
            memcpy((void *)knot[i], (void *)floatDat, (ncps[i] + ord[i]) * sizeof(float));
            floatDat += ncps[i] + ord[i];
        }
    }
    // adjust for bspline curves
    int ncps1 = ((ncps[1] > 0 || ord[1] > 0) ? ncps[1] : 1);
    int nt = ncps[0] * ncps1;
    cps = 0;
    if (nt > 0)
    {
        if (rat)
        {
            nt *= 4;
        }
        else
        {
            nt *= 3;
        }
        cps = new float[nt];
        memcpy((void *)cps, (void *)floatDat, nt * sizeof(float));
        floatDat += nt;
    }
}

void coTetin__BSpline::addSizes(int &numInt, int &numFloat, int &numChar) const
{
    numInt += 5; // ncps[2],ord[2],rat
    int i;
    for (i = 0; i < 2; i++)
    {
        numFloat += ncps[i] + ord[i]; // knot[i]
    }
    int ncps1 = ((ncps[1] > 0 || ord[1] > 0) ? ncps[1] : 1);
    int nt = ncps[0] * ncps1;
    // cps
    if (rat)
    {
        numFloat += nt * 4;
    }
    else
    {
        numFloat += nt * 3;
    }
}

void coTetin__BSpline::getBinary(int *&intDat, float *&floatDat, char *&charDat) const
{
    *intDat++ = ncps[0];
    *intDat++ = ncps[1];
    *intDat++ = ord[0];
    *intDat++ = ord[1];
    *intDat++ = rat;
    int i;
    for (i = 0; i < 2; i++)
    {
        if (knot[i] && (ncps[i] + ord[i]) > 0)
        {
            memcpy((void *)floatDat, (void *)knot[i], (ncps[i] + ord[i]) * sizeof(float));
            floatDat += ncps[i] + ord[i];
        }
    }
    // adjust for bspline curves
    int ncps1 = ((ncps[1] > 0 || ord[1] > 0) ? ncps[1] : 1);
    int nt = ncps[0] * ncps1;
    // cps
    if (cps && nt > 0)
    {
        if (rat)
        {
            nt *= 4;
        }
        else
        {
            nt *= 3;
        }
        memcpy((void *)floatDat, (void *)cps, nt * sizeof(float));
        floatDat += nt;
    }
}
