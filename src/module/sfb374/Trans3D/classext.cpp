/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          classext.cpp  -  class extensions
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#include "classext.h"
#include "fortran.h"

// ***************************************************************************
// float rectangle class
// ***************************************************************************

void TRectF::Set(float l, float t, float r, float b)
{
    left = l;
    right = r;
    top = t;
    bottom = b;
}

// ***************************************************************************
// 2D interpolation class
// ***************************************************************************

// delete entries

void Interpolation2D::Delete()
{
    Data.ReSize(0, 0);
    dx = 0;
    dy = 0;
}

// get value

prec Interpolation2D::GetPrec(prec x, prec y)
{
    prec p;
    int i, j, imax, jmax;

    if (dx <= 0 || dy <= 0)
        return 0;
    imax = Data.GetiMax() - 1;
    jmax = Data.GetjMax() - 1;
    x = (x + dx * imax / 2.0) / dx;
    y = (y + dy * jmax / 2.0) / dy;
    if (x < 0 || x > imax || y < 0 || y > jmax)
        return 0;
    i = (int)x;
    j = (int)y;
    if (i == imax)
        i = imax - 1;
    if (j == jmax)
        j = jmax - 1;
    x -= i;
    y -= j;
    p = x * (y * Data(i + 1, j + 1) + (1 - y) * Data(i + 1, j)) + (1 - x) * (y * Data(i, j + 1) + (1 - y) * Data(i, j));
    return p;
}

// copy class

Interpolation2D &Interpolation2D::operator=(const Interpolation2D &ipol)
{
    int i, j, imax, jmax;

    if (this == &ipol)
        return *this;
    imax = ipol.GetXEntries();
    jmax = ipol.GetYEntries();
    ReSize(imax, jmax);
    dx = ipol.dx;

    dy = ipol.dy;
    for (i = 0; i < imax; i++)
        for (j = 0; j < jmax; j++)
            Data(i, j) = ipol.Get(i, j);
    return *this;
}

// save class

void Interpolation2D::Save(int unit, bool)
{
    int i, j, imax, jmax;

    if ((imax = Data.GetiMax()) <= 0 || (jmax = Data.GetjMax()) <= 0)
        return;
    unitwrite(unit, imax, "Eintraege =\t%i");
    unitwrite(unit, jmax, "\t%i\n");
    unitwrite(unit, dx, "Abstand =\t%le");
    unitwrite(unit, dy, "\t%le\n");
    for (i = 0; i < imax; i++)
        for (j = 0; j < jmax; j++)
        {
            if (j < jmax - 1)
                unitwrite(unit, (double)Data(i, j), "%le\t");
            else
                unitwrite(unit, (double)Data(i, j), "%le\n");
        }
}

// read class

void Interpolation2D::Read(int unit, bool)
{
    int i, j, imax, jmax;

    imax = readint(unit);
    jmax = readint(unit);
    dx = readreal(unit);
    dy = readreal(unit);
    ReSize(imax, jmax);
    for (i = 0; i < imax; i++)
        for (j = 0; j < jmax; j++)
            Data(i, j) = readreal(unit);
}

// write data table

ostream &operator<<(ostream &ps, const Interpolation2D &src)
{
    int i, j, imax = src.GetXEntries(), jmax = src.GetYEntries();

    ps << imax << ',' << jmax << " entries" << endl;
    ps << "spacing:\t" << src.GetDx() << ',' << src.GetDy() << endl;

    for (i = 0; i < imax; i++)
    {
        for (j = 0; j < jmax - 1; j++)
            ps << src.Get(i, j) << '\t';
        ps << src.Get(i, j) << endl;
    }

    return ps;
}

// read data table

istream &operator>>(istream &ps, Interpolation2D &src)
{
    int i, j, imax, jmax;
    char ch;
    prec p, dx, dy;

    ps >> imax >> ch >> jmax;
    ps.ignore(1000, '\t');
    ps >> dx >> ch >> dy;
    ps.ignore(1000, '\n');
    if (ps.fail())
        return ps;
    src.ReSize(imax, jmax);
    src.SetDx(dx);
    src.SetDy(dy);

    for (i = 0; i < imax; i++)
        for (j = 0; j < jmax; j++)
        {
            ps >> p;
            ps.get();
            src.Set(i, j, p);
        }
    return ps;
}

// conversion int -> bool

bool ConvBool(int i)
{
    return i != 0;
}

int ConvInt(bool b)
{
    if (b)
        return 1;
    return 0;
}
