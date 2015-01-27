/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// this file: - class Marker3DList debug output

/******************************************************************************
 *                       CGV Optical Tracking
 *
 *              license: currently no public release, all rights reserved
 *
 *
 *  developers: Marcel Lancelle, Hyosun Kim, Lars Offen
 *              2006
 *
 *       Computer Graphics & Knowledge Visualization, TU Graz, Austria
 *                       http://www.cgv.tugraz.at/
 *
 ******************************************************************************/

#include "Marker3DList.h"

#include <stdio.h>
#include <fstream>
using namespace std;
using namespace CGVOpticalTracking;

istream &CGVOpticalTracking::operator>>(istream &is, Marker3D &m)
{
    char c;
    int id;
    CGVVec3 pos;
    is >> c;
    if (c == '(')
    {
        is >> id >> c;
        if (c == ',')
        {
            is >> pos >> c;
            if (c != ')')
            {
                is.clear(ios_base::badbit);
            }
        }
        else
        {
            is.clear(ios_base::badbit);
        }
    }
    else
    {
        is.clear(ios_base::badbit);
    }
    if (is)
    {
        m = Marker3D(pos, id);
    }
    return is;
}

istream &CGVOpticalTracking::operator>>(istream &is, Marker3DList &m)
{
    char c;
    Marker3D marker(0, 0, 0);
    m.clear();

    is >> c;
    if (c == '(')
    {
        is >> m.time >> c;
        while (c == ',')
        {
            // read the following Marker3D entry
            if (is >> marker)
            {
                m.push_back(marker);
                is >> c;
            }
            else
            {
                is.clear(ios_base::badbit);
            }
        }
        if (c != ')')
        {
            is.clear(ios_base::badbit);
        }
        is >> c;
        while (is && ((c == '\n') || (c == '\r')))
        {
            is >> c;
        }
        if (is)
        {
            is.unget();
        }
    }
    else
    {
        is.clear(ios_base::badbit);
    }
    if (!is)
    {
        m.clear(); // return nothing so it's obvious something didn't work!
    }
    return is;
}
