/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// this file: - stream input for recognized targets

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

#include "RecognizedTargetList.h"
#include <string> // for getline

using namespace std;
using namespace CGVOpticalTracking;

istream &CGVOpticalTracking::operator>>(istream &is, RecognizedTarget &t)
{
    char c;
    int markerId;
    CGVVec3 pos; //,eul;
    QuaternionRot rot;
    std::string id;

    t.marker3DInd.clear();
    is >> c;
    if (c == '(')
    {
        std::getline(is, id, ','); // TODO: this ignores all \n but should stop reading there!?
        //is >> id >> c;
        is.unget();
        is >> c;
        if (c == ',')
        {
            is >> pos >> c;
            if (c == ',')
            {
                //            is >> eul >> c;
                is >> rot >> c;
                if (c == ',')
                {
                    is >> c;
                    if (c == '(')
                    {
                        is >> markerId >> c;
                        t.marker3DInd.push_back(markerId);
                        while (c == ',')
                        {
                            is >> markerId >> c;
                            t.marker3DInd.push_back(markerId);
                        }
                        if (c == ')')
                        {
                            is >> c;
                            if (c != ')')
                            {
                                is.clear(std::ios_base::badbit);
                            }
                        }
                        else
                        {
                            is.clear(std::ios_base::badbit);
                        }
                    }
                    else
                    {
                        is.clear(std::ios_base::badbit);
                    }
                }
                else
                {
                    is.clear(std::ios_base::badbit);
                }
            }
            else
            {
                is.clear(std::ios_base::badbit);
            }
        }
        else
        {
            is.clear(std::ios_base::badbit);
        }
    }
    else
    {
        is.clear(std::ios_base::badbit);
    }
    if (is)
    {
        t.name = id;
        t.position = pos;
        t.quatRotation = rot;
        //      t.m_fEuler=eul;
    }
    else
    {
        t.marker3DInd.clear();
    }
    return is;
}

istream &CGVOpticalTracking::operator>>(istream &is, RecognizedTargetList &m)
{
    char c;
    RecognizedTarget target;
    m.clear();
    while (is >> target)
    {
        m.push_back(target);
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
    return is;
}
