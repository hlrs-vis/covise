/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * Point.h
 *
 *  Created on: Jan 2, 2012
 *      Author: jw_te
 */

#pragma once

#include <string>

namespace KardanikXML
{

class Point
{
private:
public:
    Point();
    Point(float x, float y, float z);

    void SetName(const std::string &name);
    const std::string &GetName() const;

    void SetX(float x);
    float GetX() const;

    void SetY(float y);
    float GetY() const;

    void SetZ(float z);
    float GetZ() const;

private:
    std::string m_Name;

    float m_X;
    float m_Y;
    float m_Z;
};
}
