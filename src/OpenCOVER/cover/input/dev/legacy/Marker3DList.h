/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// this file: - class Marker3D
//            - class Marker3DList : a vector of Marker3D that can write to a log file or read from one

/******************************************************************************
*                       CGV Optical Tracking
*
*              license: currently no public release, all rights reserved
*
*       main developer: Hyosun Kim
*  assistant developer: Marcel Lancelle
*                       2006
*
*       Computer Graphics & Knowledge Visualization, TU Graz, Austria
*                       http://www.cgv.tugraz.at/
*
******************************************************************************/

#ifndef MARKER3DLIST_HEADER_INCLUDED
#define MARKER3DLIST_HEADER_INCLUDED

#include <vector>
#include "vec3_basetr.h"

namespace CGVOpticalTracking
{

class Marker3D
{
public:
    Marker3D(CGVVec3 _pos, int _id = -1)
        : pos(_pos)
        , id(_id)
    {
    }
    Marker3D(float _x, float _y, float _z, int _id = -1)
        : pos(_x, _y, _z)
        , id(_id){};

    inline const CGVVec3 &getPos() const
    {
        return pos;
    }
    inline int getId() const
    {
        return id;
    }

    //friend std::ostream& operator << (std::ostream& os, const Marker3D& m);
    friend std::istream &operator>>(std::istream &is, Marker3D &m);

protected:
    CGVVec3 pos;
    int id;
};

class Marker3DList : public std::vector<Marker3D>
{
public:
    //friend std::ostream& operator << (std::ostream& os, const Marker3DList& m);
    friend std::istream &operator>>(std::istream &is, Marker3DList &m);

    //bool write2File(std::string fileName);
    //bool readFromFile(std::string fileName) {return true;}; // TODO: implement
    //private:
    unsigned long time; // time (in ms) when the last of the source images was taken
};

} // namespace CGVOpticalTracking

#endif // MARKER3DLIST_HEADER_INCLUDED
