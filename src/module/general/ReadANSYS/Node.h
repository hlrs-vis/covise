/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _NODE_H_
#define _NODE_H_

struct Rotation
{
    float rotation_[9];
    float &operator[](int i)
    {
        return rotation_[i];
    }
    const float &operator[](int i) const
    {
        return rotation_[i];
    }
    enum
    {
        XX = 0,
        XY = 1,
        XZ = 2,
        YX = 3,
        YY = 4,
        YZ = 5,
        ZX = 6,
        ZY = 7,
        ZZ = 8
    };
    Rotation &operator*=(const Rotation &rhs);
};

struct Node
{
    double id_; // Da muss man später nix mehr konvertieren!
    double x_, y_, z_;
    double thxy_, thyz_, thzx_;
    Rotation Rotation_;
    Node();
    void MakeRotation();
};
#endif
