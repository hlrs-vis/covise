/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// this file: - class RecognizedTarget
//            - class RecognizedTargetList: contains all recognized targets
//            - note: this is a stripped down version of RecognizedTargetListExt

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

#ifndef RECOGNIZEDTARGETLIST_HEADER_INCLUDED
#define RECOGNIZEDTARGETLIST_HEADER_INCLUDED

#include <vector>
#include <iostream>
#include "vec3_basetr.h"
#include "quaterniontr.hpp"

namespace CGVOpticalTracking
{

class RecognizedTarget
{
public:
    //friend std::ostream& operator << (std::ostream& os, const RecognizedTarget& t);
    friend std::istream &operator>>(std::istream &is, RecognizedTarget &t);
    RecognizedTarget(int size = 0)
    {
        marker3DInd.resize(size, -1);
    }

    //   inline const CGVVec3& getPosition()   const { return m_fPos; }
    //   inline const CGVVec3& getEulerAngle() const { return m_fEuler; }

    //inline void   setPosition(float x, float y, float z);
    //inline void   setEulerAngle(float x, float y, float z);
    //inline void   setRotation(float angle, float axis_x, float axis_y, float axis_z); // angle in rad

    //inline bool   associateVertexWithMarker(size_t v, int m);
    //	virtual const std::string getName() const {return name;};
    //	virtual void setName(const std::string name) {this->name = name;};
    //private:
    std::string name;
    std::vector<int> marker3DInd; // matched vertices have an index to the marker, -1 otherwise

    // position and angles are calculated by Target6DoFExtraction
    CGVVec3 position;
    //CGVVec3 m_fEuler;
    QuaternionRot quatRotation;
    // TODO: maybe have some value indicating how reliably it is matched / how accurate it might be
};

class RecognizedTargetList : public std::vector<RecognizedTarget>
{
public:
    //friend std::ostream& operator << (std::ostream& os, const RecognizedTargetList<T>& t);
    friend std::istream &operator>>(std::istream &is, RecognizedTargetList &t);

private:
    unsigned long time; // time (in ms) when the last of the source images was taken
};

/*******************************************************************
 *                          inlines                                *
 *******************************************************************/

//std::ostream& operator << (std::ostream& os, const RecognizedTarget& t);
//std::istream& operator >> (std::istream& is, RecognizedTarget& t);
//std::ostream& operator << (std::ostream& os, const RecognizedTargetList& m);
//std::istream& operator >> (std::istream& is, RecognizedTargetList& m);

} // namespace CGVOpticalTracking

#endif // RECOGNIZEDTARGETLIST_HEADER_INCLUDED
