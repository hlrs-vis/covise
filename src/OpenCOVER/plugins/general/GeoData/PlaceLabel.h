/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PLACELABEL_H
#define PLACELABEL_H

/*! \file
 \brief  create a billboard banner like label

 \author Daniela Rainer
 \author (C) 2001
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date
 */

// label always faces viewer
// line is z axis
//
//    | TEXT
//    |
//    |

#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Array>

#include <util/coExport.h>
#include <cover/coBillboard.h>

namespace osg
{
class MatrixTransform;
class Geode;
class Geometry;
};

namespace osgText
{
class Text;
};

class PlaceLabel
{
public:
    PlaceLabel(const std::string &value, const osg::Vec3 &position, osg::ref_ptr<osg::Group> parent, int size = 0);

    void reposition();

private:
    std::string value;
    osg::Vec3 position;
    int size = 0;

    osg::ref_ptr<osg::MatrixTransform> transform;

    osg::ref_ptr<opencover::coBillboard> billboard;
    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<osgText::Text> text;
    osg::ref_ptr<osg::Geometry> lineGeometry;

    float lineLength = 200.f; // meters
    float fontSize = 20.f; // meters
    // bool keepPositionInScene = true;
    // bool moveToCam = false;
    // bool depthScale = false;
};
#endif
