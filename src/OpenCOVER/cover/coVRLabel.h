/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_VR_LABEL
#define CO_VR_LABEL

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

#include "coBillboard.h"
#include <util/coExport.h>
namespace opencover
{
class COVEREXPORT coVRLabel
{

private:
    osg::MatrixTransform *posTransform;
    osg::ref_ptr<coBillboard> billboard;
    osg::Geode *geode;
    osgText::Text *text;
    float offset;
    osg::Vec3Array *lc, *qc;
    osg::Geometry *lineGeoset, *quadGeoset;
    osg::Geode *label;
    osg::Vec3 position;
    bool keepPositionInScene;
    float distanceFromCamera;
    bool moveToCam;

    osg::Vec3f moveToCamera(const osg::Vec3f &point, float distance);

public:
    // creates scene graph
    //
    //         scene
    //           |
    //         posTransform
    //           |
    //      coBillboard
    //       |      |
    //    coText lineGeode
    //           |      |
    //      lineGeoset quadGeoset
    coVRLabel(const char *name, float fontsize, float lineLen, osg::Vec4 fgcolor, osg::Vec4 bgcolor);
    ~coVRLabel();

    // after construction you may decide to attach posTransform to
    // a given osg::Group *anchor instead of cover->getScene()
    // but manipulate the position of the label as if
    // it were attached to the scene

    //!!!!!!!!!!!!
    // Does not work correctly!!!!!
    //!!!!!!!!!!!!

    void reAttachTo(osg::Group *anchor);

    // position in world coordinates
    void setPosition(const osg::Vec3 &pos);
    void setPositionInScene(const osg::Vec3 &pos);

    // move label towards camera to keep above geometry
    void keepDistanceFromCamera(bool enable, float distanceFromCamera = 50.0f);

    // rotation mode
    /*  enum RotationMode
        {
            STOP_ROT = 0,
            AXIAL_ROT,
            POINT_ROT_EYE,
            POINT_ROT_WORLD
        };*/

    void setRotMode(coBillboard::RotationMode mode);
    //RotationMode getMode() const;

    // update the linelen
    void setLineLen(float lineLen);

    // update the label string
    void setString(const char *name);

    void setFGColor(osg::Vec4 fgc);

    // show label
    void show();

    // hide label
    void hide();

    // show line
    void showLine();

    // hide line
    void hideLine();
    // RotationMode _mode;

    void update();
};
}
#endif
