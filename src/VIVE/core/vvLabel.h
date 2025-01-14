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

#include <vsg/maths/vec3.h>
#include <vsg/maths/vec4.h>

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

#include "vvBillboard.h"
#include <util/coExport.h>
namespace vive
{
class VVCORE_EXPORT vvLabel
{

private:
    vsg::MatrixTransform *posTransform;
    vsg::ref_ptr<vvBillboard> billboard;
    osg::Geode *geode;
    osgText::Text *text;
    float offset;
    vsg::vec3Array *lc, *qc;
    vsg::Node *lineGeoset, *quadGeoset;
    osg::Geode *label;
    vsg::vec3 position;
    bool keepPositionInScene;
    float distanceFromCamera;
    bool moveToCam;
    bool depthScale;

    vsg::vec3 moveToCamera(const vsg::vec3 &point, float distance);

public:
    // creates scene graph
    //
    //         scene
    //           |
    //         posTransform
    //           |
    //      vvBillboard
    //       |      |
    //    coText lineGeode
    //           |      |
    //      lineGeoset quadGeoset
    vvLabel(const char *name, float fontsize, float lineLen, vsg::vec4 fgcolor, vsg::vec4 bgcolor);
    ~vvLabel();

    // after construction you may decide to attach posTransform to
    // a given vsg::Group *anchor instead of vv->getScene()
    // but manipulate the position of the label as if
    // it were attached to the scene

    //!!!!!!!!!!!!
    // Does not work correctly!!!!!
    //!!!!!!!!!!!!

    void reAttachTo(vsg::Group *anchor);

    // position in world coordinates
    void setPosition(const vsg::vec3 &pos);
    void setPositionInScene(const vsg::vec3 &pos);

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

    void setRotMode(vvBillboard::RotationMode mode);
    //RotationMode getMode() const;

    // update the linelen
    void setLineLen(float lineLen);

    // update the label string
    void setString(const char *name);

    void setFGColor(vsg::vec4 fgc);

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

    void setDepthScale(bool s) { depthScale = s; }
};
}
#endif
