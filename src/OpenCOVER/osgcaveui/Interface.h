/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_INTERFACE_H_
#define _CUI_INTERFACE_H_

// OSG:
#include <osg/ref_ptr>
#include <osgUtil/SceneView>

#include "Widget.h"

namespace cui
{
/** This class interfaces the rendering framework with osgCaveUI.
    There are three pre-defined scenegraph nodes to connect geometry with.
    They are arranged in the following way:
    <pre>

             worldRoot
                 |
             roomRoot
            |     |         \          \         \          \ 
     headRoot   leftRoot  rightRoot  frontRoot  floorRoot fishRoot
    </pre>

  The following variables abbreviate coordinate systems:
  w = world
  r = room
  h = head
  m = mouse (=wand=hand)

  Define DOLLAR_G to use with Brown University's $G VR framework. This will not work right
  away but at least be a start.

  @author Jurgen Schulze (jschulze@ucsd.edu)
  */
class CUIEXPORT Interface
{
public:
    enum DisplayType ///< type of (virtual) environment rendered in
    {
        CAVE,
        FISHTANK,
        DESKTOP
    };
    osg::Matrix _w2r; ///< world to room coordinates
    osg::Matrix _w2h; ///< world to head coordinates
    osg::Matrix _r2w; ///< room to world coordinates
    osg::Matrix _r2h; ///< room to head coordinates
    osg::Matrix _h2r; ///< head to room coordinates
    osg::Matrix _h2w; ///< head to world coordinates

    Interface();
    virtual ~Interface();
    virtual osg::ref_ptr<osg::ClearNode> getWorldRoot();
    virtual osg::ref_ptr<osg::MatrixTransform> getRoomRoot();
    virtual osg::ref_ptr<osg::MatrixTransform> getHeadRoot();
    virtual osg::ref_ptr<osg::MatrixTransform> getLeftRoot();
    virtual osg::ref_ptr<osg::MatrixTransform> getRightRoot();
    virtual osg::ref_ptr<osg::MatrixTransform> getRightRaveRoot();
    virtual osg::ref_ptr<osg::MatrixTransform> getFrontRoot();
    virtual osg::ref_ptr<osg::MatrixTransform> getFloorRoot();
    virtual osg::ref_ptr<osg::MatrixTransform> getFishRoot();
    virtual osg::ref_ptr<osgUtil::SceneView> getSceneView();
    virtual void addWorldChild(osg::Node *);
    virtual void addRoomChild(osg::Node *);
    virtual void addHeadChild(osg::Node *);
    virtual void addLeftChild(osg::Node *);
    virtual void addRightChild(osg::Node *);
    virtual void addRightRaveChild(osg::Node *);
    virtual void addFrontChild(osg::Node *);
    virtual void addFloorChild(osg::Node *);
    virtual void addFishChild(osg::Node *);
    virtual void removeWorldChild(osg::Node *);
    virtual void removeRoomChild(osg::Node *);
    virtual void removeHeadChild(osg::Node *);
    virtual void removeLeftChild(osg::Node *);
    virtual void removeRightChild(osg::Node *);
    virtual void removeRightRaveChild(osg::Node *);
    virtual void removeFrontChild(osg::Node *);
    virtual void removeFloorChild(osg::Node *);
    virtual void removeFishChild(osg::Node *);
    virtual void setSmallFeatureCulling(bool newState);
    virtual void draw();
    virtual DisplayType getDisplayType();
    virtual void setUseZbuffer(int);
#ifdef DOLLAR_G
    static void convertGlue2OSG(Wtransf &, osg::Matrix &);
    static void convertOSG2Glue(osg::Matrix &, Wtransf &);
#endif
    static bool isChild(osg::Node *, osg::Node *);
    void setOSGLibraryPath();

protected:
    osg::ref_ptr<osgUtil::SceneView> sceneview;
    osg::ref_ptr<osg::ClearNode> worldRoot; ///< root node of virtual world
    osg::ref_ptr<osg::MatrixTransform> roomRoot; ///< root node of real world (eg, Cave)
    osg::ref_ptr<osg::MatrixTransform> headRoot; ///< root node of head coordinate system (center between eyes)
    osg::ref_ptr<osg::MatrixTransform> leftRoot; ///< root node of left Cave wall
    ///< root node of right Cave wall
    osg::ref_ptr<osg::MatrixTransform> rightRoot;
    ///< root node of front Cave wall
    osg::ref_ptr<osg::MatrixTransform> frontRoot;
    ///< root node of Cave floor
    osg::ref_ptr<osg::MatrixTransform> floorRoot;
    osg::ref_ptr<osg::MatrixTransform> fishRoot; ///< root node of right monitor of Fishtank
    ///< root node of right Rave wall
    osg::ref_ptr<osg::MatrixTransform> rightRaveRoot;
    osg::Matrix filterTrackerData(osg::Matrix &);
    DisplayType _display;
    int _zbuf_flag;
};
}
#endif
