/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef ORTHOGRAPHICSNAPSHOT_H
#define ORTHOGRAPHICSNAPSHOT_H
/****************************************************************************\
 **                                                            (C)2010 HLRS  **
 **                                                                          **
 ** Description: OrthographicSnapShot Plugin                                 **
 **                                                                          **
 **                                                                          **
 ** Author: Frank Naegele                                                    **
 **                                                                          **
 ** History:                                                                 **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include <cover/coVRPluginSupport.h>
#include <cover/coTabletUI.h>

#include <osg/Camera>

#include <string>

namespace opencover
{
class coVRSceneHandler;
class coVRSceneView;
}

using namespace covise;
using namespace opencover;

class OrthographicSnapShot : public coVRPlugin, public coTUIListener
{
    friend class DrawCallback;

    //!##########################//
    //! Functions                //
    //!##########################//

public:
    OrthographicSnapShot();
    virtual ~OrthographicSnapShot();

    bool init();

    void preFrame();

protected:
    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual void tabletReleaseEvent(coTUIElement *tUIItem);

private:
    void initUI();
    void deleteUI();

    void prepareSnapshot();
    inline void cameraCallbackExit() const;

    std::string suggestFileName(std::string suggestedDir);

    //!##########################//
    //! Members                  //
    //!##########################//

private:
    mutable bool removeCamera_;
    mutable bool doSnap_;
    bool hijackCam_;

    bool createScreenshot_;
    bool createHeightmap_;

    std::string filename_;
    std::string heightFilename_;

    double xPos_;
    double yPos_;
    double width_;
    double height_;

    double scale_;

    // osg //
    //
    osg::ref_ptr<osg::Image> image_;
    osg::ref_ptr<osg::Camera::DrawCallback> drawCallback_;
    osg::ref_ptr<osg::Camera> pBufferCamera_;

    // TUI //
    //
    coTUITab *tuiSnapTab;

    coTUIButton *tuiSnapButton;
    coTUIToggleButton *tuiHijackButton;

    coTUIEditIntField *tuiResolutionX;
    coTUIEditIntField *tuiResolutionY;

    coTUIEditFloatField *tuiXPos;
    coTUIEditFloatField *tuiYPos;
    coTUIEditFloatField *tuiWidth;
    coTUIEditFloatField *tuiHeight;

    coTUIEditField *tuiFileName;
    coTUIEditField *tuiHeightmapFileName;

    coTUIToggleButton *tuiToggleSnapshot;
    coTUIToggleButton *tuiToggleHeightmap;
};

#endif
