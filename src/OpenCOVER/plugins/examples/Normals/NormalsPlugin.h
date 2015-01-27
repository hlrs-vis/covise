/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _NORMALS_PLUGIN_H
#define _NORMALS_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2001 ZAIK  **
 **                                                                          **
 ** Description: Normals Plugin (draw normals)                               **
 **                                                                          **
 ** Author: Martin Aumueller (aumueller@uni-koeln.de)                        **
 **                                                                          **
\****************************************************************************/

#include <cover/coVRPluginSupport.h>
#include <cover/coVRTui.h>

using namespace covise;
using namespace opencover;

#include "Normals.h"

class NormalsPlugin : public coVRPlugin, public coTUIListener
{
public:
    NormalsPlugin();
    ~NormalsPlugin();
    bool init();

    // this will be called on every key press/release
    void keyEvent(int type, int keySym, int mod);
    void tabletEvent(coTUIElement *tUIItem);

private:
    void applyState();

    osg::ref_ptr<osgUtil::VertexNormals> vertexNormals;
    osg::ref_ptr<osgUtil::SurfaceNormals> faceNormals;

    enum NormalsState
    {
        NORM_NONE = 0,
        NORM_FACE = 1,
        NORM_VERTEX = 2,
        NORM_ALL = NORM_FACE | NORM_VERTEX
    };
    NormalsState normalsState;

    coTUITab *tuiTab;
    coTUIToggleButton *faceNorm, *vertNorm;
    coTUILabel *scaleLabel;
    coTUIFloatSlider *scaleSlider;
    float scaleValue;
    coTUIButton *update;
};
#endif
