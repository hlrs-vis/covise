/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
// Description:   ModelViewer
//
// Author:        Jurgen Schulze (jschulze@ucsd.edu)
//
// Creation Date: 2006-09-17
//
// **************************************************************************

#ifndef _MODEL_VIEWER_PLUGIN_H_
#define _MODEL_VIEWER_PLUGIN_H_

#include <OpenVRUI/coMenu.h>

namespace vrui
{
class coFrame;
class coPanel;
class coButtonMenuItem;
class coPopupHandle;
class coButton;
class coPotiItem;
class coLabelItem;
}

using namespace vrui;
using namespace opencover;

/** Plugin to load any data file OSG supports.
  @author Jurgen Schulze (jschulze@ucsd.edu)
*/
class ModelViewer : public coVRPlugin, public coMenuListener, public coButtonActor, public coValuePotiActor
{
    coButtonMenuItem *modelViewerMenuItem;
    coPanel *panel;
    coFrame *frame;
    coPopupHandle *handle;

    void menuEvent(coMenuItem *);

protected:
    void potiValueChanged(float oldvalue, float newvalue, coValuePoti *poti, int context);

public:
    ModelViewer();
    ~ModelViewer();
    bool init();
    void buttonEvent(coButton *);
    void preFrame();
};

#endif

// EOF
