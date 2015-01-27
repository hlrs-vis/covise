/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
// Description:   CaveVOX
//
// Author:        Jurgen Schulze (jschulze@ucsd.edu)
//
// Creation Date: 2005-12-14
//
// **************************************************************************

#ifndef _CAVE_VOX_PLUGIN_H_
#define _CAVE_VOX_PLUGIN_H_

#include "coMenu.h"
class coFrame;
class coPanel;
class coButtonMenuItem;
class coPopupHandle;
class coButton;
class coPotiItem;
class coLabelItem;

/** Plugin to explore volume data sets in VR-environments.
  @author Jurgen Schulze (jschulze@ucsd.edu)
*/
class CaveVOX : public coMenuListener, public coButtonActor, public coValuePotiActor
{
    coButtonMenuItem *caveVOXMenuItem;
    coPanel *panel;
    coFrame *frame;
    coPopupHandle *handle;

    void menuEvent(coMenuItem *);

protected:
    void potiValueChanged(float oldvalue, float newvalue, coValuePoti *poti, int context);

public:
    CaveVOX(coVRPlugin *m);
    ~CaveVOX();
    void buttonEvent(coButton *);
    void preFrame();
};

#endif

// EOF
