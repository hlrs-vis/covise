/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
// Description:   PDB Sequence Display
//
// Author:        Sendhil Panchadsaram (sendhilp@gmail.com)
//
// Creation Date: 8/15/2006
//
// **************************************************************************

#ifndef _PDB_SEQ_DISPLAY_PLUGIN_H_
#define _PDB_SEQ_DISPLAY_PLUGIN_H_

#include <OpenVRUI/coMenu.h>
#include <string>
#include <vector>
#include <osg/Group>
#include <osg/Switch>
#include <osg/AutoTransform>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/BlendFunc>
#include <osg/BlendEquation>
#include <osg/AlphaFunc>
#include "CChain.h"
#include <sys/stat.h>
#include <cover/coVRPlugin.h>

static const std::string WGET("wget -np ");
static const std::string PDB_EXT(".pdb");

namespace vrui
{
class coSlider;
class coFrame;
class coPanel;
class coButtonMenuItem;
class coSubMenuItem;
class coCheckboxMenuItem;
class coPopupHandle;
class coButton;
class coPotiItem;
class coLabelItem;
}

namespace opencover
{
class coVRPlugin;
class SystemCover;
}

using namespace vrui;
using namespace opencover;

struct MessageData
{
    float x;
    float y;
    float z;
    bool on;
    char filename[5];
};

class PDBSequenceDisplay : public coVRPlugin, public coMenuListener, public coButtonActor, public coValuePotiActor
{
    coButtonMenuItem *PDBSequenceDisplayMenuItem;
    coPanel *panel;
    coFrame *frame;
    coPopupHandle *handle;
    coLabel **labelChar;
    coLabel **labelNum;
    coLabel *labelSequence;
    coLabel *labelTitle;
    coLabel *labelChain;
    coLabel *labelChange;
    coButton *testButton;
    //	coButton** buttonArray;
    coButton *tempButton;
    coButton *upButton;
    coButton *downButton;
    coButton *leftButton;
    coButton *rightButton;
    coButton *firstButton;
    coButton *lastButton;
    coButton *showMarkerButton;
    coVRPlugin *modulePointer;

    std::string curChain;
    std::string CacheDirectory;
    int curChainPos;
    int curChainEndPos;
    int curChainNum;
    std::vector<CChain> overallChain;
    CChain oneChain;
    CProtein myProtein;

    // has name of current in memory protein data
    std::string currentProtein;

    //Menus
    coSubMenuItem *selectMenu;
    coCheckboxMenuItem *sequenceDisplayButton;
    coRowMenu *selectRow;

    // switch node for attaching vrml scenes
    osg::ref_ptr<osg::Switch> mainNode; //new

    void menuEvent(coMenuItem *);

private:
    void ChangeLabel(int);
    void ChangeChain(int);
    void GoFirst();
    void GoLast();
    void ChangeProtein(std::string);
    void SendCoordinates();
    void GotoChainAndPos(std::string chain, int pos);

protected:
    void potiValueChanged(float oldvalue, float newvalue, coValuePoti *poti, int context);

public:
    PDBSequenceDisplay();
    ~PDBSequenceDisplay();
    bool init();
    void buttonEvent(coButton *);
    void preFrame();
    void message(int type, int, const void *buf);
    int FileExists(const char *filename);
};

#endif

// EOF
