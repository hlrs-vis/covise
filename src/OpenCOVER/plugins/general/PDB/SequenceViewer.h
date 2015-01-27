/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <config/CoviseConfig.h>
#include <cstdio>
#include <OpenVRUI/coPopupHandle.h>
#include <OpenVRUI/coRectButtonGeometry.h>
#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coFlatPanelGeometry.h>
#include <OpenVRUI/coPopupHandle.h>
#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coTextButtonGeometry.h>
#include <cover/coVRMSController.h>
#include <osg/Matrix>
#include <osg/Vec4>
#include "TabbedDialogPanel.h"
#include <cover/coVRPlugin.h>

#define MOVE_MARK 164

struct SequenceMessage
{
    float x;
    float y;
    float z;
    bool on;
    char filename[5];
};

class SequenceViewer : public coButtonActor
{
public:
    struct markpoint
    {
        float x;
        float y;
        float z;
    };

    SequenceViewer(coVRPlugin *p);
    ~SequenceViewer();
    void buttonEvent(coButton *);
    void menuUpdate(coButton *);
    void load(std::string name);
    void set(std::string name);
    void setLoc(float x, float y, float z);
    void remove(std::string name);
    void clear();
    void clearDisplay();
    void setVisible(bool vis);

    std::pair<osg::Matrix, osg::Matrix> &getInverse(std::string name);

protected:
    void setChain(int chain);
    coPopupHandle *handle;
    coFrame *frame;
    TabbedDialogPanel *panel;
    std::string pdbpath, _name;
    std::map<std::string, std::vector<std::vector<std::pair<struct markpoint, std::string> > > > locations;
    std::map<std::string, std::vector<std::vector<coToggleButton *> > > buttons;
    std::map<std::string, std::pair<osg::Matrix, osg::Matrix> > inversemap;
    std::map<std::string, int> pcount;
    std::map<std::string, std::vector<coToggleButton *> > cbuttons;
    float bwidth, bheight, pwidth;
    int visiblechain, selectedc, selecteda;
    float offset;
    float space;
    coVRPlugin *PDBptr;
};
