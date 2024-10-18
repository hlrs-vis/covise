/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CSV_NODE_PLUGIN_H
#define _CSV_NODE_PLUGIN_H

#include <util/common.h>

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>

#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>
#include <util/byteswap.h>
#include <net/covise_connect.h>

#include <util/coTypes.h>

#include <vrml97/vrml/config.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/coEventQueue.h>
#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/System.h>
#include <vrml97/vrml/Viewer.h>
#include <vrml97/vrml/VrmlScene.h>
#include <vrml97/vrml/VrmlNamespace.h>
#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFBool.h>
#include <vrml97/vrml/VrmlMFFloat.h>
#include <vrml97/vrml/VrmlSFInt.h>
#include <vrml97/vrml/VrmlMFInt.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>
#include <cover/ui/Owner.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Label.h>
#include <cover/coTabletUI.h>

using namespace vrml;
using namespace opencover;
using namespace covise;
class PLUGINEXPORT LabelInfo
{
public:
    LabelInfo(const std::string &l, int64_t sta, int64_t sto) { label = l; start = sta; stop = sto; };
    std::string label;
    int64_t start;
    int64_t stop;
};

class PLUGINEXPORT gpsData
{
public:
    uint64_t timestamp;
    float lat;
    float lon;
    float alt;
    float velocity;
};

class PLUGINEXPORT VrmlNodeCSV : public VrmlNodeChild
{
public:
    // Define the fields of CSV nodes
    static void initFields(VrmlNodeCSV *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeCSV(VrmlScene *scene = 0);
    VrmlNodeCSV(const VrmlNodeCSV &n);

    virtual VrmlNodeCSV *toCSV() const;

    void eventIn(double timeStamp, const char *eventName,
                 const VrmlField *fieldValue);

    virtual void render(Viewer *);

    bool isEnabled()
    {
        return d_enabled.get();
    }
    int numFloats;
    float *floatValues;
    std::vector<gpsData> path;

private:
    // Fields
    VrmlSFBool d_enabled;
    VrmlSFInt d_numColumns;
    VrmlSFInt d_numRows;
    VrmlSFInt d_row;
    VrmlSFString d_fileName;
    VrmlSFString d_labelFileName;
    VrmlSFString d_gpsFileName;

    // eventOuts
    VrmlMFFloat d_floats;
    std::vector<float *> rows;
    bool loadFile(const std::string &fileName);
    bool loadLabelFile(const std::string &fileName);
    bool loadGPSFile(const std::string &fileName);
    bool changedFile;
    bool changedLabelFile;
    std::string MenuLabel;

    int numColumns = -1;
    int RowCount = 0;
};

class CSVPlugin : public coVRPlugin, public opencover::ui::Owner
{
public:
    CSVPlugin();
    ~CSVPlugin();
    bool init();
    static CSVPlugin *plugin;
    void createMenu(const std::string &label);
    ui::Menu *labelMenu;
    ui::Label *currentLabel;
    uint64_t currentLabelNumber;
    void updateLabel(int64_t ts);
    std::vector<LabelInfo> labels;
    virtual void setTimestep(int t);

    bool update();

    coTUIEarthMap *tuiEarthMap;
    coTUITab *CSVTab;
    VrmlNodeCSV* CSVNode;

private:
};
#endif
