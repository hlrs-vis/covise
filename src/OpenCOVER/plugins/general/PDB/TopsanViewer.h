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
#include "TabbedDialogPanel.h"
#include <cover/coVRMSController.h>

class TopsanViewer
{
public:
    struct topsanData
    {
        std::string title;
        std::string site;
        std::string id;
        std::string name;
        std::string source;
        std::string refid;
        std::string weight;
        std::string residues;
        std::string isoelec;
        std::string seq;
        std::string ligands;
        std::string metals;
        std::string summary;
    };

    struct nontopsanData
    {
        std::string title;
        std::string source;
        std::string author;
        std::string resolution;
    };

    TopsanViewer();
    ~TopsanViewer();
    void load(std::string name);
    void set(std::string name);
    void remove(std::string name);
    void clear();
    void setVisible(bool vis);

protected:
    TopsanViewer::nontopsanData *parsePDB(std::string name);

    coPopupHandle *topsanHandle;
    coFrame *topsanFrame;
    TabbedDialogPanel *topsanPanel;
    bool visible;
    std::string _name, _tsDir;
    std::map<std::string, std::pair<TopsanViewer::topsanData *, int> > loadedData;
    std::map<std::string, std::pair<struct nontopsanData *, int> > nottopsanData;
    std::map<std::string, std::vector<std::string> > _imagemap;
};
