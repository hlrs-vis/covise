#include <cover/coVRPlugin.h>
#include "coFileBrowser.h"
#include <osgcaveui/CUI.h>
#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coPopupHandle.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <osg/MatrixTransform>
#include <map>

#include "SizedPanel.h"

using namespace vrui;
using namespace opencover;

class FileBrowserPlugin : public coVRPlugin, public cui::coFileBrowserListener, public coMenuListener
{
  public:
    FileBrowserPlugin();
    ~FileBrowserPlugin() override;
    bool init() override;
    void menuEvent(coMenuItem*) override;
    void preFrame() override;

    void updateOSGCaveUI();

    bool fileBrowserEvent(cui::coFileBrowser*, std::string&, std::string&, int, int) override;

    void message(int, int, int, const void*) override;

    coPopupHandle * handle;
    coFrame * frame;
    SizedPanel * panel;

    float scale, pWidth, pHeight;

    coButtonMenuItem * _fbButton;
    std::map<std::string, cui::coFileBrowserListener *> _extmap;
    std::string home;
    osg::MatrixTransform * root;

    cui::coFileBrowser * browser;
    cui::Interaction * _interaction;

    std::vector<std::pair< int, BrowserMessage * > > _storedMessages;

    bool _isInit;
};
