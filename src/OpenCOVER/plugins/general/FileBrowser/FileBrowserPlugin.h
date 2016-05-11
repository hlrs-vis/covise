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
    ~FileBrowserPlugin();
    bool init();
    void menuEvent(coMenuItem*);
    void preFrame();

    void updateOSGCaveUI();

    virtual bool fileBrowserEvent(cui::coFileBrowser*, std::string&, std::string&, int, int);

    void message(int, int, const void*);

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
