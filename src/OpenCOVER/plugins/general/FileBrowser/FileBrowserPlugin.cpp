#include "FileBrowserPlugin.h"
#include <osg/MatrixTransform>
#include <osg/Matrix>
#include <OpenVRUI/coFlatPanelGeometry.h>
#include <OpenVRUI/coInteraction.h>
#include <OpenVRUI/coMenu.h>
#include <config/CoviseConfig.h>
#include <algorithm>

using namespace std;
using namespace cui;
using namespace osg;
COVERPLUGIN(FileBrowserPlugin)

FileBrowserPlugin::FileBrowserPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    _isInit = false;
}

FileBrowserPlugin::~FileBrowserPlugin()
{

}

bool FileBrowserPlugin::init()
{

    scale = 1.0;

    handle = new coPopupHandle(string("FileBrowser"));
    frame = new coFrame();
    panel = new SizedPanel(new coFlatPanelGeometry(coUIElement::BLACK));

    _interaction = new cui::Interaction(cover->getScene(), cover->getObjectsRoot());
    _interaction->_wandR->setCursorType(cui::InputDevice::INVISIBLE);
    _interaction->_head->setCursorType(cui::InputDevice::NONE);
    
    if (const char *covisedir = getenv("COVISEDIR"))
    {
        home = string(covisedir);
    }
    else
    {
      cerr << "COVISEDIR not set." << endl;
      home = "/";
    }

    browser = new cui::coFileBrowser(home, _interaction);
    browser->addListener(this);

    root = new MatrixTransform();
    Matrix m;

    m.makeTranslate(0.0, browser->getHeight(), 5.0);
    root->setMatrix(m);

    root->addChild(browser->getNode());

    //cerr << "Width: " << browser->getWidth() << " Height: " << browser->getHeight() << endl;

    dynamic_cast<MatrixTransform *>(dynamic_cast<OSGVruiTransformNode *>(panel->getDCS())->getNodePtr())->addChild(root);
    pWidth = browser->getWidth();
    pHeight = browser->getHeight(); 
    panel->setSize(pWidth, pHeight);

    handle->addElement(frame);
    frame->addElement(panel);
    handle->setVisible(false);

    browser->_handle = handle;
 
    _fbButton = new coButtonMenuItem("File Browser");
    cover->getMenu()->add(_fbButton);
    _fbButton->setMenuListener(this);

    _isInit = true;
    return true;
}

void FileBrowserPlugin::menuEvent(coMenuItem* menuItem)
{
    if(menuItem == _fbButton)
    {
        browser->changeDir(home);
        handle->setVisible(true);
    }
}

void FileBrowserPlugin::preFrame()
{
    if(_storedMessages.size() > 0)
    {
	for(int i = 0; i < _storedMessages.size(); i++)
	{
        message(0, _storedMessages[i].first, 0, (void *)_storedMessages[i].second);
	    delete _storedMessages[i].second;
	}
	_storedMessages.clear();
    }
    if(pWidth != browser->getWidth() || pHeight != browser->getHeight())
    {
        pWidth = browser->getWidth();
        pHeight = browser->getHeight();
        panel->setSize(pWidth, pHeight);
        Matrix m;
    
        m.makeTranslate(0.0, pHeight, 5.0);
        root->setMatrix(m);
    }
    updateOSGCaveUI(); 
}

void FileBrowserPlugin::updateOSGCaveUI()
{
  static int lastStatus=0;
  static int lastButton=0;
  static int lastState=0;
  int button=0, state=0;
  int newStatus;

  _interaction->_wandR->setI2W(cover->getPointerMat());
  _interaction->_head->setI2W(cover->getViewerMat());

  newStatus = cover->getPointerButton()->getState();
  switch(newStatus)
  {
    case 0:   button = 0; state = 0; break;
    case vruiButtons::ACTION_BUTTON:   button = 0; state = 1; break;
    case vruiButtons::XFORM_BUTTON:  button = 1; state = 1; break;
    case vruiButtons::DRIVE_BUTTON: button = 2; state = 1; break;
    default: break;
  }
  if (lastStatus != newStatus)
  {
    if (lastState==1)
    {
      _interaction->_wandR->buttonStateChanged(lastButton, 0);
    }
    if (state==1)
    {
      _interaction->_wandR->buttonStateChanged(button, 1);
    }
  }

  lastStatus = newStatus;
  lastButton = button;
  lastState = state;

  _interaction->action();

}

void FileBrowserPlugin::message(int toWhom, int type, int, const void * buf)
{    
    struct BrowserMessage * bm = (BrowserMessage *)buf;
    if(!_isInit)
    {

	BrowserMessage * tempbp = new BrowserMessage;
	*tempbp = * bm;
	_storedMessages.push_back(pair< int, BrowserMessage * >(type, tempbp));

	return;
    }

    switch((BrowserMessageFlag)type)
    {
        case REGISTER_EXT:
        {
          string temp = bm->ext;
          std::transform(temp.begin(), temp.end(), temp.begin(), ::tolower);
          if(_extmap.find(temp) == _extmap.end())
          {
              _extmap[temp] = bm->plugin;
              browser->addExt(temp, bm->preview);
          }
          bm->plugin->_fileBrowser = browser;
          break;
        }
        case RELEASE_EXT:
        {
          string temp = bm->ext;
          std::transform(temp.begin(), temp.end(), temp.begin(), ::tolower);
          if(_extmap.find(temp) != _extmap.end())
          {
              if(_extmap[temp] == bm->plugin)
              {
                  _extmap.erase(temp);
                  browser->removeExt(temp);
              }
          }
          break;
        }
    }
}

bool FileBrowserPlugin::fileBrowserEvent(coFileBrowser* fb, string & file, string & ext, int button, int state)
{
    //cerr << "Getting Browser Event.\n";
    _extmap[ext]->fileBrowserEvent(fb, file, ext, button, state);
    return true;
}
