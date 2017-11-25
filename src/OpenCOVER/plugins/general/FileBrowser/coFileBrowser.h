#ifndef _CO_FILEBROWSER_H_
#define _CO_FILEBROWSER_H_

// C++:
#include <string>
#include <vector>
#include <map>

// Local:
#include <osgcaveui/Widget.h>
#include <osgcaveui/Panel.h>
#include <osgcaveui/Button.h>

#include <OpenVRUI/coPopupHandle.h>

using namespace vrui;
using namespace opencover;

namespace cui
{

class coFileBrowserListener;

class coFileBrowser : public CardListener
{
  public:
    coFileBrowser(std::string&, cui::Interaction*);
    virtual ~coFileBrowser();
    bool cardButtonEvent(Card*, int, int);
    bool cardCursorUpdate(Card*, InputDevice*);
    std::string getFileName();
    virtual void addListener(coFileBrowserListener*);
    void setVisible(bool);
    osg::Node * getNode();
    float getWidth();
    float getHeight();
    void addExt(std::string ext, bool preview);
    void removeExt(std::string ext);
    void changeDir(std::string dir);

    vrui::coPopupHandle * _handle;
 
  protected:
    std::string _browserDirectory;
    std::vector<std::string> _fileNames;
    std::vector<std::string> _folderNames;
    std::vector<cui::Button*> _fileButtons;
    std::vector<cui::Button*> _folderButtons;
    std::string _selectedFile;
    std::map<std::string, bool> _extensions;
    cui::Interaction* _interaction;
    cui::Panel* _browserPanel;
    
    osg::Node * root;

    char _delim;    // directory delimiter (/ or \)
    std::list<coFileBrowserListener*> _listeners;
  
    void createFolderIcons();
    void createFileIcons();
    bool makeFileList(std::string&, std::vector<std::string>&);
    void removePreviews();
    void sortAlphabetically(std::vector<std::string>&);
};

class coFileBrowserListener
{
    public:
      coFileBrowserListener() { _fileBrowser = NULL; }
      virtual ~coFileBrowserListener() {}
      void setFileBrowserVisible(bool v)
      {
        if(_fileBrowser != NULL)
        {
          _fileBrowser->setVisible(v);
        }
      }
      void changeFileBrowserDir(std::string s)
      {
        if(_fileBrowser != NULL)
        {
          _fileBrowser->changeDir(s);
        }
      }
      virtual bool fileBrowserEvent(coFileBrowser*, std::string&, std::string&, int, int) = 0;
      coFileBrowser * _fileBrowser;
};

}

enum BrowserMessageFlag
{
    REGISTER_EXT,
    RELEASE_EXT
};

struct BrowserMessage
{
    cui::coFileBrowserListener * plugin;
    char ext[25];
    bool preview;
};


#endif

// EOF
