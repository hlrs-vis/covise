/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_FILEBROWSER_H_
#define _CUI_FILEBROWSER_H_

// C++:
#include <string>
#include <vector>

// Local:
#include "Widget.h"
#include "Panel.h"
#include "Button.h"

namespace cui
{

class FileBrowserListener;

class CUIEXPORT FileBrowser : public CardListener
{
public:
    FileBrowser(std::string &, std::string &, cui::Interaction *);
    virtual ~FileBrowser();
    bool cardButtonEvent(Card *, int, int);
    bool cardCursorUpdate(Card *, InputDevice *);
    std::string getFileName();
    virtual void addListener(FileBrowserListener *);
    void setVisible(bool);

protected:
    std::string _browserDirectory;
    std::vector<std::string> _fileNames;
    std::vector<std::string> _folderNames;
    std::vector<cui::Button *> _fileButtons;
    std::vector<cui::Button *> _folderButtons;
    std::string _selectedFile;
    std::string _extension;
    cui::Interaction *_interaction;
    cui::Panel *_browserPanel;
    char _delim; // directory delimiter (/ or \)
    std::list<FileBrowserListener *> _listeners;

    void createFolderIcons();
    void createFileIcons();
    bool makeFileList(std::string &, std::vector<std::string> &);
    void sortAlphabetically(std::vector<std::string> &);
};

class CUIEXPORT FileBrowserListener
{
public:
    virtual ~FileBrowserListener()
    {
    }
    /** @param file browser
          @param file name (incl. path)
          @param button
          @param state
          @return true if event code handles event completely so it shouldn't do anything else
      */
    virtual bool fileBrowserEvent(FileBrowser *, std::string &, int, int) = 0;
};
}
#endif

// EOF
