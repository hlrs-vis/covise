/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVR_FILE_MANAGER_H
#define COVR_FILE_MANAGER_H

/*! \file
 \brief  load files and manipulate file names

 \author (C)
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date
 */

#include <util/coExport.h>
#include <list>
#include <limits.h>
#include <util/coStringMultiHash.h>
#include <map>
#include <osg/ref_ptr>
#include <osg/Texture2D>

#include <OpenVRUI/coUpdateManager.h>

namespace osg
{
class Node;
class Group;
class Texture2D;
}

namespace osgText
{
class Font;
}

namespace opencover
{

class coTUIFileBrowserButton;
class coVRIOReader;

typedef struct
{
    int (*loadUrl)(const char *filename, osg::Group *parent, const char *covise_key);
    int (*loadFile)(const char *filename, osg::Group *parent, const char *covise_key);
    int (*replaceFile)(const char *filename, osg::Group *parent, const char *covise_key);
    int (*unloadFile)(const char *filename, const char *covise_key);
    const char *extension;
} FileHandler;

class COVEREXPORT coVRFileManager : public vrui::coUpdateable
{
public:
    static coVRFileManager *instance();

    const char *findFileExt(const char *filename);

    // returns the full path for file
    const char *getName(const char *file);

    // load a OSG or VRML97 or other (via plugin) file
    osg::Node *loadFile(const char *file, coTUIFileBrowserButton *fb = NULL, osg::Group *parent = NULL, const char *covise_key = "");

    // replace the last loaded Performer or VRML97 file
    osg::Node *replaceFile(const char *file, coTUIFileBrowserButton *fb = NULL, osg::Group *parent = NULL, const char *covise_key = "");

    // reload the previously loaded /*VRML*/ file
    void reloadFile();

    // unload the previously loaded file
    void unloadFile();

    osg::Node *getLastModelNode();

    // load an icon file, looks in covise/icons/$LookAndFeel or covise/icons
    // returns NULL, if nothing found
    osg::Node *loadIcon(const char *filename);

    // loads a font
    // fontname can be NULL, which loads the default specified in the config (DroidSansFallbackFull.ttf if not in config), or "myfont.ttf"
    std::string getFontFile(const char *fontname);
    osg::ref_ptr<osgText::Font> loadFont(const char *fontname);

    // load an texture, looks in covise/icons/$LookAndFeel or covise/icons for filename.rgb
    // returns NULL, if nothing found
    osg::Texture2D *loadTexture(const char *texture);

    // tries to fopen() fileName
    // returns true if exists otherwise false
    bool fileExist(const char *fileName);

    // builds filename for icon files, looks in covise/icons/$LookAndFeel or covise/icons for filename.rgb
    const char *buildFileName(const char *);

    // register a loader, ... for a file type
    int registerFileHandler(const FileHandler *handler);
    int registerFileHandler(coVRIOReader *handler);

    // unregister a loader, ... for a file type
    int unregisterFileHandler(const FileHandler *handler);
    int unregisterFileHandler(coVRIOReader *handler);

    // get list of extensioins as required by a filebrowser
    std::string getFilterList();

    // get a loader for a file type, if available
    const FileHandler *getFileHandler(const char *extension);
    coVRIOReader *findIOHandler(const char *extension);

    // find a loader, load plugin if no loader available
    const FileHandler *findFileHandler(const char *extension);

    coTUIFileBrowserButton *getMatchingFileBrowserInstance(std::string keyFileName);

    bool IsDefFBSet();

    void SetDefaultFB(coTUIFileBrowserButton *fb);

    virtual bool update();

private:
    // Get the configured font style.
    int coLoadFontDefaultStyle();

    char *lastFileName;
    char *lastCovise_key;
    osg::Node *lastNode;

    typedef std::list<const FileHandler *> FileHandlerList;
    FileHandlerList fileHandlerList;

    typedef std::list<coVRIOReader *> IOReaderList;
    IOReaderList ioReaderList;

    covise::coStringMultiHash<osg::ref_ptr<osg::Texture2D> > textureList;
    std::map<std::string, coTUIFileBrowserButton *> fileFBMap;
    coTUIFileBrowserButton *mDefaultFB;

    struct IOReadOperation
    {
        IOReadOperation()
        {
            reader = 0;
            filename = "";
            group = 0;
        }
        coVRIOReader *reader;
        std::string filename;
        osg::Group *group;
    };

    typedef std::map<std::string, std::list<IOReadOperation> > ReadOperations;
    ReadOperations readOperations;

    coVRFileManager();
    ~coVRFileManager();
};
}
#endif
