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
#include <map>
#include <memory>
#include <vsg/core/ref_ptr.h>
#include <vsg/text/Font.h>
#include <vsg/state/Sampler.h>
#include <string>
#include <vector>
#include <vrb/client/SharedState.h>
#include <OpenVRUI/coUpdateManager.h>

class vvSidecarConfigBridge;

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
namespace covise
{
class Message;
}
namespace vive
{

namespace ui
{
class Owner;
class Group;
class FileBrowser;
}

class vvTUIFileBrowserButton;
class vvIOReader;

struct LoadedFile;

class VVCORE_EXPORT Url
{
public:
    Url(const std::string &url);
    static Url fromFileOrUrl(const std::string &furl);
    static std::string decode(const std::string &str, bool path=false);

    std::string str() const;
    operator std::string() const;

    std::string extension() const;
    bool valid() const;
    bool isLocal() const;

    const std::string &scheme() const;
    const std::string &authority() const;
    const std::string &path() const;
    const std::string &query() const;
    const std::string &fragment() const;

private:

    bool m_valid = false;

    std::string m_scheme;
    std::string m_authority;
    bool m_haveAuthority = false;
        std::string m_userinfo;
        std::string m_host;
        std::string m_port;
    std::string m_path;
    std::string m_query;
    std::string m_fragment;

    Url();
};


typedef struct
{
    int (*loadUrl)(const Url &url, vsg::Group *parent, const char *covise_key);
    int (*loadFile)(const char *filename, vsg::Group *parent, const char *covise_key);
    int (*unloadFile)(const char *filename, const char *covise_key);
    const char *extension;
} FileHandler;



class VVCORE_EXPORT vvFileManager : public vrui::coUpdateable
{
    friend struct LoadedFile;
    static vvFileManager *s_instance;

public:
    ~vvFileManager();
    static vvFileManager *instance();

    std::string findFileExt(const Url &url);

    // returns the full path for file
    const char *getName(const char *file);
    //if filePath starts with sharedDataPath, return true and remove sharedDataPath from filePath. Ignore upper/lower case differences
    bool makeRelativeToSharedDataLink(std::string &fileName);
	//return true if fileName contains tmp path
	bool isInTmpDir(const std::string& fileName);

    //return true if fineName is not in tmp dir and 
    //not in the remoteFetchDir of remoteFetchDir is shared (from config)
    bool isInSharedDir(const std::string &fileName);
    void checkRemoteFetchDirShared();

    //changes fileName to be relative to basePath
	bool makeRelativePath(std::string& fileName, const std::string& basePath);
    
    std::string findFile(const std::string &fileName);
    //search file locally, in sharedData and then try to remote fetch the file(if activated) until a the file gets found. Return "" if no file found.
    //"where" can be set to the partner id that should provide the file 
    
    std::string findOrGetFile(const std::string &fileName, int where = 0);
    // load a OSG or VRML97 or other (via plugin) file
    vsg::ref_ptr<vsg::Node> loadFile(const char *file, vvTUIFileBrowserButton *fb = NULL, vsg::Group *parent = NULL, const char *covise_key = "");

    // replace the last loaded Performer or VRML97 file
    vsg::ref_ptr<vsg::Node> replaceFile(const char *file, vvTUIFileBrowserButton *fb = NULL, vsg::Group *parent = NULL, const char *covise_key = "");

    // reload the previously loaded /*VRML*/ file
    void reloadFile();

    // unload the previously loaded file
    void unloadFile(const char *file=NULL);

    // set name of a file to store Viewpoints; if unset, this is derived from the loaded FileName
    void setViewPointFile(const std::string &viewPointFile);

    // getName of a file to store Viewpoints
    // this is derived from the loaded FileName
    std::string getViewPointFile();

    // load an icon file, looks in covise/icons/$LookAndFeel or covise/icons
    // returns NULL, if nothing found
    vsg::ref_ptr<vsg::Node> loadIcon(const std::string& filename);

    // loads a font
    // fontname can be NULL, which loads the default specified in the config (DroidSansFallbackFull.ttf if not in config), or "myfont.ttf"
    std::string getFontFile(const char *fontname);
    vsg::ref_ptr<vsg::Font> loadFont(const char *fontname);

    // load an texture, looks in covise/icons/$LookAndFeel or covise/icons for filename.rgb
    // returns NULL, if nothing found
    vsg::ref_ptr<vsg::Image>& loadTexture(const char *texture);

    // tries to fopen() fileName
    // returns true if exists otherwise false
    bool fileExist(const char *fileName);
	bool fileExist(const std::string& fileName);
    // builds filename for icon files, looks in covise/icons/$LookAndFeel or covise/icons for filename.rgb
    const char *buildFileName(const char *);

    // register a loader, ... for a file type
    int registerFileHandler(const FileHandler *handler);
    int registerFileHandler(vvIOReader *handler);

    // unregister a loader, ... for a file type
    int unregisterFileHandler(const FileHandler *handler);
    int unregisterFileHandler(vvIOReader *handler);

    // get list of extensions as required by a filebrowser
    std::string getFilterList();

    // get list of extensions for saving as required by a filebrowser
    std::string getWriteFilterList();

    // get a loader for a file type, if available
    const FileHandler *getFileHandler(const char *extension);
    vvIOReader *findIOHandler(const char *extension);

    // find a loader, load plugin if no loader available
    const FileHandler *findFileHandler(const char *extension);

    vvTUIFileBrowserButton *getMatchingFileBrowserInstance(std::string keyFileName);

    bool IsDefFBSet();

    void SetDefaultFB(vvTUIFileBrowserButton *fb);

    virtual bool update();
    //send a requested File to vrb
    void sendFile(covise::TokenBuffer &tb);

	///request the file from vrb -> file gets copied to configured dir
	std::string remoteFetch(const std::string &filePath, int fileOwner = -1);
    //download thie file from url to the same dir as remoteFetch
    std::string httpFetch(const std::string &url);
    //parse obj file and request the used material files
    void fetchObjMaterials(const std::string & localPath, const std::string &remotePath, int fileOwner);

    ///compares the url with m_sharedFiles. If found returns its position in, else -1;
	size_t getFileId(const std::string &url);

	///get the filename + extension from a path: path/fileName -> fileName
	std::string getFileName(const std::string& filePath);

    const std::string &getMainFile() const;

    void initUI();

private:
    // Get the configured font style.
    int coLoadFontDefaultStyle();
	//set in 'config/system/vrb/RemoteFetch value = "on" to enable remote fetch in local usr tmp directory. 
	bool remoteFetchEnabled = false;
    bool remoteFetchHashPrefix = true;
    // set in 'config/system/vrb/RemoteFetch path="your path" to chose a differen directory to remote Fetch to.
    std::string remoteFetchPath, remoteFetchPathTmp;
    bool remoteFetchDirExists = false;
    bool remoteFetchPathShared = false;
    std::string viewPointFile;
    int m_loadCount = 0;
    std::unique_ptr<ui::Owner> m_owner;
    ui::Group *m_fileGroup = nullptr;
    int uniqueNumber = 0;
    vsg::ref_ptr<vsg::Font> defaultFont;

    typedef std::list<const FileHandler *> FileHandlerList;
    FileHandlerList fileHandlerList;

    typedef std::list<vvIOReader *> IOReaderList;
    IOReaderList ioReaderList;

    typedef std::map<std::string, vsg::ref_ptr<vsg::Image> > TextureMap;
    TextureMap textureList;
    std::map<std::string, vvTUIFileBrowserButton *> fileFBMap;
    vvTUIFileBrowserButton *mDefaultFB = nullptr;

    struct IOReadOperation
    {
        IOReadOperation()
        {
            reader = 0;
            filename = "";
            group = 0;
        }
        vvIOReader *reader;
        std::string filename;
        vsg::Group *group;
    };

    typedef std::map<std::string, std::list<IOReadOperation> > ReadOperations;
    ReadOperations readOperations;

    vvFileManager();
    LoadedFile *m_lastFile = nullptr;
    LoadedFile *m_loadingFile = nullptr;
    std::map<std::string, LoadedFile *> m_files;
	typedef std::pair<std::string, int> fileOwner;
	///map of fileowners(client id) and file paths
	typedef std::vector<fileOwner> fileOwnerList;
    vrb::SharedState<fileOwnerList> m_sharedFiles;
    void loadPartnerFiles();
    struct Compare {
        bool operator()(const std::string& first, const std::string& second) {
            return first.size() > second.size();
        }
    };
    ///returns the full path of the symbolic link that points to the shared data
    ///the link should be in COVISE_PATH/.. and named sharedData
    void getSharedDataPath();
    std::string m_sharedDataLink;
	///convert all backslashes to forward slashes
    void convertBackslash(std::string &path);
	///cut the filename from path:  path/fileName --> path
    std::string cutFileName(const std::string & filePath);
    ///removs non-aphanumeric characters
    std::string reduceToAlphanumeric(const std::string & str);

	///writes content into a file unter remotefetchPath/hash/filename . Returns the path to the file or "" on failure
	std::string writeRemoteFetchedFile(const std::string& filePath, const char* content, int size);
	///compares the filePaths of m_sharedFiels wit filePath and returns the best matching fileOwner
	int guessFileOwner(const std::string& filePath);
	bool serializeFile(const std::string& fileName, covise::TokenBuffer& tb);
	std::vector<std::unique_ptr<covise::Message>> m_sendFileMessages;
    void createRemoteFetchDir();

    //utility
    
    ///replaces all occurences of environmentvariables (%env$ on win or $env/ on unix) with the first entry (delimited by ';')
    static std::string resolveEnvs(const std::string& s);
    ///return the substring of s until the delimiter(delimiter is cut off)
    static std::string cutStringAt(const std::string &s, char delimiter);

    std::string m_defaultFontFile;

    std::unique_ptr<vvSidecarConfigBridge> m_settings;
    std::string m_mainFile; // the first file that was loaded
};
}
#endif
