/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef coIDATA_H_
#define coIDATA_H_
#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/message.h>
#else
#include <wce_msg.h>
#endif

namespace opencover
{
/**
 * The class provides an abstract interface for the FileBrowser
 * This way access to different data objects from the FileBrowser
 * itself can be handled. Some methods which are common for different
 * data access methods are implemented directly in this abstract class
 * thereby the implementation does not completely stay to the pattern
 * of interfaces.
 * @author Michael Braitmaier
 * @date 2007-01-12
 */
class IData
{
public:
    /**
       * The standard constructor for class initialization
       */
    IData(void);

    /**
       * The standard destructor for cleaning up used memory of
       * the class.
       */
    virtual ~IData(void);

    /**
       * DEPRECATED
       * Enumeration type specifying different locations the data
       * objects can get their data from. Currently three types
       * of locations are available.
       * LocalMachine - instructs the data object to retrieve file
       *				  and directory information only from the local machine
       * RemoteMachine - allows filesystem information retrieval from any computer
       *				   on the network. This however needs a mechanism for
       *				   communication which has to be implemented in a specialized
       *				   class of IData.
       * AccessGridStore allows data retrieval through usage of a special webservice
       *				   library to directly access a venue server associated FTP
       *				   server.
       */
    enum DataType
    {
        VRBDATA = 1,
        AGDATA = 2,
        LOCALDATA = 3
    };

    /**
       * Requests the list of directories contained in a given path.
       * No files are listed in the result.
       * @param path - the path to be searched for the directories.
       * @return a QStringList pointer containing all the directories in path
       */
    virtual void reqDirectoryList(std::string path, int pId) = 0;
    virtual void setDirectoryList(covise::Message &msg) = 0;

    /**
       * Requests the list of files contained in a given path.
       * No directories are listed in the result.
       * @param path - the path to be searched for the files.
       * @return a QStringList pointer containing all the files in path
       */
    virtual void reqFileList(std::string path, int pId) = 0;
    virtual void setFileList(covise::Message &msg) = 0;

    /**
       * This function determines the directories of the current location's
       * home directory. The result is given as a List of strings
       * @return a QStringList pointer containing all the directories in path
       */
    virtual void reqHomeDir(int pId) = 0;

    /**
       * This function determines the files of the current location's
       * home directory. The result is given as a List of strings
       * @return a QStringList pointer containing all the files in path
       */
    virtual void reqHomeFiles(int pId) = 0;

    /**
       * This function determines the directory on the next level above the current one.
       * This corresponds to a "one directory level up" command.
       * @param basePath - a path can be specified which should be used for
       *					 determining the next directory level above. However
       *					 if it is unspecified the member containing the current
       *					 path will be used.
       * @return a QString containing the desired directory.
       */
    //virtual void reqDirUp(QString basePath) = 0;

    /**
       * Allows to specify the mode the data object should work in
       * which as a consequence determines where filesystem information
       * will be retrieved from.
       */
    void setLocation(std::string ip = "");

    /**
       * Returns the current Path as set in the objects attribute.
       * @return QString - current path used by data object.
       */
    std::string getCurrentPath();
    void setCurrentPath(std::string path);

    /**
       * Allows to set the filter used to create the list of files
       * with getFileList(). The filter is specified as a string
       * containing the extension of the desirred files, e.g.
       * "*.net"
       * @param filter - the filter to be set in the format "*.<ext>"
       */
    void setFilter(std::string filter);

    DataType getType()
    {
        return mType;
    };
    //virtual void reqClientList(int pId) = 0;

    virtual void *getTmpFileHandle(bool sync = false) = 0;
    virtual std::string getTmpFilename(const std::string url, int id) = 0;

    virtual void reqDrives(int pId) = 0;

    virtual void setSelectedPath(std::string path) = 0;
    virtual std::string getSelectedPath() = 0;

protected:
    /**
       * Attribute for storing the currently used location
       * for filesystem data retrieval.
       * Can contain one of the two following values:
       * "Local/127.0.0.1" for search on the local machine of the requesting
       *				     FileBrowser
       * "<ip-address>"    for search on the file system of the given IP-address
       *				     associated machine( either VRB server or OpenCover
       *				     client)
       */
    std::string mIP;

    /**
       * Attribute for storing the latest retrieved directory list
       * as a chache.
       */
    //std::vector<string> mDirectoryList;

    /**
       * Attribute for storing the latest retrieved file list
       * as a chache.
       */
    //std::vector<string> mFileList;

    /**
       * Attribute for storing the current path to retrieve
       * file and directories from as long as not overridden
       * by method parameters.
       */
    std::string mCurrentPath;

    /**
       * Attribute for storing the currently used filter for
       * retrieving file and directories.
       * Contains the filter as a string in the format
       * "*.<file-extension>"
       */
    std::string mFilter;

    //LocationType mLocationType;

    DataType mType;

    std::string mFileLocation;
};
}
#endif
