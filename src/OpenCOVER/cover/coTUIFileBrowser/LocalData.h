/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef LOCALDATA_H
#define LOCALDATA_H

#include "IData.h"
#include <net/message.h>
#include <cover/coTabletUI.h>
#include <vrb/client/VRBClient.h>

namespace opencover
{
class coTUIElement;

/**
 * @Brief:This class encapsulates functionality to retrieve files from
 *        the local file system.
 * @Desc: This class encapsulates functionality to retrieve files from
 *        the local file system. Therefore it uses links in the
 *        following format:
 *
 *		  file://<location>/<path-to-file>/<filename>
 *
 *		  Based on the protocol identifier the class distinguishes for
 *        responsibility for this URL.
 */

class LocalData : public IData
{
public:
    LocalData(coTUIElement *elem = nullptr);
    ~LocalData() override;
    void reqDirectoryList(std::string path, int pId) override;
    void setDirectoryList(const covise::Message &msg) override;
    void reqFileList(std::string path, int pId) override;
    void setFileList(const covise::Message &msg) override;

    void reqHomeDir(int pId) override;
    void reqHomeFiles(int pId) override;
    void setHomeDir();

    void reqDirUp(std::string basePath = "", int pId = 0);
    void setCurDir(covise::Message &msg);

    //	LocationType getType();
    std::string getTmpFilename(const std::string url, int id) override;
    void *getTmpFileHandle(bool sync = false) override;
    void reqDrives(int pId) override;

    void setFile(std::string file);

    void setSelectedPath(std::string path) override;
    std::string getSelectedPath() override;

    std::string resolveToAbsolute(const std::string &dir);

private:
    coTUIElement *mTUIElement;
    int mId;
    std::string mFile;
};
}
#endif
