/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include "IRemoteData.h"
#include <net/message.h>
#include <vrb/client/VRBClient.h>
#include "../vvTabletUI.h"

/**
 * @Brief:This class encapsulates functionality to retrieve files from a VRB
 *        server or a connected VRB client. Therefore it uses links in the
 * @Desc: This class encapsulates functionality to retrieve files from a VRB
 *        server or a connected VRB client. Therefore it uses links in the
 *        following format:
 *
 *		  vrb://<location>/<path-to-file>/<filename>
 *
 *		  Based on the protocol identifier the class distinguishes for
 *        responsibility for this URL.
 */
namespace vive
{
class VRBData : public IRemoteData
{
public:
    VRBData(vvTUIElement *elem = nullptr);
    void reqDirectoryList(std::string path, int pId) override;
    void setDirectoryList(const covise::Message &msg) override;
    void reqFileList(std::string path, int pId) override;
    void setFileList(const covise::Message &msg) override;

    void reqHomeDir(int pId) override;
    void reqHomeFiles(int pId) override;

    void reqDirUp(std::string basePath = "");

    vrb::VRBClient *getVRB();
    void setId(int id);
    int getId();
    void setCurDir(const covise::Message &msg);
    void reqClientList(int pId) override;
    void setClientList(const covise::Message &msg) override;
    void setDrives(const covise::Message &msg);

    std::string getTmpFilename(const std::string url, int id) override;
    void *getTmpFileHandle(bool sync = false) override;
    void reqDrives(int pId) override;

    void setRemoteFileList(const covise::Message &msg);
    void setRemoteDirList(const covise::Message &msg);
    void setRemoteDir(const covise::Message &msg, std::string absPath);
    void setRemoteDrives(const covise::Message &msg);
    void setRemoteFile(const covise::Message &msg);

    void reqRemoteFile(std::string filename, int pId);
    void setFile(const covise::Message &msg);

    void setSelectedPath(std::string path) override;
    std::string getSelectedPath() override;
    bool VRBWait();

    void reqGlobalLoad(std::string url, int pId);

private:
    vvTUIElement *mTUIElement;
    vrb::VRBClient *mVrbc;
    std::string mTmpFileName;
    int mId;
};
}
