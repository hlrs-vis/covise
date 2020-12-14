/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRBDATA_H_
#define VRBDATA_H_

#include "IRemoteData.h"
#include <net/message.h>
#include <vrb/client/VRBClient.h>
#include <cover/coTabletUI.h>

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
namespace opencover
{
class VRBData : public IRemoteData
{
public:
    VRBData(coTUIElement *elem = NULL);
    ~VRBData(void);
    void reqDirectoryList(std::string path, int pId);
    void setDirectoryList(covise::Message &msg);
    void reqFileList(std::string path, int pId);
    void setFileList(covise::Message &msg);

    void reqHomeDir(int pId);
    void reqHomeFiles(int pId);

    void reqDirUp(std::string basePath = "");

    vrb::VRBClient *getVRB();
    void setId(int id);
    int getId();
    void setCurDir(covise::Message &msg);
    void reqClientList(int pId);
    void setClientList(covise::Message &msg);
    void setDrives(covise::Message &msg);

    std::string getTmpFilename(const std::string url, int id);
    void *getTmpFileHandle(bool sync = false);
    void reqDrives(int pId);

    void setRemoteFileList(covise::Message &msg);
    void setRemoteDirList(covise::Message &msg);
    void setRemoteDir(covise::Message &msg, std::string absPath);
    void setRemoteDrives(covise::Message &msg);
    void setRemoteFile(covise::Message &msg);

    void reqRemoteFile(std::string filename, int pId);
    void setFile(covise::Message &msg);

    void setSelectedPath(std::string path);
    std::string getSelectedPath();
    bool VRBWait();

    void reqGlobalLoad(std::string url, int pId);

private:
    coTUIElement *mTUIElement;
    vrb::VRBClient *mVrbc;
    std::string mTmpFileName;
    int mId;
};
}
#endif
