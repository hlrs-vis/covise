/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef AGDATA_H_
#define AGDATA_H_

#include "IRemoteData.h"
#include <net/message.h>
#include <cover/coTabletUI.h>
#include <DataStore.h>

#include <util/coExport.h>

namespace covise
{
class coTUIElement;
}
namespace vrb{
class VRBClient;
}
namespace opencover
{
/**
 * @Brief:This class encapsulates functionality to retrieve files from an
 *		  AccessGrid datastore server.
 * @Desc: This class encapsulates functionality to retrieve files from an
 *		  AccessGrid datastore server. Therefore it uses links in the
 *        following format:
 *
 *		  agtk3://<location>/<venue>/<path-to-file>/<filename>
 *
 *		  Based on the protocol identifier the class distinguishes for
 *        responsibility for this URL.
 *		  It is yet undecided how to resolve to the real URL which starts
 *        with https. A possible solution might be to just replace
 *		  agtk3 --> https.
 */

class AGData : public IRemoteData
{
public:
    AGData(coTUIElement *elem = NULL);
    ~AGData(void);
    void reqDirectoryList(std::string path, int pId);
    void setDirectoryList(Message &msg);
    void reqFileList(std::string path, int pId);
    void setFileList(Message &msg);

    void reqHomeDir(int pId);
    void reqHomeFiles(int pId);
    void reqClientList(int pId);
    void setClientList(Message &msg);
    void reqDrives(int pId);

    void reqDirUp(std::string basePath = "");

    void setVRBC(VRBClient *vrbc);
    VRBClient *getVRB();
    void setId(int id);
    void setCurDir(Message &msg);

    std::string getTmpFilename(const std::string url, int id);
    void *getTmpFileHandle(bool sync = false);
    void setFile(std::string Filename);

    void setSelectedPath(std::string path);
    std::string getSelectedPath();

private:
    coTUIElement *mTUIElement;
    VRBClient *mVrbc;
    int mId;
    CDataStore *mDataStore;
    std::string mLastFile;
};
}
#endif
