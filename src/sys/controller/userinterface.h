/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CONTROLLER_USERINTERFACE_H
#define CONTROLLER_USERINTERFACE_H

#include "subProcess.h"
#include "moduleInfo.h"

namespace covise{
namespace controller{

struct RemoteHost;
class CRBModule;

struct UIOptions{
    enum Type
    {
        gui,
        python,
        miniGui,
        nogui
    }type = gui;
    std::string pyFile;
    bool iconify = false;
    bool maximize = false;
};

class Userinterface : public SubProcess
{

public:
    static const SubProcess::Type moduleType = SubProcess::Type::UI;
    Userinterface(const RemoteHost& host, const std::string &execName);
    virtual ~Userinterface();
    enum Status
    {
        Master,
        Slave,
        Mirror,
        Init,
        LASTDUMMY
    };
    static const char *getStatusName(Status s);
    void sendCurrentNetToUI(const std::string &filename = std::string{}) const;

    void setStatus(Status status);
    Status status() const;
    void changeStatus(Status status);
    void changeMaster(const RemoteHost& master);

    virtual bool start(const UIOptions &options, bool restart) = 0;
    bool restart(const UIOptions &options);
    void quit();

protected:
    static const std::array<const char *, static_cast<int>(LASTDUMMY)> statusNames; //to put into messages
    Status m_status;
    bool rendererIsPossible;
    bool rendererIsActive;
    void updateUI();
};


struct MapEditor : Userinterface
{
    MapEditor(const RemoteHost& host);

    bool start(const UIOptions &options, bool restart) override;
};

struct WsInterface : Userinterface
{
    static ModuleInfo wsInterfaceInfo;
    WsInterface(const RemoteHost &host);
    bool start(const UIOptions &options, bool restart) override;
};

struct PythonInterface : Userinterface{

    PythonInterface(const RemoteHost& host, const std::string &execName);
    bool start(const UIOptions &options,bool restart) override;

};


} // namespace controller
} // namespace covise

#endif // !CONTROLLER_USERINTERFACE_H