#ifndef MSG_CRB_EXEC_H
#define MSG_CRB_EXEC_H

#include <net/message_macros.h>
#include <util/coExport.h>
#include <iosfwd>
#include <vector>
#include <string>

#include <vrb/client/VrbCredentials.h>

namespace covise{
    class TokenBuffer;
    class MessageSenderInterface;
    class Message;
    enum class ExecFlag
    {
        Normal,
        Debug,
        Memcheck
    };
    TokenBuffer &operator<<(TokenBuffer &tb, ExecFlag);
    TokenBuffer &operator>>(TokenBuffer &tb, ExecFlag&);
    std::ostream &operator<<(std::ostream &os, ExecFlag);

    DECL_MESSAGE_CLASS(CRB_EXEC, COMSGEXPORT,
     ExecFlag, flag,
     const char *, name,
     int, controllerPort,
     const char *, controllerIp,
     int, moduleCount,
     const char *, moduleId,
     const char *, moduleIp,
     const char *, moduleHostName,
     const char *, category,
     int, vrbClientIdOfController,
     vrb::VrbCredentials, vrbCredentials,
     std::vector<std::string>, params)
     
    COMSGEXPORT std::vector<std::string> getCmdArgs(const CRB_EXEC &exec);
    COMSGEXPORT CRB_EXEC getExecFromCmdArgs(int argC, char* argV[]);


} //covise

#endif //!NET_CONCRETE_MESSAGES_H