#ifndef NET_CONCRETE_MESSAGES_H
#define NET_CONCRETE_MESSAGES_H

#include "message_macros.h"
#include <util/coExport.h>
#include <iosfwd>
#include <vector>
#include <string>
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

    DECL_MESSAGE_CLASS(CRB_EXEC, NETEXPORT,
     ExecFlag, flag,
     char *, name,
     int, port,
     char *, localIp,
     int, moduleCount,
     char *, moduleId,
     char *, moduleIp,
     char *, moduleHostName,
     char *, displayIp,
     char *, category,
     std::vector<std::string>, params)
    //port and moduleCount are dummies to hold the correspondig char ptrs
    NETEXPORT std::vector<const char *> getCmdArgs(const CRB_EXEC &exec, std::string& port, std::string&moduleCount);



} //covise

#endif //!NET_CONCRETE_MESSAGES_H