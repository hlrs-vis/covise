#include "message_macros.h"
#include <util/coExport.h>
#include <iosfwd>
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

    DECL_MESSAGE_CLASS(CRB_EXEC, NETEXPORT, char *, name, char *, cat, char *, param, char *, localIp, char *, moduleIp, char *, displayIp, char *, moduleHostName, char *, instance, int, port, int, moduleCount, ExecFlag, flag)





} //covise