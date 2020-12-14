#include "concrete_messages.h"
#include "message.h"
#include "message_sender_interface.h"
#include "message_types.h"
#include "tokenbuffer.h"
#include "tokenbuffer_util.h"

#include <cassert>
#include <iostream>

namespace covise{

TokenBuffer &operator<<(TokenBuffer &tb, ExecFlag flag){
    tb << static_cast<int>(flag);
    return tb;
}

std::ostream &operator<<(std::ostream &os, ExecFlag flag){
    os << static_cast<int>(flag);
    return os;
}

TokenBuffer &operator>>(TokenBuffer &tb, ExecFlag& flag){
    int i;
    tb >> i;
    flag = static_cast<ExecFlag>(i);
    return tb;
}

IMPL_MESSAGE_CLASS(CRB_EXEC, char*, name, char*, cat, char*, param, char*, localIp, char*, moduleIp, char*, displayIp, char*, moduleHostName, char*, instance, int, port, int, moduleCount, ExecFlag, flag)

} //covise



