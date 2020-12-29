#include "concrete_messages.h"
#include "message.h"
#include "message_sender_interface.h"
#include "message_types.h"
#include "tokenbuffer.h"
#include "tokenbuffer_util.h"
#include "tokenbuffer_serializer.h"

#include <cassert>
#include <iostream>
#include <algorithm>
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

IMPL_MESSAGE_CLASS(CRB_EXEC, 
     ExecFlag, flag,
     char *, name,
     int, port,
     char *, localIp,
     int, moduleCount,
     char *, moduleId,
     char *, moduleIp,
     char *, moduleHostName,
     char *, displayIp,
     char*, vrbSession,
     char *, category,
     std::vector<std::string>, params)

std::vector<const char *> getCmdArgs(const CRB_EXEC &exec, std::string& port, std::string&moduleCount){
    port = std::to_string(exec.port);
    moduleCount = std::to_string(exec.moduleCount);
    
    size_t l = 9;
    l += exec.params.size();
    std::vector<const char *> args(l);
    size_t pos = 0;
    args[pos++] = exec.name;
    for(auto & arg : exec.params)
    {
        args[pos++] = arg.c_str();
    }
    args[pos++] = port.c_str();
    args[pos++] = exec.localIp;
    args[pos++] = moduleCount.c_str();
    args[pos++] = exec.moduleId;
    args[pos++] = exec.moduleIp;
    args[pos++] = exec.moduleHostName;
    args[pos++] = exec.displayIp;
    args[pos++] = "dummy";
    args.erase(std::remove(args.begin(), args.end(), nullptr), args.end());
    args.erase(std::remove_if(args.begin(), args.end(), [](const char* c) {return strcmp(c, "") == 0; }), args.end());
    args[args.size() - 1] = nullptr;
    return args;
}

} //covise



