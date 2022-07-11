#include "coGRColorBarPluginMsg.h"
using namespace grmsg;
coGRColorBarPluginMsg::coGRColorBarPluginMsg(Type t)
: coGRMsg(coGRMsg::COLOR_BAR_PLUGIN)
, m_type(t)
{
    addToken(std::to_string((int)t).c_str());
}

coGRColorBarPluginMsg::coGRColorBarPluginMsg(const char *msg)
:coGRMsg(msg)
{
    m_type = (Type)std::stoi(extractFirstToken());
}