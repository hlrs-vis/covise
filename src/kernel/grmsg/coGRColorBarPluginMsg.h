#ifndef COGR_COLOR_BAR_MSG_H
#define COGR_COLOR_BAR_MSG_H
#include "coGRMsg.h"
namespace grmsg
{
class GRMSGEXPORT coGRColorBarPluginMsg : public coGRMsg
{
public:
    enum Type
    {
        ShowColormap
    };
    coGRColorBarPluginMsg(Type t);
    coGRColorBarPluginMsg(const char *msg);
protected:

private:
    Type m_type;
};
} // namespace grmsg

#endif // COGR_COLOR_BAR_MSG_H