
#include <net/message_macros.h>
#include <vector>
#include <string>
namespace test
{
    #define NOEXPORT

    enum class VrbMessageType
    {
        Launcher,
        Avatar
    };

    DECL_MESSAGE_CLASS(VRB_MESSAGE, NOEXPORT, int, messageType, int, clientID, std::vector<std::string>, args)

    DECL_MESSAGE_WITH_SUB_CLASSES(VRB_LOAD_SESSION, VrbMessageType, )
    DECL_SUB_MESSAGE_CLASS(VRB_LOAD_SESSION, VrbMessageType, Launcher, NOEXPORT, int, clientID, std::vector<std::string>, args)
    DECL_SUB_MESSAGE_CLASS(VRB_LOAD_SESSION, VrbMessageType, Avatar, NOEXPORT, std::string, name, std::vector<float>, pos)

    void test_message_macros();

    

} // namespace test


