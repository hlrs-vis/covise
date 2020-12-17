
#include "message_macros.h"

#include <net/message.h>
#include <net/message_sender_interface.h>
#include <net/message_types.h>
#include <net/tokenbuffer.h>
#include <net/tokenbuffer_serializer.h>
#include <net/tokenbuffer_util.h>

#include <cassert>
#include <iostream>
#include <cassert>

namespace test
{
    IMPL_MESSAGE_CLASS(VRB_MESSAGE, int, messageType, int, clientID, std::vector<std::string>, args)

    IMPL_MESSAGE_WITH_SUB_CLASSES(VRB_LOAD_SESSION, VrbMessageType)
    IMPL_SUB_MESSAGE_CLASS(VRB_LOAD_SESSION, VrbMessageType, Launcher, int, clientID, std::vector<std::string>, args)
    IMPL_SUB_MESSAGE_CLASS(VRB_LOAD_SESSION, VrbMessageType, Avatar, std::string, name, std::vector<float>, pos)

    void testMessageClass()
    {
        std::vector<std::string> args;
        args.push_back("test1");
        args.push_back("test2");
        VRB_MESSAGE vrbMessage{5, 123, args};
        assert(vrbMessage.messageType == 5);
        assert(vrbMessage.clientID == 123);
        assert(vrbMessage.args.size() == 2);
        assert(vrbMessage.args[0] == "test1");
        assert(vrbMessage.args[1] == "test2");
        std::cerr << vrbMessage << std::endl;
        covise::TokenBuffer tb;
        tb << vrbMessage;
        covise::Message msg{tb};
        msg.type = covise::COVISE_MESSAGE_VRB_MESSAGE;
        VRB_MESSAGE vrbMessage2{msg};
        assert(vrbMessage2.messageType == 5);
        assert(vrbMessage2.clientID == 123);
        assert(vrbMessage2.args.size() == 2);
        assert(vrbMessage2.args[0] == "test1");
        assert(vrbMessage2.args[1] == "test2");
        std::cerr << vrbMessage2 << std::endl;
    }

    void testMessageSubCLass()
    {
        std::vector<std::string> args;
        args.push_back("test1");
        args.push_back("test2");
        VRB_LOAD_SESSION_Launcher launcher{154, args};
        std::cerr << launcher << std::endl;

        assert(launcher.clientID == 154);
        assert(launcher.args.size() == 2);
        assert(launcher.args[1] == "test2");
        covise::TokenBuffer tb;
        tb << launcher;
        covise::Message msg{tb};
        msg.type = covise::COVISE_MESSAGE_VRB_LOAD_SESSION;
        VRB_LOAD_SESSION m{msg};
        switch (m.type)
        {
        case VrbMessageType::Avatar:
            assert(false);
            break;
        case VrbMessageType::Launcher:
            auto &launcher2 = m.unpackOrCast<VRB_LOAD_SESSION_Launcher>();
            assert(launcher2.clientID == 154);
            assert(launcher2.args.size() == 2);
            assert(launcher2.args[1] == "test2");
            std::cerr << launcher2 << std::endl;
            break;
        }
    }

    void test_message_macros()
    {
        testMessageClass();
        testMessageSubCLass();
    }

} // namespace test
