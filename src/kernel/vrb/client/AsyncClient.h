#ifndef VRB_ASYNC_CLIENT_H
#define VRB_ASYNC_CLIENT_H

#include "VRBClient.h"
#include <util/coExport.h>
#include <net/message_macros.h>

#include <list>
#include <mutex>
#include <boost/optional.hpp>
#include <net/message.h>
#include <functional>
#include <thread>
#include <chrono>
#include <set>
#include <condition_variable>
namespace vrb
{
    template <typename Message>
    typename std::enable_if<Message::Style == covise::detail::MessageSyle::Primary, boost::optional<Message>>::type createMesage(const covise::Message &msg)
    {
        return boost::optional<Message>{msg};
    }

    template <typename Message>
    typename std::enable_if<Message::Style == covise::detail::MessageSyle::SubMessage, boost::optional<Message>>::type createMesage(const covise::Message &msg)
    {
        typename Message::Base base{msg};
        if (base.type == Message::subType)
        {
            return boost::optional<Message>{base.template createDerived<Message>()};
        }
        return boost::optional<Message>{};
    }
    //thread save wait and poll. poll only returns messages that have been checked by all waiting threads. 
    class VRBCLIENTEXPORT AsyncClient : public VRBClientBase
    {
        using VRBClientBase::VRBClientBase;

    public:
        template <typename Message>
        Message wait(std::function<bool(const Message &)> evaluation)
        {
            return waitMessage<Message>([evaluation](const covise::Message &msg)
                                        {
                                            if (msg.type == Message::MessageType)
                                            {
                                                auto retval = createMesage<Message>(msg);
                                                if (retval && evaluation(*retval))
                                                {
                                                   return std::move(retval);
                                                }
                                            }
                                            return boost::optional<Message>{};
                                        });
        }

        covise::Message wait(std::function<bool(const covise::Message &)> evaluation);

        template <typename Message>
        Message wait(bool evaluation(const Message &))
        {
            return wait(std::function<bool(const Message &)>{evaluation});
        }

        template <typename Message>
        Message wait()
        {
            return wait(+[](const Message &)
                        { return true; });
        }
        bool poll(covise::Message *msg);
        void addMessage(const covise::Message &msg);

    private:
        struct CountedMessage
        {
            covise::Message message;
            std::set<size_t> waitersChecked;
        };
        std::list<CountedMessage> m_messages;
        std::set<size_t> m_waitIds;
        std::mutex m_mutex;
        std::condition_variable m_waitForNewMessages;

        template <typename Retval>
        Retval waitMessage(std::function<boost::optional<Retval>(const covise::Message &)> fnct)
        {
            auto id = createWaitId();
            while (true)
            {
                {
                    std::lock_guard<std::mutex> g{m_mutex};
                    for (auto i = m_messages.begin(); i != m_messages.end(); i++)
                    {
                        auto retval = fnct(i->message);
                        if (retval)
                        {
                            m_messages.erase(i);
                            m_waitIds.erase(id);
                            return std::move(*retval);
                        }
                        i->waitersChecked.insert(id);
                    }
                }
                waitForNewMesages();
            }
        }

        size_t createWaitId();
        void waitForNewMesages();
        bool checkedByAll(const CountedMessage &msg) const;
    };
}

#endif //VRB_ASYNC_CLIENT_H