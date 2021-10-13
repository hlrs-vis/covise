#include "AsyncClient.h"
#include <messages/PROXY.h>

using namespace vrb;
using namespace covise;

bool AsyncClient::poll(covise::Message *msg)
{
    if (isSlave)
        return false;
    std::lock_guard<std::mutex> g{m_mutex};
    for (auto i = m_messages.begin(); i != m_messages.end(); ++i)
    {
        if (checkedByAll(*i))
        {
            *msg = i->message;
            i = m_messages.erase(i);
            return true;
        }
    }
    bool newMessages = false;
    while (sConn->check_for_input())
    {
        auto msg = m_messages.insert(m_messages.begin(), CountedMessage{});
        sConn->recv_msg(&msg->message);
        newMessages = true;
    }
    if (newMessages)
    {
        m_waitForNewMessages.notify_all();
    }
    return false;
}

void AsyncClient::addMessage(const covise::Message &msg)
{
    std::lock_guard<std::mutex> g{m_mutex};
    m_messages.push_back(CountedMessage{msg});
    m_waitForNewMessages.notify_all();
}

covise::Message AsyncClient::wait(std::function<bool(const covise::Message &)> evaluation)
{
    return waitMessage<covise::Message>([evaluation](const covise::Message &msg)
                       {
                           boost::optional<covise::Message> retval;
                           if (evaluation(msg))
                           {
                               retval = msg;
                           }
                           return retval;
                       });
}

size_t AsyncClient::createWaitId()
{
    std::lock_guard<std::mutex> g{m_mutex};
    size_t id = 0;
    while (!m_waitIds.insert(id).second)
    {
        ++id;
    }
    return id;
}

void AsyncClient::waitForNewMesages()
{
    std::unique_lock<std::mutex> lk(m_mutex);
    m_waitForNewMessages.wait(lk);
}

bool AsyncClient::checkedByAll(const CountedMessage &msg) const
{
    for (auto id : m_waitIds)
    {
        if (msg.waitersChecked.find(id) == msg.waitersChecked.end())
            return false;
    }
    return true;
}
