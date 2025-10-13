#ifndef COVER_DATACLIENT_OBSERVERHANDLE_H
#define COVER_DATACLIENT_OBSERVERHANDLE_H

#include "export.h"

#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace opencover{namespace dataclient{

class Client;


//automativally unobserves the node on deletion
class DATACLIENTEXPORT ObserverHandle{
    public:
        friend class Client;
        ObserverHandle() = default;
        ObserverHandle(size_t id,  Client *client);
        bool operator==(size_t id) const;
        bool operator==(const ObserverHandle &other) const;
        bool operator<(const ObserverHandle &other) const;
    private:
        struct Deleter{
            ~Deleter();
            size_t m_id = 0;
            Client *m_client = nullptr;
        };
        std::shared_ptr<Deleter> m_deleter;
        // ClientNode *m_node = nullptr; 
    };

}}

#endif // COVER_DATACLIENT_OBSERVERHANDLE_H