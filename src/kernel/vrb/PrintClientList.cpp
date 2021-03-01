#include "PrintClientList.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>

using namespace vrb;

template <typename T>
size_t getStringifiedSize(const T &t)
{
    std::stringstream ss;
    ss.operator<<(t);
    return ss.str().size();
}

template <typename Retval>
size_t longestStringT(const std::vector<const RemoteClient *> &clients, const std::function<Retval(const RemoteClient *)> &func)
{
    auto f = std::max_element(clients.begin(), clients.end(), [func](const RemoteClient *cl1, const RemoteClient *cl2) {
        return getStringifiedSize(func(cl1)) < getStringifiedSize(func(cl2));
    });
    return getStringifiedSize(func(*f));
}

template <>
size_t longestStringT(const std::vector<const RemoteClient *> &clients, const std::function<std::string(const RemoteClient *)> &func)
{
    auto f = std::max_element(clients.begin(), clients.end(), [func](const RemoteClient *cl1, const RemoteClient *cl2) {
        return func(cl1).size() < func(cl2).size();
    });
    return func(*f).size();
}

std::string allignedString(const std::string &s, size_t space)
{
    assert(s.size() <= space);
    std::string ns(space, ' ');
    ns.replace(0, s.size(), s);
    return ns;
}

void vrb::printClientInfo(const std::vector<const RemoteClient *> &clients)
{
    size_t distance = 3;
    std::array<const char *, 5> headings{"ID", "Name", "Email", "Hostname", "Master"};
    
    std::array<std::function<std::string(const RemoteClient *)>, headings.max_size()> functions;
    functions[0] = [](const RemoteClient *cl) { return std::to_string(cl->ID());    };
    functions[1] = [](const RemoteClient *cl) { return cl->userInfo().userName ;        };
    functions[2] = [](const RemoteClient *cl) { return cl->userInfo().email ;       };
    functions[3] = [](const RemoteClient *cl) { return cl->userInfo().hostName ;    };
    functions[4] = [](const RemoteClient *cl) { return cl->isMaster() ? "true" : "false" ;    };


    std::array<size_t, headings.max_size()> max;
    for (size_t i = 0; i < functions.size(); i++)
    {
        max[i] = longestStringT(clients, functions[i]);
        max[i] = std::max(max[i], strlen(headings[i])) + distance;
    }
    for (size_t i = 0; i < functions.size(); i++)
    {
        std::cerr << allignedString(headings[i], max[i]);
    }
    std::cerr << std::endl;
    for (const auto cl : clients)
    {
        for (size_t i = 0; i < functions.size(); i++)
        {
            std::cerr << allignedString(functions[i](cl), max[i]);
        }
        std::cerr << std::endl;
    }
    std::cerr << std::endl;
}

void vrb::printClientInfo(const std::vector<RemoteClient> &clients)
{
    std::vector<const RemoteClient *> clientPtrs(clients.size());
    std::transform(clients.begin(), clients.end(), clientPtrs.begin(), [](const RemoteClient &cl) {
        return &cl;
    });
    printClientInfo(clientPtrs);
}

void vrb::printClientInfo(const std::vector<std::unique_ptr<vrb::RemoteClient>> &clients)
{
    std::vector<const RemoteClient *> clientPtrs(clients.size());
    std::transform(clients.begin(), clients.end(), clientPtrs.begin(), [](const std::unique_ptr<RemoteClient> &cl) {
        return cl.get();
    });
    printClientInfo(clientPtrs);
}