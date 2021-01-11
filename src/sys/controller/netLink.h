/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CONTROLLER_NETLINK_H
#define CONTROLLER_NETLINK_H

#include <string>

namespace covise
{
class Message;
namespace controller
{
struct NetModule;

class netlink
{
    std::string name;
    std::string instanz;
    std::string host;
    const controller::NetModule *mptr;
    /* evtl weitere Eintraege bzgl. synchro. */

public:
    netlink();
    const std::string &get_name() const
    {
        return name;
    };
    const std::string &get_instanz() const
    {
        return instanz;
    };
    const std::string &get_host() const
    {
        return host;
    };
    void set_name(const std::string &str);
    void set_instanz(const std::string &str);
    void set_host(const std::string &str);
    void set_mod(const NetModule *ptr)
    {
        mptr = ptr;
    };
    const NetModule *get_mod()
    {
        return mptr;
    };
    void del_link(const std::string &lname, const std::string &lnr, const std::string &lhost);
    /* evtl. unnoetig */
    void send_message(const Message *msg, const std::string &loc_nr);
};
    } // namespace controller

} // namespace covise

#endif