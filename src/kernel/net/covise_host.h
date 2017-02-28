/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EC_HOST_H
#define EC_HOST_H

#include <util/coTypes.h>
#include <iostream>
#include <string>

namespace covise
{

class NETEXPORT Host
{
    friend class Socket;

private:
    unsigned char char_address[4];
    std::string m_address;
    bool m_addressValid;

    std::string m_name;
    bool m_nameValid;

    void HostNumeric(const char *n);
    void HostSymbolic(const char *n);

    void setAddress(const char *n);
    void setName(const char *n);

    void get_char_address(unsigned char *c) const;

    const unsigned char *get_char_address() const;

public:
    static std::string lookupHostname(const char *numericIp);
    static std::string lookupIpAddress(const char *hostname);

    Host();
    Host(const char *n, bool numeric = false);
    Host(unsigned long a);
    Host(const Host &h);
    ~Host();

    Host &operator=(const Host &h);

    uint32_t get_ipv4() const;

    const char *getName() const;
    const char *getAddress() const;
    bool hasValidName() const;
    bool hasValidAddress() const;

    bool hasRoutableAddress() const;

    void print()
    {
#ifdef DEBUG
        std::cerr << "Hostname: " << address << std::endl;
#endif
    }
};
}
#endif
