#ifndef NET_PROGRAM_TYPE_H
#define NET_PROGRAM_TYPE_H

#include <util/coExport.h>

#include <array>
#include <ostream>

namespace covise{
class TokenBuffer;

enum class Program
{
    covise,
    opencover,
    coviseDaemon,
    crb,
    LAST_DUMMY
};
namespace detail{
    typedef std::array<const char *, static_cast<int>(Program::LAST_DUMMY)> ProgramContainer;
    constexpr ProgramContainer programNames{
        "covise",
        "opencover",
        "coviseDaemon",
        "crb",
        };
}

struct NETEXPORT ProgramNames
{
    const char *operator[](Program p) const;
    const char *operator[](size_t p) const;
    detail::ProgramContainer::const_iterator begin() const;
    detail::ProgramContainer::const_iterator end() const;
    constexpr size_t size() const;
};
constexpr ProgramNames programNames;

NETEXPORT covise::TokenBuffer &operator<<(covise::TokenBuffer &tb, const covise::Program &userType);
NETEXPORT covise::TokenBuffer &operator>>(covise::TokenBuffer &tb, covise::Program &userType);
NETEXPORT std::ostream &operator<<(std::ostream &os, const covise::Program &userInfo);    

}

#endif // !NET_PROGRAM_TYPE_H