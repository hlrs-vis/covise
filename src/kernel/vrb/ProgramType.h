#ifndef VRB_PROGRAM_TYPE_H
#define VRB_PROGRAM_TYPE_H

#include <util/coExport.h>

#include <array>
#include <ostream>

namespace covise{
    class TokenBuffer;
}

namespace vrb{

enum class Program
{
    covise,
    opencover,
    VrbRemoteLauncher,
    crb,
    crbProxy,
    LAST_DUMMY
};
namespace detail{
    typedef std::array<const char *, static_cast<int>(Program::LAST_DUMMY)> ProgramContainer;
    constexpr ProgramContainer programNames{
        "covise",
        "opencover",
        "VrbRemoteLauncher",
        "crb",
        "crbProxy"
        };
}

struct VRBEXPORT ProgramNames
{
    const char *operator[](Program p) const;
    const char *operator[](size_t p) const;
    detail::ProgramContainer::const_iterator begin() const;
    detail::ProgramContainer::const_iterator end() const;
    constexpr size_t size() const;
};
constexpr ProgramNames programNames;

VRBEXPORT covise::TokenBuffer &operator<<(covise::TokenBuffer &tb, const vrb::Program &userType);
VRBEXPORT covise::TokenBuffer &operator>>(covise::TokenBuffer &tb, vrb::Program &userType);
VRBEXPORT std::ostream &operator<<(std::ostream &os, const vrb::Program &userInfo);    

}

#endif // !VRB_PROGRAM_TYPE_H