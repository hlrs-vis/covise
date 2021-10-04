#include "program_type.h"

#include "tokenbuffer.h"

using namespace covise;

const char *ProgramNames::operator[](Program p) const
{
    return detail::programNames[static_cast<int>(p)];
}
const char *ProgramNames::operator[](size_t p) const
{
    return detail::programNames[static_cast<int>(p)];
}
detail::ProgramContainer::const_iterator ProgramNames::begin() const
{
    return detail::programNames.begin();
}
detail::ProgramContainer::const_iterator ProgramNames::end() const
{
    return detail::programNames.end();
}

constexpr size_t ProgramNames::size() const
{
    return detail::programNames.size();
}

covise::TokenBuffer &covise::operator<<(covise::TokenBuffer &tb, const covise::Program &p){
    tb << static_cast<int>(p);
    return tb;
}

covise::TokenBuffer &covise::operator>>(covise::TokenBuffer &tb, covise::Program &p){
    int type;
    tb >> type;
    p = static_cast<Program>(type);
    return tb;
}

std::ostream &covise::operator<<(std::ostream &os, const covise::Program &p){
    os << programNames[p];
    return os;
}