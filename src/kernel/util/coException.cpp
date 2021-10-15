#include "coException.h"


namespace covise {


exception::exception(const std::string &what): m_what(what)
{
}

exception::~exception() throw()
{}

const char *exception::what() const throw()
{
    return m_what.c_str();
}

const char *exception::info() const throw()
{
    return m_info.c_str();
}


} //namespace covise
