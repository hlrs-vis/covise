#pragma once

#include <exception>
#include <string>


namespace virvo
{

namespace fileio
{

struct exception : public std::exception
{

  exception()                       : message_("unknown fileio error") {}
  exception(std::string const& msg) : message_(msg) {}
  virtual ~exception() throw() {}

  char const* what() const throw()
  {
    return message_.c_str();
  }

private:

  std::string message_;

};


struct wrong_dimensions : public exception
{
  wrong_dimensions() : exception("data must be three dimensional") {}
};


struct unsupported_datatype : public exception
{
  unsupported_datatype() : exception("data type stored in file is not supported") {}
};


} // fileio

} // virvo


