/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// string_parse.h
// string_parse class declaration
//
////////////////////////////////////////////////////////////////////////

#ifndef __STRING_PARSE_H
#define __STRING_PARSE_H

#include "common_include.h"

#include <iosfwd>

class string_parse : public std::string
{
public:
    inline string_parse();
    inline string_parse(const char name[]);
    string_parse(std::ifstream &file);

public:
    string_parse &readword(std::string &text);
    string_parse &readquoted(std::string &text, char iquote, char fquote);
    string_parse &readquoted(std::string &text, char quote);
    string_parse &nocomment(const std::string &comstr);
    string_parse &nocomment(const std::string &icomstr, const std::string &fcomstr);
    string_parse &skipwhite();
    inline string_parse &rewind();

    inline bool eos() const;

    inline bool operator!() const;

private:
    const_iterator pos;
};

inline string_parse::string_parse()
    : std::string()
{
}

inline string_parse::string_parse(const char name[])
    : std::string(name)
{
}

inline string_parse &string_parse::rewind()
{
    pos = begin();
    return *this;
}

bool string_parse::eos() const
{
    return pos >= end();
}

bool string_parse::operator!() const
{
    return eos();
}

#endif // __STRING_PARSE_H
