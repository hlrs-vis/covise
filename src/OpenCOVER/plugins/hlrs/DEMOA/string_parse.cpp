/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// string_parse.cpp
// string_parse class definition
//
////////////////////////////////////////////////////////////////////////

#include "string_parse.h"

#ifdef WIN32
#pragma warning(disable : 4786)
#endif

#include <cctype>
#include <cstdlib>

#include <fstream>
#include <iostream>

string_parse::string_parse(std::ifstream &file)
    : std::string()
{
    while (!file.eof())
    {
        char c;
        if (file.get(c))
        {
            *this += c;
        }
    }
    pos = begin();
}

string_parse &string_parse::nocomment(const std::string &icomstr,
                                      const std::string &fcomstr)
{
    typedef std::string::size_type ST;

    ST start = 0U;
    ST maxpos = size() - 1;
    const ST offset = fcomstr.size();

    while (start <= maxpos)
    {
        const ST lopos = find(icomstr, start);
        if (lopos > maxpos)
        {
            break;
        }
        const ST hipos = find(fcomstr, lopos);
        if (hipos > maxpos)
        {
            std::cerr << "Error! Missing termination '"
                      << fcomstr
                      << "' of comment '"
                      << substr(lopos, find('\n', lopos) - lopos)
                      << "...'. Exiting"
                      << std::endl;
            std::exit(0);
        }
        erase(lopos, hipos - lopos + offset);
        start = lopos;
        maxpos = size() - 1;
    }
    pos = begin();
    return *this;
}

string_parse &string_parse::nocomment(const std::string &comstr)
{
    typedef std::string::size_type ST;

    ST start = 0U;
    ST maxpos = size() - 1;

    while (start <= maxpos)
    {
        const ST lopos = find(comstr, start);
        if (lopos == npos)
        {
            break;
        }
        ST hipos = find('\n', lopos);
        if (hipos == npos)
        {
            hipos = maxpos + 1;
        }
        erase(lopos, hipos - lopos);
        start = lopos;
        maxpos = size() - 1;
    }
    pos = begin();
    return *this;
}

string_parse &string_parse::readword(std::string &text)
{
    text.clear();
    skipwhite();
    while (!std::isspace(static_cast<int>(*pos)) && !eos())
    {
        text += *pos++;
    }
    return *this;
}

string_parse &string_parse::readquoted(std::string &text, const char iquote,
                                       const char fquote)
{
    const_iterator savepos;
    text.clear();
    skipwhite();

    if (*pos != iquote)
    {
        std::cerr << "Error! Initial quote not found: '";
        while (*pos != '\n' && pos != end())
        {
            std::cerr << *pos++;
        }
        if (pos == end())
            return *this;
    }
    savepos = pos;
    ++pos;
    if (pos == end())
        return *this;
    int level = 0;
    while (*pos != fquote || level)
    {
        if (pos >= end())
        {
            std::cerr << "Error! Missing terminating quote '";
            while (*savepos != '\n' && savepos != end())
            {
                std::cerr << *savepos++;
            }
            std::cerr << "...'. Exiting." << std::endl;
            std::exit(0);
        }
        if (iquote != fquote)
        { // Count quotation levels within block
            if (*pos == iquote)
            {
                ++level;
            }
            if (*pos == fquote)
            {
                --level;
            }
        }
        text += *pos++;
    }
    ++pos;
    return *this;
}

string_parse &string_parse::readquoted(std::string &text, char quote)
{
    readquoted(text, quote, quote);
    return *this;
}

string_parse &string_parse::skipwhite()
{
    while (!eos() && std::isspace(static_cast<int>(*pos++)))
        ;
    --pos;
    return *this;
}
