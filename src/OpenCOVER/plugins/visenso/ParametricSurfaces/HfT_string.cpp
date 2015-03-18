/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <sstream>
#include <math.h>
#include "HfT_string.h"

std::string::size_type HfT_replace_char_in_string(std::string *str, char oldc, char newc)
{
    std::string::size_type pos = 0, repl = 0;

    while ((pos = str->find_first_of(oldc)) != std::string::npos)
    {
        (*str)[pos] = newc;
        repl++;
    }

    return (repl);
}
std::string HfT_double_to_string(double d)
{
    if (fabs(d) < 0.0000001)
        d = 0;
    std::stringstream ss;
    ss << d;
    std::string dstr = ss.str();
    return (dstr);
}
double HfT_string_to_double(std::string str)
{
    std::stringstream ss;
    double value = 0;
    ss << str;
    ss >> value;
    return (value);
}
std::string HfT_int_to_string(int d)
{
    std::stringstream ss;
    ss << d;
    std::string dstr = ss.str();
    return (dstr);
}
int HfT_string_to_int(std::string str)
{
    std::stringstream ss;
    int value = 0;
    ss << str;
    ss >> value;
    return (value);
}
std::string HfT_replace_string_in_string(std::string instr, std::string repstr, std::string newstr)
{
    int pos;
    while (true)
    {
        pos = instr.find(repstr);
        if (pos == -1)
            break;
        else
        {
            instr.erase(pos, repstr.length());
            instr.insert(pos, newstr);
        }
    }
    return instr;
}
std::string HfT_cut_LastBlanks_in_string(std::string str)
{
    std::string::reverse_iterator rit;
    std::string::reverse_iterator rend;
    std::string cutstr;
    unsigned int i = 0;

    for (rit = str.rbegin(); rit < str.rend(); rit++)
    {
        if (*rit == ' ')
        {
            i++;
        }
        else
            break;
    }
    cutstr = std::string(str, 0, str.size() - i);
    return cutstr;
}
std::string HfT_cut_LastDigits_in_string(std::string str, unsigned int round)
{
    std::string::reverse_iterator rit;
    std::string::reverse_iterator rend;
    std::string cutstr;
    bool found = false;
    unsigned int i = 0;

    for (rit = str.rbegin(); rit < str.rend(); rit++)
    {
        if (*rit == '.')
            found = true;
        i++;
        if (found)
            break;
    }
    if (found)
        cutstr = std::string(str, 0, str.size() - i + round + 1);
    else
        cutstr = str;
    return cutstr;
}
