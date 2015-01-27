/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _USE_LEXERS_H_
#define _USE_LEXERS_H_

#include <fstream>
using namespace std;

#include <unistd.h>
#include <stdio.h>

#include <string>
#ifdef __sgi
using namespace std;
#endif

template <class Lexer, class Service>
class useLexers
{
private:
    string simDir_;

public:
    useLexers(const char *simDir)
        : simDir_(simDir)
    {
    }
    int run(const char *Iname, const char *Oname)
    {
        string iname(simDir_);
        iname += Iname;
        ifstream topoIn(iname.c_str());
        if (!topoIn.rdbuf()->is_open())
        {
            Service::sendWarning("Could not open input lexer file");
            return -1;
        }
        string oname(simDir_);
        oname += Oname;
        ofstream topoOut(oname.c_str());
        if (!topoOut.rdbuf()->is_open())
        {
            Service::sendWarning("Could not open output lexer file");
            return -1;
        }
        Lexer tlexer(&topoIn, &topoOut);
        while (tlexer.yylex() != 0)
        {
        }
        unlink(iname.c_str());
        if (rename(oname.c_str(), iname.c_str()) != 0)
        {
            Service::sendWarning("Could not rename a _k file");
            return -1;
        }
        return 0;
    }
};
#endif
