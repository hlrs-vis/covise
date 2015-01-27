/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          main.h  -  global variables
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#ifndef _MAIN_H_

#define _MAIN_H_

#include <iostream>
#include <string>
#include <strstream>
#include <map>

#include "arrays.h"

#ifndef GUI
#define VERSION 1.05
#else
#include <qapplication.h>
#endif

const prec PI = 3.14159;
const prec TWO_PI = 6.28319;
const prec TWO_OVER_PI = 0.63662;
const prec PI_OVER_TWO = 1.57080;

extern string inname; // name of input file
extern string outname; // name of output file
extern string abortname; // name of abort file

extern void init_program(const char *filename);

#ifdef GUI
class TextView;

extern QApplication *papp; // application
extern TextView *pout; // default output stream
extern TextView *perr; // default error stream
#else
extern ostream *pout; // default output stream
extern ostream *perr; // default error stream
#endif
extern ostream *plog; // default log stream
extern ostream *pdebug; // default debug stream
extern istream *pin; // default input stream

extern prec RefLength; // length scale
extern prec RefTime; // time scale
extern prec RefVelocity; // velocity scale

void ResetVariables(); // set variables to default
void UpdateVariables(); // update dependent variables
bool CheckHeader(istream &, char *); // check header in file

inline istream &endl(istream &ps)
{
    while ((ps.get()) != '\n')
        if (ps.fail())
            break;
    return ps;
}

inline istream &tab(istream &ps)
{
    while ((ps.get()) != '\t')
        if (ps.fail())
            break;
    return ps;
}

inline ostream &tab(ostream &ps)
{
    return ps << '\t';
}

// find character

class fc
{
public:
    fc(char c)
    {
        ch = c;
    }

    char ch;
};

inline istream &operator>>(istream &ps, fc _fc)
{
    while ((ps.get()) != _fc.ch)
        if (ps.fail())
            break;
    return ps;
}

// check for string

class checkstring
{
public:
    checkstring(const char *ps, bool *p)
    {
        pstr = ps;
        pb = p;
    }

    const char *pstr;
    bool *pb;
};

istream &operator>>(istream &, checkstring);

// check for strings

class checkstrings
{
public:
    checkstrings(const char *ps[], int *p)
    {
        slist = ps;
        pi = p;
    }

    const char **slist;
    int *pi;
};

istream &operator>>(istream &, checkstrings);

// ***************************************************************************
// script class
// ***************************************************************************

#ifdef WIN32
#pragma warning(disable : 4786) // disable debug warning for long names
#endif

class TScript
{
public:
    TScript()
    {
        sScript = "start";
        iLine = -1;
        iBracket = 0;
    }

    int execute();
    int executeCommand(string &);
    int getNumLines() const;
    int getLineIndex(int);
    void getBracketStart(int &);
    prec getValue(istrstream &); // read number
    prec getExpression(istrstream &); // evaluate expression
    void Save(int unit);
    void Read(int unit);

    string sScript; // script
    int iLine; // current line in script
    int iBracket; // bracket level in script
    map<int, int> iloopend; // end value of loop (bracket index)
    map<int, int> iloopvar; // loop variable (bracket index)
    map<int, int> ivariables; // interger variables
    map<int, prec> pvariables; // prec variables

    enum ScriptStatus
    {
        Ok = 0,
        JumpBracket = 1,
        Error = -10,
        SyntaxError = -11,
        Aborted = -12,
        START = 2
    };
};

extern TScript script; // script class

extern ostream &operator<<(ostream &, const TScript &);
extern istream &operator>>(istream &, TScript &);

template <class T>
inline ostream &operator<<(ostream &os, map<int, T> m)
{
    int imax;
    map<int, T>::iterator it;

    imax = m.size();
    os << "elements:" << tab << imax << endl;
    it = m.begin();
    while (it != m.end())
    {
        os << it->first << tab << it->second << endl;
        it++;
    }
    return os;
}

template <class T>
inline istream &operator>>(istream &is, map<int, T> m)
{
    int i, imax, j;
    T t;

    m.clear();
    is >> tab >> imax >> endl;
    i = 0;
    while (i < imax)
    {
        is >> j >> tab >> t >> endl;
        m[j] = t;
        i++;
        j = m.size();
    }
    return is;
}

void read_file(const char *);
void write_file(const char *);
#endif
