/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2003 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class PyFile                          ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 24.03.2003                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <cstdio>
#include "PyFile.h"
#include <util/string_util.h>

// static helpers
static std::string
pyQuote(const std::string &str, const char *quotmark = "\"")
{
    std::string ret = replace(str, "\\", "\\\\", -1);
    char repl[3] = { '\\', quotmark ? *quotmark : '\"', '\0' };
    return replace(ret, quotmark, repl, -1);
}

static std::string
pyReplace(const std::string &str, const char *c, const char *fill = "")
{
    size_t pos = str.find_first_of(c);

    std::string fStr(fill);
    std::string ret(str);
    size_t last(0);

    std::string res;
    while (pos != std::string::npos)
    {
        res = res + str.substr(last, pos - last) + fStr;
        last = pos + 1;
        pos = str.find_first_of(c, pos + 1);
        ret = res;
    }
    if (pos != str.size() - 1)
    {
        res = res + str.substr(last, str.size() - last);
        ret = res;
    }
    return ret;
}

//
// Constructor
//
PyFile::PyFile()
{
}

PyFile::PyFile(Skeletons *skels, ostream &sDiag)
    : NetFile(skels, sDiag)
{
}

//
// Destructor
//
PyFile::~PyFile()
{
}

std::string
PyFile::createArgumentStr(const ParamSkel &p) const
{

    std::string argStr("( ");

    // create Argument due to type of parameter
    if (p.getType() == "Vector" || p.getType() == "FloatVector" || p.getType() == "IntVector")
    {
        std::string vStr = strip(p.getValue());

        int fill(0);
        size_t pos = vStr.find_first_of(" ");
        if (pos != std::string::npos)
            fill++;
        pos = vStr.find_first_of(" ", pos + 1);
        if (pos != std::string::npos)
            fill++;

        if (fill == 0)
        {
            vStr = vStr + std::string(" 0 0");
        }

        if (fill == 1)
        {
            vStr = vStr + std::string(" 0");
        }

        vStr = pyReplace(vStr, " ", ", ");

        argStr = argStr + vStr;
    }
    else if (p.getType() == "Scalar" || p.getType() == "FloatScalar" || p.getType() == "IntScalar")
    {
        std::string vStr = strip(p.getValue());
        argStr = argStr + vStr;
    }
    else if (p.getType() == "String" || p.getType() == "Boolean" || p.getType() == "BrowserFilter")
    {
        std::string vStr = strip(p.getValue());
        argStr = argStr + std::string("\"") + vStr + std::string("\"");
    }
    else if (p.getType() == "Browser")
    {
        std::string vStr = strip(p.getValue());

#if 0
      size_t pos = vStr.find_first_of(" ");
      if ( pos != std::string::npos )
      {
         vStr = vStr.substr(0,pos);
      }
#endif
        argStr = argStr + std::string("\"") + pyQuote(vStr) + std::string("\"");
    }
    else if ((p.getType() == "Choice") || (p.getType() == "choice"))
    {
        std::string vStr = strip(p.getValue());
        size_t pos = vStr.find_first_of(" ");
        if (pos != std::string::npos)
        {
            vStr = vStr.substr(0, pos);
        }
        argStr = argStr + vStr;
    }
    else if (p.getType() == "Slider" || p.getType() == "FloatSlider" || p.getType() == "IntSlider")
    {
        std::string vStr = strip(p.getValue());
        vStr = pyReplace(vStr, " ", ", ");

        argStr = argStr + vStr;
    }
    else
    {
        argStr += "\"" + pyQuote(strip(p.getValue())) + "\"";
    }

    argStr = argStr + std::string(" )");
    return argStr;
}

ostream &
operator<<(ostream &s, const PyFile &f)
{

    s << "#" << endl;
    s << "#   converted with COVISE (C) map_converter" << endl;
    s << "#   from " << f.inputFileName_ << endl;
    s << "#" << endl;

    s << endl;

    s << "#" << endl;
    s << "# create global net" << endl;
    s << "#" << endl;
    s << "theNet = net()" << endl;

    for (int i = 0; i < f.modules_.size(); ++i)
    {
        // create py-module
        std::string modName(f.modules_[i].getName());
        int instance(f.modules_[i].getNetIndex());
        char nn[32];
        sprintf(nn, "%d", instance);
        std::string varName(modName + std::string("_") + std::string(nn));
        s << "#" << endl;
        s << "# MODULE: " << modName << endl;
        s << "#" << endl;
        s << varName + std::string(" = ") + modName + std::string("()") << endl;
        // add it to the net
        s << "theNet.add( " << varName << " )" << endl;
        // add also the position
        s << varName + std::string(".setPos( ");
        sprintf(nn, "%d", f.modules_[i].getXPos());
        s << nn + std::string(",");
        sprintf(nn, "%d", f.modules_[i].getYPos());
        s << nn + std::string(" )") << endl;
        // the parameters of Renderer should stay invisible
        if (modName != std::string("Renderer"))
        {

            s << "#" << endl;
            s << "# set parameter values " << endl;
            s << "#" << endl;

            // create set functions for each module's parameter
            int j(0);
            while (!f.modules_[i].getParam(j).empty())
            {
                ParamSkel p = f.modules_[i].getParam(j);
                //std::cerr << "paraname: " << p.getName() << ", val: " << p.getValue() << std::endl;
                std::string cParamName = pyReplace(p.getName(), "/");
                cParamName = pyReplace(cParamName, ":");
                cParamName = pyReplace(cParamName, "(");
                cParamName = pyReplace(cParamName, ")");
                cParamName = pyReplace(cParamName, "*");
                cParamName = pyReplace(cParamName, "-");
                cParamName = pyReplace(cParamName, "+");
                cParamName = pyReplace(cParamName, " ", "_");
                s << varName + std::string(".set_") + cParamName + f.createArgumentStr(p) << endl;
                j++;
            }
        }
    }

    s << "#" << endl;
    s << "# CONNECTIONS" << endl;
    s << "#" << endl;

    for (int i = 0; i < f.connections_.size(); ++i)
    {
        Connection ci(f.connections_[i]);
        s << "theNet.connect( "
          << ci.frmModName() + std::string("_") + ci.frmModIdx()
          << ", \"" << ci.frmPort()
          << "\", " << ci.toModName() + std::string("_") + ci.toModIdx()
          << ", \"" << ci.toPort()
          << "\" )" << endl;
    }

    s << "#" << endl;
    s << "# uncomment the following line if you want your script to be executed after loading" << endl;
    s << "#" << endl;
    s << "#runMap()" << endl;

    s << "#" << endl;
    s << "# uncomment the following line if you want exit the COVISE-Python interface" << endl;
    s << "#" << endl;
    s << "#sys.exit()" << endl;

    return s;
}
