/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS coFeedbackParser
//
// This class @@@
//
// Initial version: 2004-07-14 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

#include "coFeedbackParser.h"
#include <assert.h>

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Static Variable initializers
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// read from istr untl an non-escaped delim char appears
static void readString(istream &istr, string &str, char delim)
{
    string readPart;
    str = "";
    bool readMore;
    do
    {
        readMore = false;
        std::getline(istr, readPart, delim);
        str += readPart;
        int lastIdx = str.size() - 1;
        char lastChar = str[lastIdx];
        if (lastChar == '\\')
        {
            readMore = true;
            str[lastIdx] = delim;
        }
    } while (!istr.eof() && readMore);
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Constructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

coFeedbackParser::coFeedbackParser(const string &interactorAttrib)
{
    istringstream istr(interactorAttrib);

    // 1st must be an 'X', otherwise it's not correct
    char char_X;

    istr >> char_X;

    if (char_X != 'X')
    {
        cerr << "Illegal INTERACTOR argument" << endl;
        return;
    }

    string coFeedback;

    istr >> module_ >> instance_ >> host_ >> plugin_ >> coFeedback;

    if (coFeedback != "coFeedback:")
    {
        cerr << "Illegal INTERACTOR argument" << endl;
        return;
    }

    int numPara, numStrings;
    istr >> numPara >> numStrings;

    while (istr >> char_X)
        if (char_X == '!')
            break;

    string xx;
    int i;
    for (i = 0; i < numPara; i++)
    {
        ParaDesc desc;
        readString(istr, desc.name, '!');
        readString(istr, desc.type, '!');
        readString(istr, desc.value, '!');
        paraMap_[desc.name] = desc;
    }

    for (i = 0; i < numStrings; i++)
    {
        string value;
        readString(istr, value, '!');
        stringList_.push_back(value);
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

coFeedbackParser::~coFeedbackParser()
{
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Operations
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

bool
coFeedbackParser::getIntScalar(const string &name, int &res)
{
    string value;
    if (!findPara(name, "INTSCA", value))
        return false;

    res = atoi(value.c_str());
    return true;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

bool
coFeedbackParser::getBoolean(const string &name, bool &res)
{
    string value;
    if (!findPara(name, "INTSCA", value))
        return false;

    if (value == "FALSE")
    {
        res = false;
        return true;
    }

    if (value == "TRUE")
    {
        res = true;
        return true;
    }

    return false;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

bool coFeedbackParser::getFloatScalar(const string &name, float &res)
{
    string value;
    if (!findPara(name, "FLOSCA", value))
        return false;

    res = atof(value.c_str());
    return true;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

bool coFeedbackParser::getFloatVector(const string &name,
                                      float &x, float &y, float &z)
{
    string value;
    if (!findPara(name, "FLOVEC", value))
        return false;

    int numVec;
    int sres = sscanf(value.c_str(), "%d %f %f %f", &numVec, &x, &y, &z);

    if (sres < 4 || numVec != 3)
    {
        cerr << name << "is not a 3d FLOVEC parameter" << endl;
    }

    return true;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

bool coFeedbackParser::getFloatVector(const string &name, int idx,
                                      float &val)
{
    string value;
    if (!findPara(name, "FLOVEC", value))
        return false;

    istringstream istr(value);
    int i;
    for (i = 0; i < idx; i++)
        if (!(istr >> val))
        {
            cerr << name << " does not have " << idx + 1 << " compoments" << endl;
            return false;
        }

    return true;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

bool coFeedbackParser::getIntVector(const string &name, int &x, int &y, int &z)
{
    string value;
    if (!findPara(name, "INTVEC", value))
        return false;

    int numVec;
    int sres = sscanf(value.c_str(), "%d %d %d %d", &numVec, &x, &y, &z);

    if (sres < 4 || numVec != 3)
    {
        cerr << name << " is not a 3d INTVEC parameter" << endl;
    }

    return true;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

bool coFeedbackParser::getIntVector(const string &name, int idx, int &val)
{
    string value;
    if (!findPara(name, "INTVEC", value))
        return false;

    istringstream istr(value);
    int i;
    for (i = 0; i < idx; i++)
        if (!(istr >> val))
        {
            cerr << name << " does not have " << idx + 1 << " compoments" << endl;
            return false;
        }

    return true;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

bool coFeedbackParser::getChoice(const string &name, int &val)
{
    string value;
    if (!findPara(name, "CHOICE", value))
        return false;

    istringstream istr(value);

    // number of choice labels
    int numLabels;

    if (!(istr >> numLabels))
    {
        cerr << "INTERACTOR element " << name << " illegally formatted" << endl;
        return true;
    }

    // read choice value
    if (istr >> val)
        return true;
    else
    {
        cerr << "INTERACTOR element " << name << " illegally formatted" << endl;
        return false;
    }
}

bool coFeedbackParser::getChoice(const string &name, int &val,
                                 list<string> &choices)
{
    string value;
    if (!findPara(name, "CHOICE", value))
        return false;

    istringstream istr(value);

    // number of choice labels
    int numLabels;

    if (!(istr >> numLabels))
    {
        cerr << "INTERACTOR element " << name << " illegally formatted" << endl;
        return false;
    }

    // read choice value
    if (!(istr >> val))
    {
        cerr << "INTERACTOR element " << name << " illegally formatted" << endl;
        return false;
    }

    // skip 1st $ sign
    string label;
    readString(istr, label, '$');

    // read labels
    choices.clear();
    int i;
    for (i = 0; i < numLabels; i++)
    {
        readString(istr, label, '$');
        choices.push_back(label);
    }

    return true;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Attribute request/set functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void coFeedbackParser::print(ostream &str)
{
    str << "******************************************************" << endl;
    map<string, ParaDesc>::iterator pIter;
    str << "** coFeedbackParser: Found " << paraMap_.size() << " Parameters" << endl;
    for (pIter = paraMap_.begin(); pIter != paraMap_.end(); pIter++)
    {
        str << "** "
            << std::setw(25) << pIter->second.name
            << std::setw(10) << pIter->second.type << "   "
            << pIter->second.value
            << endl;
    }

    str << "******************************************************" << endl;
    vector<string>::iterator sIter;
    str << "** coFeedbackParser: Found " << stringList_.size() << " Strings" << endl;
    for (sIter = stringList_.begin(); sIter != stringList_.end(); sIter++)
    {
        str << "** \"" << (*sIter) << "\"" << endl;
    }
    str << "******************************************************" << endl;
}

// get Feedback parameters
const string &coFeedbackParser::moduleName()
{
    return module_;
}

int coFeedbackParser::moduleInstance()
{
    return instance_;
}

string coFeedbackParser::moduleNameInstance()
{
    char buffer[1024];
    sprintf(buffer, "%d", instance_);
    return module_ + "_" + buffer;
}

const string &coFeedbackParser::moduleHost()
{
    return host_;
}

const string &coFeedbackParser::pluginName()
{
    return plugin_;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Internally used functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

bool coFeedbackParser::findPara(const string &name, const string &type, string &value)
{
    map<string, ParaDesc>::iterator iter;
    iter = paraMap_.find(name);
    if (iter != paraMap_.end())
    {
        if ((*iter).second.type == type)
        {
            value = (*iter).second.value;
            return true;
        }
        else
            cerr << "parameter " << name << " is not " << type << endl;
    }
    else
    {
        cerr << " did not find parameter " << name << endl;
    }
    return false;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++ Prevent auto-generated functions by assert or implement
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/// Copy-Constructor: NOT IMPLEMENTED
coFeedbackParser::coFeedbackParser(const coFeedbackParser &)
{
    assert(0);
}

/// Assignment operator: NOT IMPLEMENTED
coFeedbackParser &coFeedbackParser::operator=(const coFeedbackParser &)
{
    assert(0);
    return *this;
}

/// Default constructor: NOT IMPLEMENTED
coFeedbackParser::coFeedbackParser()
{
    assert(0);
}
