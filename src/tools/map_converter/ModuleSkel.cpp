/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class ParamSkel                       ++
// ++                                     ModuleSkel                      ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 10.01.2001                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "ModuleSkel.h"
#include "Translations.h"
#include <util/string_util.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

/////////////////////////// class SkelObj ////////////////////////////////

SkelObj::SkelObj()
    : empty_(1)
    , altNames_(NULL)
    , numAltNames_(0)
{
}

SkelObj::SkelObj(const std::string &name)
    : empty_(0)
    , name_(name)
    , altNames_(NULL)
    , numAltNames_(0)
{
}

SkelObj::SkelObj(const SkelObj &obj)
    : empty_(obj.empty_)
    , name_(obj.name_)
    , altNames_(NULL)
    , numAltNames_(obj.numAltNames_)
{
    altNames_ = new std::string[numAltNames_];
    int i;
    for (i = 0; i < numAltNames_; ++i)
    {
        altNames_[i] = obj.altNames_[i];
    }
}

void
SkelObj::addAltName(const std::string &name)
{
    std::string *tmp = new std::string[numAltNames_ + 1];
    int i;
    for (i = 0; i < numAltNames_; ++i)
    {
        tmp[i] = altNames_[i];
    }

    delete[] altNames_;

    tmp[numAltNames_] = name;
    ++numAltNames_;

    altNames_ = tmp;
}

// return the object name
const std::string
SkelObj::getName() const
{
    return name_;
}

int
SkelObj::nameValid(const std::string &name) const
{
    for (int i = 0; i < numAltNames_; ++i)
    {
        if (name == altNames_[i])
            return 1;
    }

    if (name == name_)
        return 1;

    // return FALSE if no name was found
    return 0;
}

SkelObj::~SkelObj()
{
    delete[] altNames_;
}

/////////////////////////// class ParamSkel ////////////////////////////////

// Constructors
ParamSkel::ParamSkel()
    : SkelObj()
    , type_("")
    , value_("")
    , desc_("")
    , mode_("IMM")
    , select_(-1)
    , vectDim_(-1)
{
}

ParamSkel::ParamSkel(const char *name,
                     const char *type,
                     const char *def,
                     const char *desc,
                     const char *mode,
                     const int &sel)
    : SkelObj(name)
    , type_(type)
    , value_(def)
    , desc_(desc)
    , mode_(mode)
    , select_(sel)
{
    setVectDim();
}

ParamSkel::ParamSkel(const ParamSkel &p)
    : SkelObj(p)
    , type_(p.type_)
    , value_(p.value_)
    , desc_(p.desc_)
    , mode_(p.mode_)
    , select_(p.select_)
    , vectDim_(p.vectDim_)
{
}

// assignment operator
const ParamSkel &
    ParamSkel::
    operator=(const ParamSkel &p)
{
    if (this == &p)
        return *this;
    empty_ = p.empty_;
    name_ = p.name_;
    type_ = p.type_;
    value_ = p.value_;
    desc_ = p.desc_;
    mode_ = p.mode_;
    select_ = p.select_;
    numAltNames_ = p.numAltNames_;
    vectDim_ = p.vectDim_;

    altNames_ = new std::string[numAltNames_];
    int i;
    for (i = 0; i < numAltNames_; ++i)
    {
        altNames_[i] = p.altNames_[i];
    }

    return *this;
}

ostream &
operator<<(ostream &s, const ParamSkel &para)
{
    s << para.name_ << endl;
    s << para.type_ << endl;
    s << para.desc_ << endl;
    s << para.enforceDim() << endl;
    s << "" << endl;
    s << para.select_ << endl;

    return s;
}

// destructor
ParamSkel::~ParamSkel()
{
}

void
ParamSkel::setVectDim()
{
    // do nothing if *this is not a vector parameter
    if (type_ != "Vector" && type_ != "IntVector" && type_ != "FloatVector")
        return;

    std::string value(value_);

    int idx(value.find_first_of(" "));
    int i(-1);
    int cnt(0);

    while (idx != std::string::npos)
    {
        if (!value.substr(i + 1, idx - i - 1).empty())
            cnt++;
        i = idx;
        idx = value.find_first_of(" ", idx + 1);
    }

    int len(value.size());
    if (!value.substr(i + 1, len - i).empty())
        cnt++;

    //    cerr << "ParamSkel::setVectDim() dimension of <" << value << "> : " << cnt << endl;

    vectDim_ = cnt;
}

// return vlaue_ string with vectDim_ Komponents
std::string
ParamSkel::enforceDim() const
{
    // simply return value if *this is not a vector parameter
    if (type_ != "Vector" && type_ != "IntVector" && type_ != "FloatVector")
        return value_;

    std::string value(strip(value_));

    int idx(value.find_first_of(" "));
    int i(-1);
    std::string ret;

    int j;
    for (j = 0; j < vectDim_ - 1; ++j)
    {
        if (!value.substr(i + 1, idx - i - 1).empty())
            ret = ret + value.substr(i + 1, idx - i - 1) + std::string(" ");
        i = idx;
        idx = value.find_first_of(" ", idx + 1);
    }

    if (idx == std::string::npos)
    {
        int len(value.size());
        if (!value.substr(i + 1, len - i).empty())
            ret = ret + value.substr(i + 1, len - i);
    }
    else
    {
        if (!value.substr(i + 1, idx - i - 1).empty())
            ret = ret + value.substr(i + 1, idx - i - 1);
    }

    return ret;
}

void
ParamSkel::setValue(const std::string &val)
{
    value_ = val;
}

/////////////////////////// class ChoiceParamSkel ////////////////////////////////

ChoiceParamSkel::ChoiceParamSkel()
    : ParamSkel()
{
}

ChoiceParamSkel::ChoiceParamSkel(const char *name,
                                 const char *type,
                                 const char *val,
                                 const char *desc,
                                 const char *mode,
                                 const int &sel)
    :

    ParamSkel(name, type, val, desc, mode, sel)
{
}

ChoiceParamSkel::ChoiceParamSkel(const ParamSkel &p)
    : ParamSkel(p)
{
}

// sets value_ to a reasonable value
void
ChoiceParamSkel::checkValue(const std::string &val)
{
    // return if empty val or empty value_;
    if (val.empty() || value_.empty())
        return;

    // always accept IMM choices as-is  rm: add count check here
    if (mode_ == "IMM")
    {
        value_ = val;
        return;
    }

    // we assume value_ as the default and parse it
    std::string pat(" ");
    int end = val.find_first_of(pat.c_str());
    int beg = 0;
    int cnt = 0;

    int actChoice = 0;
    std::string actEntry;

    // we obtain the name of the chosen entry
    while (end != std::string::npos)
    {
        std::string part(val.substr(beg, (end - beg)));
        if (cnt == 0)
        {
            actChoice = atoi(part.c_str());
        }
        else
        {
            if (cnt == actChoice) // in this case this is correct the number of choices start at 1
            {
                actEntry = part;
            }
        }
        beg = end + pat.size();
        end = val.find_first_of(pat.c_str(), beg);
        cnt++;
    }

    // now we check if the default-choices in value_ contain actEntry and modify
    // value_ accordingly

    int pos = 0;
    cnt = 0;
    end = value_.find_first_of(pat.c_str());
    int firstBlnk = end;
    // we obtain the name of the chosen entry
    while (end != std::string::npos)
    {
        std::string part(value_.substr(beg, (end - beg)));
        if (cnt > 0)
        {
            if (part == actEntry)
            {
                pos = cnt;
            }
        }
        beg = end + pat.size();
        end = value_.find_first_of(pat.c_str(), beg);
        cnt++;
    }
    int numDefChoices = cnt;

    int len = value_.size() - firstBlnk;
    std::string defTail(value_.substr(firstBlnk, len));

    defTail = strip(defTail.c_str()); //Note: strip( CoNewString ) and strip( char * ) are distinct

    if (pos == 0)
    {

        cerr << " WARNING: the choice "
             << actEntry
             << " could not be found in parameter "
             << name_ << endl;

        if (actChoice <= numDefChoices)
        {
            char num[55];
            sprintf(num, "%d", actChoice);
            value_ = std::string(num) + std::string(" ") + defTail;
            cerr << "           using choice # " << actChoice << " instead! " << endl;
        }
        else
        {
            cerr << "           using default for choice parameter "
                 << endl;
        }
        return;
    }
    else
    {
        char num[55];
        sprintf(num, "%d", pos);
        value_ = std::string(num) + std::string(" ") + defTail;
    }
    return;
}

/////////////////////////// class PortSkel ////////////////////////////////

// Constructors
PortSkel::PortSkel()
    : SkelObj()
    , type_("")
    , desc_("")
    , genDep_("")
    , intPortName_("")
    , what_(PIN)
    , parent_("")
    , modIdx_(1)
    , replacePolicy_(Translations::NONE)
{
}

PortSkel::PortSkel(const char *name,
                   const char *type,
                   const char *text,
                   const char *genDep,
                   const PortCharacter &pc)
    : SkelObj(name)
    , type_(type)
    , desc_(text)
    , genDep_(genDep)
    , intPortName_("")
    , what_(pc)
    , netIdx_(1)
    , parent_("")
    , modIdx_(1)
    , replacePolicy_(Translations::NONE)

{
}

PortSkel::PortSkel(const PortSkel &p)
    : SkelObj(p)
    , type_(p.type_)
    , desc_(p.desc_)
    , genDep_(p.genDep_)
    , intPortName_(p.intPortName_)
    , what_(p.what_)
    , netIdx_(p.netIdx_)
    , parent_(p.parent_)
    , modIdx_(p.modIdx_)
    , replacePolicy_(p.replacePolicy_)
{
}

// assignment operator
const PortSkel &
    PortSkel::
    operator=(const PortSkel &p)
{
    if (this == &p)
        return *this;
    name_ = p.name_;
    type_ = p.type_;
    desc_ = p.desc_;
    genDep_ = p.genDep_;
    intPortName_ = p.intPortName_;
    what_ = p.what_;
    empty_ = p.empty_;
    netIdx_ = p.netIdx_;
    parent_ = p.parent_;
    modIdx_ = p.modIdx_;
    numAltNames_ = p.numAltNames_;
    replacePolicy_ = p.replacePolicy_;

    altNames_ = new std::string[numAltNames_];
    int i;
    for (i = 0; i < numAltNames_; ++i)
    {
        altNames_[i] = p.altNames_[i];
    }

    return *this;
}

void
PortSkel::setParentInfo(const std::string &parent, const int &num)
{
    parent_ = parent;
    modIdx_ = num;

    intPortName_ = std::string("");
}

int
PortSkel::nameValid(const std::string &name) const
{
    if (replacePolicy_ == Translations::NONE)
    {
        //	cerr << "PortSkel::nameValid(..) replacePolicy NONE" << endl;
        for (int i = 0; i < numAltNames_; ++i)
        {
            if (name == altNames_[i])
                return 1;
        }

        if (name == name_)
            return 1;
    }

    if (replacePolicy_ == Translations::TRANSLATIONS)
    {
        for (int i = 0; i < numAltNames_; ++i)
        {
            if (name == altNames_[i])
                return 1;
        }

        //	if (name == name_) return 1;
    }
    // return FALSE if no name was found
    return 0;
}

void
PortSkel::setNetIdx(const int &idx)
{
    netIdx_ = idx;

    intPortName_ = std::string("");
};

// destructor
PortSkel::~PortSkel()
{
}

ostream &operator<<(ostream &s, const PortSkel &prt)
{
    s << prt.name_ << endl;
    s << prt.type_ << endl;
    s << prt.desc_ << endl;
    s << prt.genDep_ << endl;
    s << prt.intPortName_ << endl;

    return s;
}

/////////////////////////// class ModuleSkel ////////////////////////////////
ModuleSkeleton::ModuleSkeleton()
    : SkelObj()
    , desc_("")
    , group_("")
    , host_("")
    , numParams_(0)
    , numInPorts_(0)
    , numOutPorts_(0)
    , X_(0)
    , Y_(0)
    , netIndex_(1)
    , orgNetIndex_(1)
    , params_(NULL)
    , unUsedParams_(NULL)
    , ports_(NULL)
    , altGroups_(NULL)
    , numAltGroups_(0)
{
}

ModuleSkeleton::ModuleSkeleton(const char *name, const char *group, const char *desc)
    : SkelObj(name)
    , desc_(desc)
    , group_(group)
    , host_("")
    , numParams_(0)
    , numInPorts_(0)
    , numOutPorts_(0)
    , X_(0)
    , Y_(0)
    , netIndex_(1)
    , orgNetIndex_(1)
    , params_(NULL)
    , unUsedParams_(NULL)
    , ports_(NULL)
    , altGroups_(NULL)
    , numAltGroups_(0)
    , numOrgPortNames_(0)
{
}

ModuleSkeleton::ModuleSkeleton(const ModuleSkeleton &rm)
    : SkelObj(rm)
    , desc_(rm.desc_)
    , group_(rm.group_)
    , host_(rm.host_)
    , numParams_(rm.numParams_)
    , numInPorts_(rm.numInPorts_)
    , numOutPorts_(rm.numOutPorts_)
    , X_(rm.X_)
    , Y_(rm.Y_)
    , netIndex_(rm.netIndex_)
    , orgNetIndex_(rm.orgNetIndex_)
    , orgModName_(rm.orgModName_)
    , numAltGroups_(rm.numAltGroups_)
    , numOrgPortNames_(rm.numOrgPortNames_)

{
    int i; // congratulations to SGI
    params_ = new ParamSkel[numParams_];
    unUsedParams_ = new int[numParams_];

    for (i = 0; i < numParams_; ++i)
    {
        params_[i] = rm.params_[i];
        unUsedParams_[i] = rm.unUsedParams_[i];
    }

    int numPorts = numInPorts_ + numOutPorts_;
    ports_ = new PortSkel[numPorts];
    for (i = 0; i < numPorts; ++i)
        ports_[i] = rm.ports_[i];

    altGroups_ = new std::string[numAltGroups_];
    for (i = 0; i < numAltGroups_; ++i)
        altGroups_[i] = rm.altGroups_[i];
}

const ModuleSkeleton &
    ModuleSkeleton::
    operator=(const ModuleSkeleton &rm)
{
    if (this == &rm)
        return *this;

    numParams_ = rm.numParams_;
    numInPorts_ = rm.numInPorts_;
    numOutPorts_ = rm.numOutPorts_;
    name_ = rm.name_;
    group_ = rm.group_;
    host_ = rm.host_;
    desc_ = rm.desc_;
    empty_ = rm.empty_;
    X_ = rm.X_;
    Y_ = rm.Y_;
    netIndex_ = rm.netIndex_;
    orgNetIndex_ = rm.orgNetIndex_;
    orgModName_ = rm.orgModName_;
    numAltNames_ = rm.numAltNames_;
    numAltGroups_ = rm.numAltGroups_;
    numOrgPortNames_ = rm.numOrgPortNames_;

    delete[] params_;
    delete[] ports_;
    delete[] unUsedParams_;
    delete[] altNames_;

    params_ = new ParamSkel[numParams_];
    unUsedParams_ = new int[numParams_];
    int i;
    for (i = 0; i < numParams_; ++i)
    {
        params_[i] = rm.params_[i];
        unUsedParams_[i] = rm.unUsedParams_[i];
    }
    int numPorts = numInPorts_ + numOutPorts_;
    ports_ = new PortSkel[numPorts];
    for (i = 0; i < numPorts; ++i)
        ports_[i] = rm.ports_[i];

    altNames_ = new std::string[numAltNames_];
    for (i = 0; i < numAltNames_; ++i)
        altNames_[i] = rm.altNames_[i];

    altGroups_ = new std::string[numAltGroups_];
    for (i = 0; i < numAltGroups_; ++i)
        altGroups_[i] = rm.altGroups_[i];

    return *this;
}

// destructor
ModuleSkeleton::~ModuleSkeleton()
{
    delete[] params_;
    delete[] ports_;
    delete[] unUsedParams_;
    //      delete [] altNames_;
    delete[] altGroups_;
}

// add a port to the Module
void
ModuleSkeleton::add(const PortSkel &port)
{
    int numPorts = numInPorts_ + numOutPorts_;
    PortSkel *tmp = new PortSkel[numPorts + 1];
    int i;
    for (i = 0; i < numPorts; ++i)
    {
        tmp[i] = ports_[i];
    }

    delete[] ports_;

    i = numPorts;
    tmp[i] = port;
    tmp[i].setParentInfo(name_, i);

    if (port.getCharacter() == PIN)
        numInPorts_++;
    else
        numOutPorts_++;

    ports_ = tmp;
}

// add a parameter
void
ModuleSkeleton::add(const ParamSkel &param)
{
    ParamSkel *tmp = new ParamSkel[numParams_ + 1];
    int *iTmp = new int[numParams_ + 1];

    int i;
    for (i = 0; i < numParams_; ++i)
    {
        tmp[i] = params_[i];
        iTmp[i] = unUsedParams_[i];
    }

    delete[] params_;
    delete[] unUsedParams_;

    tmp[numParams_] = param;
    iTmp[numParams_] = 1;

    numParams_++;

    params_ = tmp;
    unUsedParams_ = iTmp;
}

void
ModuleSkeleton::deleteAllParams()
{
    delete[] params_;
    delete[] unUsedParams_;
    params_ = NULL;
    unUsedParams_ = NULL;
    numParams_ = 0;
}

const ParamSkel &
ModuleSkeleton::getParam(const std::string &name, const std::string &type)
{
    int i;
    for (i = 0; i < numParams_; ++i)
    {
        if ((params_[i].nameValid(name)) && (type == params_[i].getType() || (type == "Vector" && (params_[i].getType() == "FloatVector" || params_[i].getType() == "IntVector")) || (type == "Scalar" && (params_[i].getType() == "FloatScalar" || params_[i].getType() == "IntScalar"))))
        {
            unUsedParams_[i] = -1;
            return params_[i];
        }
    }
    return emptyParam_;
}

ParamSkel
ModuleSkeleton::getParam(const int &i) const
{
    if ((i >= 0) && (i < numParams_))
    {
        return params_[i];
    }
    return emptyParam_;
}

const PortSkel &
ModuleSkeleton::getPort(const std::string &name)
{
    int i;
    int numPorts = numInPorts_ + numOutPorts_;
    for (i = 0; i < numPorts; ++i)
    {
        if (ports_[i].nameValid(name))
        {
            return ports_[i];
        }
    }
    return emptyPort_;
}

int
ModuleSkeleton::getUnusedParams(ParamSkel *pArray, const int &n)
{
    int cnt = 0;
    if (pArray)
    {
        int i = 0;
        while ((i < numParams_) && (cnt <= n))
        {
            if (unUsedParams_[i] > 0)
            {
                // 			cerr << "ModuleSkeleton::getUnusedParams(..) found unused param "
                // 			     << i
                // 			     << " : "
                // 			     << params_[i].getName() << endl;
                pArray[cnt] = params_[i];
                ++cnt;
            }
            ++i;
        }
    }
    else
    {
        int i;
        for (i = 0; i < numParams_; ++i)
        {
            if (unUsedParams_[i] > 0)
                cnt++;
        }
    }

    return cnt;
}

void
ModuleSkeleton::checkPortPolicy(const std::string &pName, const int &done)
{
    int i;
    int numPorts = numInPorts_ + numOutPorts_;
    //

    if (!done)
    {
        for (i = 0; i < numPorts; ++i)
        {
            if (ports_[i].getName() == pName)
            {
                numOrgPortNames_++;
            }
        }
    }
    else
    {
        if (numOrgPortNames_ == numPorts)
        {
            for (i = 0; i < numPorts; ++i)
            {
                ports_[i].setReplacePolicy(Translations::NONE);
            }
            // 	    cerr << "ModuleSkeleton::checkPortPolicy(..) reset port replace policy for module "
            // 		 << name_ << endl;
        }
    }
}

ostream &operator<<(ostream &s, const ModuleSkeleton &mod)
{
    s << "# Module " << mod.name_ << endl;
    s << mod.name_ << endl;
    s << mod.netIndex_ << endl;
    s << mod.host_ << endl;
    s << mod.group_ << endl;
    s << mod.desc_ << endl;
    s << mod.X_ << endl;
    s << mod.Y_ << endl;
    s << mod.numInPorts_ << endl;

    int i;
    int totNum = mod.numInPorts_ + mod.numOutPorts_;
    for (i = 0; i < totNum; ++i)
    {
        if (mod.ports_[i].getCharacter() == PIN)
            s << mod.ports_[i];
    }

    s << mod.numOutPorts_ << endl;

    for (i = 0; i < totNum; ++i)
    {
        if (mod.ports_[i].getCharacter() == POUT)
            s << mod.ports_[i];
    }

    s << mod.numParams_ << endl;

    for (i = 0; i < mod.numParams_; ++i)
    {
        s << mod.params_[i];
    }

    s << 0 << endl;

    return s;
}

// add an alternative group name the module
void
ModuleSkeleton::addAltGroup(const std::string &grp)
{

    std::string *tmp = new std::string[numAltGroups_ + 1];
    int i;
    for (i = 0; i < numAltGroups_; ++i)
    {
        tmp[i] = altGroups_[i];
    }

    delete[] altGroups_;

    tmp[numAltGroups_] = grp;
    ++numAltGroups_;

    altGroups_ = tmp;
}

int
ModuleSkeleton::groupValid(const std::string &group)
{
    int i;

    if (group == group_)
        return 1;

    for (i = 0; i < numAltGroups_; ++i)
    {
        if (group == altGroups_[i])
        {
            return 1;
        }
    }

    return 0;
}

void
ModuleSkeleton::setNetIndex(const int &idx)
{
    netIndex_ = idx;

    int i;
    int totNum = numInPorts_ + numOutPorts_;

    for (i = 0; i < totNum; ++i)
    {
        ports_[i].setNetIdx(netIndex_);
    }
}
