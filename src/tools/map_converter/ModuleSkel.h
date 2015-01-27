/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description: Module Converter                                       ++
// ++              Adopt COVISE net files to changed module structures    ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 09.01.2001                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ifndef MODULE_SKELS_H
#define MODULE_SKELS_H

#include <iostream>

enum PortCharacter
{
    PIN,
    POUT
};

//
// base class for all module skeletons
//

class SkelObj
{
public:
    SkelObj();

    SkelObj(const std::string &name);

    SkelObj(const SkelObj &obj);

    virtual const std::string getName() const;

    virtual int empty() const
    {
        return empty_;
    };

    // add an alternative name for parameter obj.
    virtual void addAltName(const std::string &name);
    // returns 1 if name is either the parameter's true name or an alias to it
    virtual int nameValid(const std::string &name) const;

    virtual ~SkelObj();

protected:
    int empty_;
    std::string name_;

    std::string *altNames_;
    int numAltNames_;
};

//store parameter information
class ParamSkel : public SkelObj
{
public:
    ParamSkel();
    ParamSkel(const char *name,
              const char *type,
              const char *val,
              const char *desc,
              const char *mode,
              const int &sel = -1);

    ParamSkel(const ParamSkel &p);

    const ParamSkel &operator=(const ParamSkel &p);

    const std::string &getType() const
    {
        return type_;
    };
    const std::string &getValue() const
    {
        return value_;
    };
    const std::string &getDesc() const
    {
        return desc_;
    };
    const std::string &getMode() const
    {
        return mode_;
    };

    // set methods for data which can be changed individually by the user
    virtual void setValue(const std::string &val);
    virtual void setSelect(const int &sel)
    {
        select_ = sel;
    };

    friend std::ostream &operator<<(std::ostream &s, const ParamSkel &para);

    virtual ~ParamSkel();

protected:
    std::string type_;
    std::string value_;
    std::string desc_;
    std::string mode_;
    int select_;

private:
    // the dimensionality of a vector parameter is set only at the initialization
    // of a module, in the copy-constructur and the assignment operator !!!
    void setVectDim();
    // enforce a value-string with the correct dimensionality
    std::string enforceDim() const;
    int vectDim_;
};

//
// Stores parameter info for choice parameters
// The class has the same functionality as ParamSkel but the
// overloaded setValue method includes a consistency-check of the
// choice parameters' entries.
//
class ChoiceParamSkel : public ParamSkel
{
public:
    ChoiceParamSkel();
    ChoiceParamSkel(const char *name,
                    const char *type,
                    const char *val,
                    const char *desc,
                    const char *mode,
                    const int &sel = -1);

    ChoiceParamSkel(const ParamSkel &p);

    virtual void setValue(const std::string &val)
    {
        checkValue(val);
    };

private:
    // performs consistency-check of the choice's value
    void checkValue(const std::string &val);
};

// store port information
class PortSkel : public SkelObj
{
public:
    PortSkel();
    PortSkel(const char *name,
             const char *type,
             const char *text,
             const char *genDep,
             const PortCharacter &pc);

    PortSkel(const PortSkel &p);

    const PortSkel &operator=(const PortSkel &p);

    const std::string &getType() const
    {
        return type_;
    };
    const std::string &getDesc() const
    {
        return desc_;
    };
    PortCharacter getCharacter() const
    {
        return what_;
    };
    const std::string &getGenDepStr() const
    {
        return genDep_;
    };
    const std::string &getCoObjName() const
    {
        return intPortName_;
    };

    void setReplacePolicy(const int &p)
    {
        replacePolicy_ = p;
    };

    void setNetIdx(const int &idx);

    void setParentInfo(const std::string &parent, const int &num);

    virtual int nameValid(const std::string &name) const;

    friend std::ostream &operator<<(std::ostream &s, const PortSkel &mod);

    virtual ~PortSkel();

private:
    std::string type_;
    std::string desc_;
    std::string genDep_;
    std::string intPortName_;
    PortCharacter what_;
    int netIdx_;
    std::string parent_;
    int modIdx_;
    int replacePolicy_;
};

// store a complete COVISE module profile
class ModuleSkeleton : public SkelObj
{
public:
    ModuleSkeleton();
    ModuleSkeleton(const char *name, const char *group, const char *desc);
    ModuleSkeleton(const ModuleSkeleton &rm);

    const ModuleSkeleton &operator=(const ModuleSkeleton &rm);

    virtual ~ModuleSkeleton();

    const int &getNumParams()
    {
        return numParams_;
    };
    const int &getNumInPorts()
    {
        return numInPorts_;
    };
    const int &getNumOutPorts()
    {
        return numOutPorts_;
    };

    // add port information to the module
    void add(const PortSkel &port);
    // add parameter information to the module
    void add(const ParamSkel &param);

    const std::string &getGroup() const
    {
        return group_;
    };
    const std::string &getHost() const
    {
        return host_;
    };
    const std::string &getDesc() const
    {
        return desc_;
    };
    const int &getNetIndex() const
    {
        return netIndex_;
    };
    const int &getOrgNetIndex() const
    {
        return orgNetIndex_;
    };
    const int &getXPos() const
    {
        return X_;
    };
    const int &getYPos() const
    {
        return Y_;
    };
    const std::string &getOrgModName() const
    {
        return orgModName_;
    };

    // returns 1 if group is either the module's true group or an alias to it
    int groupValid(const std::string &group);

    // query an parameter by name and return corresponding param obj.
    // returns an empty obj. if obj. with name does not exist
    const ParamSkel &getParam(const std::string &name, const std::string &type);

    // returns parameter by (internal) index. If the index not exists an
    // empty parameter is returned.
    // - you may want to remove it later -
    ParamSkel getParam(const int &i) const;

    // query an port by name and return corresponding port obj.
    // returns an empty obj. if obj. withe name does not exist
    const PortSkel &getPort(const std::string &name);

    // returns the number of param objects which have never been queried
    // during the lifetime of *this pArray contains array if ParamSkel
    // pArray has to be allocated outside the method and contains n elements
    int getUnusedParams(ParamSkel *pArray, const int &n);

    void setHost(const std::string &host)
    {
        host_ = host;
    };
    void setPos(int &x, int &y)
    {
        X_ = x;
        Y_ = y;
    };
    void setNetIndex(const int &idx);
    void setOrgNetIndex(const int &idx)
    {
        orgNetIndex_ = idx;
    };
    void setOrgModName(const std::string &nm)
    {
        orgModName_ = nm;
    };
    void setDesc(const std::string &d)
    {
        desc_ = d;
    };

    // deletes all parameter information in *this
    void deleteAllParams();

    // add an alternative group the module could belong to
    void addAltGroup(const std::string &grp);

    // check the port replace policy
    // the  ports replace policy indicates how port names are matched. If set to
    // Translations::NONE port names are identical if they are either identical to
    // the original port name obtained directly from the module or one of its alias names.
    // If set to Translations::TRANSLATIONS port names are identical if they match ONLY the
    // alias names.
    // done=0: an internal flag is set if pName matches the orig. name of one port
    // done=1: perform the real check - the port replace policy is reset if all so far
    //         registered port names match the orig. port names
    void checkPortPolicy(const std::string &pName, const int &done = 0);

    // be aware that parameter information is not written to ostream by default
    friend std::ostream &operator<<(std::ostream &s, const ModuleSkeleton &mod);

private:
    std::string desc_;
    std::string group_;
    std::string host_;

    int numParams_;
    int numInPorts_;
    int numOutPorts_;
    int X_;
    int Y_;
    int netIndex_;
    int orgNetIndex_;
    std::string orgModName_;

    ParamSkel *params_;

    int *unUsedParams_;

    PortSkel *ports_;

    ParamSkel emptyParam_; // dummy for error conditions
    PortSkel emptyPort_; // dummy for error conditions

    std::string *altGroups_;
    int numAltGroups_;
    int numOrgPortNames_;
};
#endif
