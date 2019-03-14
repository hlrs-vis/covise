/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef REGCLASS_H
#define REGCLASS_H

#include <net/tokenbuffer.h>
#include <map>
#include <set>
#include <memory>
#include <util/coExport.h>
#include <fstream>

class coCharBuffer;
namespace covise
{
class VRBClient;
}

namespace vrb
{
class clientRegClass;
class clientRegVar;
class serverRegVar;
class serverRegClass;
class regClassObserver;
class regVarObserver;
class VrbClientRegistry;
class SessionID;

template<class variableType>
class regClass
{
public:
    typedef std::map<const std::string, std::shared_ptr<variableType>> VariableMap;
    regClass<variableType>(const std::string &n, int ID)
        : name(n)
        , classID(ID)
        , isDel(false)
    {
    }
    /// get Class ID
    int getID()
    {
        return (classID);
    }
    void setID(int id)
    {
        classID = id;
    }
    const std::string &getName()
    {
        return (name);
    }
    ///creates a  a regvar entry  in the map
    void append(variableType * var)
    {
        myVariables[var->getName()].reset(var);
    }
    /// getVariableEntry, returns NULL if not found
    variableType *getVar(const std::string &n)
    {
        auto it = myVariables.find(n);
        if (it == myVariables.end())
        {
            return (NULL);
        }
        return it->second.get();
    }
    /// remove a Variable
    void deleteVar(const std::string &n)
    {
        myVariables.erase(n);
    }
    /// remove some Variables
    void deleteAllNonStaticVars()
    {
        typename VariableMap::iterator it = myVariables.begin();
        while (it != myVariables.end())
        {
            if (it->second->isStatic())
            {
                it = myVariables.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }
    bool isDeleted()
    {
        return isDel;
    }
    void setDeleted(bool isdeleted = true)
    {
        isDel = isdeleted;
        for (const auto var : myVariables)
        {
            var.second->setDeleted(isdeleted);
        }
    }
    ///write the classname and all variables in a .vrbreg file
    void writeClass(std::ofstream &file) {
        file << name;
        file << "\n";
        file << "{";
        file << "\n";
        for (const auto var : myVariables)
        {
            var.second->writeVar(file);
            file << "\n";
        }
        file << "}";

    }
    ///reads the name and value out of stream, return false if class has no variable
    void readVar(std::ifstream &file)
    {

        while (true)
        {
            std::string varName = "invalid";
            int valueSize = -1;
            file >> varName;
            if (varName == "}")
            {
                return;
            }
            varName.pop_back();
            file >> valueSize;
            char *value = new char[valueSize];
            file.read(value, valueSize);
            covise::TokenBuffer tb(value, valueSize);
            myVariables[varName] = createVar(varName, std::move(tb));
            delete[] value; //createVar did copy the tokenbuffer
        }

    };
    virtual std::shared_ptr<variableType> createVar(const std::string &name, covise::TokenBuffer &&value) = 0;
    ~regClass()
    {
    };
protected:
    std::string name;
    int classID = -1;
    bool isDel;
    VariableMap myVariables;
};

template<class classType>
class regVar
{
protected:
    covise::TokenBuffer value;
    std::string name;
    classType *myClass;
    bool staticVar;
    bool isDel;

public:

    regVar(classType *c, const std::string &n, covise::TokenBuffer &v, bool s = 1)
    {
        myClass = c;
        name = n;
        staticVar = s;
        setValue(v);
        isDel = false;
    }
    ~regVar()
    {
        value.delete_data();
    }
    /// returns the value
    covise::TokenBuffer &getValue()
    {
        value.rewind();
        return (value);
    };
    /// returns the class of this variable
    classType *getClass()
    {
        return (myClass);
    };
    /// set value
    inline void setValue(const covise::TokenBuffer &v)
    {
        value.copy(v);
    }
    /// returns true if this Var is static
    int isStatic()
    {
        return (staticVar);
    };
    /// returns the Name
    const std::string &getName()
    {
        return (name);
    };

    bool isDeleted()
    {
        return isDel;
    }
    void setDeleted(bool isdeleted = true)
    {
        isDel = isdeleted;
    }
    void writeVar(std::ofstream &file) 
    {
        file << "    " << name << "; ";
        int length = value.get_length();
        file << length;
        file.write(value.get_data(), value.get_length());
    }
};


class VRBEXPORT clientRegClass : public regClass<clientRegVar>
{
private:
    regClassObserver *_observer = nullptr; //local observer class
    int lastEditor;
    VrbClientRegistry *registry;
public:
    clientRegClass(const std::string &n, int ID, VrbClientRegistry *reg);
    regClassObserver *getLocalObserver()
    {
        return _observer;
    }
    ///attach a observer to the regClass
    void attach(regClassObserver *ob)
    {
        _observer = ob;
    }
    int getLastEditor()
    {
        return lastEditor;
    }
    void setLastEditor(int lastEditor);
    void notifyLocalObserver();
    void resubscribe(const SessionID &sessionID);
    void subscribe(regClassObserver *obs, const SessionID &sessionID);
    covise::VRBClient *getRegistryClient();
    VariableMap &getAllVariables();
    std::shared_ptr<clientRegVar> createVar(const std::string &name, covise::TokenBuffer &&value) override;
};
class VRBEXPORT clientRegVar : public regVar<clientRegClass>
{
private:
    regVarObserver *_observer;
    int lastEditor;
public:
    using regVar::regVar;
    ///returns the clent side observer
    regVarObserver * getLocalObserver()
    {
        return _observer;
    }
    void notifyLocalObserver();
    void subscribe(regVarObserver *ob, const SessionID &sessionID);

    //void attach(regVarObserver *ob)
    //{
    //    _observer = ob;
    //}
    int getLastEditor()
    {
        return lastEditor;
    }
    void setLastEditor(int lastEditor)
    {
        this->lastEditor = lastEditor;
    }
};




class VRBEXPORT regClassObserver
{
public:
    virtual void update(clientRegClass *theChangedClass) = 0;
};
class VRBEXPORT regVarObserver
{
public:
    virtual void update(clientRegVar *theChangedVar) = 0;
};
}
#endif
