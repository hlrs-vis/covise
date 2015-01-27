/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef regVar_H
#define regVar_H

#include <util/coDLList.h>

class netModule;
class coCharBuffer;
class regClass;
class regVar;
class coRegistry;

class observerList
{
    int numObservers;
    int *observers;

public:
    observerList()
    {
        numObservers = 0;
        observers = NULL;
    };
    void addObserver(int recvID);
    void removeObserver(int recvID);
    void serveObservers(regVar *v);
    void informDeleteObservers(regVar *v);
    void copyObservers(regClass *v);

    ~observerList()
    {
        delete[] observers;
    };
};

class regVar
{
private:
    char *value;
    char *name;
    regClass *myClass;
    observerList observers;
    int staticVar;

public:
    regVar(regClass *c, const char *n, const char *v, int s = 1);
    /// returns the value
    const char *getValue()
    {
        return (value);
    };
    /// returns the class of this variable
    regClass *getClass()
    {
        return (myClass);
    };
    /// set value
    void setValue(const char *v);
    /// returns true if this Var is static
    int isStatic()
    {
        return (staticVar);
    };
    /// returns the Name
    const char *getName()
    {
        return (name);
    };
    /// send Value to recvID
    void update(int recvID);
    /// send Value UIs depending on UI variable RegistryMode
    void updateUIs();
    /// add an observer to my list
    void observe(int recvID)
    {
        observers.addObserver(recvID);
        update(recvID);
    };
    /// remove an observer to my list
    void unObserve(int recvID)
    {
        observers.removeObserver(recvID);
    };
    /// get list of Observers
    observerList *getOList()
    {
        return (&observers);
    };
    /**
       * add Variables to Script
       */
    void saveNetwork(coCharBuffer &cb);
    ~regVar();
};

class regClass : public covise::coDLPtrList<regVar *>
{
private:
    char *name;
    int classID;
    observerList observers;

public:
    regClass(const char *n, int ID);
    /// returns the value
    const char *getName()
    {
        return (name);
    };
    /// add a new observer to this class or variable of this class
    void observe(int recvID, const char *variableName = NULL);
    /// remove an observer from this class or variable of this class
    void unObserve(int recvID, const char *variableName = NULL);
    /// get Class ID
    int getID()
    {
        return (classID);
    };
    /// getVariableEntry, returns NULL if not found
    regVar *getVar(const char *name);
    /// get list of Observers
    observerList *getOList()
    {
        return (&observers);
    };
    /// remove a Variable
    void deleteVar(const char *n);
    /// remove some Variables
    void deleteAllNonStaticVars();
    /**
       * add this Class to Script
       */
    void saveNetwork(coCharBuffer &cb);
    ~regClass()
    {
        delete[] name;
    };
};
/**
 * List of userinterface Variables
 * @author Uwe Woessner
 * @version 1.0
 */
class coRegistry : public covise::coDLPtrList<regClass *>
{
public:
    /// constructor initializes Variables with values from yac.config:regVariables
    int regMode;
    coRegistry();
    /// getClassEntry, returns NULL if not found
    regClass *getClass(const char *name, int ID = 0);
    /// set a Value or create new Entry
    void setVar(const char *className, int ID, const char *name, const char *value);
    /// create new Entry
    void create(const char *className, int ID, const char *name, int s = 0);
    /// remove an Entry
    void deleteEntry(const char *className, int ID, const char *name);
    /// remove all Entries from one Module
    void deleteEntry(int moduleID);
    /// add a new observer
    void observe(const char *className, int ID, int recvID, const char *variableName = NULL);
    /// remove an observer
    void unObserve(const char *className, int ID, int recvID, const char *variableName = NULL);
    /// remove all observers for this ID
    void unObserve(int recvID);
    /// get a boolean Variable
    int isTrue(const char *className, int ID, const char *name, int def = 0);
    /**
       * add Registry to Script
       */
    void saveNetwork(coCharBuffer &cb);
    static coRegistry *instance;
};
#endif
