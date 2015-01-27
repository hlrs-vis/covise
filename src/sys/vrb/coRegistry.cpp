/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include "coRegistry.h"
#include "VRBClientList.h"
#include <util/unixcompat.h>
#include <net/tokenbuffer.h>

using namespace std;
using namespace covise;

int debugMode = 0;
int dataCaching = 0;

coRegistry *coRegistry::instance = NULL;
void observerList::addObserver(int recvID)
{
    int i, *newList = new int[numObservers + 1];
    for (i = 0; i < numObservers; i++)
        newList[i] = observers[i];
    delete[] observers;
    observers = newList;
    observers[numObservers] = recvID;
    numObservers++;
}

void observerList::removeObserver(int recvID)
{
    int i;
    for (i = 0; i < numObservers; i++)
    {
        if (observers[i] == recvID)
        {
            i++;
            while (i <= numObservers)
            {
                observers[i - 1] = observers[i];
                i++;
            }
            numObservers--;
            break;
        }
    }
}

void observerList::copyObservers(regClass *c)
{
    int i;
    for (i = 0; i < numObservers; i++)
    {
        c->observe(observers[i], NULL);
    }
}

void observerList::serveObservers(regVar *v)
{
    int i;
    TokenBuffer sb;
    sb << v->getClass()->getName();
    sb << v->getClass()->getID();
    sb << v->getName();
    sb << v->getValue();
    for (i = 0; i < numObservers; i++)
    {
        clients.sendMessageToID(sb, observers[i], COVISE_MESSAGE_VRB_REGISTRY_ENTRY_CHANGED);
    }
}

void observerList::informDeleteObservers(regVar *v)
{
    int i;
    TokenBuffer sb;
    sb << v->getClass()->getName();
    sb << v->getClass()->getID();
    sb << v->getName();
    sb << v->getValue();
    for (i = 0; i < numObservers; i++)
    {
        clients.sendMessageToID(sb, observers[i], COVISE_MESSAGE_VRB_REGISTRY_ENTRY_DELETED);
    }
}

regVar::regVar(regClass *c, const char *n, const char *v, int s)
{
    myClass = c;
    value = NULL;
    name = new char[strlen(n) + 1];
    strcpy(name, n);
    setValue(v);
    staticVar = s;
}

regVar::~regVar()
{
    observers.informDeleteObservers(this);
    getClass()->getOList()->informDeleteObservers(this);
    delete[] name;
    delete[] value;
}

regClass::regClass(const char *n, int ID)
{
    name = new char[strlen(n) + 1];
    strcpy(name, n);
    classID = ID;
}

void regVar::setValue(const char *v)
{
    delete[] value;
    if (!v)
        v = "";
    value = new char[strlen(v) + 1];
    strcpy(value, v);
}

void regVar::updateUIs()
{
    /*if(registry.regMode==0)
       return;
   TokenBuffer sb;
   sb<<myClass->getName();
   sb<<myClass->getID();
   sb<<getName();
   sb<<getValue();
   if(registry.regMode==2)
   {
       hostlist->sendUI(sb,coCtrlMsg::REGISTRY_ENTRY_CHANGED,coMsg::CTRL);
   }
   else if((registry.regMode==1)&&(myClass->getID()==0))
   {
   hostlist->sendUI(sb,coCtrlMsg::REGISTRY_ENTRY_CHANGED,coMsg::CTRL);
   }*/
}

void regVar::update(int recvID)
{
    TokenBuffer sb;
    sb << myClass->getName();
    sb << myClass->getID();
    sb << getName();
    sb << getValue();
    clients.sendMessageToID(sb, recvID, COVISE_MESSAGE_VRB_REGISTRY_ENTRY_CHANGED);
}

/// set a Value or create new Entry
void coRegistry::setVar(const char *className, int ID, const char *name, const char *value)
{
    if (ID == 0)
    {
        if (strcmp(className, "UI") == 0)
        {
            if (strcmp(name, "RegistryMode") == 0)
            {
                regMode = 0;
                if (strcasecmp(value, "ALL") == 0)
                    regMode = 2;
                else if (strcasecmp(value, "Global Only") == 0)
                    regMode = 1;
            }
            else if (strcmp(name, "DebugMode") == 0)
            {
                debugMode = 0;
                if (strcasecmp(value, "true") == 0)
                    debugMode = 1;
                else if (strcasecmp(value, "on") == 0)
                    debugMode = 1;
                else if (value[0] == '1')
                    debugMode = 1;
            }
            else if (strcmp(name, "DataCaching") == 0)
            {
                dataCaching = 0;
                if (strcasecmp(value, "None") == 0)
                    dataCaching = 0;
                else if (strcasecmp(value, "Low") == 0)
                    dataCaching = 1;
                else if (strcasecmp(value, "High") == 0)
                    dataCaching = 2;
            }
        }
    }
    regClass *rc = getClass(className, ID);
    if (!rc)
    {
        regClass *grc;
        rc = new regClass(className, ID);
        append(rc);
        grc = getClass(className, 0);
        if (grc)
        {
            // we have a generic observer for this class name, copy observers
            grc->getOList()->copyObservers(rc);
        }
    }
    regVar *rv = rc->getVar(name);
    if (rv)
    {
        rv->setValue(value);
    }
    else
    {
        rv = new regVar(rc, name, value);
        rc->append(rv);
    }
    rc->getOList()->serveObservers(rv);
    rv->getOList()->serveObservers(rv);
    rv->updateUIs();
}

/// create new Entry
void coRegistry::create(const char *className, int ID, const char *name, int s)
{
    regClass *rc = getClass(className, ID);
    if (!rc)
    {
        regClass *grc;
        rc = new regClass(className, ID);
        append(rc);
        grc = getClass(className, 0);
        if (grc)
        {
            // we have a generic observer for this class name, copy observers
            grc->getOList()->copyObservers(rc);
        }
    }
    regVar *rv = rc->getVar(name);
    if (!rv)
    {
        rv = new regVar(rc, name, "", s);
        rc->append(rv);
    }
    rc->getOList()->serveObservers(rv);
}

/// get a boolean Variable
int coRegistry::isTrue(const char *className, int ID, const char *name, int def)
{
    regClass *rc = getClass(className, ID);
    if (rc)
    {
        regVar *rv = rc->getVar(name);
        if (rv)
        {
            const char *v = rv->getValue();
            if (v[0] == '1')
                return 1;
            if (v[0] == '0')
                return 0;
            if (strcasecmp(v, "true") == 0)
                return 1;
            if (strcasecmp(v, "false") == 0)
                return 0;
            if (strcasecmp(v, "on") == 0)
                return 1;
            if (strcasecmp(v, "off") == 0)
                return 0;
        }
        return def;
    }
    return def;
}

void coRegistry::deleteEntry(const char *className, int ID, const char *name)
{
    if (!className)
    {
        return;
    }
    reset();
    while (current())
    {
        if ((ID == 0) || (current()->getID() == ID))
        {
            if (!strcmp(current()->getName(), className))
            {
                current()->deleteVar(name);
            }
        }
        next();
    }
}

void coRegistry::deleteEntry(int modID)
{
    if (!modID)
    {
        return;
    }
    reset();
    while (current())
    {
        if (current()->getID() == modID)
        {
            current()->deleteAllNonStaticVars();
        }
        next();
    }
}

regClass *coRegistry::getClass(const char *name, int ID)
{
    if (!name)
    {
        return (NULL);
    }
    reset();
    while (current())
    {
        if ((ID == 0) || (current()->getID() == ID))
        {
            if (!strcmp(current()->getName(), name))
                return (current());
        }
        next();
    }
    //cerr << "Class " << name << " not found!\n";
    return (NULL);
}

void coRegistry::unObserve(int recvID)
{
    reset();
    while (current())
    {
        current()->unObserve(recvID);
        next();
    }
}

void coRegistry::observe(const char *className, int ID, int recvID, const char *variableName)
{
    if (!className)
    {
        return;
    }
    int foundOne = 0;
    reset();
    while (current())
    {
        if ((ID == 0) || (current()->getID() == ID))
        {
            if (!strcmp(current()->getName(), className))
            {
                current()->observe(recvID, variableName);
                foundOne = 1;
            }
        }
        next();
    }
    if (!foundOne)
    {
        regClass *rc = getClass(className, ID);
        if (!rc)
        {
            rc = new regClass(className, ID);
            append(rc);
        }
        rc->observe(recvID, variableName);
    }
}

void coRegistry::unObserve(const char *className, int ID, int recvID, const char *variableName)
{
    if (!className)
    {
        return;
    }
    int foundOne = 0;
    reset();
    while (current())
    {
        if ((ID == 0) || (current()->getID() == ID))
        {
            if (!strcmp(current()->getName(), className))
            {
                current()->unObserve(recvID, variableName);
                foundOne = 1;
            }
        }
        next();
    }
    if (!foundOne)
    {
        if (variableName)
            cerr << "Variable " << variableName << " not found in class " << className << " ID: " << ID << endl;
        else
            cerr << "Class " << className << " ID: " << ID << " not found" << endl;
    }
}

void regClass::observe(int recvID, const char *variableName)
{
    if (variableName)
    {
        regVar *rv = getVar(variableName);
        if (!rv)
        {
            rv = new regVar(this, variableName, "coNULL");
            append(rv);
        }
        rv->observe(recvID);
    }
    else
    {
        observers.addObserver(recvID);
        reset();
        regVar *cv;
        while ((cv = current()))
        {
            cv->observe(recvID);
            next();
        }
    }
}

void regClass::unObserve(int recvID, const char *variableName)
{
    if (variableName)
    {
        regVar *rv = getVar(variableName);
        if (rv)
        {
            rv->unObserve(recvID);
        }
    }
    else
    {
        observers.removeObserver(recvID);
        reset();
        regVar *cv;
        while ((cv = current()))
        {
            cv->unObserve(recvID);
            next();
        }
    }
}

regVar *regClass::getVar(const char *n)
{
    if (!n)
    {
        return (NULL);
    }
    reset();
    while (current())
    {
        if (!strcmp(current()->getName(), n))
            return (current());
        next();
    }
    //cerr << "Var " << n << " not found in Class "<< name <<"!\n";
    return (NULL);
}

void regClass::deleteVar(const char *n)
{
    if (!n)
    {
        return;
    }
    reset();
    while (current())
    {
        if (!strcmp(current()->getName(), n))
        {
            remove();
            return;
        }
        next();
    }
    cerr << "Var " << n << " not found in Class " << name << "!\n";
}

void regClass::deleteAllNonStaticVars()
{
    reset();
    while (current())
    {
        if (!(current()->isStatic()))
        {
            remove();
        }
        else
        {
            next();
        }
    }
}

/*
void coRegistry::saveNetwork(coCharBuffer &cb)
{
    cb+="# Global Registry\n#\n\n";
    reset();
   regClass *rc;
    while ((rc=current()))
    {
       if(rc->getID() == 0)
         rc->saveNetwork(cb);
        next();
}
cb+="\n# End Global Registry\n#\n";
}

void regClass::saveNetwork(coCharBuffer &cb)
{
reset();
regVar *cv;
while ((cv=current()))
{
if((classID==0)||(cv->isStatic()))
{
cb+="coSet ";
if(classID!=0)
{
netModule *m=network->getModule(classID);
if(m)
{
cb+=m->scriptName();
cb+=' ';
}
else
cb+="ModuleNotFound ";
}
cb+=name;
cb+="::";
cb+=cv->getName();
cb+=" \"";
cb+=cv->getValue();
cb+='\"';
cb+="\n";
}
next();
}
cb+="\n# End Userinterface Variables\n#\n";
}*/

coRegistry::coRegistry()
{
    instance = this;
    setNoDelete();
    regMode = 0;
}
