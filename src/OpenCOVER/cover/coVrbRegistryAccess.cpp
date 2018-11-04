/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C)1998 RUS                                **
 **                Computing Center University of Stuttgart                **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Author: D. Rantzau                                                     **
 ** Date:  15.07.98  V1.0                                                  **
\**************************************************************************/
#include "coVRPluginSupport.h"
#include "coVrbRegistryAccess.h"
#include "coVRMSController.h"
#include <vrbclient/VRBClient.h>
#include <vrbclient/VRBMessage.h>
#include <net/message.h>
#include <net/message_types.h>
#include "coVRCommunication.h"
#include "OpenCOVER.h"

#include <sys/stat.h>

using namespace covise;
using namespace opencover;
coVrbRegistryAccess *coVrbRegistryAccess::instance = NULL;

//==========================================================================
//
// class coVrbRegistryAccess
//
//==========================================================================
coVrbRegistryAccess::coVrbRegistryAccess(int ID)
{
    assert(!instance);
    _ID = ID;
    _entryList = new coDLPtrList<coVrbRegEntry *>();
    instance = this;
}

coVrbRegistryAccess::~coVrbRegistryAccess()
{

    // delete entry list including all entries!
    delete _entryList;
    instance = NULL;
}

void coVrbRegistryAccess::setID(int id)
{
    int oldID = _ID;
    _ID = id;
    // update ID of all own registry entries
    _entryList->reset();
    while (_entryList->current())
    {
        if (_entryList->current()->getID() == oldID)
        {
            _entryList->current()->setID(id);
            _entryList->current()->changedByMe(true);
            _entryList->current()->updateVRB();
        }
        _entryList->next();
    }
}

void coVrbRegistryAccess::addEntry(coVrbRegEntry *e)
{

    _entryList->append(e);
}

void coVrbRegistryAccess::removeEntry(coVrbRegEntry *e)
{

    _entryList->reset();
    if (_entryList->find(e))
        _entryList->remove();
}

void coVrbRegistryAccess::sendMsg(TokenBuffer &tb, int type)
{
    if (vrbc)
    {
        Message m(tb);
        m.type = type;
        vrbc->sendMessage(&m);
    }
}

void coVrbRegistryAccess::updateVRB()
{
    // attach to the list
    _entryList->reset();
    while (_entryList->current())
    {
        _entryList->current()->updateVRB();
        _entryList->next();
    }
}

void coVrbRegistryAccess::subscribeClass(const char *cl, int ID, coVrbRegEntryObserver *ob)
{
    if (cl == NULL)
        return;

    // attach to the list
    _entryList->reset();
    while (_entryList->current())
    {
        if ((strcmp(_entryList->current()->getClass(), cl) == 0L) && (_entryList->current()->isClassOnlyEntry()) && (_entryList->current()->getID() == ID))

            // entry already existing
            return;
        _entryList->next();
    }

    coVrbRegEntry *entry = new coVrbRegEntry(cl, ID, NULL);
    entry->attach(ob);
    _entryList->append(entry);

    TokenBuffer tb;
    // compose message
    tb << cl;
    tb << ID;
    tb << _ID;
    // inform controller about creation
    if (_ID >= 0)
        sendMsg(tb, COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_CLASS);
}

coVrbRegEntry *coVrbRegistryAccess::subscribeVar(const char *cl, int ID, const char *var, coVrbRegEntryObserver *ob)
{
    if ((cl == NULL) || (var == NULL))
        return NULL;

    // attach to the list
    _entryList->reset();
    while (_entryList->current())
    {
        if ((strcmp(_entryList->current()->getClass(), cl) == 0L) && (strcmp(_entryList->current()->getVar(), cl) == 0L && ((_entryList->current()->getID() == ID))))
            // entry already existing
            return _entryList->current();
        _entryList->next();
    }

    coVrbRegEntry *entry = new coVrbRegEntry(cl, ID, var);
    entry->attach(ob);
    _entryList->append(entry);

    TokenBuffer tb;
    // compose message
    tb << cl;
    tb << ID;
    tb << var;
    tb << _ID;
    // inform controller about creation
    if (_ID >= 0)
        sendMsg(tb, COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_VARIABLE);
    return entry;
}

void coVrbRegistryAccess::unsubscribeClass(const char *cl, int ID)
{
    if (cl == NULL)
        return;

    // attach to the list
    _entryList->reset();
    while (_entryList->current())
    {
        if ((strcmp(_entryList->current()->getClass(), cl) == 0L) && (_entryList->current()->isClassOnlyEntry()) && ((_entryList->current()->getID() == ID)))
        {
            // found
            TokenBuffer tb;
            // compose message
            tb << cl;
            tb << ID;
            tb << _ID;
            if (_ID >= 0)
                sendMsg(tb, COVISE_MESSAGE_VRB_REGISTRY_UNSUBSCRIBE_CLASS);
            // remove the found current entry
            _entryList->remove();
            return;
        }
        _entryList->next();
    }
}

void coVrbRegistryAccess::unsubscribeVar(const char *cl, int ID, const char *var)
{

    if ((cl == NULL) || (var == NULL))
        return;

    // attach to the list
    _entryList->reset();
    while (_entryList->current())
    {
        if ((strcmp(_entryList->current()->getClass(), cl) == 0L) && (strcmp(_entryList->current()->getVar(), cl) == 0L) && ((_entryList->current()->getID() == ID)))
        {
            // found
            TokenBuffer tb;
            // compose message
            tb << cl;
            tb << ID;
            tb << var;
            tb << _ID;
            if (_ID >= 0)
                sendMsg(tb, COVISE_MESSAGE_VRB_REGISTRY_UNSUBSCRIBE_VARIABLE);
            // remove the found current entry
            _entryList->remove();
            return;
        }
        _entryList->next();
    }
}

void coVrbRegistryAccess::createVar(const char *cl, const char *var, int flag)
{
    if ((cl == NULL) || (var == NULL))
        return;

    // compose message
    TokenBuffer tb;
    tb << cl;
    tb << _ID; // this module ID
    tb << var;
    tb << flag;
    if (_ID >= 0)
        sendMsg(tb, COVISE_MESSAGE_VRB_REGISTRY_CREATE_ENTRY);
}

void coVrbRegistryAccess::setVar(const char *cl, const char *var, const char *val)
{
    if ((cl == NULL) || (var == NULL))
        return;

    // compose message
    TokenBuffer tb;
    tb << cl;
    tb << _ID; // this module ID
    tb << var;
    if (val == NULL)
        tb << "";
    else
        tb << val;
    if (_ID >= 0)
        sendMsg(tb, COVISE_MESSAGE_VRB_REGISTRY_SET_VALUE);

    // attach to the list
    _entryList->reset();
    while (_entryList->current())
    {
        if ((strcmp(_entryList->current()->getClass(), cl) == 0L) && (strcmp(_entryList->current()->getVar(), cl) == 0L))
        {
            _entryList->current()->changedByMe();
            // entry already existing
            return;
        }
        _entryList->next();
    }

    coVrbRegEntry *entry = new coVrbRegEntry(cl, _ID, var);
    entry->setVal(val);
    entry->changedByMe();
    _entryList->append(entry);
}

void coVrbRegistryAccess::destroyVar(const char *cl, const char *var)
{
    if ((cl == NULL) || (var == NULL))
        return;

    // compose message
    TokenBuffer tb;
    tb << cl;
    tb << _ID; // this module ID
    tb << var;
    if (_ID >= 0)
        sendMsg(tb, COVISE_MESSAGE_VRB_REGISTRY_DELETE_ENTRY);
}

// called by controller if registry entry has changed
void coVrbRegistryAccess::update(TokenBuffer &tb, int reason)
{
    char *cl;
    int clID;
    char *name;
    char *val;

    switch (reason)
    {

    case COVISE_MESSAGE_VRB_REGISTRY_ENTRY_CHANGED:

        tb >> cl;
        tb >> clID;
        tb >> name;
        tb >> val;

        // call all class specific observers
        _entryList->reset();

        while (_entryList->current())
        {
            if (!strcmp(_entryList->current()->getClass(), cl))
            {
                _entryList->current()->setVar((const char *)name);
                _entryList->current()->setVal((const char *)val);
                _entryList->current()->setID(clID);
                // call observers
                _entryList->current()->setChanged();
                _entryList->current()->changedByMe(false);
            }
            _entryList->next();
        }

        break;

    case COVISE_MESSAGE_VRB_REGISTRY_ENTRY_DELETED:

        tb >> cl;
        tb >> clID;
        tb >> name;

        // inform observers
        _entryList->reset();
        while (_entryList->current())
        {
            if (!strcmp(_entryList->current()->getClass(), cl))
            {
                if (strcmp(_entryList->current()->getVar(), name) == 0L)
                {

                    if (clID == 0 || (_entryList->current()->getID() == clID))
                    {
                        // inform observer
                        _entryList->current()->setDeleted();
                        // delete entries
                        _entryList->remove();
                    }
                }
            }
            _entryList->next();
        }

        break;

    } // switch
}

void coVrbRegEntry::setValue(const char *val)
{
    if ((_cl.c_str() == NULL) || (_var.c_str() == NULL))
        return;

    _val = val;
    // compose message
    TokenBuffer tb;
    tb << _cl;
    tb << _ID; // this module ID
    tb << _var;
    if (_val.c_str() == NULL)
        tb << "";
    else
        tb << _val;
    if (_ID >= 0)
        coVrbRegistryAccess::instance->sendMsg(tb, COVISE_MESSAGE_VRB_REGISTRY_SET_VALUE);
    changedByMe();
}

void coVrbRegEntry::updateVRB()
{
    if (_cl.c_str() == NULL)
        return;

    if (_observer)
    {
        if (_var.c_str() == NULL)
        {
            TokenBuffer tb;
            // compose message
            tb << _cl;
            tb << _ID;
            tb << coVrbRegistryAccess::instance->getID();
            // inform controller about creation
            if (_ID >= 0)
                coVrbRegistryAccess::instance->sendMsg(tb, COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_CLASS);
        }
        else
        {
            TokenBuffer tb;
            // compose message
            tb << _cl;
            tb << _ID;
            tb << _var;
            tb << coVrbRegistryAccess::instance->getID();
            // inform controller about creation
            if (_ID >= 0)
                coVrbRegistryAccess::instance->sendMsg(tb, COVISE_MESSAGE_VRB_REGISTRY_SUBSCRIBE_VARIABLE);
        }
    }

    if (_var.c_str() == NULL)
        return;

    if (_changedByMe)
    {
        // compose message
        TokenBuffer tb;
        tb << _cl;
        tb << _ID; // this module ID
        tb << _var;
        if (_val.c_str() == NULL)
            tb << "";
        else
            tb << _val;
        if (_ID >= 0)
            coVrbRegistryAccess::instance->sendMsg(tb, COVISE_MESSAGE_VRB_REGISTRY_SET_VALUE);
    }
}

//==========================================================================
//
// class coVrbRegEntry
//
//==========================================================================
coVrbRegEntry::coVrbRegEntry(const char *cl, int ID, const char *var)
{
    _val = "";
    _ID = ID;
    _isDeleted = false;
    _changedByMe = false;
    _observer = NULL;

    if (cl)
    {

        // create a registry entry
        _cl = cl;
    }
    else
        return;

    if (var)
    {
        _isClassOnly = false;
        _var = var;
    }
    else
    {
        _isClassOnly = true;
        _var = "";
    }
}

void coVrbRegEntry::attach(coVrbRegEntryObserver *ob)
{
    _observer = ob;
}

void coVrbRegEntry::detach(coVrbRegEntryObserver *)
{

    _observer = NULL;
}

void coVrbRegEntry::notify(int interestType)
{

    if (_observer)
    {
        if (_observer->getObserverType() == interestType)
        {
            _observer->update(this);
        }
    }
}

coVrbRegEntry::~coVrbRegEntry()
{
}
