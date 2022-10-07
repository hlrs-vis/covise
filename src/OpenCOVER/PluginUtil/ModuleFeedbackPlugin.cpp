/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ModuleFeedbackPlugin.h"
#include "ModuleFeedbackManager.h"

#include <cover/RenderObject.h>
#include <cover/coInteractor.h>
#include <list>
#include <algorithm>

using namespace opencover;

std::list<ModuleFeedbackManager *> ModuleFeedbackPlugin::_ComplexModuleList;

// ----------------------------------------------------------------
// construction / destruction
// ----------------------------------------------------------------
ModuleFeedbackPlugin::ModuleFeedbackPlugin(const char *pluginName)
: coVRPlugin(pluginName)
{
}

ModuleFeedbackPlugin::~ModuleFeedbackPlugin()
{
    myInteractions_.clear();
    for (auto &m: _ComplexModuleList)
    {
        // delete the tracerInteraction
        delete m;
    }
    _ComplexModuleList.clear();
}

void ModuleFeedbackPlugin::newInteractor(const RenderObject *container, coInteractor *i)
{
    if (strcmp(i->getPluginName(), "ColorBars"))
        return;

    if (!container)
        return;

    if (!container->getName())
        return;

    std::string objname(container->getName());

    ModuleFeedbackManager *mgr = NULL;
    for (auto &m: _ComplexModuleList)
    {
        if (m->compare(objname.c_str()))
        {
            mgr = m;
            break;
        }
    }

    if (mgr)
    {
        mgr->addColorbarInteractor(i);
    }
}

// ----------------------------------------------------------------
// manipulation (add / remove)
// ----------------------------------------------------------------

// assume the module name in inter is the same as in _inter
// in ComplexModuleInteraction::compare(coInteractor *)
// we only check the module instance
void
ModuleFeedbackPlugin::add(const RenderObject *container, coInteractor *inter)
{
    //fprintf(stderr,"ModuleFeedbackPlugin::add\n");
    // do we have this object in our list
    bool found = false;
    for (auto *m: _ComplexModuleList)
    {
        if (m->compare(inter))
        {
            // we already have this object, we replace the object name
            m->update(container, inter);
            found = true;
            break;
        }
    }
    if (!found)
    {
        //fprintf(stderr,"not found\n");

        // we don't have this object, append it to list and add to submenu
        ModuleFeedbackManager *newManager = NewModuleFeedbackManager(container, inter, NULL, getName());

        if (newManager)
        {
            // fprintf(stderr,"newManager\n");
            _ComplexModuleList.push_back(newManager);
            newManager->update(container, inter);
            myInteractions_.push_back(newManager);
        }
    }
}

void
ModuleFeedbackPlugin::add(const RenderObject *container, const RenderObject *geomobj)
{
    if (geomobj->isSet() || geomobj->isGeometry())
        return;

    // do we have this object in our list
    const char *name = container ? container->getName() : geomobj->getName();

    // go through all interactions and check if it is already there
    bool found = false;
    for (auto *m: _ComplexModuleList)
    {
        if (m->compare(name))
        {
            // we already have this object, but does it really belong to me?
            for (auto *i: myInteractions_)
            {
                if (i->compare(name))
                {
                    i->update(container, geomobj);
                }
            }
            found = true;
            break;
        }
    }
    if (!found)
    {
        // we don't have this object, append it to list and add to submenu
        ModuleFeedbackManager *newManager = NewModuleFeedbackManager(container, NULL, geomobj, getName());
        _ComplexModuleList.push_back(newManager);
        newManager->update(container, geomobj);
        myInteractions_.push_back(newManager);
    }
}

void
ModuleFeedbackPlugin::remove(const char *objName)
{
    if (!objName)
        return;

    bool found = false;
    auto it = std::find_if(myInteractions_.begin(), myInteractions_.end(),
                           [objName](const ModuleFeedbackManager *i) { return i->compare(objName); });
    if (it != myInteractions_.end())
    {
        myInteractions_.erase(it);
        found = true;
    }

    if (found)
    {
        auto it = std::find_if(_ComplexModuleList.begin(), _ComplexModuleList.end(),
                               [objName](const ModuleFeedbackManager *m) { return m->compare(objName); });
        if (it != _ComplexModuleList.end())
        {
            delete *it;
            _ComplexModuleList.erase(it);
        }
    }
}

void
ModuleFeedbackPlugin::preFrame()
{
    // for each plugin preFrame is called
    // a plugin handles objects with feedback from several modules, therefore we
    // have to call preFrame for every ModuleFeedbackManager which belongs to this plugin
    for (auto *m: _ComplexModuleList)
    {
        if (m->comparePlugin(getName()))
            m->preFrame();
    }
}

void
ModuleFeedbackPlugin::handleSetCaseMsg(const char *objectName, const char *caseName)
{
    //fprintf(stderr,"ModuleFeedbackPlugin::handleSetCaseMsg(%s, %s)\n", objectName, caseName);

    for (auto *i: myInteractions_)
    {
        if (i->compare(objectName))
        {
            //fprintf(stderr,"\tfound object %s and set case %s\n", objectName, caseName);
            i->setCaseFromGui(caseName);
        }
    }
}

void
ModuleFeedbackPlugin::handleSetNameMsg(const char *coviseObjectName, const char *newName)
{
    //fprintf(stderr,"ModuleFeedbackPlugin::handleSetNameMsg(%s, %s)\n", coviseObjectName, newName);

    for (auto *i: myInteractions_)
    {
        if (i->compare(coviseObjectName))
        {
            //fprintf(stderr,"\tfound object %s and set case %s\n", objectName, caseName);
            i->setNameFromGui(newName);
        }
    }
}

void
ModuleFeedbackPlugin::handleGeoVisibleMsg(const char *objectName, bool visibility)
{
    //fprintf(stderr,"\nModuleFeedbackPlugin::handleGeoVisibleMsg(%s, %d)\n", objectName, visibility);

    for (auto *i: myInteractions_)
    {
        if (i->compare(objectName))
        {
            //fprintf(stderr,"\tfound object %s and set visibility\n", objectName);
            i->setHideFromGui(!visibility);
            break;
        }
    }
}

void
ModuleFeedbackPlugin::setMatrix(const char *objectName, float *row0, float *row1, float *row2, float *row3)
{
    for (auto *i: myInteractions_)
    {
        if (i->compare(objectName))
        {
            i->setMatrix(row0, row1, row2, row3);
            break;
        }
    }
}

void
ModuleFeedbackPlugin::addNodeToCase(const char *objectName, osg::Node *node)
{
    //fprintf(stderr,"\nModuleFeedbackPlugin::addNodeToCase(%s)\n", objectName);

    for (auto *i: myInteractions_)
    {
        if (i->compare(objectName))
        {
            i->addNodeToCase(node);
            break;
        }
    }
}

void ModuleFeedbackPlugin::getSyncInteractors(coInteractor *inter_) // collect all interactors from other modules which should be syncronized with this one
// the following methods will set parameters on all those intersctors
{
    interactors.clear();
    const char *syncGroup = NULL;
    int nu = inter_->getNumUser();
    if (nu > 0)
    {
        for (int i = 0; i < nu; i++)
        {
            const char *ud = inter_->getString(i);
            if (strncmp(ud, "SYNCGROUP=", 10) == 0)
            {
                syncGroup = ud + 10;
            }
        }
    }
    if (syncGroup != NULL)
    {
        for (auto *i: myInteractions_)
        {
            if (i->getInteractor() == inter_)
            {
                if (i->getSyncState() == false) // don't sync
                {
                    interactors.clear();
                    interactors.push_back(inter_);
                    break;
                }
            }
            const char *syncGroup_ = NULL;
            int nu = i->getInteractor()->getNumUser();
            if ( nu > 0)
            {
                for (int k = 0; k < nu; k++)
                {
                    const char *ud = i->getInteractor()->getString(k);
                    if (strncmp(ud, "SYNCGROUP=", 10) == 0)
                    {
                        syncGroup_ = ud + 10;
                    }
                }
            }
            if (syncGroup_ != NULL && strcmp(syncGroup_, syncGroup) == 0)
            {
                interactors.push_back(i->getInteractor());
            }
        }
    }
    else
    {
        interactors.push_back(inter_);
    }
}
/// execute the Modules
void ModuleFeedbackPlugin::executeModule()
{
    for (std::list<coInteractor *>::iterator it = interactors.begin(); it != interactors.end(); it++)
    {
        (*it)->executeModule();
    }
}

/// set Boolean Parameter
void ModuleFeedbackPlugin::setBooleanParam(const char *name, int val)
{
    for (std::list<coInteractor *>::iterator it = interactors.begin(); it != interactors.end(); it++)
    {
        (*it)->setBooleanParam(name, val);
    }
}

/// set float scalar parameter
void ModuleFeedbackPlugin::setScalarParam(const char *name, float val)
{
    for (std::list<coInteractor *>::iterator it = interactors.begin(); it != interactors.end(); it++)
    {
        (*it)->setScalarParam(name, val);
    }
}

/// set int scalar parameter
void ModuleFeedbackPlugin::setScalarParam(const char *name, int val)
{
    for (std::list<coInteractor *>::iterator it = interactors.begin(); it != interactors.end(); it++)
    {
        (*it)->setScalarParam(name, val);
    }
}

/// set float slider parameter
void ModuleFeedbackPlugin::setSliderParam(const char *name, float min, float max, float value)
{
    for (std::list<coInteractor *>::iterator it = interactors.begin(); it != interactors.end(); it++)
    {
        (*it)->setSliderParam(name, min, max, value);
    }
}

/// set int slider parameter
void ModuleFeedbackPlugin::setSliderParam(const char *name, int min, int max, int value)
{
    for (std::list<coInteractor *>::iterator it = interactors.begin(); it != interactors.end(); it++)
    {
        (*it)->setSliderParam(name, min, max, value);
    }
}

/// set float Vector Param
void ModuleFeedbackPlugin::setVectorParam(const char *name, int numElem, float *field)
{
    for (std::list<coInteractor *>::iterator it = interactors.begin(); it != interactors.end(); it++)
    {
        (*it)->setVectorParam(name, numElem, field);
    }
}
void ModuleFeedbackPlugin::setVectorParam(const char *name, float u, float v, float w)
{
    for (std::list<coInteractor *>::iterator it = interactors.begin(); it != interactors.end(); it++)
    {
        (*it)->setVectorParam(name, u, v, w);
    }
}

/// set int vector parameter
void ModuleFeedbackPlugin::setVectorParam(const char *name, int numElem, int *field)
{
    for (std::list<coInteractor *>::iterator it = interactors.begin(); it != interactors.end(); it++)
    {
        (*it)->setVectorParam(name, numElem, field);
    }
}
void ModuleFeedbackPlugin::setVectorParam(const char *name, int u, int v, int w)
{
    for (std::list<coInteractor *>::iterator it = interactors.begin(); it != interactors.end(); it++)
    {
        (*it)->setVectorParam(name, u, v, w);
    }
}

/// set string parameter
void ModuleFeedbackPlugin::setStringParam(const char *name, const char *val)
{
    for (std::list<coInteractor *>::iterator it = interactors.begin(); it != interactors.end(); it++)
    {
        (*it)->setStringParam(name, val);
    }
}

/// set choice parameter, pos starts with 0
void ModuleFeedbackPlugin::setChoiceParam(const char *name, int pos)
{
    for (std::list<coInteractor *>::iterator it = interactors.begin(); it != interactors.end(); it++)
    {
        (*it)->setChoiceParam(name, pos);
    }
}

void ModuleFeedbackPlugin::setChoiceParam(const char *name, int num, const char *const *list, int pos)
{
    for (std::list<coInteractor *>::iterator it = interactors.begin(); it != interactors.end(); it++)
    {
        (*it)->setChoiceParam(name, num, list, pos);
    }
}
