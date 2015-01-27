/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "classmanager.h"

#include <stdio.h>

using namespace Tools;

ClassManager::ClassManager()
{
#ifdef DEBUG_MODE
    cout << "ClassManager::ClassManager()" << endl;
#endif

    this->nextObjID = 0;

    classInfos.clear();
    objects.clear();
    singles.clear();
    singleIDs.clear();

    outMan = NULL;

    statMan = NULL;
}

ClassManager::~ClassManager()
{
#ifdef DEBUG_MODE
    cout << "ClassManager::~ClassManager()" << endl;
#endif

    cout << "ClassManager::~ClassManager()" << endl;
    this->clear();
}

bool ClassManager::addClass(string className, ClassInfo *classInfo)
{
#ifdef DEBUG_MODE
    cout << "ClassManager::addClass(): " << className << endl;
#endif

    map<string, ClassInfo *>::iterator infoIterator = classInfos.find(className);

    if (infoIterator != classInfos.end())
        return false;
    else
        classInfos.insert(pair<string, ClassInfo *>(className, classInfo));

    return true;
}

BaseObject *ClassManager::getObject(string className)
{
#ifdef DEBUG_MODE
    cout << "ClassManager::getObject(): " << className << " ";
#endif

    BaseObject *obj = NULL;

    map<string, ClassInfo *>::iterator infoIterator
        = this->classInfos.find(className);

    if (infoIterator != classInfos.end())
    {
        ClassInfo *classInfo = infoIterator->second;

        if (classInfo->getMaxObj() == 0)
        {
            cout << "ClassManager: class " << className << " is blocked. I can't create any objects of it. Bailing out..."
                 << endl;
            exit(-1);
        }

        if (classInfo->getMaxObj() == 1 && classInfo->maxObjReached())
        {
            obj = getSingle(className);

            if (obj != NULL)
            {
#ifdef DEBUG_MODE
                cout << "got single with ID " << obj->getID() << endl;
#endif

                return obj;
            }
        }

        obj = classInfo->New(++this->nextObjID);

        if (obj == NULL)
        {
            cout << "ClassManager: could not create object with ID " << this->nextObjID << ". Bailing out.." << endl;

            exit(-1);
        }

#ifdef DEBUG_MODE
        cout << "created new object with ID " << obj->getID() << endl;
#endif

        if (!obj->init())
        {
            cout << "could not init object with ID " << obj->getID()
                 << " of class " << className << endl;
            cout << "Bailing out.... " << endl;

            exit(-1);
        }

#ifdef DEBUG_MODE
        cout << "inserted new object with ID " << obj->getID() << endl;
#endif

        objects.insert(pair<int, BaseObject *>(obj->getID(), obj));

        if (classInfo->getMaxObj() == 1)
            addSingle(className, obj->getID(), obj);
    }
    else
    {
        cout << "ClassManager: unknown class " << className << ". The given name is either mispelled or unknown. Bailing out. " << endl;
        exit(-1);
    }

    if (obj == NULL)
    {
        cout << "ClassManager: couldn't create an object of class " << className << ". Could be a memory problem. Bailing out. " << endl;
        exit(-1);
    }

    return obj;
}

bool ClassManager::deleteObject(int objectID)
{
#ifdef DEBUG_MODE
    cout << "ClassManager::deleteObject(" << objectID << ")" << endl;
#endif

    map<int, BaseObject *>::iterator objectIterator = objects.find(objectID);

    if (objectIterator != objects.end())
    {
        BaseObject *bObj = objectIterator->second;

        if (bObj != NULL)
        {
            delete bObj;

            bObj = NULL;
        }

        objects.erase(objectID);
        removeSingle(objectID);

        return true;
    }

    return false;
}

OutputManagerBase *ClassManager::getOutputMan()
{
    if (outMan == NULL)
        outMan = (OutputManagerBase *)getObject("Tools::OutputManager");

    return outMan;
}

void ClassManager::selfTest()
{
    cout << "ClassManager: initiating self test... " << endl;
    cout << endl;

    map<int, BaseObject *>::iterator objectIterator = objects.begin();

    while (objectIterator != objects.end())
    {
        BaseObject *bObj = objectIterator->second;

        if (bObj != NULL)
        {
#ifdef DEBUG_MODE
            cout << "Classmanager: deleting object #" << bObj->getID() << endl;
#endif

            this->objects.erase(objectIterator->first);
            removeSingle(bObj->getID());
            delete bObj;
            bObj = NULL;
        }

        ++objectIterator;
    }

    if (this->objects.size() == 0)
        objects.clear();

    objects.clear();
    singles.clear();
    singleIDs.clear();
}

bool ClassManager::addSingle(string className,
                             int objectID,
                             Tools::BaseObject *object)
{
    map<string, BaseObject *>::iterator singleIterator = singles.find(className);
    map<int, string>::iterator idIterator = singleIDs.find(objectID);

    if (singleIterator == singles.end())
    {
#ifdef DEBUG_MODE
        cout << "inserting new single with ID " << object->getID() << endl;
#endif

        singles.insert(pair<string, Tools::BaseObject *>(className, object));

        if (idIterator == singleIDs.end())
            singleIDs.insert(pair<int, string>(objectID, className));
    }

    return true;
}

bool ClassManager::removeSingle(int objectID)
{
#ifdef DEBUG_MODE
    cout << "ClassManager::removeSingle(" << objectID << ")" << endl;
#endif

    map<int, string>::iterator idIterator = singleIDs.find(objectID);

    if (idIterator != singleIDs.end())
    {
        map<string, BaseObject *>::iterator singleIterator
            = singles.find(idIterator->second);

        if (singleIterator != singles.end())
            singles.erase(idIterator->second);

        singleIDs.erase(objectID);
    }

    return true;
}

BaseObject *ClassManager::getSingle(string className)
{
    BaseObject *object = NULL;
    map<string, BaseObject *>::iterator singleIterator
        = singles.find(className);

    if (singleIterator != singles.end())
        object = singleIterator->second;

    return object;
}

StatisticManagerBase *ClassManager::getStatMan()
{
    if (statMan == NULL)
    {
        map<string, ClassInfo *>::iterator infoIterator = classInfos.find("Tools::StatisticManager");

        if (infoIterator != classInfos.end())
        {
            ClassInfo *classInfo = infoIterator->second;

            statMan = (StatisticManagerBase *)classInfo->New(++this->nextObjID);

#ifdef DEBUG_MODE
            cout << "ClassManager::getStatMan(): created statMan with ID "
                 << statMan->getID() << endl;
#endif
        }
    }

    if (statMan == NULL)
    {
        cout << "could not get statMan. Bailing out." << endl;
        exit(-1);
    }

    return statMan;
}

void ClassManager::clear()
{
    cout << "ClassManager::clear()" << endl;

    BaseObject *bObj = NULL;

    if (!classInfos.empty())
    {
        classInfos.clear();
    }

    if (!objects.empty())
    {
        map<int, BaseObject *>::iterator objectIterator = objects.begin();

        while (objectIterator != objects.end())
        {
            bObj = objectIterator->second;

            if (bObj != NULL)
            {
#ifdef DEBUG_MODE2
                cout << "class manager: remove object of class "
                     << bObj->getClassName() << " with ID " << bObj->getID()
                     << endl;
#endif

                delete bObj;
                bObj = NULL;
            }

            printf("deleting\n");
            ++objectIterator;
        }
    }

    classInfos.clear();
    classInfos.swap(classInfos);
    objects.clear();
    objects.swap(objects);

    singles.clear();
    singles.swap(singles);
    singleIDs.clear();
    singleIDs.swap(singleIDs);

    printf("printing\n");

    statMan->print();

    cout << "printed" << endl;

    outMan = NULL;

    if (statMan != NULL)
    {
#ifdef DEBUG_MODE
        cout << "class manager: remove statMan with ID " << statMan->getID()
             << endl;
#endif

        delete statMan;
        statMan = NULL;
    }
}
