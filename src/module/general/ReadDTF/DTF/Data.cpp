/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DTF/Data.cpp
 * @brief contains implementation of class DTF::Data
 * @author Alexander Martinez <kubus3561@gmx.de>
 * created
 */
#include "Data.h"

using namespace DTF;

CLASSINFO_OBJ(ClassInfo_DTFData, Data, "DTF::Data", INT_MAX);

Data::Data()
    : Tools::BaseObject()
{
    INC_OBJ_COUNT(getClassName());
}

Data::Data(string className, int objectID)
    : Tools::BaseObject(className, objectID)
{
    file = NULL;

    INC_OBJ_COUNT(getClassName());
}

Data::~Data()
{
    sims.clear();
    DEC_OBJ_COUNT(getClassName());
}

bool Data::init()
{
    sims.clear();

    return true;
}

bool Data::read(string fileName)
{
    clear();

    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    DTF_Lib::LibIF *libif = (DTF_Lib::LibIF *)cm->getObject("DTF_Lib::LibIF");
    DTF_Lib::File *libFile = (DTF_Lib::File *)(*libif)("File");
    DTF::File *file = (DTF::File *)cm->getObject("DTF::File");

    int numSims = 0;

    if (libif->setFileName(fileName))
    {
        if (libFile->queryNumSims(numSims))
            for (int i = 1; i <= numSims; i++)
            {
                Sim *sim = (DTF::Sim *)cm->getObject("DTF::Sim");

                if (!sim->read(fileName, i))
                {
                    cm->deleteObject(sim->getID());
                    break;
                }

                if (sims.find(i) == sims.end())
                    sims.insert(pair<int, Sim *>(i, sim));

                sim = NULL;
            }

        File *fileObj = (File *)cm->getObject("DTF::File");

        if (file->read(fileName))
            this->file = fileObj;

        libFile->close();
        return true;
    }

    return false;
}

File *Data::getFile()
{
    return this->file;
}

Sim *Data::getSim(int simNum)
{
    Sim *sim = NULL;
    map<int, Sim *>::iterator simIterator = sims.find(simNum);

    if (simIterator != sims.end())
        sim = simIterator->second;

    return sim;
}

int Data::getNumSims()
{
    return this->sims.size();
}

void Data::clear()
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    map<int, Sim *>::iterator simIterator = sims.begin();

    while (simIterator != sims.end())
    {
        Sim *sim = simIterator->second;

        if (sim != NULL)
        {
            sim->clear();
            cm->deleteObject(sim->getID());
            sim = NULL;
        }

        ++simIterator;
    }

    sims.clear();

    if (file != NULL)
    {
        cm->deleteObject(file->getID());
        file = NULL;
    }
}

void Data::print()
{
    cout << "---------------------------------------------------------" << endl;
    cout << "printing DTF data: " << endl;
    cout << "---------------------------------------------------------" << endl;

    if (file != NULL)
        file->print();

    cout << "---------------------------------------------------------" << endl;
    cout << "simulations: " << sims.size() << endl;

    cout << "---------------------------------------------------------" << endl;

    map<int, Sim *>::iterator simIterator = this->sims.begin();

    while (simIterator != this->sims.end())
    {
        Sim *sim = simIterator->second;

        cout << "- simulation " << simIterator->first << ": ";

        if (sim != NULL)
            sim->print();

        ++simIterator;
    }
}
