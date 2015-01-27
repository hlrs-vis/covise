/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ZD.h"
#include <cstdio>

using namespace DTF;

CLASSINFO_OBJ(ClassInfo_DTFZD, ZD, "DTF::ZD", INT_MAX);

ZD::ZD()
    : Tools::BaseObject()
{
    INC_OBJ_COUNT(getClassName());
}

ZD::ZD(string className, int objectID)
    : Tools::BaseObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

ZD::~ZD()
{
    DEC_OBJ_COUNT(getClassName());
}

bool ZD::read(int simNum)
{
    clear();
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    DTF_Lib::LibIF *libIF = (DTF_Lib::LibIF *)cm->getObject("DTF_Lib::LibIF");
    DTF_Lib::Zone *zone = (DTF_Lib::Zone *)(*libIF)("Zone");
    DTF_Lib::VirtualZone *vZone = (DTF_Lib::VirtualZone *)(*zone)("VirtualZone");

    DTF_Lib::DataElement *zdInfo = (DTF_Lib::DataElement *)cm->getObject("DTF_Lib::DataElement");

    string type = "";
    bool retVal = false;
    vector<string> names;
    string name = "";
    vector<double> data;

    if (vZone->queryVZDNames(simNum, names))
    {
        vector<string>::iterator namesIterator = names.begin();

        while (namesIterator != names.end())
        {
            name = *namesIterator;

            if (vZone->queryVZDbyName(simNum, name, *zdInfo))
            {
                DTF_Lib::DataTypes *dt = Tools::Singleton<DTF_Lib::DataTypes>::getInstance();

                dt->toString(zdInfo->getDataType(), type);

                if ((type == "dtf_double") || (type == "dtf_single"))
                    if (vZone->readVZDbyName(simNum, name, data))
                        dblData.insert(pair<string, vector<double> >(name, data));
            }

            data.clear();
            zdInfo->clear();

            ++namesIterator;
        }

        retVal = true;
    }

    if (zdInfo != NULL)
    {
        zdInfo->clear();
        cm->deleteObject(zdInfo->getID());
    }

    return retVal;
}

map<string, vector<double> > ZD::getData()
{
    return this->dblData;
}

void ZD::clear()
{
    dblData.clear();
    dataName = "";
}

void ZD::print()
{
    string typeStr = "";
    DTF_Lib::DataTypes *dt = Tools::Singleton<DTF_Lib::DataTypes>::getInstance();

    dt->toString(dataType, typeStr);

    cout << "zone data array: " << endl;
    cout << "-----------------" << endl;

    cout << "name: " << this->dataName << endl;
    cout << "type: " << typeStr << endl;

    int length = 0;

    length = dblData.size();

    cout << "length: " << length << endl;

    if (length != 0)
    {
        map<string, vector<double> >::iterator i = dblData.begin();

        cout << "data: " << endl;
        while (i != dblData.end())
        {
            vector<double> data = i->second;

            cout << "name: " << i->first << endl;

            for (unsigned int j = 0; j < data.size(); j++)
                cout << data[j] << endl;

            ++i;
        }
    }
}
