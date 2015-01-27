/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ReadRECOMDx.h"
#include "DxFile.h"
#include <util/coviseCompat.h>
#include <api/coFeedback.h>
#include <do/coDoData.h>

ReadRECOMDx::ReadRECOMDx(int argc, char *argv[])
    : coModule(argc, argv, "Read module for IBM data explorer")
    , action(NULL)
{
    p_UnstrGrid = addOutputPort("grid",
                                "StructuredGrid|UnstructuredGrid|Polygons",
                                "Grid Object");

    int index;
    for (index = 0; index < maxDataPorts; index++)
    {
        char name[64];
        snprintf(name, 64, "Scalar%c", (char)('A' + index));
        p_ScalarData[index] = addOutputPort(name, "Float",
                                            "Scalar data on object");
    }
    p_filePath = addFileBrowserParam("filename", "Browser");
    p_filePath->setValue(".", "*.dx;*.DX/*");

    p_defaultIsLittleEndian = addBooleanParam("DefaultIsLittleEndian",
                                              "Default is little endian");
    p_defaultIsLittleEndian->setValue(1);

    const char *value = "---";
    for (index = 0; index < maxDataPorts; index++)
    {
        char name[64];
        snprintf(name, 64, "Scalar_Port_%d", index);
        p_ScalarChoice[index] = addChoiceParam(name, name);
        p_ScalarChoice[index]->setValue(1, &value, 0);
    }
}

ReadRECOMDx::~ReadRECOMDx()
{
}

void ReadRECOMDx::traverse(std::vector<std::vector<DxObject *> > &ports,
                           bool generateOutput = false)
{
    if (!action)
        return;

    coDistributedObject **usgSet = NULL;

    std::vector<std::vector<DxObject *> > newPorts;

    bool output = true;
    int port = 0;
    std::vector<std::vector<DxObject *> >::iterator pi;

    int objectIndex = 0;
    int numObjects = -1;

    for (pi = ports.begin(); pi != ports.end(); pi++)
    {
        float setMin = FLT_MAX, setMax = -FLT_MAX;
        std::string label;

        objectIndex = 0;

        if (generateOutput && !usgSet)
        {
            usgSet = new coDistributedObject *[pi->size()];
            if (numObjects == -1)
                numObjects = pi->size();
            else if (numObjects != pi->size())
                Covise::sendError("found %d objects, should be %d", pi->size(),
                                  numObjects);

            for (int index = 0; index < pi->size(); index++)
                usgSet[index] = NULL;
        }

        coDistributedObject **dataSet = NULL;
        if (generateOutput)
            dataSet = new coDistributedObject *[pi->size()];

        std::vector<DxObject *> newObjects;
        std::vector<DxObject *>::iterator i;
        for (i = pi->begin(); i != pi->end(); i++)
        {
            DxObject *object = *i;
            if (object)
            {
                if (label == "")
                {
                    const char *data = object->getData();
                    if (data)
                    {
                        std::map<std::string, std::string>::iterator gi = objects.find(data);
                        if (gi != objects.end())
                            label = gi->second;
                    }
                }

                switch (object->getObjectClass())
                {
                case Parser::MULTIGRID:
                case Parser::GROUP:
                {
                    MemberList mList = object->getMemberList();
                    for (int m = 0; m < mList.size(); m++)
                    {
                        newObjects.push_back(mList[m]->getObject());
                        output = false;
                    }
                    break;
                }

                case Parser::FIELD:
                {
                    if (!generateOutput)
                        newObjects.push_back(object);
                    else
                    {
                        if (!usgSet[objectIndex])
                        {
                            // TODO: test if grid ids really match
                            DxObject *dxPos = NULL;
                            DxObject *dxCon = NULL;
                            DxObjectMap::iterator d = action->arrays_.find(object->getPositions());
                            if (d != action->arrays_.end())
                                dxPos = d->second;
                            d = action->arrays_.find(object->getConnections());
                            if (d != action->arrays_.end())
                                dxCon = d->second;
                            usgSet[objectIndex] = generateUSG(dxPos, dxCon, p_UnstrGrid);
                        }
                        DxObject *dxData = NULL;
                        DxObjectMap::iterator d = action->arrays_.find(object->getData());
                        if (d != action->arrays_.end())
                            dxData = d->second;

                        float min, max;
                        dataSet[objectIndex] = generateData(dxData, p_ScalarData[port], min, max);
                        if (min < setMin)
                            setMin = min;
                        if (max > setMax)
                            setMax = max;
                    }
                    break;
                }

                default:
                    if (generateOutput)
                        dataSet[objectIndex] = NULL;
                    break;
                }
            }
            else
            {
                if (generateOutput)
                    dataSet[objectIndex] = NULL;
                newObjects.push_back(NULL);
            }
            objectIndex++;
        }

        if (generateOutput)
        {
            for (int index = 0; index < objectIndex; index++)
            {
                if (dataSet[index])
                {
                    char min[16], max[16];
                    snprintf(min, 16, "%f", setMin);
                    snprintf(max, 16, "%f", setMax);
                    dataSet[index]->addAttribute("MIN", min);
                    dataSet[index]->addAttribute("MAX", max);
                    if (label != "")
                        dataSet[index]->addAttribute("LABEL", label.c_str());
                }
            }
            p_ScalarData[port]->setCurrentObject(new coDoSet(p_ScalarData[port]->getObjName(), pi->size(), dataSet));
        }

        newPorts.push_back(newObjects);
        port++;
    }

    if (!generateOutput)
        traverse(newPorts, output);
    else
    {
        p_UnstrGrid->setCurrentObject(
            new coDoSet(p_UnstrGrid->getObjName(), numObjects, usgSet));
    }
}

int ReadRECOMDx::compute(const char *)
{
    if (!action)
        return STOP_PIPELINE;

    std::vector<std::vector<DxObject *> > ports;

    for (int index = 0; index < maxDataPorts; index++)
    {
        std::vector<DxObject *> obj;

        std::string name = p_ScalarChoice[index]->getActLabel();
        DxObject *object = NULL;
        DxObjectMap::iterator d = action->objects_.find(name.c_str());
        if (d != action->objects_.end())
            object = d->second;

        obj.push_back(object);
        ports.push_back(obj);
    }
    traverse(ports);

    return CONTINUE_PIPELINE;
}

void ReadRECOMDx::param(const char *p, bool inMapLoading)
{
    if (inMapLoading)
        return;

    std::string name(p);
    if (name == "filename")
    {
        // First parse the dx-File and collect all
        // informations of the structure

        const char *fileName = p_filePath->getValue();

        if (p_defaultIsLittleEndian->getValue())
            action = new actionClass(Parser::LSB);
        else
            action = new actionClass(Parser::MSB);

        Parser *parser = new Parser(action, fileName);
        if (!parser->isOpen())
        {
            Covise::sendError("file %s could not be opened", fileName);
            delete action;
            action = NULL;
            delete parser;
            return;
        }
        parser->yyparse();

        if (!parser->isCorrect())
        {
            delete action;
            action = NULL;
            delete parser;
            return;
        }

        DxObjectMap::iterator d = action->objects_.find("default");
        if (d != action->objects_.end())
        {
            DxObject *o = d->second;
            MemberList mList = o->getMemberList();

            std::vector<std::string> items;
            items.push_back("---");
            for (int m = 0; m < mList.size(); m++)
            {
                addToGroup(mList[m]->getName(), mList[m]->getObject()->getName());
                items.push_back(mList[m]->getObject()->getName());
            }

            for (int index = 0; index < maxDataPorts; index++)
            {
                int selected = index + 1 < items.size() ? index + 1 : 0;
                p_ScalarChoice[index]->setValue(items.size(), items, selected);
            }
        }
    }
}

coDoUnstructuredGrid *ReadRECOMDx::generateUSG(DxObject *positions,
                                               DxObject *connections,
                                               coOutputPort *port)
{
    if (!positions || !connections)
        return NULL;

    const char *fileName = p_filePath->getValue();
    DxFile *dataFile = new DxFile(fileName, true);
    if (!dataFile->isValid())
        return NULL;

    int nconn = connections->getShape() * connections->getItems();
    int ncoord = positions->getItems();
    int nelem = connections->getItems();

    char *name = new char[128];
    snprintf(name, 128, "%s_%s", port->getObjName(), positions->getName());

    coDoUnstructuredGrid *usg = new coDoUnstructuredGrid(name, nelem, nconn, ncoord, 1);

    int *el, *cl, *tl;
    float *xc, *yc, *zc;
    usg->getAddresses(&el, &cl, &xc, &yc, &zc);
    usg->getTypeList(&tl);

    dataFile->readCoords(xc, 1.0, yc, 1.0, zc, 1.0,
                         positions->getDataOffset(),
                         positions->getItems(),
                         positions->getByteOrder());

    int offs = connections->getDataOffset();
    int shap = connections->getShape();
    int ite = connections->getItems();
    int bo = connections->getByteOrder();

    dataFile->readConnections(cl, offs, shap, ite, bo);

    std::string type = connections->getElementType();
    int gridType;
    int numVerts;

    if (type == "cubes")
    {
        gridType = TYPE_HEXAEDER;
        numVerts = 8;
    }
    else if (type == "quads")
    {
        gridType = TYPE_QUAD;
        numVerts = 4;
    }
    else if (type == "triangles")
    {
        gridType = TYPE_TRIANGLE;
        numVerts = 3;
    }
    else
    {
        Covise::sendError("The element type %s is not yet supported",
                          type.c_str());
    }

    int pos = 0;
    for (int index = 0; index < nelem; index++)
    {
        el[index] = pos;
        tl[index] = gridType;
        pos += numVerts;
    }

    return usg;
}

coDistributedObject *ReadRECOMDx::generateData(DxObject *object,
                                               coOutputPort *port,
                                               float &min, float &max)
{
    if (!object)
        return NULL;

    const char *fileName = p_filePath->getValue();
    DxFile *dataFile = new DxFile(fileName, true);
    if (!dataFile->isValid())
        return NULL;

    // if the data object does not contain a rank, it is scalar
    int rank = object->getRank();
    if (!rank)
        rank = 1;

    char *name = new char[128];
    snprintf(name, 128, "%s_%s", port->getObjName(), object->getName());

    coDistributedObject *data;
    float **dataPointers = new float *[rank];

    switch (rank)
    {
    case 1:
    {
        data = new coDoFloat(name, object->getItems());
        ((coDoFloat *)data)->getAddress(&dataPointers[0]);
        break;
    }

    case 3:
    {
        data = new coDoVec3(name, object->getItems());
        ((coDoVec3 *)data)->getAddresses(&dataPointers[0], &dataPointers[1], &dataPointers[2]);
        break;
    }
    }

    dataFile->readData(dataPointers, object->getDataOffset(), rank,
                       object->getItems(), object->getByteOrder(), min, max);

    delete[] dataPointers;

    return data;
}

void ReadRECOMDx::addToGroup(std::string object, std::string group)
{
    if (!action)
        return;

    DxObjectMap::iterator d = action->objects_.find(object.c_str());

    if (d != action->objects_.end())
    {

        switch (d->second->getObjectClass())
        {
        case Parser::MULTIGRID:
        case Parser::GROUP:
        {
            MemberList mList = d->second->getMemberList();
            for (int m = 0; m < mList.size(); m++)
                addToGroup(mList[m]->getObject()->getName(), group);

            break;
        }

        case Parser::FIELD:
        {
            objects[d->second->getData()] = group;
            break;
        }

        default:
            break;
        }
    }
}

MODULE_MAIN(Reader, ReadRECOMDx)
