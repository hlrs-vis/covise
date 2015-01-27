/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "SPartToU.h"
#include <util/coString.h>
#include "Map1D.h"

int
main(int argc, char *argv[])
{
    SPartToU *application = new SPartToU;

    application->start(argc, argv);

    return 0;
}

inline int
SPartToU::IndicesToNode(int x, int y, int z)
{
    return ((x * size_y_ + y) * size_z_ + z);
}

int
SPartToU::compute()
{
    // only non-null if we get "good", non-dummy data
    index_ = NULL;
    x_c_ = NULL;
    y_c_ = NULL;
    z_c_ = NULL;

    // Open objects, do not care about data objects
    // check types and dimensions
    if (Diagnose())
    {
        return FAIL;
    }

    // prepare a map with nodes marked with 0 or +1
    // count first those nodes
    int c_x, c_y, c_z;
    ia<int> nodesForU;

    int numSelems = (size_x_ - 1) * (size_y_ - 1) * (size_z_ - 1);
    int *markElems = new int[numSelems];
    memset(markElems, 0, numSelems * sizeof(int));

    for (c_x = 0; c_x < size_x_; ++c_x)
    {
        int base_x = c_x * size_y_;
        for (c_y = 0; c_y < size_y_; ++c_y)
        {
            int base_y = (base_x + c_y) * size_z_;
            for (c_z = 0; c_z < size_z_; ++c_z)
            {
                int node = base_y + c_z;
                if (!index_ || index_[node] != 0) // FIXME
                {
                    // mark up to 8 elements
                    RegisterElements(c_x, c_y, c_z, markElems);
                }
            }
        }
    }

    // Now we register the nodes as required according to the regis. elements
    int *markNodes = new int[size_x_ * size_y_ * size_z_];
    memset(markNodes, 0, size_x_ * size_y_ * size_z_ * sizeof(int));
    for (c_x = 0; c_x < size_x_ - 1; ++c_x)
    {
        int base_x = c_x * (size_y_ - 1);
        for (c_y = 0; c_y < size_y_ - 1; ++c_y)
        {
            int base_y = (base_x + c_y) * (size_z_ - 1);
            for (c_z = 0; c_z < size_z_ - 1; ++c_z)
            {
                int elem = base_y + c_z;
                if (markElems[elem] == 8)
                {
                    RegisterNodes(c_x, c_y, c_z, markNodes);
                }
            }
        }
    }

    for (c_x = 0; c_x < size_x_; ++c_x)
    {
        int base_x = c_x * size_y_;
        for (c_y = 0; c_y < size_y_; ++c_y)
        {
            int base_y = (base_x + c_y) * size_z_;
            for (c_z = 0; c_z < size_z_; ++c_z)
            {
                int node = base_y + c_z;
                if (markNodes[node] != 0)
                {
                    nodesForU.push_back(node);
                }
            }
        }
    }
    delete[] markNodes;

    Map1D mapNodes4u(nodesForU.size(), nodesForU.getArray());

    // loop over elements 4u,
    // check if they have at least one node marked with 0 or +1
    ia<int> elList;
    ia<int> tlList;
    int countVertices = 0;
    ia<int> connList;
    for (c_x = 0; c_x < size_x_ - 1; ++c_x)
    {
        int base_x = c_x * (size_y_ - 1);
        for (c_y = 0; c_y < size_y_ - 1; ++c_y)
        {
            int base_y = (base_x + c_y) * (size_z_ - 1);
            for (c_z = 0; c_z < size_z_ - 1; ++c_z)
            {
                int elem = base_y + c_z;
                if (markElems[elem] == 8)
                {
                    // trivially fill element list
                    elList.push_back(countVertices);
                    countVertices += 8;
                    tlList.push_back(TYPE_HEXAEDER);
                    // use the node map in order to fill the connectivity list
                    connList.push_back(mapNodes4u[IndicesToNode(c_x, c_y, c_z)]);
                    connList.push_back(mapNodes4u[IndicesToNode(c_x + 1, c_y, c_z)]);
                    connList.push_back(mapNodes4u[IndicesToNode(c_x + 1, c_y + 1, c_z)]);
                    connList.push_back(mapNodes4u[IndicesToNode(c_x, c_y + 1, c_z)]);
                    connList.push_back(mapNodes4u[IndicesToNode(c_x, c_y, c_z + 1)]);
                    connList.push_back(mapNodes4u[IndicesToNode(c_x + 1, c_y, c_z + 1)]);
                    connList.push_back(mapNodes4u[IndicesToNode(c_x + 1, c_y + 1, c_z + 1)]);
                    connList.push_back(mapNodes4u[IndicesToNode(c_x, c_y + 1, c_z + 1)]);
                }
            }
        }
    }
    // fill coordinate arrays
    ia<float> x_c;
    ia<float> y_c;
    ia<float> z_c;
    int node;
    for (node = 0; node < nodesForU.size(); ++node)
    {
        x_c.push_back(x_c_[nodesForU.getArray()[node]]);
        y_c.push_back(y_c_[nodesForU.getArray()[node]]);
        z_c.push_back(z_c_[nodesForU.getArray()[node]]);
    }

    // make output grid object
    coDoUnstructuredGrid *outUGrid = new coDoUnstructuredGrid(p_outUGrid_->getObjName(),
                                                              elList.size(), connList.size(),
                                                              nodesForU.size(),
                                                              elList.getArray(), connList.getArray(),
                                                              x_c.getArray(), y_c.getArray(),
                                                              z_c.getArray(), tlList.getArray());
    p_outUGrid_->setCurrentObject(outUGrid);

    // fill also data arrays
    createGridData(nodesForU);

    delete[] markElems;
    return SUCCESS;
}

SPartToU::SPartToU()
    : coSimpleModule("Create an UNSGRD out of a part of a STRGRD")
{
    p_inSGrid_ = addInputPort("InSGrid", "coDoStructuredGrid", "input structured grid");
    p_codes_ = addInputPort("Codes", "coDoIntArr", "Input codes");
    int port;
    for (port = 0; port < NUM_DATA_PORTS; ++port)
    {
        coString name("InData_");
        coString descr("Input data port ");
        char buf[64];
        sprintf(buf, "%d", port);
        name += buf;
        descr += buf;
        p_inData_[port] = addInputPort(name,
                                       "coDoFloat|coDoVec3", descr);
        p_inData_[port]->setRequired(0);
    }
    // output grid ports
    p_outUGrid_ = addOutputPort("outUGrid", "coDoUnstructuredGrid",
                                "output unstructured grid");
    for (port = 0; port < NUM_DATA_PORTS; ++port)
    {
        coString name("OutGridData_");
        coString descr("Output grid data port ");
        char buf[64];
        sprintf(buf, "%d", port);
        name += buf;
        descr += buf;
        p_outData_[port] = addOutputPort(name,
                                         "coDoFloat|coDoVec3", descr);
    }

    // ports polygon ports
    p_outPoly_ = addOutputPort("outPoly", "coDoPolygons",
                               "output polygons");
    for (port = 0; port < NUM_DATA_PORTS; ++port)
    {
        coString name("OutPolyData_");
        coString descr("Output poly data port ");
        char buf[64];
        sprintf(buf, "%d", port);
        name += buf;
        descr += buf;
        p_outDataPoly_[port] = addOutputPort(name,
                                             "coDoFloat|coDoVec3", descr);
    }
}

int
SPartToU::Diagnose()
{
    // check types
    coDistributedObject *sgrid = p_inSGrid_->getCurrentObject();
    if (!sgrid || !sgrid->objectOk() || !sgrid->isType("STRGRD"))
    {
        sendError("Could not get a correct STRGRD object");
        return -1;
    }

    coDoStructuredGrid *SGrid = dynamic_cast<coDoStructuredGrid *>(sgrid);
    SGrid->getGridSize(&size_x_, &size_y_, &size_z_);
    SGrid->getAddresses(&x_c_, &y_c_, &z_c_);

    coDistributedObject *codes = p_codes_->getCurrentObject();
    if (!codes || !codes->objectOk() || !codes->isType("INTARR"))
    {
        sendError("Could not get a correct INTARR object");
        return -1;
    }

    // get and check dimensions
    coDoIntArr *Codes = dynamic_cast<coDoIntArr *>(codes);
    int codesSize = Codes->get_dim(0);
    if (codesSize != 0 && codesSize != size_x_ * size_y_ * size_z_)
    {
        sendError("Codes and grid: non-matching number of points");
        return -1;
    }
    else if (codesSize == 0)
    {
        index_ = NULL;
    }
    else
    {
        Codes->getAddress(&index_);
    }

    return 0;
}

void
    // mark up to 8 elements
    SPartToU::RegisterElements(int c_x, int c_y, int c_z, int *markElems)
{
    int e_x, e_y, e_z;
    int base_x = c_x - 1;
    int base_y = c_y - 1;
    int base_z = c_z - 1;

    for (e_x = base_x; e_x <= c_x; ++e_x)
    {
        if (e_x < 0 || e_x >= size_x_ - 1)
        {
            continue;
        }
        for (e_y = base_y; e_y <= c_y; ++e_y)
        {
            if (e_y < 0 || e_y >= size_y_ - 1)
            {
                continue;
            }
            for (e_z = base_z; e_z <= c_z; ++e_z)
            {
                if (e_z < 0 || e_z >= size_z_ - 1)
                {
                    continue;
                }
                int elem = (e_x * (size_y_ - 1) + e_y) * (size_z_ - 1) + e_z;
                ++markElems[elem];
            }
        }
    }
}

void
    //mark all nodes
    SPartToU::RegisterNodes(int c_x, int c_y, int c_z, int *markNodes)
{
    ++markNodes[IndicesToNode(c_x, c_y, c_z)];
    ++markNodes[IndicesToNode(c_x, c_y, c_z + 1)];
    ++markNodes[IndicesToNode(c_x, c_y + 1, c_z)];
    ++markNodes[IndicesToNode(c_x, c_y + 1, c_z + 1)];
    ++markNodes[IndicesToNode(c_x + 1, c_y, c_z)];
    ++markNodes[IndicesToNode(c_x + 1, c_y, c_z + 1)];
    ++markNodes[IndicesToNode(c_x + 1, c_y + 1, c_z)];
    ++markNodes[IndicesToNode(c_x + 1, c_y + 1, c_z + 1)];
}

void
SPartToU::copyAttributesToOutObj(coInputPort **input_ports,
                                 coOutputPort **output_ports,
                                 int i)
{
    int j;
    if (i == 0 || i == NUM_DATA_PORTS + 1)
    {
        j = 0;
    }
    else if (i <= NUM_DATA_PORTS)
    {
        j = i + 1;
    }
    else if (i > NUM_DATA_PORTS + 1)
    {
        j = i - NUM_DATA_PORTS;
    }
    if (input_ports[j] && output_ports[i])
        copyAttributes(output_ports[i]->getCurrentObject(), input_ports[j]->getCurrentObject());
}

void
SPartToU::createGridData(const ia<int> &ulist)
{
    int port;
    int size = ulist.size();
    for (port = 0; port < NUM_DATA_PORTS; ++port)
    {
        coDistributedObject *data = p_inData_[port]->getCurrentObject();
        coDistributedObject *odata = NULL;
        float *uo = NULL, *vo = NULL, *wo = NULL;
        float *ui = NULL, *vi = NULL, *wi = NULL;
        const char *oname = p_outData_[port]->getObjName();
        if (!data)
        {
            continue;
        }
        else if (!data->objectOk())
        {
            sendWarning("Data is not OK");
            continue;
        }
        else if (data->isType("STRSDT"))
        {
            odata = new coDoFloat(oname, size);
            dynamic_cast<coDoFloat *>(odata)->getAddress(&uo);
            dynamic_cast<coDoFloat *>(data)->getAddress(&ui);
            int node;
            const int *uarray = ulist.getArray();
            for (node = 0; node < size; ++node)
            {
                uo[node] = ui[uarray[node]];
            }
        }
        else if (data->isType("STRVDT"))
        {
            odata = new coDoVec3(oname, size);
            dynamic_cast<coDoVec3 *>(odata)->getAddresses(&uo, &vo, &wo);
            dynamic_cast<coDoVec3 *>(data)->getAddresses(&ui, &vi, &wi);
            int node;
            for (node = 0; node < size; ++node)
            {
                uo[node] = ui[ulist.getArray()[node]];
                vo[node] = vi[ulist.getArray()[node]];
                wo[node] = wi[ulist.getArray()[node]];
            }
        }
        p_outData_[port]->setCurrentObject(odata);
    }
}
