/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ReadANSYS.h"
#include <stdlib.h>
#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoSet.h>

extern const char *dofname[];
extern const char *exdofname[];

int compar(const void *l, const void *r)
{
    int *left = (int *)l;
    int *right = (int *)r;
    if (*left < *right)
        return -1;
    if (*left == *right)
        return 0;
    return 1;
}

int
ReadANSYS::SetNodeChoices()
{
    const char **Choices = NULL;
    std::vector<int> variable_codes;
    open_err_ = readRST_.OpenFile(FileName_ /*p_rst_->getValue()*/);
    // loop over rstheader_.numsets_ and accumulate
    // in variable_codes non repeated codes
    int numset;
    for (numset = 1; numset <= readRST_.getNumTimeSteps(); ++numset)
    {
        int ErrorReadSHDR = readRST_.ReadSHDR(numset);
        if (ErrorReadSHDR)
        {
            sendError("SetNodeChoices: Error in ReadSHDR");
            return ErrorReadSHDR;
        }
        // the important info is in solheader_.numdofs_,
        // solheader_.dof_, solheader_.numexdofs_, solheader_.exdof_
        int new_dof;
        for (new_dof = 0; new_dof < readRST_.solheader_.numdofs_; ++new_dof)
        {
            int old_dof;
            int repeated = 0;
            for (old_dof = 0; old_dof < variable_codes.size(); ++old_dof)
            {
                if (readRST_.solheader_.dof_[new_dof] == variable_codes[old_dof])
                {
                    // dof is repeated
                    repeated = 1;
                    break;
                }
            }
            if (!repeated)
            {
                variable_codes.push_back(readRST_.solheader_.dof_[new_dof]);
            }
        }
        // now add extra dofs
        int extra_new_dof;
        for (extra_new_dof = 0; extra_new_dof < readRST_.solheader_.numexdofs_; ++extra_new_dof)
        {
            int old_dof;
            int repeated = 0;
            for (old_dof = 0; old_dof < variable_codes.size(); ++old_dof)
            {
                if (readRST_.solheader_.exdof_[extra_new_dof] == variable_codes[old_dof] - EX_OFFSET)
                {
                    // dof is repeated
                    repeated = 1;
                    break;
                }
            }
            if (!repeated)
            {
                variable_codes.push_back(readRST_.solheader_.exdof_[extra_new_dof] + EX_OFFSET);
            }
        }
    }
    // now order variable_codes;
    int *var_codes = &variable_codes[0];
    qsort(var_codes, variable_codes.size(), sizeof(int), compar);

    std::vector<int> my_variable_codes;
    int code;
    for (code = 0; code < variable_codes.size(); ++code)
    {
        my_variable_codes.push_back(variable_codes[code]);
        size_t my_size = my_variable_codes.size();
        // extra dofs are always scalar fields
        switch (variable_codes[code])
        {
        case 3:
            if (my_size >= 3
                && my_variable_codes[my_size - 2] == 2
                && my_variable_codes[my_size - 3] == 1)
            {
                my_variable_codes.push_back(variable_codes[code] + V_OFFSET);
            }
            break;
        case 6:
            if (my_size >= 3
                && my_variable_codes[my_size - 2] == 5
                && my_variable_codes[my_size - 3] == 4)
            {
                my_variable_codes.push_back(variable_codes[code] + V_OFFSET);
            }
            break;
        case 9:
            if (my_size >= 3
                && my_variable_codes[my_size - 2] == 8
                && my_variable_codes[my_size - 3] == 7)
            {
                my_variable_codes.push_back(variable_codes[code] + V_OFFSET);
            }
            break;
        case 12:
            if (my_size >= 3
                && my_variable_codes[my_size - 2] == 11
                && my_variable_codes[my_size - 3] == 10)
            {
                my_variable_codes.push_back(variable_codes[code] + V_OFFSET);
            }
            break;
        }
    }
    Choices = new const char *[1 + my_variable_codes.size()];
    Choices[0] = "none";
    int choice;
    for (choice = 0; choice < my_variable_codes.size(); ++choice)
    {
        if (my_variable_codes[choice] < V_OFFSET)
        {
            Choices[choice + 1] = dofname[my_variable_codes[choice] - 1];
        }
        else if (my_variable_codes[choice] < EX_OFFSET)
        {
            int substract = my_variable_codes[choice] - V_OFFSET;
            switch (substract)
            {
            case 3:
                Choices[choice + 1] = "U";
                break;
            case 6:
                Choices[choice + 1] = "ROT";
                break;
            case 9:
                Choices[choice + 1] = "A";
                break;
            case 12:
                Choices[choice + 1] = "V";
                break;
            }
        }
        else
        {
            int substract = my_variable_codes[choice] - EX_OFFSET;
            Choices[choice + 1] = exdofname[substract - 1];
        }
    }
    if (!inMapLoading && !p_file_name_->isConnected())
    {
        p_nsol_->setValue((int)my_variable_codes.size() + 1, Choices, 0);
    }
    // set the correct state for DOFOptions_
    DOFOptions_.num_options_ = (int)my_variable_codes.size() + 1;
    delete[] DOFOptions_.options_;
    delete[] DOFOptions_.codes_;
    DOFOptions_.options_ = new std::string[DOFOptions_.num_options_];
    DOFOptions_.codes_ = new std::vector<int>[DOFOptions_.num_options_];
    DOFOptions_.options_[0] = "none";
    DOFOptions_.codes_[0].push_back(0);
    for (choice = 0; choice < my_variable_codes.size(); ++choice)
    {
        DOFOptions_.options_[choice + 1] = Choices[choice + 1];
        if (my_variable_codes[choice] < V_OFFSET)
        {
            DOFOptions_.codes_[choice + 1].push_back(my_variable_codes[choice]);
        }
        else if (my_variable_codes[choice] < EX_OFFSET)
        {
            int substract = my_variable_codes[choice] - V_OFFSET;
            DOFOptions_.codes_[choice + 1].push_back(substract - 2);
            DOFOptions_.codes_[choice + 1].push_back(substract - 1);
            DOFOptions_.codes_[choice + 1].push_back(substract - 0);
        }
        else
        {
            DOFOptions_.codes_[choice + 1].push_back(my_variable_codes[choice]);
        }
    }
    delete[] Choices;
    return 0;
}

void
ReadANSYS::MakeGridAndObjects(const std::string &gridName,
                              std::vector<int> &e_l,
                              std::vector<int> &v_l,
                              std::vector<float> &x_l,
                              std::vector<float> &y_l,
                              std::vector<float> &z_l,
                              std::vector<int> &t_l,
                              const std::string &dataName,
                              const float *const *field,
                              FieldType ftype,
                              const std::string &matName,
                              const int *materials,
                              std::vector<coDistributedObject *> &grid_set_list,
                              std::vector<coDistributedObject *> &data_set_list,
                              std::vector<coDistributedObject *> &mat_set_list)
{
    // first we have to make a list of marks for all nodes...
    // bit 0 -> on: bit in geometry
    // bit 1 -> on: bit in geometry and has data
    // bit 2 -> on: bit in geometry and is in grid with data
    // bit 3 -> on: bit in geometry and is in grid without data
    int *marks = new int[x_l.size()];
    int *markElems = new int[e_l.size()];
    memset(marks, 0, sizeof(int) * x_l.size());
    memset(markElems, 0, sizeof(int) * e_l.size());
    int elem, vert;
    // Set bit 0
    for (elem = 0; elem < e_l.size(); ++elem)
    {
        for (vert = 0; vert < UnstructuredGrid_Num_Nodes[t_l[elem]]; ++vert)
        {
            marks[v_l[e_l[elem] + vert]] = 1;
        }
    }
    int node;
    // Set bit 1
    for (node = 0; node < x_l.size(); ++node)
    {
        if (field[0][node] != ReadRST::FImpossible_
            && (ftype == SCALAR
                || (field[1][node] != ReadRST::FImpossible_
                    && field[2][node] != ReadRST::FImpossible_)))
        {
            marks[node] |= 2;
        }
    }
    for (elem = 0; elem < e_l.size(); ++elem)
    {
        int allNodesHaveData = 1;
        for (vert = 0; vert < UnstructuredGrid_Num_Nodes[t_l[elem]]; ++vert)
        {
            node = v_l[e_l[elem] + vert];
            if ((marks[node] & 2) == 0)
            {
                allNodesHaveData = 0;
                break;
            }
        }
        // Set bit 2
        if (allNodesHaveData)
        {
            for (vert = 0; vert < UnstructuredGrid_Num_Nodes[t_l[elem]]; ++vert)
            {
                node = v_l[e_l[elem] + vert];
                marks[node] |= 4;
            }
            markElems[elem] = 1;
        }
        // Set bit 3
        else
        {
            for (vert = 0; vert < UnstructuredGrid_Num_Nodes[t_l[elem]]; ++vert)
            {
                node = v_l[e_l[elem] + vert];
                marks[node] |= 8;
            }
        }
    }

    // make a map for decoding nodes with data and without data
    std::vector<int> nodes_with_data;
    std::vector<int> nodes_without_data;
    std::vector<float> x_l_data;
    std::vector<float> y_l_data;
    std::vector<float> z_l_data;
    std::vector<float> field_data[3];
    std::vector<float> x_l_no_data;
    std::vector<float> y_l_no_data;
    std::vector<float> z_l_no_data;
    for (node = 0; node < x_l.size(); ++node)
    {
        if (marks[node] & 4)
        {
            nodes_with_data.push_back(node);
            x_l_data.push_back(x_l[node]);
            y_l_data.push_back(y_l[node]);
            z_l_data.push_back(z_l[node]);
            field_data[0].push_back(field[0][node]);
            if (ftype == VECTOR)
            {
                field_data[1].push_back(field[1][node]);
                field_data[2].push_back(field[2][node]);
            }
        }
        if (marks[node] & 8)
        {
            nodes_without_data.push_back(node);
            x_l_no_data.push_back(x_l[node]);
            y_l_no_data.push_back(y_l[node]);
            z_l_no_data.push_back(z_l[node]);
        }
    }
    int *nodes = NULL;
    if (nodes_with_data.size())
    {
        nodes = &nodes_with_data[0];
    }
    Map1D nodesDataDecode((int)nodes_with_data.size(), &nodes_with_data[0]);
    if (nodes_without_data.size())
    {
        nodes = &nodes_without_data[0];
    }
    Map1D nodesNoDataDecode((int)nodes_without_data.size(), nodes);

    // now create data for grid with data (and without)
    std::vector<int> e_l_data;
    std::vector<int> v_l_data;
    std::vector<int> t_l_data;
    std::vector<int> m_l_data;
    std::vector<int> e_l_no_data;
    std::vector<int> v_l_no_data;
    std::vector<int> t_l_no_data;
    std::vector<int> m_l_no_data;

    int v_count_data = 0;
    int v_count_no_data = 0;

    for (elem = 0; elem < e_l.size(); ++elem)
    {
        if (markElems[elem])
        {
            e_l_data.push_back(v_count_data);
            m_l_data.push_back(materials[elem]);
            v_count_data += UnstructuredGrid_Num_Nodes[t_l[elem]];
            for (vert = 0; vert < UnstructuredGrid_Num_Nodes[t_l[elem]]; ++vert)
            {
                v_l_data.push_back(nodesDataDecode[v_l[e_l[elem] + vert]]);
            }
            t_l_data.push_back(t_l[elem]);
        }
        else
        {
            e_l_no_data.push_back(v_count_no_data);
            m_l_no_data.push_back(materials[elem]);
            v_count_no_data += UnstructuredGrid_Num_Nodes[t_l[elem]];
            for (vert = 0; vert < UnstructuredGrid_Num_Nodes[t_l[elem]]; ++vert)
            {
                v_l_no_data.push_back(nodesNoDataDecode[v_l[e_l[elem] + vert]]);
            }
            t_l_no_data.push_back(t_l[elem]);
        }
    }

    // now create objects
    if (t_l_no_data.size() == 0) // the whole grid has data
    {
        coDoUnstructuredGrid *p_grid = new coDoUnstructuredGrid(gridName, (int)e_l_data.size(),
                                                                (int)v_l_data.size(), (int)x_l_data.size(),
                                                                &e_l_data[0],
                                                                &v_l_data[0],
                                                                &x_l_data[0],
                                                                &y_l_data[0],
                                                                &z_l_data[0],
                                                                &t_l_data[0]);
        p_grid->addAttribute("COLOR", "White");
        grid_set_list.push_back(p_grid);
        if (ftype == SCALAR)
        {
            data_set_list.push_back(new coDoFloat(dataName,
                                                  (int)field_data[0].size(),
                                                  &field_data[0][0]));
        }
        else if (ftype == VECTOR)
        {
            data_set_list.push_back(new coDoVec3(dataName,
                                                 (int)field_data[0].size(),
                                                 &field_data[0][0],
                                                 &field_data[1][0],
                                                 &field_data[2][0]));
        }
        data_set_list[data_set_list.size() - 1]->addAttribute("SPECIES", DOFOptions_.options_[h_nsol_->getIValue()].c_str());
        int matSize = (int)m_l_data.size();
        coDoIntArr *matOut = new coDoIntArr(matName, 1, &matSize);
        memcpy(matOut->getAddress(), &m_l_data[0], m_l_data.size() * sizeof(int));
        mat_set_list.push_back(matOut);
    }
    else // there is a grid without data
    {
        std::string gridNameData(gridName);
        gridNameData += "_data";
        std::string gridNameNoData(gridName);
        gridNameNoData += "_no_data";
        std::string dataNameData(dataName);
        dataNameData += "_data";
        std::string dataNameNoData(dataName); // for the dummy
        dataNameNoData += "_no_data";
        std::string matNameData(matName);
        matNameData += "_data";
        std::string matNameNoData(matName);
        matNameNoData += "_no_data";

        coDistributedObject *grid_list[3];
        coDistributedObject *data_list[3];
        coDistributedObject *mat_list[3];

        grid_list[2] = NULL;
        data_list[2] = NULL;
        mat_list[2] = NULL;

        // grid
        grid_list[0] = new coDoUnstructuredGrid(gridNameData, (int)e_l_data.size(),
                                                (int)v_l_data.size(), (int)x_l_data.size(),
                                                &e_l_data[0],
                                                &v_l_data[0],
                                                &x_l_data[0],
                                                &y_l_data[0],
                                                &z_l_data[0],
                                                &t_l_data[0]);
        grid_list[0]->addAttribute("COLOR", "White");
        grid_list[1] = new coDoUnstructuredGrid(gridNameNoData, (int)e_l_no_data.size(),
                                                (int)v_l_no_data.size(), (int)x_l_no_data.size(),
                                                &e_l_no_data[0],
                                                &v_l_no_data[0],
                                                &x_l_no_data[0],
                                                &y_l_no_data[0],
                                                &z_l_no_data[0],
                                                &t_l_no_data[0]);
        grid_list[1]->addAttribute("COLOR", "White");
        // material
        int matSize = int(m_l_data.size());
        coDoIntArr *matOutData = new coDoIntArr(matNameData, 1, &matSize);
        memcpy(matOutData->getAddress(), &m_l_data[0], m_l_data.size() * sizeof(int));
        mat_list[0] = matOutData;
        matSize = int(m_l_no_data.size());
        coDoIntArr *matOutNoData = new coDoIntArr(matNameNoData, 1, &matSize);
        memcpy(matOutNoData->getAddress(), &m_l_no_data[0], m_l_no_data.size() * sizeof(int));
        mat_list[1] = matOutNoData;
        // data
        if (ftype == SCALAR)
        {
            data_list[0] = new coDoFloat(dataNameData,
                                         (int)field_data[0].size(),
                                         &field_data[0][0]);
            data_list[1] = new coDoFloat(dataNameNoData, 0);
        }
        else if (ftype == VECTOR)
        {
            data_list[0] = new coDoVec3(dataNameData,
                                        (int)field_data[0].size(),
                                        &field_data[0][0],
                                        &field_data[1][0],
                                        &field_data[2][0]);
            data_list[1] = new coDoVec3(dataNameNoData, 0);
        }
        if (!p_file_name_->isConnected())
        {
            data_list[0]->addAttribute("SPECIES", p_nsol_->getActLabel());
            data_list[1]->addAttribute("SPECIES", p_nsol_->getActLabel());
        }
        grid_set_list.push_back(new coDoSet(gridName, grid_list));
        data_set_list.push_back(new coDoSet(dataName, data_list));
        mat_set_list.push_back(new coDoSet(matName, mat_list));
    }
    delete[] marks;
    delete[] markElems;
}

void
ReadANSYS::useReadANSYSAttribute(const coDistributedObject *inName)
{
    const char *wert = NULL;
    if (inName == NULL)
    {
        return;
    }
    wert = inName->getAttribute("READ_ANSYS");
    if (wert == NULL) // perhaps the attribute is hidden in a set structure
    {
        if (inName->isType("SETELE"))
        {
            int no_elems;
            const coDistributedObject *const *setList = ((coDoSet *)(inName))->getAllElements(&no_elems);
            int elem;
            for (elem = 0; elem < no_elems; ++elem)
            {
                useReadANSYSAttribute(setList[elem]);
            }
        }
        return;
    }
    istringstream pvalues(wert);
    char *value = new char[strlen(wert) + 1];
    while (pvalues.getline(value, strlen(wert) + 1))
    {
        int param;
        for (param = 0; param < hparams_.size(); ++param)
        {
            hparams_[param]->load(value);
        }
    }
    delete[] value;
}
