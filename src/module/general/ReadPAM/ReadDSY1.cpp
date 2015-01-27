/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#define COIDENT "$Header: /vobs/covise/src/application/general/READ_PAM/ReadDSY1.cpp /main/vir_main/12 29-Oct-2001.15:56:38 dirk_te $"
#include <util/coIdent.h>
#include <do/coDoIntArr.h>

// "Deep" functions
// ReadDSY1.cpp contains mainly functions called from ReadDSY.cpp

#include "ReadDSY.h"

// AQUI
// tails of the set element names
const char *Beschriftungen[noTypes] = {
    "Solid", "Shell", "Tool", "Bar",
    "Beam", "Bar1", "Spring", "SphJoint", "FTJoint",
    "SptWeld", "Jet", "KineJoint", "MISWeld"
#ifdef _INCLUDE_SPH_
    ,
    "SPH"
#endif
};

coDistributedObject *dummy(std::string &name, int type)
{
    coDistributedObject *res = 0;
    switch (type)
    {
    case coStringObj::SCALAR:
        res = new coDoFloat(name, 0);
        break;
    case coStringObj::VECTOR:
        res = new coDoVec3(name, 0);
        break;
        // other cases in construction...
    }
    return res;
}

coDistributedObject *ReadDSY::realObj(std::string &name, int this_state,
                                      coStringObj::ElemType e_type, int first, int last,
                                      int type, INDEX ind)
{
    float zeit;
    int cell_type;
    coDistributedObject *res = 0;
    float *varval = new float[type * (last - first + 1)];
    for (cell_type = 0; cell_type < noTypes; ++cell_type)
    {
        if (e_type == Types[cell_type])
        {
            ReadCellVar(cell_type, &this_state, &type, ind, &first, &last, &zeit, varval);
            break;
        }
    }
    if (rtn_)
    {
        Covise::sendInfo("Could not read some variables");
        delete[] varval;
        return (dummy(name, type));
    }
    int i;
    float *u_c, *v_c, *w_c;
    switch (type)
    {
    case coStringObj::SCALAR:
    {
        coDoFloat *s_data = new coDoFloat(name,
                                          last - first + 1);
        s_data->getAddress(&u_c);
        memcpy(u_c, varval, sizeof(float) * (last - first + 1));
        res = s_data;
    }
    break;
    case coStringObj::VECTOR:
    {
        coDoVec3 *v_data = new coDoVec3(name,
                                        last - first + 1);
        v_data->getAddresses(&u_c, &v_c, &w_c);
        for (i = 0; i < last - first + 1; ++i)
        {
            u_c[i] = varval[3 * i + 0];
            v_c[i] = varval[3 * i + 1];
            w_c[i] = varval[3 * i + 2];
        }
        res = v_data;
    }
    break;
    }
    delete[] varval;
    return res;
}

coDistributedObject *ReadDSY::realTensorObj(std::string &name, int this_state,
                                            coStringObj::ElemType e_type, int first, int last,
                                            coDoTensor::TensorType ttype, int *req_label_ind)
{
    float zeit;
    int cell_type;
    int i;
    INDEX ind;
    // first fill ind...
    for (i = 0; i < ttype; ++i)
    {
        ind[i] = cell_contents_[req_label_ind[i]].index_[0];
    }
    float *varval;
    coDoTensor *t_data = new coDoTensor(name,
                                        last - first + 1, ttype);
    t_data->getAddress(&varval);
    for (cell_type = 0; cell_type < noTypes; ++cell_type)
    {
        if (e_type == Types[cell_type])
        {
            int ttype_int = ttype;
            ReadCellVar(cell_type, &this_state, &ttype_int, ind, &first, &last, &zeit, varval);
            break;
        }
    }
    if (rtn_)
    {
        Covise::sendInfo("Could not read some variables");
        return (new coDoTensor(name, 0, ttype));
    }
    return t_data;
}

int ReadDSY::globalVarNotInFile(INDEX ind)
{
    int res = 1; // in pple not in file
    int one = 1;
    int time;
    float varval[1];
    dsysta_(&nstate_);
    for (time = 1; time <= nstate_; ++time)
    {
        dsyhgl_(&time, &time, &one, ind, varval, &rtn_);
        if (rtn_ != 0)
        {
            break;
        }
    }
    if (time > nstate_)
        res = 0;
    return res;
}

int ReadDSY::cellVarNotInFile(coStringObj::ElemType cell_type, INDEX ind)
{
    int res = 1; // in pple not in file
    int one = 1;
    float zeit;
    int i, time;
    float varval[1];
    dsysta_(&nstate_);
    for (i = 0; i < noTypes; ++i)
    {
        if (Types[i] == cell_type)
        {
            for (time = 1; time <= nstate_; ++time)
            {
                ReadCellVar(i, &time, &one, ind, &one, &one, &zeit, varval);
                if (rtn_ == 0)
                {
                    res = 0;
                    break;
                }
            }
        }
    }
    return res;
}

coDoSet *ReadDSY::cellObjAtTime(std::string name, int time, int req_ind)
{
    std::string obj_name(name);
    char obj_name_app[32];
    sprintf(obj_name_app, "_%d", time);
    obj_name += obj_name_app;

    int cell_type;
    int no_elements = 0;
    for (cell_type = 0; cell_type < noTypes; ++cell_type)
        no_elements += (no_entities_[cell_type] > 0);

    coDistributedObject **setList = new coDistributedObject *[no_elements + 1];

    setList[no_elements] = 0;

    int element = 0;
    int this_state = fromIntToDSY(time);
    char realtime[32];
    sprintf(realtime, "%.5e", zeit_[this_state - 1]);

    for (cell_type = 0; cell_type < noTypes; ++cell_type)
        if (no_entities_[cell_type])
        {
            std::string entity_name(obj_name);
            entity_name += "_";
            entity_name += Beschriftungen[cell_type];
            if (cell_contents_[req_ind].ElemType_ == Types[cell_type])
            {
                setList[element] = realObj(entity_name, this_state,
                                           Types[cell_type], 1, no_entities_[cell_type],
                                           cell_contents_[req_ind].ObjType_,
                                           cell_contents_[req_ind].index_);
            }
            else
            {
                // create dummy
                setList[element] = dummy(entity_name,
                                         cell_contents_[req_ind].ObjType_);
            }
            char species[25];
            stripSpecies(species, cell_contents_[req_ind].ObjName_);
            setList[element]->addAttribute("SPECIES", species);
            setList[element]->addAttribute("REALTIME", realtime);
            ++element;
        }

    coDoSet *theObject = new coDoSet(obj_name, setList);

    int i;
    for (i = 0; i < no_elements; ++i)
        delete setList[i];
    delete[] setList;

    return theObject;
}

// req_ind refers to the tensor port: 0...TENSOR_PORTS
coDoSet *ReadDSY::tensorObjAtTime(const TensDescriptions &tdesc, int time, int req_ind)
{
    std::string obj_name(tdesc.requests[req_ind]);
    char obj_name_app[32];
    sprintf(obj_name_app, "_%d", time);
    obj_name += obj_name_app;

    int cell_type;
    int no_elements = 0;
    for (cell_type = 0; cell_type < noTypes; ++cell_type)
        no_elements += (no_entities_[cell_type] > 0);

    coDistributedObject **setList = new coDistributedObject *[no_elements + 1];

    setList[no_elements] = 0;

    int element = 0;
    int this_state = fromIntToDSY(time);

    for (cell_type = 0; cell_type < noTypes; ++cell_type)
        if (no_entities_[cell_type])
        {
            std::string entity_name(obj_name);
            entity_name += "_";
            entity_name += Beschriftungen[cell_type];
            if (cell_contents_[tdesc.req_label[9 * req_ind]].ElemType_ == Types[cell_type])
            {
                setList[element] = realTensorObj(entity_name, this_state,
                                                 Types[cell_type], 1, no_entities_[cell_type],
                                                 tdesc.ttype[req_ind],
                                                 &tdesc.req_label[9 * req_ind]);
            }
            else
            {
                // create dummy
                setList[element] = new coDoTensor(entity_name, 0, tdesc.ttype[req_ind]);
            }
            /* Sorry, species for tensors not yet implemented
               char species[25];
               stripSpecies(species,cell_contents_[req_ind].ObjName_);
               setList[element]->addAttribute("SPECIES",species);
      */
            ++element;
        }

    coDoSet *theObject = new coDoSet(obj_name, setList);

    int i;
    for (i = 0; i < no_elements; ++i)
        delete setList[i];
    delete[] setList;

    return theObject;
}

void ReadDSY::stripSpecies(char *species, std::string &title)
{
    int i, j;
    species[24] = '\0';
    for (i = 23; i >= 0; --i)
    {
        if (title[i] == ' ')
            title[i] = '\0';
        if (title[i] != '\0')
            break;
    }
    // Now we want to eliminate subchains like lVEC or gVEC
    // i is either -1 or something like a letter or a ')'
    if (i >= 5)
    {
        // search for first occurrence of '\0'
        for (j = 0; j < 25; ++j)
        {
            if (title[j] == '\0')
                break;
        }
        i = j - 1;
        if (i >= 5 && (strncmp("lVEC", &title[i - 4], 4) == 0 || strncmp("gVEC", &title[i - 4], 4) == 0) && title[i - 5] == ' ')
        {
            title[i - 5] = ')';
            title[i - 4] = '\0';
        }
    }

    strncpy(species, title.c_str(), 24);
}

coDoSet *ReadDSY::gridAtTime(std::string objName, std::string matName,
                             std::string elaName,
#ifdef _LOCAL_REFERENCES_
                             std::string refName,
#endif
                             int time,
                             coDistributedObject **matElem,
                             coDistributedObject **elaElem
#ifdef _LOCAL_REFERENCES_
                             ,
                             coDistributedObject **refElem
#endif
                             )
{
    std::string grid_name(objName);
    std::string material_name(matName);
    std::string elabel_name(elaName);
#ifdef _LOCAL_REFERENCES_
    std::string reference_name(refName);
#endif
    char grid_name_app[32];
    sprintf(grid_name_app, "_%d", time);
    grid_name += grid_name_app;
    material_name += grid_name_app;
    elabel_name += grid_name_app;
#ifdef _LOCAL_REFERENCES_
    reference_name += grid_name_app;
#endif

    int no_elements = 0;
    int cell_type;
    for (cell_type = 0; cell_type < noTypes; ++cell_type)
        no_elements += (no_entities_[cell_type] > 0);

    coDistributedObject **setList = new coDistributedObject *[no_elements + 1];
    coDistributedObject **setMatList = new coDistributedObject *[no_elements + 1];
    coDistributedObject **setElaList = new coDistributedObject *[no_elements + 1];
#ifdef _LOCAL_REFERENCES_
    coDistributedObject **setRefList = new coDistributedObject *[no_elements + 1];
    setRefList[no_elements] = 0;
#endif

    setList[no_elements] = 0;
    setMatList[no_elements] = 0;
    setElaList[no_elements] = 0;

    int element = 0;

    int j, k, el_p;
    int *el, *cl, *tl;
    float *x_c, *y_c, *z_c;

    // room for node coordinates
    float *node_coor = 0;
    if (ndim_ * numnp_)
        node_coor = new float[ndim_ * numnp_];
    // coordinates
    int indice[3];
    indice[0] = 1;
    indice[1] = 2;
    indice[2] = 3;
    int this_state = fromIntToDSY(time);
    int one = 1;

    float zeit;

    dsywno_(&this_state, &ndim_, indice, &one, &numnp_,
            &zeit, node_coor, &rtn_);

    float *node_disp = 0;
    if (scale_ != 1.0)
    {
        node_disp = new float[ndim_ * numnp_];
        int ind_disp[3] = { 4, 5, 6 };
        dsywno_(&this_state, &ndim_, ind_disp, &one, &numnp_, &zeit,
                node_disp, &rtn_);
    }

    // First the materials
    for (cell_type = 0; cell_type < noTypes; ++cell_type)
        if (no_entities_[cell_type])
        {
            std::string material_entity_name = material_name;
            material_entity_name += "_";
            material_entity_name += Beschriftungen[cell_type];
            coDoIntArr *entityMaterial = new coDoIntArr(material_entity_name, 1,
                                                        &no_entities_[cell_type]);
            setMatList[element++] = entityMaterial;
            int *mat_addr = entityMaterial->getAddress();
            for (j = 0; j < no_entities_[cell_type]; ++j)
            {
                mat_addr[j] = entity_conn_[cell_type][connNumbers[cell_type] * j];
            }
        }

    // Now the labels
    for (cell_type = 0, element = 0; cell_type < noTypes; ++cell_type)
        if (no_entities_[cell_type])
        {
            std::string elabel_entity_name = elabel_name;
            elabel_entity_name += "_";
            elabel_entity_name += Beschriftungen[cell_type];
            coDoIntArr *entityELabel = new coDoIntArr(elabel_entity_name, 1,
                                                      &no_entities_[cell_type]);
            setElaList[element++] = entityELabel;
            int *ela_addr = entityELabel->getAddress();
            for (j = 0; j < no_entities_[cell_type]; ++j)
            {
                ela_addr[j] = entity_label[cell_type][j];
            }
        }

    // Now the grids
    // solid and other grids (but neither shells nor tools!!)
    for (cell_type = 0, element = 0; cell_type < noTypes; ++cell_type)
        if (no_entities_[cell_type] && Types[cell_type] != coStringObj::SHELL && Types[cell_type] != coStringObj::TOOL)
        {
            std::string grid_entity_name = grid_name;
            grid_entity_name += "_";
            grid_entity_name += Beschriftungen[cell_type];
            coDoUnstructuredGrid *entityGrid = new coDoUnstructuredGrid(grid_entity_name,
                                                                        no_entities_[cell_type],
                                                                        (connNumbers[cell_type] - 1) * no_entities_[cell_type],
                                                                        entity_nodes_[cell_type], 1);
            entityGrid->getAddresses(&el, &cl, &x_c, &y_c, &z_c);
            entityGrid->getTypeList(&tl);
            // connectivity
            for (j = 0; j < no_entities_[cell_type]; ++j)
            {
                el_p = (connNumbers[cell_type] - 1) * j;
                el[j] = el_p;
                for (k = 1; k < connNumbers[cell_type]; ++k)
                    cl[el_p + k - 1] = node_entity_map_[cell_type][entity_conn_[cell_type][connNumbers[cell_type] * j + k]];
                tl[j] = coviseType[cell_type];
            }
            // coordinates
            for (j = 0; j < entity_nodes_[cell_type]; ++j)
            {
                x_c[j] = node_coor[ndim_ * node_map_[node_entity_label_[cell_type][j]] + 0];
                y_c[j] = node_coor[ndim_ * node_map_[node_entity_label_[cell_type][j]] + 1];
                if (scale_ != 1.0)
                {
                    x_c[j] += (scale_ - 1.0) * node_disp[ndim_ * node_map_[node_entity_label_[cell_type][j]] + 0];
                    y_c[j] += (scale_ - 1.0) * node_disp[ndim_ * node_map_[node_entity_label_[cell_type][j]] + 1];
                }
                if (ndim_ == 3)
                {
                    z_c[j] = node_coor[ndim_ * node_map_[node_entity_label_[cell_type][j]] + 2];
                    if (scale_ != 1.0)
                    {
                        z_c[j] += (scale_ - 1.0) * node_disp[ndim_ * node_map_[node_entity_label_[cell_type][j]] + 2];
                    }
                }
                else
                {
                    z_c[j] = 0.0;
                }
            }
            setList[element++] = entityGrid;
        }
    // shell and tool grid
    int i;
    for (i = 1; i < 3; ++i)
        if (no_entities_[i])
        {
            std::string grid_shell_name = grid_name;
            grid_shell_name += "_";
            grid_shell_name += Beschriftungen[i];
            coDoUnstructuredGrid *shellGrid = 0;
            if (i == 1)
                shellGrid = new coDoUnstructuredGrid(
                    grid_shell_name,
                    no_entities_[1],
                    4 * shells4 + 3 * shells3,
                    entity_nodes_[1], 1);
            else
                shellGrid = new coDoUnstructuredGrid(
                    grid_shell_name,
                    no_entities_[2],
                    4 * tools4 + 3 * tools3,
                    entity_nodes_[2], 2);
            shellGrid->getAddresses(&el, &cl, &x_c, &y_c, &z_c);
            shellGrid->getTypeList(&tl);
            // connectivity
            for (j = 0, el_p = 0; j < no_entities_[i]; ++j)
            {
                el[j] = el_p;
                cl[el_p + 0] = node_entity_map_[i][entity_conn_[i][5 * j + 1]];
                cl[el_p + 1] = node_entity_map_[i][entity_conn_[i][5 * j + 2]];
                cl[el_p + 2] = node_entity_map_[i][entity_conn_[i][5 * j + 3]];
                if (entity_conn_[i][5 * j + 4] != 0 && entity_conn_[i][5 * j + 3] != entity_conn_[i][5 * j + 4])
                {
                    // bilinear 4-node shell element
                    cl[el_p + 3] = node_entity_map_[i][entity_conn_[i][5 * j + 4]];
                    el_p += 4;
                    tl[j] = TYPE_QUAD;
                }
                else
                {
                    // degenerate or C^0 element
                    el_p += 3;
                    tl[j] = TYPE_TRIANGLE;
                }
            }
            // coordinates
            for (j = 0; j < entity_nodes_[i]; ++j)
            {
                x_c[j] = node_coor[ndim_ * node_map_[node_entity_label_[i][j]] + 0];
                y_c[j] = node_coor[ndim_ * node_map_[node_entity_label_[i][j]] + 1];
                if (scale_ != 1.0)
                {
                    x_c[j] += (scale_ - 1.0) * node_disp[ndim_ * node_map_[node_entity_label_[i][j]] + 0];
                    y_c[j] += (scale_ - 1.0) * node_disp[ndim_ * node_map_[node_entity_label_[i][j]] + 1];
                }
                if (ndim_ == 3)
                {
                    z_c[j] = node_coor[ndim_ * node_map_[node_entity_label_[i][j]] + 2];
                    if (scale_ != 1.0)
                    {
                        z_c[j] += (scale_ - 1.0) * node_disp[ndim_ * node_map_[node_entity_label_[i][j]] + 2];
                    }
                }
                else
                {
                    z_c[j] = 0.0;
                }
            }
            setList[element++] = shellGrid;
        }

    // rehash setList so that the order is solid, shells, tools, bars ....
    int grids_to_move = (no_entities_[1] > 0) + (no_entities_[2] > 0);
    coDistributedObject *tmp[2];
    switch (grids_to_move)
    {
    case 2:
        tmp[0] = setList[element - 2];
        tmp[1] = setList[element - 1];
        break;
    case 1:
        tmp[0] = setList[element - 1];
        break;
    default:
        break;
    }
    for (i = element - grids_to_move - 1; i >= (no_entities_[0] > 1); --i)
    {
        setList[i + grids_to_move] = setList[i];
    }
    for (i = 0; i < grids_to_move; ++i)
        setList[i + (no_entities_[0] > 1)] = tmp[i];

// references
#ifdef _LOCAL_REFERENCES_
    for (cell_type = 0, element = 0; cell_type < noTypes; ++cell_type)
        if (no_entities_[cell_type])
        {
            std::string reference_entity_name = reference_name;
            reference_entity_name += "_";
            reference_entity_name += Beschriftungen[cell_type];
            coDoMat3 *entityReference = new coDoMat3(reference_entity_name,
                                                     localRef[cell_type] * no_entities_[cell_type]);
            setRefList[element] = entityReference;
            //coDoMat3::Reference *ref_addr;
            float *ref_addr;
            if (localRef[cell_type])
            {
                entityReference->getAddress(&ref_addr);
                fillReferences(cell_type, ref_addr, setList[element], node_coor);
            }
            ++element;
        }
#endif
    // REALTIME and COLOR attributes for grids
    char realtime[32];
    sprintf(realtime, "%.5e", zeit_[this_state - 1]);
    for (element = 0; setList[element]; ++element)
    {
        setList[element]->addAttribute("REALTIME", realtime);
        setList[element]->addAttribute("COLOR", "White");
    }

    coDoSet *theGrid = new coDoSet(grid_name, setList);
    *matElem = new coDoSet(material_name, setMatList);
    *elaElem = new coDoSet(elabel_name, setElaList);
#ifdef _LOCAL_REFERENCES_
    *refElem = new coDoSet(reference_name, setRefList);
#endif

    for (i = 0; i < no_elements; ++i)
    {
        delete setList[i];
        delete setMatList[i];
        delete setElaList[i];
#ifdef _LOCAL_REFERENCES_
        delete setRefList[i];
#endif
    }
    delete[] setList;
    delete[] setMatList;
    delete[] setElaList;
#ifdef _LOCAL_REFERENCES_
    delete[] setRefList;
#endif

    delete[] node_coor;
    delete[] node_disp;

    return theGrid;
}

coDoSet *ReadDSY::scalarNodal(const char *name, const char *species,
                              int this_state, float *node_vars, int offset, int num_vars)
{
    char realtime[32];
    sprintf(realtime, "%.5e", zeit_[this_state - 1]);

    int no_elements = 0;
    int cell_type;
    for (cell_type = 0; cell_type < noTypes; ++cell_type)
        no_elements += (no_entities_[cell_type] > 0);

    float *u_c;

    coDistributedObject **setList = new coDistributedObject *[no_elements + 1];
    setList[no_elements] = 0;

    int i, element = 0;

    for (cell_type = 0; cell_type < noTypes; ++cell_type)
        if (no_entities_[cell_type])
        {
            std::string grid_entity_name = name;
            grid_entity_name += "_";
            grid_entity_name += Beschriftungen[cell_type];
            coDoFloat *varInEntity = new coDoFloat(
                grid_entity_name, entity_nodes_[cell_type]);

            varInEntity->getAddress(&u_c);
            for (i = 0; i < entity_nodes_[cell_type]; ++i)
            {
                u_c[i] = node_vars[offset + num_vars * node_map_[node_entity_label_[cell_type][i]]];
            }
            setList[element] = varInEntity;
            setList[element]->addAttribute("SPECIES", species);
            setList[element]->addAttribute("REALTIME", realtime);
            ++element;
        }

    coDoSet *data = new coDoSet(name, setList);

    for (i = 0; i < no_elements; ++i)
        delete setList[i];
    delete[] setList;

    return data;
}

coDoSet *ReadDSY::vectorNodal(const char *name, const char *species,
                              int this_state, float *node_vars, int offset1, int offset2,
                              int offset3, int num_vars)
{
    char realtime[32];
    sprintf(realtime, "%.5e", zeit_[this_state - 1]);

    int no_elements = 0;
    int cell_type;
    for (cell_type = 0; cell_type < noTypes; ++cell_type)
        no_elements += (no_entities_[cell_type] > 0);
    float *u_c, *v_c, *w_c;

    coDistributedObject **setList = new coDistributedObject *[no_elements + 1];
    setList[no_elements] = 0;

    int i, element = 0;

    for (cell_type = 0; cell_type < noTypes; ++cell_type)
        if (no_entities_[cell_type])
        {
            std::string grid_entity_name(name);
            grid_entity_name += "_";
            grid_entity_name += Beschriftungen[cell_type];
            coDoVec3 *varInEntity = new coDoVec3(
                grid_entity_name, entity_nodes_[cell_type]);

            varInEntity->getAddresses(&u_c, &v_c, &w_c);
            for (i = 0; i < entity_nodes_[cell_type]; ++i)
            {
                u_c[i] = node_vars[offset1 + num_vars * node_map_[node_entity_label_[cell_type][i]]];
                v_c[i] = node_vars[offset2 + num_vars * node_map_[node_entity_label_[cell_type][i]]];
                w_c[i] = node_vars[offset3 + num_vars * node_map_[node_entity_label_[cell_type][i]]];
            }
            setList[element] = varInEntity;
            std::string co_species(species);
            if (co_species.find(" (VEC)", co_species.length() - 6) != std::string::npos)
                co_species.resize(co_species.length() - 6);
            setList[element]->addAttribute("SPECIES", co_species.c_str());
            setList[element]->addAttribute("REALTIME", realtime);
            ++element;
        }

    coDoSet *data = new coDoSet(name, setList);

    for (i = 0; i < no_elements; ++i)
        delete setList[i];
    delete[] setList;

    return data;
}

void ReadDSY::ReadCellVar(int cell_type, int *nstate, int *numvar,
                          int *indice, int *ifirst, int *ilast, float *zeit, float *varval)
{
    switch (cell_type)
    {
    case 0: // solid
        dsywbr_(nstate, numvar, indice, ifirst, ilast, zeit, varval, &rtn_);
        break;
    case 1: // shells
        dsywsh_(nstate, numvar, indice, ifirst, ilast, zeit, varval, &rtn_);
        break;
    case 2: // tools
        dsywto_(nstate, numvar, indice, ifirst, ilast, zeit, varval, &rtn_);
        break;
#ifdef _INCLUDE_SPH_
    case 13: // SPH
        dsywsp_(nstate, numvar, indice, ifirst, ilast, zeit, varval, &rtn_);
        break;
#endif
    case 3: // bars
    {
        float *oneD_varval = new float[(*numvar) * no_1D_entities_];
        int one = 1;
        int last = no_1D_entities_;
        dsywbe_(nstate, numvar, indice, &one, &last, zeit, oneD_varval, &rtn_);
        int count1D, j;
        for (count1D = 0, j = 0; count1D < no_1D_entities_; ++count1D)
        {
            if (oneD_conn_[6 * count1D + 4] == 1 && oneD_conn_[6 * count1D + 5] <= 204)
            {
                if (*ifirst <= j + 1 && *ilast >= j + 1)
                    memcpy(varval + (*numvar) * j, oneD_varval + (*numvar) * count1D,
                           sizeof(float) * (*numvar));
                ++j;
                if (j == *ilast - *ifirst + 1)
                    break;
            }
        }
        delete[] oneD_varval;
    }
    break;
    case 5: // bar1
    {
        float *oneD_varval = new float[(*numvar) * no_1D_entities_];
        int one = 1;
        int last = no_1D_entities_;
        dsywbe_(nstate, numvar, indice, &one, &last, zeit, oneD_varval, &rtn_);
        int count1D, j;
        for (count1D = 0, j = 0; count1D < no_1D_entities_; ++count1D)
        {
            if (oneD_conn_[6 * count1D + 4] == 1 && oneD_conn_[6 * count1D + 5] == 205)
            {
                if (*ifirst <= j + 1 && *ilast >= j + 1)
                    memcpy(varval + (*numvar) * j, oneD_varval + (*numvar) * count1D,
                           sizeof(float) * (*numvar));
                ++j;
                if (j == *ilast - *ifirst + 1)
                    break;
            }
        }
        delete[] oneD_varval;
    }
    break;
    case 4: // beams
    {
        float *oneD_varval = new float[(*numvar) * no_1D_entities_];
        int one = 1;
        int last = no_1D_entities_;
        dsywbe_(nstate, numvar, indice, &one, &last, zeit, oneD_varval, &rtn_);
        int count1D, j;
        for (count1D = 0, j = 0; count1D < no_1D_entities_; ++count1D)
        {
            if (oneD_conn_[6 * count1D + 4] == 2)
            {
                if (*ifirst <= j + 1 && *ilast >= j + 1)
                    memcpy(varval + (*numvar) * j, oneD_varval + (*numvar) * count1D,
                           sizeof(float) * (*numvar));
                ++j;
                if (j == *ilast - *ifirst + 1)
                    break;
            }
        }
        delete[] oneD_varval;
    }
    break;
    case 6: // SPRING
    case 7: // SPH_JOINT
    case 8: // FLX_TOR_JOINT
    case 9: // SPOTWELD
    case 10: // JET
    case 11: // KINE_JOINT
    case 12: // MESI_SPOTWELD
    {
        float *oneD_varval = new float[(*numvar) * no_1D_entities_];
        int one = 1;
        int last = no_1D_entities_;
        dsywbe_(nstate, numvar, indice, &one, &last, zeit, oneD_varval, &rtn_);
        int count1D, j;
        for (count1D = 0, j = 0; count1D < no_1D_entities_; ++count1D)
        {
            if (oneD_conn_[6 * count1D + 4] == cell_type - 3)
            {
                if (*ifirst <= j + 1 && *ilast >= j + 1)
                    memcpy(varval + (*numvar) * j, oneD_varval + (*numvar) * count1D,
                           sizeof(float) * (*numvar));
                ++j;
                if (j == *ilast - *ifirst + 1)
                    break;
            }
        }
        delete[] oneD_varval;
    }
    break;
    }
}

// finds a title of 4 letters (as used for 1D elements)
int ReadDSY::FindTitle(const char *tit, int no_titles, const char *titles)
{
    int i;
    for (i = 0; i < no_titles; ++i)
    {
        if (strncmp(tit, titles + 32 * i + 24, 4) == 0)
            return i;
    }
    return -1;
}

// lumps vector components into a full vector object
// given the titles of the three components
void ReadDSY::Lump(const char *tit1, const char *tit2, const char *tit3,
                   const char *titv, int no_titles, char *titles)
{
    int wo1, wo2, wo3;
    wo1 = FindTitle(tit1, no_titles, titles);
    wo2 = FindTitle(tit2, no_titles, titles);
    wo3 = FindTitle(tit3, no_titles, titles);
    if (wo1 >= 0 && wo2 >= 0 && wo3 >= 0)
    {
        // we have a full vector
        strncpy(titles + 32 * wo1, titv, 24);
        //      memset(titles+32*wo1+24,' ',6);
        titles[32 * wo1 + 30] = '2';
        titles[32 * wo1 + 31] = '1';
        strncpy(titles + 32 * wo2, titv, 24);
        //      memset(titles+32*wo2+24,' ',6);
        titles[32 * wo2 + 30] = '2';
        titles[32 * wo2 + 31] = '2';
        strncpy(titles + 32 * wo3, titv, 24);
        //      memset(titles+32*wo3+24,' ',6);
        titles[32 * wo3 + 30] = '2';
        titles[32 * wo3 + 31] = '3';
    }
}

void ReadDSY::FindVectors(coStringObj::ElemType etype, int no_titles, char *titles)
{
    switch (etype)
    {
    case coStringObj::SPRING:
        Lump("FAXI", "FSSH", "FTSH", "Force (spring lVEC)", no_titles, titles);
        Lump("MTOR", "MSN1", "MTN1", "Moment (spring lVEC)", no_titles, titles);
        Lump("DAXI", "RTOR", "RSN1", "Displ. (spring lVEC)", no_titles, titles);
        Lump("RTN1", "RSN2", "RTN2", "Theta (spring lVEC)", no_titles, titles);
        break;
    case coStringObj::SPH_JOINT:
        Lump("FAXI", "FSSH", "FTSH", "Force (sphJ lVEC)", no_titles, titles);
        Lump("DAXI", "RTOR", "RSN1", "Displ. (sphJ lVEC)", no_titles, titles);
        break;
    case coStringObj::FLX_TOR_JOINT:
        Lump("FAXI", "FSSH", "FTSH", "Force (ftJ lVEC)", no_titles, titles);
        Lump("DAXI", "RTOR", "RSN1", "R. disp. (ftJ lVEC)", no_titles, titles);
        break;
    case coStringObj::KINE_JOINT:
        Lump("FAXI", "FSSH", "FTSH", "Force (kJ lVEC)", no_titles, titles);
        Lump("DAXI", "RTOR", "RSN1", "R. disp. (kJ lVEC)", no_titles, titles);
        break;
    case coStringObj::MESI_SPOTWELD:
        Lump("FTSH", "MTOR", "MSN1", "Force (MIS gVEC)", no_titles, titles);
        break;
    default:
        break;
    }
}

// return 0 if variable is meaningfull, -1 otherwise
int ReadDSY::Translations(coStringObj::ElemType etype, char *title)
{
    // AQUI
    const int BarMag = 2;
    const char *BarTranslations[] = {
        "FAXI", "Axial force (bar)",
        "DAXI", "Axial elong. (bar)"
    };
    const int Bar1Mag = 3;
    const char *Bar1Translations[] = {
        "FAXI", "Axial force (bar)",
        "DAXI", "Axial elong. (bar)",
        "RTN1", "Length gain (bar1)"
    };
    const int SpringMag = 12;
    const char *SpringTanslations[] = {
        "FAXI", "F_R (spring)",
        "FSSH", "F_S (spring)", "FTSH", "F_T (spring)", "MTOR", "M_R (spring)",
        "MSN1", "M_S (spring)", "MTN1", "M_T (spring)", "DAXI", "D_R (spring)",
        "RTOR", "D_S (spring)", "RSN1", "D_T (spring)", "RTN1", "ANG_R (spring)",
        "RSN2", "ANG_S (spring)", "RTN2", "ANG_T (spring)"
    };

    const int SphJMag = 12;
    const char *SphJTanslations[] = {
        "FAXI", "F_R (sphJ)",
        "FSSH", "F_S (sphJ)", "FTSH", "F_T (sphJ)", "MTOR", "M_R (sphJ)",
        "MSN1", "M_S (sphJ)", "MTN1", "M_T (sphJ)", "DAXI", "D_R (sphJ)",
        "RTOR", "D_S (sphJ)", "RSN1", "D_T (sphJ)", "RTN1", "PHI (sphJ)",
        "RSN2", "THETA (sphJ)", "RTN2", "PSI (sphJ)"
    };

    const int FTJMag = 14;
    const char *FTJTanslations[] = {
        "FAXI", "F_R (ftJ)",
        "FSSH", "F_S (ftJ)", "FTSH", "F_T (ftJ)", "MTOR", "FlxMom. (ftJ)",
        "MSN1", "FlxDep. (ftJ)", "MTN1", "TorMom. (ftJ)",
        "MSN2", "DampMom. (ftJ)", "MTN2", "FricMom. (ftJ)", "DAXI", "D_R (ftJ)",
        "RTOR", "D_S (ftJ)", "RSN1", "D_T (ftJ)", "RTN1", "ALPHA (ftJ)",
        "RSN2", "GAMMA (ftJ)", "RTN2", "BETA (ftJ)"
    };

    const int SptWMag = 2;
    const char *SptWTranslations[] = {
        "FAXI", "Tens. force (sptW)",
        "FSSH", "Tang. force (sptW)"
    };

    const int JetMag = 6;
    const char *JetTranslations[] = {
        "FAXI", "Infl. pressure",
        "FSSH", "Exh. pressure",
        "FTSH", "Jet velocity",
        "MTOR", "Jet length",
        "MSN1", "Jet force",
        "MTN1", "Max. imp. veloc."
    };

    const int KJMag = 12;
    const char *KJTranslations[] = {
        "FAXI", "F_R (kJ)",
        "FSSH", "F_S (kJ)",
        "FTSH", "F_T (kJ)",
        "MTOR", "Flex. Mom. (kJ)",
        "MSN1", "Flex. Dep. (kJ)",
        "MTN1", "Tor. Mom. (kJ)",
        "DAXI", "D_R (kJ)",
        "RTOR", "D_S (kJ)",
        "RSN1", "D_T (kJ)",
        "RTN1", "R Ang. (kJ)",
        "RSN2", "S' Ang. (kJ)",
        "RTN2", "T'' Ang. (kJ)"
    };

    const int MISMag = 9;
    const char *MISTranslations[] = {
        "FAXI", "Normal force (MIS)",
        "FSSH", "Shear force (MIS)",
        "FTSH", "F_X (MIS)",
        "MTOR", "F_Y (MIS)",
        "MSN1", "F_Z (MIS)",
        "DAXI", "Elem. length (MIS)",
        "RTOR", "Contact energy (MIS)",
        "RSN1", "MaxAbsContEnergy (MIS)",
        "RTN1", "Rupt. criterion (MIS)",
    };

    int res = -1;
    int noMag;
    const char **translations;

    switch (etype)
    {
    case coStringObj::SOLID:
    case coStringObj::SHELL:
    case coStringObj::TOOL:
    case coStringObj::BEAM:
#ifdef _INCLUDE_SPH_
    case coStringObj::SPH:
#endif
        res = 0;
        break;
    case coStringObj::BAR:
        noMag = BarMag;
        translations = BarTranslations;
        break;
    case coStringObj::BAR1:
        noMag = Bar1Mag;
        translations = Bar1Translations;
        break;
    case coStringObj::SPRING:
        noMag = SpringMag;
        translations = SpringTanslations;
        break;
    case coStringObj::SPH_JOINT:
        noMag = SphJMag;
        translations = SphJTanslations;
        break;
    case coStringObj::FLX_TOR_JOINT:
        noMag = FTJMag;
        translations = FTJTanslations;
        break;
    case coStringObj::SPOTWELD:
        noMag = SptWMag;
        translations = SptWTranslations;
        break;
    case coStringObj::JET:
        noMag = JetMag;
        translations = JetTranslations;
        break;
    case coStringObj::KINE_JOINT:
        noMag = KJMag;
        translations = KJTranslations;
        break;
    case coStringObj::MESI_SPOTWELD:
        noMag = MISMag;
        translations = MISTranslations;
        break;
    default:
        res = 0;
        break;
    }
    if (res == 0)
        return 0;
    int i;
    for (i = 0; i < noMag; ++i)
    {
        if (strncmp(title + 24, translations[2 * i], 4) == 0)
        {
            if (title[30] == ' ' || title[30] == '1')
                strncpy(title, translations[2 * i + 1], 24);
            return 0;
        }
    }
    return -1;
}

void ReadDSY::setScale(float scal)
{
    int i;
    int one = 1;
    int ind[3] = { 4, 5, 6 };
    float zeit;
    float varval[3];
    int numvar = 3;
    for (i = 1; i <= nstate_; ++i)
    {
        // check only for the first node
        dsywno_(&i, &numvar, ind, &one, &one, &zeit, varval, &rtn_);
        if (rtn_)
        {
            scale_ = 1.0;
            useDisplacements_ = 0;
            return;
        }
    }
    scale_ = scal;
    useDisplacements_ = 1;
}

const int X = 0;
const int Y = 1;
const int Z = 2;
const int XX = 0;
const int XY = 1;
const int XZ = 2;
const int YX = 3;
const int YY = 4;
const int YZ = 5;
const int ZX = 6;
const int ZY = 7;
const int ZZ = 8;

void setIdentity(float *matrix)
{
    matrix[XX] = 1.0;
    matrix[XY] = 0.0;
    matrix[XZ] = 0.0;
    matrix[YX] = 0.0;
    matrix[YY] = 1.0;
    matrix[YZ] = 0.0;
    matrix[ZX] = 0.0;
    matrix[ZY] = 0.0;
    matrix[ZZ] = 1.0;
}

void writeRef(float *ref_addr,
              int *conn, float *x_c, float *y_c, float *z_c)
{
    float vector[3];
    float len;
    // first column
    vector[X] = x_c[conn[1]] - x_c[conn[0]];
    vector[Y] = y_c[conn[1]] - y_c[conn[0]];
    vector[Z] = z_c[conn[1]] - z_c[conn[0]];
    len = sqrt(vector[X] * vector[X] + vector[Y] * vector[Y] + vector[Z] * vector[Z]);
    if (len != 0.0)
    {
        ref_addr[XX] = vector[X] / len;
        ref_addr[YX] = vector[Y] / len;
        ref_addr[ZX] = vector[Z] / len;
    }
    else
    {
        setIdentity(ref_addr);
        return;
    }
    // second column
    vector[X] = x_c[conn[2]] - x_c[conn[0]];
    vector[Y] = y_c[conn[2]] - y_c[conn[0]];
    vector[Z] = z_c[conn[2]] - z_c[conn[0]];
    len = vector[X] * ref_addr[XX];
    len += vector[Y] * ref_addr[YX];
    len += vector[Z] * ref_addr[ZX];
    vector[X] -= len * ref_addr[XX];
    vector[Y] -= len * ref_addr[YX];
    vector[Z] -= len * ref_addr[ZX];
    len = sqrt(vector[X] * vector[X] + vector[Y] * vector[Y] + vector[Z] * vector[Z]);
    if (len != 0.0)
    {
        ref_addr[XY] = vector[X] / len;
        ref_addr[YY] = vector[Y] / len;
        ref_addr[ZY] = vector[Z] / len;
    }
    else
    {
        setIdentity(ref_addr);
        return;
    }
    // third column
    ref_addr[XZ] = ref_addr[YX] * ref_addr[ZY] - ref_addr[ZX] * ref_addr[YY];
    ref_addr[YZ] = ref_addr[ZX] * ref_addr[XY] - ref_addr[XX] * ref_addr[ZY];
    ref_addr[ZZ] = ref_addr[XX] * ref_addr[YY] - ref_addr[YX] * ref_addr[XY];
}

void ReadDSY::fillReferences(int cell_type, float *ref_addr,
                             coDistributedObject *grid_obj, const float *node_coor)
{
    coDoUnstructuredGrid *grid = (coDoUnstructuredGrid *)(grid_obj);
    int *el, *cl;
    int cl1D[3] = { 0, 1, 2 };
    float x_c1D[3];
    float y_c1D[3];
    float z_c1D[3] = { 0.0, 0.0, 0.0 };
    int e, c, p;
    int i, j;
    int label1, label2; // labels for nodes that are not in the grid
    int global1, global2; // labels for nodes that are not in the grid
    float *x_c, *y_c, *z_c;

    grid->getGridSize(&e, &c, &p);
    grid->getAddresses(&el, &cl, &x_c, &y_c, &z_c);

    switch (Types[cell_type])
    {
    case coStringObj::SHELL:
        for (i = 0; i < e; ++i)
        {
            writeRef(ref_addr + 9 * i, cl + el[i], x_c, y_c, z_c);
        }
        break;
    case coStringObj::BEAM: // beams springs and joints
        for (i = 0, j = 0; i < no_1D_entities_; ++i)
        {
            if (oneD_conn_[6 * i + 4] == 2)
            {
                // j is the partial (for this entity) number, i the 1D-ordering number
                x_c1D[0] = x_c[cl[el[j]]];
                y_c1D[0] = y_c[cl[el[j]]];
                z_c1D[0] = z_c[cl[el[j]]];
                x_c1D[1] = x_c[cl[el[j] + 1]];
                y_c1D[1] = y_c[cl[el[j] + 1]];
                z_c1D[1] = z_c[cl[el[j] + 1]];
                label1 = oneD_conn_[6 * i + 2];
                global1 = node_map_[label1];
                x_c1D[2] = node_coor[ndim_ * global1];
                y_c1D[2] = node_coor[ndim_ * global1 + 1];
                if (ndim_ == 3)
                    z_c1D[2] = node_coor[ndim_ * global1 + 2];
                writeRef(ref_addr + 9 * j, cl1D, x_c1D, y_c1D, z_c1D);
                ++j;
            }
        }
        break;
    case coStringObj::SPRING:
    case coStringObj::SPH_JOINT:
    case coStringObj::FLX_TOR_JOINT:
        for (i = 0, j = 0; i < no_1D_entities_; ++i)
        {
            if (oneD_conn_[6 * i + 4] == Types[cell_type] - 100)
            {
                x_c1D[0] = x_c[cl[el[j]]];
                y_c1D[0] = y_c[cl[el[j]]];
                z_c1D[0] = z_c[cl[el[j]]];
                label1 = oneD_conn_[6 * i + 2];
                global1 = node_map_[label1];
                label2 = oneD_conn_[6 * i + 3];
                global2 = node_map_[label2];
                x_c1D[1] = node_coor[ndim_ * global1];
                y_c1D[1] = node_coor[ndim_ * global1 + 1];
                x_c1D[2] = node_coor[ndim_ * global2];
                y_c1D[2] = node_coor[ndim_ * global2 + 1];
                if (ndim_ == 3)
                {
                    z_c1D[1] = node_coor[ndim_ * global1 + 2];
                    z_c1D[2] = node_coor[ndim_ * global2 + 2];
                }
                writeRef(ref_addr + 9 * j, cl1D, x_c1D, y_c1D, z_c1D);
                ++j;
            }
        }
        break;
    default:
        break;
    }
}
