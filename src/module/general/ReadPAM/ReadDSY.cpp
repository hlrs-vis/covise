/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#define COIDENT "$Header: /vobs/covise/src/application/general/READ_PAM/ReadDSY.cpp /main/vir_main/2 18-Dec-2001.11:08:15 we_te $"
#include <util/coIdent.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>

// ReadDSY.cpp conatins mainly interface functions

#include "ReadDSY.h"
#include <config/CoviseConfig.h>

ReadDSY::ReadDSY()
    : unit(1)
    , theSame_(0)
{
    isvel_ = isacc_ = isadd_ = isdisp_ = 0;
    int i;
    for (i = 0; i < noTypes; ++i)
        node_entity_label_[i] = 0;
    oneD_conn_ = oneD_label_ = 0;

    CELL_REDUCE = coCoviseConfig::getInt("Module.ReadPAM.CellVarLimit", INT_MAX);
    GLOBAL_REDUCE = coCoviseConfig::getInt("Module.ReadPAM.GlobalVarLimit", INT_MAX);
    MAT_GLOBAL_REDUCE = coCoviseConfig::getInt("Module.ReadPAM.MatGlobalVarLimit", INT_MAX);
    TS_GLOBAL_REDUCE = coCoviseConfig::getInt("Module.ReadPAM.TsGlobalVarLimit", INT_MAX);
    CI_GLOBAL_REDUCE = coCoviseConfig::getInt("Module.ReadPAM.CiGlobalVarLimit", INT_MAX);
    RW_GLOBAL_REDUCE = coCoviseConfig::getInt("Module.ReadPAM.RwGlobalVarLimit", INT_MAX);
    AB_GLOBAL_REDUCE = coCoviseConfig::getInt("Module.ReadPAM.AbGlobalVarLimit", INT_MAX);
    ABCH_GLOBAL_REDUCE = coCoviseConfig::getInt("Module.ReadPAM.AbChGlobalVarLimit", INT_MAX);
    ABW_GLOBAL_REDUCE = coCoviseConfig::getInt("Module.ReadPAM.AbWGlobalVarLimit", INT_MAX);
}

void mydsytit(int *type, int *no_titles, char *titles, int titles_len)
{
    // AQUI
    int typ1D = 5;
    *no_titles = 0;

    switch (*type)
    {
    case coStringObj::SOLID:
    case coStringObj::SHELL:
    case coStringObj::NODAL_ADDI:
    case coStringObj::GLOBAL:
    case coStringObj::BAR:
    case coStringObj::BEAM:
#ifdef _INCLUDE_SPH_
    case coStringObj::SPH:
#endif
        dsytit_(type, no_titles, titles, titles_len);
        break;
    case coStringObj::BAR1:
    case coStringObj::SPRING:
    case coStringObj::SPH_JOINT:
    case coStringObj::FLX_TOR_JOINT:
    case coStringObj::SPOTWELD:
    case coStringObj::JET:
    case coStringObj::KINE_JOINT:
    case coStringObj::MESI_SPOTWELD:
        dsytit_(&typ1D, no_titles, titles, titles_len);
    }

    /*
      if(*type == coStringObj::SOLID || *type == coStringObj::SHELL || *type == coStringObj::NODAL_ADDI || *type == coStringObj::GLOBAL || *type == coStringObj::BAR || *type == coStringObj::BEAM){
         dsytit_(type,no_titles,titles);
      } else if( *type == coStringObj::BAR1 || *type == coStringObj::SPRING ||
          *type == coStringObj::SPH_JOINT || *type == coStringObj::FLX_TOR_JOINT ||
          *type == coStringObj::SPOTWELD || *type == coStringObj::JET ||
          *type == coStringObj::KINE_JOINT || *type == coStringObj::MESI_SPOTWELD){
         int typ1D=5;
         dsytit_(&typ1D,no_titles,titles);
      }
   */
}

void ReadDSY::mydsyhal(int *type, int *nall, int *iall, int *rtn)
{
    int oneDtype = 5;
    switch (*type)
    {
    case coStringObj::SOLID:
    case coStringObj::SHELL:
    case coStringObj::TOOL:
    case coStringObj::NODAL_ADDI:
#ifdef _INCLUDE_SPH_
    case coStringObj::SPH:
#endif
        dsyhal_(type, nall, iall, rtn);
        break;
    case coStringObj::BAR:
        dsyhal_(type, nall, iall, rtn);
        *nall = no_entities_[3];
        if (no_entities_[3] == 0)
            *iall = 0;
        break;
    case coStringObj::BAR1:
        dsyhal_(type, nall, iall, rtn);
        *nall = no_entities_[5];
        if (no_entities_[5] == 0)
            *iall = 0;
        break;
    case coStringObj::BEAM:
        dsyhal_(type, nall, iall, rtn);
        *nall = no_entities_[4];
        if (no_entities_[4] == 0)
            *iall = 0;
        break;
    case coStringObj::SPRING:
    case coStringObj::SPH_JOINT:
    case coStringObj::FLX_TOR_JOINT:
    case coStringObj::SPOTWELD:
    case coStringObj::JET:
    case coStringObj::KINE_JOINT:
    case coStringObj::MESI_SPOTWELD:
        dsyhal_(&oneDtype, nall, iall, rtn);
        *nall = no_entities_[*type - 100 + 3];
        if (*nall == 0)
            *iall = 0;
        break;
    default:
        break;
        *nall = 0;
        *iall = 0;
        *rtn = 28;
    }
}

coDistributedObject **ReadDSY::globalObj(int no_objects,
                                         std::string *objName, int *req_label)
{
    int numvar;
    int one = 1;
    float *varval;
    if (no_objects == 0)
        return 0;
    coDistributedObject **theObjects = new coDistributedObject *[no_objects];
    int i, j;

    dsyopn_(path_glo_.c_str(), &rtn_, path_glo_.length());
    if (rtn_ != 0)
    {
        Covise::sendError("Error when opening DSY or THP file");
        return 0;
    }

    int no_states;
    dsysta_(&no_states);

    for (i = 0; i < no_objects; ++i)
    {
        int matOrder = global_contents_[req_label[i]].position_;
        switch (global_contents_[req_label[i]].ObjType_)
        {
        case coStringObj::SCALAR:
            numvar = 1;
            theObjects[i] = new coDoFloat(objName[i], no_states);
            theObjects[i]->addAttribute("SPECIES",
                                        global_contents_[req_label[i]].ObjName_.c_str());
            ((coDoFloat *)(theObjects[i]))->getAddress(&varval);
            switch (global_contents_[req_label[i]].ElemType_)
            {
            case coStringObj::GLOBAL:
                if (global_contents_[req_label[i]].ObjName_ != "Time")
                    dsyhgl_(&one, &no_states, &numvar,
                            global_contents_[req_label[i]].index_, varval, &rtn_);
                else
                    dsytim_(&no_states, varval);
                break;
            // special globals
            case coStringObj::MATERIAL:
                dsyhma_(&one, &no_states, &one, &matOrder, &matOrder, &one,
                        global_contents_[req_label[i]].index_, varval, &rtn_);
                break;
            case coStringObj::TRANS_SECTION:
                dsyhse_(&one, &no_states, &one, &matOrder, &matOrder, &one,
                        global_contents_[req_label[i]].index_, varval, &rtn_);
                break;
            case coStringObj::CONTACT_INTERFACE:
                dsyhct_(&one, &no_states, &one, &matOrder, &matOrder, &one,
                        global_contents_[req_label[i]].index_, varval, &rtn_);
                break;
            case coStringObj::RIGID_WALL:
                dsyhrw_(&one, &no_states, &one, &matOrder, &matOrder, &one,
                        global_contents_[req_label[i]].index_, varval, &rtn_);
                break;
            case coStringObj::AIRBAG:
                dsyhba_(&one, &no_states, &one, &matOrder, &matOrder, &one,
                        global_contents_[req_label[i]].index_, varval, &rtn_);
                break;
            case coStringObj::AIRBAG_CHAM:
                dsyhch_(&one, &no_states, &one, &matOrder, &matOrder, &one,
                        global_contents_[req_label[i]].index_, varval, &rtn_);
                break;
            case coStringObj::AIRBAG_WALL:
                dsyhwa_(&one, &no_states, &one, &matOrder, &matOrder, &one,
                        global_contents_[req_label[i]].index_, varval, &rtn_);
                break;
            default:
                Covise::sendError("ReadDSY::globalObj: Error: Unknown "
                                  "element-type.  Ignored.");
            }
            break;
        case coStringObj::VECTOR:
            numvar = 3;
            float *u, *v, *w;
            varval = new float[no_states * numvar];
            dsyhgl_(&one, &no_states, &numvar, global_contents_[req_label[i]].index_,
                    varval, &rtn_);
            theObjects[i] = new coDoVec3(objName[i], no_states);
            ((coDoVec3 *)(theObjects[i]))->getAddresses(&u, &v, &w);
            for (j = 0; j < no_states; ++j)
            {
                u[j] = varval[3 * j];
                v[j] = varval[3 * j + 1];
                w[j] = varval[3 * j + 2];
            }
            delete[] varval;
            break;
        default:
            break;
        }
    }
    dsyclo_();
    return theObjects;
}

// objName: names of objects attached to ports
coDoSet **ReadDSY::cellObj(int no_objects, std::string *objName, int *req_label)
{
    if (no_objects == 0)
        return 0;
    coDoSet **theObjects = new coDoSet *[no_objects];
    int i, time;

    dsyopn_(path_.c_str(), &lrec, path_.length());

    if (rtn_ != 0)
    {
        Covise::sendError("Error when opening DSY file");
        return 0;
    }

    char timeAttr[16];
    sprintf(timeAttr, "1 %d", noTimeReq_);

    for (i = 0; i < no_objects; ++i)
    {
        coDistributedObject **setList = new coDistributedObject *[noTimeReq_ + 1];
        setList[noTimeReq_] = 0;
        for (time = 0; time < noTimeReq_; ++time)
            setList[time] = cellObjAtTime(objName[i], time, req_label[i]);
        theObjects[i] = new coDoSet(objName[i], setList);
        theObjects[i]->addAttribute("TIMESTEP", timeAttr);
        for (time = 0; time < noTimeReq_; ++time)
            delete setList[time];
        delete[] setList;
    }
    dsyclo_();
    return theObjects;
}

coDoSet **ReadDSY::tensorObj(const TensDescriptions &tdesc)
{
    if (tdesc.no_request == 0)
        return 0;
    coDoSet **theObjects = new coDoSet *[tdesc.no_request];
    int i, time;

    dsyopn_(path_.c_str(), &rtn_, path_.length());

    if (rtn_ != 0)
    {
        Covise::sendError("Error when opening DSY file");
        return 0;
    }

    char timeAttr[16];
    sprintf(timeAttr, "1 %d", noTimeReq_);

    for (i = 0; i < tdesc.no_request; ++i)
    {
        coDistributedObject **setList = new coDistributedObject *[noTimeReq_ + 1];
        setList[noTimeReq_] = 0;
        for (time = 0; time < noTimeReq_; ++time)
            setList[time] = tensorObjAtTime(tdesc, time, i);
        theObjects[i] = new coDoSet(tdesc.requests[i], setList);
        theObjects[i]->addAttribute("TIMESTEP", timeAttr);
        for (time = 0; time < noTimeReq_; ++time)
            delete setList[time];
        delete[] setList;
    }
    dsyclo_();
    return theObjects;
}

coDoSet **ReadDSY::nodalObj(int no_objects, std::string *objName, int *req_label)
{
    // prepare naming
    int num_vars = 0;
    int i, j;
    std::string *obj_name;

    if (no_objects == 0)
        return 0;

    obj_name = new std::string[no_objects];
    coDistributedObject ***setList = new coDistributedObject **[no_objects];
    for (i = 0; i < no_objects; ++i)
    {
        // lists for sets
        setList[i] = new coDistributedObject *[noTimeReq_ + 1];
        setList[i][noTimeReq_] = 0;
        // work out num_vars
        num_vars += contents_[req_label[i]].ObjType_;
    }

    float *node_vars = new float[num_vars * numnp_];
    int *index = new int[num_vars];
    int offset = 0;
    for (i = 0; i < no_objects; ++i)
    {
        switch (contents_[req_label[i]].ObjType_)
        {
        case coStringObj::SCALAR:
            index[offset++] = contents_[req_label[i]].index_[0];
            break;
        case coStringObj::VECTOR:
            index[offset++] = contents_[req_label[i]].index_[0];
            index[offset++] = contents_[req_label[i]].index_[1];
            // if(ndim_ == 3)
            index[offset++] = contents_[req_label[i]].index_[2];
            break;
        case coStringObj::TENSOR_2D:
            index[offset++] = contents_[req_label[i]].index_[0];
            index[offset++] = contents_[req_label[i]].index_[1];
            index[offset++] = contents_[req_label[i]].index_[2];
            index[offset++] = contents_[req_label[i]].index_[3];
            break;
        case coStringObj::TENSOR_3D:
            index[offset++] = contents_[req_label[i]].index_[0];
            index[offset++] = contents_[req_label[i]].index_[1];
            index[offset++] = contents_[req_label[i]].index_[2];
            index[offset++] = contents_[req_label[i]].index_[3];
            index[offset++] = contents_[req_label[i]].index_[4];
            index[offset++] = contents_[req_label[i]].index_[5];
            index[offset++] = contents_[req_label[i]].index_[6];
            index[offset++] = contents_[req_label[i]].index_[7];
            index[offset++] = contents_[req_label[i]].index_[8];
            break;
        default:
            Covise::sendError("ReadDSY::nodalObj: Error: unknown object "
                              "type.");
            return 0;
        }
    }
    Map1D mapIndices(num_vars, index);

    dsyopn_(path_.c_str(), &rtn_, path_.length());

    if (rtn_ != 0)
    {
        Covise::sendError("Error when opening DSY file");
        return 0;
    }

    // nstate_ ...
    for (i = 0; i < noTimeReq_; ++i)
    {
        // create set list elements

        // names
        char obj_name_app[32];
        sprintf(obj_name_app, "_%d", i);
        for (j = 0; j < no_objects; ++j)
        {
            obj_name[j] = objName[j];
            obj_name[j] += obj_name_app;
        }

        // read variables
        int this_state = fromIntToDSY(i);
        int one = 1;

        float zeit;

        dsywno_(&this_state, &num_vars, index, &one, &numnp_,
                &zeit, node_vars, &rtn_);

        // write variables
        int offset1;
        int offset2;
        int offset3;
        char species[25];
        for (j = 0; j < no_objects; ++j)
        {
            stripSpecies(species, contents_[req_label[j]].ObjName_);
            switch (contents_[req_label[j]].ObjType_)
            {
            case coStringObj::SCALAR:
                offset = mapIndices[contents_[req_label[j]].index_[0]];
                setList[j][i] = scalarNodal(obj_name[j].c_str(), species, this_state,
                                            node_vars, offset, num_vars);
                break;
            case coStringObj::VECTOR:
                offset1 = mapIndices[contents_[req_label[j]].index_[0]];
                offset2 = mapIndices[contents_[req_label[j]].index_[1]];
                offset3 = mapIndices[contents_[req_label[j]].index_[2]];
                setList[j][i] = vectorNodal(obj_name[j].c_str(), species, this_state,
                                            node_vars, offset1,
                                            offset2, offset3, num_vars);
                break;
            case coStringObj::TENSOR_2D:
            case coStringObj::TENSOR_3D:
                // this may not be reached unless aux.h is modified
                // to take tensors into account...
                Covise::sendWarning("Tensors not yet implemented");
                break;
            default:
                Covise::sendError("ReadDSY::nodalObj: Error: unknown object "
                                  "type.");
                return 0;
            }
        } // end loop over objects
    } // end loop over states

    dsyclo_();

    // now make the sets with time steps
    coDoSet **theObjects = new coDoSet *[no_objects];
    char timeAttr[16];
    sprintf(timeAttr, "1 %d", noTimeReq_);
    for (i = 0; i < no_objects; ++i)
    {
        //      obj_name[i] = objName[i];
        theObjects[i] = new coDoSet(objName[i], setList[i]);
        theObjects[i]->addAttribute("TIMESTEP", timeAttr);
        for (j = 0; j < noTimeReq_; ++j)
            delete setList[i][j];
        delete[] setList[i];
    }
    delete[] setList;

    delete[] obj_name;
    delete[] node_vars;
    delete[] index;

    return theObjects;
}

// returns the grid in the form of time steps
coDoSet *ReadDSY::grid(const char *objName, const char *matName,
                       const char *elaName,
#ifdef _LOCAL_REFERENCES_
                       const char *refName,
#endif
                       float scal)
{

    dsyopn_(path_.c_str(), &rtn_, path_.length());

    if (rtn_ != 0)
    {
        Covise::sendError("Error when opening DSY file");
        return 0;
    }

    // read number of states
    dsysta_(&nstate_);
    zeit_ = new float[nstate_];
    dsytim_(&nstate_, zeit_);

    // set scale if displacements are available
    setScale(scal);
    coDistributedObject **setList = new coDistributedObject *[noTimeReq_ + 1];
    setList[noTimeReq_] = 0;
    // we prepare also lists for materials
    coDistributedObject **setMatList = new coDistributedObject *[noTimeReq_ + 1];
    setMatList[noTimeReq_] = 0;
    // we prepare also lists for labels
    coDistributedObject **setElaList = new coDistributedObject *[noTimeReq_ + 1];
    setElaList[noTimeReq_] = 0;
// we prepare also lists for references
#ifdef _LOCAL_REFERENCES_
    coDistributedObject **setRefList = new coDistributedObject *[noTimeReq_ + 1];
    setRefList[noTimeReq_] = 0;
#endif

    // prepare naming
    int i, j;
    std::string grid_name;
    int cell_type;

    //   int *entity_label[noTypes]; // made private class data

    for (cell_type = 0; cell_type < noTypes; ++cell_type)
    {
        // connectivity
        entity_conn_[cell_type] = 0;
        if (connNumbers[cell_type] * no_entities_[cell_type])
            entity_conn_[cell_type] = new int[connNumbers[cell_type] * no_entities_[cell_type]];
        // entity_conn_ have nodes and also materials after Connectivity
        Connectivity(cell_type);
        // (this->*ConnFuncArray[cell_type])(entity_conn_[cell_type]);

        // labels
        entity_label[cell_type] = new int[no_entities_[cell_type]];
        Labels(cell_type, entity_label[cell_type]);
        // (this->*LabelFuncArray[cell_type])(entity_label[cell_type]);
    }
    // 1 for shells
    count_shells(entity_conn_[1], &shells4, &shells3, coStringObj::SHELL);
    // 2 for tools
    count_shells(entity_conn_[2], &tools4, &tools3, coStringObj::TOOL);
    // node label
    int *node_label = new int[numnp_];
    dsylno_(node_label);
    node_map_.setMap(numnp_, node_label);

    // nodes of  elements
    int *marks = 0;
    if (numnp_)
        marks = new int[numnp_];

    // set up node_entity_map_
    for (cell_type = 0; cell_type < noTypes; ++cell_type)
    {
        memset(marks, 0, numnp_ * sizeof(int));
        int *this_conn = entity_conn_[cell_type];
        for (i = 0; i < no_entities_[cell_type]; ++i)
        {
            for (j = 1; j < connNumbers[cell_type]; ++j) // j=0 would be the material
            {
                // relevant for shells
                if (this_conn[connNumbers[cell_type] * i + j])
                    marks[node_map_[this_conn[connNumbers[cell_type] * i + j]]] = 1;
            }
        }
        // count "entity" nodes, gather their labels
        entity_nodes_[cell_type] = 0;
        for (i = 0; i < numnp_; ++i)
        {
            entity_nodes_[cell_type] += marks[i];
        }
        node_entity_label_[cell_type] = 0;
        if (entity_nodes_[cell_type])
            node_entity_label_[cell_type] = new int[entity_nodes_[cell_type]];

        for (i = 0, j = 0; i < numnp_; ++i)
        {
            if (marks[i])
            {
                node_entity_label_[cell_type][j++] = node_label[i];
            }
        }
        node_entity_map_[cell_type].setMap(entity_nodes_[cell_type],
                                           node_entity_label_[cell_type]);
    }

    // solid, shells ,..... nodes already counted and mapped
    delete[] marks;

    // nstate_ grids...
    for (i = 0; i < noTimeReq_; ++i)
    {
        if (isdisp_ == 0 && i > 0)
        {
            setList[i] = setList[0];
            setList[0]->incRefCount();
            setMatList[i] = setMatList[0];
            setMatList[0]->incRefCount();
            setElaList[i] = setElaList[0];
            setElaList[0]->incRefCount();
#ifdef _LOCAL_REFERENCES_
            setRefList[i] = setRefList[0];
            setRefList[0]->incRefCount();
#endif
            continue;
        }
        setList[i] = gridAtTime(objName, matName, elaName,
#ifdef _LOCAL_REFERENCES_
                                refName,
#endif
                                i, &setMatList[i], &setElaList[i]
#ifdef _LOCAL_REFERENCES_
                                ,
                                &setRefList[i]
#endif
                                );
    } //end of for over states ...

    dsyclo_();

    coDoSet *theGrid = new coDoSet(objName, setList);
    theMaterials_ = new coDoSet(matName, setMatList);
    theELabels_ = new coDoSet(elaName, setElaList);
#ifdef _LOCAL_REFERENCES_
    theReferences_ = new coDoSet(refName, setRefList);
#endif

    char timeAttr[16];
    sprintf(timeAttr, "1 %d", noTimeReq_);
    theGrid->addAttribute("TIMESTEP", timeAttr);
    // in almost all occasions the time steps will be "copies"
    // of the same. But it would be too risky to exploit this
    // by reusing the first time step of theMaterials_ without
    // any check: the mesh might have been dynamically refined?
    theMaterials_->addAttribute("TIMESTEP", timeAttr);
    theELabels_->addAttribute("TIMESTEP", timeAttr);
#ifdef _LOCAL_REFERENCES_
    theReferences_->addAttribute("TIMESTEP", timeAttr);
#endif

    for (i = 0; i < noTimeReq_; ++i)
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
    /* free in destructor: we need it for the nodal data!!!
      delete [] node_solid_label;
      delete [] node_shell_label;
   */
    for (i = 0; i < noTypes; ++i)
    {
        delete[] entity_conn_[i];
        delete[] entity_label[i];
    }
    delete[] node_label;
    return theGrid;
}

// Find out which nodal variables are available:
//     - displacements
//     - velocities
//     - acceleration
//     - additional variables
whichContents ReadDSY::scanContents(const char *path)
{

    if (strcmp(path_.c_str(), path) == 0)
    {
        theSame_ = 1;
        return contents_;
    }

    theSame_ = 0;

    contents_.reset();

    path_ = path;

    dsyopn_(path_.c_str(), &rtn_, path_.length());

    if (rtn_ != 0)
    {
        Covise::sendError("scanContents: Error when opening DSY file");
        contents_.clear();
        return contents_;
    }
    dsynod_(&isdisp_, &isvel_, &isacc_, &isadd_, &rtn_);
    if (isdisp_ == 1)
    {
        INDEX ind;
        ind[0] = 4;
        ind[1] = 5;
        ind[2] = 6;
        contents_.add("Displacements (VEC)", coStringObj::VECTOR,
                      coStringObj::NODAL, ind);
    }
    if (isvel_ == 1)
    {
        INDEX ind;
        ind[0] = 7;
        ind[1] = 8;
        ind[2] = 9;
        contents_.add("Velocity (VEC)", coStringObj::VECTOR,
                      coStringObj::NODAL, ind);
    }
    if (isacc_ == 1)
    {
        INDEX ind;
        ind[0] = 10;
        ind[1] = 11;
        ind[2] = 12;
        contents_.add("Acceleration (VEC)", coStringObj::VECTOR,
                      coStringObj::NODAL, ind);
    }
    if (isadd_ != 0)
    {
        // get also additional nodal variables....
        int node_type = 12; // see documentation
        int no_titles;
        char *titles;
        int titles_len = 32 * isadd_;
        titles = new char[titles_len];
        mydsytit(&node_type, &no_titles, titles, titles_len);
        if (no_titles == isadd_) // the titles are available
        {
            // use mnemonic insteaf of full title...
            char mnemonic[25]; // No!!! Do use full name!!!!
            int co_title;
            INDEX ind;
            ind[0] = 13;
            for (co_title = 0; co_title < no_titles; ++co_title)
            {
                strncpy(mnemonic, &titles[32 * co_title + 0], 24);
                mnemonic[24] = 0;
                // @@@
                if (1 || titles[32 * co_title + 30] == '1' || titles[32 * co_title + 30] == ' ') //SCALAR
                {
                    contents_.add(mnemonic, coStringObj::SCALAR,
                                  coStringObj::NODAL, ind, '1');
                } // VECTOR
                else if (titles[32 * co_title + 30] == '2')
                {
                    contents_.add(mnemonic, coStringObj::VECTOR,
                                  coStringObj::NODAL, ind, titles[32 * co_title + 31]);
                } // TENSOR_2D
                else if (titles[32 * co_title + 30] == '3')
                {
                    contents_.add(mnemonic, coStringObj::TENSOR_2D,
                                  coStringObj::NODAL, ind, titles[32 * co_title + 31]);
                } // TENSOR_3D
                else if (titles[32 * co_title + 30] == '4')
                {
                    contents_.add(mnemonic, coStringObj::TENSOR_3D,
                                  coStringObj::NODAL, ind, titles[32 * co_title + 31]);
                }
                ++ind[0];
            }
            // the previous for loop is only OK if all variables
            // were scalars, other cases need compression!!!
            // otherwise we would request a vector when in fact
            // only a single component of it is declared (component_)
            // in an entry of the contents list
            // @@@
            // contents_.compress(13); // 13 is the lowest index for additional
            // nodal vars
        } // titles are not available: assume no add. vars
        else if (no_titles == 0)
        {
        }
        else
        {
            Covise::sendError("no_titles different from isadd_!!!");
        }
        delete[] titles;
    }
    dsyclo_();
    return contents_;
}

whichContents ReadDSY::scanGlobalContents(const char *gloPath)
{
    const char GlobalMnemonics[] = "Global var. ";
    /*
      if(theSame_ ){
         return global_contents_;
      }
      global_contents_.reset();

      if(!gloPath) path_glo_ = path_;
      else
   */
    path_glo_ = gloPath;
    global_contents_.reset();

    dsyopn_(path_glo_.c_str(), &rtn_, path_glo_.length());
    if (rtn_ != 0)
    {
        Covise::sendWarning("Error when opening DSY or THP file");
        global_contents_.clear();
        return global_contents_;
    }
    dsyhva_(&iglob_, &nmat_, &imat_, &nsect_, &isect_, &nctct_,
            &ictct_, &nrgdw_, &irgdw_, &nrbg_, &irbg_);

    int int_global_type = coStringObj::GLOBAL;
    int no_titles;
    char *titles;
    int titles_len = 32 * iglob_;
    titles = new char[titles_len];
    dsytit_(&int_global_type, &no_titles, titles, titles_len);

    INDEX ind;
    ind[0] = 1; // Is this correct??? Yes, hopefully...
    if (no_titles != 0 && no_titles == iglob_)
    {
        if (no_titles > GLOBAL_REDUCE)
            no_titles = GLOBAL_REDUCE;

        char mnemonic[25]; // No!!! Do use full title
        int co_title;
        for (co_title = 0; co_title < no_titles; ++co_title)
        {
            if (globalVarNotInFile(ind))
                continue;
            strncpy(mnemonic, &titles[32 * co_title + 0], 24);
            mnemonic[24] = 0;
            // @@@
            if (1 || titles[32 * co_title + 30] == '1' || titles[32 * co_title + 30] == ' ') //SCALAR
            {
                global_contents_.add(mnemonic, coStringObj::SCALAR,
                                     coStringObj::GLOBAL, ind, '1');
            } // VECTOR
            else if (titles[32 * co_title + 30] == '2')
            {
                global_contents_.add(mnemonic, coStringObj::VECTOR,
                                     coStringObj::GLOBAL, ind, titles[32 * co_title + 31]);
            } // TENSOR_2D
            else if (titles[32 * co_title + 30] == '3')
            {
                global_contents_.add(mnemonic, coStringObj::TENSOR_2D,
                                     coStringObj::GLOBAL, ind, titles[32 * co_title + 31]);
            } // TENSOR_3D
            else if (titles[32 * co_title + 30] == '4')
            {
                global_contents_.add(mnemonic, coStringObj::TENSOR_3D,
                                     coStringObj::GLOBAL, ind, titles[32 * co_title + 31]);
            }
            ++ind[0];
        }
        // @@@
        // global_contents_.compress(1);
    } // titles are not available
    else if (no_titles == 0)
    {
        if (iglob_ > GLOBAL_REDUCE)
            iglob_ = GLOBAL_REDUCE;
        Covise::sendInfo("No titles available for global variables");
        int var;
        std::string mnemonic;
        char tail[16];
        for (var = 0; var < iglob_; ++var)
        {
            if (globalVarNotInFile(ind))
                continue;
            mnemonic = GlobalMnemonics;
            sprintf(tail, "%d", var);
            mnemonic += tail;
            // provisional
            global_contents_.add(mnemonic.c_str(), coStringObj::SCALAR,
                                 coStringObj::GLOBAL, ind, '1');
            ++ind[0];
        }
    }
    else
    {
        Covise::sendError("no_titles different from iglob_!!!");
    }
    // time is considered a global variable
    global_contents_.add("Time", coStringObj::SCALAR,
                         coStringObj::GLOBAL, ind, '1');
    delete[] titles;

    // now we may add other kind of "global" objects:
    // variables for mat., tr. sec, co. i., rw., ....

    addSpecialGlobals(coStringObj::MATERIAL);
    addSpecialGlobals(coStringObj::TRANS_SECTION);
    addSpecialGlobals(coStringObj::CONTACT_INTERFACE);
    addSpecialGlobals(coStringObj::RIGID_WALL);
    addSpecialGlobals(coStringObj::AIRBAG);
    addSpecialGlobals(coStringObj::AIRBAG_CHAM);
    addSpecialGlobals(coStringObj::AIRBAG_WALL);

    dsyclo_();
    return global_contents_;
}

// Find out which variables are available:
//     for the cell case.
// Assume that scanContents has been previously called

whichContents ReadDSY::scanCellContents()
{
    // use if no titles are available
    // AQUI
    const char *CellMnemonics[noTypes] = {
        "Solid ", "Shell ",
        "Tool  ", "Bar ", "Beam ", "Bar1 ",
        "Spring ", "SphJ ", "FlTorJ ",
        "SptWeld ", "Jet ", "KineJ ",
        "MISptW "
#ifdef _INCLUDE_SPH_
        ,
        "SPH"
#endif
    };

    if (theSame_)
    {
        return cell_contents_;
    }

    cell_contents_.reset();

    dsyopn_(path_.c_str(), &rtn_, path_.length());

    initEntities();

    if (rtn_ != 0)
    {
        Covise::sendError("Error when opening DSY file");
        return cell_contents_;
    }
    int ind_type;
    coStringObj::ElemType cell_type;
    int nall;
    int iall; // number of variables
    for (ind_type = 0; ind_type < noTypes; ++ind_type)
    {
        cell_type = Types[ind_type];
        int int_cell_type = cell_type;
        mydsyhal(&int_cell_type, &nall, &iall, &rtn_);

        if (nall != 0 && iall != 0 /* && Types[ind_type] != coStringObj::TOOL */)
        {
            // tool variables have no titles...
            int no_titles;
            char *titles;
            int titles_len = 32 * iall;
            titles = new char[titles_len];
            mydsytit(&int_cell_type, &no_titles, titles, titles_len);
            // DEBUG: for 1D elements print titles
            // printTitles(int_cell_type,no_titles,titles);
            if (no_titles < 0)
                no_titles = 0;

            if (no_titles == iall) // the titles are available
            {
                if (no_titles > CELL_REDUCE)
                {
                    no_titles = CELL_REDUCE;
                }

                char mnemonic[25]; // use full title, not mnemonic!!!
                int co_title;
                INDEX ind;
                ind[0] = 1; // Is this correct??? Yes, hopefully...
                // It will have to be modified for 1D elems
                FindVectors(cell_type, no_titles, titles);
                for (co_title = 0; co_title < no_titles; ++co_title)
                {
                    if (cellVarNotInFile(cell_type, ind))
                        continue;
                    // translates if necessary (all 1D elems but not beams)
                    // if there are no translations available, and
                    // the title is not acceptable, then DO NOT ADD,
                    // but increase ind[0]
                    if (Translations(cell_type, &titles[32 * co_title]) == 0)
                    {
                        strncpy(mnemonic, &titles[32 * co_title + 0], 24);
                        mnemonic[24] = 0;
                        // @@@
                        if (1 || titles[32 * co_title + 30] == '1' || titles[32 * co_title + 30] == ' ') //SCALAR
                        {
                            cell_contents_.add(mnemonic, coStringObj::SCALAR,
                                               cell_type, ind, '1');
                        } // VECTOR
                        else if (titles[32 * co_title + 30] == '2')
                        {
                            cell_contents_.add(mnemonic, coStringObj::VECTOR,
                                               cell_type, ind, titles[32 * co_title + 31]);
                        } // TENSOR_2D
                        else if (titles[32 * co_title + 30] == '3')
                        {
                            cell_contents_.add(mnemonic, coStringObj::TENSOR_2D,
                                               cell_type, ind, titles[32 * co_title + 31]);
                        } // TENSOR_3D
                        else if (titles[32 * co_title + 30] == '4')
                        {
                            cell_contents_.add(mnemonic, coStringObj::TENSOR_3D,
                                               cell_type, ind, titles[32 * co_title + 31]);
                        }
                    }
                    ++ind[0];
                }
                // the previous for loop is only OK if all variables
                // were scalars, other cases need compression!!!
                // otherwise we would request a vector when in fact
                // only a single component of it is declared (component_)
                // in an entry of the contents list
                // @@@
                // cell_contents_.compress(1); // 1 is the lowest index for
                // cell vars
            } // titles are not available
            else if (no_titles == 0)
            {
                if (iall > CELL_REDUCE)
                {
                    iall = CELL_REDUCE;
                }
                Covise::sendInfo("No titles available for some cell variables");
                int var;
                std::string mnemonic;
                char tail[16];
                INDEX ind;
                ind[0] = 1;
                for (var = 0; var < iall; ++var)
                {
                    if (cellVarNotInFile(cell_type, ind))
                        continue;
                    mnemonic = CellMnemonics[ind_type];
                    sprintf(tail, "%d", var);
                    mnemonic += tail;
                    // provisional
                    cell_contents_.add(mnemonic.c_str(), coStringObj::SCALAR,
                                       cell_type, ind, '1');
                    ++ind[0];
                }
            }
            else
            {
                Covise::sendError("no_titles different from isadd_!!!");
            }
            delete[] titles;
        }
    }

    dsyclo_();
    return cell_contents_;
}

// counts how many shells are triangles and how many are quads
void ReadDSY::count_shells(const int *shell_conn, int *shells4, int *shells3,
                           coStringObj::ElemType type)
{
    int j, itype;
    *shells4 = 0;
    *shells3 = 0;
    if (!shell_conn || (type != coStringObj::SHELL && type != coStringObj::TOOL))
        return;
    switch (type)
    {
    case coStringObj::SHELL:
        itype = 1;
        break;
    case coStringObj::TOOL:
        itype = 2;
        break;
    default:
        Covise::sendError("ReadDSY::count_shells: Error: Reached "
                          "unreachable point in module-execution.");
        return;
    }
    for (j = 0; j < no_entities_[itype]; ++j)
    {
        if (shell_conn[5 * j + 4] != 0 && shell_conn[5 * j + 3] != shell_conn[5 * j + 4])
        {
            (*shells4)++;
        }
        else
        {
            (*shells3)++;
        }
    }
}

void ReadDSY::Connectivity(int cell_type)
{
    // 0 -> solids
    // 1 -> shells
    // 2 -> tools
    // 3 -> bars
    // 4 -> beams

    switch (cell_type)
    {
    case 0: // solid
        dsybri_(entity_conn_[0]);
        break;
#ifdef _INCLUDE_SPH_
    case 13: // SPH
    {
        int i, tmp;
        dsysph_(entity_conn_[13]);
        // swap node label and material label
        for (i = 0; i < no_entities_[13]; ++i)
        {
            tmp = entity_conn_[13][2 * i + 0]; // node label
            // material
            entity_conn_[13][2 * i + 0] = entity_conn_[13][2 * i + 1];
            entity_conn_[13][2 * i + 1] = tmp;
        }
    }
    break;
#endif
    case 1: // shells
        dsyshe_(entity_conn_[1]);
        break;
    case 2: // tools
        dsytoo_(entity_conn_[2]);
        break;
    case 3: // bars
    {
        int count1D, j;
        for (count1D = 0, j = 0; count1D < no_1D_entities_; ++count1D)
        {
            // "simple" bars have type 1 and material between 201??? and 204
            if (oneD_conn_[6 * count1D + 4] == 1 && oneD_conn_[6 * count1D + 5] <= 204)
            {
                entity_conn_[3][3 * j + 0] = oneD_conn_[6 * count1D + 5];
                entity_conn_[3][3 * j + 1] = oneD_conn_[6 * count1D + 0];
                entity_conn_[3][3 * j + 2] = oneD_conn_[6 * count1D + 1];
                ++j;
            }
        }
    }
    break;
    case 5: // bar1
    {
        int count1D, j;
        for (count1D = 0, j = 0; count1D < no_1D_entities_; ++count1D)
        {
            // "simple" bars hava type 1 and material between 201??? and 204
            // bar1 elements have material 205, if the people of
            // the esi group have told me the truth.
            if (oneD_conn_[6 * count1D + 4] == 1 && oneD_conn_[6 * count1D + 5] == 205)
            {
                entity_conn_[5][3 * j + 0] = oneD_conn_[6 * count1D + 5];
                entity_conn_[5][3 * j + 1] = oneD_conn_[6 * count1D + 0];
                entity_conn_[5][3 * j + 2] = oneD_conn_[6 * count1D + 1];
                ++j;
            }
        }
    }
    break;
    case 4: // beams
    {
        int count1D, j;
        for (count1D = 0, j = 0; count1D < no_1D_entities_; ++count1D)
        {
            // beams hava type 2
            if (oneD_conn_[6 * count1D + 4] == 2)
            {
                entity_conn_[4][3 * j + 0] = oneD_conn_[6 * count1D + 5];
                entity_conn_[4][3 * j + 1] = oneD_conn_[6 * count1D + 0];
                entity_conn_[4][3 * j + 2] = oneD_conn_[6 * count1D + 1];
                ++j;
            }
        }
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
        int count1D, j;
        for (count1D = 0, j = 0; count1D < no_1D_entities_; ++count1D)
        {
            if (oneD_conn_[6 * count1D + 4] == cell_type - 3)
            {
                entity_conn_[cell_type][3 * j + 0] = oneD_conn_[6 * count1D + 5];
                entity_conn_[cell_type][3 * j + 1] = oneD_conn_[6 * count1D + 0];
                entity_conn_[cell_type][3 * j + 2] = oneD_conn_[6 * count1D + 1];
                ++j;
            }
        }
    }
    break;
    }
}

void ReadDSY::Labels(int cell_type, int *label)
{
    switch (cell_type)
    {
    case 0: // solid
        dsylbr_(label);
        break;
#ifdef _INCLUDE_SPH_
    case 13: // SPH
        dsyths_(&no_entities_[13], label);
        break;
#endif
    case 1: // shells
        dsylsh_(label);
        break;
    case 2: // tools
        dsylto_(label);
        break;
    case 3: // bars
    {
        int count1D, j;
        for (count1D = 0, j = 0; count1D < no_1D_entities_; ++count1D)
        {
            if (oneD_conn_[6 * count1D + 4] == 1 && oneD_conn_[6 * count1D + 5] <= 204)
            {
                label[j++] = oneD_label_[count1D];
            }
        }
    }
    break;
    case 5: // bar1
    {
        int count1D, j;
        for (count1D = 0, j = 0; count1D < no_1D_entities_; ++count1D)
        {
            if (oneD_conn_[6 * count1D + 4] == 1 && oneD_conn_[6 * count1D + 5] == 204)
            {
                label[j++] = oneD_label_[count1D];
            }
        }
    }
    break;
    case 4: // beams
    {
        int count1D, j;
        for (count1D = 0, j = 0; count1D < no_1D_entities_; ++count1D)
        {
            if (oneD_conn_[6 * count1D + 4] == 2)
            {
                label[j++] = oneD_label_[count1D];
            }
        }
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
        int count1D, j;
        for (count1D = 0, j = 0; count1D < no_1D_entities_; ++count1D)
        {
            if (oneD_conn_[6 * count1D + 4] == cell_type - 3)
            {
                label[j++] = oneD_label_[count1D];
            }
        }
    }
    break;
    }
}

void ReadDSY::initEntities(void)
{
    // 0 -> solids
    // 1 -> shells
    // 2 -> tools
    // 3 -> bars
    // 4 -> beams
    dsyvar_(&ndim_, &numnp_, &no_entities_[0], &no_entities_[3],
            &no_entities_[1], &no_entities_[2]);

    no_1D_entities_ = no_entities_[3];

    // no_entities_[3] has now all 1D elements, not only bars!!!
    delete[] oneD_conn_;
    delete[] oneD_label_;
    oneD_conn_ = new int[6 * no_1D_entities_];
    oneD_label_ = new int[no_1D_entities_];
    dsylbe_(oneD_label_);
    int count1D;
    int no_bars = 0;
    dsybea_(oneD_conn_);
    // AQUI
    // bar bookkeeping
    for (count1D = 0; count1D < no_1D_entities_; ++count1D)
    {
        if (oneD_conn_[6 * count1D + 4] == 1 && oneD_conn_[6 * count1D + 5] <= 204)
        {
            ++no_bars;
        }
    }
    no_entities_[3] = no_bars;
    // beam bookkeeping
    no_entities_[4] = bookkeeping(2);
    // bar1 bookkeeping
    int no_bar1 = 0;
    for (count1D = 0; count1D < no_1D_entities_; ++count1D)
    {
        if (oneD_conn_[6 * count1D + 4] == 1 && oneD_conn_[6 * count1D + 5] == 205)
        {
            ++no_bar1;
        }
    }
    no_entities_[5] = no_bar1;
    // spring bookkeeping
    no_entities_[6] = bookkeeping(3);
    // spherical joint bookkeeping
    no_entities_[7] = bookkeeping(4);
    // flexion torsion joint bookkeeping
    no_entities_[8] = bookkeeping(5);
    // spotweld bookkeeping
    no_entities_[9] = bookkeeping(6);
    // jet bookkeeping
    no_entities_[10] = bookkeeping(7);
    // kinematic joint bookkeeping
    no_entities_[11] = bookkeeping(8);
    // mesh-independent spotweld bookkeeping
    no_entities_[12] = bookkeeping(9);
#ifdef _INCLUDE_SPH_
    // SPH bookkeeping
    // use dsyhal
    int itype = coStringObj::SPH;
    int dummy;
    dsyhal_(&itype, &no_entities_[13], &dummy, &rtn_);
    if (rtn_ == 28)
    {
        Covise::sendInfo("dsyhal: type 16 for SPH not supported");
        no_entities_[13] = 0;
    }
/*
      int itype=21;  // see docu of dsyalt_!!!!, do not use 16!!!
      dsyalt_(&itype,&no_entities_[13],&rtn_);
      // rtn_ = 29;
      if(rtn_ == 28){
         Covise::sendInfo("dsyalt: type 21 for SPH not supported");
         no_entities_[13] = 0;
      } else if(rtn_ == 29){
         Covise::sendInfo("dsyalt: not a time history file");
         // use dsyhal
         itype=coStringObj::SPH;
   int dummy;
   dsyhal_(&itype,&no_entities_[13],&dummy,&rtn_);
   if(rtn_ == 28){
   Covise::sendInfo("dsyhal: type 16 for SPH not supported");
   no_entities_[13] = 0;
   }
   }
   */
#endif
}

int ReadDSY::bookkeeping(int dsybeano)
{
    int count1D;
    int no = 0;
    for (count1D = 0; count1D < no_1D_entities_; ++count1D)
    {
        if (oneD_conn_[6 * count1D + 4] == dsybeano)
        {
            ++no;
        }
    }
    return no;
}

void ReadDSY::clean()
{
    int i;
    for (i = 0; i < noTypes; ++i)
    {
        delete[] node_entity_label_[i];
        node_entity_label_[i] = 0;
    }
    delete[] zeit_;
    /*
         delete [] oneD_conn_;
         delete [] oneD_label_;
         oneD_conn_ = oneD_label_ = 0;
   */
}
