/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#ifndef _WIN32
#define COIDENT "$Header: /vobs/covise/src/application/general/READ_PAM/ReadPAM.cpp /main/vir_main/11 10-Aug-2001.11:50:12 dirk_te $"
#include <util/coIdent.h>
#endif

#include "ReadPAM.h"

ReadPam::ReadPam(int argc, char *argv[])
    : coModule(argc, argv, "Read  Pam Data")
{
    const char *IniChoice[1] = { "none" };
    // parameters
    p_dsy = addFileBrowserParam("DSY_file", "DSY file");
    p_dsy->setValue("/var/tmp/dsy.DSY", "*.DSY");

    p_thp = addFileBrowserParam("THP_file", "THP file");
    p_thp->setValue("/var/tmp/thp.THP", "*.THP");

    p_scale = addFloatParam("scale", "Displacement scale");
    p_scale->setValue(1.0);

    p_times = addInt32VectorParam("timeSteps", "select time steps");
    p_times->setValue(1, 1, 1); // minimum, maximum, jump

    int i;
    for (i = 0; i < NODAL_PORTS; ++i)
    {
        std::string param_name("Nodal_Var");
        std::string param_descr("Choose nodal variable ");
        char tail[16];
        sprintf(tail, "%d", i + 1);
        param_name += tail;
        param_descr += tail;
        p_nodal_ch[i] = addChoiceParam(param_name.c_str(), param_descr.c_str());
        p_nodal_ch[i]->setValue(1, IniChoice, 0);
    }

    for (i = 0; i < CELL_PORTS; ++i)
    {
        std::string param_name("Cell_Var");
        std::string param_descr("Choose cell variable ");
        char tail[16];
        sprintf(tail, "%d", i + 1);
        param_name += tail;
        param_descr += tail;
        p_cell_ch[i] = addChoiceParam(param_name.c_str(), param_descr.c_str());
        p_cell_ch[i]->setValue(1, IniChoice, 0);
    }

    p_file = addBooleanParam("DSY_or_THP", "DSY or THP");
    p_file->setValue(1);

    for (i = 0; i < GLOBAL_PORTS; ++i)
    {
        std::string param_name("Global_Var");
        std::string param_descr("Choose global variable ");
        char tail[16];
        sprintf(tail, "%d", i + 1);
        param_name += tail;
        param_descr += tail;
        p_global_ch[i] = addChoiceParam(param_name.c_str(), param_descr.c_str());
        p_global_ch[i]->setValue(1, IniChoice, 0);
    }

    int TportNo;
    char tail[16];
    const char *Component[9] = { "XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ" };
    p_Tport = paraSwitch("Port Select :", "use if you want to output a full tensor");
    for (TportNo = 0; TportNo < TENSOR_PORTS; ++TportNo)
    {
        std::string TportTitle("Tensor port ");
        sprintf(tail, "%d", TportNo + 1);
        TportTitle += tail;
        std::string param_pre_name("T");
        param_pre_name += tail;
        param_pre_name += "_Component_";
        paraCase(TportTitle.c_str());
        for (i = 0; i < 9; ++i)
        {
            std::string param_name(param_pre_name);
            std::string param_descr("Component ");
            param_name += Component[i];
            param_descr += Component[i];
            p_Tcomponents[TportNo][i] = addChoiceParam(param_name.c_str(), param_descr.c_str());
            p_Tcomponents[TportNo][i]->setValue(1, IniChoice, 0);
        }
        paraEndCase();
    }
    paraEndSwitch();

    // ports
    p_grid = addOutputPort("meshOut", "UnstructuredGrid", "output mesh");

    for (i = 0; i < NODAL_PORTS; ++i)
    {
        std::string port_name("nodalData");
        std::string port_descr("nodal data ");
        char tail[16];
        sprintf(tail, "%d", i + 1);
        port_name += tail;
        port_descr += tail;
        p_nodal_obj[i] = addOutputPort(port_name.c_str(), "Float|Vec3", port_descr.c_str());
    }
    for (i = 0; i < CELL_PORTS; ++i)
    {
        std::string port_name("cellData");
        std::string port_descr("cell data ");
        char tail[16];
        sprintf(tail, "%d", i + 1);
        port_name += tail;
        port_descr += tail;
        p_cell_obj[i] = addOutputPort(port_name.c_str(), "Float|Vec3", port_descr.c_str());
    }
    for (i = 0; i < GLOBAL_PORTS; ++i)
    {
        std::string port_name("globalData");
        std::string port_descr("global data ");
        char tail[16];
        sprintf(tail, "%d", i + 1);
        port_name += tail;
        port_descr += tail;
        p_global_obj[i] = addOutputPort(port_name.c_str(), "Float|Vec3", port_descr.c_str());
    }
    for (i = 0; i < TENSOR_PORTS; ++i)
    {
        std::string port_name("tensorData");
        std::string port_descr("tensor data ");
        char tail[16];
        sprintf(tail, "%d", i + 1);
        port_name += tail;
        port_descr += tail;
        p_tensor_obj[i] = addOutputPort(port_name.c_str(), "Tensor", port_descr.c_str());
    }

    p_materials = addOutputPort("Materials", "IntArr", "Element materials");
    p_elementL = addOutputPort("Elem_labels", "IntArr", "Element labels");
#ifdef _LOCAL_REFERENCES_
    p_references = addOutputPort("References", "coDoMat3", "Element loc. coord. sys.");
#endif
    dsy_ok_ = 0;
    thp_ok_ = 0;
}

void ReadPam::postInst()
{
    p_thp->disable();
}

ReadPam::~ReadPam()
{
}

int ReadPam::compute(const char *)
{
    // tensor object descriptions
    if (!dsy_ok_)
    {
        sendError("Problem with DSY file");
        return FAIL;
    }
    if (!thp_ok_ && !p_file->getValue())
    {
        sendError("Problem with THP file");
        return FAIL;
    }
    if (p_times->getValue(0) > p_times->getValue(1))
    {
        sendError("Minimum time may not be larger than the maximum one");
        p_times->setValue(0, p_times->getValue(1));
        return FAIL;
    }
    if (p_times->getValue(1) > noStates_)
    {
        sendError("Maximum time is too large");
        p_times->setValue(1, noStates_);
        return FAIL;
    }
    if (p_times->getValue(0) < 1)
    {
        sendError("Minimum time may not be smaller than 1");
        p_times->setValue(0, 1);
        return FAIL;
    }
    if (p_times->getValue(2) < 1)
    {
        sendError("Jump value for time steps must be positive");
        p_times->setValue(2, 1);
        return FAIL;
    }

    readDSY.setTimeRequest(p_times->getValue(0), p_times->getValue(1),
                           p_times->getValue(2));

    TensDescriptions tensDescriptions;

    if (fillDescriptions(tensDescriptions) < 0)
    {
        return FAIL;
    }

    // the grid
    coDistributedObject *Grid = readDSY.grid(p_grid->getObjName(), p_materials->getObjName(),
                                             p_elementL->getObjName(),
#ifdef _LOCAL_REFERENCES_
                                             p_references->getObjName(),
#endif
                                             p_scale->getValue());
    if (!Grid)
        return FAIL;
    p_grid->setCurrentObject(Grid);

    // materials and references
    p_materials->setCurrentObject(readDSY.materials());
    p_elementL->setCurrentObject(readDSY.eleLabels());
#ifdef _LOCAL_REFERENCES_
    p_references->setCurrentObject(readDSY.references());
#endif

    // may we use a scale different from 1.0?
    if (p_scale->getValue() != 1.0 && readDSY.getScale() == 1.0)
    {
        sendInfo("Displacements not available for all time steps. Assume scale=1");
        p_scale->setValue(1.0);
    }

    // nodal objects
    int i, j;
    int no_request = 0;

    for (i = 0; i < NODAL_PORTS; ++i)
    {
        if (p_nodal_ch[i]->getValue() > 0)
            ++no_request;
    }

    std::string *requests = 0;
    int *req_label = 0;
    if (no_request)
    {
        requests = new std::string[no_request];
        req_label = new int[no_request];
    }

    for (i = 0, j = 0; i < NODAL_PORTS; ++i)
    {
        if (p_nodal_ch[i]->getValue() > 0)
        {
            requests[j] = p_nodal_obj[i]->getObjName();
            req_label[j] = p_nodal_ch[i]->getValue();
            ++j;
        }
    }

    coDoSet **nodal_obj = readDSY.nodalObj(no_request, requests, req_label);

    if (nodal_obj)
    {
        for (i = 0, j = 0; i < NODAL_PORTS; ++i)
        {
            if (p_nodal_ch[i]->getValue() > 0)
            {
                p_nodal_obj[i]->setCurrentObject(nodal_obj[j]);
                ++j;
                if (j == no_request)
                    break;
            }
        }
    }

    delete[] nodal_obj;
    delete[] requests;
    delete[] req_label;

    // cell objects
    no_request = 0;

    for (i = 0; i < CELL_PORTS; ++i)
    {
        if (p_cell_ch[i]->getValue() > 0)
            ++no_request;
    }

    requests = 0;
    req_label = 0;
    if (no_request)
    {
        requests = new std::string[no_request];
        req_label = new int[no_request];
    }

    for (i = 0, j = 0; i < CELL_PORTS; ++i)
    {
        if (p_cell_ch[i]->getValue() > 0)
        {
            requests[j] = p_cell_obj[i]->getObjName();
            req_label[j] = p_cell_ch[i]->getValue();
            ++j;
        }
    }

    coDoSet **cell_obj = readDSY.cellObj(no_request, requests, req_label);

    if (cell_obj)
    {
        for (i = 0, j = 0; i < CELL_PORTS; ++i)
        {
            if (p_cell_ch[i]->getValue() > 0)
            {
                p_cell_obj[i]->setCurrentObject(cell_obj[j]);
                ++j;
                if (j == no_request)
                    break;
            }
        }
    }

    delete[] cell_obj;
    delete[] requests;
    delete[] req_label;

    // tensor objects
    coDoSet **tensor_obj = readDSY.tensorObj(tensDescriptions);
    setTensorObj(tensor_obj, tensDescriptions);

    // global objects
    no_request = 0;

    for (i = 0; i < GLOBAL_PORTS; ++i)
    {
        if (p_global_ch[i]->getValue() > 0)
            ++no_request;
    }

    requests = 0;
    req_label = 0;
    if (no_request)
    {
        requests = new std::string[no_request];
        req_label = new int[no_request];
    }

    for (i = 0, j = 0; i < GLOBAL_PORTS; ++i)
    {
        if (p_global_ch[i]->getValue() > 0)
        {
            requests[j] = p_global_obj[i]->getObjName();
            req_label[j] = p_global_ch[i]->getValue();
            ++j;
        }
    }

    coDistributedObject **global_obj = readDSY.globalObj(no_request, requests, req_label);

    if (global_obj)
    {
        for (i = 0, j = 0; i < GLOBAL_PORTS; ++i)
        {
            if (p_global_ch[i]->getValue() > 0)
            {
                p_global_obj[i]->setCurrentObject(global_obj[j]);
                ++j;
                if (j == no_request)
                    break;
            }
        }
    }

    delete[] global_obj;
    delete[] requests;
    delete[] req_label;

    readDSY.clean();

    return SUCCESS;
}

void ReadPam::param(const char *paramName, bool in_map_loading)
{
    whichContents contents;
    whichContents cell_contents;
    whichContents global_contents;

    contents.reset();
    cell_contents.reset();
    global_contents.reset();

    if (strcmp(p_dsy->getName(), paramName) == 0 || strcmp(p_thp->getName(), paramName) == 0 || strcmp(p_Tport->getName(), paramName) == 0 || strcmp(p_file->getName(), paramName) == 0)
    {
        if (strcmp(p_dsy->getName(), paramName) == 0)
        {
            int i;
            sendInfo("Scanning DSY file for nodal variables, please wait...");
            contents = readDSY.scanContents(p_dsy->getValue());
            sendInfo("...scanning done");
            dsy_ok_ = (contents.no_options() > 0);
            if (dsy_ok_)
            {
                noStates_ = readDSY.getNoStates();
            }
            sendInfo("Scanning DSY file for cell variables, please wait...");
            cell_contents = readDSY.scanCellContents();
            sendInfo("...scanning done");
            if (in_map_loading)
            {
                for (i = 0; i < TENSOR_PORTS; ++i)
                    cell_contents_old_[i] = cell_contents;
            }
            if (!in_map_loading && p_file->getValue())
            {
                sendInfo("Scanning DSY file for global variables, please wait...");
                global_contents = readDSY.scanGlobalContents(p_dsy->getValue());
                sendInfo("...scanning done");
            }
        }
        if (strcmp(p_thp->getName(), paramName) == 0 && !in_map_loading && !p_file->getValue())
        {
            sendInfo("Scanning THP file for global variables, please wait...");
            global_contents = readDSY.scanGlobalContents(p_thp->getValue());
            sendInfo("...scanning done");
            thp_ok_ = (global_contents.no_options() > 0);
        }
        if (strcmp(p_file->getName(), paramName) == 0)
        {
            if (p_file->getValue() == 1)
            {
                sendInfo("Scanning DSY file for global variables, please wait...");
                global_contents = readDSY.scanGlobalContents(p_dsy->getValue());
                sendInfo("...scanning done");
                p_thp->disable();
            }
            else
            {
                sendInfo("Scanning THP file for global variables, please wait...");
                global_contents = readDSY.scanGlobalContents(p_thp->getValue());
                sendInfo("...scanning done");
                thp_ok_ = (global_contents.no_options() > 0);
                p_thp->enable();
            }
        }
        if (!in_map_loading)
        {
            int i;
            const char **theOptions;
            if (strcmp(p_dsy->getName(), paramName) == 0 && dsy_ok_)
            {
                p_times->setValue(1, noStates_, 1);
                // choices of nodal variables
                theOptions = new const char *[contents.no_options()];
                for (i = 0; i < contents.no_options(); ++i)
                {
                    theOptions[i] = contents[i].ObjName_.c_str();
                }
                std::string none;
                char buffer[32];
                for (i = 0; i < NODAL_PORTS; ++i)
                {
                    sprintf(buffer, "%d: none", i + 1);
                    p_nodal_ch[i]->setValue(contents.no_options(),
                                            theOptions, 0);
                    none = "nodalData";
                    none += buffer;
                    p_nodal_obj[i]->setInfo(none.c_str());
                }
                delete[] theOptions;
                // choices of cell variables
                theOptions = new const char *[cell_contents.no_options()];
                for (i = 0; i < cell_contents.no_options(); ++i)
                {
                    theOptions[i] = cell_contents[i].ObjName_.c_str();
                }
                for (i = 0; i < CELL_PORTS; ++i)
                {
                    sprintf(buffer, "%d: none", i + 1);
                    none = "cellData";
                    none += buffer;
                    p_cell_ch[i]->setValue(cell_contents.no_options(),
                                           theOptions, 0);
                    p_cell_obj[i]->setInfo(none.c_str());
                }
                // choices for making tensors...
                //    ... use only "scalar" components
                int scalarOrder, component;
                for (i = 0; i < TENSOR_PORTS; ++i)
                    cell_contents_old_[i] = cell_contents;
                for (i = 0, scalarOrder = 0; i < cell_contents.no_options(); ++i)
                {
                    if (cell_contents[i].ObjType_ == coStringObj::SCALAR || cell_contents[i].ObjType_ == coStringObj::NONE)
                    {
                        theOptions[scalarOrder] = cell_contents[i].ObjName_.c_str();
                        ++scalarOrder;
                    }
                }
                for (i = 0; i < TENSOR_PORTS; ++i)
                    for (component = 0; component < 9; ++component)
                        p_Tcomponents[i][component]->setValue(scalarOrder,
                                                              theOptions, 0);
                // end of tensor options
                delete[] theOptions;
            }
            if (strcmp(p_Tport->getName(), paramName) == 0 && p_Tport->getValue() > 0 && dsy_ok_
                /*&&
                        !((cell_contents = readDSY.getCellContents())==
                               cell_contents_old_[p_Tport->getValue()-1])*/
                )
            {
                cell_contents = readDSY.getCellContents();
                cell_contents_old_[p_Tport->getValue() - 1] = cell_contents;
                // choices of cell variables
                theOptions = new const char *[cell_contents.no_options()];
                for (i = 0; i < cell_contents.no_options(); ++i)
                {
                    theOptions[i] = cell_contents[i].ObjName_.c_str();
                }
                // choices for making tensors...
                //    ... use only "scalar" components
                int scalarOrder, component;
                for (i = 0, scalarOrder = 0; i < cell_contents.no_options(); ++i)
                {
                    if (cell_contents[i].ObjType_ == coStringObj::SCALAR || cell_contents[i].ObjType_ == coStringObj::NONE)
                    {
                        theOptions[scalarOrder] = cell_contents[i].ObjName_.c_str();
                        ++scalarOrder;
                    }
                }
                for (component = 0; component < 9; ++component)
                    p_Tcomponents[p_Tport->getValue() - 1][component]->hide();
                for (component = 0; component < 9; ++component)
                    p_Tcomponents[p_Tport->getValue() - 1][component]->setValue(scalarOrder, theOptions, 1);
                for (component = 0; component < 9; ++component)
                    p_Tcomponents[p_Tport->getValue() - 1][component]->show();
                // end of tensor options
                delete[] theOptions;
            }
            // p_global_ch: these choice lists should only be
            // modified in the following cases:
            // DSY file changed && DSY/THP switch is on ||
            // THP file changed && DSY/THP switch is off ||
            // DSY/THP switch changed
            if (global_contents.no_options())
                if ((strcmp(p_dsy->getName(), paramName) == 0 && p_file->getValue()) || (strcmp(p_thp->getName(), paramName) == 0 && !p_file->getValue()) || strcmp(p_file->getName(), paramName) == 0)
                {
                    theOptions = new const char *[global_contents.no_options()];
                    for (i = 0; i < global_contents.no_options(); ++i)
                    {
                        theOptions[i] = global_contents[i].ObjName_.c_str();
                    }
                    std::string none;
                    char buffer[32];
                    for (i = 0; i < GLOBAL_PORTS; ++i)
                    {
                        sprintf(buffer, "%d: none", i + 1);
                        p_global_ch[i]->setValue(global_contents.no_options(),
                                                 theOptions, 0);
                        none = "globalData";
                        none += buffer;
                        p_global_obj[i]->setInfo(none.c_str());
                    }
                    delete[] theOptions;
                }
        }
    }

    // change port title dynamically
    // when we choose some item from the choice lists
    if (!in_map_loading)
    {
        int i;
        int choice;
        char ret[26];
        // first nodal ports...
        if (dsy_ok_)
            for (i = 0; i < NODAL_PORTS; ++i)
            {
                if (strcmp(p_nodal_ch[i]->getName(), paramName) == 0)
                {
                    // nodal_choice[i] = p_nodal_ch[i]->getValue();
                    choice = p_nodal_ch[i]->getValue();
                    if (choice > 0)
                        // get variable title
                        p_nodal_obj[i]->setInfo(readDSY.getTitle(choice, 0, ret));
                    else
                        p_nodal_obj[i]->setInfo("none");
                }
            }
        // cell ports...
        if (dsy_ok_)
            for (i = 0; i < CELL_PORTS; ++i)
            {
                if (strcmp(p_cell_ch[i]->getName(), paramName) == 0)
                {
                    if ((choice = p_cell_ch[i]->getValue()) > 0)
                        // get variable title
                        p_cell_obj[i]->setInfo(readDSY.getTitle(choice, 1, ret));
                    else
                        p_cell_obj[i]->setInfo("none");
                }
            }
        // global ports...
        if ((dsy_ok_ && p_file->getValue()) || (thp_ok_ && !p_file->getValue()))
            for (i = 0; i < GLOBAL_PORTS; ++i)
            {
                if (strcmp(p_global_ch[i]->getName(), paramName) == 0)
                {
                    if ((choice = p_global_ch[i]->getValue()) > 0)
                        // get variable title
                        p_global_obj[i]->setInfo(readDSY.getTitle(choice - 1, 2, ret));
                    else
                        p_global_obj[i]->setInfo("none");
                }
            }
    }
}

const int XX = 0;
const int XY = 1;
const int XZ = 2;
const int YX = 3;
const int YY = 4;
const int YZ = 5;
const int ZX = 6;
const int ZY = 7;
const int ZZ = 8;
// descriptions for tensor objects
int ReadPam::fillDescriptions(TensDescriptions &tensDescriptions)
{
    int port, countObj;
    coDoTensor::TensorType types[TENSOR_PORTS];

    // first check if the user is not mixing variables defined
    // in different subgrids
    int param;
    whichContents cell_contents = readDSY.getCellContents();
    // count types NONE and SCALAR...
    int i, none_scalar;
    for (i = 0, none_scalar = 0; i < cell_contents.no_options(); ++i)
    {
        if (cell_contents[i].ObjType_ == coStringObj::SCALAR || cell_contents[i].ObjType_ == coStringObj::NONE)
        {
            ++none_scalar;
        }
    }
    int *mapScalar = new int[none_scalar]; // DELETE hier
    for (i = 0, none_scalar = 0; i < cell_contents.no_options(); ++i)
    {
        if (cell_contents[i].ObjType_ == coStringObj::SCALAR || cell_contents[i].ObjType_ == coStringObj::NONE)
        {
            mapScalar[none_scalar] = i;
            ++none_scalar;
        }
    }

    coStringObj::ElemType e_type = coStringObj::NODAL; // init-value chosen arbitrary to be able to compile on linux (mw 10102005)
    for (port = 0; port < TENSOR_PORTS; ++port)
    {
        int noPrevious = 1; // flag to detect if we have alredy found a previous object
        for (param = 0; param < 9; ++param)
        {
            if (!noPrevious && cell_contents[mapScalar[p_Tcomponents[port][param]->getValue()]].ObjType_ != coStringObj::NONE
                && cell_contents[mapScalar[p_Tcomponents[port][param]->getValue()]].ElemType_ != e_type)
            {
                delete[] mapScalar;
                sendError("Tensor construction: you may not mix variables defined for different element types");
                return -1;
            }
            if (noPrevious && cell_contents[mapScalar[p_Tcomponents[port][param]->getValue()]].ObjType_ != coStringObj::NONE)
            {
                e_type = cell_contents[mapScalar[p_Tcomponents[port][param]->getValue()]].ElemType_;
                noPrevious = 0;
            }
        }
    }
    delete[] mapScalar;

    // deleted by
    tensDescriptions.markPort = new char[TENSOR_PORTS];
    // tensDescriptions' destructor
    memset(tensDescriptions.markPort, 'N', TENSOR_PORTS);

    for (port = 0, countObj = 0; port < TENSOR_PORTS; ++port)
    {
        // test for every port and for every tensor type
        if (isNull(port))
        {
            types[port] = coDoTensor::UNKNOWN;
            continue;
        }
        else if (isS2D(port))
        {
            tensDescriptions.markPort[port] = 'Y';
            types[port] = coDoTensor::S2D;
            ++countObj;
        }
        else if (isS3D(port))
        {
            tensDescriptions.markPort[port] = 'Y';
            types[port] = coDoTensor::S3D;
            ++countObj;
        }
        else if (isF2D(port))
        {
            tensDescriptions.markPort[port] = 'Y';
            types[port] = coDoTensor::F2D;
            ++countObj;
        }
        else if (isF3D(port))
        {
            tensDescriptions.markPort[port] = 'Y';
            types[port] = coDoTensor::F3D;
            ++countObj;
        }
        else
        {
            sendError("Cannot make tensor for port %d", port + 1);
            return -1;
        }
    }

    tensDescriptions.no_request = countObj;
    tensDescriptions.req_label = new int[9 * countObj];
    tensDescriptions.requests = new std::string[countObj];
    tensDescriptions.ttype = new coDoTensor::TensorType[countObj];

    for (port = 0, countObj = 0; port < TENSOR_PORTS; ++port)
    {
        if (types[port] != coDoTensor::UNKNOWN)
        {
            tensDescriptions.requests[countObj] = p_tensor_obj[port]->getObjName();
            tensDescriptions.ttype[countObj] = types[port];
            switch (types[port])
            {
            case coDoTensor::S2D:
                tensDescriptions.req_label[9 * countObj + 0] = p_Tcomponents[port][XX]->getValue();
                tensDescriptions.req_label[9 * countObj + 1] = p_Tcomponents[port][YY]->getValue();
                if (p_Tcomponents[port][XY]->getValue() != 0)
                    tensDescriptions.req_label[9 * countObj + 2] = p_Tcomponents[port][XY]->getValue();
                else
                    tensDescriptions.req_label[9 * countObj + 2] = p_Tcomponents[port][YX]->getValue();
                break;
            case coDoTensor::S3D:
                tensDescriptions.req_label[9 * countObj + 0] = p_Tcomponents[port][XX]->getValue();
                tensDescriptions.req_label[9 * countObj + 1] = p_Tcomponents[port][YY]->getValue();
                tensDescriptions.req_label[9 * countObj + 2] = p_Tcomponents[port][ZZ]->getValue();
                if (p_Tcomponents[port][XY]->getValue() != 0)
                    tensDescriptions.req_label[9 * countObj + 3] = p_Tcomponents[port][XY]->getValue();
                else
                    tensDescriptions.req_label[9 * countObj + 3] = p_Tcomponents[port][YX]->getValue();
                if (p_Tcomponents[port][YZ]->getValue() != 0)
                    tensDescriptions.req_label[9 * countObj + 4] = p_Tcomponents[port][YZ]->getValue();
                else
                    tensDescriptions.req_label[9 * countObj + 4] = p_Tcomponents[port][ZY]->getValue();
                if (p_Tcomponents[port][ZX]->getValue() != 0)
                    tensDescriptions.req_label[9 * countObj + 5] = p_Tcomponents[port][ZX]->getValue();
                else
                    tensDescriptions.req_label[9 * countObj + 5] = p_Tcomponents[port][XZ]->getValue();
                break;
            case coDoTensor::F2D:
                tensDescriptions.req_label[9 * countObj + 0] = p_Tcomponents[port][XX]->getValue();
                tensDescriptions.req_label[9 * countObj + 1] = p_Tcomponents[port][XY]->getValue();
                tensDescriptions.req_label[9 * countObj + 2] = p_Tcomponents[port][YX]->getValue();
                tensDescriptions.req_label[9 * countObj + 3] = p_Tcomponents[port][YY]->getValue();
                break;
            case coDoTensor::F3D:
                tensDescriptions.req_label[9 * countObj + 0] = p_Tcomponents[port][XX]->getValue();
                tensDescriptions.req_label[9 * countObj + 1] = p_Tcomponents[port][XY]->getValue();
                tensDescriptions.req_label[9 * countObj + 2] = p_Tcomponents[port][XZ]->getValue();
                tensDescriptions.req_label[9 * countObj + 3] = p_Tcomponents[port][YX]->getValue();
                tensDescriptions.req_label[9 * countObj + 4] = p_Tcomponents[port][YY]->getValue();
                tensDescriptions.req_label[9 * countObj + 5] = p_Tcomponents[port][YZ]->getValue();
                tensDescriptions.req_label[9 * countObj + 6] = p_Tcomponents[port][ZX]->getValue();
                tensDescriptions.req_label[9 * countObj + 7] = p_Tcomponents[port][ZY]->getValue();
                tensDescriptions.req_label[9 * countObj + 8] = p_Tcomponents[port][ZZ]->getValue();
                break;
            default:
                sendError("Tensor type not supported");
                break;
            }
            ++countObj;
        }
    }
    return 0;
}

int ReadPam::isNull(int port)
{
    int i;
    for (i = 0; i < 9; ++i)
    {
        if (p_Tcomponents[port][i]->getValue() != 0)
            return 0;
    }
    return 1;
}

int ReadPam::isS2D(int port)
{
    // XX YY have to be selected
    if (p_Tcomponents[port][XX]->getValue() == 0 || p_Tcomponents[port][YY]->getValue() == 0)
        return 0;
    // Z? ?Z components are required to be unselected
    if (p_Tcomponents[port][XZ]->getValue() != 0 || p_Tcomponents[port][ZX]->getValue() != 0 || p_Tcomponents[port][YZ]->getValue() != 0 || p_Tcomponents[port][ZY]->getValue() != 0 || p_Tcomponents[port][ZZ]->getValue() != 0)
        return 0;
    // either XY or YX have to be selected
    if (p_Tcomponents[port][XY]->getValue() == 0 && p_Tcomponents[port][YX]->getValue() == 0)
        return 0;
    // if XY and YX are different, at least one of them has to be unselected
    if (p_Tcomponents[port][XY]->getValue() != p_Tcomponents[port][YX]->getValue()
        && p_Tcomponents[port][XY]->getValue() != 0 && p_Tcomponents[port][YX]->getValue() != 0)
        return 0;

    return 1;
}

int ReadPam::isS3D(int port)
{
    // XX YY ZZ have to be selected
    // other components are required to be unselected
    if (p_Tcomponents[port][XX]->getValue() == 0 || p_Tcomponents[port][YY]->getValue() == 0 || p_Tcomponents[port][ZZ]->getValue() == 0)
        return 0;
    // either XY or YX have to be selected
    if (p_Tcomponents[port][XY]->getValue() == 0 && p_Tcomponents[port][YX]->getValue() == 0)
        return 0;
    // if XY and YX are different, at least one of them has to be unselected
    if (p_Tcomponents[port][XY]->getValue() != p_Tcomponents[port][YX]->getValue()
        && p_Tcomponents[port][XY]->getValue() != 0 && p_Tcomponents[port][YX]->getValue() != 0)
        return 0;
    // either YZ or ZY have to be selected
    if (p_Tcomponents[port][YZ]->getValue() == 0 && p_Tcomponents[port][ZY]->getValue() == 0)
        return 0;
    // if YZ and ZY are different, at least one of them has to be unselected
    if (p_Tcomponents[port][YZ]->getValue() != p_Tcomponents[port][ZY]->getValue()
        && p_Tcomponents[port][YZ]->getValue() != 0 && p_Tcomponents[port][ZY]->getValue() != 0)
        return 0;
    // either ZX or XZ have to be selected
    if (p_Tcomponents[port][ZX]->getValue() == 0 && p_Tcomponents[port][XZ]->getValue() == 0)
        return 0;
    // if ZX and XZ are different, at least one of them has to be unselected
    if (p_Tcomponents[port][ZX]->getValue() != p_Tcomponents[port][XZ]->getValue()
        && p_Tcomponents[port][ZX]->getValue() != 0 && p_Tcomponents[port][XZ]->getValue() != 0)
        return 0;

    return 1;
}

int ReadPam::isF2D(int port)
{
    if (p_Tcomponents[port][XX]->getValue() == 0 || p_Tcomponents[port][YY]->getValue() == 0 || p_Tcomponents[port][XY]->getValue() == 0 || p_Tcomponents[port][YX]->getValue() == 0)
        return 0;
    if (p_Tcomponents[port][XZ]->getValue() != 0 || p_Tcomponents[port][ZX]->getValue() != 0 || p_Tcomponents[port][ZY]->getValue() != 0 || p_Tcomponents[port][YZ]->getValue() != 0 || p_Tcomponents[port][ZZ]->getValue() != 0)
        return 0;
    return 1;
}

int ReadPam::isF3D(int port)
{
    int i;
    for (i = 0; i < 9; ++i)
        if (p_Tcomponents[port][i]->getValue() == 0)
            return 0;

    return 1;
}

void ReadPam::setTensorObj(coDoSet **tensObj, TensDescriptions &tensDescriptions)
{
    int port, output;
    for (port = 0, output = 0; port < TENSOR_PORTS; ++port)
    {
        if (tensDescriptions.markPort[port] == 'Y')
        {
            p_tensor_obj[port]->setCurrentObject(tensObj[output]);
            ++output;
        }
    }
}

MODULE_MAIN(IO, ReadPam)
