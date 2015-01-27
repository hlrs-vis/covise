/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                                     ++
// ++ COVISE CGNS Reader                                                  ++
// ++ Author: Vlad                                                        ++
// ++                                                                     ++
// ++               St. Petersburg Polytechnical University               ++
// ++**********************************************************************/

//
//  Capabilities:
//	-- Supported only these UNSTRUCTURED grids:
//		--HEXA8,TETRA4 3d usg;
//		--QUAD_4,TRI_3 2d usg;
//		--MIXED usg with HEXA_8,TETRA_4,PYRA_5,TRI_3,QUAD_4, PENTA_6 elements;

//  -- This module can read multiple base moving mesh data
//	-- This module can read multiple file moving mesh data, use %03d for 3-digit index
//  -- This module can read multiple zone bases WITH SOLUTIONS OF THE SAME FORMAT

//	-- This module can reads FIRST solution in zone, only ONE coord block in zone
//  -- grid sections selection is not good yet.

#include "ReadCGNS.h"
#include <cgnslib.h>
#include <do/coDoPoints.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>

#include <vector>
#include <algorithm>
#include <string>
#include <sstream>

string int2str(int n)
{
    std::ostringstream o;
    o << n;
    return o.str();
}
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ReadCGNS::ReadCGNS(int argc, char *argv[])
    : coModule(argc, argv, "CGNS Reader alpha")
{
    cout << "ReadCGNS started. " << endl;

    param_file = addFileBrowserParam("SGNS_File", "Select CGNS file to read");
    //param_file->setValue("/home/cave/develop/nozzleSteamOP1.cgns","*.cgns");
    param_file->setValue("/home/cave/data/vlad", "*.cgns");

    param_load_2d = addBooleanParam("Load_Only_2D", "Load Only 2D meshes");
    param_load_2d->setValue(false);
    param_use_string = addBooleanParam("Use_Sections_string", "Use String for mesh sections selection");
    param_use_string->setValue(false);
    param_sections_string = addStringParam("Sections_String", "Enter section numbers here");
    param_sections_string->setValString("1");

    param_vx = addChoiceParam("vel_x", "Select VelX solution number");
    param_vy = addChoiceParam("vel_y", "Select Vely solution number");
    param_vz = addChoiceParam("vel_z", "Select Velz solution number");

    param_f[0] = addChoiceParam("float_1", "Select Solution number");
    param_f[1] = addChoiceParam("float_2", "Select Solution number");
    param_f[2] = addChoiceParam("float_3", "Select Solution number");
    param_f[3] = addChoiceParam("float_4", "Select Solution number");

    param_use_file_timesteps = addBooleanParam("Use_file_timesteps", "Check to read multiple file timesteps");
    param_first_file_idx = addInt32Param("First_file_index", "First file index");
    param_last_file_idx = addInt32Param("Last_file_index", "Last file index");
    param_use_file_timesteps->setValue(false);
    param_first_file_idx->setValue(1);
    param_last_file_idx->setValue(1);
    /*
    param_read_single_zone=addBooleanParam("RESERVED_Read_single_zone","(RESERVED)Check it to read only one zone");
    param_zone=addInt32Param("RESERVED_Zone","(RESERVED)Zone to read");
    param_read_single_zone->setValue(false);
    param_zone->setValue(1);
*/
    out_mesh = addOutputPort("mesh", "UnstructuredGrid", "Unstructured Grid");

    out_float[0] = addOutputPort("Density", "Float", "Density");
    out_float[1] = addOutputPort("Pressure", "Float", "Pressure");
    out_float[2] = addOutputPort("Nu_bar", "Float", "Nu_bar");
    out_float[3] = addOutputPort("Steam_Quality", "Float", "Steam Quality");

    out_velocity = addOutputPort("Velocity", "Vec3", "Velocity");
}

void ReadCGNS::param(const char *paramName, bool /*inMapLoading*/)
{

    //TODO:Optimize param()

    string fileParam = "SGNS_File"; //,float1param="float_1";
    //sendInfo("ReadCGNS::param()=====Parameter changed:%s , %d",paramName,(int)inMapLoading);

    //if (true/*(fileParam==paramName)*//*&&(!inMapLoading)*/) //need to check every param

    int index_file, error; //file handling

    if (param_use_file_timesteps->getValue())
    {
        char fname[300];
        indexed_file_name(fname, param_first_file_idx->getValue());
        sendWarning("ReadCGNS::param(): Multifile. Using name '%s'", fname);
        error = cg_open(fname, CG_MODE_READ, &index_file);
    }
    else
        error = cg_open(param_file->getValue(), CG_MODE_READ, &index_file);

    if (!error)
    {
        //			sendInfo("param: cg_open - File opened.");
        // -----now we read only base#1,zone#1---
        int ibase = 1, izone = 1;

        // ============Bases=========
        int numbases = 0;
        // for the first base
        int cell_dim = 0, phys_dim = 0;
        char basename[100];
        //=============zones================
        int numzones = 0;

        //for the first zone
        cgsize_t zonesize[3]; //3 sizes for unstructured grid
        char zonename[100];
        CGNS_ENUMT(ZoneType_t) zonetype;

        //===========the code=================

        //1. How many bases do we have
        cg_nbases(index_file, &numbases);

        // I should cycle the bases , but now i'll use the first one.
        // However, multiple bases=multiple cases, so numbases >1 is very rare.

        // 2. read base info
        cg_base_read(index_file, ibase, basename, &cell_dim, &phys_dim);
        //		sendInfo("param: Bases: %d. Base #1: name=%s, cell_dim= %d, phys_dim=%d ",numbases,basename,cell_dim,phys_dim);

        //3. read zone info
        error = cg_nzones(index_file, ibase, &numzones);
        if (error)
            sendInfo("param: nzones error=%d", error);
        error = cg_zone_read(index_file, ibase, izone, zonename, zonesize);
        error = cg_zone_type(index_file, ibase, izone, &zonetype);
        //	sendInfo("param: Zones:%d.     Zone#1: name=%s , type=%d",numzones,zonename,zonetype);
        //	sendInfo("param: Sizes for unstructured grid: Verts=%d Elements=%d BoundVerts=%d",zonesize[0],zonesize[1],zonesize[2]);

        /// Making solution array

        int nsols = 0;
        int isol = 1;
        error = cg_nsols(index_file, ibase, izone, &nsols); //check the number of existing solutions
        //sendInfo("param: Solutions: %d",nsols);

        int nfields = 0;
        error = cg_nfields(index_file, ibase, izone, isol, &nfields); //check the number of existing fields
        //			sendInfo("param: Number of fields: %d",nfields);

        char fieldname[100];
        CGNS_ENUMT(DataType_t) fieldtype;

        vector<string> stparams;
        stparams.push_back("None");

        for (int i = 1; i <= nfields; ++i)
        {
            error = cg_field_info(index_file, ibase, izone, isol, i, &fieldtype, fieldname);
            if (error)
            {
                sendError("param: Cannot read solution %d field info", i);
                return;
            }
            stparams.push_back(fieldname);
        }
        //			sendWarning("param: xv=%d ; yv=%d",param_vx->getValue(),param_vy->getValue());

        int parm = -1;

        //sendWarning ("Setting params %s",stparams[0].c_str());
        parm = param_vx->getValue();
        if (parm == -1)
            parm = 0;
        param_vx->setValue(stparams.size(), stparams, parm);
        parm = param_vy->getValue();
        if (parm == -1)
            parm = 0;
        param_vy->setValue(stparams.size(), stparams, parm);
        parm = param_vz->getValue();
        if (parm == -1)
            parm = 0;
        param_vz->setValue(stparams.size(), stparams, parm);

        for (int i = 0; i < 4; ++i)
        {
            parm = param_f[i]->getValue();

            if (parm == -1)
                parm = 0;
            param_f[i]->setValue(stparams.size(), stparams, parm);

            //TODO: Not important. change port description when variable changes.
            //Maybe this has to be done at initialisation, it doesn't work here

            out_float[i]->setInfo("sendinfo"); //Doesn't work!
            //sendWarning("setting port info");
        }

        error = cg_close(index_file);
        //		sendInfo("ReadCGNS::param(): cg_close: error=%d ",error);
    } // cgns file open
    else
    {
        //waring only if param is file name
        if (fileParam == paramName)
            sendWarning("param: Cannot read CGNS file!");
    }
}
void ReadCGNS::postInst()
{
    //param_vx->setValue(param_vx->getValue());
    //sendWarning("postInst: vx=%d  vy=%d  vz=%d",param_vx->getValue(),param_vy->getValue(),param_vz->getValue());
}
/*---------------------------------------------------
 *  int ReadCGNS::read_params(params &p)
 *  Reads UI parameters into p structure
-----------------------------------------------------*/
int ReadCGNS::read_params(params &p)
{
    p.b_load_2d = param_load_2d->getValue();
    p.b_use_string = param_use_string->getValue();
    p.sections_string = param_sections_string->getValue();

    p.param_vx = param_vx->getValue();
    p.param_vy = param_vy->getValue();
    p.param_vz = param_vz->getValue();

    for (int i = 0; i < 4; ++i)
        p.param_f[i] = param_f[i]->getValue();

    return 0;
}

bool ReadCGNS::indexed_file_name(char *s, int n)
{
    if (sprintf(s, param_file->getValue(), n) < 0)
        return false;
    return true;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int ReadCGNS::compute(const char *port)
{
    (void)port;

    params_out();
    // get params
    params p;
    read_params(p);

    int index_file, error; //file handling

    int numtimesteps = 0; //no of timesteps
    int idxlo = 0, idxhi = 0;

    // Decision: animation(timestep>1) or still image (timestep=1)
    if (param_use_file_timesteps->getValue())
    {
        sendInfo("CGNS reader started, reading multiple files");

        idxlo = param_first_file_idx->getValue();
        idxhi = param_last_file_idx->getValue();
        numtimesteps = idxhi - idxlo + 1;
        if ((idxlo < 1) || (idxhi < 1) || (numtimesteps < 1))
        {
            sendError("ReadCGNS::compute(): FATAL ERROR! File indices are wrong, check them.");
            return FAIL;
        }

        // Testing file name string format
        string s;
        s.append(param_file->getValue());
        if ((s.find("%d") == string::npos) && (s.find("%0") == string::npos))
        {
            sendError("ReadCGNS::compute(): FATAL ERROR! Invalid  file name format for timesteps");
            return FAIL;
        }

        // Testing file name sequence
        /*
		char name[300];
		int i_file;
		for (int i=0;i<numtimesteps;++i)
		{
			indexed_file_name(name,idxlo+i);
			sendInfo("%s",name);
			error=cg_open(name,CG_MODE_READ,&i_file);
			if(error)
			{
				sendError("File Can't be opened, error=%d !",error);
				return FAIL;
			}
			error=cg_close(i_file);
			if(error)
			{
				sendError("Error on file closing, error=%d !",error);
				return FAIL;
			}
		}//for
		return FAIL; //debug
		*/

    } // if use_file_timesteps
    else // don't use file timesteps
    {
        sendInfo("CGNS Reader started, reading single file. Input file name: %s ", param_file->getValue());
        error = cg_open(param_file->getValue(), CG_MODE_READ, &index_file);
        if (error)
        {
            sendError("File Can't be opened, error=%d !", error);
            return FAIL;
        }
        sendInfo("File opened with cg_open.");
        //How many bases do we have
        int numbases = 0;
        cg_nbases(index_file, &numbases);
        // I should cycle the bases , but now i'll use the first one.
        sendInfo("Bases: %d.", numbases);
        numtimesteps = numbases;
    }

    // 2. read bases
    if (numtimesteps == 1) //single step&base
    {
        sendWarning("Reading only ONE base");
        //base read starts here
        int ibase = 1; // we've got only 1 base#1---
        base b(index_file, ibase, p); //create base object
        if (b.read() == FAIL) //read base object
        {
            sendError("ReadCGNS::compute(): ERROR! Cannot read the base");
            return FAIL;
        }
        //Creating COVISE objects and ports from arrays
        out_objs bo;
        create_base_objs(b, bo, -1);
        set_output_objs(bo);
    }
    else
    {
        //arrays for coDoSet
        vector<coDistributedObject *> gridobj(numtimesteps + 1);
        vector<vector<coDistributedObject *> > floatobj(4, vector<coDistributedObject *>(numtimesteps + 1));
        vector<coDistributedObject *> velobj(numtimesteps + 1);
        //filling arrays
        for (int timestep = 0; timestep < numtimesteps; ++timestep)
        {
            sendWarning("Reading TIMESTEP %d", timestep);
            //base read starts here
            {
                //for multibase animation
                int ibase = timestep + 1; // -----now we read only base#1---

                if (param_use_file_timesteps->getValue())
                {
                    // file should be opened for each timestep for file_timesteps
                    ibase = 1; //read only first base for file_timesteps

                    char name[300];
                    indexed_file_name(name, idxlo + timestep);
                    error = cg_open(name, CG_MODE_READ, &index_file);
                    if (error)
                    {
                        sendError("File Can't be opened, error=%d, name=%s !", error, name);
                        return FAIL;
                    }
                }

                base b(index_file, ibase, p); //create a base object
                if (b.read() == FAIL) //read the base
                {
                    sendError("ReadCGNS::compute(): ERROR! Cannot read base %d", ibase);
                    return FAIL;
                }

                if (param_use_file_timesteps->getValue())
                {
                    error = cg_close(index_file);
                    if (error)
                    {
                        sendError("Error on file closing, error=%d, index=%d!", error, timestep + idxlo);
                        return FAIL;
                    }
                }

                //for file-based animation ibase=1

                //Creating COVISE objects and ports from arrays
                out_objs bo;
                create_base_objs(b, bo, timestep);

                gridobj[timestep] = bo.gridobj;
                for (int i = 0; i < 4; ++i)
                    floatobj[i][timestep] = bo.floatobj[i];
                velobj[timestep] = bo.velobj;
            } //-----deleting base-----------
        } //for
        //terminating arrays
        gridobj[numtimesteps] = NULL;
        for (int i = 0; i < 4; ++i)
            floatobj[i][numtimesteps] = NULL;
        velobj[numtimesteps] = NULL;

        sendWarning("Creating TIMESTEPS coDoSet");

        out_objs mod_out;
        const char *name;

        string attrval = string("0 ") + int2str(numtimesteps - 1);

        name = out_mesh->getObjName();

        mod_out.gridobj = NULL;
        if (gridobj[0] != NULL)
        {
            mod_out.gridobj = new coDoSet(name, &gridobj[0]);
            mod_out.gridobj->addAttribute("TIMESTEP", attrval.c_str());
            if (mod_out.gridobj == NULL)
                sendError("Cannot create coDoSet!");
        }

        mod_out.velobj = NULL;
        if (velobj[0] != NULL)
        {
            name = out_velocity->getObjName();
            mod_out.velobj = new coDoSet(name, &velobj[0]);
            mod_out.velobj->addAttribute("TIMESTEP", attrval.c_str());
        }

        for (int i = 0; i < 4; ++i)
        {
            mod_out.floatobj[i] = NULL;
            if (floatobj[i][0] != NULL)
            {
                name = out_float[i]->getObjName();
                mod_out.floatobj[i] = new coDoSet(name, &floatobj[i][0]);
                mod_out.floatobj[i]->addAttribute("TIMESTEP", attrval.c_str());
            }
        }
        set_output_objs(mod_out);
    }

    if (!param_use_file_timesteps->getValue()) //closing file for non-timestep
    {
        error = cg_close(index_file);
        sendInfo("cg_close: error=%d ", error);
    }
    //return FAIL;
    return SUCCESS;
}

/*---------------------------------------------------------
 * int ReadCGNS::create_base_objs(base &b,out_objs &objs)
 * Creates distributed objects of a base
 * base 	&b 		-- source base
 * out_objs &objs	-- structure with resulting objects
 * int 		nubmer	-- number to add to name (for output
 *         TIMESTEP coDoSet, (-1) == don't add anything)
 *---------------------------------------------------------*/
int ReadCGNS::create_base_objs(base &b, out_objs &objs, int number)
{
    //creating COVISE objects with correct names
    /////////////////////////
    //creating float object
    for (int i = 0; i < 4; ++i)
    {
        objs.floatobj[i] = NULL;
        const char *floatobjname;

        string str;
        str.append(out_float[i]->getObjName());
        if (number != -1)
            str.append(string("_") + int2str(number));

        floatobjname = str.c_str();

        if (floatobjname)
        {
            sendInfo("ReadCGNS::create_base_objs(): float objname=%s", floatobjname);
            if (b.is_single_zone())
                objs.floatobj[i] = b.create_do(floatobjname, T_FLOAT, i);
            else
                objs.floatobj[i] = b.create_do_set(floatobjname, T_FLOAT, i);
        }
        else
            sendError("Can not set float array to output port!");
    }

    //creating COVISE vector object for velocity

    objs.velobj = NULL;
    const char *velobjname;

    string str;
    str.append(out_velocity->getObjName());
    if (number != -1)
        str.append(string("_") + int2str(number));

    velobjname = str.c_str();
    if (velobjname)
    {
        sendInfo("ReadCGNS::create_base_objs(): vel objname=%s", velobjname);
        if (b.is_single_zone())
            objs.velobj = b.create_do(velobjname, T_VEC3);
        else
            objs.velobj = b.create_do_set(velobjname, T_VEC3);
    }

    //Preparing COVISE object for grid

    //-----creating unstructured grid object and set it for output---
    objs.gridobj = NULL;
    const char *usgobjname;

    str.clear();
    str.append(out_mesh->getObjName());
    if (number != -1)
        str.append(string("_") + int2str(number));

    usgobjname = str.c_str();
    if (usgobjname)
    {
        sendInfo("ReadCGNS::create_base_objs(): usg objname=%s", usgobjname);
        if (b.is_single_zone())
            objs.gridobj = b.create_do(usgobjname, T_GRID);
        else
            objs.gridobj = b.create_do_set(usgobjname, T_GRID);
    }
    return SUCCESS;
}

/*-----------------------------------------------
 * int ReadCGNS::set_output_objs(out_objs &objs)
 * Sets objs.* output objects to output ports
 *----------------------------------------------*/
int ReadCGNS::set_output_objs(out_objs &objs)
{
    out_mesh->setCurrentObject(objs.gridobj);
    out_velocity->setCurrentObject(objs.velobj);
    for (int i = 0; i < 4; ++i)
        out_float[i]->setCurrentObject(objs.floatobj[i]);
    return SUCCESS;
}

/*==============================================
  void ReadCGNS::params_out()
 Writes dialog params to stdout. For debug
==============================================*/
void ReadCGNS::params_out()
{
    cout << "========Dialog params================" << endl;
    cout << "vx_param=" << param_vx->getValue() << ";  vy_param=" << param_vy->getValue() << " vz_param=" << param_vz->getValue() << endl;
    cout << "field1=" << param_f[0]->getValue() << endl;
    cout << "field2=" << param_f[1]->getValue() << endl;
    cout << "field3=" << param_f[2]->getValue() << endl;
    cout << "field4=" << param_f[3]->getValue() << endl;
    cout << "======================================" << endl;
}

/////////////////////
/// BASE CLASS
////////////////////

/*-----------------------------------
 * base::base(int i_file,int i_base,params _p)
 * constructor
 * i_file -- opened cgns file handle
 * i_base -- base no
 * _p -- parameters structure
 *-------------------------------------*/
base::base(int i_file, int i_base, params _p)
{
    p = _p;
    index_file = i_file;
    ibase = i_base;

    // read(); good only without error handling
}

/*---------------------------------------------------
 * int base::read()
 * reads base from file
 *---------------------------------------------------*/
int base::read()
{
    if (zones.size() != 0) // if the base has been already read (that must not happen!)
    {
        zones.clear();
        CoviseBase::sendWarning("base::read(): WARNING! Reading the base that's already read, clearing zones.");
    }

    cg_base_read(index_file, ibase, basename, &cell_dim, &phys_dim);
    CoviseBase::sendInfo("base::read(): name=%s, cell_dim= %d, phys_dim=%d ", basename, cell_dim, phys_dim);

    //3. read zone info
    //=============zones================
    int numzones = 0;

    //for the first zone

    error = cg_nzones(index_file, ibase, &numzones);
    if (error)
    {
        CoviseBase::sendError("nzones error=%d", error);
        return FAIL;
    }

    CoviseBase::sendInfo("base::read(): Zones:%d. ", numzones);

    //	zones.push_back(zone(index_file,ibase,1,p));
    //	zones[0].read();

    //============read zones(temp!!!)
    for (int i = 0; i < numzones; ++i)
    {
        zones.push_back(zone(index_file, ibase, i + 1, p));
        if (zones[i].read() == FAIL)
        {
            CoviseBase::sendError("base::read(): ERROR! Cannot read zone no %d", i);
            return FAIL;
        }
    }
    return SUCCESS;
}

/*--------------------------------------------------------
 *  coDoSet *base::create_do_set()
 * Creates coDoSet for multizone base
 *
 * params are similar to zone::create_do():
 * type -- data type {T_GRID,T_VEC3, T_FLOAT}
 * scal_no -- number of scalar float field for T_FLOAT
 *-----------------------------------------------------*/
coDoSet *base::create_do_set(const char *name, int type, int scal_no)
{

    cout << cout_green << "base::create_do_set(): type=" << type << "; scal_no=" << scal_no << endl;
    cout << "======CREATING coDoSet========" << cout_norm << endl;

    vector<coDistributedObject *> d_obj(zones.size() + 1);

    string s;

    for (int i = 0; i < zones.size(); ++i)

    {
        s = string(name) + "_doset_" + int2str(i);
        cout << s << endl;
        d_obj[i] = zones[i].create_do(s.c_str(), type, scal_no);
    }
    d_obj[zones.size()] = NULL;

    if (d_obj[0] == NULL)
    {
        cout << cout_red << "base::create_do_set(): ERROR! First DO ptr is NULL." << cout_norm << endl;
        return NULL;
    }

    coDoSet *do_set = new coDoSet(name, &d_obj[0]);

    return do_set;
}
/*------------------------------------------------------------
 * coDistributedObject *base::create_do (const char *name,int type, int scal_no=0)
 *
 * Creates COVISE distributed object of a given type for 1st zone of the base
 * int  -- can be {T_GRID, T_VEC3, T_FLOAT}
 * int scal -- number of scalar field from 0 to 3 for t_FLOAT
 *-------------------------------------------------------------*/
coDistributedObject *base::create_do(const char *name, int type, int scal_no)
{
    return zones[0].create_do(name, type, scal_no);
}

/*--------------------------------------------
 *  bool base::single_zone()
 * returns true if the base has only one zone
 *--------------------------------------------*/
bool base::is_single_zone()
{
    return zones.size() == 1;
}

MODULE_MAIN(Examples, ReadCGNS)
