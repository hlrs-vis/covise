/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * zone.cpp
 *
 *	CGNS Zone reader
 *
 *  Created on: 25.01.2010
 *      Author: Vlad
 */

#include "zone.h"

// compatibilty with ubuntu system CGNS
#ifndef CGNS_ENUMT
#define CGNS_ENUMT(e) e
#endif
#ifndef CGNS_ENUMV
#define CGNS_ENUMV(e) e

typedef int cgsize_t;
#endif

/*------------------------------
 *  Constructor
 *-----------------------------*/

zone::zone(int i_file, int i_base, int i_zone, params _p)
{
    index_file = i_file;
    ibase = i_base;
    izone = i_zone;
    p = _p;
}

/*-----------------------------------------------------
 * int zone::read()
 *
 * reads a zone and fills internal arrays with data
 * arrays must be empty!
 *---------------------------------------------------*/
int zone::read()
{
    error = cg_zone_read(index_file, ibase, izone, zonename, zonesize);
    error = cg_zone_type(index_file, ibase, izone, &zonetype);

    CoviseBase::sendInfo(" zone::read() started: name=%s , type=%d, num=%d", zonename, zonetype, izone);

    cout << cout_cyan << cout_underln << "==============zone::read() started======================" << endl
         << "zone::read() name=" << zonename << ", type=" << zonetype << ", no=" << izone << cout_norm << endl;

    cout << "Sizes for unstructured grid: Verts=" << zonesize[0] << ", Elements=" << zonesize[1] << "BoundVerts=" << zonesize[2] << endl;

    //4,5  Reading coords

    fx.insert(fx.begin(), zonesize[0], 0);
    fy.insert(fy.begin(), zonesize[0], 0);
    fz.insert(fz.begin(), zonesize[0], 0);

    //cout << "LALALALALAL!!!!!!!!!!!!!!!!!!!!!!!"<<endl<<fx.size()<<"="<<zonesize[0]<<endl;
    if ((error = read_coords(fx, fy, fz)))
        return FAIL;

    //6. Reading Grid data (Sections)
    // Elements are the parts of geometry (grids and so on) which refers to coords array
    cout << "zone::read(): ====================Reading grid Sections=================" << endl;

    int numsections = 0;
    error = cg_nsections(index_file, ibase, izone, &numsections);
    cout << "zone::read(): Number of sections:" << numsections << endl;

    vector<int> sections; //array of selected section indices
    select_sections(sections);

    int sectionsread = 0; //How many sections we have read?
    for (int ind = 0; ind < sections.size(); ++ind) // for_each ?
    {
        int isection = sections[ind];
        sectionsread++;
        if ((error = read_one_section(isection, conn, elem, tl)))
            return FAIL;
    }
    cout << "zone::read():" << sectionsread << " sections read.";
    if (sectionsread == 0)
    {
        CoviseBase::sendError("zone::read(): Can not read any section!");
        cout << cout_red << cout_underln << "zone::read(): Can not read any section!" << cout_norm << endl;
        return FAIL;
    }

    // connectivity indices should be decremented after loading
    for (int i = 0; i < conn.size(); ++i)
        conn[i]--; //decrementing for C++ Covise indexes from zero
    cout << "zone::read(): Connectivities after decrement: conn[0]=" << conn[0] << ", conn[9]=" << conn[8] << ", conn[" << (int)conn.size() - 1 << "]=" << conn.back() << endl;

    //7. Reading Solution

    cout << "zone::read():===============Reading Solution=================" << endl;

    int nsols = 0;
    error = cg_nsols(index_file, ibase, izone, &nsols); //check the number of existing solutions
    cout << "zone::read(): Solutions: " << nsols << endl;

    int isol = 1; //now I know that I have got 1 solution
    char solname[100];
    CGNS_ENUMT(GridLocation_t) solloc;
    error = cg_sol_info(index_file, ibase, izone, isol, solname, &solloc);
    cout << "zone::read(): Solution #1: Name= " << solname << ", Location=" << solloc << endl;

    int nfields = 0;
    error = cg_nfields(index_file, ibase, izone, isol, &nfields); //check the number of existing fields
    cout << "zone::read(): Number of fields: " << nfields << endl;

    //Loading all scalar float fields

    //is solution is  vertex based, load from rmin to rmax (=zonesize[3])

    int solmin = 0, solmax = 0;

    switch (solloc)
    {
    case CGNS_ENUMV(Vertex): //vertex based
        solmax = fx.size();
        solmin = 1;
        cout << "zone::read(): Solution is vertex based" << endl;
        break;
    case CGNS_ENUMV(CellCenter): //cell based
        solmin = 1; //universal
        solmax = tl.size();
        cout << "zone::read(): Solution is cell based" << endl;
        break;

    default:
        CoviseBase::sendError("zone::read(): Solution location type %d is not supported", (int)solloc);
        cout << cout_red << cout_underln << "zone::read(): Solution location type " << (int)solloc << " is not supported" << cout_norm << endl;
        return FAIL;
    }

    //scalars load
    // loads in vector <float> scalar[4];

    for (int i = 0; i < 4; ++i)
    {
        scalar[i].insert(scalar[i].begin(), solmax - solmin + 1, 0);
        //check!  field load func. needs right size of arrays.
        int error = 0;
        error = read_field(isol, p.param_f[i], solmin, solmax, scalar[i]);
        if (error)
        {
            cout << cout_magenta << "zone::read(): WARNING! cannot read scalar field no " << i + 1 << "; clearing array" << cout_norm << endl;
            scalar[i].clear();
            //break;// return FAIL;
        }
    }

    //velocity load
    cout << "zone::read():----------Reading velocity-----------" << endl;

    error = 0; //temp!

    fvx.insert(fvx.begin(), solmax - solmin + 1, 0);
    fvy.insert(fvy.begin(), solmax - solmin + 1, 0);
    fvz.insert(fvz.begin(), solmax - solmin + 1, 0);

    //CHECK  Important! field load func. needs right size of arrays!

    int vector_error = 0;

    error = read_field(isol, p.param_vx, solmin, solmax, fvx);
    if (error)
        vector_error = 1;

    error = read_field(isol, p.param_vy, solmin, solmax, fvy);
    if (error)
        vector_error = 2;

    error = read_field(isol, p.param_vz, solmin, solmax, fvz);
    if (error)
        vector_error = 3;

    if (vector_error)
    {
        cout << cout_magenta << "zone::read(): WARNING! vector field array no " << vector_error << " cannot be loaded!" << endl;
        cout << "zone::read(): Clearing all 3 vector arrays." << cout_norm << endl;

        fvx.clear();
        fvy.clear();
        fvz.clear();
    }

    return SUCCESS;
}

/*-------------------------------------------------------------------------------------
 * int zone::read_coords(int index_file,int ibase,int izone, //OBS!
 * vector <float> &fx, vector <float> &fy, vector <float> &fz)
 *
 * Reads 3 first coord arrays in zone into 3 float vectors of size=zonesize[0].
 * Datatypes must be the same now for every coord array.
 *
 * index_file,ibase,izone -- obsolete, not used now!
 * vectors -- float vectors with size=zonesize[0] (number of points). Don't use empty vectors!
 */
int zone::read_coords(vector<float> &fx, vector<float> &fy, vector<float> &fz)
{
    //=========GridCoordinates_t=========
    int error = 0;
    int numgrids = 0, igrid = 1; //i know that I have 1 "grid"
    char gridname[100];

    cout << cout_cyan << "zone::read_coords():==========read_coords started================" << cout_norm << endl;
    //???????
    /*
		 * Then I should use these functions to determine presence of coordinate arrays blocks in zone
		 * there is one such block in my file: "GridCoordinates"
		 * cg_ngrids(file,base,zone, int *ngrids); -- number of grid coord arrays block
		 * cg_grid_read(f,b,z,ngr, char *gridcoordname); -- name of specific grid arrays block
		 *
		 * maybe this needed if there are many grid coord blocks
		 *
		 * UPDATE: I don't need igrid (grid No) to load coord arrays at all! it's strange.
		 */
    error = cg_ngrids(index_file, ibase, izone, &numgrids);
    error = cg_grid_read(index_file, ibase, izone, igrid, gridname);
    cout << "zone::read_coords(): Ngrids=" << numgrids << ".  Gridcoordinates_t #1 name:" << gridname << endl;

    //5. =================Reading the coordinates===============

    int ncoords = 0;

    // determine the number of coord arrays
    cg_ncoords(index_file, ibase, izone, &ncoords);
    cout << "zone::read_coords(): Ncoord=" << ncoords << endl;

    if (ncoords < 3) //need at least 3 arrays
    {
        CoviseBase::sendError("zone::read_coords(): ERROR! Zone has less than 3 coordinate arrays!");
        return FAIL;
    }

    //determine datatypes and names of coord arrays
    vector<CGNS_ENUMT(DataType_t)> datatypes;
    vector<string> coordnames;

    for (int icoord = 1; icoord <= ncoords; ++icoord)
    {
        char coordname[100];
        CGNS_ENUMT(DataType_t) datatype;
        cg_coord_info(index_file, ibase, izone, icoord, &datatype, coordname);
        coordnames.push_back((string)coordname);

        datatypes.push_back(datatype);
        cout << "zone::read_coords(): Datatype=" << (int)datatypes[icoord - 1] << " for coord No " << icoord << ", name=" << coordnames[icoord - 1] << endl;
    }

    cgsize_t rmin = 1; //minimal CGNS index of coord array
    //int rmax=zonesize[0]; //maximal CGNS index of coord array
    cgsize_t rmax = fx.size(); // vector size MUST be = zonesize(0)

    //Datatypes must be the same now for every coord array; function reads 3 first arrays as X,Y,Z

    switch (datatypes[0])
    {
    case CGNS_ENUMV(RealDouble):
    {
        vector<double> x(fx.size()); //=zonesize[0]
        vector<double> y(fy.size());
        vector<double> z(fz.size());

        error = cg_coord_read(index_file, ibase, izone, coordnames[0].c_str(), CGNS_ENUMV(RealDouble), &rmin, &rmax, &x[0]);
        error = cg_coord_read(index_file, ibase, izone, coordnames[1].c_str(), CGNS_ENUMV(RealDouble), &rmin, &rmax, &y[0]);
        error = cg_coord_read(index_file, ibase, izone, coordnames[2].c_str(), CGNS_ENUMV(RealDouble), &rmin, &rmax, &z[0]);

        copy(x.begin(), x.end(), fx.begin());
        copy(y.begin(), y.end(), fy.begin());
        copy(z.begin(), z.end(), fz.begin());
    }
    break;
    case CGNS_ENUMV(RealSingle):
        cout << "zone::read_coords(): Reading Float type coordinates:" << endl;
        error = cg_coord_read(index_file, ibase, izone, coordnames[0].c_str(), CGNS_ENUMV(RealSingle), &rmin, &rmax, &fx[0]);
        error = cg_coord_read(index_file, ibase, izone, coordnames[1].c_str(), CGNS_ENUMV(RealSingle), &rmin, &rmax, &fy[0]);
        error = cg_coord_read(index_file, ibase, izone, coordnames[2].c_str(), CGNS_ENUMV(RealSingle), &rmin, &rmax, &fz[0]);
        break;
    default:
        CoviseBase::sendError("zone::read_coords(): type %d is not supported.", (int)datatypes[0]);
        cout << cout_red << cout_underln << "zone::read_coord(): type " << (int)datatypes[0] << "is not supported" << cout_norm << endl;
        return FAIL;
    }
    //The library can load float type coords into double type array
    // I don't know why

    if (error)
    {
        CoviseBase::sendError("zone::read_coords(): Cannot read coordinates");
        cout << cout_red << cout_underln << "zone::read_coord(): Cannot read coordinates" << cout_norm << endl;
        return FAIL;
    }

    cout << "read_coords: X [0]=" << fx[0] << "; [" << (int)fx.size() - 1 << "]=" << fx.back() << endl;
    cout << "read_coords: Y [0]=" << fy[0] << "; [" << (int)fy.size() - 1 << "]=" << fy.back() << endl;
    cout << "read_coords: Z [0]=" << fz[0] << "; [" << (int)fz.size() - 1 << "]=" << fz.back() << endl;

    return SUCCESS;
}

/*------------------------------------------------------------------------------------
 * int zone::read_one_section(int index_file, int ibase, int izone, // obs.!
 * int isection, vector <int> &conn, vector <int> &elem, vector <int> &tl)
 *
 * Reads an USG section and INSERTS it into 3 vectors (which can be not empty)
 *
 * index_file,ibase,izone,isection -- as usual for CGNS functions
 *  conn, elem, tl -- vectors to INSERT an USG section
 *------------------------------------------------------------------------------------*/

int zone::read_one_section(int isection, vector<int> &conn, vector<int> &elem, vector<int> &tl)
{

    char elemsectname[100];
    int bdry = 0, parentflg = 0;
    cgsize_t start = 0, end = 0, eldatasize = 0;
    CGNS_ENUMV(ElementType_t) etype;

    //Reading Section (reading Connectivity, then creating Type list and element list)
    error = cg_section_read(index_file, ibase, izone, isection, elemsectname, &etype, &start, &end, &bdry, &parentflg);
    error = cg_ElementDataSize(index_file, ibase, izone, isection, &eldatasize);

    cout << cout_cyan << "zone::read_one_section() started: Section#" << isection << ", Name=" << elemsectname << ", type=" << etype << ", start=" << start << ", end=" << end << ", boundary=" << bdry << ", parentflg=" << parentflg << " eldatasize=" << eldatasize << cout_norm << endl;

    //int parentdata;
    // reading connectivity array

    size_t connfirst = conn.size(); //Index of first conn. element (in this iteration) for tl and elem creation.

    vector<cgsize_t> conntemp; // temporary array for cleaned connectivity for this iteration
    conntemp.insert(conntemp.begin(), eldatasize, 0);
    error = cg_elements_read(index_file, ibase, izone, isection, &conntemp[0], NULL);

    // creating elements and types array for COVISE unstructured grid
    switch (etype)
    {
    case CGNS_ENUMV(HEXA_8): //=17

        tl.insert(tl.end(), end - start + 1, TYPE_HEXAEDER); //all elements are hexaeders
        for (int i = 0; i < end - start + 1; i++)
            elem.push_back(connfirst + 8 * i); //all elements (hexaeders) are of size 8
        conn.insert(conn.end(), conntemp.begin(), conntemp.end());
        cout << "zone::read_one_section(): Section " << isection << " has CG_HEXA_8 type, sizes=tl:" << (int)tl.size() << " elem:" << (int)elem.size() << endl;
        break;
    case CGNS_ENUMV(TETRA_4): //=10
        tl.insert(tl.end(), end - start + 1, TYPE_TETRAHEDER);
        for (int i = 0; i < end - start + 1; ++i)
            elem.push_back(connfirst + 4 * i); // indices begin from connfirst
        conn.insert(conn.end(), conntemp.begin(), conntemp.end());
        cout << "zone::read_one_section(): Section " << isection << " has CG_TETRA_4 type, sizes=tl:" << (int)tl.size() << " elem:" << (int)elem.size() << endl;
        break;
    case CGNS_ENUMV(QUAD_4):
        tl.insert(tl.end(), end - start + 1, TYPE_QUAD);
        for (int i = 0; i < end - start + 1; ++i)
            elem.push_back(connfirst + 4 * i); // indices begin from connfirst
        conn.insert(conn.end(), conntemp.begin(), conntemp.end());
        cout << "zone::read_one_section(): Section " << isection << " has CG_QUAD_4 type, sizes=tl:" << (int)tl.size() << " elem:" << (int)elem.size() << endl;
        break;
    case CGNS_ENUMV(TRI_3):
        tl.insert(tl.end(), end - start + 1, TYPE_TRIANGLE);
        for (int i = 0; i < end - start + 1; ++i)
            elem.push_back(connfirst + 3 * i); // indices begin from connfirst
        conn.insert(conn.end(), conntemp.begin(), conntemp.end());
        cout << "zone::read_one_section(): Section " << isection << " has CG_TRI_3 type, sizes=tl:" << (int)tl.size() << " elem:" << (int)elem.size() << endl;
        break;
    case CGNS_ENUMV(MIXED): // =20 ; now loads with conn. type member deletion
    {
        std::cout << "zone::read_one_section(): MIXED grid load begins" << endl;
        cgsize_t i = 0; //working with conntemp array from begin
        cgsize_t erases = 0; //amount of erased element types in conn. array

        while (i < conntemp.size())
        {
            switch (conntemp[i]) //cycling through element_type connectivity array elements
            {
            case CGNS_ENUMV(HEXA_8): //CG_HEXA_8 = 17
                tl.push_back(TYPE_HEXAEDER); //pushing one more element_type into type list
                erases++;
                conn.insert(conn.end(), &conntemp[i + 1], (&conntemp[i + 1]) + 8);
                elem.push_back(connfirst + i + 1 - erases); //pushing 1st element member (cleaned conn) index into element list
                i += 9; // 8+1					// goto next element_type connectivity element
                break;
            case CGNS_ENUMV(TETRA_4): //CG_TETRA_4 = 10
                tl.push_back(TYPE_TETRAHEDER);
                erases++;
                conn.insert(conn.end(), &conntemp[i + 1], (&conntemp[i + 1]) + 4);
                elem.push_back(connfirst + i + 1 - erases);
                i += 5; //4+1
                break;
            case CGNS_ENUMV(PYRA_5): //CG_PYRA_5 = 12
                tl.push_back(TYPE_PYRAMID);
                erases++;
                conn.insert(conn.end(), &conntemp[i + 1], (&conntemp[i + 1]) + 5);
                elem.push_back(connfirst + i + 1 - erases);
                i += 6;
                break;
            case CGNS_ENUMV(PENTA_6): //CG_PENTA_6=14
                tl.push_back(TYPE_PRISM);
                erases++;
                conn.insert(conn.end(), &conntemp[i + 1], (&conntemp[i + 1]) + 6);
                elem.push_back(connfirst + i + 1 - erases);
                i += 7;
                break;
            //	2D primitives
            case CGNS_ENUMV(TRI_3): //=5
                tl.push_back(TYPE_TRIANGLE);
                erases++;
                conn.insert(conn.end(), &conntemp[i + 1], (&conntemp[i + 1]) + 3);
                elem.push_back(connfirst + i + 1 - erases);
                i += 4;
                break;
            case CGNS_ENUMV(QUAD_4): //=7
                tl.push_back(TYPE_QUAD);
                erases++;
                conn.insert(conn.end(), &conntemp[i + 1], (&conntemp[i + 1]) + 4);
                elem.push_back(connfirst + i + 1 - erases);
                i += 5;
                break;

            default:
                CoviseBase::sendError("zone::read_one_section(): Element of mixed unstructured grid is not supported: %ld, i=%ld", (long)conntemp[i], (long)i);
                cout << cout_red << cout_underln << "zone::read_one_section(): Element of mixed unstructured grid is not supported: " << conntemp[i] << ", i=" << i << cout_norm << endl;
                return FAIL;
            } //switch conntemp
        } //while conntemp
        cout << "erases=" << erases << "  conntemp.size=" << conntemp.size()
             << "conn.size =" << conn.size() << endl;

        cout << "zone::read_one_section(): Section " << isection << " has MIXED type, sizes=tl:" << (int)tl.size() << " elem:" << (int)elem.size() << endl;
    } // case section_type
    break;

    default:
        CoviseBase::sendError("read_one_section: Loading section %d ERROR: Element type number %d not supported", isection, etype);
        cout << cout_red << cout_underln << "zone::read_one_section(): Loading section  " << isection << "ERROR: Element type number " << etype << " is not supported." << cout_norm << endl;
        return FAIL;
    }
    //conntemp.clear();
    return SUCCESS;
}

//----------------------------------------------------------------------------------------
//int zone::select_sections(int index_file,int ibase,int izone,params &p, //this params are obsolete
//                         vector <int> &sections)
//
// Selects grid sections that we want to load
// returns vector <int> &section -- array of section indices to load
//----------------------------------------------------------------------------------------

int zone::select_sections(vector<int> &sections)
{
    int numsections = 0;

    vector<string> sectnames;

    cout << cout_cyan << "zone::select_sections() started ===========================" << cout_norm << endl;

    error = cg_nsections(index_file, ibase, izone, &numsections);
    cout << "zone::select_sections() Number of sections:" << numsections << endl;

    if (p.b_use_string) //using sections string
    {
        cout << "zone::select_sections(): Using section string" << endl;
        string paramstr; //,tempstr;
        paramstr = p.sections_string;

        cout << "zone::select_sections(): string= " << paramstr << endl;

        size_t i;
        int isection;
        while ((i = paramstr.find_first_of(" ,")) != string::npos)
        {
            istringstream tempstr(paramstr.substr(0, i));
            paramstr = paramstr.substr(i + 1);

            if (tempstr >> isection)
            {
                sections.push_back(isection);
                sectnames.push_back(int2str(isection)); //temp!!!
            }
            else
            {
                CoviseBase::sendError("select_sections: Cannot convert sections string!");
                return FAIL;
            }
        } //while
    } //use sections string
    else //loading only 2d or only 3d
    {
        bool need2d = p.b_load_2d;
        if (need2d)
            cout << "zone::select_sections(): Loading only 2D meshes" << endl;
        else
            cout << "select_sections: Loading only 3D meshes" << endl;

        for (int isection = 1; isection <= numsections; ++isection)
        {

            char elemsectname[100];
            int bdry = 0, parentflg = 0;
            cgsize_t start = 0, end = 0;
            CGNS_ENUMT(ElementType_t) etype;

            //Reading Section (reading Connectivity, then creating Type list and element list)
            error = cg_section_read(index_file, ibase, izone, isection, elemsectname, &etype, &start, &end, &bdry, &parentflg);
            string s = elemsectname;
            switch (etype)
            {
            case CGNS_ENUMV(TRI_3):
                if (need2d)
                {
                    sections.push_back(isection);
                    sectnames.push_back(s);
                }
                break;
            case CGNS_ENUMV(QUAD_4):
                if (need2d)
                {
                    sections.push_back(isection);
                    sectnames.push_back(s);
                }
                break;
            case CGNS_ENUMV(TETRA_4):
                if (!need2d)
                {
                    sections.push_back(isection);
                    sectnames.push_back(s);
                }
                break;
            case CGNS_ENUMV(HEXA_8):
                if (!need2d)
                {
                    sections.push_back(isection);
                    sectnames.push_back(s);
                }
                break;
            case CGNS_ENUMV(MIXED):
            {
                vector<cgsize_t> conntemp; // have to load the array to determine if the section is 2d if 3d

                cgsize_t eldatasize = 0;
                error = cg_ElementDataSize(index_file, ibase, izone, isection, &eldatasize);
                conntemp.insert(conntemp.begin(), eldatasize, 0);
                error = cg_elements_read(index_file, ibase, izone, isection, &conntemp[0], NULL);
                if (error)
                {
                    CoviseBase::sendError("zone::select_sections(): Cannot load MIXED-type array!");
                    return FAIL;
                }

                bool is3d = IsMixed3D(conntemp);
                if (!is3d && p.b_load_2d)
                {
                    sections.push_back(isection);
                    sectnames.push_back(s);
                    cout << "zone::select_sections(): Mixed Section " << isection << " is 2D , adding ..." << endl;
                }
                if ((is3d) && (!p.b_load_2d))
                {
                    sections.push_back(isection);
                    sectnames.push_back(s);
                    cout << "zone::select_sections(): Mixed Section " << isection << " is 3D , adding ..." << endl;
                }
                conntemp.clear(); //don't needed???
            }
            break;
            default:
                CoviseBase::sendError("zone::select_sections(): ERROR: Element %d type number %d not supported", isection, etype);
                return FAIL;
            } //switch etype
        } //for
    }
    cout << "zone::select_section():SELECTED SECTIONS-------------------------\n";
    for (int i = 0; i < sections.size(); ++i)
    {
        cout << sections[i] << ": " << sectnames[i] << "\n";
    }
    cout << "total  " << sections.size() << " sections selected.\n-----end of zone::select_sections--------------------" << endl;

    return SUCCESS;
}

//------------------------------------------------
// bool zone::IsMixed3D (const vector <int> &conntemp)
//
// Checks if mixed grid section has CG_HEXA_8, CG_TETRA_4, CG_PYRA_5 or CG_PENTA_6 elements
// conntemp -- MIXED-type CGNS section array
// Also returns false if there is an element of not supported type
//-------------------------------------------------

bool zone::IsMixed3D(const vector<cgsize_t> &conntemp)
{
    cgsize_t i = 0;
    while (i < conntemp.size())
    {
        switch (conntemp[i]) //cycling through element_type connectivity array elements
        {
        case CGNS_ENUMV(HEXA_8): //CG_HEXA_8 = 17
            return true;
            i += 9; // 8+1	?????? don't need int this function
            break;
        case CGNS_ENUMV(TETRA_4): //CG_TETRA_4 = 10
            return true;
            i += 5; //4+1
            break;
        case CGNS_ENUMV(PYRA_5): //CG_PYRA_5 = 12
            return true;
            i += 6;
            break;
        case CGNS_ENUMV(PENTA_6): //CG_PENTA_6=14
            return true;
            //i+=7;
            break;
        //	2D primitives
        case CGNS_ENUMV(TRI_3): //=5
            i += 4;
            break;
        case CGNS_ENUMV(QUAD_4): //=7
            i += 5;
            break;

        default:
            CoviseBase::sendError("IsMixed3D: Element of mixed unstructured grid is not supported: %ld, i=%ld", (long)conntemp[i], (long)i);
            return false;
        } //switch conntemp
    } //while conntemp
    return false;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++          Solution reading functions              +++
//+++                                                  +++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//  for_each() function
/*

struct func_copyd2f // for STL for_each
{
	const double *dbl;
	float *flt;
	func_copyd2f(const double *_dbl, float *_flt): dbl(_dbl),flt(_flt) {}
	void operator () (int i) {flt[i]=dbl[i];}
};
*/

/*========================================================================
 * Loading the ifield plot field
 * params:
 *
 * most parameters are from cg_* functions
 *
 * isol -- solution number
 * ifield -- float field to load
 *
 * start,end -- CGNS grid element or point indices: start (usually 1) and end (usually number of elements or points) indices
 *
 * Return value:
 * 0 at success
 * -1 if isol=0 or solution type is not supported
 * cgns error value on reading error
 *=========================================================================*/
int zone::read_field(int isol, int ifield, cgsize_t start, cgsize_t end, vector<float> &fvar)
{

    char fieldname[100];
    CGNS_ENUMT(DataType_t) fieldtype;
    int error = 0;

    cout << cout_cyan << "zone::read_field() started. zone=" << izone << " field=" << ifield << cout_norm << endl;

    //Get field info
    error = cg_field_info(index_file, ibase, izone, isol, ifield, &fieldtype, fieldname);

    if (ifield == 0) //Null solution field index selected in user interface
    {
        cout << cout_magenta << "zone::read_field(): WARNING: Attempt to load NULL solution, skipping..." << cout_norm << endl;
        return -1;
    }

    if (error)
    {
        cout << cout_red << "zone::read_field(): WARNING: Cannot load field no. " << ifield << " ; check properties dialog. " << cout_norm << endl;
        ;
        return error;
    }
    cout << "zone::read_field(): Reading field " << ifield << "; type=" << fieldtype << ";name = " << fieldname << endl;

    //read field
    switch (fieldtype)
    {
    case CGNS_ENUMV(RealDouble): //When datatype is double, read in double array and copy to float
    {
        vector<double> var(end - start + 1);
        error = cg_field_read(index_file, ibase, izone, isol, fieldname, fieldtype, &start, &end, &var[0]);
        cout << "zone::read_field(DOUBLE): " << fieldname << "[" << start - 1 << "]=" << var[start - 1] << " , " << fieldname << "[" << end - 1 << "]=" << var[end - 1] << endl;
        //for_each(var.begin(),var.end(),func_copyd2f(&var[0],&fvar[0])); //STL copy with for_each
        if (error == 0)
            copy(var.begin(), var.end(), fvar.begin()); //STL copy with "copy"
        else
        {
            cout << cout_red << "zone::read_field(): Cannot load field, possibly bad field id in module params." << cout_norm << endl;
        }
    }
    break;
    case CGNS_ENUMV(RealSingle): //when datatype is float, simply read the data into float array
        error = cg_field_read(index_file, ibase, izone, isol, fieldname, fieldtype, &start, &end, &fvar[0]);
        cout << "zone::read_field(FLOAT): " << fieldname << "[" << start - 1 << "]=" << fvar[start - 1] << " , " << fieldname << "[" << end - 1 << "]=" << fvar[end - 1] << endl;
        if (error != 0)
        {
            cout << cout_red << "zone::read_field(): Cannot load field, possibly bad field id in module params." << cout_norm << endl;
        }

        break;
    default:
        cout << cout_red << "read_field: Array Datatype is not supported" << cout_norm << endl;
        return -1;
    }

    return error;
}

/////////////////////////////////////////////////////////
//  Distribured object creation
////////////////////////////////////////////////////////

/*------------------------------------------------------------
 * coDistributedObject *zone::create_do (const char *name,int type, int scal_no=0)
 *
 * Creates COVISE distributed object of a given type
 * int  -- can be {T_GRID, T_VEC3, T_FLOAT}
 * int scal -- number of scalar field from 0 to 3
 *-------------------------------------------------------------*/
coDistributedObject *zone::create_do(const char *name, int type, int scal_no)
{
    coDistributedObject *d_obj;
    switch (type)
    {
    case T_GRID:
        d_obj = create_do_grid(name);
        break;
    case T_FLOAT:
        d_obj = create_do_scalar(name, scal_no);
        break;
    case T_VEC3:
        d_obj = create_do_vec(name);
        break;
    default:
        cout << cout_red << "zone::create_do(): ERROR! Invalid object type." << cout_norm << endl;
        return NULL;
    }
    return d_obj;
}

/*---------------------------------------------------------------------
 * coDoUnstructuredGrid *zone::create_do_grid (const char *usgobjname)
 *
 * creates COVISE grid object
 * const char * usgobjname -- object name
 *----------------------------------------------------------------------*/
coDoUnstructuredGrid *zone::create_do_grid(const char *usgobjname)
{
    coDoUnstructuredGrid *gridobj;
    gridobj = new coDoUnstructuredGrid(usgobjname, elem.size(), conn.size(), zonesize[0],
                                       &elem[0], &conn[0], &fx[0], &fy[0], &fz[0], &tl[0]);
    return gridobj;
}

/*-------------------------------------------------------
 * coDoVec3 *zone::create_do_vec(const char *velobjname)
 *
 * creates COVISE float vector object
 * const char * usgobjname -- object name
 *-------------------------------------------------------*/
coDoVec3 *zone::create_do_vec(const char *velobjname)
{
    if (fvx.size() == 0)
    {
        cout << cout_magenta << "zone::create_do_vec(): WARNING! Empty VECTOR array, returning NULL. "
             << "Zone=" << izone << "; size=" << fvx.size() << cout_norm << endl;
        return NULL;
    }
    coDoVec3 *velobj;
    velobj = new coDoVec3(velobjname, fvx.size(), &fvx[0], &fvy[0], &fvz[0]);
    return velobj;
}
/*---------------------------------------------------------
 * coDoFloat *zone::create_do_scalar(const char *floatobjname, int n)
 *
 * creates COVISE float object
 * const char * usgobjname -- object name
 * n -- number of scalar field (0-3)
 */
coDoFloat *zone::create_do_scalar(const char *floatobjname, int n)
{
    if ((n < 0) || (n > 3))
    {
        cout << "\e[31;4m zone::create_do_scalar(): ERROR! scalar index is out of range! \e[0m" << endl;
        return NULL;
    }
    if (scalar[n].size() == 0)
    {
        cout << cout_magenta << "zone::create_do_scalar(): WARNING! Empty SCALAR array, returning NULL;"
             << "Zone=" << izone << " no=" << n + 1 << "; size=" << scalar[n].size() << cout_norm << endl;

        return NULL;
    }
    coDoFloat *floatobj = NULL;
    floatobj = new coDoFloat(floatobjname, scalar[n].size(), &scalar[n][0]);

    //cout<<"zone::create_do_scalar(): zone="<<izone<<" no="<<n+1<<"; size="<<scalar[n].size()<<"; floatobj="<<floatobj<<endl;
    return floatobj;
}
