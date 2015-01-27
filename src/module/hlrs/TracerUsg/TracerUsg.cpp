/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/**************************************************************************\ 
**                                                           (C)1996 RUS  **
**                                                                        **
** Description: COVISE Tracer3_USG create streamlines 			  **
**		of unstructured grids using adjacency information 	  **
**              (single block)                                            **
**		Uses 2nd order and 4th order integration with adaptive    **
**		stepsize control. Adjacency information is computed in	  **
**		a pre process and stored in a list (memory expensive)	  **
**		or computed on the fly(slower)				  **
**		Contains possibility of point reduction on output and	  **
**									  **
**		Input: geometrie field, data field (vector)		  **
**		Output: Streamlines 			 		  **
**			Datafield (vorticity of line (scalar))		  **
**			Datafield (vorticity of line (vector))		  **
**                                                                        **
**                                                                        **
**                                                                        **
**                             (C) 1996                                   **
**                Computer Center University of Stuttgart                 **
**                            Allmandring 30                              **
**                            70550 Stuttgart                             **
**                                                                        **
**                                                                        **
** Author:  Oliver Heck                                                   **
**                                                                        **
**                                                                        **
** Date:  26.03.96  V1.0  						  **
**                                                                        **
** 								          **
\**************************************************************************/

//
// Covise include stuff
//
#include "TracerUsg.h"
#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <float.h>

#ifdef _WIN32
#include <math.h>
#include <stdio.h>
#endif

#define MAX_PIB 100
#define NUM_STEPS 10.0

// functions
//
void rk2_tet(float[], float[], float[], float[], float[],
             float[], float[], float[], float[], float, float[]);
void rk4_tet(float[], float[], float[], float[], float[],
             float[], float[], float[], float[], float *, float[]);
void vort_tet(const float[], const float[], const float[], const float[], const float[],
              const float[], const float[], const float[], const float[], float[]);
void hex2tet(int ind, int i, int *tel, int *tcl);
void pyra2tet(int ind, int i, int *tel, int *tcl);
void prism2tet(int ind, int i, int *tel, int *tcl);
void tracer(int start, float *startp, int *no_points, ia<float> &x_out, ia<float> &y_out, ia<float> &z_out,
            ia<float> &v_out, ia<float> &v_x_out, ia<float> &v_y_out, ia<float> &v_z_out, int *no_vert, ia<int> &vert_list, int startpointNo);

void print_adjacency();

int create_cnbr(int, int *, int *, int *, int **, int **);
int create_single_cnbr(int, int, int *, int *, int *, int **);
int isin_tetra(const float[], const float[], const float[], const float[], const float[]);
int compare(const void *, const void *);
int comparef(const void *, const void *);
void intersect(ia<int> &Xref, ia<int> &Yref, ia<int> &Zref, ia<int> &inbox);
int find_startcell(float *);
int find_startcell_fast(float *);
int find_actualcell(int, float *);
float stepsize(int m, float px[3]);
float tetra_vol(const float[], const float[], const float[], const float[]);
//   float p_ang(float[], float[], float[]);
float dout(const float[]);
float Min_of(float[], int);
float Max_of(float[], int);
void typicalDim(float *);
int *loopedArray;
float loopDelta;
float loopDelta2;
long MaxPoints;
float Speed_thres;
int Connectivity;
//   int lookForCellMethod;
float MaxSearchFactor;

double p_ang(float[], float[], float[]);
int read_velocity(int actual_cell, const float *px, float *velocity);
int check_diff(const float *old_velocity, const float *new_velocity);
int ExpandSetList(const coDistributedObject *const *objects, int h_many, const char *type,
                  ia<const coDistributedObject *> &out_array);

inline float sqr(float val)
{
    val *= val;
    return val;
}

// works out a determinant
inline float vol3Vect(const float *p0, const float *p1, const float *p2)
{
    float res;
    res = (*p0) * (*(p1 + 1)) * (*(p2 + 2));
    res += (*(p0 + 1)) * (*(p1 + 2)) * (*(p2));
    res += (*(p0 + 2)) * (*p1) * (*(p2 + 1));
    res -= (*p0) * (*(p1 + 2)) * (*(p2 + 1));
    res -= (*(p0 + 2)) * (*(p1 + 1)) * (*(p2));
    res -= (*(p0 + 1)) * (*p1) * (*(p2 + 2));
    return res;
}

// global variables
ListNeighbourhood cuc_list;
static int actual_cell_type;
float trace_eps = 0.00001f;
float trace_len = 0.00001f;
int *el, *cl, *tl;
int *ctl; // cell type list from coDoIntArr
CellPair **amil; // arbitrary mesh interface list
int intpol_dir, set_num_elem, option, stp_control; //get choice of interpolation method
int reduce, data_out_type;
int numelem, numconn, numcoord;
long numstart, numstart_min, numstart_max;
int trace_num;
int startStyle;
int box_flag, opt2 /*,  opt2_old*/;
int onceIsEnoughInit;
int onceIsEnoughFactor;
const char *color[10] = {
    "yellow", "green", "blue", "red", "violet", "chocolat",
    "linen", "pink", "crimson", "indigo"
};

//  adjacency vars
int cuc_count, *gpt_nbr_count = NULL;
int *cuc = NULL, *cuc_pos = NULL;
int cnbr_count;
int *cnbr = NULL, *cnbr_pos = NULL;
; //cnbr -> cellneighbours

float *x_in, *y_in, *z_in; // positions data
float *u_in, *v_in, *w_in; // vorticity data
float *ct_in; // cell type data
float startp[3], startpoint1[3], startpoint2[3]; // startpoint
float /* startNormal[3],*/ startDirection[3];
float box_length = 0.;

const char *dtype, *gtype, *gtype_test; // data type, grid type
char *GridIn, *DataIn;
// Not used    char *colorn,
char *startpoint;
char *ameshPath;

int notASet;
// block handling
int NoOfBlocks;
int CurrentBlock;
ia<int> SetElementChecked;

// new search method
Application::MinMaxList XList(Application::X);
Application::MinMaxList YList(Application::Y);
Application::MinMaxList ZList(Application::Z);

//  Shared memory data
coDoIntArr *cell_type_in = NULL;
ia<const coDistributedObject *> grid_in_list;
ia<const coDistributedObject *> uv_data_in_list;
ia<const coDistributedObject *> v_data_in_list;

int main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();
    return 0;
}

//======================================================================
// Called before module exits
//======================================================================
void Application::quit(void *)
{
    //
    // ...... delete your data here .....
    //
    Covise::log_message(__LINE__, __FILE__, "Quitting now");
}

//
// static stub callback functions calling the real class
// member functions
//

void Application::quitCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->quit(callbackData);
}

void Application::computeCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->compute(callbackData);
}

//
//
//..........................................................................
//
//

void CellPair::addCellPair(CellPair *other)
{
    int *tmp = new int[NoOfAMesh + other->NoOfAMesh];
    int i;
    for (i = 0; i < NoOfAMesh; i++)
        tmp[i] = listOfAMesh[i];
    for (i = NoOfAMesh; i < NoOfAMesh + other->NoOfAMesh; i++)
        tmp[i] = other->listOfAMesh[i - NoOfAMesh];
    delete[] listOfAMesh;
    listOfAMesh = tmp;
    NoOfAMesh += other->NoOfAMesh;
    if (NoOfCells + other->NoOfCells > 0)
    {
        tmp = new int[((NoOfCells + other->NoOfCells) / CellBlocks + 1) * CellBlocks];
        for (i = 0; i < NoOfCells; i++)
            tmp[i] = listOfCells[i];
        for (i = NoOfCells; i < NoOfCells + other->NoOfCells; i++)
            tmp[i] = other->listOfCells[i - NoOfCells];
        delete[] listOfCells;
        listOfCells = tmp;
        NoOfCells += other->NoOfCells;
        delete other;
    }
}

void CellPair::addCell(int other)
{
    if ((NoOfCells % CellBlocks) == 0)
    {
        cerr << "adding new chunk\n";
        int *tmp = new int[NoOfCells + CellBlocks];
        for (int i = 0; i < NoOfCells; i++)
            tmp[i] = listOfCells[i];
        delete[] listOfCells;
        listOfCells = tmp;
    }
    listOfCells[NoOfCells] = other;
    NoOfCells++;
}

class CellTreeNode // a tree that holds all cellPairs
{
private:
    CellTreeNode *left;
    CellTreeNode *right;
    CellPair *data;

public:
    CellTreeNode(CellPair *d)
        : left(NULL)
        , right(NULL)
        , data(d){};
    void addNode(CellPair *d)
    {
        if (data->cellNo > d->cellNo)
        {
            if (left)
                left->addNode(d);
            else
                left = new CellTreeNode(d);
        }
        else if (data->cellNo < d->cellNo)
        {
            if (right)
                right->addNode(d);
            else
                right = new CellTreeNode(d);
        }
        else if (data->cellNo == d->cellNo)
        {
            cerr << "adding equal node: " << d->cellNo << endl;
            data->addCellPair(d);
        }
    };
    int findMax()
    {
        if (right)
            return right->findMax();
        else
            return data->cellNo;
    }
    void fillArray(CellPair **arr) // converts tree into an array with direct indices
    {
        if (left)
            left->fillArray(arr);
        arr[data->cellNo] = data;
        cerr << "filling " << data->cellNo << endl;
        if (right)
            right->fillArray(arr);
        return;
    }
};

int Application::createAmeshList(char *aMeshPath)
{
    static char *actPath = NULL;
    FILE *hdl;
    int i, c1, c2;
    CellPair *cell;
    CellTreeNode *rootCellTypeList;

    if (actPath)
        if (strcmp(ameshPath, actPath) == 0)
            return NoOfArbMesh;

    actPath = aMeshPath;

    NoOfArbMesh = 0;
    hdl = fopen(actPath, "r");
    if (hdl == NULL)
        return 0;

    int iret = fscanf(hdl, "%d %d", &c1, &c2);
    if (iret != 2)
        cerr << "createAmeshList: wrong no. of arguments, read " << iret << endl;
    cell = new CellPair(c1, c2);
    rootCellTypeList = new CellTreeNode(cell);

    while (fscanf(hdl, "%d %d", &c1, &c2) == 2)
    {
        cell = new CellPair(c1, c2);
        rootCellTypeList->addNode(cell);
        cell = new CellPair(c2, c1);
        rootCellTypeList->addNode(cell);
    }

    NoOfArbMesh = rootCellTypeList->findMax();
    cerr << "max = " << NoOfArbMesh << endl;
    ArbMeshIntList = new CellPair *[NoOfArbMesh];
    for (i = 0; i < NoOfArbMesh; i++)
        ArbMeshIntList[i] = NULL;
    rootCellTypeList->fillArray(ArbMeshIntList);

    for (i = 0; i < numelem; i++)
    {
        if (ctl[i] > NoOfArbMesh)
            cerr << "ctl[" << i << "] too high: " << ctl[i] << endl;
        if (ArbMeshIntList[ctl[i]])
            ArbMeshIntList[ctl[i]]->addCell(i);
    }

    amil = ArbMeshIntList;

    return NoOfArbMesh;
}

void Application::compute(void *)
{
    ia<const coDistributedObject *> returnObject;
    //      Distributed Objects
    const coDistributedObject *data_obj, *in[3];
    coDoLines *lineobj = NULL;
    const coDistributedObject *const *in_set_elements = NULL;
    char *Mesh, *Data;

    //      local variables
    int i, j, k;
    int data_anz;
    int no_points, no_vert;
    int t_no_points = 0, t_no_vert = 0, t_no_lines = 0;

    ia<int> vert_list(SMALL_GRID), t_vert_list(SMALL_GRID);

    int *t_line_list;
    int direction;
    int startcell_count, *startcell, *startblock, *startp_ind;

    ia<float> s_out(SMALL_GRID);
    ia<float> t_s_out(SMALL_GRID);
    ia<float> x_out(SMALL_GRID);
    ia<float> y_out(SMALL_GRID);
    ia<float> z_out(SMALL_GRID);
    ia<float> t_x_out(SMALL_GRID);
    ia<float> t_y_out(SMALL_GRID);
    ia<float> t_z_out(SMALL_GRID);
    ia<float> vx_out(SMALL_GRID);
    ia<float> vy_out(SMALL_GRID);
    ia<float> vz_out(SMALL_GRID);
    ia<float> t_vx_out(SMALL_GRID);
    ia<float> t_vy_out(SMALL_GRID);
    ia<float> t_vz_out(SMALL_GRID);

    char *Lines, *DataOut, *VelOut;
    char buf[1000];

    //	get input data object names

    Mesh = Covise::get_object_name("meshIn");
    if (Mesh == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for 'meshIn'");
        return;
    }
    in[0] = coDistributedObject::createFromShm(Mesh);

    Data = Covise::get_object_name("dataIn");
    if (Data == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for 'dataIn'");
        return;
    }
    in[1] = coDistributedObject::createFromShm(Data);

    in[2] = 0L;

    //	get output data object	names
    Lines = Covise::get_object_name("lines");
    if (Lines == NULL)
    {
        Covise::sendError("Object name not correct for 'lines'");
        return;
    }

    DataOut = Covise::get_object_name("dataOut");
    if (DataOut == NULL)
    {
        Covise::sendError("Object name not correct for 'dataOut'");
        return;
    }

    VelOut = Covise::get_object_name("velocOut");
    if (VelOut == NULL)
    {
        Covise::sendError("Object name not correct for 'velocOut'");
        return;
    }

    //      get parameters
    //get choice of creation cnbr_list (1:as stored array; 2:on the fly)
    //Covise::get_choice_param("nbrs_comp", &opt2);
    opt2 = 2;

    //get number of startpoints
    Covise::get_slider_param("no_startp", &numstart_min, &numstart_max, &numstart);

    //get startpoints
    for (i = 0; i < 3; i++)
    {
        // 1st start point
        Covise::get_vector_param("startpoint1", i, &startpoint1[i]);
        // 2nd start point
        Covise::get_vector_param("startpoint2", i, &startpoint2[i]);
        //      Covise::get_vector_param( "normal", i, &startNormal[i]);
        Covise::get_vector_param("direction", i, &startDirection[i]);
    }
    Covise::get_browser_param("amesh_path", &ameshPath);

    Covise::get_choice_param("startStyle", &startStyle);
    if (startStyle > 2) // just plane&line supported at the moment
        startStyle = 1;

    //get choice of interpolation method
    Covise::get_choice_param("option", &option);

    //get choice of stepsize control
    Covise::get_choice_param("stp_control", &stp_control);

    //get direction of interpolation
    Covise::get_choice_param("tdirection", &direction);
    intpol_dir = direction;

    //get choice of point reduction on output geometry
    Covise::get_choice_param("reduce", &reduce);

    //get choice of output type
    Covise::get_choice_param("whatout", &data_out_type);

    //get choice of output type
    Covise::get_scalar_param("trace_eps", &trace_eps);
    //get choice of output type
    Covise::get_scalar_param("trace_len", &trace_len);
    loopDelta = 0.0;
    Covise::get_scalar_param("loopDelta", &loopDelta);
    loopDelta2 = loopDelta * loopDelta;

    Covise::get_scalar_param("MaxPoints", &MaxPoints);
    Covise::get_scalar_param("MaxSearchFactor", &MaxSearchFactor);
    Covise::get_scalar_param("Speed_thres", &Speed_thres);
    Covise::get_boolean_param("Connectivity", &Connectivity);
    //	Covise::get_choice_param("searchMethod", &lookForCellMethod);

    //      retrieve grid object from shared memory

    if (DataOut == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for 'dataOut'");
        return;
    }

    if (VelOut == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for 'velocOut'");
        return;
    }

    HandleTimeSteps timeSteps(in, Lines, DataOut, VelOut);

    int co_time;
    onceIsEnoughFactor = 0; // once is enough for warnings even for several time steps
    if (timeSteps.ok())
        for (co_time = 0; co_time < timeSteps.how_many(); co_time++)
        {

            SetElementChecked.schleifen();
            grid_in_list.schleifen();
            uv_data_in_list.schleifen();
            v_data_in_list.schleifen();

            t_no_points = t_no_vert = t_no_lines = 0;

            // Reset names for this time step
            timeSteps.setTime(co_time, in, &Lines, &DataOut, &VelOut);
            // write in in[0], in[1], in[2] the pointers to the relevant
            // objects for this time step

            data_obj = in[0];

            CurrentBlock = 0; // sl: we never know if the CurrentBlock
            //     at the end of a previous calculation
            //     will be meaningful now.

            if (data_obj != 0L)
            {
                gtype = data_obj->getType(); // get the grid type
                if (strcmp(gtype, "SETELE") == 0)
                {
                    notASet = 0;
                    int in_set_len;
                    in_set_elements = ((const coDoSet *)data_obj)->getAllElements(&in_set_len);
                    NoOfBlocks = ExpandSetList(in_set_elements, in_set_len, "UNSGRD", grid_in_list);
                    if (NoOfBlocks)
                    {
                        data_obj = grid_in_list[0];
                        gtype = data_obj->getType();
                    }
                }
                else
                {
                    // we have a simple object
                    notASet = 1;
                    NoOfBlocks = 1;
                }
                // SetElementChecked = new int[NoOfBlocks];
                for (i = 0; i < NoOfBlocks; i++)
                    SetElementChecked[i] = 0;

                if (strcmp(gtype, "UNSGRD") == 0 && NoOfBlocks > 0)
                {
                    grid_in_list[0] = data_obj;
                    coDoUnstructuredGrid *grid_in = (coDoUnstructuredGrid *)(data_obj);
                    grid_in->getGridSize(&numelem, &numconn, &numcoord);
                    grid_in->getAddresses(&el, &cl, &x_in, &y_in, &z_in);
                    grid_in->getTypeList(&tl);
                }
                else
                {
                    Covise::sendError("ERROR: Data object 'meshIn' has wrong data type");
                    return;
                }
            }
            else
            {
                Covise::sendError("ERROR: Data object 'meshIn' can't be accessed in shared memory");
                return;
            }

            if (numelem == 0 || numconn == 0 || numcoord == 0)
            {
                Covise::sendWarning("WARNING: Data object 'meshIn' is empty");
            }

            {
                int i;
                for (i = 0; i < grid_in_list.size(); ++i)
                {
                    if (cuc_list.search(
                            (coDoUnstructuredGrid *)(grid_in_list[i])) == -1)
                    {
                        cuc_list.schleifen(); // this might be improved, but not today
                        break;
                    }
                }
            }

            data_obj = in[1];
            if (data_obj != 0L)
            {
                dtype = data_obj->getType();

                if (strcmp(dtype, "SETELE") == 0)
                {
                    int in_set_len;
                    in_set_elements = ((coDoSet *)data_obj)->getAllElements(&in_set_len);
                    if (ExpandSetList(in_set_elements, in_set_len,
                                      // Try STRVDT
                                      "STRVDT", v_data_in_list) == NoOfBlocks)
                    {
                        data_obj = v_data_in_list[0];
                        dtype = data_obj->getType();
                    }
                    else if (ExpandSetList(in_set_elements, in_set_len,
                                           // Try USTVDT
                                           "USTVDT", uv_data_in_list) == NoOfBlocks)
                    {
                        data_obj = uv_data_in_list[0];
                        dtype = data_obj->getType();
                    }
                    else
                    {
                        Covise::sendError("ERROR: Sets for grid and data have different number of matching elements");
                        return;
                    }
                }

                if (strcmp(dtype, "STRVDT") == 0)
                {
                    v_data_in_list[0] = data_obj;
                    coDoVec3 *v_data_in = (coDoVec3 *)(data_obj);
                    data_anz = v_data_in->getNumPoints();
                    v_data_in->getAddresses(&u_in, &v_in, &w_in);
                }
                else if (strcmp(dtype, "USTVDT") == 0)
                {
                    uv_data_in_list[0] = data_obj;
                    coDoVec3 *uv_data_in = (coDoVec3 *)(data_obj);
                    data_anz = uv_data_in->getNumPoints();
                    uv_data_in->getAddresses(&u_in, &v_in, &w_in);
                }
                else
                {
                    Covise::sendError("ERROR: Data object 'dataIn' has wrong data type");
                    return;
                }
            }
            else
            {
                Covise::sendError("ERROR: Data object 'dataIn' can't be accessed in shared memory");
                return;
            }

            // check dimensions
            if (data_anz != numcoord)
            {
                Covise::sendError("ERROR: Data Object's dimension doesn't match Grid ones");
                return;
            }

            data_obj = in[2];
            amil = 0L;
            if (data_obj != 0L)
            {
                dtype = data_obj->getType();

                if (strcmp(dtype, "SETELE") == 0)
                {
                    in_set_elements = ((coDoSet *)data_obj)->getAllElements(&i);
                    data_obj = in_set_elements[0];
                    dtype = data_obj->getType();
                }

                if (strcmp(dtype, "INTARR") == 0)
                {
                    cell_type_in = (coDoIntArr *)data_obj;
                    data_anz = cell_type_in->getDimension(0);
                    cell_type_in->getAddress(&ctl);
                    createAmeshList(ameshPath);
                }
                else
                {
                    Covise::sendError("ERROR: Data object 'CellTypeIn' has wrong data type");
                    return;
                }
            }
            else
            {
                //Covise::sendWarning("WARNING: Data object 'CellTypeIn' can't be accessed in shared memory");
                //return(returnObject);
            }

            if (data_anz == 0)
            {
                Covise::sendWarning("WARNING: Data object 'CellTypeIn' is empty");
            }

            //	Begin tracing
            if (loopDelta2 > 0.0)
                loopedArray = new int[numstart];
            xStart = new float[numstart];
            yStart = new float[numstart];
            zStart = new float[numstart];
            startcell = new int[numstart];
            startblock = new int[numstart];
            startp_ind = new int[numstart];
            startcell_count = 0;
            t_line_list = new int[(numstart * 4) + 1];
            t_line_list[0] = 0;

            if (numstart < 1)
            {
                Covise::sendError("ERROR:  Number of startpoints is < 1");
                return;
            }

            computeStartPoints();

            //	find the startcells
            // onceIsEnoughInit is reset to 0 for each time step
            for (i = 0, onceIsEnoughInit = 0; i < numstart; i++)
            {
                if (loopDelta2 > 0.0)
                    loopedArray[i] = 0;
                startp[0] = xStart[i];
                startp[1] = yStart[i];
                startp[2] = zStart[i];

                //calling startcell functions
                setCurrentBlock(CurrentBlock); // otherwise find_startcell_fast
                // crashes if the module is run twice
                j = find_startcell_fast(startp);
                if (j == -1)
                {
                }
                else if (j == -2)
                {
                    return;
                }
                else
                {
                    startp_ind[startcell_count] = i;
                    // a cell number
                    startblock[startcell_count] = CurrentBlock;
                    // does not uniquely identify a cell
                    // we need also a block number
                    startcell[startcell_count++] = j;
                }
            }

            onceIsEnoughInit = 1; // do not send messages if a cell is not
            // found except when looking for the initial cells

            if (startcell_count == 0)
            {
                createDummies(Lines, DataOut, VelOut, returnObject, buf);
                Covise::sendInfo("INFO:  No startpoints found");
                timeSteps.addToTArrays(&returnObject[0]);
                continue;
            }

            for (i = 0; i < startcell_count; i++)
            {
                if (direction == 3)
                {

                    intpol_dir = 2;
                    trace_num = i;

                    startp[0] = xStart[startp_ind[i]];
                    startp[1] = yStart[startp_ind[i]];
                    startp[2] = zStart[startp_ind[i]];

                    j = startcell[i];
                    //compute the streamlines
                    setCurrentBlock(startblock[i] /*,naehe_list*/);
                    try
                    {
                        tracer(j, startp, &no_points, x_out, y_out, z_out, s_out,
                               vx_out, vy_out, vz_out, &no_vert, vert_list, i);
                        for (k = 0; k < no_points; k++)
                        {
                            //positions-comp-x
                            t_x_out[t_no_points + no_points - 1 - k] = x_out[k];
                            //positions-comp-y
                            t_y_out[t_no_points + no_points - 1 - k] = y_out[k];
                            //positions-comp-z
                            t_z_out[t_no_points + no_points - 1 - k] = z_out[k];
                            //data-out (scalar)
                            t_s_out[t_no_points + no_points - 1 - k] = s_out[k];
                            //data-out (vector x-comp)
                            t_vx_out[t_no_points + no_points - 1 - k] = vx_out[k];
                            //data-out (vector y-comp)
                            t_vy_out[t_no_points + no_points - 1 - k] = vy_out[k];
                            //data-out (vector z-comp)
                            t_vz_out[t_no_points + no_points - 1 - k] = vz_out[k];
                        }

                        for (k = 0; k < no_vert; k++)
                        {
                            t_vert_list[t_no_vert + k] = vert_list[k] + t_no_points;
                        }
                        t_line_list[t_no_lines] = t_no_vert;
                        t_no_vert += no_vert;
                        t_no_points += no_points;

                        t_no_lines += 1;
                    }
                    catch (...)
                    {
                        Covise::sendError("Could not accomplish the integration");
                        timeSteps.Destroy();
                        return;
                    }
                }

                trace_num = i;

                startp[0] = xStart[startp_ind[i]];
                startp[1] = yStart[startp_ind[i]];
                startp[2] = zStart[startp_ind[i]];

                j = startcell[i];

                //compute the streamlines
                if (direction == 3)
                {
                    intpol_dir = 1;
                }
                if ((loopDelta2 == 0.0) || (loopedArray[i] == 0))
                {
                    setCurrentBlock(startblock[i]);
                    try
                    {
                        tracer(j, startp, &no_points, x_out, y_out, z_out, s_out,
                               vx_out, vy_out, vz_out, &no_vert, vert_list, i);
                        if (intpol_dir == 1)
                        {
                            for (k = 0; k < no_points; k++)
                            {
                                //positions-comp-x
                                t_x_out[t_no_points + k] = x_out[k];
                                //positions-comp-y
                                t_y_out[t_no_points + k] = y_out[k];
                                //positions-comp-z
                                t_z_out[t_no_points + k] = z_out[k];
                                //data-out (scalar)
                                t_s_out[t_no_points + k] = s_out[k];
                                //data-out (vector x-comp)
                                t_vx_out[t_no_points + k] = vx_out[k];
                                //data-out (vector y-comp)
                                t_vy_out[t_no_points + k] = vy_out[k];
                                //data-out (vector z-comp)
                                t_vz_out[t_no_points + k] = vz_out[k];
                            }
                        }
                        else
                        {
                            for (k = 0; k < no_points; k++)
                            {
                                //positions-comp-x
                                t_x_out[t_no_points + no_points - 1 - k] = x_out[k];
                                //positions-comp-y
                                t_y_out[t_no_points + no_points - 1 - k] = y_out[k];
                                //positions-comp-z
                                t_z_out[t_no_points + no_points - 1 - k] = z_out[k];
                                //data-out (scalar)
                                t_s_out[t_no_points + no_points - 1 - k] = s_out[k];
                                //data-out (vector x-comp)
                                t_vx_out[t_no_points + no_points - 1 - k] = vx_out[k];
                                //data-out (vector y-comp)
                                t_vy_out[t_no_points + no_points - 1 - k] = vy_out[k];
                                //data-out (vector z-comp)
                                t_vz_out[t_no_points + no_points - 1 - k] = vz_out[k];
                            }
                        }

                        for (k = 0; k < no_vert; k++)
                        {
                            t_vert_list[t_no_vert + k] = vert_list[k] + t_no_points;
                        }
                        if (direction != 3)
                        {
                            t_line_list[t_no_lines] = t_no_vert;
                            t_no_lines += 1;
                        }
                        t_no_vert += no_vert;
                        t_no_points += no_points;
                    }
                    catch (...)
                    {
                        Covise::sendError("Could not accomplish the integration");
                        timeSteps.Destroy();
                        return;
                    }
                } // end of if(loopedArray[i]==0)
            } // end of for loop over possible trejectories

            // by Lars Frenzel
            if (t_no_lines == 1)
            {
                returnObject[0] = lineobj = new coDoLines(Lines, t_no_points, &t_x_out[0], &t_y_out[0], &t_z_out[0], t_no_vert, &t_vert_list[0], t_no_lines, t_line_list);
                returnObject[1] /*= data_out */ = new coDoFloat(DataOut, t_no_points, &t_s_out[0]);
                returnObject[2] /*= vel_out  */ = new coDoVec3(VelOut, t_no_points, &t_vx_out[0], &t_vy_out[0], &t_vz_out[0]);

                lineobj->addAttribute("COLOR", "red");
                lineobj->addAttribute("TRACER_DATA_OBJECT", DataOut);
                lineobj->addAttribute("TRACER_VEL_OBJECT", VelOut);
                // and attribute for interaction
                // depending on style !!!
                switch (startStyle)
                {
                case 2: // plane
                    sprintf(buf, "P%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
                    break;

                default: // line
                    sprintf(buf, "T%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
                    break;
                }
                lineobj->addAttribute("FEEDBACK", buf);

                //delete	 lineobj;
                //delete	 data_out;
                //delete   	vel_out;
            }
            else // more than one line
            {
                coDoSet *set_lines_out = NULL;

                coDistributedObject **line_elements = NULL;
                coDistributedObject **data_elements = NULL;
                coDistributedObject **vel_elements = NULL;

                // lines
                float *this_x, *this_y, *this_z;
                int this_num_points, this_num_vert;
                int *this_vl, *this_ll;
                int c;

                // scalar-data
                float *this_scalar;

                // vector data
                float *this_vector_x, *this_vector_y, *this_vector_z;

                // malloc
                line_elements = new coDistributedObject *[t_no_lines + 1];
                data_elements = new coDistributedObject *[t_no_lines + 1];
                vel_elements = new coDistributedObject *[t_no_lines + 1];

                line_elements[t_no_lines] = NULL;
                data_elements[t_no_lines] = NULL;
                vel_elements[t_no_lines] = NULL;

                // we malloc far too much but it doesn't matter
                // Antwort: Ach, Du Schlingel. It does matter!!!
                this_x = new float[t_no_points];
                this_y = new float[t_no_points];
                this_z = new float[t_no_points];

                this_vl = new int[t_no_vert];
                this_ll = new int[1];

                this_scalar = new float[t_no_points];
                this_vector_x = new float[t_no_points];
                this_vector_y = new float[t_no_points];
                this_vector_z = new float[t_no_points];

                // create set of lines instead of linestrips
                for (i = 0; i < t_no_lines; i++)
                {

                    // compute current element
                    j = t_line_list[i];
                    if (i == t_no_lines - 1)
                        k = t_no_vert;
                    else
                        k = t_line_list[i + 1];

                    this_ll[0] = 0;
                    this_num_vert = 0;
                    this_num_points = 0;

                    for (c = 0; j < k; j++, c++)
                    {
                        // line
                        this_x[c] = t_x_out[t_vert_list[j]];
                        this_y[c] = t_y_out[t_vert_list[j]];
                        this_z[c] = t_z_out[t_vert_list[j]];

                        this_vl[c] = c;

                        // scalar
                        this_scalar[c] = t_s_out[t_vert_list[j]];

                        // vector
                        this_vector_x[c] = t_vx_out[t_vert_list[j]];
                        this_vector_y[c] = t_vy_out[t_vert_list[j]];
                        this_vector_z[c] = t_vz_out[t_vert_list[j]];

                        this_num_vert++;
                        this_num_points++;
                    }

                    sprintf(buf, "%s_%d", Lines, i);
                    line_elements[i] = new coDoLines(buf, this_num_points, this_x, this_y, this_z, this_num_vert, this_vl, 1, this_ll);
                    line_elements[i]->addAttribute("COLOR", "red");

                    sprintf(buf, "%s_%d", DataOut, i);
                    data_elements[i] = new coDoFloat(buf, this_num_points, this_scalar);

                    line_elements[i]->addAttribute("TRACER_DATA_OBJECT", buf);
                    sprintf(buf, "%s_%d", VelOut, i);
                    vel_elements[i] = new coDoVec3(buf, this_num_points, this_vector_x, this_vector_y, this_vector_z);

                    line_elements[i]->addAttribute("TRACER_VEL_OBJECT", buf);
                }

                // create sets
                returnObject[0] = set_lines_out = new coDoSet(Lines, line_elements);
                returnObject[1] /*=set_data_out */ = new coDoSet(DataOut, data_elements);
                returnObject[2] /*=set_vel_out  */ = new coDoSet(VelOut, vel_elements);

                // attributes
                set_lines_out->addAttribute("COLOR", "red");
                // and attribute for interaction
                // depending on style !!!
                switch (startStyle)
                {
                case 2: // plane
                    sprintf(buf, "P%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
                    break;

                default: // line
                    sprintf(buf, "T%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
                    break;
                }
                set_lines_out->addAttribute("FEEDBACK", buf);

                // clean up
                for (i = 0; i < t_no_lines; i++)
                {
                    delete line_elements[i];
                    delete data_elements[i];
                    delete vel_elements[i];
                }
                delete[] line_elements;
                delete[] data_elements;
                delete[] vel_elements;

                delete[] this_x;
                delete[] this_y;
                delete[] this_z;

                delete[] this_vl;
                delete[] this_ll;

                delete[] this_scalar;
                delete[] this_vector_x;
                delete[] this_vector_y;
                delete[] this_vector_z;
            }

            // delete	 data_obj;

            delete[] t_line_list;

            if (loopDelta2 > 0.0)
                delete[] loopedArray;
            delete[] xStart;
            delete[] yStart;
            delete[] zStart;
            delete[] startcell;
            delete[] startblock;
            delete[] startp_ind;

            timeSteps.addToTArrays(&returnObject[0]);
        } // for over co_time
    delete in[0];
    delete in[1];
    timeSteps.createOutput(returnObject);
    return;
}

//=====================================================================
// compare function for qsort library function
//=====================================================================
int compare(const void *v1, const void *v2)
{

    if (*((int *)v1) < *((int *)v2))
        return -1;
    else if (*((int *)v1) == *((int *)v2))
        return 0;
    else
        return 1;
}

int comparef(const void *v1, const void *v2)
{

    if (*((float *)v1) < *((float *)v2))
        return -1;
    else if (*((float *)v1) == *((float *)v2))
        return 0;
    else
        return 1;
}

//=====================================================================
// function to create cell_neighbors list
//=====================================================================
int create_cnbr(int /*cuc_count*/, int *cuc, int *cuc_pos, int *arr_count, int **arr, int **arr_pos)
{
    int i, j, k, l;
    int arr_val, arr_size, offset, mark, tmp_pos;
    int *pt_nbr_count;
    int tmpl1[400], *tmpl1ind;

    if (tl[0] == TYPE_HEXAEDER)
    {
        arr_size = numelem * 26;
    }
    else
    {
        arr_size = numelem * 72;
    }
    *arr = new int[arr_size];
    *arr_pos = new int[numelem + 1]; //easier handling ->size of cell_nbr=cell_nbr_pos[numelem+1]
    pt_nbr_count = new int[numcoord];

    memset(*arr, -1, arr_size * sizeof(int));
    memset(pt_nbr_count, 0, numcoord * sizeof(int));
    memset(*arr_pos, 0, (numelem + 1) * sizeof(int));

    //create the pt_nbr_countlist
    for (i = 0; i < numconn; i++)
    {
        pt_nbr_count[cl[i]]++;
    }

    for (i = 0; i < numelem; i++)
    {
        memset(tmpl1, -1, 400 * sizeof(int));
        tmpl1ind = tmpl1;
        tmp_pos = 0;

        //find the offset in cl-list
        offset = UnstructuredGrid_Num_Nodes[tl[i]];

        //put the cells from every point in a single cell
        for (j = 0; j < offset; j++)
        {
            for (k = 0; k < pt_nbr_count[cl[el[i] + j]]; k++)
            {
                *(tmpl1ind++) = cuc[cuc_pos[cl[el[i] + j]] + k];
            }
        }

        l = 0;
        mark = 0;
        while (mark != -1)
        {
            if (tmpl1[l] == i)
                tmpl1[l] = -2;
            mark = tmpl1[l++];
        }
        //sort(tmpl1,0,l-1);
        qsort(tmpl1, l - 1, sizeof(int), compare);

        //find the beginning in the list
        l = 0;
        while (tmpl1[l] == -2)
            l++;

        //fill the cellnbr and cell_nbr_pos lists
        arr_val = -1;
        while (tmpl1[l] != -1)
        {
            if (tmpl1[l] != arr_val)
            {
                if ((*arr_pos)[i] + tmp_pos > arr_size)
                {
                    Covise::sendInfo("INFO:  Not enough memory allocated for cnbr-list -> realloc memory");
                    //realloc arr with 1.5 * old arr_size
                    arr_size = arr_size + arr_size / 2 + 1;
                    *arr = (int *)realloc(*arr, arr_size * sizeof(int));
                    if (*arr == NULL)
                    {
                        Covise::sendError("ERROR:  Allocation of more memory failed!");
                        return 0;
                    }
                }
                (*arr)[(*arr_pos)[i] + tmp_pos++] = tmpl1[l];
                (*arr_pos)[i + 1] = (*arr_pos)[i] + tmp_pos;
                arr_val = tmpl1[l];
                l++;
            }
            else
                l++;
        }
    }
    *arr_count = (*arr_pos)[numelem + 1];
    delete[] pt_nbr_count;
    return 1;
}

//=====================================================================
// function to convert hexaeder to tetraeder
//=====================================================================
void hex2tet(int ind, int i, int *tel, int *tcl)
{
    int j;

    // fill the tel list
    for (j = 0; j < 5; j++)
    {
        tel[j] = j * 4;
    }

    // fill the tcl list
    if (ind > 0)
    {
        //positive decomposition
        tcl[0] = cl[el[i]];
        tcl[1] = cl[el[i] + 4];
        tcl[2] = cl[el[i] + 5];
        tcl[3] = cl[el[i] + 7];

        tcl[4] = cl[el[i]];
        tcl[5] = cl[el[i] + 5];
        tcl[6] = cl[el[i] + 2];
        tcl[7] = cl[el[i] + 7];

        tcl[8] = cl[el[i]];
        tcl[9] = cl[el[i] + 1];
        tcl[10] = cl[el[i] + 2];
        tcl[11] = cl[el[i] + 5];

        tcl[12] = cl[el[i]];
        tcl[13] = cl[el[i] + 2];
        tcl[14] = cl[el[i] + 3];
        tcl[15] = cl[el[i] + 7];

        tcl[16] = cl[el[i] + 2];
        tcl[17] = cl[el[i] + 5];
        tcl[18] = cl[el[i] + 6];
        tcl[19] = cl[el[i] + 7];
    }
    else if (ind < 0)
    {
        //negative decomposition
        tcl[0] = cl[el[i] + 3];
        tcl[1] = cl[el[i] + 4];
        tcl[2] = cl[el[i] + 6];
        tcl[3] = cl[el[i] + 7];

        tcl[4] = cl[el[i] + 1];
        tcl[5] = cl[el[i] + 3];
        tcl[6] = cl[el[i] + 4];
        tcl[7] = cl[el[i] + 6];

        tcl[8] = cl[el[i] + 1];
        tcl[9] = cl[el[i] + 2];
        tcl[10] = cl[el[i] + 3];
        tcl[11] = cl[el[i] + 6];

        tcl[12] = cl[el[i]];
        tcl[13] = cl[el[i] + 1];
        tcl[14] = cl[el[i] + 3];
        tcl[15] = cl[el[i] + 4];

        tcl[16] = cl[el[i] + 1];
        tcl[17] = cl[el[i] + 4];
        tcl[18] = cl[el[i] + 5];
        tcl[19] = cl[el[i] + 6];
    }

    return;
}

//=====================================================================
// function to convert pyramid to tetraheder
//=====================================================================
void pyra2tet(int ind, int i, int *tel, int *tcl)
{
    // fill the tel list
    tel[0] = 0;
    tel[1] = 4;

    // fill the tcl list
    if (ind > 0)
    {
        //positive decomposition
        tcl[0] = cl[el[i]];
        tcl[1] = cl[el[i] + 1];
        tcl[2] = cl[el[i] + 3];
        tcl[3] = cl[el[i] + 4];

        tcl[4] = cl[el[i] + 1];
        tcl[5] = cl[el[i] + 2];
        tcl[6] = cl[el[i] + 3];
        tcl[7] = cl[el[i] + 4];
    }
    else if (ind < 0)
    {
        //negative decomposition
        tcl[0] = cl[el[i]];
        tcl[1] = cl[el[i] + 1];
        tcl[2] = cl[el[i] + 2];
        tcl[3] = cl[el[i] + 4];

        tcl[4] = cl[el[i]];
        tcl[5] = cl[el[i] + 2];
        tcl[6] = cl[el[i] + 3];
        tcl[7] = cl[el[i] + 4];
    }

    return;
}

//=====================================================================
// function to convert prism to tetraheder
//=====================================================================
void prism2tet(int ind, int i, int *tel, int *tcl)
{
    int j;

    // fill the tel list
    for (j = 0; j < 3; j++)
    {
        tel[j] = j * 4;
    }

    // fill the tcl list
    if (ind > 0)
    {
        //positive decomposition
        tcl[0] = cl[el[i]];
        tcl[1] = cl[el[i] + 1];
        tcl[2] = cl[el[i] + 4];
        tcl[3] = cl[el[i] + 5];

        tcl[4] = cl[el[i]];
        tcl[5] = cl[el[i] + 1];
        tcl[6] = cl[el[i] + 3];
        tcl[7] = cl[el[i] + 5];

        tcl[8] = cl[el[i] + 1];
        tcl[9] = cl[el[i] + 2];
        tcl[10] = cl[el[i] + 3];
        tcl[11] = cl[el[i] + 5];
    }
    else if (ind < 0)
    {
        //negative decomposition
        tcl[0] = cl[el[i] + 2];
        tcl[1] = cl[el[i] + 3];
        tcl[2] = cl[el[i] + 4];
        tcl[3] = cl[el[i] + 5];

        tcl[4] = cl[el[i]];
        tcl[5] = cl[el[i] + 1];
        tcl[6] = cl[el[i] + 2];
        tcl[7] = cl[el[i] + 4];

        tcl[8] = cl[el[i]];
        tcl[9] = cl[el[i] + 2];
        tcl[10] = cl[el[i] + 3];
        tcl[11] = cl[el[i] + 4];
    }

    return;
}

//=====================================================================
// find the startcell (sequentiell)
//=====================================================================
int find_startcell(float *p)
{
    int i, j;
    int tmp_el[5], tmp_cl[20];

    float p0[3], p1[3], p2[3], p3[3];

    // look for cell
    for (i = 0; i < numelem; i++)
    {
        switch (tl[i])
        {

        case TYPE_HEXAEDER:
        {
            hex2tet(1, i, tmp_el, tmp_cl);
            for (j = 0; j < 5; j++)
            {
                p0[0] = x_in[tmp_cl[tmp_el[j]]];
                p0[1] = y_in[tmp_cl[tmp_el[j]]];
                p0[2] = z_in[tmp_cl[tmp_el[j]]];

                p1[0] = x_in[tmp_cl[tmp_el[j] + 1]];
                p1[1] = y_in[tmp_cl[tmp_el[j] + 1]];
                p1[2] = z_in[tmp_cl[tmp_el[j] + 1]];

                p2[0] = x_in[tmp_cl[tmp_el[j] + 2]];
                p2[1] = y_in[tmp_cl[tmp_el[j] + 2]];
                p2[2] = z_in[tmp_cl[tmp_el[j] + 2]];

                p3[0] = x_in[tmp_cl[tmp_el[j] + 3]];
                p3[1] = y_in[tmp_cl[tmp_el[j] + 3]];
                p3[2] = z_in[tmp_cl[tmp_el[j] + 3]];

                if (isin_tetra(p, p0, p1, p2, p3) == 1)
                    return i;
            }
        }
        break;

        case TYPE_PRISM:
        {
            prism2tet(1, i, tmp_el, tmp_cl);
            for (j = 0; j < 3; j++)
            {
                p0[0] = x_in[tmp_cl[tmp_el[j]]];
                p0[1] = y_in[tmp_cl[tmp_el[j]]];
                p0[2] = z_in[tmp_cl[tmp_el[j]]];

                p1[0] = x_in[tmp_cl[tmp_el[j] + 1]];
                p1[1] = y_in[tmp_cl[tmp_el[j] + 1]];
                p1[2] = z_in[tmp_cl[tmp_el[j] + 1]];

                p2[0] = x_in[tmp_cl[tmp_el[j] + 2]];
                p2[1] = y_in[tmp_cl[tmp_el[j] + 2]];
                p2[2] = z_in[tmp_cl[tmp_el[j] + 2]];

                p3[0] = x_in[tmp_cl[tmp_el[j] + 3]];
                p3[1] = y_in[tmp_cl[tmp_el[j] + 3]];
                p3[2] = z_in[tmp_cl[tmp_el[j] + 3]];

                if (isin_tetra(p, p0, p1, p2, p3) == 1)
                    return i;
            }
        }
        break;

        case TYPE_PYRAMID:
        {
            pyra2tet(1, i, tmp_el, tmp_cl);
            for (j = 0; j < 2; j++)
            {
                p0[0] = x_in[tmp_cl[tmp_el[j]]];
                p0[1] = y_in[tmp_cl[tmp_el[j]]];
                p0[2] = z_in[tmp_cl[tmp_el[j]]];

                p1[0] = x_in[tmp_cl[tmp_el[j] + 1]];
                p1[1] = y_in[tmp_cl[tmp_el[j] + 1]];
                p1[2] = z_in[tmp_cl[tmp_el[j] + 1]];

                p2[0] = x_in[tmp_cl[tmp_el[j] + 2]];
                p2[1] = y_in[tmp_cl[tmp_el[j] + 2]];
                p2[2] = z_in[tmp_cl[tmp_el[j] + 2]];

                p3[0] = x_in[tmp_cl[tmp_el[j] + 3]];
                p3[1] = y_in[tmp_cl[tmp_el[j] + 3]];
                p3[2] = z_in[tmp_cl[tmp_el[j] + 3]];

                if (isin_tetra(p, p0, p1, p2, p3) == 1)
                    return i;
            }
        }
        break;

        case TYPE_TETRAHEDER:
        {
            p0[0] = x_in[cl[el[i]]];
            p0[1] = y_in[cl[el[i]]];
            p0[2] = z_in[cl[el[i]]];

            p1[0] = x_in[cl[el[i] + 1]];
            p1[1] = y_in[cl[el[i] + 1]];
            p1[2] = z_in[cl[el[i] + 1]];

            p2[0] = x_in[cl[el[i] + 2]];
            p2[1] = y_in[cl[el[i] + 2]];
            p2[2] = z_in[cl[el[i] + 2]];

            p3[0] = x_in[cl[el[i] + 3]];
            p3[1] = y_in[cl[el[i] + 3]];
            p3[2] = z_in[cl[el[i] + 3]];

            if (isin_tetra(p, p0, p1, p2, p3) == 1)
                return i;
        }
        break;

        default:
            return -2;
        }
    }
    return -1;
}

//=====================================================================
// find the actualcell (sequentiell)
//=====================================================================
int find_actualcell(int m, float *head)
{
    int i, j;
    int tmp_el[5], tmp_cl[20];

    float p0[3], p1[3], p2[3], p3[3];
    actual_cell_type = 1;
    // look for cell
    if (opt2 != 2)
    {
        Covise::sendError("ERROR:  Wrong option");
        return -2;
        //	exit(0);
    }
    else
    {
        cuc_list.retrieve((coDoUnstructuredGrid *)grid_in_list[CurrentBlock],
                          &cuc_count, &cuc, &cuc_pos, &gpt_nbr_count);
        if (create_single_cnbr(m, cuc_count, cuc, cuc_pos, &cnbr_count, &cnbr) != 1)
        {
            Covise::sendError("ERROR:  In creating neighbor information for single cell");
            return -2;
        }
        for (i = 0; i < cnbr_count; i++)
        {
            switch (tl[cnbr[i]])
            {

            case TYPE_HEXAEDER:
            {
                hex2tet(1, cnbr[i], tmp_el, tmp_cl);
                for (j = 0; j < 5; j++)
                {
                    p0[0] = x_in[tmp_cl[tmp_el[j]]];
                    p0[1] = y_in[tmp_cl[tmp_el[j]]];
                    p0[2] = z_in[tmp_cl[tmp_el[j]]];

                    p1[0] = x_in[tmp_cl[tmp_el[j] + 1]];
                    p1[1] = y_in[tmp_cl[tmp_el[j] + 1]];
                    p1[2] = z_in[tmp_cl[tmp_el[j] + 1]];

                    p2[0] = x_in[tmp_cl[tmp_el[j] + 2]];
                    p2[1] = y_in[tmp_cl[tmp_el[j] + 2]];
                    p2[2] = z_in[tmp_cl[tmp_el[j] + 2]];

                    p3[0] = x_in[tmp_cl[tmp_el[j] + 3]];
                    p3[1] = y_in[tmp_cl[tmp_el[j] + 3]];
                    p3[2] = z_in[tmp_cl[tmp_el[j] + 3]];

                    if (isin_tetra(head, p0, p1, p2, p3) == 1)
                        return cnbr[i];
                }
            }
            break;

            case TYPE_PRISM:
            {
                prism2tet(1, cnbr[i], tmp_el, tmp_cl);
                for (j = 0; j < 3; j++)
                {
                    p0[0] = x_in[tmp_cl[tmp_el[j]]];
                    p0[1] = y_in[tmp_cl[tmp_el[j]]];
                    p0[2] = z_in[tmp_cl[tmp_el[j]]];

                    p1[0] = x_in[tmp_cl[tmp_el[j] + 1]];
                    p1[1] = y_in[tmp_cl[tmp_el[j] + 1]];
                    p1[2] = z_in[tmp_cl[tmp_el[j] + 1]];

                    p2[0] = x_in[tmp_cl[tmp_el[j] + 2]];
                    p2[1] = y_in[tmp_cl[tmp_el[j] + 2]];
                    p2[2] = z_in[tmp_cl[tmp_el[j] + 2]];

                    p3[0] = x_in[tmp_cl[tmp_el[j] + 3]];
                    p3[1] = y_in[tmp_cl[tmp_el[j] + 3]];
                    p3[2] = z_in[tmp_cl[tmp_el[j] + 3]];

                    if (isin_tetra(head, p0, p1, p2, p3) == 1)
                        return cnbr[i];
                }
            }
            break;

            case TYPE_PYRAMID:
            {
                pyra2tet(1, cnbr[i], tmp_el, tmp_cl);
                for (j = 0; j < 2; j++)
                {
                    p0[0] = x_in[tmp_cl[tmp_el[j]]];
                    p0[1] = y_in[tmp_cl[tmp_el[j]]];
                    p0[2] = z_in[tmp_cl[tmp_el[j]]];

                    p1[0] = x_in[tmp_cl[tmp_el[j] + 1]];
                    p1[1] = y_in[tmp_cl[tmp_el[j] + 1]];
                    p1[2] = z_in[tmp_cl[tmp_el[j] + 1]];

                    p2[0] = x_in[tmp_cl[tmp_el[j] + 2]];
                    p2[1] = y_in[tmp_cl[tmp_el[j] + 2]];
                    p2[2] = z_in[tmp_cl[tmp_el[j] + 2]];

                    p3[0] = x_in[tmp_cl[tmp_el[j] + 3]];
                    p3[1] = y_in[tmp_cl[tmp_el[j] + 3]];
                    p3[2] = z_in[tmp_cl[tmp_el[j] + 3]];

                    if (isin_tetra(head, p0, p1, p2, p3) == 1)
                        return cnbr[i];
                }
            }
            break;

            case TYPE_TETRAHEDER:
            {
                p0[0] = x_in[cl[el[cnbr[i]]]];
                p0[1] = y_in[cl[el[cnbr[i]]]];
                p0[2] = z_in[cl[el[cnbr[i]]]];

                p1[0] = x_in[cl[el[cnbr[i]] + 1]];
                p1[1] = y_in[cl[el[cnbr[i]] + 1]];
                p1[2] = z_in[cl[el[cnbr[i]] + 1]];

                p2[0] = x_in[cl[el[cnbr[i]] + 2]];
                p2[1] = y_in[cl[el[cnbr[i]] + 2]];
                p2[2] = z_in[cl[el[cnbr[i]] + 2]];

                p3[0] = x_in[cl[el[cnbr[i]] + 3]];
                p3[1] = y_in[cl[el[cnbr[i]] + 3]];
                p3[2] = z_in[cl[el[cnbr[i]] + 3]];

                if (isin_tetra(head, p0, p1, p2, p3) == 1)
                    return cnbr[i];
            }
            break;

            default:
                return -2;
            }
        }
        // This is apparently a repetition, but note
        // the subtle but important difference
        // that now the first argument of the hex2tet (etc)
        // functions is -1 instead of 1!!!!
        // Not all readers guarantee that the element
        // orientation is correct!!!!
        actual_cell_type = -1;
        for (i = 0; i < cnbr_count; i++)
        {
            switch (tl[cnbr[i]])
            {

            case TYPE_HEXAEDER:
            {
                hex2tet(actual_cell_type, cnbr[i], tmp_el, tmp_cl);
                for (j = 0; j < 5; j++)
                {
                    p0[0] = x_in[tmp_cl[tmp_el[j]]];
                    p0[1] = y_in[tmp_cl[tmp_el[j]]];
                    p0[2] = z_in[tmp_cl[tmp_el[j]]];

                    p1[0] = x_in[tmp_cl[tmp_el[j] + 1]];
                    p1[1] = y_in[tmp_cl[tmp_el[j] + 1]];
                    p1[2] = z_in[tmp_cl[tmp_el[j] + 1]];

                    p2[0] = x_in[tmp_cl[tmp_el[j] + 2]];
                    p2[1] = y_in[tmp_cl[tmp_el[j] + 2]];
                    p2[2] = z_in[tmp_cl[tmp_el[j] + 2]];

                    p3[0] = x_in[tmp_cl[tmp_el[j] + 3]];
                    p3[1] = y_in[tmp_cl[tmp_el[j] + 3]];
                    p3[2] = z_in[tmp_cl[tmp_el[j] + 3]];

                    if (isin_tetra(head, p0, p1, p2, p3) == 1)
                        return cnbr[i];
                }
            }
            break;

            case TYPE_PRISM:
            {
                prism2tet(actual_cell_type, cnbr[i], tmp_el, tmp_cl);
                for (j = 0; j < 3; j++)
                {
                    p0[0] = x_in[tmp_cl[tmp_el[j]]];
                    p0[1] = y_in[tmp_cl[tmp_el[j]]];
                    p0[2] = z_in[tmp_cl[tmp_el[j]]];

                    p1[0] = x_in[tmp_cl[tmp_el[j] + 1]];
                    p1[1] = y_in[tmp_cl[tmp_el[j] + 1]];
                    p1[2] = z_in[tmp_cl[tmp_el[j] + 1]];

                    p2[0] = x_in[tmp_cl[tmp_el[j] + 2]];
                    p2[1] = y_in[tmp_cl[tmp_el[j] + 2]];
                    p2[2] = z_in[tmp_cl[tmp_el[j] + 2]];

                    p3[0] = x_in[tmp_cl[tmp_el[j] + 3]];
                    p3[1] = y_in[tmp_cl[tmp_el[j] + 3]];
                    p3[2] = z_in[tmp_cl[tmp_el[j] + 3]];

                    if (isin_tetra(head, p0, p1, p2, p3) == 1)
                        return cnbr[i];
                }
            }
            break;

            case TYPE_PYRAMID:
            {
                pyra2tet(actual_cell_type, cnbr[i], tmp_el, tmp_cl);
                for (j = 0; j < 2; j++)
                {
                    p0[0] = x_in[tmp_cl[tmp_el[j]]];
                    p0[1] = y_in[tmp_cl[tmp_el[j]]];
                    p0[2] = z_in[tmp_cl[tmp_el[j]]];

                    p1[0] = x_in[tmp_cl[tmp_el[j] + 1]];
                    p1[1] = y_in[tmp_cl[tmp_el[j] + 1]];
                    p1[2] = z_in[tmp_cl[tmp_el[j] + 1]];

                    p2[0] = x_in[tmp_cl[tmp_el[j] + 2]];
                    p2[1] = y_in[tmp_cl[tmp_el[j] + 2]];
                    p2[2] = z_in[tmp_cl[tmp_el[j] + 2]];

                    p3[0] = x_in[tmp_cl[tmp_el[j] + 3]];
                    p3[1] = y_in[tmp_cl[tmp_el[j] + 3]];
                    p3[2] = z_in[tmp_cl[tmp_el[j] + 3]];

                    if (isin_tetra(head, p0, p1, p2, p3) == 1)
                        return cnbr[i];
                }
            }
            break;

            case TYPE_TETRAHEDER:
            {
                p0[0] = x_in[cl[el[cnbr[i]]]];
                p0[1] = y_in[cl[el[cnbr[i]]]];
                p0[2] = z_in[cl[el[cnbr[i]]]];

                p1[0] = x_in[cl[el[cnbr[i]] + 1]];
                p1[1] = y_in[cl[el[cnbr[i]] + 1]];
                p1[2] = z_in[cl[el[cnbr[i]] + 1]];

                p2[0] = x_in[cl[el[cnbr[i]] + 2]];
                p2[1] = y_in[cl[el[cnbr[i]] + 2]];
                p2[2] = z_in[cl[el[cnbr[i]] + 2]];

                p3[0] = x_in[cl[el[cnbr[i]] + 3]];
                p3[1] = y_in[cl[el[cnbr[i]] + 3]];
                p3[2] = z_in[cl[el[cnbr[i]] + 3]];

                if (isin_tetra(head, p0, p1, p2, p3) == 1)
                    return cnbr[i];
            }
            break;

            default:
                return -2;
            }
        }

        //      cerr << "now checking for arbitrary mesh interfaces ...\n";
        if (amil && amil[ctl[m]] != 0)
        {
            for (int k = 0; k < amil[ctl[m]]->NoOfAMesh; k++)
            {
                int neighbCellType = amil[ctl[m]]->listOfAMesh[k];
                if (cnbr)
                {
                    delete[] cnbr;
                    printf("delete cnbr: %p\n", cnbr);
                }
                cnbr = new int[amil[neighbCellType]->NoOfCells];
                printf("   new cnbr: %p\n", cnbr);
                for (i = 0; i < amil[neighbCellType]->NoOfCells; i++)
                    cnbr[i] = amil[neighbCellType]->listOfCells[i];
                actual_cell_type = -1;
                //                cerr << "hitting AMI, checking " << amil[neighbCellType]->NoOfCells <<
                //                        " new cells\nconnecting cell type " << ctl[m] << " and " << neighbCellType << endl;
                for (i = 0; i < amil[neighbCellType]->NoOfCells; i++)
                {
                    switch (tl[cnbr[i]])
                    {

                    case TYPE_HEXAEDER:
                    {
                        hex2tet(actual_cell_type, cnbr[i], tmp_el, tmp_cl);
                        for (j = 0; j < 5; j++)
                        {
                            p0[0] = x_in[tmp_cl[tmp_el[j]]];
                            p0[1] = y_in[tmp_cl[tmp_el[j]]];
                            p0[2] = z_in[tmp_cl[tmp_el[j]]];

                            p1[0] = x_in[tmp_cl[tmp_el[j] + 1]];
                            p1[1] = y_in[tmp_cl[tmp_el[j] + 1]];
                            p1[2] = z_in[tmp_cl[tmp_el[j] + 1]];

                            p2[0] = x_in[tmp_cl[tmp_el[j] + 2]];
                            p2[1] = y_in[tmp_cl[tmp_el[j] + 2]];
                            p2[2] = z_in[tmp_cl[tmp_el[j] + 2]];

                            p3[0] = x_in[tmp_cl[tmp_el[j] + 3]];
                            p3[1] = y_in[tmp_cl[tmp_el[j] + 3]];
                            p3[2] = z_in[tmp_cl[tmp_el[j] + 3]];

                            if (isin_tetra(head, p0, p1, p2, p3) == 1)
                            {
                                //                               cerr << "return TYPE_HEXAEDER\n";
                                return cnbr[i];
                            }
                        }
                    }
                    break;

                    case TYPE_PRISM:
                    {
                        prism2tet(actual_cell_type, cnbr[i], tmp_el, tmp_cl);
                        for (j = 0; j < 3; j++)
                        {
                            p0[0] = x_in[tmp_cl[tmp_el[j]]];
                            p0[1] = y_in[tmp_cl[tmp_el[j]]];
                            p0[2] = z_in[tmp_cl[tmp_el[j]]];

                            p1[0] = x_in[tmp_cl[tmp_el[j] + 1]];
                            p1[1] = y_in[tmp_cl[tmp_el[j] + 1]];
                            p1[2] = z_in[tmp_cl[tmp_el[j] + 1]];

                            p2[0] = x_in[tmp_cl[tmp_el[j] + 2]];
                            p2[1] = y_in[tmp_cl[tmp_el[j] + 2]];
                            p2[2] = z_in[tmp_cl[tmp_el[j] + 2]];

                            p3[0] = x_in[tmp_cl[tmp_el[j] + 3]];
                            p3[1] = y_in[tmp_cl[tmp_el[j] + 3]];
                            p3[2] = z_in[tmp_cl[tmp_el[j] + 3]];

                            if (isin_tetra(head, p0, p1, p2, p3) == 1)
                            {
                                //                               cerr << "return TYPE_PRISM\n";
                                return cnbr[i];
                            }
                        }
                    }
                    break;

                    case TYPE_PYRAMID:
                    {
                        pyra2tet(actual_cell_type, cnbr[i], tmp_el, tmp_cl);
                        for (j = 0; j < 2; j++)
                        {
                            p0[0] = x_in[tmp_cl[tmp_el[j]]];
                            p0[1] = y_in[tmp_cl[tmp_el[j]]];
                            p0[2] = z_in[tmp_cl[tmp_el[j]]];

                            p1[0] = x_in[tmp_cl[tmp_el[j] + 1]];
                            p1[1] = y_in[tmp_cl[tmp_el[j] + 1]];
                            p1[2] = z_in[tmp_cl[tmp_el[j] + 1]];

                            p2[0] = x_in[tmp_cl[tmp_el[j] + 2]];
                            p2[1] = y_in[tmp_cl[tmp_el[j] + 2]];
                            p2[2] = z_in[tmp_cl[tmp_el[j] + 2]];

                            p3[0] = x_in[tmp_cl[tmp_el[j] + 3]];
                            p3[1] = y_in[tmp_cl[tmp_el[j] + 3]];
                            p3[2] = z_in[tmp_cl[tmp_el[j] + 3]];

                            if (isin_tetra(head, p0, p1, p2, p3) == 1)
                            {
                                //                               cerr << "return TYPE_PYRAMID\n";
                                return cnbr[i];
                            }
                        }
                    }
                    break;

                    case TYPE_TETRAHEDER:
                    {
                        p0[0] = x_in[cl[el[cnbr[i]]]];
                        p0[1] = y_in[cl[el[cnbr[i]]]];
                        p0[2] = z_in[cl[el[cnbr[i]]]];

                        p1[0] = x_in[cl[el[cnbr[i]] + 1]];
                        p1[1] = y_in[cl[el[cnbr[i]] + 1]];
                        p1[2] = z_in[cl[el[cnbr[i]] + 1]];

                        p2[0] = x_in[cl[el[cnbr[i]] + 2]];
                        p2[1] = y_in[cl[el[cnbr[i]] + 2]];
                        p2[2] = z_in[cl[el[cnbr[i]] + 2]];

                        p3[0] = x_in[cl[el[cnbr[i]] + 3]];
                        p3[1] = y_in[cl[el[cnbr[i]] + 3]];
                        p3[2] = z_in[cl[el[cnbr[i]] + 3]];

                        if (isin_tetra(head, p0, p1, p2, p3) == 1)
                        {
                            //                           cerr << "return TYPE_TETRAHEDER\n";
                            return cnbr[i];
                        }
                    }
                    break;

                    default:
                        //                        cerr << "return default\n";
                        return -2;
                    }
                }
            }
        }
        //        cerr << "return -1 a\n";
        return -1;
    }
    //    cerr << "return -1 b\n";
    return -1;
}

//=====================================================================
//compute the stepsize at a point
//=====================================================================
float stepsize(float px[3], float p0[3], float p1[3], float p2[3], float p3[3],
               float v0[3], float v1[3], float v2[3], float v3[3])
{
    float dt, vq, vol;
    float v[3];

    //vorticity at px
    vort_tet(px, p0, p1, p2, p3, v0, v1, v2, v3, v);
    vq = sqrt(sqr(v[0]) + sqr(v[1]) + sqr(v[2]));
    if (vq == 0.0)
        return (0.0);

#if defined(__sgi) || defined(_WIN32)
    vol = fabsf(tetra_vol(p0, p1, p2, p3));
    dt = 1 / vq * powf(vol, 1.0f / 3.0f) / NUM_STEPS;
#else
    vol = fabs(tetra_vol(p0, p1, p2, p3));
    dt = 1 / vq * pow(vol, 1. / 3.) / NUM_STEPS;
#endif

    return dt;
}

//=====================================================================
// function to compute the volume of a tetrahedra cell
//=====================================================================
inline float tetra_vol(const float p0[3], const float p1[3], const float p2[3], const float p3[3])
{
    //returns the volume of the tetrahedra cell
    float vol;
    float diff1_0 = p1[0] - p0[0];
    float diff1_1 = p1[1] - p0[1];
    float diff1_2 = p1[2] - p0[2];
    float diff2_0 = p2[0] - p0[0];
    float diff2_1 = p2[1] - p0[1];
    float diff2_2 = p2[2] - p0[2];
    float diff3_0 = p3[0] - p0[0];
    float diff3_1 = p3[1] - p0[1];
    float diff3_2 = p3[2] - p0[2];

    vol = (diff2_1 * diff3_2 - diff3_1 * diff2_2) * diff1_0;
    vol += (diff2_2 * diff3_0 - diff3_2 * diff2_0) * diff1_1;
    vol += (diff2_0 * diff3_1 - diff3_0 * diff2_1) * diff1_2;
    vol *= 0.16666666666667f;

    return vol;
}

//=====================================================================
// is the point inside the tetrahedra cell
//=====================================================================
int isin_tetra(const float px[3], const float p0[3], const float p1[3], const float p2[3], const float p3[3])
{
    //returns 1 if point px is inside the tetrahedra cell, else 0
    float vg, w0, w1, w2, w3;

    vg = fabsf(tetra_vol(p0, p1, p2, p3));

    w0 = fabsf(tetra_vol(px, p1, p2, p3));

    w1 = fabsf(tetra_vol(p0, px, p2, p3));

    w2 = fabsf(tetra_vol(p0, p1, px, p3));

    w3 = fabsf(tetra_vol(p0, p1, p2, px));

    if (w0 + w1 + w2 + w3 <= vg * (1. + trace_eps))
        return 1;

    else
        return 0;
}

//=====================================================================
// compute the vorticity inside the tetrahedra cell
//=====================================================================
void vort_tet(const float px[3], const float p0[3], const float p1[3], const float p2[3], const float p3[3],
              const float v0[3], const float v1[3], const float v2[3], const float v3[3],
              float vx[3])
{
    //input p0 p1 p2 p3 v0 v1 v2 v3
    //output vx
    float w, w0, w1, w2, w3;

    // sl: The 5 fabsf below were not correct!!! This was not good for extrapolations
    w = tetra_vol(p0, p1, p2, p3);
    w0 = tetra_vol(px, p1, p2, p3) / w;
    w1 = tetra_vol(p0, px, p2, p3) / w;
    w2 = tetra_vol(p0, p1, px, p3) / w;
    w3 = tetra_vol(p0, p1, p2, px) / w;

    vx[0] = w0 * v0[0] + w1 * v1[0] + w2 * v2[0] + w3 * v3[0];
    vx[1] = w0 * v0[1] + w1 * v1[1] + w2 * v2[1] + w3 * v3[1];
    vx[2] = w0 * v0[2] + w1 * v1[2] + w2 * v2[2] + w3 * v3[2];

    return;
}

//=====================================================================
// 2nd order Runge Kutta function (Heun's scheme)
//=====================================================================
void rk2_tet(float px[3], float p0[3], float p1[3], float p2[3], float p3[3],
             float v0[3], float v1[3], float v2[3], float v3[3], float dt,
             float phead[3])
{
    //computes the new headposition for the streamline (phead[x|y|z])
    float tmp_pos[3], vn[3], vnp1[3];

    vort_tet(px, p0, p1, p2, p3, v0, v1, v2, v3, vn);

    tmp_pos[0] = px[0] + dt * vn[0];
    tmp_pos[1] = px[1] + dt * vn[1];
    tmp_pos[2] = px[2] + dt * vn[2];

    vort_tet(tmp_pos, p0, p1, p2, p3, v0, v1, v2, v3, vnp1);

    phead[0] = px[0] + dt * ((vn[0] + vnp1[0]) / 2.0f);
    phead[1] = px[1] + dt * ((vn[1] + vnp1[1]) / 2.0f);
    phead[2] = px[2] + dt * ((vn[2] + vnp1[2]) / 2.0f);

    return;
}

//=====================================================================
// 4th order Runge Kutta function
//=====================================================================
void rk4_tet(float px[3], float p0[3], float p1[3], float p2[3], float p3[3],
             float v0[3], float v1[3], float v2[3], float v3[3], float h,
             float phead[3])
{
    //computes the new headposition for the streamline (phead[x|y|z])
    float a[3], b[3], c[3], a2[3], b2[3];

    //a=v(x0)
    vort_tet(px, p0, p1, p2, p3, v0, v1, v2, v3, a);

    a2[0] = px[0] + a[0] * h / 2.0f;
    a2[1] = px[1] + a[1] * h / 2.0f;
    a2[2] = px[2] + a[2] * h / 2.0f;

    //b=v(x0+h/2)
    vort_tet(a2, p0, p1, p2, p3, v0, v1, v2, v3, b);

    b2[0] = px[0] + a[0] * h;
    b2[1] = px[1] + a[1] * h;
    b2[2] = px[2] + a[2] * h;

    //c=v(x0+h)
    vort_tet(b2, p0, p1, p2, p3, v0, v1, v2, v3, c);

    phead[0] = px[0] + (a[0] + 4 * b[0] + c[0]) * h / 6.0f;
    phead[1] = px[1] + (a[1] + 4 * b[1] + c[1]) * h / 6.0f;
    phead[2] = px[2] + (a[2] + 4 * b[2] + c[2]) * h / 6.0f;

    return;
}

//=====================================================================
// compute a particle trace
//=====================================================================
void tracer(int start, float startp[3], int *no_points,
            ia<float> &x_out, ia<float> &y_out, ia<float> &z_out,
            ia<float> &v_out, ia<float> &v_x_out, ia<float> &v_y_out,
            ia<float> &v_z_out, int *no_vert, ia<int> &vert_list, int startpointNo)
{
    (void)startpointNo;
    int i, j, is_in_domain, is_in_subdomain, is_in_cell;
    int tmp_el[5], tmp_cl[20], tmp_count;
    int actual_cell;
    int count_reduced = 0;
    // if there are more than 1000 reduced points
    // between two consecutive written points, do not use
    // reduction
    int flag;

    float px[3], p0[3], p1[3], p2[3], p3[3], phead[3], pn_1[3], p2n_1[3], p2x[3];
    float v0[3], v1[3], v2[3], v3[3], tmp_v[3];

    float dt = 0.0, dt_min, dt_max;
    float loopDist, oldLoopDist = 0.0f;

    float curr_len = 0.0;

    actual_cell = start;
    actual_cell_type = 1;

    // add startpoint
    x_out[0] = px[0] = p2x[0] = startp[0];
    y_out[0] = px[1] = p2x[1] = startp[1];
    z_out[0] = px[2] = p2x[2] = startp[2];
    *no_points = 0;
    int restarted = 1;
    *no_vert = 0;

    //while head of trace is inside domain
    is_in_domain = 1;
    while ((is_in_domain == 1) && (curr_len < trace_len))
    {

        switch (tl[actual_cell])
        {

        case TYPE_HEXAEDER:
        {
            hex2tet(actual_cell_type, actual_cell, tmp_el, tmp_cl);
            tmp_count = 5;
        }
        break;

        case TYPE_PRISM:
        {
            prism2tet(actual_cell_type, actual_cell, tmp_el, tmp_cl);
            tmp_count = 3;
        }
        break;

        case TYPE_PYRAMID:
        {
            pyra2tet(actual_cell_type, actual_cell, tmp_el, tmp_cl);
            tmp_count = 2;
        }
        break;

        case TYPE_TETRAHEDER:
        {
            tmp_el[0] = 0;
            for (j = 0; j < 4; j++)
            {
                tmp_cl[j] = cl[el[actual_cell] + j];
            }
            tmp_count = 1;
        }
        break;

        default:
        {
            Covise::sendInfo("INFO: Unsupported cell type detected, stop computation");
            return;
        }
        }

        is_in_subdomain = 1;
        while ((is_in_subdomain == 1) && (curr_len < trace_len))
        {
            for (i = 0; i < tmp_count; i++)
            {
                //find head of trace inside subdomain
                p0[0] = x_in[tmp_cl[tmp_el[i]]];
                p0[1] = y_in[tmp_cl[tmp_el[i]]];
                p0[2] = z_in[tmp_cl[tmp_el[i]]];

                p1[0] = x_in[tmp_cl[tmp_el[i] + 1]];
                p1[1] = y_in[tmp_cl[tmp_el[i] + 1]];
                p1[2] = z_in[tmp_cl[tmp_el[i] + 1]];

                p2[0] = x_in[tmp_cl[tmp_el[i] + 2]];
                p2[1] = y_in[tmp_cl[tmp_el[i] + 2]];
                p2[2] = z_in[tmp_cl[tmp_el[i] + 2]];

                p3[0] = x_in[tmp_cl[tmp_el[i] + 3]];
                p3[1] = y_in[tmp_cl[tmp_el[i] + 3]];
                p3[2] = z_in[tmp_cl[tmp_el[i] + 3]];

                is_in_subdomain = isin_tetra(px, p0, p1, p2, p3);
                if (is_in_subdomain == 1)
                {
                    break;
                }
            }
            //if found, compute the streamline in the single cell
            if (is_in_subdomain == 1)
            {
                // vorticity field from data set
                v0[0] = u_in[tmp_cl[tmp_el[i]]];
                v0[1] = v_in[tmp_cl[tmp_el[i]]];
                v0[2] = w_in[tmp_cl[tmp_el[i]]];

                v1[0] = u_in[tmp_cl[tmp_el[i] + 1]];
                v1[1] = v_in[tmp_cl[tmp_el[i] + 1]];
                v1[2] = w_in[tmp_cl[tmp_el[i] + 1]];

                v2[0] = u_in[tmp_cl[tmp_el[i] + 2]];
                v2[1] = v_in[tmp_cl[tmp_el[i] + 2]];
                v2[2] = w_in[tmp_cl[tmp_el[i] + 2]];

                v3[0] = u_in[tmp_cl[tmp_el[i] + 3]];
                v3[1] = v_in[tmp_cl[tmp_el[i] + 3]];
                v3[2] = w_in[tmp_cl[tmp_el[i] + 3]];

                if (intpol_dir == 2)
                {
                    for (j = 0; j < 3; j++)
                    {
                        v0[j] *= -1;
                        v1[j] *= -1;
                        v2[j] *= -1;
                        v3[j] *= -1;
                    }
                }

                //compute the initial stepsize for integration
                if (restarted || (*no_points) == 0)
                {
                    dt = stepsize(px, p0, p1, p2, p3, v0, v1, v2, v3);
                    restarted = 0;
                }

                is_in_cell = 1;
                while (is_in_cell == 1)
                {
                    //compute the new position
                    switch (option)
                    {
                    case 1:
                    {
                        flag = 0; // get new point for output????
                        while (flag < 1)
                        {
                            rk2_tet(px, p0, p1, p2, p3, v0, v1, v2, v3, dt, phead);
                            if ((px[0] == phead[0]) && (px[1] == phead[1]) && (px[2] == phead[2]))
                            {
                                // sl: In this case we have to include
                                //     px in the solution anyway...
                                vort_tet(px, p0, p1, p2, p3, v0, v1, v2, v3, tmp_v);
                                x_out[*no_points] = px[0];
                                y_out[*no_points] = px[1];
                                z_out[*no_points] = px[2];
                                if (intpol_dir == 2)
                                {
                                    tmp_v[0] *= -1.0;
                                    tmp_v[1] *= -1.0;
                                    tmp_v[2] *= -1.0;
                                }
                                v_out[*no_points] = dout(tmp_v);
                                v_x_out[*no_points] = tmp_v[0];
                                v_y_out[*no_points] = tmp_v[1];
                                v_z_out[*no_points] = tmp_v[2];
                                vert_list[*no_vert] = *no_vert;
                                (*no_points)++;
                                (*no_vert)++;
                                return;
                            }
                            // Local step control
                            if (stp_control == 1 || *no_points <= 1)
                            {
                                dt = stepsize(phead, p0, p1, p2, p3, v0, v1, v2, v3);
                                flag = 2; // we have got the new point????
                            }
                            else if (stp_control == 2 && *no_points > 1)
                            {
                                // < 5 degrees
                                if (p_ang(pn_1, px, phead) > .9961)
                                {
                                    dt *= 1.8f;
                                    dt_max = 3.0f * stepsize(px, p0, p1, p2, p3, v0, v1, v2, v3);
                                    dt = (dt > dt_max ? dt_max : dt);
                                    flag = 2;
                                }
                                // > 12 degrees
                                else if (p_ang(pn_1, px, phead) < .978)
                                {
                                    dt /= 2.;
                                    dt_min = 0.1f * stepsize(px, p0, p1, p2, p3, v0, v1, v2, v3);
                                    if (dt < dt_min)
                                    {
                                        dt = dt_min;
                                        flag = 2;
                                    }
                                    else
                                    {
                                        flag = 0; // we still have to get a new point
                                    }
                                }
                                else
                                    flag = 2;
                            }
                            else if (stp_control == 3 && *no_points > 1)
                            {
                                // < 2 degrees
                                if (p_ang(pn_1, px, phead) > .99939)
                                {
                                    dt *= 1.8f;
                                    dt_max = 2.0f * stepsize(px, p0, p1, p2, p3, v0, v1, v2, v3);
                                    dt = (dt > dt_max ? dt_max : dt);
                                    flag = 2;
                                }
                                // > 6 degrees
                                else if (p_ang(pn_1, px, phead) < .9945)
                                {
                                    dt /= 2.;
                                    dt_min = 0.01f * stepsize(px, p0, p1, p2, p3, v0, v1, v2, v3);
                                    if (dt < dt_min)
                                    {
                                        dt = dt_min;
                                        flag = 2;
                                    }
                                    else
                                    {
                                        flag = 0;
                                    }
                                }
                                else
                                    flag = 2;
                            }
                        } // end loop for new point while(flag ....)
                    } // case end
                    break;
                    default:
                    {
                        flag = 0;
                        while (flag < 1)
                        {
                            rk4_tet(px, p0, p1, p2, p3, v0, v1, v2, v3, dt, phead);
                            if ((px[0] == phead[0]) && (px[1] == phead[1]) && (px[2] == phead[2]))
                            {
                                // cerr << "return 2\n";
                                // sl: In this case we have to include
                                //     px in the solution anyway...
                                vort_tet(px, p0, p1, p2, p3, v0, v1, v2, v3, tmp_v);
                                x_out[*no_points] = px[0];
                                y_out[*no_points] = px[1];
                                z_out[*no_points] = px[2];
                                if (intpol_dir == 2)
                                {
                                    tmp_v[0] *= -1.0;
                                    tmp_v[1] *= -1.0;
                                    tmp_v[2] *= -1.0;
                                }
                                v_out[*no_points] = dout(tmp_v);
                                v_x_out[*no_points] = tmp_v[0];
                                v_y_out[*no_points] = tmp_v[1];
                                v_z_out[*no_points] = tmp_v[2];
                                vert_list[*no_vert] = *no_vert;
                                (*no_points)++;
                                (*no_vert)++;
                                // end of addition
                                return;
                            }
                            if (stp_control == 1 || *no_points <= 1)
                            {
                                dt = stepsize(phead, p0, p1, p2, p3, v0, v1, v2, v3);
                                flag = 2;
                            }
                            else if (stp_control == 2 && *no_points > 1)
                            {
                                // < 5 degrees
                                if (p_ang(pn_1, px, phead) > .9961)
                                {
                                    dt *= 1.8f;
                                    dt_max = 3.0f * stepsize(px, p0, p1, p2, p3, v0, v1, v2, v3);
                                    dt = (dt > dt_max ? dt_max : dt);
                                    flag = 2;
                                }
                                // > 12 degrees
                                else if (p_ang(pn_1, px, phead) < .978)
                                {
                                    dt /= 2.0f;
                                    dt_min = 0.1f * stepsize(px, p0, p1, p2, p3, v0, v1, v2, v3);
                                    if (dt < dt_min)
                                    {
                                        dt = dt_min;
                                        flag = 2;
                                    }
                                    else
                                    {
                                        flag = 0;
                                    }
                                }
                                else
                                    flag = 2;
                            }
                            else if (stp_control == 3 && *no_points > 1)
                            {
                                // < 2 degrees
                                if (p_ang(pn_1, px, phead) > .99939)
                                {
                                    dt *= 1.8f;
                                    //dt_max = 2. * stepsize(px, p0, p1, p2, p3, v0, v1, v2, v3);
                                    //dt = (dt > dt_max ? dt_max : dt);
                                    flag = 2;
                                }
                                // > 6 degrees
                                else if (p_ang(pn_1, px, phead) < .9945)
                                {
                                    dt /= 2.0f;
                                    dt_min = 0.01f * stepsize(px, p0, p1, p2, p3, v0, v1, v2, v3);
                                    if (dt < dt_min)
                                    {
                                        dt = dt_min;
                                        flag = 2;
                                    }
                                    else
                                    {
                                        flag = 0;
                                    }
                                }
                                else
                                    flag = 2;
                            }
                        } // end while
                    }
                    break;
                    } // end switch

                    curr_len += sqrt((px[0] - phead[0]) * (px[0] - phead[0]) + (px[1] - phead[1]) * (px[1] - phead[1]) + (px[2] - phead[2]) * (px[2] - phead[2]));

                    // sl: This may work in some cases, but in general
                    //     it will not detect what it is supposed to detect
                    // warum?
                    if (loopDelta > 0.0) // loop Detection is turned on!
                    {
                        loopDist = ((phead[0] - startp[0]) * (phead[0] - startp[0]) + (phead[1] - startp[1]) * (phead[1] - startp[1]) + (phead[2] - startp[2]) * (phead[2] - startp[2]));
                        if (loopDist < loopDelta2)
                        {
                            if (curr_len > loopDelta * 2)
                            {
                                // make sure we left the startpoint
                                if (loopDist > oldLoopDist)
                                {
                                    loopedArray[startpointNo] = 1;
                                    // write startpoint direct to output
                                    pn_1[0] = p2n_1[0] = px[0];
                                    pn_1[1] = p2n_1[1] = px[1];
                                    pn_1[2] = p2n_1[2] = px[2];

                                    x_out[*no_points] = startp[0];
                                    y_out[*no_points] = startp[1];
                                    z_out[*no_points] = startp[2];
                                    vort_tet(px, p0, p1, p2, p3, v0, v1, v2, v3, tmp_v);
                                    v_out[*no_points] = dout(tmp_v);
                                    v_x_out[*no_points] = tmp_v[0];
                                    v_y_out[*no_points] = tmp_v[1];
                                    v_z_out[*no_points] = tmp_v[2];
                                    px[0] = p2x[0] = phead[0];
                                    px[1] = p2x[1] = phead[1];
                                    px[2] = p2x[2] = phead[2];
                                    (*no_points)++;

                                    vert_list[*no_vert] = *no_vert;
                                    (*no_vert)++;
                                    cerr << "return 3\n";
                                    return;
                                }
                                oldLoopDist = loopDist;
                            }
                        }
                        else
                            oldLoopDist = FLT_MAX;
                    } // end of loop detection

                    // Instead we try something less cute, but
                    // more robust. The idea is to prevent a situation
                    // in which the trajectory slowly "converges" to a point,
                    // in which case, the process would consume a lot of CPU
                    // time (and memory!!!) for nothing worth the effort
                    if (*no_points >= MaxPoints)
                    {
                        Covise::sendInfo("Integration interrupted because the maximum number of points was reached. Increase MaxPoints if you want to integrate further");
                        return;
                    }

                    // write the new position to tmp arrays
                    // reduce cellpoints for output
                    if ((reduce == 2 || reduce == 3) && *no_points >= 1)
                    {
                        if (reduce == 2)
                        {
                            // do not reduce in this case
                            if (p_ang(p2n_1, p2x, phead) < .9995)
                            {
                                count_reduced = 0;

                                pn_1[0] = px[0];
                                pn_1[1] = px[1];
                                pn_1[2] = px[2];

                                p2n_1[0] = p2x[0];
                                p2n_1[1] = p2x[1];
                                p2n_1[2] = p2x[2];

                                x_out[*no_points] = px[0];
                                y_out[*no_points] = px[1];
                                z_out[*no_points] = px[2];
                                vort_tet(px, p0, p1, p2, p3, v0, v1, v2, v3, tmp_v);
                                if (intpol_dir == 2)
                                {
                                    tmp_v[0] *= -1.0;
                                    tmp_v[1] *= -1.0;
                                    tmp_v[2] *= -1.0;
                                }
                                v_out[*no_points] = dout(tmp_v);
                                v_x_out[*no_points] = tmp_v[0];
                                v_y_out[*no_points] = tmp_v[1];
                                v_z_out[*no_points] = tmp_v[2];
                                px[0] = p2x[0] = phead[0];
                                px[1] = p2x[1] = phead[1];
                                px[2] = p2x[2] = phead[2];

                                vert_list[*no_vert] = *no_vert;

                                (*no_points)++;
                                (*no_vert)++;
                            } // now reduce
                            else
                            {
                                count_reduced++;
                                if (count_reduced == 1000)
                                {
                                    Covise::sendWarning("Please, do not use line reduction");
                                    return;
                                }

                                pn_1[0] = px[0];
                                pn_1[1] = px[1];
                                pn_1[2] = px[2];

                                px[0] = phead[0];
                                px[1] = phead[1];
                                px[2] = phead[2];
                            }
                        } // reduce==3
                        else
                        {
                            if (p_ang(p2n_1, p2x, phead) < .9986)
                            {
                                count_reduced = 0;
                                pn_1[0] = px[0];
                                pn_1[1] = px[1];
                                pn_1[2] = px[2];

                                p2n_1[0] = p2x[0];
                                p2n_1[1] = p2x[1];
                                p2n_1[2] = p2x[2];

                                x_out[*no_points] = px[0];
                                y_out[*no_points] = px[1];
                                z_out[*no_points] = px[2];
                                vort_tet(px, p0, p1, p2, p3, v0, v1, v2, v3, tmp_v);
                                if (intpol_dir == 2)
                                {
                                    tmp_v[0] *= -1.0;
                                    tmp_v[1] *= -1.0;
                                    tmp_v[2] *= -1.0;
                                }
                                v_out[*no_points] = dout(tmp_v);
                                v_x_out[*no_points] = tmp_v[0];
                                v_y_out[*no_points] = tmp_v[1];
                                v_z_out[*no_points] = tmp_v[2];
                                px[0] = p2x[0] = phead[0];
                                px[1] = p2x[1] = phead[1];
                                px[2] = p2x[2] = phead[2];

                                vert_list[*no_vert] = *no_vert;

                                (*no_points)++;
                                (*no_vert)++;
                            }
                            else // reduce in this case
                            {
                                count_reduced++;
                                if (count_reduced == 1000)
                                {
                                    Covise::sendWarning("Please, do not use line reduction");
                                    return;
                                }

                                pn_1[0] = px[0];
                                pn_1[1] = px[1];
                                pn_1[2] = px[2];

                                px[0] = phead[0];
                                px[1] = phead[1];
                                px[2] = phead[2];
                            }
                        }
                    }
                    else // no reduction: write direct to output
                    {
                        pn_1[0] = p2n_1[0] = px[0];
                        pn_1[1] = p2n_1[1] = px[1];
                        pn_1[2] = p2n_1[2] = px[2];

                        x_out[*no_points] = px[0];
                        y_out[*no_points] = px[1];
                        z_out[*no_points] = px[2];
                        vort_tet(px, p0, p1, p2, p3, v0, v1, v2, v3, tmp_v);
                        if (intpol_dir == 2)
                        {
                            tmp_v[0] *= -1.0;
                            tmp_v[1] *= -1.0;
                            tmp_v[2] *= -1.0;
                        }
                        v_out[*no_points] = dout(tmp_v);
                        v_x_out[*no_points] = tmp_v[0];
                        v_y_out[*no_points] = tmp_v[1];
                        v_z_out[*no_points] = tmp_v[2];
                        px[0] = p2x[0] = phead[0];
                        px[1] = p2x[1] = phead[1];
                        px[2] = p2x[2] = phead[2];

                        vert_list[*no_vert] = *no_vert;

                        (*no_points)++;
                        (*no_vert)++;
                    }

                    is_in_cell = isin_tetra(phead, p0, p1, p2, p3);
                } // end while(is_in_cell)
            } // end if(is_in_subdomain)
        } // end  while((is_in_subdomain == 1)&&(curr_len<trace_len))
        // we have got out of the cell or curr_len>=trace_len
        if (curr_len >= trace_len)
        {
            Covise::sendInfo("Integrated trajectory reached trace_len limit");
            return;
        }
        int tryit = 0;
        int this_cell = actual_cell;
        // look for neighbors in cnbr_list
        while (((actual_cell = find_actualcell(this_cell, phead)) == -1) && (tryit < 5))
        {
            if (phead[0] == px[0] && phead[1] == px[1] && phead[2] == px[2])
            {
                tryit = 5;
                break;
            }
            phead[0] = px[0] + (phead[0] - px[0]) * 1.5f;
            phead[1] = px[1] + (phead[1] - px[1]) * 1.5f;
            phead[2] = px[2] + (phead[2] - px[2]) * 1.5f;
            //Covise::sendInfo("INFO: retried");
            tryit++;
        }

        int thisBlock = CurrentBlock;
        // notASet==0 may be eliminated from this line
        //            in case that we have a simple grid
        //            but with a "false" connectivity
        //            (probably introduced during refinement)
        //
        if ((Connectivity == 0 || notASet == 0) && actual_cell < 0 && tryit == 5)
        {
            // cerr << "tryit == 5\n";
            // look for other grids
            actual_cell = find_startcell_fast(phead);
            // cerr << "actual_cell now: " << actual_cell << endl;
        }

        if (actual_cell < 0)
        {
            return;
        }
        else if (thisBlock == CurrentBlock)
        {
            restarted = 1;
            is_in_domain = 1;
        }
        else
        {
            // We have moved to another block, there was a wall between them??
            // If the answer is positive we have to stop the integration!!!
            float new_velocity[3];
            float old_velocity[3];

            // vort_tet does not introduce any additional sign flipping
            vort_tet(px, p0, p1, p2, p3, v0, v1, v2, v3, old_velocity);
            // the same applies to read_velocity
            if (read_velocity(actual_cell, phead, new_velocity) == 0 && check_diff(old_velocity, new_velocity))
            {
                restarted = 1;
                is_in_domain = 1;
            }
            else
            {
                // Some problem or wall detected, I am off
                return;
            }
        }
    } // end of while((is_in_domain == 1)&&(curr_len<trace_len))
    if (curr_len >= trace_len)
    {
        Covise::sendInfo("Integrated trajectory reached trace_len limit");
    }
    return;
}

//=====================================================================
//print adjacency information
//=====================================================================
void print_adjacency()
{
    int i, j, k;
    FILE *file1 = fopen("adjacency.out", "w");

    fprintf(file1, "\nZellnachbarn:");
    for (i = 0; i < numelem; i++)
    {
        fprintf(file1, "\nZelle[%d]: ", i);
        k = 0;
        for (j = cnbr_pos[i]; j < cnbr_pos[i + 1]; j++)
        {
            fprintf(file1, "%d, ", cnbr[j]);
            k++;
        }
        fprintf(file1, "\nSumme: %d", k);
        fprintf(file1, "\n");
    }
    fclose(file1);
}

//=====================================================================
//returns the cos(angle) from the deviation of the pathline between 2 steps
//=====================================================================
double p_ang(float p0[3], float p1[3], float p2[3])
{
    double cos_tn;

    cos_tn = ((double)(p1[0] - p0[0]) * (double)(p2[0] - p1[0]) + (double)(p1[1] - p0[1]) * (double)(p2[1] - p1[1]) + (double)(p1[2] - p0[2]) * (double)(p2[2] - p1[2])) / (sqrt(sqr((double)(p1[0] - p0[0])) + sqr((double)(p1[1] - p0[1])) + sqr((double)(p1[2] - p0[2]))) * sqrt(sqr((double)(p2[0] - p1[0])) + sqr((double)(p2[1] - p1[1])) + sqr((double)(p2[2] - p1[2]))));

    return cos_tn;
}

//=====================================================================
//returns the scalar of chosen vorticity
//=====================================================================
float dout(const float t[3])
{
    float s;

    switch (data_out_type)
    {

    case 1:
    {
        s = sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
    }
    break;

    case 2:
    {
        s = t[0];
    }
    break;

    case 3:
    {
        s = t[1];
    }
    break;

    case 4:
    {
        s = t[2];
    }
    break;

    case 5:
    {
        s = (float)trace_num;
    }
    break;

    default:
    {
        Covise::sendInfo("Info: Choice of whatout incorrect -> DataOut = 0");
        s = 0;
    }
    }

    return s;
}

//=====================================================================
//return the minimum of an float array
//=====================================================================
float Min_of(float array[], int length)
{
    int i;
    float val = array[0];

    for (i = 1; i < length; i++)
    {
        val = (val <= array[i]) ? val : array[i];
    }
    return val;
}

//=====================================================================
//return the maximum of an float array
//=====================================================================
float Max_of(float array[], int length)
{
    int i;
    float val = array[0];

    for (i = 1; i < length; i++)
    {
        val = (val >= array[i]) ? val : array[i];
    }
    return val;
}

//=====================================================================
// function to find the cell neighbors of a cell
//=====================================================================
int create_single_cnbr(int actual_cell, int /*cuc_count*/, int *cuc, int *cuc_pos, int *arr_count, int **arr)
{
    int i = 0, j, k, l;
    int arr_val, offset, tmp_pos;
    ia<int> tmpl1(400);
    int arr_lim = 2000;

    delete[] * arr;
    *arr = new int[arr_lim];

    //   memset(gpt_nbr_count, 0, numcoord*sizeof(int));
    // not necessary, oder? memset(*arr, -1, arr_lim*sizeof(int));

    //   memset(&tmpl1[0], -1, 400*sizeof(int));
    //   tmpl1ind=tmpl1;
    tmp_pos = 0;

    //find the offset in cl-list
    offset = UnstructuredGrid_Num_Nodes[tl[actual_cell]];

    //put the cells from every point in a single cell

    // we have an old gpt_nbr_count
    /* oooo
   if(notASet==0){   // execute only if we have a set
   delete [] gpt_nbr_count;
   gpt_nbr_count = new int[numcoord];
   memset(gpt_nbr_count, 0, numcoord*sizeof(int));
   //create the gpt_nbr_countlist
   strcpy(lastGridGpt,lastGrid);
   for(i=0;i<numconn;i++){
   gpt_nbr_count[cl[i]]++;
   }
   }
   */

    for (j = 0; j < offset; j++)
    {
        for (k = 0; k < gpt_nbr_count[cl[el[actual_cell] + j]]; k++)
        {
            tmpl1[i++] = cuc[cuc_pos[cl[el[actual_cell] + j]] + k];
        }
    }
    tmpl1[i] = -1;

    /*
   l=0;
   mark=0;
   while(mark!=-1){
   if(tmpl1[l]==i) tmpl1[l]=-2;
   mark=tmpl1[l++];
   }
   */
    //sort(tmpl1,0,l-1);
    l = 0;
    while (tmpl1[l] != -1)
        l++;
    qsort(&tmpl1[0], l, sizeof(int), compare);
    //   qsort(tmpl1, l-1, sizeof(int), compare);

    //find the beginning in the list
    /*
   l=0;
   while(tmpl1[l] == -2) l++;
   */

    //fill the cellnbr list
    l = 0;
    arr_val = -1;
    while (tmpl1[l] != -1)
    {
        if (tmpl1[l] != arr_val)
        {
            if (tmp_pos >= arr_lim)
            {
                // we have to reallocate!!!
                int *arr_real;
                arr_real = new int[2 * arr_lim];
                memcpy(arr_real, *arr, arr_lim * sizeof(int));
                delete[] * arr;
                *arr = arr_real;
                arr_lim *= 2;
                /*
            Covise::sendError("ERROR:  Not enough memory allocated for cnbr-list");
            return 0;
            */
            }
            (*arr)[tmp_pos++] = tmpl1[l];
            arr_val = tmpl1[l];
        }
        l++;
    }
    *arr_count = tmp_pos;
    return 1;
}

//=====================================================================
// find the startcell (fast)
//=====================================================================
int find_startcell_fast(float *p)
{
    int i, j, k, mark;
    int tmp_el[5], tmp_cl[20];
    ia<int> tmp_inbox, inbox;
    int tmp_inbox_count, inbox_count;
    ia<int> pib;
    try
    {
        pib[MAX_PIB - 1] = 0;
    }
    catch (...)
    {
        cerr << "Construction failure of infinite array with " << MAX_PIB << "elements" << endl;
        Covise::sendWarning("WARNING: Construction failure of infinite array");
        return -2;
    }
    int pib_count;

    float p0[3], p1[3], p2[3], p3[3];
    //   float tp[3];
    float tmp_length[3];
    float x_min, x_max, y_min, y_max, z_min, z_max;

    float factor; // sl: Diese Zuweisung ist hier vorzeitig = 2,
    //     weil es immer initialisiert werden muss, wenn
    //     wir in einem anderen Block suchen

    int CheckedAllBlocks = 0;
    for (i = 0; i < NoOfBlocks; i++)
        SetElementChecked[i] = 0;

    while (!CheckedAllBlocks)
    {

        // create ref box
        if (numelem > 0)
            typicalDim(tmp_length);
        /*
      switch(tl[0]) {

      case TYPE_HEXAEDER:
      {
      tmp_length[0] = 0.0;
      tmp_length[1] = 0.0;
      tmp_length[2] = 0.0;

      int elem_loop;   // RISKY, OPTIMISTIC METHOD
      // this assumes that all elements
      // are the same type!!!!!!
      int inc_loop=numelem/1000;
      int count_elems=0;
      if(inc_loop==0) inc_loop=1;
      for(elem_loop = 0;elem_loop < numelem;elem_loop += inc_loop) {
      //Covise::sendInfo("INFO: DataSet -> TYPE_HEXAGON");
      for(i=0;i<8;i++)
      {
      xb[i] = x_in[cl[el[elem_loop]+i]];
      yb[i] = y_in[cl[el[elem_loop]+i]];
      zb[i] = z_in[cl[el[elem_loop]+i]];
      }

      tmp_length[0] += Max_of(xb, 8) - Min_of(xb, 8);
      tmp_length[1] += Max_of(yb, 8) - Min_of(yb, 8);
      tmp_length[2] += Max_of(zb, 8) - Min_of(zb, 8);
      count_elems++;
      }
      tmp_length[0]/=count_elems;
      tmp_length[1]/=count_elems;
      tmp_length[2]/=count_elems;

      }
      break;

      case TYPE_PRISM:
      {
      //Covise::sendInfo("INFO: DataSet -> TYPE_PRISM");
      for(i=0;i<6;i++)
      {
      xb[i] = x_in[cl[el[0]+i]];
      yb[i] = y_in[cl[el[0]+i]];
      zb[i] = z_in[cl[el[0]+i]];
      }

      tmp_length[0] = Max_of(xb, 6) - Min_of(xb, 6);
      tmp_length[1] = Max_of(yb, 6) - Min_of(yb, 6);
      tmp_length[2] = Max_of(zb, 6) - Min_of(zb, 6);

      }
      break;

      case TYPE_PYRAMID:
      {
      //Covise::sendInfo("INFO: DataSet -> TYPE_PYRAMID");
      for(i=0;i<5;i++)
      {
      xb[i] = x_in[cl[el[0]+i]];
      yb[i] = y_in[cl[el[0]+i]];
      zb[i] = z_in[cl[el[0]+i]];
      }

      tmp_length[0] = Max_of(xb, 5) - Min_of(xb, 5);
      tmp_length[1] = Max_of(yb, 5) - Min_of(yb, 5);
      tmp_length[2] = Max_of(zb, 5) - Min_of(zb, 5);

      }
      break;

      case TYPE_TETRAHEDER:
      {
      for(i=0;i<4;i++)
      {
      //Covise::sendInfo("INFO: DataSet -> TYPE_TETRAHEDER");
      xb[i] = x_in[cl[el[0]+i]];
      yb[i] = y_in[cl[el[0]+i]];
      zb[i] = z_in[cl[el[0]+i]];
      }

      tmp_length[0] = Max_of(xb, 4) - Min_of(xb, 4);
      tmp_length[1] = Max_of(yb, 4) - Min_of(yb, 4);
      tmp_length[2] = Max_of(zb, 4) - Min_of(zb, 4);

      }
      break;

      default:
      {
      cerr << "return: wrong type\n";
      return -2;
      }
      }
      */

        pib_count = 0;
        factor = 2;
        // find all coords inside the box
        while (pib_count < 8 && numelem > 0)
        {
            //cerr << "factor: "<< factor <<"\n";
            if ((factor > MaxSearchFactor) || (factor <= 0))
            {
                // cerr << "Factor not in range\n";
                if (onceIsEnoughFactor == 0)
                {
                    Covise::sendInfo("MaxSearchFactor exceeded. This may be normal and harmless if the initial point is far from the nearest element, which is also usual when using sets. In a pathological case it may be due to large irregularities in the grid, only in this case try with a larger MaxSearchFactor");
                    onceIsEnoughFactor = 1;
                }
                // this is not valid with sets:   return(-1);
                pib_count = 0;
                break;
            }
            box_length = factor * Max_of(tmp_length, 3);
            //       cout << "box_length:" << box_length << endl;
            box_flag = 1;

            // compute the absolute position of the box
            x_min = p[0] - box_length * 0.5f;
            x_max = p[0] + box_length * 0.5f;
            y_min = p[1] - box_length * 0.5f;
            y_max = p[1] + box_length * 0.5f;
            z_min = p[2] - box_length * 0.5f;
            z_max = p[2] + box_length * 0.5f;
            pib_count = 0;

            for (i = 0; i < numcoord; i++) //every coord
            {

                if (x_in[i] > x_min && x_in[i] < x_max)
                {
                    if (y_in[i] > y_min && y_in[i] < y_max)
                    {
                        if (z_in[i] > z_min && z_in[i] < z_max)
                        {
                            try
                            {
                                pib[pib_count] = i;
                            }
                            catch (...)
                            {
                                Covise::sendWarning("Problem with assignation to array element");
                                return -2;
                            }
                            pib_count++;
                        }
                    }
                }
            }
            factor++;
            //       cout << "factor:    " << factor << endl;
            //       cout << "pib_count: " << pib_count << endl;
        }
        //if(pib_count == 0){
        //    Covise::sendInfo("INFO: No points found in trap box - trying sequential module");
        //    k = find_startcell(p);

        //    delete[] pib;

        //    return k;
        //}

        //cerr << "\nNo of elements in pib-list (find_startpoint_fast) " << pib_count << endl;

        // compute the size of inbox
        tmp_inbox_count = 0;
        for (i = 0; i < pib_count; i++)
        {
            tmp_inbox_count += cuc_pos[pib[i] + 1] - cuc_pos[pib[i]];
        }

        // tmp_inbox = new int[tmp_inbox_count];

        // filling tmp_inbox list
        k = 0;
        for (i = 0; i < pib_count; i++)
        {
            for (j = cuc_pos[pib[i]]; j < cuc_pos[pib[i] + 1]; j++)
            {
                tmp_inbox[k] = cuc[j];
                k++;
            }
        }

        if (k)
        {
            qsort(&tmp_inbox[0], k, sizeof(int), compare);

            // create the inbox list
            // inbox = new int[tmp_inbox_count];
            inbox_count = 0;
            mark = tmp_inbox[0];
            inbox[0] = tmp_inbox[0];
            inbox_count = 1;

            for (i = 1; i < k; i++)
            {
                if (tmp_inbox[i] != mark)
                {
                    inbox[inbox_count] = tmp_inbox[i];
                    mark = tmp_inbox[i];
                    inbox_count++;
                }
            }
        }
        else
            inbox_count = 0;

        // look for cell inside the box
        if (numelem > 0 && factor <= MaxSearchFactor)
            for (i = 0; i < inbox_count; i++)
            {
                switch (tl[inbox[i]])
                {

                case TYPE_HEXAEDER:
                {
                    hex2tet(1, inbox[i], tmp_el, tmp_cl);
                    for (j = 0; j < 5; j++)
                    {
                        p0[0] = x_in[tmp_cl[tmp_el[j]]];
                        p0[1] = y_in[tmp_cl[tmp_el[j]]];
                        p0[2] = z_in[tmp_cl[tmp_el[j]]];

                        p1[0] = x_in[tmp_cl[tmp_el[j] + 1]];
                        p1[1] = y_in[tmp_cl[tmp_el[j] + 1]];
                        p1[2] = z_in[tmp_cl[tmp_el[j] + 1]];

                        p2[0] = x_in[tmp_cl[tmp_el[j] + 2]];
                        p2[1] = y_in[tmp_cl[tmp_el[j] + 2]];
                        p2[2] = z_in[tmp_cl[tmp_el[j] + 2]];

                        p3[0] = x_in[tmp_cl[tmp_el[j] + 3]];
                        p3[1] = y_in[tmp_cl[tmp_el[j] + 3]];
                        p3[2] = z_in[tmp_cl[tmp_el[j] + 3]];

                        if (isin_tetra(p, p0, p1, p2, p3) == 1)
                        {
                            k = inbox[i];

                            /*****************************************************************
                        if(lookForInit == 1)
                        cerr << "return: inital cell found\n";
                        *****************************************************************/
                            return k;
                        }
                    }
                }
                break;

                case TYPE_PRISM:
                {
                    prism2tet(1, inbox[i], tmp_el, tmp_cl);
                    for (j = 0; j < 3; j++)
                    {
                        p0[0] = x_in[tmp_cl[tmp_el[j]]];
                        p0[1] = y_in[tmp_cl[tmp_el[j]]];
                        p0[2] = z_in[tmp_cl[tmp_el[j]]];

                        p1[0] = x_in[tmp_cl[tmp_el[j] + 1]];
                        p1[1] = y_in[tmp_cl[tmp_el[j] + 1]];
                        p1[2] = z_in[tmp_cl[tmp_el[j] + 1]];

                        p2[0] = x_in[tmp_cl[tmp_el[j] + 2]];
                        p2[1] = y_in[tmp_cl[tmp_el[j] + 2]];
                        p2[2] = z_in[tmp_cl[tmp_el[j] + 2]];

                        p3[0] = x_in[tmp_cl[tmp_el[j] + 3]];
                        p3[1] = y_in[tmp_cl[tmp_el[j] + 3]];
                        p3[2] = z_in[tmp_cl[tmp_el[j] + 3]];

                        if (isin_tetra(p, p0, p1, p2, p3) == 1)
                        {
                            k = inbox[i];

                            /******************************************************************
                        if(lookForInit == 1)
                        cerr << "return: inital cell found\n";
                        *******************************************************************/
                            return k;
                        }
                    }
                }
                break;

                case TYPE_PYRAMID:
                {
                    pyra2tet(1, inbox[i], tmp_el, tmp_cl);
                    for (j = 0; j < 2; j++)
                    {
                        p0[0] = x_in[tmp_cl[tmp_el[j]]];
                        p0[1] = y_in[tmp_cl[tmp_el[j]]];
                        p0[2] = z_in[tmp_cl[tmp_el[j]]];

                        p1[0] = x_in[tmp_cl[tmp_el[j] + 1]];
                        p1[1] = y_in[tmp_cl[tmp_el[j] + 1]];
                        p1[2] = z_in[tmp_cl[tmp_el[j] + 1]];

                        p2[0] = x_in[tmp_cl[tmp_el[j] + 2]];
                        p2[1] = y_in[tmp_cl[tmp_el[j] + 2]];
                        p2[2] = z_in[tmp_cl[tmp_el[j] + 2]];

                        p3[0] = x_in[tmp_cl[tmp_el[j] + 3]];
                        p3[1] = y_in[tmp_cl[tmp_el[j] + 3]];
                        p3[2] = z_in[tmp_cl[tmp_el[j] + 3]];

                        if (isin_tetra(p, p0, p1, p2, p3) == 1)
                        {
                            k = inbox[i];

                            /***************************************************************
                        if(lookForInit == 1)
                        cerr << "return: inital cell found\n";
                        ****************************************************************/
                            return k;
                        }
                    }
                }
                break;

                case TYPE_TETRAHEDER:
                {
                    p0[0] = x_in[cl[el[inbox[i]]]];
                    p0[1] = y_in[cl[el[inbox[i]]]];
                    p0[2] = z_in[cl[el[inbox[i]]]];

                    p1[0] = x_in[cl[el[inbox[i]] + 1]];
                    p1[1] = y_in[cl[el[inbox[i]] + 1]];
                    p1[2] = z_in[cl[el[inbox[i]] + 1]];

                    p2[0] = x_in[cl[el[inbox[i]] + 2]];
                    p2[1] = y_in[cl[el[inbox[i]] + 2]];
                    p2[2] = z_in[cl[el[inbox[i]] + 2]];

                    p3[0] = x_in[cl[el[inbox[i]] + 3]];
                    p3[1] = y_in[cl[el[inbox[i]] + 3]];
                    p3[2] = z_in[cl[el[inbox[i]] + 3]];

                    if (isin_tetra(p, p0, p1, p2, p3) == 1)
                    {
                        k = inbox[i];

                        /*********************************************************
                     if(lookForInit == 1)
                     cerr << "return: inital cell found\n";
                     *********************************************************/
                        return k;
                    }
                }
                break;

                default:
                    cerr << "return: wrong type\n";
                    return -2;
                } // switch ends
            } // for loop over cells inside the box ends (and if statement)

        SetElementChecked[CurrentBlock] = 1;
#ifdef _VERBOSE_MODE_
        cerr << "CurrentBlock: " << CurrentBlock << endl;
        cerr << "SetElementChecked:";
        for (i = 0; i < NoOfBlocks; i++)
            cerr << " " << SetElementChecked[i];
        cerr << endl;
#endif

        for (i = 0; i < NoOfBlocks && SetElementChecked[i] == 1; i++)
            ;
        if (i == NoOfBlocks)
            CheckedAllBlocks = 1;
        else
        {
            CurrentBlock = i;
            coDoUnstructuredGrid *p_thisgrid = (coDoUnstructuredGrid *)(grid_in_list[CurrentBlock]);
            p_thisgrid->getGridSize(&numelem, &numconn, &numcoord);
            p_thisgrid->getAddresses(&el, &cl, &x_in, &y_in, &z_in);
            p_thisgrid->getTypeList(&tl);
            // ooo strcpy(lastGrid,p_thisgrid->getName());
            cuc_list.retrieve(p_thisgrid, &cuc_count, &cuc, &cuc_pos, &gpt_nbr_count);
            // p_thisgrid->getNeighborList(&cuc_count, &cuc, &cuc_pos);
            if (strcmp(dtype, "STRVDT") == 0)
            {
                coDoVec3 *p_thisdata = (coDoVec3 *)(v_data_in_list[CurrentBlock]);
                p_thisdata->getAddresses(&u_in, &v_in, &w_in);
            }
            else if (strcmp(dtype, "USTVDT") == 0)
            {
                coDoVec3 *p_thisdata = (coDoVec3 *)(uv_data_in_list[CurrentBlock]);
                p_thisdata->getAddresses(&u_in, &v_in, &w_in);
            }
        }
    } // end for loop over inbox_count elements
    if (onceIsEnoughInit == 0)
    {
        onceIsEnoughInit = 1;
        Covise::sendInfo("Could not find initial cells for some trajectory or trajectories");
    }
    //       cerr << "return: no cell found\n";
    return -1;
}

void Application::computeStartPoints()
{
    int i, j;
    int n0, n1;
    float v[3], o[3];
    float v0[3], v1[3];

    //   float myDir[3];

    float s, r;
    float s0, s1;
    //float sD2;

    o[0] = startpoint1[0];
    o[1] = startpoint1[1];
    o[2] = startpoint1[2];

    v[0] = startpoint2[0] - o[0];
    v[1] = startpoint2[1] - o[1];
    v[2] = startpoint2[2] - o[2];

    if (numstart != 1)
    {
        s = 1.0f / (float)(numstart - 1); // numstart could be 1!!!!
    }
    else
    {
        startStyle = 1;
        s = 0.0;
    }
    if (v[0] == 0.0 && v[1] == 0.0 && v[1] == 0.0)
    {
        startStyle = 1;
    }

    // here we go
    switch (startStyle)
    {

    case 2: // plane
    {
        // given:  startpoint1, startpoint2, startDirection, startNormal
        //
        // we have to compute: s0, s1 first, then n0, n1
        //

        // normize all vectors first

        r = 1.0f / sqrt((startDirection[0] * startDirection[0]) + (startDirection[1] * startDirection[1]) + (startDirection[2] * startDirection[2]));
        startDirection[0] *= r;
        startDirection[1] *= r;
        startDirection[2] *= r;

        r = v[0] * startDirection[0] + v[1] * startDirection[1] + v[2] * startDirection[2];

        v0[0] = startDirection[0] * r;
        v0[1] = startDirection[1] * r;
        v0[2] = startDirection[2] * r;

        v1[0] = v[0] - v0[0];
        v1[1] = v[1] - v0[1];
        v1[2] = v[2] - v0[2];

        /* This does not work sometimes... (even divisions by 0 are possible)
         r = 1.0 / sqrt((startNormal[0]*startNormal[0])+(startNormal[1]*startNormal[1])+(startNormal[2]*startNormal[2]));
         startNormal[0] *= r;
         startNormal[1] *= r;
         startNormal[2] *= r;

         myDir[0] = startNormal[1]*startDirection[2] - startNormal[2]*startDirection[1];
         myDir[1] = startNormal[2]*startDirection[0] - startNormal[0]*startDirection[2];
         myDir[2] = startNormal[0]*startDirection[1] - startNormal[1]*startDirection[0];
         // that should be normized by definition, so we have nothing to do

         // the following algorithm is standard, though it is taken
         // from GraphicGems I, side 304
         //

         // v1 x v2 = v0  (for speed up)
         v0[0] = startDirection[1]*myDir[2] - startDirection[2]*myDir[1];
         v0[1] = startDirection[2]*myDir[0] - startDirection[0]*myDir[2];
         v0[2] = startDirection[0]*myDir[1] - startDirection[1]*myDir[0];

         // v1 = p2 - p1
         v1[0] = startpoint2[0] - startpoint1[0];
         v1[1] = startpoint2[1] - startpoint1[1];
         v1[2] = startpoint2[2] - startpoint1[2];

         // compute r = det(v1, myDir, v0)
         r = (v1[0]*myDir[1]*v0[2])+(myDir[0]*v0[1]*v1[2])+(v0[0]*v1[1]*myDir[2]) - \ 
         (v0[0]*myDir[1]*v1[2])-(v1[0]*v0[1]*myDir[2])-(myDir[0]*v1[1]*v0[2]);

         // and divide by v0^2
         s0 = (v0[0]*v0[0]) + (v0[1]*v0[1]) + (v0[2]*v0[2]);
         if( s0==0.0 )
         fprintf(stderr, "s0 = 0.0 !!!\n");

         r /= s0;

         // so upper left point is at
         s1 = -r;     	// ulp = startpoint1 + s1*myDir

         // and compute r = det(v1, startDirection, v0)
         r = (v1[0]*startDirection[1]*v0[2])+(startDirection[0]*v0[1]*v1[2])+(v0[0]*v1[1]*startDirection[2]) - \ 
         (v0[0]*startDirection[1]*v1[2])-(v1[0]*v0[1]*startDirection[2])-(startDirection[0]*v1[1]*v0[2]);
         r /= s0;

         // so lower right point is at
         s0 = r;	// lrp = startpoint1 + s0*startDirection

         // we flipped 'em somehow....damn
         r = s0;
         s0 = -s1;
         s1 = -r;

         fprintf(stderr, "s0: %19.16f   s1: %19.16f\n", s0, s1);

         // now compute "stepsize" and "stepcount"

         n1 = (int)sqrt(fabsf(((float)numstart)*(s1/s0)));
         n0 = numstart / n1;

         s0 /= ((float)n0);
         s1 /= ((float)n1);

         v0[0] = startDirection[0];
         v0[1] = startDirection[1];
         v0[2] = startDirection[2];

         v1[0] = myDir[0];
         v1[1] = myDir[1];
         v1[2] = myDir[2];

         fprintf(stderr, "numStart will be %d\n", n1*n0);

         numstart = n1*n0;
         //numStartX = n0;
         //numStartY = n1;

         */
        r = sqrt(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]);
        s = sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
        if (r > s)
        {
            n1 = int(sqrt(numstart * s / r));
            if (n1 <= 1)
                n1 = 2;
            n0 = numstart / n1;
            if (n0 <= 1)
                n0 = 2;
        }
        else
        {
            n0 = int(sqrt(numstart * r / s));
            if (n0 <= 1)
                n0 = 2;
            n1 = numstart / n0;
            if (n1 <= 1)
                n1 = 2;
        }
        numstart = n0 * n1;
        s0 = 1.0f / (n0 - 1);
        s1 = 1.0f / (n1 - 1);

        for (j = 0; j < n1; j++)
            for (i = 0; i < n0; i++)
            {
                xStart[i + (j * n0)] = o[0] + v0[0] * s0 * (float)i + v1[0] * s1 * (float)j;
                yStart[i + (j * n0)] = o[1] + v0[1] * s0 * (float)i + v1[1] * s1 * (float)j;
                zStart[i + (j * n0)] = o[2] + v0[2] * s0 * (float)i + v1[2] * s1 * (float)j;
            }
    }
    break;

    default: // connecting line between both startpoints
    {
        for (i = 0; i < numstart; i++)
        {
            xStart[i] = o[0] + v[0] * s * (float)i;
            yStart[i] = o[1] + v[1] * s * (float)i;
            zStart[i] = o[2] + v[2] * s * (float)i;
        }
    }
    break;
    }

    // done
    return;
}

void Application::setCurrentBlock(int i)
{

    CurrentBlock = i;
    coDoUnstructuredGrid *p_thisgrid = (coDoUnstructuredGrid *)(grid_in_list[CurrentBlock]);
    p_thisgrid->getGridSize(&numelem, &numconn, &numcoord);
    p_thisgrid->getAddresses(&el, &cl, &x_in, &y_in, &z_in);
    p_thisgrid->getTypeList(&tl);
    cuc_list.retrieve(p_thisgrid, &cuc_count, &cuc, &cuc_pos, &gpt_nbr_count);
    if (strcmp(dtype, "STRVDT") == 0)
    {
        coDoVec3 *p_thisdata = (coDoVec3 *)(v_data_in_list[CurrentBlock]);
        p_thisdata->getAddresses(&u_in, &v_in, &w_in);
    }
    else if (strcmp(dtype, "USTVDT") == 0)
    {
        coDoVec3 *p_thisdata = (coDoVec3 *)(uv_data_in_list[CurrentBlock]);
        p_thisdata->getAddresses(&u_in, &v_in, &w_in);
    }
}

/*
void Application::fillNaehe(int block,ia<Naehe>& naehe_list){
int i,j,last;
float cx,cy,cz;

if(naehe_list[block].good == 1) return;

naehe_list[block].good = 1;
for(i=0;i<numelem;++i){
naehe_list[block].center_x[i] = 0.0;
naehe_list[block].center_y[i] = 0.0;
naehe_list[block].center_z[i] = 0.0;
naehe_list[block].radius2[i] = 0.0;
if(i==numelem-1){
last=numconn;
} else {
last=cl[i+1];
}
for(j=cl[i];j<last;++j){
naehe_list[block].center_x[i] += x_in[j];
naehe_list[block].center_y[i] += y_in[j];
naehe_list[block].center_z[i] += z_in[j];
}
naehe_list[block].center_x[i] /= (last - cl[i]);
naehe_list[block].center_y[i] /= (last - cl[i]);
naehe_list[block].center_z[i] /= (last - cl[i]);
cx=naehe_list[block].center_x[i];
cy=naehe_list[block].center_y[i];
cz=naehe_list[block].center_z[i];
for(j=cl[i];j<last;++j){
naehe_list[block].radius2[i] += (cx-x_in[j])*(cx-x_in[j]);
naehe_list[block].radius2[i] += (cy-y_in[j])*(cy-y_in[j]);
naehe_list[block].radius2[i] += (cz-z_in[j])*(cz-z_in[j]);
}
}
}
*/
//  read_velocity(actual_cell,phead,new_velocity);
int read_velocity(int actual_cell, const float *px, float *velocity)
{
    int i, j, is_in_subdomain = 0;
    int tmp_el[5], tmp_cl[20], tmp_count;
    int actual_cell_type = 1;
    float p0[3];
    float p1[3];
    float p2[3];
    float p3[3];
    float v0[3];
    float v1[3];
    float v2[3];
    float v3[3];

    switch (tl[actual_cell])
    {

    case TYPE_HEXAEDER:
    {
        hex2tet(actual_cell_type, actual_cell, tmp_el, tmp_cl);
        tmp_count = 5;
    }
    break;

    case TYPE_PRISM:
    {
        prism2tet(actual_cell_type, actual_cell, tmp_el, tmp_cl);
        tmp_count = 3;
    }
    break;

    case TYPE_PYRAMID:
    {
        pyra2tet(actual_cell_type, actual_cell, tmp_el, tmp_cl);
        tmp_count = 2;
    }
    break;

    case TYPE_TETRAHEDER:
    {
        tmp_el[0] = 0;
        for (j = 0; j < 4; j++)
        {
            tmp_cl[j] = cl[el[actual_cell] + j];
        }
        tmp_count = 1;
    }
    break;

    default:
    {
        return -1;
    }
    }

    for (i = 0; i < tmp_count; i++)
    {
        //find head of trace inside subdomain
        p0[0] = x_in[tmp_cl[tmp_el[i]]];
        p0[1] = y_in[tmp_cl[tmp_el[i]]];
        p0[2] = z_in[tmp_cl[tmp_el[i]]];

        p1[0] = x_in[tmp_cl[tmp_el[i] + 1]];
        p1[1] = y_in[tmp_cl[tmp_el[i] + 1]];
        p1[2] = z_in[tmp_cl[tmp_el[i] + 1]];

        p2[0] = x_in[tmp_cl[tmp_el[i] + 2]];
        p2[1] = y_in[tmp_cl[tmp_el[i] + 2]];
        p2[2] = z_in[tmp_cl[tmp_el[i] + 2]];

        p3[0] = x_in[tmp_cl[tmp_el[i] + 3]];
        p3[1] = y_in[tmp_cl[tmp_el[i] + 3]];
        p3[2] = z_in[tmp_cl[tmp_el[i] + 3]];

        is_in_subdomain = isin_tetra(px, p0, p1, p2, p3);
        if (is_in_subdomain == 1)
        {
            break;
        }
    }
    //if found, compute the streamline in the single cell
    if (is_in_subdomain == 1)
    {
        // vorticity field from data set
        v0[0] = u_in[tmp_cl[tmp_el[i]]];
        v0[1] = v_in[tmp_cl[tmp_el[i]]];
        v0[2] = w_in[tmp_cl[tmp_el[i]]];

        v1[0] = u_in[tmp_cl[tmp_el[i] + 1]];
        v1[1] = v_in[tmp_cl[tmp_el[i] + 1]];
        v1[2] = w_in[tmp_cl[tmp_el[i] + 1]];

        v2[0] = u_in[tmp_cl[tmp_el[i] + 2]];
        v2[1] = v_in[tmp_cl[tmp_el[i] + 2]];
        v2[2] = w_in[tmp_cl[tmp_el[i] + 2]];

        v3[0] = u_in[tmp_cl[tmp_el[i] + 3]];
        v3[1] = v_in[tmp_cl[tmp_el[i] + 3]];
        v3[2] = w_in[tmp_cl[tmp_el[i] + 3]];

        // We introduce the same sign flipping
        // used in the case at issue
        if (intpol_dir == 2)
        {
            for (j = 0; j < 3; j++)
            {
                v0[j] *= -1;
                v1[j] *= -1;
                v2[j] *= -1;
                v3[j] *= -1;
            }
        }

        vort_tet(px, p0, p1, p2, p3, v0, v1, v2, v3, velocity);
        return 0;
    }
    else
    {
        return -1;
    }
}

// Check if the difference is greater than some relative amount...
int check_diff(const float *old_velocity, const float *new_velocity)
{
    float diff[3];
    float magnitudes[2];

    // A negative Speed_thres is to be interpreted
    // as if it were infinite, thus achieving the
    // effect that a wall is never detected
    if (Speed_thres < 0.0)
        return 1;

    magnitudes[0] = sqrt(new_velocity[0] * new_velocity[0] + new_velocity[1] * new_velocity[1] + new_velocity[2] * new_velocity[2]);
    magnitudes[1] = sqrt(old_velocity[0] * old_velocity[0] + old_velocity[1] * old_velocity[1] + old_velocity[2] * old_velocity[2]);

    diff[0] = new_velocity[0] - old_velocity[0];
    diff[1] = new_velocity[1] - old_velocity[1];
    diff[2] = new_velocity[2] - old_velocity[2];

    return (sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            < Speed_thres * Max_of(magnitudes, 2));
}

int ExpandSetList(const coDistributedObject *const *objects, int h_many, const char *type,
                  ia<const coDistributedObject *> &out_array)
{
    int i, j;
    int count = 0;
    const char *obj_type;
    out_array.schleifen();
    for (i = 0; i < h_many; ++i)
    {
        obj_type = objects[i]->getType();
        if (strcmp(obj_type, type) == 0)
        {
            out_array[count++] = objects[i];
        }
        else if (strcmp(obj_type, "SETELE") == 0)
        {
            ia<const coDistributedObject *> partial;
            const coDistributedObject *const *in_set_elements;
            int in_set_len;
            in_set_elements = ((const coDoSet *)(objects[i]))->getAllElements(&in_set_len);
            int add = ExpandSetList(in_set_elements, in_set_len, type, partial);
            for (j = 0; j < add; ++j)
            {
                out_array[count++] = partial[j];
            }
        }
    }
    return count;
}

// the Names below refer to output objects!!!
Application::HandleTimeSteps::HandleTimeSteps(const coDistributedObject *in[], const char *lineName,
                                              const char *dataName, const char *veloName)
    : outLineName(lineName)
    , outDataName(dataName)
    , outVeloName(veloName)
    , in_grid_elements(0)
    , in_velo_elements(0)
{
    int no_timesteps_velo;
    if (in[0] == NULL || in[1] == NULL || !(in[0]->objectOk()) || !(in[1]->objectOk()))
    {
        is_ok = 0;
        no_timesteps = 0;
        timeStepFlag = 0;
        Covise::sendError("Input objects are not OK");
    }
    else if (strcmp(in[0]->getType(), "SETELE") == 0 && strcmp(in[1]->getType(), "SETELE") == 0 && in[0]->getAttribute("TIMESTEP"))
    {
        in_grid_elements = ((coDoSet *)(in[0]))->getAllElements(&no_timesteps);
        in_velo_elements = ((coDoSet *)(in[1]))->getAllElements(&no_timesteps_velo);
        if (no_timesteps != no_timesteps_velo)
        {
            Covise::sendError("Number of time steps in grid and velocity object do not match");
            is_ok = 0;
            timeStepFlag = 1;
        }
        else
        {
            is_ok = 1;
            timeStepFlag = 1;
        }
    } // One of the input objects is not a set or the grid has no time steps -> Ignore time steps
    else
    {
        is_ok = 1;
        no_timesteps = 1;
        timeStepFlag = 0;
    }
}

// In case of time steps the in[?] elements have to point to
// the corresponding element in in_grid_elements or in_velo_elements,
// that have been set by the HandleTimeSteps constructor.
// line_name data_name and velo_name are to point to the strings
// in thisLineName thisDataName and thisVeloName respectively, which
// are set by this function
void Application::HandleTimeSteps::setTime(int no_step, const coDistributedObject *in[],
                                           char **line_name, char **data_name, char **velo_name)
{
    char buf[64];
    if (is_ok && timeStepFlag)
    {

        time_step = no_step;

        in[0] = in_grid_elements[no_step];
        in[1] = in_velo_elements[no_step];

        sprintf(buf, "_%d", no_step);
        thisLineName = outLineName + buf;
        thisDataName = outDataName + buf;
        thisVeloName = outVeloName + buf;

        *line_name = (char *)thisLineName.c_str();
        *data_name = (char *)thisDataName.c_str();
        *velo_name = (char *)thisVeloName.c_str();
    }
}

void Application::HandleTimeSteps::addToTArrays(const coDistributedObject **returnObjects)
{
    if (is_ok && timeStepFlag)
    {
        timeLineElements[time_step] = returnObjects[0];
        timeDataElements[time_step] = returnObjects[1];
        timeVeloElements[time_step] = returnObjects[2];
    }
}

void DestroyObject(coDistributedObject *p_obj)
{
    int no_ele;
    coDistributedObject *const *list;

    if (p_obj->isType("SETELE"))
    {
        list = const_cast<coDistributedObject *const *>(((coDoSet *)(p_obj))->getAllElements(&no_ele));
        int ele;
        for (ele = 0; ele < no_ele; ++ele)
            DestroyObject(list[ele]);
    }
    p_obj->destroy();
}

void Application::HandleTimeSteps::Destroy(void)
{
    if (is_ok && timeStepFlag)
    {
        int i;
        for (i = 0; i < time_step; ++i)
        {
            DestroyObject(const_cast<coDistributedObject *>(timeLineElements[i]));
            DestroyObject(const_cast<coDistributedObject *>(timeDataElements[i]));
            DestroyObject(const_cast<coDistributedObject *>(timeVeloElements[i]));
        }
    }
}

void Application::HandleTimeSteps::createOutput(ia<const coDistributedObject *> &returnObject)
{
    if (!is_ok || !timeStepFlag)
    {
        return;
    }
    timeLineElements[no_timesteps] = 0;
    timeDataElements[no_timesteps] = 0;
    timeVeloElements[no_timesteps] = 0;

    returnObject[0] = new coDoSet(outLineName, &timeLineElements[0]);
    returnObject[1] = new coDoSet(outDataName, &timeDataElements[0]);
    returnObject[2] = new coDoSet(outVeloName, &timeVeloElements[0]);

    char buf[64];
    sprintf(buf, "1 %d", no_timesteps);
    const_cast<coDistributedObject *>(returnObject[0])->addAttribute("TIMESTEP", buf);
    const_cast<coDistributedObject *>(returnObject[1])->addAttribute("TIMESTEP", buf);
    const_cast<coDistributedObject *>(returnObject[2])->addAttribute("TIMESTEP", buf);
}

#define _RADICAL_DUMMY_

void Application::createDummies(const char *Lines, const char *DataOut,
                                const char *VelOut,
                                ia<const coDistributedObject *> &returnObject,
                                char *buf)
{
    int *zahlen = 0;
    float *zero_f = 0;
#ifdef _RADICAL_DUMMY_
    // Hereby we create a stripped dummy
    coDoLines *lines = new coDoLines(Lines, 0, 0, 0);
    returnObject[0] = lines;
    returnObject[1] = new coDoFloat(DataOut, 0);
    returnObject[2] = new coDoVec3(VelOut, 0);
#else
    // Hereby we create dummies that have at least the initial point
    zahlen = new int[numstart];
    zero_f = new float[numstart];
    int i;
    for (i = 0; i < numstart; ++i)
    {
        zahlen[i] = i;
        zero_f[i] = 0.0;
    }

    coDoLines *lines = new coDoLines(Lines, numstart, xStart, yStart, zStart,
                                     numstart, zahlen, numstart, zahlen);
    returnObject[0] = lines;
    returnObject[1] = new coDoFloat(DataOut, numstart, zero_f);
    returnObject[2] = new coDoVec3(VelOut, numstart, zero_f, zero_f, zero_f);
#endif
    // attributes
    lines->addAttribute("COLOR", "red");
    // and attribute for interaction
    // depending on style !!!
    switch (startStyle)
    {
    case 2: // plane
        sprintf(buf, "P%s\n%s\n%s\n", Covise::get_module(),
                Covise::get_instance(), Covise::get_host());
        break;

    default: // line
        sprintf(buf, "T%s\n%s\n%s\n", Covise::get_module(),
                Covise::get_instance(), Covise::get_host());
        break;
    }
    lines->addAttribute("FEEDBACK", buf);
    delete[] zahlen;
    delete[] zero_f;
}

void Application::MinMaxList::load(coDoUnstructuredGrid *grid)
{
    if (strcmp(gridName, grid->getName()) == 0)
    {
        return;
    }
    else
    {
        strcpy(gridName, grid->getName());
    }

    grid->getGridSize(&no_e, &no_c, &no_p);
    grid->getAddresses(&el, &cl, &x_c, &y_c, &z_c);
    int elem;

    iaMinMax.clean();

    for (elem = 0; elem < no_e; elem++)
    {
        iaMinMax[elem].elem = elem; // this will be reordered afterwards...
        getMinMax(&iaMinMax[elem].min, &iaMinMax[elem].max, elem);
    }
    qsort(&iaMinMax[0], no_e, sizeof(MinMax), comparef);
}

void Application::MinMaxList::getMinMax(float *mn, float *mx, int elem)
{
    int max_node, node;
    *mn = FLT_MAX;
    *mx = FLT_MIN;
    if (elem == no_e - 1)
    {
        max_node = no_c;
    }
    else
    {
        max_node = el[elem + 1];
    }
    for (node = el[elem]; node < max_node; ++node)
    {
        switch (xyz)
        {
        case Application::X:
            if (*mn > x_c[cl[node]])
                *mn = x_c[cl[node]];
            if (*mx < x_c[cl[node]])
                *mx = x_c[cl[node]];
            break;
        case Application::Y:
            if (*mn > y_c[cl[node]])
                *mn = y_c[cl[node]];
            if (*mx < y_c[cl[node]])
                *mx = y_c[cl[node]];
            break;
        case Application::Z:
            if (*mn > z_c[cl[node]])
                *mn = z_c[cl[node]];
            if (*mx < z_c[cl[node]])
                *mx = z_c[cl[node]];
            break;
        }
    }
}

void Application::MinMaxList::search(float position, ia<int> &out)
{
    out.clean(); // important for intersect...
    int i, count = 0;
    for (i = 0; i < no_e; ++i)
    {
        if (iaMinMax[i].min > position)
            break;
        if (iaMinMax[i].max >= position)
            out[count++] = iaMinMax[i].elem;
    }
    qsort(&out[0], out.size(), sizeof(int), compare); // important for bsearch
}

// Xref Yref Zref have been ordered!!!!!!
void intersect(ia<int> &Xref, ia<int> &Yref, ia<int> &Zref, ia<int> &inbox)
{
    inbox.clean();
    int i, count = 0;
    if (Xref.size() <= Yref.size() && Xref.size() <= Zref.size())
    {
        for (i = 0; i < Xref.size(); ++i)
        {
            if (bsearch(&Xref[i], &Yref[0], Yref.size(), sizeof(int), compare) && bsearch(&Xref[i], &Zref[0], Zref.size(), sizeof(int), compare))
            {
                inbox[count++] = Xref[i];
            }
        }
    }
    else if (Yref.size() <= Xref.size() && Yref.size() <= Zref.size())
    {
        for (i = 0; i < Yref.size(); ++i)
        {
            if (bsearch(&Yref[i], &Xref[0], Xref.size(), sizeof(int), compare) && bsearch(&Yref[i], &Zref[0], Zref.size(), sizeof(int), compare))
            {
                inbox[count++] = Yref[i];
            }
        }
    }
    else
    {
        for (i = 0; i < Zref.size(); ++i)
        {
            if (bsearch(&Zref[i], &Xref[0], Xref.size(), sizeof(int), compare) && bsearch(&Zref[i], &Yref[0], Yref.size(), sizeof(int), compare))
            {
                inbox[count++] = Zref[i];
            }
        }
    }
}

void typicalDim(float *tmp_length)
{
    float xb[8], yb[8], zb[8];
    int i, elem_loop;
    int no_vert;
    int inc_loop = numelem / 1000;
    int count_elems = 0;

    tmp_length[0] = 0.0;
    tmp_length[1] = 0.0;
    tmp_length[2] = 0.0;
    if (inc_loop == 0)
        inc_loop = 1;
    for (elem_loop = 0; elem_loop < numelem; elem_loop += inc_loop)
    {
        if (elem_loop < numelem - 1)
        {
            no_vert = el[elem_loop + 1] - el[elem_loop];
        }
        else
        {
            no_vert = numconn - el[elem_loop];
        }
        for (i = 0; i < no_vert; i++)
        {
            xb[i] = x_in[cl[el[elem_loop] + i]];
            yb[i] = y_in[cl[el[elem_loop] + i]];
            zb[i] = z_in[cl[el[elem_loop] + i]];
        }

        tmp_length[0] += Max_of(xb, no_vert) - Min_of(xb, no_vert);
        tmp_length[1] += Max_of(yb, no_vert) - Min_of(yb, no_vert);
        tmp_length[2] += Max_of(zb, no_vert) - Min_of(zb, no_vert);
        count_elems++;
    }
    tmp_length[0] /= count_elems;
    tmp_length[1] /= count_elems;
    tmp_length[2] /= count_elems;
}
