/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _APPLICATION_H
#define _APPLICATION_H
/**************************************************************************\ 
 **                                                           (C)1996 RUS  **
 **                                                                        **
 ** Description:  COVISE Tracer_USG application module	                  **
 **                                                                        **
 **                                                                        **
 **                             (C) 1996                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Uwe Woessner, Oliver Heck                                     **
 **                                                                        **
 **                                                                        **
 ** Date:  12.02.96  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <appl/CoviseAppModule.h>
#include <util/coviseCompat.h>
#include <do/coDoIntArr.h>
#include <do/coDoLines.h>
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>
#include "coIA.h"

// holds arbitrary mesh cell type pairs and the cells that have this type

class CellPair
{
private:
    enum
    {
        CellBlocks = 1000
    };

public:
    int cellNo; // cell Type Number
    int NoOfAMesh; // Number of cell types that interface to this cell type
    int *listOfAMesh; // list of cell types that interface to this cell type
    int NoOfCells; // number of cells that have this type
    int *listOfCells; // list of cells that have this type
    CellPair(int No)
        : cellNo(No)
        , NoOfAMesh(0)
        , listOfAMesh(NULL)
        , NoOfCells(0)
        , listOfCells(NULL){};
    CellPair(int No, int other)
        : cellNo(No)
        , NoOfAMesh(1)
        , NoOfCells(0)
        , listOfCells(NULL)
    {
        listOfAMesh = new int[1];
        listOfAMesh[0] = other;
    }
    ~CellPair()
    {
        delete[] listOfAMesh;
        delete[] listOfCells;
    }
    void addAMeshCell(int other)
    {
        int *tmp = new int[NoOfAMesh + 1];
        for (int i = 0; i < NoOfAMesh; i++)
            tmp[i] = listOfAMesh[i];
        delete[] listOfAMesh;
        listOfAMesh = tmp;
        listOfAMesh[NoOfAMesh] = other;
        NoOfAMesh++;
    }
    void addCell(int other);
    void addCellPair(CellPair *other);
};

// sl: infinite array template class
/*
template<class T> class ia{
private:
   T *array;
   int length;
   int max_reference;
   ia(const ia&);  // copy constr. and Zuweisung nicht implementiert:
                   // Nachmacher sind nicht erwuenscht
   ia& operator=(const ia&);
public:

int num_elems() const{return max_reference+1;}
ia(int i=0){
array=0;
if(i<0) i=0;
length=i;
if(i){
try{
array=new T[i];
} catch(...){
cerr << "Memory allocation request for " << i << " elements failed"<<endl;
length=0;
throw;
}
}
max_reference = -1;
}

void clean(){
max_reference=-1;
}

void schleifen(){
delete [] array;
array=0;
length=0;
clean();
}

~ia(){delete [] array;}

T& operator[](int i){
if(i>max_reference) max_reference = i;
if(i>=length){
T *tmp;
int new_length;
if(length==0){
new_length=i+1;
} else {
new_length= ((i/length)+1)*length;
}
try{
tmp=new T[new_length];
} catch(...){
cerr << "Memory allocation request for " << new_length << " elements failed"<<endl;
throw;
}
if(length){
memcpy(tmp,array,sizeof(T)*length); // Not valid for all classes!!!
delete [] array;
}
array=tmp;
length=new_length;
return array[i];
} else if(i<0){
cerr << "Array referenced with negative index... Delivering the 0th term"<<endl;
if(length==0){
length=1;
array=new T[1];
}
return array[0];
} else {
return array[i];
}
}
};
*/

struct neighbourhood
{
    char *name_;
    int cuc_count_;
    int *cuc_;
    int *cuc_pos_;
    int *gpt_nbr_count_;
    neighbourhood()
    {
        name_ = (char *)"";
        cuc_count_ = 0;
        gpt_nbr_count_ = cuc_pos_ = cuc_ = 0;
    }
    void load(coDoUnstructuredGrid *grid)
    {
        int i, numelem, numconn, numcoord;
        name_ = grid->getName();
        grid->getGridSize(&numelem, &numconn, &numcoord);
        if (numelem)
        {
            grid->getNeighborList(&cuc_count_, &cuc_, &cuc_pos_);
        }
        int *el, *cl;
        float *x_in, *y_in, *z_in;
        grid->getAddresses(&el, &cl, &x_in, &y_in, &z_in);
        gpt_nbr_count_ = new int[numcoord];
        memset(gpt_nbr_count_, 0, numcoord * sizeof(int));
        for (i = 0; i < numconn; i++)
        {
            gpt_nbr_count_[cl[i]]++;
        }
    }
    ~neighbourhood();
};

class ListNeighbourhood // this class works only with one unique instance
{
    static int doNotDelete;
    friend struct neighbourhood;
    ia<neighbourhood> array;

public:
    int search(coDoUnstructuredGrid *grid)
    {
        int i;
        if (array.size() == 0)
            return -1;
        for (i = 0; i < array.size(); ++i)
        {
            if (strcmp(grid->getName(), array[i].name_) == 0)
                return i;
        }
        return -1;
    }
    void schleifen()
    {
        array.schleifen();
    }
    void retrieve(coDoUnstructuredGrid *grid, int *count, int **cuco, int **cuco_pos,
                  int **gpto_nbr_count)
    {
        int where = search(grid);
        if (where < 0)
        {
            where = array.size();
            doNotDelete = 1;
            array[where].load(grid);
            doNotDelete = 0;
        }
        *count = array[where].cuc_count_;
        *cuco = array[where].cuc_;
        *cuco_pos = array[where].cuc_pos_;
        *gpto_nbr_count = array[where].gpt_nbr_count_;
    }
};

int ListNeighbourhood::doNotDelete = 0;

neighbourhood::~neighbourhood()
{
    if (ListNeighbourhood::doNotDelete == 0)
    {
        delete[] cuc_;
        delete[] cuc_pos_;
        delete[] gpt_nbr_count_;
    }
}

//class Application : public CoviseAppModule
class Application
{
public:
    enum Coords
    {
        X = 0,
        Y = 1,
        Z = 2
    };
    enum
    {
        FLINTSTONE = 1,
        THREE_LISTS = 2
    };

    class MinMaxList
    {
        Application::Coords xyz;
        char gridName[128];
        int no_e;
        int no_c;
        int no_p;
        int *el;
        int *cl;
        float *x_c;
        float *y_c;
        float *z_c;
        struct MinMax
        {
            float min;
            float max;
            int elem;
        };
        ia<MinMax> iaMinMax;
        void getMinMax(float *, float *, int);

    public:
        MinMaxList(Application::Coords direction)
            : xyz(direction)
        {
            gridName[0] = '\0';
        }
        void load(coDoUnstructuredGrid *);
        void search(float, ia<int> &);
    };

    enum // initial size for some "infinite" arrays
    {
        SMALL_GRID = 2000
    };
    // that keep in compute the trajectories

private:
    void createDummies(const char *Lines, const char *DataOut,
                       const char *VelOut,
                       ia<const coDistributedObject *> &returnObject,
                       char *buf);

    class HandleTimeSteps
    {
    private:
        int is_ok;
        int no_timesteps;
        int time_step;
        int timeStepFlag;
        std::string outLineName; // The names for output objects
        std::string outDataName;
        std::string outVeloName;
        std::string thisLineName; // The names for elements of output sets in case of time steps
        std::string thisDataName;
        std::string thisVeloName;
        // array of pointers
        const coDistributedObject *const *in_grid_elements;
        // to elements in time steps
        const coDistributedObject *const *in_velo_elements;
        // In these infinte arrays below we keep the elements of
        // the output sets in case of time steps (addToTArrays
        // writes to these objects)
        // They will be used by createOutput
        ia<const coDistributedObject *> timeLineElements;
        ia<const coDistributedObject *> timeDataElements;
        ia<const coDistributedObject *> timeVeloElements;

    public:
        // a.i.
        HandleTimeSteps(const coDistributedObject *[], const char *, const char *, const char *);
        int ok(void)
        {
            return is_ok;
        }
        int how_many(void)
        {
            return no_timesteps;
        }
        // a.i.
        void setTime(int, const coDistributedObject *[], char **, char **, char **);
        void addToTArrays(const coDistributedObject **); // a. i.
        void createOutput(ia<const coDistributedObject *> &);
        void Destroy(void);
    };

    /*
          struct Naehe{
             int good;
             ia<float> radius2;
             ia<float> center_x;
             ia<float> center_y;
             ia<float> center_z;
             Naehe(){good=0;}
          };
          void fillNaehe(int i,ia<Naehe>& );
      */
    void setCurrentBlock(int /*,ia<Naehe>&*/);

    // callback stub functions
    //
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

    // private member functions
    //
    void compute(void *callbackData);
    void quit(void *callbackData);

    // private member functions
    //
    // called whenever the module is executed
    //      coDistributedObject **compute(coDistributedObject **, char **);
    float *xStart, *yStart, *zStart;
    int NoOfArbMesh;
    CellPair **ArbMeshIntList;

public:
    Application(int argc, char *argv[])
    {
        Covise::set_module_description("Generate streamlines from an unstr. data set");
        Covise::add_port(INPUT_PORT, "meshIn", "UnstructuredGrid", "input mesh");
        Covise::add_port(INPUT_PORT, "dataIn", "Vec3|Vec3", "input data");
        Covise::add_port(INPUT_PORT, "cellTypeIn", "IntArr", "cell type data");
        Covise::set_port_required("cellTypeIn", 0);
        Covise::add_port(OUTPUT_PORT, "lines", "Lines", "Streamline");
        Covise::add_port(OUTPUT_PORT, "dataOut", "Float", "Output data");
        Covise::add_port(OUTPUT_PORT, "velocOut", "Vec3", "Output velocity");
        Covise::add_port(PARIN, "no_startp", "IntSlider", "number of startpoints");
        Covise::add_port(PARIN, "startpoint1", "FloatVector", "Startpoint1 for Streamline");
        Covise::add_port(PARIN, "startpoint2", "FloatVector", "Startpoint2 for Streamline");

        /*
                     Covise::add_port(PARIN, "normal", "FloatVector", "...");
                     Covise::set_port_default("normal", "0.0 0.0 1.0");
         */

        Covise::add_port(PARIN, "direction", "FloatVector", "...");
        Covise::set_port_default("direction", "1.0 0.0 0.0");

        // Mesh and Data file names
        char buffer1[128];
        char *cov_path = getenv("COVISEDIR");
        if (cov_path)
            sprintf(buffer1, "%s/data/ *", cov_path);
        else
            sprintf(buffer1, "/ *");
        Covise::add_port(PARIN, "amesh_path", "Browser", "Amesh path");
        Covise::set_port_default("amesh_path", buffer1);
        Covise::add_port(PARIN, "option", "Choice", "Method of interpolation");
        Covise::add_port(PARIN, "stp_control", "Choice", "stepsize control");
        Covise::add_port(PARIN, "tdirection", "Choice", "direction of interpolation");
        Covise::add_port(PARIN, "reduce", "Choice", "Point reduction on output geometry");
        Covise::add_port(PARIN, "whatout", "Choice", "Component of output data");
        Covise::add_port(PARIN, "trace_eps", "FloatScalar", "Epsilon, be careful with this");
        Covise::add_port(PARIN, "trace_len", "FloatScalar", "Maximum length of trace");
        Covise::add_port(PARIN, "startStyle", "Choice", "how to compute starting-points");
        Covise::set_port_default("startStyle", "1 line plane");
        //Covise::set_port_default("nbrs_comp","2 pre_process on_the_fly");
        Covise::set_port_default("no_startp", "1 10 2");
        Covise::set_port_default("startpoint1", "1.0 1.0 1.0");
        Covise::set_port_default("startpoint2", "1.0 2.0 1.0");
        Covise::set_port_default("option", "2 rk2 rk4");
        Covise::set_port_default("stp_control", "1 position 5<ang<12 2<ang<6");
        //		     Covise::set_port_default("tdirection","3 forward backward both xyplane cube");
        Covise::set_port_default("tdirection", "3 forward backward both ");
        Covise::set_port_default("reduce", "1 off 3deg 5deg");
        Covise::set_port_default("whatout", "1 mag v_x v_y v_z number");
        Covise::set_port_default("trace_eps", "0.00001");
        Covise::set_port_default("trace_len", "1.0");
        Covise::add_port(PARIN, "loopDelta", "FloatScalar", "Loop detection");
        Covise::set_port_default("loopDelta", "0.0");
        Covise::add_port(PARIN, "MaxPoints", "IntScalar", "Max. length in point lists");
        Covise::set_port_default("MaxPoints", "10000");
        Covise::add_port(PARIN, "MaxSearchFactor", "FloatScalar", "Max. factor for the searching of the initial point");
        Covise::set_port_default("MaxSearchFactor", "100.0");
        Covise::add_port(PARIN, "Speed_thres", "FloatScalar", "Relative speed threshold for wall detection");
        Covise::set_port_default("Speed_thres", "0.2");
        Covise::add_port(PARIN, "Connectivity", "Boolean", "Elements are seamless connected or not");
        Covise::set_port_default("Connectivity", "TRUE");
        /*
                    Covise::add_port(PARIN,"searchMethod","Choice","Method for finding the initial cell");
                    Covise::set_port_default("searchMethod","1 Box Lists");
         */

        Covise::init(argc, argv);
        Covise::set_start_callback(Application::computeCallback, this);
        Covise::set_quit_callback(Application::quitCallback, this);

        //         char *in_names[] = {"meshIn", "dataIn", "cellTypeIn", NULL};
        //         char *out_names[] = {"lines", "dataOut", "velocOut", NULL};
        //         setPortNames( in_names, out_names );

        //         setCallbacks();
    }

    void run()
    {
        Covise::main_loop();
    }
    void computeStartPoints();
    int createAmeshList(char *);

    ~Application()
    {
    }
};
#endif // _APPLICATION_H
