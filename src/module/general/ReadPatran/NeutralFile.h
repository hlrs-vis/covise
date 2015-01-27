/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _NEUTRALFILE_H_
#define _NEUTRALFILE_H_

#include <do/coDoUnstructuredGrid.h>
#include <util/coviseCompat.h>
#include "istreamFTN.h"

int Max(int, int);
int idcmp(const void *id1, const void *id2);
int binsearch(int x, int *v, int n);

const int MAXLINE = 82; // maximum length of input line

// element shape
enum SHAPE
{
    BAR = 2,
    TRI,
    QUAD,
    TET,
    WEDGE = 7,
    HEX
};

class Element;
class Node;
class Component;

/* PATRAN Neutral File (formatted version) */
/* ( PATRAN Version 2.5 )                  */
class NeutralFile
{

private:
    char title[80];
    istreamFTN in;
    struct HeaderCard
    {
        int it; // Packet type;
        int id; // Identification number
        int iv; // additional ID
        int kc; // Card count
        int n1, n2, n3, n4, n5;
    } header;

    struct Summary
    {
        int nnodes; // Number of nodes
        int nelements; // Number of elements
        int nm; // Number of materials
        int np; // Number of element properties
        int ncf; // Number of coordinate frames
        char subtitle[80];
    } summary;

    int maxnode; // maximum node number
    int maxelem; // maximum element number
    Node *nodeTab;
    Element *elemTab;
    Component *compTab;
    int maxTemperatures;
    int numTemperatures;

public:
    // number of vertices for given cell shape
    int *IdElemRef;
    int *ElemIdRef;
    int maxId;

    static const int numVert[9];

    int num_nodes; // Number of nodes
    int num_elements; // Number of elements
    int num_components; // Number of components
    int num_connections;

    int *nodeMap; // array enables mapping of PATRAN Node ID
    // to COVISE vertex number

    int *elemMap; // array enables mapping of PATRAN Element ID
    // to COVISE element number

    // Member functions

    NeutralFile(const char *filename);
    ~NeutralFile();

    void eval_num_connections();

    // skipping function
    void skip_lines(ifstream &input, int numlines, bool may_be_empty = false);

    Node *findNodeID(int id);
    Element *findElementID(int id);

    int isValid()
    {
        return (num_nodes != 0);
    };

    // get
    int getMaxnode(void);
    int getMaxelem(void);
    inline SHAPE getShape(int elemNo);
    void getMesh(
        int *elPtr, int *clPtr, int *tlPtr,
        float *xPtr, float *yPtr, float *zPtr,
        int *typPtr);
    int hasTemperatures()
    {
        return numTemperatures;
    };
    void getTemperatures(float *temp);
};

class Node
{
    friend class NeutralFile;

private:
    int id; // Node ID
    float coord[3]; // Cartesian Coordinate of Node
    float temperature;

public:
    void set(int id);
};

class Element
{
    friend class NeutralFile;

private:
    int id; // Element ID
    SHAPE shape;
    int pid; // Property ID
    int lnodes[8]; // Element corner nodes

public:
    Element();
    // set and get
    void set(int id, SHAPE shp);
    inline SHAPE getShape();
};

class Component
{
    friend class NeutralFile;

public:
    typedef enum _TYPE
    {
        grid = 1,
        node = 5,
        bar,
        triangle,
        quadrilateral,
        tetrahedron,
        wedge = 11,
        hexahedron
    } TYPE;

private:
    int id; // Component number
    int num; // 2 times the number of data pairs
    char name[80]; // Component name
    struct dataPair
    {
        friend class Component;
        TYPE ntype;
        int id;
    } *tab;

public:
    Component();
    Component(int id, int num);
    ~Component();

    // operators
    Component &operator=(const Component &comp);

    // set
    void set(int id, int num);
};
#endif
