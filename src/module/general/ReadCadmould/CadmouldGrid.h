/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CADMOULD_GRID_H_
#define __CADMOULD_GRID_H_
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS CadmouldGrid
//
// This class @@@
//
// Initial version: 2002-03-25 [we]
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
//

/**
 * Class @@@
 *
 */
class CadmouldGrid
{
public:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Constructors / Destructor
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /** Constructor
       *
       */
    CadmouldGrid(const char *filename);

    /// Destructor : virtual in case we derive objects
    virtual ~CadmouldGrid();

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Operations
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Attribute request/set functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /** Check whether grid was read correctly
       * @return  0=ok, >0 = errno of failing op, -1 other error
       */
    int getState();

    /// Get number of groups
    int numGroups();

    /// get grid size of group (0..numGroups(()-1)
    void gridSizes(int groupNo, int &numElem, int &numConn, int &numVert);

    /// fill grid tables in given fields (0..numGroups(()-1)
    void copyTables(int groupNo, int *elemList, int *typeList, int *connList,
                    float *x, float *y, float *z);

    /// get the Cadmould index of this group
    int getGroupID(int groupNo);

    /// copy thickness data into given field
    void copyThickness(int groupNo, float *thick);

    /// get the Vertex mapping of a group into the global field
    const int *globalVertex(int group);

    /// get number of vertices
    int getNumVert()
    {
        return d_numVert;
    }

protected:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Attributes
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // state of this object: =0 means ok, otherwise errno
    // of failing operation or -1 for other problems
    int d_state;

    // grid sizes
    int d_numElem; // number of elements in grid
    int d_numVert; // number of vertices in grid

    // element list type
    typedef struct
    {
        int node[3]; // vertices, -1 un [2] means line element
        float thickness; // element thickness or beam diameter
        int groupID; // group number, <0 for inlet parts
        int nodeNo; // we re-sort, here original element number
    } ElemRecord;

    // element list
    ElemRecord *d_elem;

    // number of existing groups
    int d_numGroups;

    // groups : group ID and complete grids in Covise storage
    typedef struct
    {

        // Cadmould group ID
        int groupID;

        // complete covise grid
        int numElem;
        int numConn;
        int numVert;
        int *elemList;
        int *connList;
        int *typeList;
        float *x, *y, *z;
        float *thickness;

        // 1st and last idx in global orted list
        int firstIdx, lastIdx;

        // my node numbers in the global field
        int *globalNode;
    } Group;

    Group *d_group;

    // backward mapping from global element to group/element in group
    typedef struct
    {
        int nodeNo; // index of data value - we sort by this
        int groupIdx; // index of group in group list
        int valIdx; // index of value in its group
    } GroupMap;

    GroupMap *d_map;

private:
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  Internally used functions
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++  prevent auto-generated bit copy routines by default
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// comparison fct. for qsort
    static int elemComp(const void *e1void, const void *e2void);

    /// comparison fct. for qsort
    static int mapComp(const void *e1void, const void *e2void);

    /// Copy-Constructor: NOT IMPLEMENTED
    CadmouldGrid(const CadmouldGrid &);

    /// Assignment operator: NOT IMPLEMENTED
    CadmouldGrid &operator=(const CadmouldGrid &);

    /// Default constructor: NOT IMPLEMENTED
    CadmouldGrid();
};
#endif
