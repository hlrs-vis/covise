/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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

#include "CadmouldGrid.h"
#include <util/coviseCompat.h>

#undef DUMP

// to get the TYPE_TRIANGLE and TYPE_BAR constants
#include <do/coDoUnstructuredGrid.h>

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Static Variable initializers
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Inline/static utilities
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// non-reentrant function: clip a part out of a string
inline const char *partstring(const char *string, int start, int numchars)
{
    assert(numchars < 255);
    static char buffer[256];
    strncpy(buffer, string + start, numchars);
    buffer[numchars] = '\0';
    return buffer;
}

//
int CadmouldGrid::elemComp(const void *e1void, const void *e2void)
{
    const ElemRecord &e1 = *static_cast<const ElemRecord *>(e1void);
    const ElemRecord &e2 = *static_cast<const ElemRecord *>(e2void);

    // 1st level soting - after groups
    if (e1.groupID < e2.groupID)
        return -1;
    if (e1.groupID > e2.groupID)
        return 1;

    // in groups - sort after original node number
    if (e1.nodeNo < e2.nodeNo)
        return -1;
    if (e1.nodeNo > e2.nodeNo)
        return 1;

    return 0;
}

//
int CadmouldGrid::mapComp(const void *e1void, const void *e2void)
{
    const GroupMap &e1 = *static_cast<const GroupMap *>(e1void);
    const GroupMap &e2 = *static_cast<const GroupMap *>(e2void);

    // sort after original node number
    if (e1.nodeNo < e2.nodeNo)
        return -1;
    if (e1.nodeNo > e2.nodeNo)
        return 1;

    return 0;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Constructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CadmouldGrid::CadmouldGrid(const char *filename)
{
    /// if we don't get to the end of the C'tor, it's not ok.
    d_state = -1;

    // for d_tor...
    d_numGroups = 0;
    d_elem = NULL;
    d_group = NULL;
    d_map = NULL;

    // try to open the file
    FILE *file = fopen(filename, "r");

    // too bad - we can't omen it
    if (!file)
    {
        d_state = errno;
        return;
    }

    // now read the file - via line buffer to make sure we capture structure
    char linebuffer[256];

    // 1st: read the sizes
    char *res = fgets(linebuffer, 255, file);
    if (!res)
    {
        d_state = errno;
        fclose(file);
        return;
    }
    linebuffer[255] = '\0'; // make sure it's terminated

    int kenner; // might be in, but will then be ignored
    int readFields = sscanf(linebuffer, "%d %d %d", &d_numVert, &d_numElem, &kenner);

#ifdef DUMP
    // and dump it
    FILE *dump = fopen("dump", "w");
    fprintf(dump, "Cadmould dump of file %s\n\n", filename);
    fprintf(dump, "Sizes: %d nodes, %d vertices\n\n", d_numElem, d_numVert);
#endif

    // we must have read at least two fields here
    if (readFields < 2)
    {
        fclose(file);
        return;
    }

    // allocate the field
    d_elem = new ElemRecord[d_numElem];
    if (!d_elem)
    {
        fclose(file);
        return;
    }

    ///////  Read Nodes //////////////////////////////////////////

    // read node data
    for (int i = 0; i < d_numElem; ++i)
    {
        // read a line
        res = fgets(linebuffer, 255, file);
        if (!res)
        {
            d_state = errno;
            fclose(file);
            return;
        }
        linebuffer[255] = '\0'; // make sure it's terminates

        // analyse fortran fields: 3I10, F10.3, I2
        //FORTRAN!
        d_elem[i].node[0] = atoi(partstring(linebuffer, 0, 10)) - 1;
        d_elem[i].node[1] = atoi(partstring(linebuffer, 10, 10)) - 1;
        d_elem[i].node[2] = atoi(partstring(linebuffer, 20, 10)) - 1;
        d_elem[i].thickness = atoi(partstring(linebuffer, 30, 10));
        d_elem[i].groupID = atoi(partstring(linebuffer, 40, 2));
        d_elem[i].nodeNo = i;
    }

    // and now we sort it by groups
    qsort(d_elem, d_numElem, sizeof(ElemRecord), CadmouldGrid::elemComp);

#ifdef DUMP
    fprintf(dump, "%-7s: %-7s %-7s %-7s  %-7s  %-2s  %7s\n--------------------------------------------------------------\n",
            "idx", "node[0]", "node[1]", "node[2]", "thickn", "GR", "nodeNo");
    for (int i = 0; i < d_numElem; ++i)
    {
        fprintf(dump, "%7i: %7d %7d %7d  %7f  %2d  %7i\n",
                i,
                d_elem[i].node[0],
                d_elem[i].node[1],
                d_elem[i].node[2],
                d_elem[i].thickness,
                d_elem[i].groupID,
                d_elem[i].nodeNo);
    }
#endif

    // get number of groups
    int actGrp = d_elem[0].groupID;
    d_numGroups = 1;
    for (int i = 1; i < d_numElem; ++i)
        if (d_elem[i].groupID != actGrp)
        {
            d_numGroups++;
            actGrp = d_elem[i].groupID;
        }

    // create back mapping and count group sizes
    d_map = new GroupMap[d_numElem];
    d_group = new Group[d_numGroups];

    // set 1st group here - easier implementation.
    actGrp = d_elem[0].groupID;
    int actGrpIdx = 0;
    int actValIdx = -1;
    d_group[0].groupID = d_elem[0].groupID;
    d_group[0].numElem = 0;
    d_group[0].firstIdx = 0;

    // fill in everything - remember elements are sorted by group
    for (int i = 0; i < d_numElem; ++i)
    {
        // create new group
        if (d_elem[i].groupID != actGrp)
        {
            d_group[actGrpIdx].lastIdx = i - 1;
            actGrpIdx++;
            d_group[actGrpIdx].firstIdx = i;
            d_group[actGrpIdx].groupID = d_elem[i].groupID;
            d_group[actGrpIdx].numElem = 1;

            actGrp = d_elem[i].groupID;
            actValIdx = 0;
        }
        // new element in existing group
        else
        {
            d_group[actGrpIdx].numElem++;
            actValIdx++;
        }

        // create backward mapping
        d_map[i].groupIdx = actGrpIdx;
        d_map[i].valIdx = actValIdx;
        d_map[i].nodeNo = d_elem[i].nodeNo;
    }
    d_group[actGrpIdx].lastIdx = d_numElem - 1;

#ifdef DUMP
    fprintf(dump, "\n @@@ Group List : %d Groups\n\n", d_numGroups);
    for (int i = 0; i < d_numGroups; i++)
        fprintf(dump, "Group %7d: %6d elements, indices %6d - %6d\n",
                d_group[i].groupID, d_group[i].numElem,
                d_group[i].firstIdx, d_group[i].lastIdx);
#endif

    // now sort the backward mappings
    qsort(d_map, d_numElem, sizeof(GroupMap), CadmouldGrid::mapComp);

#ifdef DUMP
    fprintf(dump, "\n @@@ Mapping : %d Elements\n\n", d_numElem);
    fprintf(dump, "%6s: %8s %8s %8s\n-----------------------------------------------------\n",
            "idx", "nodeNo", "groupIdx", "valIdx");
    for (int i = 0; i < d_numElem; i++)
        fprintf(dump, "%6d: %8d %8d %8d\n", i,
                d_map[i].nodeNo, d_map[i].groupIdx, d_map[i].valIdx);
#endif

    ///////  Read Vertices //////////////////////////////////////

    float *tempX = new float[d_numVert];
    float *tempY = new float[d_numVert];
    float *tempZ = new float[d_numVert];

#ifdef DUMP
    fprintf(dump, "\n @@@ %d Vertices:\n\n%5s: %10s %10s %10s\n------------------------------------------------------\n",
            d_numVert, "idx", "X", "Y", "Z");
#endif

    for (int i = 0; i < d_numVert; ++i)
    {
        // read a line
        res = fgets(linebuffer, 255, file);
        if (!res)
        {
            d_state = errno;
            return;
        }
        linebuffer[255] = '\0'; // make sure it's terminates

        // analyse fortran fields: 3F10.3
        tempX[i] = atof(partstring(linebuffer, 0, 10));
        tempY[i] = atof(partstring(linebuffer, 10, 10));
        tempZ[i] = atof(partstring(linebuffer, 20, 10));
#ifdef DUMP
        fprintf(dump, "%5i: %10.3f %10.3f %10.3f\n",
                i, tempX[i], tempY[i], tempZ[i]);
#endif
    }

    fclose(file);

    /// find group's vertices

    // the map from group's counting to overall, allow node=-1 for zylinder elems
    const int UNUSED = -999;

    int *map = new int[d_numElem + 2];
    map += 2; // allow -2 as index

#ifdef DUMP
    fprintf(dump, "\n @@@ Group grids:\n\n");
#endif

    // loop over group indices
    for (int grp = 0; grp < d_numGroups; grp++)
    {
        Group &group = d_group[grp];

        // mark all vertices unused
        for (int i = -2; i < d_numElem; i++)
            map[i] = UNUSED; // node=-2 reserved

        // mark and count all vertices this group uses
        int numUsed = 0;
        for (int i = group.firstIdx; i <= group.lastIdx; i++)
        {
            ElemRecord &elem = d_elem[i];
            for (int n = 0; n < 3; n++)
            {
                // we didn't have this vertex before
                if (map[elem.node[n]] == UNUSED)
                {
                    map[elem.node[n]] = numUsed;
                    numUsed++;
                }
            }
        }
        group.numVert = numUsed;

        // now create the group's vertex tables
        group.x = new float[numUsed];
        group.y = new float[numUsed];
        group.z = new float[numUsed];

        // and the map for storing the global numbers of my local vertices
        group.globalNode = new int[numUsed];
        for (int i = 0; i < numUsed; ++i)
            group.globalNode[i] = UNUSED;

        // fill all used vertices into group's vertex table
        for (int i = 0; i < d_numVert; i++)
            if (map[i] >= 0)
            {
                group.x[map[i]] = tempX[i];
                group.y[map[i]] = tempY[i];
                group.z[map[i]] = tempZ[i];
                group.globalNode[map[i]] = i;
            }

        // now create element lists of group
        group.elemList = new int[group.numElem];
        // won't use all when zylinders are there
        group.connList = new int[group.numElem * 3];
        group.typeList = new int[group.numElem];
        group.thickness = new float[group.numElem];

        // loop over group's part of element list
        int *elemPtr = group.elemList;
        int *connPtr = group.connList;
        int *typePtr = group.typeList;
        float *thickPtr = group.thickness;
        int connCount = 0;

        for (int el = group.firstIdx; el <= group.lastIdx; ++el)
        {

            ElemRecord &elem = d_elem[el];

            if (elem.node[2] == -2) // zylinder element
            {
                *elemPtr++ = connCount;
                *connPtr++ = map[elem.node[0]];
                *connPtr++ = map[elem.node[1]];
                *typePtr++ = TYPE_BAR;
                connCount += 2;
            }
            else // triangle element
            {
                *elemPtr++ = connCount;
                *connPtr++ = map[elem.node[0]];
                *connPtr++ = map[elem.node[1]];
                *connPtr++ = map[elem.node[2]];
                *typePtr++ = TYPE_TRIANGLE;
                connCount += 3;
            }

            *thickPtr++ = elem.thickness;
        }
        group.numConn = connCount;

#ifdef DUMP

        fprintf(dump, "\n +++ Group %d: %d elem, %d conn, %d vertices\n",
                group.groupID, group.numElem, group.numConn, group.numVert);

        fprintf(dump, "\n Element list:\n");
        for (int i = 0; i < group.numElem; i++)
        {
            if (group.typeList[i] == TYPE_BAR)
                fprintf(dump, "%6i: [BAR  ] %6d %6d\n", i,
                        group.connList[group.elemList[i]],
                        group.connList[group.elemList[i] + 1]);
            else if (group.typeList[i] == TYPE_TRIANGLE)
                fprintf(dump, "%6i: [TRIANG] %6d %6d %6d\n", i,
                        group.connList[group.elemList[i]],
                        group.connList[group.elemList[i] + 1],
                        group.connList[group.elemList[i] + 2]);
            else
                fprintf(dump, "%6i: [ILL]", i);
        }

        fprintf(dump, "\n Mapping local vertices -> global\n");
        for (int i = 0; i < group.numVert; i++)
        {
            fprintf(dump, "%6d -> %6d\n", i, group.globalNode[i]);
        }
#endif

    } // loop over groups

#ifdef DUMP
    fclose(dump);
#endif

    // we moved the map base pointer for elem=-1
    delete[](map - 2);

    delete[] tempX;
    delete[] tempY;
    delete[] tempZ;

    /// seems to be ok...
    d_state = 0;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Destructors
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CadmouldGrid::~CadmouldGrid()
{
    int i;
    for (i = 0; i < d_numGroups; i++)
    {
        delete[] d_group[i].elemList;
        delete[] d_group[i].connList;
        delete[] d_group[i].typeList;
        delete[] d_group[i].x;
        delete[] d_group[i].y;
        delete[] d_group[i].z;
        delete[] d_group[i].thickness;
        delete[] d_group[i].globalNode;
    }
    delete[] d_group;
    delete[] d_elem;
    delete[] d_map;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Operations
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++  Attribute request/set functions
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int CadmouldGrid::getState()
{
    return d_state;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Get number of groups
int CadmouldGrid::numGroups()
{
    return d_numGroups;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// get grid size of group (0..numGroups(()-1)
void CadmouldGrid::gridSizes(int groupNo,
                             int &numElem, int &numConn, int &numVert)
{
    numElem = d_group[groupNo].numElem;
    numConn = d_group[groupNo].numConn;
    numVert = d_group[groupNo].numVert;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// fill grid tables in given fields (0..numGroups(()-1)
void CadmouldGrid::copyTables(int groupNo, int *elemList,
                              int *typeList, int *connList,
                              float *x, float *y, float *z)
{
    Group &grp = d_group[groupNo];
    memcpy(elemList, grp.elemList, sizeof(int) * grp.numElem);
    memcpy(typeList, grp.typeList, sizeof(int) * grp.numElem);
    memcpy(connList, grp.connList, sizeof(int) * grp.numConn);
    memcpy(x, grp.x, sizeof(float) * grp.numVert);
    memcpy(y, grp.y, sizeof(float) * grp.numVert);
    memcpy(z, grp.z, sizeof(float) * grp.numVert);
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// get the Cadmould index of this group
int CadmouldGrid::getGroupID(int groupNo)
{
    return d_group[groupNo].groupID;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// copy thickness data into given field
void CadmouldGrid::copyThickness(int groupNo, float *thick)
{
    memcpy(thick, d_group[groupNo].thickness,
           sizeof(float) * d_group[groupNo].numElem);
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// get the Vertex mapping of a group into the global field
const int *CadmouldGrid::globalVertex(int group)
{
    return d_group[group].globalNode;
}
