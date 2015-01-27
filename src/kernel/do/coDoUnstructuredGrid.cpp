/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coDoUnstructuredGrid.h"
#include "coDoOctTree.h"
#include "covise_gridmethods.h"

// in this list the TYPE_... definitions in covise_unstrgrd.h can be
// used to return the number of vertices for this kind of element
namespace covise
{

int UnstructuredGrid_Num_Nodes[20] = {
    // eam:  This kind of list cannot handle polyhedral cells due to the fact
    // that they contain an arbitrary number of vertices.  The value -1 is
    // used in this case only as a reference to the cell type (TYPE_POLYHEDRON)
    // while on the remaining cases the value denotes the number of
    // vertices of the cell.

    0, 2, 3, 4, 4, 5, 6, 8, 0, 0, 1,
    -1, 0, 0, 0, 0, 0, 0, 0, 0
};
}

using namespace covise;

/**
removed from header:

coDoUnstructuredGrid(const char *n, int nelem, int nconn, int ncoord,
                    int *el, int *cl, float *xc, float *yc, float *zc,
                    int *tl, int nneighbor, int *nl, int *nli);
coDoUnstructuredGrid(const char *n, int nelem, int nconn, int ncoord,
int *el, int *cl, float *xc, float *yc, float *zc,
int nneighbor, int *nl, int *nli);
**/

/***********************************************************************\ 
 **                                                                     **
 **   Unstructured classe Routines                  Version: 1.0        **
 **                                                                     **
 **                                                                     **
 **   Description  : Classes for the handling of unstructured grids     **
 **                  in a distributed manner.                           **
 **                                                                     **
 **   Classe      : coDoUnstructuredGrid                                 **
 **                                                                     **
 **                                                                     **
 **   Copyright (C) 1993     by University of Stuttgart                 **
 **                             Computer Center (RUS)                   **
 **                             Allmandring 30                          **
 **                             7000 Stuttgart 80                       **
 **                                                                     **
 **                                                                     **
 **   Author       : Uwe Woessner                                       **
 **                                                                     **
 **   History      :                                                    **
 **                  10.02.95  Ver 1.0                                  **
 **                  27.01.97  Andreas Werner: delete connectivity      **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

#undef DEBUG

coDistributedObject *coDoUnstructuredGrid::virtualCtor(coShmArray *arr)
{
    coDistributedObject *ret;
    ret = new coDoUnstructuredGrid(coObjInfo(), arr);
    return ret;
}

int coDoUnstructuredGrid::getObjInfo(int no, coDoInfo **il) const
{
    if (no == SHM_OBJ)
    {
        (*il)[0].description = "Number of Elements";
        (*il)[1].description = "Number of Connectivity";
        (*il)[2].description = "Number of Coordinates";
        (*il)[3].description = "Elements";
        (*il)[4].description = "Connections";
        (*il)[5].description = "X Coordinates";
        (*il)[6].description = "Y Coordinates";
        (*il)[7].description = "Z Coordinates";
        (*il)[8].description = "Element Types";
        (*il)[9].description = "Number of Neighbours";
        (*il)[10].description = "Neighbour List";
        (*il)[11].description = "Neighbour Index";
        return SHM_OBJ;
    }
    else
    {
        print_error(__LINE__, __FILE__, "number wrong for object info");
        return 0;
    }
}

coDoUnstructuredGrid::~coDoUnstructuredGrid()
{
    // DO NOT delete these arrays here!!!
    //delete [] lnl;
    //delete [] lnli;
    /*
      if(!oct_tree){
         MakeOctTree();
      }
      delete oct_tree;
   */
}

coDoUnstructuredGrid::coDoUnstructuredGrid(const coObjInfo &info, coShmArray *arr)
    : coDoGrid(info)
    , oct_tree(NULL)
    , lnl(NULL)
    , lnli(NULL)
{
    setType("UNSGRD", "UNSTRUCTURED GRID");
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
}

coDoUnstructuredGrid::coDoUnstructuredGrid(const coObjInfo &info,
                                           int nelem, int nconn, int ncoord,
                                           int *el, int *cl, float *xc, float *yc,
                                           float *zc)
    : coDoGrid(info)
    , oct_tree(NULL)
    , hastypes(0)
    , hasneighbors(0)
    , lnl(NULL)
    , lnli(NULL)
{
    covise_data_list dl[SHM_OBJ];

    setType("UNSGRD", "UNSTRUCTURED GRID");
#ifdef DEBUG
    cerr << "vor store_shared coDoUnstructuredGrid\n";
#endif
    x_coord.set_length(ncoord);
    y_coord.set_length(ncoord);
    z_coord.set_length(ncoord);
    elements.set_length(nelem);
    connections.set_length(nconn);
    elementtypes.set_length(0);
    neighborlist.set_length(0);
    neighborindex.set_length(0);
    /*
      char octname[64];
      sprintf(octname,"%s_OctTree",n);
      oct_tree = new coDoOctTree(octname,nelem,nconn,ncoord,el,cl,xc,yc,zc);
   */
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&numelem;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&numconn;
    dl[2].type = INTSHM;
    dl[2].ptr = (void *)&numcoord;
    dl[3].type = INTSHMARRAY;
    dl[3].ptr = (void *)&elements;
    dl[4].type = INTSHMARRAY;
    dl[4].ptr = (void *)&connections;
    dl[5].type = FLOATSHMARRAY;
    dl[5].ptr = (void *)&x_coord;
    dl[6].type = FLOATSHMARRAY;
    dl[6].ptr = (void *)&y_coord;
    dl[7].type = FLOATSHMARRAY;
    dl[7].ptr = (void *)&z_coord;
    dl[8].type = INTSHMARRAY;
    dl[8].ptr = (void *)&elementtypes;
    dl[9].type = INTSHM;
    dl[9].ptr = (void *)&numneighbor;
    dl[10].type = INTSHMARRAY;
    dl[10].ptr = (void *)&neighborlist;
    dl[11].type = INTSHMARRAY;
    dl[11].ptr = (void *)&neighborindex;

    new_ok = store_shared_dl(SHM_OBJ, dl) != 0;

    if (!new_ok)
        return;
    int *etmp, *ctmp;
    float *xtmp, *ytmp, *ztmp;
    getAddresses(&etmp, &ctmp, &xtmp, &ytmp, &ztmp);
    memcpy(xtmp, xc, ncoord * sizeof(float));
    memcpy(ytmp, yc, ncoord * sizeof(float));
    memcpy(ztmp, zc, ncoord * sizeof(float));
    memcpy(etmp, el, nelem * sizeof(int));
    memcpy(ctmp, cl, nconn * sizeof(int));
    numconn = nconn;
    numelem = nelem;
    numcoord = ncoord;
    hastypes = false;
    hasneighbors = false;
    numneighbor = 0;
}

coDoUnstructuredGrid::coDoUnstructuredGrid(const coObjInfo &info,
                                           int nelem, int nconn, int ncoord,
                                           int *el, int *cl, float *xc, float *yc,
                                           float *zc, int *tl)
    : coDoGrid(info)
    , oct_tree(NULL)
    , lnl(NULL)
    , lnli(NULL)
{
    covise_data_list dl[SHM_OBJ];

    setType("UNSGRD", "UNSTRUCTURED GRID");
#ifdef DEBUG
    cerr << "vor store_shared coDoUnstructuredGrid\n";
#endif
    x_coord.set_length(ncoord);
    y_coord.set_length(ncoord);
    z_coord.set_length(ncoord);
    elements.set_length(nelem);
    connections.set_length(nconn);
    elementtypes.set_length(nelem);
    neighborlist.set_length(0);
    neighborindex.set_length(ncoord);
    /*
      char octname[64];
      sprintf(octname,"%s_OctTree",n);
      oct_tree = new coDoOctTree(octname,nelem,nconn,ncoord,el,cl,xc,yc,zc);
   */
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&numelem;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&numconn;
    dl[2].type = INTSHM;
    dl[2].ptr = (void *)&numcoord;
    dl[3].type = INTSHMARRAY;
    dl[3].ptr = (void *)&elements;
    dl[4].type = INTSHMARRAY;
    dl[4].ptr = (void *)&connections;
    dl[5].type = FLOATSHMARRAY;
    dl[5].ptr = (void *)&x_coord;
    dl[6].type = FLOATSHMARRAY;
    dl[6].ptr = (void *)&y_coord;
    dl[7].type = FLOATSHMARRAY;
    dl[7].ptr = (void *)&z_coord;
    dl[8].type = INTSHMARRAY;
    dl[8].ptr = (void *)&elementtypes;
    dl[9].type = INTSHM;
    dl[9].ptr = (void *)&numneighbor;
    dl[10].type = INTSHMARRAY;
    dl[10].ptr = (void *)&neighborlist;
    dl[11].type = INTSHMARRAY;
    dl[11].ptr = (void *)&neighborindex;

    new_ok = store_shared_dl(SHM_OBJ, dl) != 0;

    if (!new_ok)
        return;

    int *etmp, *ctmp, *ttmp;
    float *xtmp, *ytmp, *ztmp;
    getAddresses(&etmp, &ctmp, &xtmp, &ytmp, &ztmp);
    getTypeList(&ttmp);
    memcpy(xtmp, xc, ncoord * sizeof(float));
    memcpy(ytmp, yc, ncoord * sizeof(float));
    memcpy(ztmp, zc, ncoord * sizeof(float));
    memcpy(etmp, el, nelem * sizeof(int));
    memcpy(ctmp, cl, nconn * sizeof(int));
    memcpy(ttmp, tl, nelem * sizeof(int));
    numconn = nconn;
    numelem = nelem;
    numcoord = ncoord;
    hastypes = true;
    hasneighbors = false;
    numneighbor = 0;
}

coDoUnstructuredGrid::coDoUnstructuredGrid(const coObjInfo &info,
                                           int nelem, int nconn, int ncoord, int ht)
    : coDoGrid(info)
    , oct_tree(NULL)
    , lnl(NULL)
    , lnli(NULL)
{
    covise_data_list dl[SHM_OBJ];

    setType("UNSGRD", "UNSTRUCTURED GRID");
#ifdef DEBUG
    cerr << "vor store_shared coDoUnstructuredGrid\n";
#endif
    x_coord.set_length(ncoord);
    y_coord.set_length(ncoord);
    z_coord.set_length(ncoord);
    elements.set_length(nelem);
    connections.set_length(nconn);
    if (ht)
        elementtypes.set_length(nelem);
    else
        elementtypes.set_length(0);
    neighborlist.set_length(0);
    neighborindex.set_length(0);
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&numelem;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&numconn;
    dl[2].type = INTSHM;
    dl[2].ptr = (void *)&numcoord;
    dl[3].type = INTSHMARRAY;
    dl[3].ptr = (void *)&elements;
    dl[4].type = INTSHMARRAY;
    dl[4].ptr = (void *)&connections;
    dl[5].type = FLOATSHMARRAY;
    dl[5].ptr = (void *)&x_coord;
    dl[6].type = FLOATSHMARRAY;
    dl[6].ptr = (void *)&y_coord;
    dl[7].type = FLOATSHMARRAY;
    dl[7].ptr = (void *)&z_coord;
    dl[8].type = INTSHMARRAY;
    dl[8].ptr = (void *)&elementtypes;
    dl[9].type = INTSHM;
    dl[9].ptr = (void *)&numneighbor;
    dl[10].type = INTSHMARRAY;
    dl[10].ptr = (void *)&neighborlist;
    dl[11].type = INTSHMARRAY;
    dl[11].ptr = (void *)&neighborindex;

    new_ok = store_shared_dl(SHM_OBJ, dl) != 0;

    if (!new_ok)
        return;
    numconn = nconn;
    numelem = nelem;
    numcoord = ncoord;
    hastypes = ht;
    numneighbor = 0;
    hasneighbors = false;
}

coDoUnstructuredGrid *coDoUnstructuredGrid::cloneObject(const coObjInfo &newinfo) const
{
    int n_elem, n_conn, n_coord;
    getGridSize(&n_elem, &n_conn, &n_coord);
    float *c[3];
    int *e_l, *c_l, *t_l;
    getAddresses(&e_l, &c_l, &c[0], &c[1], &c[2]);
    getTypeList(&t_l);
    return new coDoUnstructuredGrid(newinfo, n_elem, n_conn, n_coord,
                                    e_l, c_l, c[0], c[1], c[2], t_l);
}

int coDoUnstructuredGrid::rebuildFromShm()
{
    covise_data_list dl[SHM_OBJ];

    if (shmarr == 0L)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
    }

    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&numelem;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&numconn;
    dl[2].type = INTSHM;
    dl[2].ptr = (void *)&numcoord;
    dl[3].type = INTSHMARRAY;
    dl[3].ptr = (void *)&elements;
    dl[4].type = INTSHMARRAY;
    dl[4].ptr = (void *)&connections;
    dl[5].type = FLOATSHMARRAY;
    dl[5].ptr = (void *)&x_coord;
    dl[6].type = FLOATSHMARRAY;
    dl[6].ptr = (void *)&y_coord;
    dl[7].type = FLOATSHMARRAY;
    dl[7].ptr = (void *)&z_coord;
    dl[8].type = INTSHMARRAY;
    dl[8].ptr = (void *)&elementtypes;
    dl[9].type = INTSHM;
    dl[9].ptr = (void *)&numneighbor;
    dl[10].type = INTSHMARRAY;
    dl[10].ptr = (void *)&neighborlist;
    dl[11].type = INTSHMARRAY;
    dl[11].ptr = (void *)&neighborindex;
    return restore_shared_dl(SHM_OBJ, dl);

    /* if(elementtypes.get_length() > 0)
         hastypes = true; */
}

// void coDoUnstructuredGrid::computeNeighborList()
// {
//    int ja,j,i,offset, *tmpl1;
//    int next_elem_index;
//
//    el = (int *)elements.getDataPtr();
//    cl = (int *)connections.getDataPtr();
//    tl = (int *)elementtypes.getDataPtr();
//    lnli = new int[(int)numcoord+1];
//    tmpl1 = new int[(int)numcoord];
//    memset(lnli, 0, (int)numcoord*sizeof(int));
//    for(i=0;i<(int)numconn;i++)
//    {
//       lnli[cl[i]]++;
//    }
//
//    //create the position list for cuc list -> cuc_pos
//    j=ja=0;
//    for(i=0;i<(int)numcoord;i++)
//    {
//       j += lnli[i];
//       lnli[i]=ja;
//       tmpl1[i]=ja;
//       ja=j;
//    }
//    lnli[i]=ja;
//    lnl = new int[(int)numconn];                   //size for array
//
//    //fill the cuc list
//    for(i=0;i<(int)numelem;i++)
//    {
//     // Polyhedral cells
//     if(tl[i] == TYPE_POLYHEDRON)
//     {
//         next_elem_index = (i < numelem) ? el[i + 1] : numconn;
//         offset = next_elem_index - i;
//         for(j=0;j<offset;j++)
//         {
//             lnl[tmpl1[cl[el[i]+j]]] = i;
//             tmpl1[cl[el[i]+j]]++;
//         }
//     }
//
//     // Standard cells
//     else
//     {
//         offset=UnstructuredGrid_Num_Nodes[tl[i]];
//         for(j=0;j<offset;j++)
//         {
//             lnl[tmpl1[cl[el[i]+j]]] = i;
//             tmpl1[cl[el[i]+j]]++;
//         }
//     }
//    }
//    numneighbor = numconn;
//    delete [] tmpl1;
// }

void coDoUnstructuredGrid::computeNeighborList() const
{

    int ja, j, i, offset, *tmpl1;
    int old_elem_index;
    int next_elem_index;

    el = (int *)elements.getDataPtr();
    cl = (int *)connections.getDataPtr();
    tl = (int *)elementtypes.getDataPtr();
    lnli = new int[(int)numcoord + 1];
    tmpl1 = new int[(int)numcoord];
    memset(lnli, 0, (int)numcoord * sizeof(int));

    // Calculation of the number of cells that contain a certain vertex
    int *used_vertex_list;

    used_vertex_list = new int[(int)numcoord];
    memset(used_vertex_list, -1, (int)numcoord * sizeof(int));

    for (i = 0; i < numelem; i++)
    {
        // Polyhedral cells
        if (UnstructuredGrid_Num_Nodes[tl[i]] == -1)
        {
            next_elem_index = (i < numelem - 1) ? el[i + 1] : numconn;
            for (j = el[i]; j < next_elem_index; j++)
            {
                if (used_vertex_list[cl[j]] != i)
                {
                    used_vertex_list[cl[j]] = i;
                    lnli[cl[j]]++;
                }
            }
        }

        // Standard cells
        else
        {
            next_elem_index = (i < numelem - 1) ? el[i + 1] : numconn;
            for (j = el[i]; j < next_elem_index; j++)
            {
                used_vertex_list[cl[j]] = i;
                lnli[cl[j]]++;
            }
        }
    }

    //create the position list for cuc list -> cuc_pos
    j = ja = 0;
    for (i = 0; i < (int)numcoord; i++)
    {
        j += lnli[i];
        lnli[i] = ja;
        tmpl1[i] = ja;
        ja = j;
    }
    lnli[i] = ja;
    lnl = new int[ja];
    memset(used_vertex_list, -1, (int)numcoord * sizeof(int));
    old_elem_index = el[0];

    //fill the cuc list
    for (i = 0; i < (int)numelem; i++)
    {
        // Polyhedral cells
        if (UnstructuredGrid_Num_Nodes[tl[i]] == -1)
        {
            next_elem_index = (i < numelem - 1) ? el[i + 1] : numconn;
            offset = next_elem_index - old_elem_index;
            for (j = 0; j < offset; j++)
            {
                if (used_vertex_list[cl[el[i] + j]] != i)
                {
                    used_vertex_list[cl[el[i] + j]] = i;
                    lnl[tmpl1[cl[el[i] + j]]] = i;
                    tmpl1[cl[el[i] + j]]++;
                }
            }
            old_elem_index = next_elem_index;
        }

        // Standard cells
        else
        {
            offset = UnstructuredGrid_Num_Nodes[tl[i]];
            for (j = 0; j < offset; j++)
            {
                used_vertex_list[cl[el[i] + j]] = i;
                lnl[tmpl1[cl[el[i] + j]]] = i; // eam:  lists all elements that contain a certain vertex consecutively
                tmpl1[cl[el[i] + j]]++;
            }
            old_elem_index += offset;
        }
    }
    delete[] tmpl1;
    delete[] used_vertex_list;
}

int coDoUnstructuredGrid::getNeighbor(int element, vector<int> face_nodes_list)
{
    int i;
    int j;
    int n;
    int ce;
    int n1;
    int next_elem_index;
    int check_sum;

    int *found_nodes;

    ce = -1;

    n1 = face_nodes_list[0];
    found_nodes = new int[face_nodes_list.size() - 1];

    for (i = lnli[n1]; i < lnli[n1 + 1]; i++)
    {
        ce = lnl[i];
        if (ce != element)
        {
            check_sum = 0;
            // Initialize to zero; n2, n3, ....., have not been found yet in the current cell
            memset(found_nodes, 0, (face_nodes_list.size() - 1) * sizeof(int));

            // Polyhedral cells
            if (UnstructuredGrid_Num_Nodes[tl[ce]] == -1)
            {
                next_elem_index = (ce < numelem) ? el[ce + 1] : numconn;
                for (n = 0; n < next_elem_index - el[ce]; n++)
                {
                    // Test from the second vertex on; first vertex has already been tested
                    // when searching in the neighbor list (lnli[n1]).
                    for (j = 1; j < face_nodes_list.size(); j++)
                    {
                        if (cl[el[ce] + n] == face_nodes_list[j])
                        {
                            if (found_nodes[j - 1] == 0)
                            {
                                check_sum++;
                                found_nodes[j - 1] = 1;
                                break;
                            }
                        }
                    }

                    // A total of three matching vertices should be enough
                    if (check_sum >= 2)
                    {
                        delete[] found_nodes;
                        return ce;
                    }
                }
            }

            // Standard cells
            else
            {
                for (n = 0; n < UnstructuredGrid_Num_Nodes[tl[ce]]; n++)
                {
                    for (j = 1; j < face_nodes_list.size(); j++)
                    {
                        if (cl[el[ce] + n] == face_nodes_list[j])
                        {
                            if (found_nodes[j - 1] == 0)
                            {
                                check_sum++;
                                found_nodes[j - 1] = 1;
                                break;
                            }
                        }
                    }

                    // A total of three matching vertices should be enough
                    if (check_sum >= 2)
                    {
                        delete[] found_nodes;
                        return ce;
                    }
                }
            }
        }
    }
    delete[] found_nodes;
    return -1;
}

// search for another cell than 'element' containing vertices of n1...n4
// or at least 3 polints
int coDoUnstructuredGrid::getNeighbor(int element, int n1, int n2, int n3, int n4)
{
    int f2, f3, f4, nf, ce = -1;
    int i, n;
    int next_elem_index;

    f2 = f3 = f4 = 0;
    for (i = lnli[n1]; i < lnli[n1 + 1]; i++)
    {
        ce = lnl[i];
        if (ce != element)
        {
            nf = 0;
            f2 = f3 = f4 = 0;
            // Polyhedral cells
            if (UnstructuredGrid_Num_Nodes[tl[ce]] == -1)
            {
                next_elem_index = (ce < numelem) ? el[ce + 1] : numconn;
                for (n = 0; n < next_elem_index - el[ce]; n++)
                {
                    if (cl[el[ce] + n] == n2)
                        f2 = 1;
                    else if (cl[el[ce] + n] == n3)
                        f3 = 1;
                    else if (cl[el[ce] + n] == n4)
                        f4 = 1;

                    // We test here because connectivity lists of polyhedral cells are very long
                    if (f2 + f3 + f4 >= 2)
                    {
                        return ce;
                    }
                }
            }

            // Standard cells
            else
            {
                for (n = 0; n < UnstructuredGrid_Num_Nodes[tl[ce]]; n++)
                {
                    if (cl[el[ce] + n] == n2)
                        nf++;
                    else if (cl[el[ce] + n] == n3)
                        nf++;
                    else if (cl[el[ce] + n] == n4)
                        nf++;
                }
            }

            //if(f2+f3+f4==3 || (tl[ce]<TYPE_HEXAEDER && f2+f3+f4>=2) )
            if (nf >= 2) // three matching vertices should always be good enough
                return ce;
        }
    }

    // we might have degenerated elements with the first node hanging try the second one
    for (i = lnli[n2]; i < lnli[n2 + 1]; i++)
    {
        ce = lnl[i];
        if (ce != element)
        {
            nf = 0;
            f2 = f3 = f4 = 0;

            // Polyhedral cells
            if (UnstructuredGrid_Num_Nodes[tl[ce]] == -1)
            {
                next_elem_index = (ce < numelem) ? el[ce + 1] : numconn;
                for (n = 0; n < next_elem_index - el[ce]; n++)
                {
                    if (cl[el[ce] + n] == n2)
                        f2 = 1;
                    else if (cl[el[ce] + n] == n3)
                        f3 = 1;
                    else if (cl[el[ce] + n] == n4)
                        f4 = 1;

                    // We test here because connectivity lists of polyhedral cells are very long
                    if (f2 + f3 + f4 >= 2)
                    {
                        return ce;
                    }
                }
            }

            // Standard cells
            else
            {
                for (n = 0; n < UnstructuredGrid_Num_Nodes[tl[ce]]; n++)
                {
                    if (cl[el[ce] + n] == n1)
                        nf++;
                    else if (cl[el[ce] + n] == n3)
                        nf++;
                    else if (cl[el[ce] + n] == n4)
                        nf++;
                }
            }

            //if(f2+f3+f4==3 || (tl[ce]<TYPE_HEXAEDER && f2+f3+f4>=2) )
            if (nf >= 2) // three matching vertices should always be ok
                return ce;
        }
    }
    return -1;
}

int coDoUnstructuredGrid::getNeighbor(int element, int n1, int n2, int n3)
{
    int f2, f3, ce = -1;
    int i, n;
    int next_elem_index;

    f2 = f3 = 0;
    for (i = lnli[n1]; i < lnli[n1 + 1]; i++)
    {
        ce = lnl[i];
        if (ce != element)
        {
            f2 = f3 = 0;

            // Polyhedral cells
            if (UnstructuredGrid_Num_Nodes[tl[ce]] == -1)
            {
                next_elem_index = (ce < numelem) ? el[ce + 1] : numconn;
                for (n = 0; n < next_elem_index - el[ce]; n++)
                {
                    if (cl[el[ce] + n] == n2)
                        f2 = 1;
                    else if (cl[el[ce] + n] == n3)
                        f3 = 1;
                    // We test here because connectivity lists of polyhedral cells are very long
                    if (f2 && f3)
                    {
                        return ce;
                    }
                }
            }

            // Standard cells
            else
            {
                for (n = 0; n < UnstructuredGrid_Num_Nodes[tl[ce]]; n++)
                {
                    if (cl[el[ce] + n] == n2)
                        f2 = 1;
                    else if (cl[el[ce] + n] == n3)
                        f3 = 1;
                }
            }

            if (f2 && f3)
            {
                break;
            }
        }
    }
    if (f2 && f3)
        return (ce);
    else
        return (-1);
}

int coDoUnstructuredGrid::getNeighbor(int element, int n1, int n2)
{
    int f2, ce = -1;
    int i, n;
    int next_elem_index;

    f2 = 0;
    for (i = lnli[n1]; i < lnli[n1 + 1]; i++)
    {
        ce = lnl[i];
        if (ce != element)
        {
            f2 = 0;

            // Polyhedral cells
            if (UnstructuredGrid_Num_Nodes[tl[ce]] == -1)
            {
                next_elem_index = (ce < numelem) ? el[ce + 1] : numconn;
                for (n = 0; n < next_elem_index - el[ce]; n++)
                {
                    if (cl[el[ce] + n] == n2)
                        f2 = 1;
                    // We test here because connectivity lists of polyhedral cells are very long
                    if (f2)
                    {
                        return ce;
                    }
                }
            }

            // Standard cells
            else
            {
                for (n = 0; n < UnstructuredGrid_Num_Nodes[tl[ce]]; n++)
                {
                    if (cl[el[ce] + n] == n2)
                        f2 = 1;
                }
            }

            if (f2)
            {
                break;
            }
        }
    }
    if (f2)
        return (ce);
    else
        return (-1);
}

int coDoUnstructuredGrid::getNeighbors(int element, int n1, int n2, int *neighbors)
{
    int f2, ce = -1;
    int i, n, num = -1;
    int next_elem_index;

    f2 = 0;
    for (i = lnli[n1]; i < lnli[n1 + 1]; i++)
    {
        ce = lnl[i];
        if (ce != element)
        {
            f2 = 0;

            // Polyhedral cells
            if (UnstructuredGrid_Num_Nodes[tl[ce]] == -1)
            {
                next_elem_index = (ce < numelem) ? el[ce + 1] : numconn;
                for (n = 0; n < next_elem_index - el[ce]; n++)
                {
                    if (cl[el[ce] + n] == n2)
                        f2 = 1;
                    // We test here because connectivity lists of polyhedral cells are very long
                    if (f2)
                    {
                        break;
                    }
                }
            }

            // Standard cells
            else
            {
                for (n = 0; n < UnstructuredGrid_Num_Nodes[tl[ce]]; n++)
                    if (cl[el[ce] + n] == n2)
                        f2 = 1;
            }

            if (f2)
            {
                num++;
                neighbors[i] = ce;
            }
        }
    }
    return (num);
}

int coDoUnstructuredGrid::getNeighbors(int element, int n1, int *neighbors)
{
    int ce = -1;
    int i, num = -1;

    for (i = lnli[n1]; i < lnli[n1 + 1]; i++)
    {
        ce = lnl[i];
        if (ce != element)
        {
            num++;
            neighbors[i] = ce;
        }
    }
    return num;
}

int coDoUnstructuredGrid::setSizes(int numElem, int numConn, int numCoord)
{
    if (numElem > numelem || numConn > numconn || numCoord > numcoord)
        return -1;

    numelem = numElem;
    numconn = numConn;
    numcoord = numCoord;

    return 0;
}

int coDoUnstructuredGrid::getCell(const float *point, float tolerance)
{
    coDoOctTree *cast_oct_tree = (coDoOctTree *)(oct_tree);
    if (cast_oct_tree == NULL)
    {
        char *objName = getName();
        char *newName = new char[strlen(objName) + 100];
        sprintf(newName, "%s_octTree_%d", objName, rand());
        MakeOctTree(newName);
        cast_oct_tree = (coDoOctTree *)(oct_tree);
    }
    const int *cell_list = cast_oct_tree->search(point);
    if (*cell_list == 0)
        return -1;
    int i;
    int cell = -1;
    float v_interp[3];
    for (i = 0; i < *cell_list; ++i)
    {
        cell = cell_list[i + 1];
        if (cast_oct_tree->IsInBBox(cell, numelem, point) && testACell(v_interp, point, cell, 0, 0, tolerance, NULL) == 0)
        {
            return cell;
        }
    }
    // not found
    return -1;
}

int coDoUnstructuredGrid::interpolateField(float *v_interp, const float *point,
                                           int *cell, int no_arrays, int array_dim, float tolerance,
                                           const float *const *velo) const
{
    const coDoOctTree *cast_oct_tree = (const coDoOctTree *)(oct_tree);
    if (*cell >= 0 && *cell < numelem && cast_oct_tree->IsInBBox(*cell, numelem, point)
        && testACell(v_interp, point, *cell, no_arrays, array_dim, tolerance,
                     velo) == 0)
    {
        return 0;
    }

    // if this test fails or *cell<0, apply general procedure...
    // use the oct-tree to get a list of candidate cells
    // if no cells, return -1
    // otherwise test cells
    // if the test fails for all cells, return -1
    const int *cell_list = cast_oct_tree->search(point);
    if (*cell_list == 0)
        return -1;
    int i;
    int old_cell = *cell;
    for (i = 0; i < *cell_list; ++i)
    {
        *cell = cell_list[i + 1];
        if (cast_oct_tree->IsInBBox(*cell, numelem, point) && testACell(v_interp, point, *cell, no_arrays, array_dim, tolerance, velo) == 0)
        {
            return 0;
        }
    }
    // not found
    *cell = old_cell;
    return -1;
}

int coDoUnstructuredGrid::mapScalarField(float *v_interp, const float *point,
                                         int *cell, int no_arrays, int array_dim,
                                         const float *const *velo)
{
#ifdef _DEBUG_FUNCTION_CALL_
    cout << "DO_UnstructuredGrid::mapScalarField function called..." << endl;
#endif

    mapInCell(v_interp, point, *cell, no_arrays, array_dim, velo);
    return 0;
}

int coDoUnstructuredGrid::testACell(float *v_interp, const float *point,
                                    int cell, int no_arrays, int array_dim, float tolerance,
                                    const float *const *velo) const
{
    // tetrahedronise
    int tmp_el[5], tmp_cl[20];
    float p0[3], p1[3], p2[3], p3[3];
    int i, j;
    // determine type of cell
    int cell_type;
    int *elem, *conn;
    float *x_in, *y_in, *z_in;
    getAddresses(&elem, &conn, &x_in, &y_in, &z_in);
    if (hasTypeList())
    {
        cell_type = elementtypes[cell];
    }
    else
    {
        int num_of_vert;
        if (cell < numelem - 1)
        {
            num_of_vert = elem[cell + 1] - elem[cell];
        }
        else
        {
            num_of_vert = numconn - elem[cell];
        }
        switch (num_of_vert)
        {
        case 8:
            cell_type = TYPE_HEXAEDER;
            break;
        case 6:
            cell_type = TYPE_PRISM;
            break;
        case 5:
            cell_type = TYPE_PYRAMID;
            break;
        case 4: // assume tetrahedra
            cell_type = TYPE_TETRAHEDER;
            break;
        default: // do not consider 2D, 1D or 0D elements
            return -1;
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Note:  in the case of standard cells the interpolation is performed using regular barycentric            //
    // coordinates.  Therefore the cells are subject initially to a decomposition process which divides        //
    // the cell into several tetrahedra.  Afterwards it should be determined in which tetrahedron is the     //
    // test point contained in order to perform the interpolation.  The decomposition process and in-cell  //
    // test could be avoided by using Shepard's Method in covise_gridmethods, however this procedure //
    // is not as accurate as barycentric interpolation.  In the case of polyhedral cells the only stable         //
    // interpolation method currently implemented is Shepard's Method.  A general interpolation method //
    // could be here implemented using barycentric coordinates extended to general (convex and          //
    // concave) polyhedra.                                                                                                                        //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // tetrahedronise
    switch (cell_type)
    {
    case TYPE_HEXAEDER:
    {
        // direct decomposition
        grid_methods::hex2tet(1, elem, conn, cell, tmp_el, tmp_cl);
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

            if (grid_methods::isin_tetra(point, p0, p1, p2, p3, tolerance) == 1)
            {
                if (velo)
                {
                    if (no_arrays == 3
                        && array_dim == 1)
                    {
                        grid_methods::interpolateVInHexa(v_interp, point,
                                                         velo, conn + elem[cell], x_in, y_in, z_in);
                    }
                    else
                    {
                        grid_methods::interpolateInHexa(v_interp, point,
                                                        no_arrays, array_dim, velo,
                                                        conn + elem[cell], x_in, y_in, z_in);
                    }
                }
                return 0;
            }
        }
    }
    break;
    case TYPE_PRISM:
    {
        // direct decomposition
        grid_methods::prism2tet(1, elem, conn, cell, tmp_el, tmp_cl);
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

            if (grid_methods::isin_tetra(point, p0, p1, p2, p3, tolerance) == 1)
            {
                // create a degenerate hexa
                int hexa_conn[8];
                hexa_conn[0] = conn[elem[cell]];
                hexa_conn[1] = conn[elem[cell] + 1];
                hexa_conn[2] = conn[elem[cell] + 2];
                hexa_conn[3] = conn[elem[cell] + 2];
                hexa_conn[4] = conn[elem[cell] + 3];
                hexa_conn[5] = conn[elem[cell] + 4];
                hexa_conn[6] = conn[elem[cell] + 5];
                hexa_conn[7] = conn[elem[cell] + 5];
                if (velo)
                {
                    if (no_arrays == 3
                        && array_dim == 1)
                    {
                        grid_methods::interpolateVInHexa(v_interp, point,
                                                         velo, hexa_conn, x_in, y_in, z_in);
                    }
                    else
                    {
                        grid_methods::interpolateInHexa(v_interp, point,
                                                        no_arrays, array_dim, velo,
                                                        hexa_conn, x_in, y_in, z_in);
                    }
                }
                return 0;
            }
        }
    }
    break;
    case TYPE_PYRAMID:
    {
        // direct decomposition
        grid_methods::pyra2tet(1, elem, conn, cell, tmp_el, tmp_cl);
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

            if (grid_methods::isin_tetra(point, p0, p1, p2, p3, tolerance) == 1)
            {
                if (velo)
                {
                    // create a degenerate hexa
                    int hexa_conn[8];
                    hexa_conn[0] = conn[elem[cell]];
                    hexa_conn[1] = conn[elem[cell] + 1];
                    hexa_conn[2] = conn[elem[cell] + 2];
                    hexa_conn[3] = conn[elem[cell] + 3];
                    hexa_conn[4] = conn[elem[cell] + 4];
                    hexa_conn[5] = conn[elem[cell] + 4];
                    hexa_conn[6] = conn[elem[cell] + 4];
                    hexa_conn[7] = conn[elem[cell] + 4];
                    if (no_arrays == 3
                        && array_dim == 1)
                    {
                        grid_methods::interpolateVInHexa(v_interp, point,
                                                         velo, hexa_conn, x_in, y_in, z_in);
                    }
                    else
                    {
                        grid_methods::interpolateInHexa(v_interp, point,
                                                        no_arrays, array_dim, velo,
                                                        hexa_conn, x_in, y_in, z_in);
                    }
                }
                return 0;
            }
        }
    }
    break;
    case TYPE_TETRAHEDER:
    {
        p0[0] = x_in[conn[elem[cell]]];
        p0[1] = y_in[conn[elem[cell]]];
        p0[2] = z_in[conn[elem[cell]]];

        p1[0] = x_in[conn[elem[cell] + 1]];
        p1[1] = y_in[conn[elem[cell] + 1]];
        p1[2] = z_in[conn[elem[cell] + 1]];

        p2[0] = x_in[conn[elem[cell] + 2]];
        p2[1] = y_in[conn[elem[cell] + 2]];
        p2[2] = z_in[conn[elem[cell] + 2]];

        p3[0] = x_in[conn[elem[cell] + 3]];
        p3[1] = y_in[conn[elem[cell] + 3]];
        p3[2] = z_in[conn[elem[cell] + 3]];

        if (grid_methods::isin_tetra(point, p0, p1, p2, p3, tolerance) == 1)
        {
            if (velo)
            {
                grid_methods::interpolateInTetra(v_interp, point,
                                                 no_arrays, array_dim, velo,
                                                 conn[elem[cell]], conn[elem[cell] + 1],
                                                 conn[elem[cell] + 2], conn[elem[cell] + 3],
                                                 p0, p1, p2, p3);
            }
            return 0;
        }
    }
    break;
    case TYPE_POLYHEDRON:
    {
        bool start_vertex_set;

        char inclusion_test;

        int next_elem_index;
        int start_vertex;
        int cell_radius;

        int *temp_elem_list;
        int *temp_conn_list;

        float *new_x_coord_in;
        float *new_y_coord_in;
        float *new_z_coord_in;
        float *temp_vel;

        double interpolated_data;

        vector<int> temp_elem_in;
        vector<int> temp_conn_in;
        vector<int> new_temp_elem_in;
        vector<int> new_temp_conn_in;
        vector<int> temp_vertex_list;

        grid_methods::POINT3D cell_box_min;
        grid_methods::POINT3D cell_box_max;
        grid_methods::POINT3D end_point;
        grid_methods::POINT3D particle_location;

        grid_methods::TESSELATION triangulated_cell;

        start_vertex_set = false;

        particle_location.x = (double)point[0];
        particle_location.y = (double)point[1];
        particle_location.z = (double)point[2];

        next_elem_index = (cell < numelem) ? elem[cell + 1] : numconn;

        /* Construct DO_Polygons Element and Connectivity Lists */
        for (j = elem[cell]; j < next_elem_index; j++)
        {
            if (j == elem[cell] && start_vertex_set == false)
            {
                start_vertex = conn[elem[cell]];
                temp_elem_in.push_back((int)temp_conn_in.size());
                temp_conn_in.push_back(start_vertex);
                start_vertex_set = true;
            }

            if (j > elem[cell] && start_vertex_set == true)
            {
                if (conn[j] != start_vertex)
                {
                    temp_conn_in.push_back(conn[j]);
                }
                else
                {
                    start_vertex_set = false;
                    continue;
                }
            }

            if (j > elem[cell] && start_vertex_set == false)
            {
                start_vertex = conn[j];
                temp_elem_in.push_back((int)temp_conn_in.size());
                temp_conn_in.push_back(start_vertex);
                start_vertex_set = true;
            }
        }

        /* Construct Vertex List */
        for (i = 0; i < temp_conn_in.size(); i++)
        {
            if (temp_vertex_list.size() == 0)
            {
                temp_vertex_list.push_back(temp_conn_in[i]);
            }
            else
            {
                if (find(temp_vertex_list.begin(), temp_vertex_list.end(), temp_conn_in[i]) == temp_vertex_list.end())
                {
                    temp_vertex_list.push_back(temp_conn_in[i]);
                }
            }
        }

        sort(temp_vertex_list.begin(), temp_vertex_list.end());

        /* Construct New Connectivity List */
        for (i = 0; i < temp_conn_in.size(); i++)
        {
            for (j = 0; j < temp_vertex_list.size(); j++)
            {
                if (temp_conn_in[i] == temp_vertex_list[j])
                {
                    new_temp_conn_in.push_back(j);
                    break;
                }
            }
        }

        temp_elem_list = new int[temp_elem_in.size()];
        temp_conn_list = new int[temp_conn_in.size()];
        new_x_coord_in = new float[temp_vertex_list.size()];
        new_y_coord_in = new float[temp_vertex_list.size()];
        new_z_coord_in = new float[temp_vertex_list.size()];

        for (i = 0; i < temp_elem_in.size(); i++)
        {
            temp_elem_list[i] = temp_elem_in[i];
        }

        for (i = 0; i < new_temp_conn_in.size(); i++)
        {
            temp_conn_list[i] = new_temp_conn_in[i];
        }

        /* Construct New Set of Coordinates */
        for (i = 0; i < temp_vertex_list.size(); i++)
        {
            new_x_coord_in[i] = x_in[temp_vertex_list[i]];
            new_y_coord_in[i] = y_in[temp_vertex_list[i]];
            new_z_coord_in[i] = z_in[temp_vertex_list[i]];
        }

        grid_methods::TesselatePolyhedron(triangulated_cell, (int)temp_elem_in.size(), temp_elem_list, (int)new_temp_conn_in.size(), temp_conn_list, new_x_coord_in, new_y_coord_in, new_z_coord_in);

        grid_methods::ComputeBoundingBox((int)temp_vertex_list.size(), new_x_coord_in, new_y_coord_in, new_z_coord_in, cell_box_min, cell_box_max, cell_radius /*, cell_box_vertices*/);

        inclusion_test = grid_methods::InPolyhedron(new_x_coord_in, new_y_coord_in, new_z_coord_in, cell_box_min, cell_box_max, particle_location, end_point, cell_radius, triangulated_cell);

        if (inclusion_test == 'i' || inclusion_test == 'V' || inclusion_test == 'E' || inclusion_test == 'F')
        {
            temp_vel = new float[temp_vertex_list.size()];

            if (array_dim == 1) /* Scalar or vector */
            {
                for (i = 0; i < no_arrays; i++)
                {
                    for (j = 0; j < temp_vertex_list.size(); j++)
                    {
                        temp_vel[j] = velo[i][temp_vertex_list[j]];
                    }

                    interpolated_data = grid_methods::InterpolateCellData((int)temp_vertex_list.size(), new_x_coord_in, new_y_coord_in, new_z_coord_in, temp_vel, particle_location);

                    v_interp[i] = (float)interpolated_data;
                }

                delete[] temp_vel;
            }

            else /* General case -- tensors? -- currently not implemented */
            {
            }

            delete[] temp_elem_list;
            delete[] temp_conn_list;
            delete[] new_x_coord_in;
            delete[] new_y_coord_in;
            delete[] new_z_coord_in;
            return 0;
        }
    }
    break;
    default:
        return -1;
    }
    return -1;
}

void coDoUnstructuredGrid::mapInCell(float *v_interp, const float *point,
                                     int cell, int no_arrays, int array_dim,
                                     const float *const *velo)
{
    // tetrahedronise
    //    int tmp_el[5], tmp_cl[20];
    //    float p0[3], p1[3], p2[3], p3[3];

    int i;
    int j;
    // determine type of cell
    int cell_type;
    int *elem, *conn;
    float *x_in, *y_in, *z_in;
    getAddresses(&elem, &conn, &x_in, &y_in, &z_in);
    if (hasTypeList())
    {
        cell_type = elementtypes[cell];
    }
    else
    {
        int num_of_vert;
        if (cell < numelem - 1)
        {
            num_of_vert = elem[cell + 1] - elem[cell];
        }
        else
        {
            num_of_vert = numconn - elem[cell];
        }
        switch (num_of_vert)
        {
        // In the case of polyhedral cells, cell type cannot be determined only by the number of vertices.
        // Ambiguous cases may appear, therefore type list should be always given.
        case 8:
            cell_type = TYPE_HEXAEDER;
            break;
        case 6:
            cell_type = TYPE_PRISM;
            break;
        case 5:
            cell_type = TYPE_PYRAMID;
            break;
        case 4: // assume tetrahedra
            cell_type = TYPE_TETRAHEDER;
            break;
        default: // do not consider 2D, 1D or 0D elements
            cell_type = TYPE_NONE;
            break;
        }
    }

    bool start_vertex_set;

    int next_elem_index;
    int start_vertex;

    double interpolated_data;

    int *temp_elem_list;
    int *temp_conn_list;

    float *new_x_coord_in;
    float *new_y_coord_in;
    float *new_z_coord_in;
    float *temp_vel;

    vector<int> temp_elem_in;
    vector<int> temp_conn_in;
    vector<int> new_temp_conn_in;
    vector<int> temp_vertex_list;

    grid_methods::POINT3D particle_location;

    start_vertex_set = false;

    particle_location.x = (double)point[0];
    particle_location.y = (double)point[1];
    particle_location.z = (double)point[2];

    next_elem_index = (cell < numelem) ? elem[cell + 1] : numconn;

    switch (cell_type)
    {
    case TYPE_NONE:
        // do nothing for unhandled types
        break;

    case TYPE_HEXAEDER:
    {
        /* Construct DO_Polygons Element and Connectivity Lists */
        for (j = elem[cell]; j < next_elem_index; j++)
        {
            if (j == elem[cell])
            {
                temp_elem_in.push_back((int)temp_conn_in.size());
            }

            if ((j - elem[cell]) == 4)
            {
                temp_elem_in.push_back(4);
            }

            temp_conn_in.push_back(conn[j]);
        }

        /* Construct Vertex List */
        for (i = 0; i < temp_conn_in.size(); i++)
        {
            temp_vertex_list.push_back(temp_conn_in[i]);
        }

        sort(temp_vertex_list.begin(), temp_vertex_list.end());

        /* Complete DO_Polygons Element and Connectivity Lists */
        temp_elem_in.push_back(8);

        temp_conn_in.push_back(temp_conn_in[1]);
        temp_conn_in.push_back(temp_conn_in[2]);
        temp_conn_in.push_back(temp_conn_in[6]);
        temp_conn_in.push_back(temp_conn_in[5]);

        temp_elem_in.push_back(12);

        temp_conn_in.push_back(temp_conn_in[0]);
        temp_conn_in.push_back(temp_conn_in[3]);
        temp_conn_in.push_back(temp_conn_in[7]);
        temp_conn_in.push_back(temp_conn_in[4]);

        temp_elem_in.push_back(16);

        temp_conn_in.push_back(temp_conn_in[2]);
        temp_conn_in.push_back(temp_conn_in[3]);
        temp_conn_in.push_back(temp_conn_in[7]);
        temp_conn_in.push_back(temp_conn_in[6]);

        temp_elem_in.push_back(20);

        temp_conn_in.push_back(temp_conn_in[1]);
        temp_conn_in.push_back(temp_conn_in[0]);
        temp_conn_in.push_back(temp_conn_in[4]);
        temp_conn_in.push_back(temp_conn_in[5]);
    }
    break;

    case TYPE_PRISM:
    {
        /* Construct DO_Polygons Element and Connectivity Lists */
        for (j = elem[cell]; j < next_elem_index; j++)
        {
            if (j == elem[cell])
            {
                temp_elem_in.push_back((int)temp_conn_in.size());
            }

            if ((j - elem[cell]) == 3)
            {
                temp_elem_in.push_back(3);
            }

            temp_conn_in.push_back(conn[j]);
        }

        /* Construct Vertex List */
        for (i = 0; i < temp_conn_in.size(); i++)
        {
            temp_vertex_list.push_back(temp_conn_in[i]);
        }

        /* Complete DO_Polygons Element and Connectivity Lists */
        temp_elem_in.push_back(6);

        temp_conn_in.push_back(temp_conn_in[0]);
        temp_conn_in.push_back(temp_conn_in[1]);
        temp_conn_in.push_back(temp_conn_in[4]);
        temp_conn_in.push_back(temp_conn_in[3]);

        temp_elem_in.push_back(10);

        temp_conn_in.push_back(temp_conn_in[1]);
        temp_conn_in.push_back(temp_conn_in[2]);
        temp_conn_in.push_back(temp_conn_in[5]);
        temp_conn_in.push_back(temp_conn_in[4]);

        temp_elem_in.push_back(14);

        temp_conn_in.push_back(temp_conn_in[0]);
        temp_conn_in.push_back(temp_conn_in[2]);
        temp_conn_in.push_back(temp_conn_in[5]);
        temp_conn_in.push_back(temp_conn_in[3]);
    }
    break;

    case TYPE_PYRAMID:
    {
        /* Construct DO_Polygons Element and Connectivity Lists */
        for (j = elem[cell]; j < next_elem_index; j++)
        {
            if (j == elem[cell])
            {
                temp_elem_in.push_back((int)temp_conn_in.size());
            }

            if ((j - elem[cell]) == 4)
            {
                temp_elem_in.push_back(4);
            }

            temp_conn_in.push_back(conn[j]);
        }

        /* Construct Vertex List */
        for (i = 0; i < temp_conn_in.size(); i++)
        {
            temp_vertex_list.push_back(temp_conn_in[i]);
        }

        /* Complete DO_Polygons Element and Connectivity Lists */
        temp_conn_in.push_back(temp_conn_in[1]);
        temp_conn_in.push_back(temp_conn_in[2]);

        temp_elem_in.push_back(7);

        temp_conn_in.push_back(temp_conn_in[4]);
        temp_conn_in.push_back(temp_conn_in[0]);
        temp_conn_in.push_back(temp_conn_in[1]);

        temp_elem_in.push_back(10);

        temp_conn_in.push_back(temp_conn_in[4]);
        temp_conn_in.push_back(temp_conn_in[3]);
        temp_conn_in.push_back(temp_conn_in[0]);

        temp_elem_in.push_back(13);

        temp_conn_in.push_back(temp_conn_in[4]);
        temp_conn_in.push_back(temp_conn_in[2]);
        temp_conn_in.push_back(temp_conn_in[3]);
    }
    break;

    case TYPE_TETRAHEDER:
    {
        /* Construct DO_Polygons Element and Connectivity Lists */
        for (j = elem[cell]; j < next_elem_index; j++)
        {
            if (j == elem[cell])
            {
                temp_elem_in.push_back((int)temp_conn_in.size());
            }

            if ((j - elem[cell]) == 3)
            {
                temp_elem_in.push_back(3);
            }

            temp_conn_in.push_back(conn[j]);
        }

        /* Construct Vertex List */
        for (i = 0; i < temp_conn_in.size(); i++)
        {
            temp_vertex_list.push_back(temp_conn_in[i]);
        }

        /* Complete DO_Polygons Element and Connectivity Lists */
        temp_conn_in.push_back(temp_conn_in[1]);
        temp_conn_in.push_back(temp_conn_in[2]);

        temp_elem_in.push_back(6);

        temp_conn_in.push_back(temp_conn_in[3]);
        temp_conn_in.push_back(temp_conn_in[0]);
        temp_conn_in.push_back(temp_conn_in[1]);

        temp_elem_in.push_back(9);

        temp_conn_in.push_back(temp_conn_in[3]);
        temp_conn_in.push_back(temp_conn_in[2]);
        temp_conn_in.push_back(temp_conn_in[0]);
    }
    break;

    case TYPE_POLYHEDRON:
    {
        /* Construct DO_Polygons Element and Connectivity Lists */
        for (j = elem[cell]; j < next_elem_index; j++)
        {
            if (j == elem[cell] && start_vertex_set == false)
            {
                start_vertex = conn[elem[cell]];
                temp_elem_in.push_back((int)temp_conn_in.size());
                temp_conn_in.push_back(start_vertex);
                start_vertex_set = true;
            }

            if (j > elem[cell] && start_vertex_set == true)
            {
                if (conn[j] != start_vertex)
                {
                    temp_conn_in.push_back(conn[j]);
                }

                else
                {
                    start_vertex_set = false;
                    continue;
                }
            }

            if (j > elem[cell] && start_vertex_set == false)
            {
                start_vertex = conn[j];
                temp_elem_in.push_back((int)temp_conn_in.size());
                temp_conn_in.push_back(start_vertex);
                start_vertex_set = true;
            }
        }

        /* Construct Vertex List */
        for (i = 0; i < temp_conn_in.size(); i++)
        {
            if (temp_vertex_list.size() == 0)
            {
                temp_vertex_list.push_back(temp_conn_in[i]);
            }

            else
            {
                if (find(temp_vertex_list.begin(), temp_vertex_list.end(), temp_conn_in[i]) == temp_vertex_list.end())
                {
                    temp_vertex_list.push_back(temp_conn_in[i]);
                }
            }
        }

        sort(temp_vertex_list.begin(), temp_vertex_list.end());
    }
    break;
    }

    /* Construct New Connectivity List */
    for (i = 0; i < temp_conn_in.size(); i++)
    {
        for (j = 0; j < temp_vertex_list.size(); j++)
        {
            if (temp_conn_in[i] == temp_vertex_list[j])
            {
                new_temp_conn_in.push_back(j);
                break;
            }
        }
    }

    temp_elem_list = new int[temp_elem_in.size()];
    temp_conn_list = new int[temp_conn_in.size()];
    new_x_coord_in = new float[temp_vertex_list.size()];
    new_y_coord_in = new float[temp_vertex_list.size()];
    new_z_coord_in = new float[temp_vertex_list.size()];

    for (i = 0; i < temp_elem_in.size(); i++)
    {
        temp_elem_list[i] = temp_elem_in[i];
    }

    for (i = 0; i < new_temp_conn_in.size(); i++)
    {
        temp_conn_list[i] = new_temp_conn_in[i];
    }

    /* Construct New Set of Coordinates */
    for (i = 0; i < temp_vertex_list.size(); i++)
    {
        new_x_coord_in[i] = x_in[temp_vertex_list[i]];
        new_y_coord_in[i] = y_in[temp_vertex_list[i]];
        new_z_coord_in[i] = z_in[temp_vertex_list[i]];
    }

    temp_vel = new float[temp_vertex_list.size()];

    if (array_dim == 1) /* Scalar or vector */
    {
        for (i = 0; i < no_arrays; i++)
        {
            for (j = 0; j < temp_vertex_list.size(); j++)
            {
                temp_vel[j] = velo[i][temp_vertex_list[j]];
            }

            interpolated_data = grid_methods::InterpolateCellData((int)temp_vertex_list.size(), new_x_coord_in, new_y_coord_in, new_z_coord_in, temp_vel, particle_location);

            v_interp[i] = (float)interpolated_data;
        }

        delete[] temp_vel;
    }

    else /* eam:  General case -- tensors? -- currently not implemented */
    {
    }

    delete[] temp_elem_list;
    delete[] temp_conn_list;
    delete[] new_x_coord_in;
    delete[] new_y_coord_in;
    delete[] new_z_coord_in;
}

void coDoUnstructuredGrid::MakeOctTree(const char *octtreeSurname) const
{
    if (!oct_tree)
    {
        // make oct-tree and use update_shared_dl
        //      covise_data_list dl[SHM_OBJ];
        int *e_l, *c_l;
        float *x_l, *y_l, *z_l;
        getAddresses(&e_l, &c_l, &x_l, &y_l, &z_l);
        char octname[256];
        sprintf(octname, "%s_OctTree_%s", name, octtreeSurname);
        oct_tree = new coDoOctTree(coObjInfo(octname), numelem, numconn, numcoord,
                                   e_l, c_l, x_l, y_l, z_l);
    }
}

const coDoOctTree *coDoUnstructuredGrid::GetOctTree(const coDistributedObject *reuseOct,
                                                    const char *OctTreeSurname) const
{
    if (reuseOct)
    {
        oct_tree = reuseOct;
    }
    else if (OctTreeSurname)
    {
        MakeOctTree(OctTreeSurname);
    }
    return (coDoOctTree *)(oct_tree);
}

void
coDoUnstructuredGrid::compressConnectivity()
{
    int *el, *cl, *tl;
    float *xc, *yc, *zc;

    this->getAddresses(&el, &cl, &xc, &yc, &zc);
    int nele, nconn, ncoord;
    this->getGridSize(&nele, &nconn, &ncoord);
    this->getTypeList(&tl);
    int i, next_cl, elem;
    enum ELEM_TYPE quad_type;

    std::vector<int> elist(nele);
    std::vector<enum ELEM_TYPE> tlist(nele);
    std::vector<int> clist(nconn);

    int pyrToTet = 0, pyrToTri = 0;

    for (elem = 0; elem < nele; ++elem)
    {
        elist.push_back((int)clist.size());
        std::vector<int> vertices;
        int vertex;
        std::vector<int> pattern;

        switch (tl[elem])
        {
        case TYPE_HEXAEDER:
            for (vertex = 0; vertex < 8; ++vertex)
            {
                int corner = cl[el[elem] + vertex];
                // check if it is already repeated
                int count_old_corner;
                for (count_old_corner = 0; count_old_corner < vertices.size();
                     ++count_old_corner)
                {
                    if (corner == vertices[count_old_corner])
                    {
                        break;
                    }
                }
                if (count_old_corner == vertices.size())
                {
                    vertices.push_back(corner);
                    pattern.push_back(vertex);
                }
            }
            // add vertices array to clist
            if (vertices.size() == 4)
            {
                quad_type = TYPE_QUAD;
                // for tetras the connectivity is trivial
                for (vertex = 0; vertex < 4; ++vertex)
                {
                    clist.push_back(vertices[vertex]);
                    if (cl[el[elem] + vertex] != cl[el[elem] + vertex + 4])
                    {
                        quad_type = TYPE_TETRAHEDER;
                    }
                }
                tlist.push_back(quad_type);
            }
            else if (vertices.size() == 5)
            {
                tlist.push_back(TYPE_PYRAMID);
                for (vertex = 0; vertex < 5; ++vertex)
                {
                    clist.push_back(vertices[vertex]);
                }
            }
            else if (vertices.size() == 6)
            {
                tlist.push_back(TYPE_PRISM);
                // for prisms the connectivity is NOT trivial
                // we have to use the pattern

                clist.push_back(vertices[0]);
                // the bottom face may be a line
                if (pattern[2] == 4)
                {
                    if (pattern[1] == 1)
                    {
                        clist.push_back(vertices[5]);
                        clist.push_back(vertices[2]);
                        clist.push_back(vertices[1]);
                        clist.push_back(vertices[4]);
                        clist.push_back(vertices[3]);
                    }
                    else
                    {
                        clist.push_back(vertices[2]);
                        clist.push_back(vertices[3]);
                        clist.push_back(vertices[1]);
                        clist.push_back(vertices[5]);
                        clist.push_back(vertices[4]);
                    }
                }
                // or the top face
                else if (pattern[3] == 3)
                {
                    if (pattern[4] == 4 && pattern[5] == 5)
                    {
                        clist.push_back(vertices[3]);
                        clist.push_back(vertices[4]);
                        clist.push_back(vertices[1]);
                        clist.push_back(vertices[2]);
                        clist.push_back(vertices[5]);
                    }
                    else if (pattern[4] == 6)
                    {
                        clist.push_back(vertices[3]);
                        clist.push_back(vertices[5]);
                        clist.push_back(vertices[1]);
                        clist.push_back(vertices[2]);
                        clist.push_back(vertices[4]);
                    }
                    else
                    {
                        clist.push_back(vertices[4]);
                        clist.push_back(vertices[1]);
                        clist.push_back(vertices[3]);
                        clist.push_back(vertices[5]);
                        clist.push_back(vertices[2]);
                    }
                }
                // or one side face...
                else if (pattern[3] == 4) // side joining
                {
                    clist.push_back(vertices[1]);
                    clist.push_back(vertices[2]);
                    clist.push_back(vertices[3]);
                    clist.push_back(vertices[4]);
                    clist.push_back(vertices[5]);
                }
            }
            else if (vertices.size() == 8)
            {
                tlist.push_back(TYPE_HEXAEDER);
                for (vertex = 0; vertex < 8; ++vertex)
                {
                    clist.push_back(vertices[vertex]);
                }
            }
            break;

        case TYPE_TETRAHEDER:
            for (vertex = 0; vertex < 4; ++vertex)
            {
                int corner = cl[el[elem] + vertex];
                // check if it is already repeated
                int count_old_corner;
                for (count_old_corner = 0; count_old_corner < vertices.size();
                     ++count_old_corner)
                {
                    if (corner == vertices[count_old_corner])
                    {
                        break;
                    }
                }
                if (count_old_corner == vertices.size())
                {
                    vertices.push_back(corner);
                    pattern.push_back(vertex);
                }
            }
            // add vertices array to clist
            if (vertices.size() == 3)
            {
                // TETRA -> TRIA
                for (vertex = 0; vertex < 3; ++vertex)
                {
                    clist.push_back(vertices[vertex]);
                }
                tlist.push_back(TYPE_TRIANGLE);
            }
            else if (vertices.size() == 2)
            {
                // TETRA -> BAR
                for (vertex = 0; vertex < 2; ++vertex)
                {
                    clist.push_back(vertices[vertex]);
                }
                tlist.push_back(TYPE_BAR);
            }
            else if (vertices.size() == 4)
            {
                for (vertex = 0; vertex < 4; ++vertex)
                {
                    clist.push_back(vertices[vertex]);
                }
                tlist.push_back(TYPE_TETRAHEDER);
            }
            break;

        case TYPE_PYRAMID:
            for (vertex = 0; vertex < 5; ++vertex)
            {
                int corner = cl[el[elem] + vertex];
                // check if it is already repeated
                int count_old_corner;
                for (count_old_corner = 0; count_old_corner < vertices.size();
                     ++count_old_corner)
                {
                    if (corner == vertices[count_old_corner])
                    {
                        break;
                    }
                }
                if (count_old_corner == vertices.size())
                {
                    vertices.push_back(corner);
                    pattern.push_back(vertex);
                }
            }
            // add vertices array to clist
            if (vertices.size() == 3)
            {
                // PYRA -> TRIA
                for (vertex = 0; vertex < 3; ++vertex)
                {
                    clist.push_back(vertices[vertex]);
                }
                tlist.push_back(TYPE_TRIANGLE);
                pyrToTri++;
            }
            else if (vertices.size() == 4)
            {
                // for tetras the connectivity is trivial
                for (vertex = 0; vertex < 4; ++vertex)
                {
                    clist.push_back(vertices[vertex]);
                }
                tlist.push_back(TYPE_TETRAHEDER);
                pyrToTet++;
            }
            else if (vertices.size() == 5)
            {
                for (vertex = 0; vertex < 5; ++vertex)
                {
                    clist.push_back(vertices[vertex]);
                }
                tlist.push_back(TYPE_PYRAMID);
            }
            break;

        case TYPE_QUAD:
            for (vertex = 0; vertex < 4; ++vertex)
            {
                int corner = cl[el[elem] + vertex];
                // check if it is already repeated
                int count_old_corner;
                for (count_old_corner = 0; count_old_corner < vertices.size();
                     ++count_old_corner)
                {
                    if (corner == vertices[count_old_corner])
                    {
                        break;
                    }
                }
                if (count_old_corner == vertices.size())
                {
                    vertices.push_back(corner);
                    pattern.push_back(vertex);
                }
            }
            // add vertices array to clist
            if (vertices.size() == 3)
            {
                // for tetras the connectivity is trivial
                for (vertex = 0; vertex < 3; ++vertex)
                {
                    clist.push_back(vertices[vertex]);
                }
                tlist.push_back(TYPE_TRIANGLE);
            }
            else if (vertices.size() == 4)
            {
                for (vertex = 0; vertex < 4; ++vertex)
                {
                    clist.push_back(vertices[vertex]);
                }
                tlist.push_back(TYPE_QUAD);
            }
            break;
        default: // just copy content
            tlist.push_back((enum ELEM_TYPE)tl[elem]);
            next_cl = (elem != (nele - 1)) ? el[elem + 1] : nconn;

            for (i = el[elem]; i < next_cl; i++)
            {
                clist.push_back(cl[i]);
            }
        }
    }

    if (clist.size() < nconn) // we really compressed something
    {
        for (i = 0; i < elist.size(); i++)
        {
            el[i] = elist[i];
            tl[i] = tlist[i];
        }
        for (i = 0; i < clist.size(); i++)
        {
            cl[i] = clist[i];
        }
        this->setSizes((int)elist.size(), (int)clist.size(), ncoord);

        //cerr << "coDoUnstructuredGrid::compressConnectivity() converted " << pyrToTet << " pyra to tet" << endl;

        //cerr << "coDoUnstructuredGrid::compressConnectivity() converted " << pyrToTri << " pyra to tria" << endl;
    }
}

int coDoUnstructuredGrid::getNumConnOfElement(int elem) const
{
    if (elem == numelem - 1)
    {
        return numconn - elements[elem];
    }
    else
    {
        return elements[elem + 1] - elements[elem];
    }
}
