/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// helpers for AVS
// filip sadlo
// cgl eth 2006
// $Id: avs_ext.cpp,v 1.2 2006/07/05 17:36:52 sadlof Exp $
//
// $Log: avs_ext.cpp,v $
// Revision 1.2  2006/07/05 17:36:52  sadlof
// Minor change.
//

#include <vector>
#include "avs_ext.h"

// general -----

int findInIntArray(int key, int *arr, int arr_len)
{
    for (int i = 0; i < arr_len; i++)
    {
        if (arr[i] == key)
        {
            return i;
        }
    }
    return -1;
}

#if 0 // original is in cfx_export, get it here when finished
int findInSortedIntArray(int key, int *arr, int arr_len)
{
  bool ascending = (arr[0] <= arr[arr_len-1]);

  int left = 0;
  int right = arr_len - 1;
  int mid = (left + right) / 2;

  while (1) {
    if (key == arr[mid]) {
      // found
      return mid;
    }
    else if (left >= right) {
      return -1; // not found
    }
    else if ((ascending ? key < arr[mid] : key > arr[mid])) {
      // in 'left' half
      right = mid - 1;
    }
    else {
      // in 'right' half
      left = mid + 1;
    }
    mid = (left + right) / 2;
  }
}

int insertInSortedIntArray(int key, int ascending, int *arr, int arr_len)
{ // arr must be large enough
  // changes array (indices may invalidate)
  int i;
  
  // ### linear search
  for (i=0; i<arr_len; i++) {
    if (arr[i] == key) {
      // found -> ### no multiple
      return i;
    }
    else if ((ascending ? key < arr[i]) : key > arr[i]) {
      // insert before current
      memmove(&arr[i+1], &arr[i], arr_len - i;);
      arr[i] = key;
      return i;
    }
  }
  arr[arr_len] = key;
  return arr_len;
}
#endif

bool findInString(char *key, char sep, char *string)
{
    char *lastP = string;
    char *wp = string;
    while ((wp = strchr(wp, sep)))
    {
        if ((wp > lastP) && (strncmp(lastP, key, wp - lastP) == 0))
        {
            return true;
        }
        wp++;
        lastP = wp;
    }
    if (strncmp(lastP, key, strlen(key)) == 0)
    {
        return true;
    }
    return false;
}

bool findInString(int key, char sep, char *string)
{
    char keyS[1024];
    sprintf(keyS, "%d", key);

    return findInString(keyS, sep, string);
}

// UCD -----

int ucd_nodeCompNb(UCD_structure *ucd, int veclen)
{ // returns number of AVS - UCD node components
    // 15 jul 2001
    // returns -1 if error

    int nodeCompNb = -1;
    int sum, idx;

    if (!(ucd->node_components))
    {
        printf("ucd_nodeCompNb: error: node_components = NULL\n");
        return -1;
    }

    /* === get number of AVS - node components === */
    sum = 0;
    nodeCompNb = 0;
    for (idx = 0; (idx < ucd->node_veclen) && (sum + ucd->node_components[idx] < ucd->node_veclen + 1); idx++)
    {
        sum += ucd->node_components[idx];

        if ((veclen < 0) || (veclen == ucd->node_components[idx]))
        {
            nodeCompNb++;
        }

        if (sum == ucd->node_veclen)
            break;
    }

    if (sum < ucd->node_veclen)
    {
        printf("getUCDCompNb: error: could not get number of node components\n"
               "sum=%d node_veclen=%d node_components[0]=%d\n",
               sum, ucd->node_veclen, ucd->node_components[0]);
        return -1;
    }

    return nodeCompNb;
}

int ucd_findNodeCompByVeclen(UCD_structure *ucd, int dest_veclen, int dest_idx,
                             int compare)
{ // returns index of dest_idx-th component < dest_veclen (if compare < 0)
    //                                        = dest_veclen (if compare = 0)
    //                                        > dest_veclen (if compare > 0)
    // otherwise -1

    int cNb = ucd_nodeCompNb(ucd);
    int idx = 0;
    for (int c = 0; c < cNb; c++)
    {
        if ((compare < 0) && (ucd->node_components[c] < dest_veclen) || (compare == 0) && (ucd->node_components[c] == dest_veclen) || (compare > 0) && (ucd->node_components[c] > dest_veclen))
        {
            if (idx == dest_idx)
            {
                return c;
            }
            idx++;
        }
    }
    return -1;
}

char ucd_nodeLabelDelim(UCD_structure *ucd)
{
    char labStr[10000], delimStr[512]; // TODO: ### HACK
    UCDstructure_get_node_labels(ucd, labStr, delimStr);
    return *delimStr; // TODO: multi-char delimiters possible?
}

int ucd_nodeCompLabel(UCD_structure *ucd, int comp_idx, char *dest)
{ // returns zero on success

    int compNb = ucd_nodeCompNb(ucd);
    char *ptr = ucd->node_labels;
    char delim = ucd_nodeLabelDelim(ucd);

    if (ptr && *ptr && ptr[0] == delim)
    {
        ptr++;
    }

    char *lastptr = ptr;

    for (int c = 0; c < compNb; c++)
    {
        ptr = strchr(ptr, delim);
        if (!ptr && c != (compNb - 1))
        {
            printf("ucd_nodeCompLabel: error: could not get label\n");
            return -1;
        }

        if (c == comp_idx)
        {
            int cpylen;
            if (c != compNb - 1)
                cpylen = ptr - lastptr;
            else
                cpylen = strlen(lastptr);
            strncpy(dest, lastptr, cpylen);
            dest[cpylen] = '\0';
            return 0;
        }

        ptr++; // skip '.'
        lastptr = ptr;
    }
    return -1;
}

int ucd_processCompChoice(UCD_structure *ucd, char *ucd_name,
                          char *choice, char *choice_name,
                          int dest_veclen,
                          int defaultComp)
{ // updates AVS choice and determines selected UCD component
    // ucd_name    : name of AVS UCD input parameter
    // choice      : AVS choice
    // choice_name : name of AVS choice parameter
    // dest_veclen : desired vector length of components, -1 for all components
    // defaultComp : default component (e.g. from Unstructured), if no selection
    // returns index of selected UCD component, otherwise -1

    int selected_component = -1;

    if (!(ucd->node_labels && strlen(ucd->node_labels) > 0))
        return -1;

    // ### check
    char *comp_labels = (char *)malloc(strlen(ucd->node_labels) + 1);
    comp_labels[0] = '\0';
    char selectedComponentName[256] = ""; // ###
    char defaultComponentName[256] = ""; // ###

    // get labels of components and get index of selected component
    for (int c = 0; c < ucd_nodeCompNb(ucd); c++)
    {
        if (dest_veclen < 0 || ucd->node_components[c] == dest_veclen)
        {
            char *lab = comp_labels + strlen(comp_labels);
            ucd_nodeCompLabel(ucd, c, lab);

            if (choice && strcmp(choice, lab) == 0)
            {
                selected_component = c;
                strcpy(selectedComponentName, lab);
            }

            if (c == defaultComp)
            {
                strcpy(defaultComponentName, lab);
            }

            strcat(comp_labels, ".");
        }
    }

    //printf("avs_ext: selected component = %d\n", selected_component);
    //printf("all labels: %s\nscalar labels: %s\n", ucd->node_labels, comp_labels);

    if (AVSinput_changed(ucd_name, 0))
    {
        // update
        AVSmodify_parameter(choice_name, AVS_MINVAL | AVS_MAXVAL, 0, comp_labels, ".");
    }

    // select
    if (strlen(selectedComponentName) > 0)
    {
        AVSmodify_parameter(choice_name, AVS_VALUE, selectedComponentName, "", "");
    }
    else if (strlen(defaultComponentName) > 0)
    {
        AVSmodify_parameter(choice_name, AVS_VALUE, defaultComponentName, "", "");
        selected_component = defaultComp;
    }

    free(comp_labels);

    return selected_component;
}

// --- stolen from Ronny -------------------------------------------------------

static int cellSize[] = { 1, 2, 3, 4, 4, 5, 6, 8 };
static char cellName[][10] = { "point", "line", "tri", "quad", "tet", "pyr", "prism", "hex" };

#if 0 // DELETE: replaced by below, 2006-10-02

UCD_structure *ucdClone(UCD_structure *ucd, int veclen, char *label)
{ // set veclen to zero for cloning full veclen with all components

  bool allComponents = false;
  if (veclen <= 0) {
    // all components
    allComponents = true;
    veclen = ucd->node_veclen;
  }

  UCD_structure* ucd1 = (UCD_structure *) UCDstructure_alloc( 
							     label,
							     0,							  /* structure veclen */
#if 1 // this was orig
							     70,							  /*UCD_INT | UCD_MATERIAL_IDS ???? */
#else
							     0,							  /*UCD_INT | UCD_MATERIAL_IDS ???? */
#endif
							     ucd->ncells,
							     ucd->node_conn_size,
							     0,							  /* cell veclen */
							     ucd->nnodes,
							     0,							  /* expected size of cell list */
							     veclen,						/* node veclen */
							     2);							  /* util_flag */
  
  int comp[1], *effComp, effCompCnt;
  if (!allComponents) {
    comp[0] = veclen;
    effComp = comp;
    effCompCnt = 1;
  }
  else {
    effComp = ucd->node_components;
    effCompCnt = ucd_nodeCompNb(ucd);
  }
  int active[] = {0};
  char units[] = "unit0.unit1.unit2.unit3.unit4.unit5.unit6.unit7.unit8.unit9.unit10.unit11.unit12.unit13.unit14.unit15.unit16.unit17.unit18.unit19";
  float min_node_data[1] = {-1e19};
  float max_node_data[1] = {1e19};
  
  UCDstructure_set_node_components(ucd1, effComp, effCompCnt);
  UCDstructure_set_node_active(ucd1, active);
  UCDstructure_set_node_labels(ucd1, label, ".");
  UCDstructure_set_node_units(ucd1, units, ".");
  UCDstructure_set_node_minmax (ucd1, min_node_data, max_node_data);
  
  float min[3], max[3];
  UCDstructure_get_extent(ucd, min, max);
  UCDstructure_set_extent(ucd1, min, max);
  
  memcpy(ucd1->x, ucd->x, ucd->nnodes * sizeof(float));
  memcpy(ucd1->y, ucd->y, ucd->nnodes * sizeof(float));
  memcpy(ucd1->z, ucd->z, ucd->nnodes * sizeof(float));
  
  int* dptr = ucd->node_list;
  for (int i = 0; i < ucd->ncells; i++) {
    int type = ucd->cell_type[i];
    
    UCDcell_set_information(
			    ucd1,
			    i,									/* cell */
			    (char *)(i + 1),					/* cell name */
			    cellName[type],
			    0,									  /* material */
			    type,
			    0,									  /* mid_edge_flags */
			    dptr
			    );
    dptr += cellSize[type];
  }
  
  return ucd1;
}

#else

UCD_structure *ucdClone(UCD_structure *ucd, int veclen, char *name)
{ // set veclen to zero for cloning full veclen with all components

    int components[1] = { veclen };
    return ucdClone(ucd, 1, components, name);
}

#endif

UCD_structure *ucdClone(UCD_structure *ucd, int componentCnt, int *components, char *name, char *labels, char *labelSep)
{ // set componentCnt to zero for cloning full veclen with all components
    // labels: may be NULL
    // labelSep: delimiter for labels

    int veclen = 0;
    bool allComponents = false;
    if (componentCnt <= 0)
    {
        // all components
        allComponents = true;
        veclen = ucd->node_veclen;
    }
    else
    {
        // get total veclen
        for (int c = 0; c < componentCnt; c++)
        {
            veclen += components[c];
        }
    }

    if (!labels)
    {
        labels = name;
        labelSep = ".";
    }

    UCD_structure *ucd1 = (UCD_structure *)UCDstructure_alloc(
        name,
        0, /* structure veclen */
#if 1 // this was orig
        70, /*UCD_INT | UCD_MATERIAL_IDS ???? */
#else
        0, /*UCD_INT | UCD_MATERIAL_IDS ???? */
#endif
        ucd->ncells,
        ucd->node_conn_size,
        0, /* cell veclen */
        ucd->nnodes,
        0, /* expected size of cell list */
        veclen, /* node veclen */
        2); /* util_flag */

    int *effComp, effCompCnt;
    if (!allComponents)
    {
        effComp = components;
        effCompCnt = componentCnt;
    }
    else
    {
        effComp = ucd->node_components;
        effCompCnt = ucd_nodeCompNb(ucd);
    }
    int active[] = { 0 };
    char units[] = "unit0.unit1.unit2.unit3.unit4.unit5.unit6.unit7.unit8.unit9.unit10.unit11.unit12.unit13.unit14.unit15.unit16.unit17.unit18.unit19";
    float min_node_data[1] = { -1e19 };
    float max_node_data[1] = { 1e19 };

    UCDstructure_set_node_components(ucd1, effComp, effCompCnt);
    UCDstructure_set_node_active(ucd1, active);
    UCDstructure_set_node_labels(ucd1, labels, labelSep);
    UCDstructure_set_node_units(ucd1, units, ".");
    UCDstructure_set_node_minmax(ucd1, min_node_data, max_node_data);

    float min[3], max[3];
    UCDstructure_get_extent(ucd, min, max);
    UCDstructure_set_extent(ucd1, min, max);

    memcpy(ucd1->x, ucd->x, ucd->nnodes * sizeof(float));
    memcpy(ucd1->y, ucd->y, ucd->nnodes * sizeof(float));
    memcpy(ucd1->z, ucd->z, ucd->nnodes * sizeof(float));

    int *dptr = ucd->node_list;
    for (int i = 0; i < ucd->ncells; i++)
    {
        int type = ucd->cell_type[i];

        UCDcell_set_information(
            ucd1,
            i, /* cell */
            (char *)(i + 1), /* cell name */
            cellName[type],
            0, /* material */
            type,
            0, /* mid_edge_flags */
            dptr);
        dptr += cellSize[type];
    }

    return ucd1;
}

UCD_structure *generateUniformUCD(char *name,
                                  float originX, float originY, float originZ,
                                  int cellsX, int cellsY, int cellsZ,
                                  float cellSize,
                                  int componentCnt, int *components,
                                  char *labels, char *labelSep)
{ // generates uniform hexahedral UCD
    int ncells = cellsX * cellsY * cellsZ;
    int nnodes = (cellsX + 1) * (cellsY + 1) * (cellsZ + 1);

    // get total veclen
    int veclen = 0;
    for (int c = 0; c < componentCnt; c++)
    {
        veclen += components[c];
    }

    // alloc UCD
    UCD_structure *ucd = (UCD_structure *)
        UCDstructure_alloc(name,
                           0, // data_veclen
                           0, // name flag
                           ncells, // ncells
                           ncells * 8, // cell tsize
                           0, // cell veclen
                           nnodes, // nnodes
                           0, // node_csize,
                           veclen, // ### NOTE: UCD1 defines veclen and components
                           0 // util flag
                           );

    // set coordinates and connectivity
    float *xp, *yp, *zp;
    xp = ucd->x;
    yp = ucd->y;
    zp = ucd->z;
    int n = 0;
    int c = 0;
    for (int z = 0; z < cellsZ + 1; z++)
    {
        for (int y = 0; y < cellsY + 1; y++)
        {
            for (int x = 0; x < cellsX + 1; x++)
            {
                *(xp++) = originX + x * cellSize;
                *(yp++) = originY + y * cellSize;
                *(zp++) = originZ + z * cellSize;

                if ((x < cellsX) && (y < cellsY) && (z < cellsZ))
                {

                    int nX = cellsX + 1;
                    int nXY = nX * (cellsY + 1);
                    int nXYPX = nXY + nX;
                    int list[8] = {
                        n, n + nX, n + nX + 1, n + 1,
                        n + nXY, n + nXYPX, n + nXYPX + 1, n + nXY + 1
                    };

                    UCDcell_set_information(ucd,
                                            c, // cell
                                            (char *)(c + 1), // cell name
                                            "hex",
                                            0, // material
                                            UCD_HEXAHEDRON,
                                            0, // mid_edge_flags
                                            list);

                    c++;
                }
                n++;
            }
        }
    }

    // set extent
    float min[3] = { originX, originY, originZ };
    float max[3] = { originX + cellsX * cellSize,
                     originY + cellsY * cellSize,
                     originZ + cellsZ * cellSize };
    UCDstructure_set_extent(ucd, min, max);

    // set components
    UCDstructure_set_node_components(ucd, components, componentCnt);
    UCDstructure_set_node_labels(ucd, labels, labelSep);

    return ucd;
}
