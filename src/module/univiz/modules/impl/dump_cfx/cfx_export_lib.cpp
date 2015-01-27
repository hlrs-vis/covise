/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// High level access to cfxExport.h
// Filip Sadlo
// CGL ETHZ 2006

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

extern "C" {
#include <cfxExport.h>
}
#include "cfx_export_lib.h"
#include "unstructured.h"

//#include "linalg.h"
#if 0
typedef double   vec3[3];
typedef  vec3    mat3[3];
double  vec3dot( vec3 a,  vec3 b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
void  vec3copy( vec3 a,  vec3 b) { memcpy(b, a, sizeof( vec3)); }
double  vec3sqr( vec3 a) { return  vec3dot(a, a); }
double  vec3mag( vec3 a) { return        sqrt( vec3sqr(a)); }
void  vec3cross( vec3 a,  vec3 b,  vec3 c)
{
    vec3 d;
    d[0] = a[1]*b[2] - a[2]*b[1];
    d[1] = a[2]*b[0] - a[0]*b[2];
    d[2] = a[0]*b[1] - a[1]*b[0];
    vec3copy(d, c);
}
void vec3nrm(vec3 a, vec3 b)
{
    double l = vec3mag(a);
    if (l == 0) l = 1;
    b[0] = a[0] / l;
    b[1] = a[1] / l;
    b[2] = a[2] / l;
}
inline void  vec3scal( vec3 a, double b,  vec3 c) { c[0] = a[0] * b; c[1] = a[1] * b; c[2] = a[2] * b; }
inline void  vec3zero( vec3 a) { a[0] = a[1] = a[2] = 0.0; }
inline void  vec3add( vec3 a,  vec3 b,  vec3 c)     { c[0] = a[0] + b[0]; c[1] = a[1] + b[1]; c[2] = a[2] + b[2]; }
inline void  mat3zero( mat3 a) {  vec3zero(a[0]);  vec3zero(a[1]);  vec3zero(a[2]); }
inline void  mat3add( mat3 a,  mat3 b,  mat3 c)
{
    vec3add(a[0], b[0], c[0]);
    vec3add(a[1], b[1], c[1]);
    vec3add(a[2], b[2], c[2]);
}
inline void mat3setrows(mat3 a, vec3 a0, vec3 a1, vec3 a2) 
{
    a[0][0] = a0[0]; a[0][1] = a0[1]; a[0][2] = a0[2]; 
    a[1][0] = a1[0]; a[1][1] = a1[1]; a[1][2] = a1[2];
    a[2][0] = a2[0]; a[2][1] = a2[1]; a[2][2] = a2[2];
}
inline void  mat3ident( mat3 a) {  mat3zero(a); a[0][0] = a[1][1] = a[2][2] = 1.0; }
inline void  mat3scal( mat3 a, double b,  mat3 c)
{
    vec3scal(a[0], b, c[0]);
    vec3scal(a[1], b, c[1]);
    vec3scal(a[2], b, c[2]);
}
inline void vec3outer(vec3 a, vec3 b, mat3 m)
{ // m = a * bT
  vec3 w1, w2, w3;
  vec3scal(b, a[0], w1);
  vec3scal(b, a[1], w2);
  vec3scal(b, a[2], w3);
  mat3setrows(m, w1, w2, w3);
}
inline void rotVectTomat3(vec3 v, mat3 m)
{
  // A = I * cos(theta) + (1 - cos(theta)) v * vT - eps * sin(theta)
  // eps is "kind of vorticity tensor" build from v

  double theta = vec3mag(v);
  vec3nrm(v, v);
  double sinTheta = sin(theta);
  double cosTheta = cos(theta);

  mat3 i;
  mat3ident(i);
  mat3scal(i, cosTheta, i);

  mat3 vv;
  vec3outer(v, v, vv);
  mat3scal(vv, 1 - cosTheta, vv);
  
  mat3 eps;
  mat3zero(eps);
  eps[1][0] =  v[2];
  eps[2][0] = -v[1];
  eps[2][1] =  v[0];
  eps[0][1] = -v[2];
  eps[0][2] =  v[1];
  eps[1][2] = -v[0];
  mat3scal(eps, - sinTheta, eps);

  mat3add(i, vv, m);
  mat3add(m, eps, m);
}
inline void  mat3vec( mat3 a,  vec3 b,  vec3 c)
{
    vec3 d;
    d[0] = a[0][0]*b[0] + a[0][1]*b[1] + a[0][2]*b[2];
    d[1] = a[1][0]*b[0] + a[1][1]*b[1] + a[1][2]*b[2];
    d[2] = a[2][0]*b[0] + a[2][1]*b[1] + a[2][2]*b[2];
    vec3copy(d, c);
}
inline void vec3dump(vec3 v, FILE* fp)
{
    fprintf(fp, "\n");
    fprintf(fp, "%g %g %g \n", v[0], v[1], v[2]);
}
inline void mat3dump(mat3 a, FILE* fp)
{
  fprintf(fp, "\n");
  fprintf(fp, "%15.6f %15.6f %15.6f \n", a[0][0], a[0][1], a[0][2]);
  fprintf(fp, "%15.6f %15.6f %15.6f \n", a[1][0], a[1][1], a[1][2]);
  fprintf(fp, "%15.6f %15.6f %15.6f \n", a[2][0], a[2][1], a[2][2]);
  //fprintf(fp, "%g %g %g \n", a[0][0], a[0][1], a[0][2]);
  //fprintf(fp, "%g %g %g \n", a[1][0], a[1][1], a[1][2]);
  //fprintf(fp, "%g %g %g \n", a[2][0], a[2][1], a[2][2]);
}
#endif

// porting to c++ was not easy (conflicts), sadlo 2005-12-07

#define OPTIMIZE_MEM 0 // seems to slow down and gain little #### do not activate, it is buggy

const char *findStringDel(const char *str, const char *substring, const char *delimiter)
{ // returns position of delimited substring inside str or NULL if no such
    char *delim, *next, *curr;

    if (!str || !substring || !delimiter)
        return NULL;

    char *buf = (char *)malloc(strlen(str) + 1);
    strcpy(buf, str);
    curr = buf;

    while (curr)
    {
        delim = strstr(curr, delimiter);
        if (delim)
        {
            next = delim + 1;
            *delim = '\0';
        }
        else
        {
            next = NULL;
        }

        if (strcmp(curr, substring) == 0)
        {
            const char *ret = str + (curr - buf);
            free(buf);
            return ret;
        }

        curr = next;
    }

    free(buf);

    return NULL;
}

void getNodeComponentLabel(char *node_component_labels, int c, char *res_label)
{
    char *wp = node_component_labels;
    char *lastLab = node_component_labels;
    char *del = NULL;
    for (int cc = 0; cc <= c; cc++)
    {
        lastLab = wp;
        del = strchr(wp, ';'); // #### hard coded ';'
        if (del)
        {
            wp = del + 1;
        }
        else
        {
            break;
        }
    }

    if (del)
    {
        strncpy(res_label, lastLab, del - lastLab);
        res_label[del - lastLab] = '\0';
    }
    else
    {
        strcpy(res_label, lastLab);
    }
    return;
}

#if 0 // unused and not tested
int stringCount(char *str, char *delimiter)
{
  int count = 0;

  if (strlen(str) == 0) return 0;

  while (str && *str) {
    str = strstr(curr, delimiter);
    if (str) {
      count++;
      str += strlen(delimiter);
    }
    else return count + 1;
  }
}
#endif

// paste from avs_ext
int findInIntArray(int key, int *arr, int arr_len)
{
    int i;
    for (i = 0; i < arr_len; i++)
    {
        if (arr[i] == key)
        {
            return i;
        }
    }
    return -1;
}

#if 0
int findInSortedIntArray(int key, int *arr, int arr_len)
{
  if (!arr || (arr_len < 1)) return -1; // not found
  //if (!arr || (arr_len < 1)) return -INT_MAX; // not found // ###

  int ascending = (arr[0] <= arr[arr_len-1]);

  int left = 0;
  int right = arr_len - 1;
  int mid = (left + right) / 2;

  while (1) {
    if (key == arr[mid]) {
      // found
      return mid;
    }
    else if (left >= right) {
      return -mid - 1; // not found #### val
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

int insertInSortedIntArray(int key, int ascending, int *arr, int *arr_len)
{ // arr must be large enough
  // changes array (indices may invalidate)

  if (!(*arr_len % 1000)) {
    printf("%d k array elems before insertion\r", *arr_len / 1000);
    //int i;
    //for (i=0; i<*arr_len; i++) {
    //printf(" %d", arr[i]);
    //}
    //printf("\n");
  }

#if 0
  // ### linear search
  for (i=0; i<*arr_len; i++) {
    if (arr[i] == key) {
      // found -> ### no multiple
      return i;
    }
    else if ((ascending ? key < arr[i] : key > arr[i])) {
      // insert before current
      memmove(&arr[i+1], &arr[i], (*arr_len - i)*sizeof(int));
      arr[i] = key;
      (*arr_len)++;
      return i;
    }
  }
  arr[*arr_len] = key;
  (*arr_len)++;
  return *arr_len - 1;
#else
  // binary search

  if (!arr || *arr_len < 1) {
    // no entry -> first
    arr[0] = key;
    (*arr_len)++;
    return 0;
  }

  int w = findInSortedIntArray(key, arr, *arr_len);
  if (w >= 0) {
    // found -> ### no multiple
    return w;
  }
  
  int w2 = -(w+1);
  if ((ascending ? key < arr[w2] : key > arr[w2])) {
    // insert before
    memmove(&arr[w2+1], &arr[w2], (*arr_len - w2)*sizeof(int));
    arr[w2] = key;
    (*arr_len)++;
    return w2;
  }
  else {
    if (w2+1 > *arr_len-1) {
      // no next
      // insert after
      memmove(&arr[w2+1+1], &arr[w2+1], (*arr_len - (w2+1))*sizeof(int));
      arr[w2+1] = key;
      (*arr_len)++;
      return w2+1;
    }
    else {
      // ##### seems that this branch is not needed
      if ((ascending ? key < arr[w2+1] : key > arr[w2+1])) {
	// insert after current
	memmove(&arr[w2+1+1], &arr[w2+1], (*arr_len - (w2+1))*sizeof(int));
	arr[w2+1] = key;
	(*arr_len)++;
	return w2+1;
      }
      else {
	// insert after next
	memmove(&arr[w2+2+1], &arr[w2+2], (*arr_len - (w2+2))*sizeof(int));
	arr[w2+2] = key;
	(*arr_len)++;
	return w2+2;
      }
    }
  }
#endif
}
#endif

int cvtCellNodeIDs_CFX2AVS(int *cfxNodes, int cfxType, int *avsNodes)
{ // converts cell's local node order from CFX to AVS
    // returns number of nodes converted (belonging to cell type)

    int size = 0;

    if (cfxType == cfxELEM_TET)
    {
        /* // ###### TODO this is the version according to the manuals but Ronny's ucd_check complains
    avsNodes[0] = cfxNodes[3] - 1;
    avsNodes[1] = cfxNodes[0] - 1;
    avsNodes[2] = cfxNodes[1] - 1;
    avsNodes[3] = cfxNodes[2] - 1;
    */
        avsNodes[0] = cfxNodes[0] - 1;
        avsNodes[1] = cfxNodes[1] - 1;
        avsNodes[2] = cfxNodes[2] - 1;
        avsNodes[3] = cfxNodes[3] - 1;
        size = 4;
    }
    else if (cfxType == cfxELEM_PYR)
    {
        avsNodes[0] = cfxNodes[4] - 1;
        avsNodes[1] = cfxNodes[0] - 1;
        avsNodes[2] = cfxNodes[1] - 1;
        avsNodes[3] = cfxNodes[2] - 1;
        avsNodes[4] = cfxNodes[3] - 1;
        size = 5;
    }
    else if (cfxType == cfxELEM_WDG)
    {
        avsNodes[0] = cfxNodes[3] - 1;
        avsNodes[1] = cfxNodes[4] - 1;
        avsNodes[2] = cfxNodes[5] - 1;
        avsNodes[3] = cfxNodes[0] - 1;
        avsNodes[4] = cfxNodes[1] - 1;
        avsNodes[5] = cfxNodes[2] - 1;
        size = 6;
    }
    else if (cfxType == cfxELEM_HEX)
    {
        avsNodes[0] = cfxNodes[4] - 1;
        avsNodes[1] = cfxNodes[5] - 1;
        avsNodes[2] = cfxNodes[7] - 1;
        avsNodes[3] = cfxNodes[6] - 1;
        avsNodes[4] = cfxNodes[0] - 1;
        avsNodes[5] = cfxNodes[1] - 1;
        avsNodes[6] = cfxNodes[3] - 1;
        avsNodes[7] = cfxNodes[2] - 1;
        size = 8;
    }
    else
    {
        // TODO: ####### test if other elements compatible to AVS
        int i;
        for (i = 0; i < cfxType; i++)
        {
            avsNodes[i] = cfxNodes[i] - 1;
        }
        size = cfxType; // ### ok for POINT type too?
    }
    return size;
}

int getCFXRelFaceNodeIDs(int cfxRelFace, int cfxCellType,
                         int *cellNodeIDs, int *faceNodeIDs)
{ // get ordered node IDs (1-based) belonging to relative cell face
    // returns number of vertices (nodes) that belong to face
    // TODO: pyramids (need to get appropriate data first for testing

    int vertCnt = 0;

    if (cfxCellType == cfxELEM_TET)
    {
        switch (cfxRelFace)
        {
        // these are reverse-engineered
        case 1:
        { // not tested
            faceNodeIDs[0] = cellNodeIDs[0];
            faceNodeIDs[1] = cellNodeIDs[1];
            faceNodeIDs[2] = cellNodeIDs[2];
        }
        break;
        case 2:
        { // probably ok (had no good data for testing)
            faceNodeIDs[0] = cellNodeIDs[3];
            faceNodeIDs[1] = cellNodeIDs[1];
            faceNodeIDs[2] = cellNodeIDs[0];
        }
        break;
        case 3:
        { // ok
            faceNodeIDs[0] = cellNodeIDs[3];
            faceNodeIDs[1] = cellNodeIDs[2];
            faceNodeIDs[2] = cellNodeIDs[1];
        }
        break;
        case 4:
        { // ok
            faceNodeIDs[0] = cellNodeIDs[0];
            faceNodeIDs[1] = cellNodeIDs[2];
            faceNodeIDs[2] = cellNodeIDs[3];
        }
        break;
        }
        vertCnt = 3;
    }
    else if (cfxCellType == cfxELEM_WDG)
    {
        switch (cfxRelFace)
        {
        // these are reverse-engineered
        case 1:
        { // not tested
            faceNodeIDs[0] = cellNodeIDs[0];
            faceNodeIDs[1] = cellNodeIDs[2];
            faceNodeIDs[2] = cellNodeIDs[5];
            faceNodeIDs[3] = cellNodeIDs[3];
            vertCnt = 4;
        }
        break;
        case 2:
        { // ok?
            faceNodeIDs[0] = cellNodeIDs[1];
            faceNodeIDs[1] = cellNodeIDs[0];
            faceNodeIDs[2] = cellNodeIDs[3];
            faceNodeIDs[3] = cellNodeIDs[4];
            vertCnt = 4;
        }
        break;
        case 3:
        { // ok
            faceNodeIDs[0] = cellNodeIDs[2];
            faceNodeIDs[1] = cellNodeIDs[1];
            faceNodeIDs[2] = cellNodeIDs[4];
            faceNodeIDs[3] = cellNodeIDs[5];
            vertCnt = 4;
        }
        break;
        case 4:
        { // ok
            faceNodeIDs[0] = cellNodeIDs[0];
            faceNodeIDs[1] = cellNodeIDs[1];
            faceNodeIDs[2] = cellNodeIDs[2];
            vertCnt = 3;
        }
        break;
        case 5:
        { // not tested
            faceNodeIDs[0] = cellNodeIDs[5];
            faceNodeIDs[1] = cellNodeIDs[4];
            faceNodeIDs[2] = cellNodeIDs[3];
            vertCnt = 3;
        }
        break;
        }
    }
    else if (cfxCellType == cfxELEM_HEX)
    {
        switch (cfxRelFace)
        {
        // these are reverse-engineered
        case 1:
        { // 'back'
            faceNodeIDs[0] = cellNodeIDs[0];
            faceNodeIDs[1] = cellNodeIDs[2];
            faceNodeIDs[2] = cellNodeIDs[6];
            faceNodeIDs[3] = cellNodeIDs[4];
        }
        break;
        case 2:
        { // 'front'
            faceNodeIDs[0] = cellNodeIDs[3];
            faceNodeIDs[1] = cellNodeIDs[1];
            faceNodeIDs[2] = cellNodeIDs[5];
            faceNodeIDs[3] = cellNodeIDs[7];
        }
        break;
        case 3:
        { // 'left'
            faceNodeIDs[0] = cellNodeIDs[1];
            faceNodeIDs[1] = cellNodeIDs[0];
            faceNodeIDs[2] = cellNodeIDs[4];
            faceNodeIDs[3] = cellNodeIDs[5];
        }
        break;
        case 4:
        { // 'right'
            faceNodeIDs[0] = cellNodeIDs[2];
            faceNodeIDs[1] = cellNodeIDs[3];
            faceNodeIDs[2] = cellNodeIDs[7];
            faceNodeIDs[3] = cellNodeIDs[6];
        }
        break;
        case 5:
        { // 'bottom'
            faceNodeIDs[0] = cellNodeIDs[2];
            faceNodeIDs[1] = cellNodeIDs[0];
            faceNodeIDs[2] = cellNodeIDs[1];
            faceNodeIDs[3] = cellNodeIDs[3];
        }
        break;
        case 6:
        { // 'top'
            faceNodeIDs[0] = cellNodeIDs[5];
            faceNodeIDs[1] = cellNodeIDs[4];
            faceNodeIDs[2] = cellNodeIDs[6];
            faceNodeIDs[3] = cellNodeIDs[7];
        }
        break;
        }
        vertCnt = 4;
    }
    else
    {
        printf("getCFXRelFaceNodeIDs: ERROR: unsupported cell type\n");
    }

    return vertCnt;
}

void setupTime(int isTimestep, int *timestep, int timestep_by_idx, int *nTimeDig, int *t1, int *t2, float *timeVal, char errmsg[256])
{
    if (isTimestep && *timestep == -1 && !cfxExportTimestepCount())
    {
        isTimestep = 0;
    }

    if (isTimestep)
    {
        int i;
        float f;

        if (*timestep == -1)
        {
            printf("processing all timesteps\n");
            *t1 = 1;
            *t2 = cfxExportTimestepCount() + 1;
        }
        else
        {
            int isFound = 0;
            printf("processing timestep %d\n", *timestep);
            for (i = 1; i <= cfxExportTimestepCount() + 1; i++)
            {
                if (timestep_by_idx)
                {
                    if (i == *timestep)
                    {
                        *timeVal = cfxExportTimestepTimeGet(i);
                        *t1 = *t2 = i;
                        isFound = 1;
                        break;
                    }
                }
                else
                {
                    if (cfxExportTimestepNumGet(i) == *timestep)
                    {
                        *timeVal = cfxExportTimestepTimeGet(i);
                        *t1 = *t2 = i;
                        isFound = 1;
                        break;
                    }
                }
            }
            if (!isFound)
            {
                sprintf(errmsg, "\nTimestep %d not found. "
                                "Use -f to see the list of valid timesteps.\n",
                        *timestep);
                cfxExportFatal(errmsg);
            }
        }
        // count number of digits needed to fit any timestep number
        f = (float)cfxExportTimestepCount();
        while ((f /= 10) >= 1)
            (*nTimeDig)++;
    }
    else
    {
        *timeVal = cfxExportTimestepTimeGet(cfxExportTimestepCount() + 1);
        *timestep = cfxExportTimestepNumGet(cfxExportTimestepCount() + 1);
        *t1 = *t2 = cfxExportTimestepCount() + 1;
    }

    printf("time value: %g\n", *timeVal);
}

int cfx_getInfo(const char *file_name, int level_of_interest,
                int zone,
                int crop, float cropXMin, float cropXMax,
                float cropYMin, float cropYMax,
                float cropZMin, float cropZMax,
                int time_step, int timestep_by_idx,
                int *num_tetra, int *num_pyra, int *num_wedge, int *num_hexa,
                int *nnodes,
                char *components_to_read, char *components_to_read_delimiter,
                int output_zone_id,
                int *node_veclen, int *num_node_components,
                int output_boundaryNodes, int *num_boundaries,
                float *time_val,
                int *timeStepCnt,
                int allow_zone_rotation,
                char *ucdName,
                int exportInit, int exportDone)
{ // zone: 0 for all zones (domains)
    // components_to_read: if NULL, all are read, otherwise only with those labels
    // output_boundaryNodes: nonzero for yes
    // *time_val: may be NULL
    // returns zero on success

    // ######## TODO: support timesteps

    int num_zones; // (domains)
    int counts[cfxCNT_SIZE];
    int nscalars, nvectors, nvar;

    printf("reading CFX-5 results info from <%s>\n", file_name);

    if (exportInit)
    {
#if 1 // ### WORK-AROUND
        char file_name2[256];
        strcpy(file_name2, file_name);
        num_zones = cfxExportInit(file_name2, NULL); // ### counts .. ?
#else
        num_zones = cfxExportInit(file_name, NULL); // ### counts .. ?
#endif
    }

    cfxExportSetVarParams(0 /*correct*/, level_of_interest);

    if (cfxExportZoneSet(zone, counts) < 0)
    {
        cfxExportFatal("invalid zone number");
    }

    nvar = cfxExportVariableCount(level_of_interest);
    //printf("(info1): variable count = %d\n", nvar);

    *timeStepCnt = cfxExportTimestepCount() + 1;

    if (ucdName)
    {

#if 0
    double rotationAxis[2][3];
    double angularVelocity;
    if (allow_zone_rotation && 
        cfxExportZoneIsRotating(rotationAxis, &angularVelocity)) {
      
      if (ucdName) {
        double rotationAxis[2][3];
        double angularVelocity;
        sprintf(ucdName, "%g,%g,%g %g,%g,%g %g",
                rotationAxis[0][0],  rotationAxis[0][1],  rotationAxis[0][2],
                rotationAxis[1][0],  rotationAxis[1][1],  rotationAxis[1][2],
                angularVelocity);
        printf("info: %g,%g,%g %g,%g,%g %g\n",  rotationAxis[0][0],  rotationAxis[0][1],  rotationAxis[0][2],
                rotationAxis[1][0],  rotationAxis[1][1],  rotationAxis[1][2],
                angularVelocity);
      }
    }
    else {
      if (ucdName) strcpy(ucdName, cfxExportZoneName(zone));
    }
#else
        if (ucdName)
        {
            if (cfxExportZoneName(zone))
                strcpy(ucdName, cfxExportZoneName(zone));
            else
                strcpy(ucdName, "cfx");
#if 0
      strcat(ucdName, "                                                                                                                                                                                                                                                               ");
#endif
        }
#endif
    }

    if (!crop)
    {
        // no crop
        *nnodes = cfxExportNodeCount();

        *num_tetra = counts[cfxCNT_TET];
        *num_pyra = counts[cfxCNT_PYR];
        *num_wedge = counts[cfxCNT_WDG];
        *num_hexa = counts[cfxCNT_HEX];
    }
    else
    {
        // crop

        //*nnodes = cfxExportNodeCount();
        *num_tetra = 0;
        *num_pyra = 0;
        *num_wedge = 0;
        *num_hexa = 0;

        int e, i;
        //cfxElement *elems;
        int nelems = cfxExportElementCount();

        {
            int *usedNodeIds = NULL;
            int usedNodeIdCnt = 0;

            //unsigned char *cell_typesPtr = cell_types;
            // elems = cfxExportElementList();
            //for (n=0; n<nelems; n++,elems++) {
            printf("cropping for info\n");
            for (e = 1; e <= nelems; e++)
            {
                int eType, nodeL[8];
                cfxExportElementGet(e, &eType, nodeL);

                int inside = 1;

                for (i = 0; i < eType; i++)
                {

                    double pos[3];
                    cfxExportNodeGet(nodeL[i], &pos[0], &pos[1], &pos[2]);

                    if ((pos[0] < cropXMin) || (pos[0] > cropXMax) || (pos[1] < cropYMin) || (pos[1] > cropYMax) || (pos[2] < cropZMin) || (pos[2] > cropZMax))
                    {
                        inside = 0;
                        break;
                    }
                }

                if (!inside)
                    continue;

                // count cells
                switch (eType)
                {
                case cfxELEM_TET:
                    (*num_tetra)++;
                    break;
                case cfxELEM_PYR:
                    (*num_pyra)++;
                    break;
                case cfxELEM_WDG:
                    (*num_wedge)++;
                    break;
                case cfxELEM_HEX:
                    (*num_hexa)++;
                    break;
                }

                // count nodes
                for (i = 0; i < eType; i++)
                {

                    //double pos[3];
                    //cfxExportNodeGet(nodeL[i], &pos[0], &pos[1], &pos[2]);

                    // decide if node alredy used
                    // ##### to do: binary search on sorted array
                    //if (findInIntArray(nodeL[i], usedNodeIds, usedNodeIdCnt) >= 0) {
                    if (Unstructured::findInSortedIntArray(nodeL[i], usedNodeIds, usedNodeIdCnt) >= 0)
                    {
                        // node already used
                        continue;
                    }
                    else
                    {
                        // new node

                        usedNodeIds = (int *)realloc(usedNodeIds, (usedNodeIdCnt + 1) * sizeof(int));
                        //usedNodeIds[usedNodeIdCnt] = nodeL[i];
                        //usedNodeIdCnt++;
                        Unstructured::insertInSortedIntArray(nodeL[i], 1, usedNodeIds, &usedNodeIdCnt);
                    }
                }
            }

            free(usedNodeIds);
            *nnodes = usedNodeIdCnt;
        }
    }

#if 1
    int t;
    {
        int isTimestep = 0;
        int timestep = -1;
        char errmsg[256];

        if (time_step >= 0)
        {
            timestep = time_step;
            isTimestep = 1;
        }

        // timestep setup
        {
            if (isTimestep && timestep == -1 && !cfxExportTimestepCount())
            {
                isTimestep = 0;
            }

            if (isTimestep)
            {
                int i;
                float f;

                int isFound = 0;
                printf("processing timestep %d\n", timestep);
                for (i = 1; i <= cfxExportTimestepCount() + 1; i++)
                {
                    if (timestep_by_idx)
                    {
                        if (i == timestep)
                        {
                            //timeVal = cfxExportTimestepTimeGet(i);
                            t = i;
                            isFound = 1;
                            break;
                        }
                    }
                    else
                    {
                        if (cfxExportTimestepNumGet(i) == timestep)
                        {
                            //timeVal = cfxExportTimestepTimeGet(i);
                            t = i;
                            isFound = 1;
                            break;
                        }
                    }
                }
                if (!isFound)
                {
                    sprintf(errmsg, "\nTimestep %d not found. "
                                    "Use -f to see the list of valid timesteps.\n",
                            timestep);
                    cfxExportFatal(errmsg);
                }

                // count number of digits needed to fit any timestep number
                f = (float)cfxExportTimestepCount();
                //while((f /= 10) >= 1) nTimeDig++;
            }
            else
            {
                //timeVal = cfxExportTimestepTimeGet(cfxExportTimestepCount() + 1);
                timestep = cfxExportTimestepNumGet(cfxExportTimestepCount() + 1);
                t = cfxExportTimestepCount() + 1;
            }
        }
    }

    int ts;
    ts = cfxExportTimestepNumGet(t);
    if (cfxExportTimestepSet(ts) < 0)
    {
        // fail, ### TODO: do something
    }

    if (time_val)
        *time_val = cfxExportTimestepTimeGet(t);

#if 1 // ############## 2006-03-27
    nvar = cfxExportVariableCount(level_of_interest);
//nvar = 5; // ##################################################
//printf("(info2): variable count = %d\n", nvar);
#endif

    // get nscalars, nvectors, node_veclen
    {
        int v, i;
        //int bnddat = 0; // ### to do
        int bnddat = 1; // 2007-04-24 TEST
        int dim, length;

        *node_veclen = 0;
        nscalars = nvectors = 0;
        for (v = 1; v <= nvar; v++)
        {
            cfxExportVariableSize(v, &dim, &length, &i);

            if (components_to_read && !findStringDel(components_to_read, cfxExportVariableName(v, 1), // 1: alias
                                                     components_to_read_delimiter))
            {
                continue;
            }

            //printf("nvar=%d var=%d dim=%d\n", nvar, v, dim);

            // ### this is a HACK from the CFX export template, TODO: generalize
            if ((1 != dim && 3 != dim) || (length != cfxExportNodeCount() && length != bnddat))
                continue;

            if (1 == dim)
            {
                nscalars++;
            }
            else
            {
                nvectors++;
            }

            *node_veclen += dim;
        }
        //printf("nscalars=%d nvectors=%d\n", nscalars, nvectors);
    }

#else

    // get nscalars, nvectors, node_veclen
    {
        int v, i;
        int bnddat = 0; // ### to do
        int dim, length;

        *node_veclen = 0;
        nscalars = nvectors = 0;
        for (v = 1; v <= nvar; v++)
        {
            cfxExportVariableSize(v, &dim, &length, &i);

            if (components_to_read && !findStringDel(components_to_read, cfxExportVariableName(v, 1 /*alias*/), components_to_read_delimiter))
            {
                continue;
            }

            // ### this is a HACK from the CFX export template, TODO: generalize
            if ((1 != dim && 3 != dim) || (length != *nnodes && length != bnddat))
                continue;

            if (1 == dim)
            {
                nscalars++;
            }
            else
            {
                nvectors++;
            }

            *node_veclen += dim;
        }
    }
#endif

    *num_node_components = nvectors + nscalars;

    if (output_boundaryNodes)
    {
        *num_node_components += 1;
        *node_veclen += 1;
        *num_boundaries = cfxExportBoundaryCount();
    }

    if (output_zone_id)
    {
        *num_node_components += 1;
        *node_veclen += 1;
    }

    if (exportDone)
    {
        cfxExportDone();
    }

    return 0; // success
}

int cfx_getData(const char *file_name, int level_of_interest, int zone,
                int crop, float cropXMin, float cropXMax,
                float cropYMin, float cropYMax,
                float cropZMin, float cropZMax,
                int time_step, int timestep_by_idx,
                float *node_x, float *node_y, float *node_z,
                int req_cell_type,
                int *node_list,
                unsigned char *cell_types,
                char *components_to_read, char *components_to_read_delimiter,
                float *node_data,
                int *node_components,
                char *node_component_labels,
                int fix_boundary_nodes,
                int output_zone_id,
                int output_boundaryNodes, char *boundary_node_label, char *boundary_node_labels,
                char *search_string,
                int allowZoneRotation,
                int output_boundaries, int output_regions,
                int remove_faces, double faceNormal[3],
                int **geomNodes, int *geomNodeCnt, int *geomObjCnt, int *geomNodeObjSizes, char geomObjNames[][256], int *geomPListCounts,
                char *ucdName,
                int exportInit, int exportDone)
{ // zone: 0 for all zones (domains)
    // times_tep: < 0 for no (first) timestep
    // node_x, node_y, node_z: only read if all not NULL
    // node_list: only read if not NULL
    // cell_types: only read if not NULL
    // node_data: only read if not NULL
    // node_components: only read if not NULL
    // node_component_labels: only read if not NULL
    // output_boundaryNodes: nonzero for yes
    // boundary_node_label: label of boundary to be read, boundary only read if not NULL
    // boundary_node_labels: only read if not NULL
    // returns zero on success

    // ######## TODO: support timesteps

    int num_zones; // (domains)
    int counts[cfxCNT_SIZE];
    int nscalars, nvectors, dim;
    int nnodes; // node_veclen;
    int t1, t2;
    int timestep = -1;
    int isTimestep = 0;
    int nTimeDig = 1; // number of digits in transient file suffix
    float timeVal = 0.0; // time value in the single timestep mode
    char errmsg[256];
    //int bnddat = 0; // ### to do
    int bnddat = 1; // 2007-04-24 TEST
    //int bndfix = 0; // ### to do
    int bndfix = fix_boundary_nodes;
    int alias = 1;
    int namelen, nvar;
    float *node_dataPtr = node_data;

    if (time_step >= 0)
    {
        timestep = time_step;
        isTimestep = 1;
    }

    printf("reading CFX-5 results data from <%s>\n", file_name);

    printf("trying to read:\n");
    if (node_x && node_y && node_z)
        printf("  node positions\n");
    if (node_list)
        printf("  node connectivity\n");
    if (cell_types)
        printf("  cell types\n");
    if (node_data)
        printf("  node data\n");
    if (node_components)
        printf("  node components\n");
    if (node_component_labels)
        printf("  node component labels\n");
    if (boundary_node_labels)
        printf("  boundary node labels\n");

    if (exportInit)
    {
#if 1 // ### WORK-AROUND
        char file_name2[256];
        strcpy(file_name2, file_name);
        num_zones = cfxExportInit(file_name2, NULL); // ### counts .. ?
#else
        num_zones = cfxExportInit(file_name, NULL); // ### counts .. ?
#endif
    }

    cfxExportSetVarParams(bndfix, level_of_interest);

    if (cfxExportZoneSet(zone, counts) < 0)
    {
        cfxExportFatal("invalid zone number");
    }

    printf("zone[%d] contains %d volumes:\n", zone, cfxExportVolumeCount());
    {
        int i;
        for (i = 1; i <= cfxExportVolumeCount(); i++)
        {
            printf("  volume[%d]: %s\n", i, cfxExportVolumeName(i));
        }
    }

    if (ucdName)
    {
        if (cfxExportZoneName(zone))
            strcpy(ucdName, cfxExportZoneName(zone));
        else
            strcpy(ucdName, "cfx");
#if 0
    strcat(ucdName, "                                                                                                                                                                                                                                                               ");
#endif
    }

    // for crop mode:
    int *usedNodeIds = NULL;
    int usedNodeIdCnt = 0;

    if (!crop)
    {

#if 0 // ###### 2006-03-27
    if (time_step > 0) {
      int ts = cfxExportTimestepNumGet(time_step);
      cfxExportTimestepSet(ts);
    }
#endif

        nnodes = cfxExportNodeCount();
// ### this seems to rely on timestep from previous getInfo() call .. HACK
#if 1 // ################## orig
        nvar = cfxExportVariableCount(level_of_interest);
#else
        // HACK #################################################################
        nvar = 5;
#endif
        //printf("(data1): variable count = %d\n", nvar);

        // node positions
        if (node_x && node_y && node_z)
        {
            int n;
            cfxNode *nodes;

            nodes = cfxExportNodeList();

            {
                double rotationAxis[2][3];
                double angularVelocity;
                if (allowZoneRotation && cfxExportZoneIsRotating(rotationAxis, &angularVelocity))
                {

                    cfxExportZoneMotionAction(zone, cfxMOTION_IGNORE);

                    int timestepW = timestep;
                    int nTimeDigW = nTimeDig;
                    int t1W = t1;
                    int t2W = t2;
                    float timeValW = timeVal;
                    setupTime(isTimestep, &timestepW, timestep_by_idx, &nTimeDigW, &t1W, &t2W, &timeValW, errmsg);
                    double angle = timeValW * angularVelocity;
                    vec3 rotVect;
#if 0 // ### this seems to be wrong
          rotVect[0] = rotationAxis[1][0];
          rotVect[1] = rotationAxis[1][1];
          rotVect[2] = rotationAxis[1][2];
#else // ###### HACK
                    rotVect[0] = -rotationAxis[1][2];
                    rotVect[1] = -rotationAxis[1][1];
                    rotVect[2] = -rotationAxis[1][0];
#endif
                    printf("center: %g,%g,%g axis: %g,%g,%g\n",
                           rotationAxis[0][0], rotationAxis[0][1], rotationAxis[0][2],
                           rotationAxis[1][0], rotationAxis[1][1], rotationAxis[1][2]);
                    //vec3dump(rotVect, stderr);
                    vec3nrm(rotVect, rotVect);
                    vec3scal(rotVect, angle, rotVect);
                    mat3 rotMat;
                    rotVectTomat3(rotVect, rotMat);

                    vec3 center;
                    center[0] = rotationAxis[0][0];
                    center[1] = rotationAxis[0][1];
                    center[2] = rotationAxis[0][2];

                    printf("###rotating zone by %g rad\n", angle);
                    mat3dump(rotMat, stderr);

                    if (ucdName)
                    {
#if 0
            sprintf(ucdName, "rotating %d %g,%g,%g %g,%g,%g %g                                                                      ",
                    zone, 
                    rotationAxis[0][0],  rotationAxis[0][1],  rotationAxis[0][2],
                    // ##### HACK!
                    -rotationAxis[1][2], -rotationAxis[1][1], -rotationAxis[1][0], 
                    angularVelocity);
#else
                        sprintf(ucdName, "rotating %d %g,%g,%g %g,%g,%g %g",
                                zone,
                                rotationAxis[0][0], rotationAxis[0][1], rotationAxis[0][2],
                                // ##### HACK!
                                -rotationAxis[1][2], -rotationAxis[1][1], -rotationAxis[1][0],
                                angularVelocity);
#endif
                    }

                    for (n = 0; n < nnodes; n++, nodes++)
                    {

                        vec3 pos = { nodes->x - center[0],
                                     nodes->y - center[1],
                                     nodes->z - center[2] };
                        //vec3dump(pos, stderr);
                        mat3vec(rotMat, pos, pos);
                        //vec3dump(pos, stderr);

                        pos[0] += center[0];
                        pos[1] += center[1];
                        pos[2] += center[2];

                        node_x[n] = pos[0];
                        node_y[n] = pos[1];
                        node_z[n] = pos[2];
                    }
                }
                else
                {

                    // ### this seems to enable the rotation by the library itself,
                    // however, (with cfx11sp it seems to get rotated around the
                    // wrong axis)
                    //cfxExportZoneMotionAction(zone, cfxMOTION_USE);

                    for (n = 0; n < nnodes; n++, nodes++)
                    {
                        node_x[n] = nodes->x;
                        node_y[n] = nodes->y;
                        node_z[n] = nodes->z;
                    }
                }
            }
            cfxExportNodeFree(); // ###
        }

        // node list
        {
            int n;
            cfxElement *elems;
            int nelems = cfxExportElementCount();

            {
                int *node_listPtr = node_list;
                unsigned char *cell_typesPtr = cell_types;
                elems = cfxExportElementList();
                for (n = 0; n < nelems; n++, elems++)
                {
                    if (req_cell_type == REQ_TYPE_ALL || (elems->type == cfxELEM_TET && req_cell_type == REQ_TYPE_TETRA) || (elems->type == cfxELEM_PYR && req_cell_type == REQ_TYPE_PYRAM) || (elems->type == cfxELEM_WDG && req_cell_type == REQ_TYPE_WEDGE) || (elems->type == cfxELEM_HEX && req_cell_type == REQ_TYPE_HEXA))
                    {

                        if (node_list)
                        {
                            node_listPtr += cvtCellNodeIDs_CFX2AVS(elems->nodeid,
                                                                   elems->type,
                                                                   node_listPtr);
                        }

                        if (cell_types)
                        {
                            *cell_typesPtr = elems->type;
                            cell_typesPtr++;
                        }
                    }
                }
            }
            //cfxExportElementFree(); // ###
        }
    }
    else
    {
        // crop

        int *usedNodeIdsSorted = NULL;
        int *usedNodeIdsSortedTrueIdx = NULL;

        nnodes = cfxExportNodeCount();
        // ### this seems to rely on timestep from previous getInfo() call .. HACK
        nvar = cfxExportVariableCount(level_of_interest);
        //printf("(data2): variable count = %d\n", nvar);

        // node list
        {
            int n, i;
            //cfxElement *elems;
            int nelems = cfxExportElementCount();

            {
                //int *usedNodeIds = NULL;
                //int usedNodeIdCnt = 0;

                int *node_listPtr = node_list;
                unsigned char *cell_typesPtr = cell_types;
                //elems = cfxExportElementList();
                printf("cropping for data\n");
                for (n = 1; n <= nelems; n++)
                {
                    int eType, nodeL[8];
                    cfxExportElementGet(n, &eType, nodeL);

                    int inside = 1;

                    for (i = 0; i < eType; i++)
                    {

                        double pos[3];
                        cfxExportNodeGet(nodeL[i], &pos[0], &pos[1], &pos[2]);

                        if ((pos[0] < cropXMin) || (pos[0] > cropXMax) || (pos[1] < cropYMin) || (pos[1] > cropYMax) || (pos[2] < cropZMin) || (pos[2] > cropZMax))
                        {
                            inside = 0;
                            break;
                        }
                    }

                    if (!inside)
                        continue;

                    if (req_cell_type == REQ_TYPE_ALL || (eType == cfxELEM_TET && req_cell_type == REQ_TYPE_TETRA) || (eType == cfxELEM_PYR && req_cell_type == REQ_TYPE_PYRAM) || (eType == cfxELEM_WDG && req_cell_type == REQ_TYPE_WEDGE) || (eType == cfxELEM_HEX && req_cell_type == REQ_TYPE_HEXA))
                    {

                        // count nodes
                        int nodeLNew[8];
                        for (i = 0; i < eType; i++)
                        {

                            //double pos[3];
                            //cfxExportNodeGet(nodeL[i], &pos[0], &pos[1], &pos[2]);

                            // decide if node alredy used
                            //int w = findInIntArray(nodeL[i], usedNodeIds, usedNodeIdCnt);
                            int w = Unstructured::findInSortedIntArray(nodeL[i], usedNodeIdsSorted, usedNodeIdCnt);
                            if (w >= 0)
                            {
                                // node already used
                                nodeLNew[i] = usedNodeIdsSortedTrueIdx[w] + 1;
                                continue;
                            }
                            else
                            {
                                // new node

                                usedNodeIds = (int *)realloc(usedNodeIds, (usedNodeIdCnt + 1) * sizeof(int));
                                usedNodeIdsSorted = (int *)realloc(usedNodeIdsSorted, (usedNodeIdCnt + 1) * sizeof(int));
                                usedNodeIdsSortedTrueIdx = (int *)realloc(usedNodeIdsSortedTrueIdx, (usedNodeIdCnt + 1) * sizeof(int));
                                usedNodeIds[usedNodeIdCnt] = nodeL[i];
                                nodeLNew[i] = usedNodeIdCnt + 1;
                                int w = Unstructured::insertInSortedIntArray(nodeL[i], 1, usedNodeIdsSorted, &usedNodeIdCnt);
                                //usedNodeIdCnt++;
                                memmove(&usedNodeIdsSortedTrueIdx[w + 1], &usedNodeIdsSortedTrueIdx[w], (usedNodeIdCnt - 1 - w) * sizeof(int));
                                usedNodeIdsSortedTrueIdx[w] = usedNodeIdCnt - 1;
                            }
                        }

                        // get new ids
                        //for (i=0; i<eType; i++) {
                        //nodeLNew[i] = findInSortedIntArray(nodeL[i], usedNodeIds, usedNodeIdCnt) + 1;
                        //}

                        if (node_list)
                        {
                            node_listPtr += cvtCellNodeIDs_CFX2AVS(nodeLNew, eType, node_listPtr);
                        }

                        if (cell_types)
                        {
                            *cell_typesPtr = eType;
                            cell_typesPtr++;
                        }
                    }
                }
                //free(usedNodeIds);
            }
            //cfxExportElementFree(); // ###
        }

        // node positions
        if (node_x && node_y && node_z)
        {
            int n;

            for (n = 0; n < usedNodeIdCnt; n++)
            {
                double pos[3];
                cfxExportNodeGet(usedNodeIds[n], &pos[0], &pos[1], &pos[2]);

                node_x[n] = pos[0];
                node_y[n] = pos[1];
                node_z[n] = pos[2];
            }
        }

        free(usedNodeIdsSorted);
        free(usedNodeIdsSortedTrueIdx);
    }

#if 0 // ############################ 2006-03-27
  nvar = cfxExportVariableCount(level_of_interest);
  nvar = 4; // ############################################3
#endif

    // added 2008-02-28: ################################################
    nvar = cfxExportVariableCount(level_of_interest);
    //printf("(data3): variable count = %d\n", nvar);

    // node components
    {
        int n, i;
        int dim, length;

        if (node_component_labels)
            strcpy(node_component_labels, "");

        nscalars = nvectors = namelen = 0;
        for (n = 1; n <= nvar; n++)
        {
            cfxExportVariableSize(n, &dim, &length, &i);

            if (components_to_read && !findStringDel(components_to_read, cfxExportVariableName(n, alias), components_to_read_delimiter))
            {
                continue;
            }

            if ((1 != dim && 3 != dim) || (length != cfxExportNodeCount() && length != bnddat))
                continue;

            if (1 == dim)
            {
                nscalars++;
            }
            else
            {
                nvectors++;
            }

            if (node_components)
                node_components[nscalars + nvectors - 1] = dim;
            if (node_component_labels)
            {
                if (nscalars + nvectors > 1)
                    strcat(node_component_labels, ";");
                strcat(node_component_labels, cfxExportVariableName(n, alias));
            }

            i = strlen(cfxExportVariableName(n, alias));
            if (namelen < i)
            {
                namelen = i;
            }
        }

        if (output_boundaryNodes)
        {
            if (node_components)
                node_components[nvectors + nscalars] = 1;
            if (nvectors + nscalars > 0)
                strcat(node_component_labels, ";");
            if (node_component_labels)
                strcat(node_component_labels, "boundary nodes");
        }

        if (output_zone_id)
        {
            if (node_components)
            {
                if (output_boundaryNodes)
                    node_components[nvectors + nscalars + 1] = 1;
                else
                    node_components[nvectors + nscalars] = 1;
            }
            if ((nvectors + nscalars > 0 || output_boundaryNodes) && node_component_labels)
                strcat(node_component_labels, ";");
            if (node_component_labels)
                strcat(node_component_labels, "zone");
        }
    }

    // read variables
    {
        int n, t, ts, i, length;
        float *var;

        // timestep setup
        setupTime(isTimestep, &timestep, timestep_by_idx, &nTimeDig, &t1, &t2, &timeVal, errmsg);

        for (t = t1; t <= t2; t++)
        {

            ts = cfxExportTimestepNumGet(t);

            if (cfxExportTimestepSet(ts) < 0)
            {
                continue;
            }

            printf("reading variables of timestep %d/%d\n", t, cfxExportTimestepCount() + 1);

            for (n = 1; n <= nvar; n++)
            {
                cfxExportVariableSize(n, &dim, &length, &i);

                if (!node_data || (components_to_read && !findStringDel(components_to_read, cfxExportVariableName(n, alias), components_to_read_delimiter)))
                {
                    continue;
                }

                // ### HACK (1, 3)
                if ((1 == dim || 3 == dim) && (length == nnodes || length == bnddat))
                {

                    if (!crop || !OPTIMIZE_MEM)
                    {
                        if (NULL == (var = cfxExportVariableList(n, bndfix)))
                        {
                            cfxExportFatal("error getting variable");
                            if (exportDone)
                            {
                                cfxExportDone();
                            }
                            return 1; // fail
                        }

                        printf("  %-*s ...", namelen, cfxExportVariableName(n, alias));
                        fflush(stdout);

                        if (node_data)
                        {
                            if (!crop)
                            {
                                length = nnodes * dim;
                                for (i = 0; i < length; i++, var++)
                                {
                                    node_dataPtr[i] = *var;
                                }

                                node_dataPtr += dim * nnodes;
                            }
                            else
                            {
                                // crop

                                int n;
                                for (n = 0; n < usedNodeIdCnt; n++)
                                {
                                    int d;
                                    for (d = 0; d < dim; d++)
                                    {
                                        node_dataPtr[n * dim + d] = var[(usedNodeIds[n] - 1) * dim + d];
                                    }
                                }

                                node_dataPtr += dim * usedNodeIdCnt;
                            }
                        }

                        cfxExportVariableFree(n); // ###
                    }
                    else
                    {
                        // crop
                        // ######## this mode contains a bug (data is not set correctly)
                        // fs 2006-04-24: still a bug?

                        //if (NULL == (var = cfxExportVariableList (n, bndfix))) {
                        // cfxExportFatal ("error getting variable");
                        //if (exportDone) {
                        //cfxExportDone();
                        //}
                        //return 1; // fail
                        //}

                        printf("  %-*s ...", namelen, cfxExportVariableName(n, alias));
                        fflush(stdout);

                        if (node_data)
                        {

                            // crop

                            int dim, length;
                            cfxExportVariableSize(n, &dim, &length, &bndfix);
                            float dat[256]; // #### HACK assumingmax. 256 dims

                            int n;
                            for (n = 0; n < usedNodeIdCnt; n++)
                            {
                                cfxExportVariableGet(n, bndfix, usedNodeIds[n], dat);
                                int d;
                                for (d = 0; d < dim; d++)
                                {
                                    node_dataPtr[n * dim + d] = dat[d];
                                }
                            }

                            node_dataPtr += dim * usedNodeIdCnt;
                        }

                        //cfxExportVariableFree(n); // ###
                    }
                    printf(" done\n");
                }
            }

            // boundary nodes
            if (output_boundaryNodes)
            {
                if (node_data)
                {
                    for (i = 0; i < nnodes; i++)
                    {
                        node_dataPtr[i] = 0.0;
                    }
                }

                if (boundary_node_labels)
                    strcpy(boundary_node_labels, "");

                int i;
                //printf("boundary_cnt=%d\n", cfxExportBoundaryCount());
                for (i = 1; i <= cfxExportBoundaryCount(); i++)
                {

                    if (node_data)
                    {
                        printf("boundary[%d]: name=%s, num_nodes=%d, num_faces=%d",
                               i,
                               cfxExportBoundaryName(i),
                               cfxExportBoundarySize(i, cfxREG_NODES),
                               cfxExportBoundarySize(i, cfxREG_FACES));
                    }

                    if (boundary_node_label && (strlen(boundary_node_label) > 0) && (strncmp(cfxExportBoundaryName(i), boundary_node_label, strlen(boundary_node_label)) == 0 || (strlen(search_string) > 0 && strstr(cfxExportBoundaryName(i), search_string))))
                    {

                        printf(" found boundary [%s]", boundary_node_label);

                        int *p = cfxExportBoundaryList(i, cfxREG_NODES);
                        int j;
                        if (node_data)
                        {
                            if (!crop)
                            { // ############ actually not supporting boundary nodes in crop mode
                                for (j = 0; j < cfxExportBoundarySize(i, cfxREG_NODES); j++)
                                {
                                    node_dataPtr[p[j] - 1] = 1.0;
                                }
                            }
                        }

                        printf(" -> boundary nodes");

                        //cfxExportBoundaryFree(i); // ###
                    }

                    printf("\n");

                    if (boundary_node_labels)
                    {
                        if (i > 1)
                            strcat(boundary_node_labels, ":");
                        strcat(boundary_node_labels, cfxExportBoundaryName(i));
                    }

                    cfxExportBoundaryFree(i); // ###
                }
            }

            if (output_zone_id)
            {
                if (node_data)
                {
                    for (i = 0; i < nnodes; i++)
                    {
                        node_dataPtr[i] = cfxExportZoneGet();
                    }
                }
            }
        }
    }

    // output geom
    if ((output_boundaries || output_regions) && geomNodes)
    {

        if (*geomNodes)
        {
            free(*geomNodes);
            *geomNodes = NULL;
        }
        *geomObjCnt = 0;
        *geomNodeCnt = 0;

        int plistCnt = 0;
        int plistLastCnt = 0;

        printf("exporting geom\n");

        // boundaries
        if (output_boundaries)
        {
            int bCnt = cfxExportBoundaryCount();
            printf("%d boundaries\n", bCnt);

            *geomObjCnt = bCnt;

            int unsupportedCellFaceCntTot = 0;
            int b;
            for (b = 1; b <= bCnt; b++)
            {

                int bSiz = cfxExportBoundarySize(b, cfxREG_FACES);
                printf("%d faces in boundary %d\n", bSiz, b);

                strcpy(&geomObjNames[b - 1][0], cfxExportBoundaryName(b));

                int *faces = cfxExportBoundaryList(b, cfxREG_FACES);
                int bb;

                int effFaceCnt = 0;
                int removedFaceCnt = 0;
                int unsupportedCellFaceCnt = 0;
                for (bb = 0; bb < bSiz; bb++)
                {
                    //printf("boundary[%d] face list [%d] = %d facenum=%d elemnum=%d\n", b, bb, p[bb], cfxFACENUM(p[bb]), cfxELEMNUM(p[bb]));

                    int type, nl[8];
                    cfxExportElementGet(cfxELEMNUM(faces[bb]), &type, nl);
                    if ((type != cfxELEM_HEX) && (type != cfxELEM_TET) && (type != cfxELEM_WDG))
                    {
                        unsupportedCellFaceCnt++;
                        unsupportedCellFaceCntTot++;
                        continue;
                    }

                    if (!(req_cell_type == REQ_TYPE_ALL || (type == cfxELEM_TET && req_cell_type == REQ_TYPE_TETRA) || (type == cfxELEM_PYR && req_cell_type == REQ_TYPE_PYRAM) || (type == cfxELEM_WDG && req_cell_type == REQ_TYPE_WEDGE) || (type == cfxELEM_HEX && req_cell_type == REQ_TYPE_HEXA)))
                    {
                        continue;
                    }

                    // get node IDs of face vertices
                    int nodesPerFace;
                    int pts[4] = { 1, 1, 1, 1 };
                    nodesPerFace = getCFXRelFaceNodeIDs(cfxFACENUM(faces[bb]), type, nl, pts);

                    if (remove_faces && node_x && node_y && node_z)
                    {
                        // compute normal ### assuming planar face
                        vec3 v1, v2, normal;
                        v1[0] = node_x[pts[1] - 1] - node_x[pts[0] - 1];
                        v1[1] = node_y[pts[1] - 1] - node_y[pts[0] - 1];
                        v1[2] = node_z[pts[1] - 1] - node_z[pts[0] - 1];

                        v2[0] = node_x[pts[nodesPerFace - 1] - 1] - node_x[pts[0] - 1];
                        v2[1] = node_y[pts[nodesPerFace - 1] - 1] - node_y[pts[0] - 1];
                        v2[2] = node_z[pts[nodesPerFace - 1] - 1] - node_z[pts[0] - 1];

                        vec3cross(v1, v2, normal);
                        vec3nrm(normal, normal);
                        vec3 facN;
                        vec3nrm(faceNormal, facN);

                        if (vec3dot(facN, normal) > 0.9)
                        {
                            // remove face
                            removedFaceCnt++;
                            continue;
                        }
                    }

                    *geomNodes = (int *)realloc(*geomNodes, (plistCnt + 1) * sizeof(int));
                    (*geomNodes)[plistCnt] = nodesPerFace;
                    plistCnt++;

                    // outputting node list (zero-based indices)
                    *geomNodes = (int *)realloc(*geomNodes, (plistCnt + nodesPerFace) * sizeof(int));
                    int k;
                    for (k = 0; k < nodesPerFace; k++)
                    {
                        (*geomNodes)[plistCnt] = pts[k] - 1;
                        plistCnt++;
                    }

                    effFaceCnt++;
                }
                //geomNodeObjSizes[b-1] = bSiz;
                geomNodeObjSizes[b - 1] = effFaceCnt;
                geomPListCounts[b - 1] = plistCnt - plistLastCnt;
                plistLastCnt = plistCnt;

                //printf("original face count = %d, %d faces remaining, %d unsupported faces skipped\n", bSiz, effFaceCnt, unsupportedCellFaceCnt);
            }

            printf("%d faces of unsupported type skipped\n", unsupportedCellFaceCntTot);
        }

        // regions
        if (output_regions)
        {
            const int areaForWorkaround = 1;

            int rCnt = cfxExportRegionCount();
            printf("%d regions\n", rCnt);

            int r;
            for (r = 1; r <= rCnt; r++)
            {

                int rSiz = cfxExportRegionSize(r, cfxREG_FACES);
                int *faces = cfxExportRegionList(r, cfxREG_FACES);
                int rr;

                int effFaceCnt = 0;
                int removedFaceCnt = 0;
                double area = 0.0;
                vec3 centroid = { 0.0, 0.0, 0.0 };
                int centroidCnt = 0;
                for (rr = 0; rr < rSiz; rr++)
                {
                    //printf("boundary[%d] face list [%d] = %d facenum=%d elemnum=%d\n", b, bb, p[bb], cfxFACENUM(p[bb]), cfxELEMNUM(p[bb]));

                    // get node IDs of face vertices
                    int nodesPerFace;
                    int pts[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }; // #### HACK assuming max 10 vertices per face
                    nodesPerFace = cfxExportFaceNodes(faces[rr], pts);

                    if (remove_faces && node_x && node_y && node_z)
                    {
                        // compute normal ### assuming planar face
                        vec3 v1, v2, normal;
                        v1[0] = node_x[pts[1] - 1] - node_x[pts[0] - 1];
                        v1[1] = node_y[pts[1] - 1] - node_y[pts[0] - 1];
                        v1[2] = node_z[pts[1] - 1] - node_z[pts[0] - 1];

                        v2[0] = node_x[pts[nodesPerFace - 1] - 1] - node_x[pts[0] - 1];
                        v2[1] = node_y[pts[nodesPerFace - 1] - 1] - node_y[pts[0] - 1];
                        v2[2] = node_z[pts[nodesPerFace - 1] - 1] - node_z[pts[0] - 1];

                        vec3cross(v1, v2, normal);
                        vec3nrm(normal, normal);
                        vec3 facN;
                        vec3nrm(faceNormal, facN);

                        if (vec3dot(facN, normal) > 0.9)
                        {
                            // remove face
                            removedFaceCnt++;
                            continue;
                        }
                    }

                    *geomNodes = (int *)realloc(*geomNodes, (plistCnt + 1) * sizeof(int));
                    (*geomNodes)[plistCnt] = nodesPerFace;
                    plistCnt++;

                    // outputting node list (zero-based indices)
                    *geomNodes = (int *)realloc(*geomNodes, (plistCnt + nodesPerFace) * sizeof(int));
                    int k;
                    for (k = 0; k < nodesPerFace; k++)
                    {
                        (*geomNodes)[plistCnt] = pts[k] - 1;
                        plistCnt++;
                    }

                    effFaceCnt++;

                    // workaround
                    if (areaForWorkaround)
                    {

                        // compute area
                        if (nodesPerFace == 3)
                        {
                            vec3 v1, v2, cross;

                            v1[0] = node_x[pts[1] - 1] - node_x[pts[0] - 1];
                            v1[1] = node_y[pts[1] - 1] - node_y[pts[0] - 1];
                            v1[2] = node_z[pts[1] - 1] - node_z[pts[0] - 1];

                            v2[0] = node_x[pts[nodesPerFace - 1] - 1] - node_x[pts[0] - 1];
                            v2[1] = node_y[pts[nodesPerFace - 1] - 1] - node_y[pts[0] - 1];
                            v2[2] = node_z[pts[nodesPerFace - 1] - 1] - node_z[pts[0] - 1];

                            vec3cross(v1, v2, cross);
                            area += vec3mag(cross) * 0.5;
                        }
                        else if (nodesPerFace == 4)
                        {
                            vec3 v1, v2, cross;

                            // first tria
                            v1[0] = node_x[pts[1] - 1] - node_x[pts[0] - 1];
                            v1[1] = node_y[pts[1] - 1] - node_y[pts[0] - 1];
                            v1[2] = node_z[pts[1] - 1] - node_z[pts[0] - 1];
                            v2[0] = node_x[pts[nodesPerFace - 1] - 1] - node_x[pts[0] - 1];
                            v2[1] = node_y[pts[nodesPerFace - 1] - 1] - node_y[pts[0] - 1];
                            v2[2] = node_z[pts[nodesPerFace - 1] - 1] - node_z[pts[0] - 1];
                            vec3cross(v1, v2, cross);
                            area += vec3mag(cross) * 0.5;

                            // second tria
                            v1[0] = node_x[pts[1] - 1] - node_x[pts[2] - 1];
                            v1[1] = node_y[pts[1] - 1] - node_y[pts[2] - 1];
                            v1[2] = node_z[pts[1] - 1] - node_z[pts[2] - 1];
                            v2[0] = node_x[pts[nodesPerFace - 1] - 1] - node_x[pts[2] - 1];
                            v2[1] = node_y[pts[nodesPerFace - 1] - 1] - node_y[pts[2] - 1];
                            v2[2] = node_z[pts[nodesPerFace - 1] - 1] - node_z[pts[2] - 1];
                            vec3cross(v1, v2, cross);
                            area += vec3mag(cross) * 0.5;
                        }
                        else
                        {
                            printf("area computation for fake region labels: unsupported face type\n");
                        }

                        // compute centroid
                        // ### TODO: this is fake centroid (counting vertices multiply)
                        int k;
                        for (k = 0; k < nodesPerFace; k++)
                        {
                            centroid[0] += node_x[pts[k] - 1];
                            centroid[1] += node_y[pts[k] - 1];
                            centroid[2] += node_z[pts[k] - 1];
                        }
                        centroidCnt += nodesPerFace;
                    }
                }
                //geomNodeObjSizes[b-1] = bSiz;
                geomNodeObjSizes[*geomObjCnt + r - 1] = effFaceCnt;
                geomPListCounts[*geomObjCnt + r - 1] = plistCnt - plistLastCnt;
                plistLastCnt = plistCnt;

                // workaround
                if (areaForWorkaround)
                {
                    centroid[0] /= centroidCnt;
                    centroid[1] /= centroidCnt;
                    centroid[2] /= centroidCnt;
                }

#if 1 // workaround for generic labels from cfxExport
                if (areaForWorkaround)
                {
                    // based on area of regions
                    //sprintf(&geomObjNames[*geomObjCnt + r-1][0], "a%d_%s",
                    //	  (int) (area * 100000.0), cfxExportRegionName(r));

                    // based on area and centroid of regions
                    sprintf(&geomObjNames[*geomObjCnt + r - 1][0], "a%dx%dy%dz%d_%s",
                            (int)(area * 100000.0),
                            (int)(centroid[0] * 1000.0),
                            (int)(centroid[1] * 1000.0),
                            (int)(centroid[2] * 1000.0),
                            cfxExportRegionName(r));
                }
                else
                {
                    // classification by face count does not work (same meshing of different objects)
                    sprintf(&geomObjNames[*geomObjCnt + r - 1][0], "f%d_%s",
                            rSiz, cfxExportRegionName(r));
                }
#else
                strcpy(&geomObjNames[*geomObjCnt + r - 1][0], cfxExportRegionName(r));
#endif

                printf("region[%d]: %s contains %d faces\n",
                       r, &geomObjNames[*geomObjCnt + r - 1][0], rSiz);

                //printf("original face count = %d, remaining %d faces\n", rSiz, effFaceCnt);
            }

            *geomObjCnt += rCnt;
        }

        // terminate by zero
        *geomNodes = (int *)realloc(*geomNodes, (plistCnt + 1) * sizeof(int));
        (*geomNodes)[plistCnt] = 0;
        // not incrementing plistCnt

        *geomNodeCnt += plistCnt;
    }

    if (crop)
    {
        free(usedNodeIds);
    }

    if (exportDone)
    {
        cfxExportDone();
    }

    return 0; // success
}
