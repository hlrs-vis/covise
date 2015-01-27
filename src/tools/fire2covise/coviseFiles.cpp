/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coviseFiles.h"
#include <string.h>

covOutFile::covOutFile(const char *filename)
    : d_str(filename)

{
    write("COV_LE", 6);
}

void covOutFile::writeSetHeader(int numElem)
{
    write("SETELE", 6);
    write((char *)&numElem, sizeof(int));
}

void covOutFile::writeattrib(const char **atNam,
                             const char **atVal)
{
    // count # attribs given and total space requirement
    int numAttr = 0;
    int size = sizeof(int);
    const char **nPtr = atNam, **vPtr = atVal;
    while (nPtr && *nPtr)
    {
        size += strlen(*nPtr) + strlen(*vPtr) + 2;
        nPtr++;
        vPtr++;
        numAttr++;
    }

    write((char *)&size, sizeof(int));
    write((char *)&numAttr, sizeof(int));

    int i;
    for (i = 0; i < numAttr; i++)
    {
        write((char *)atNam[i], strlen(atNam[i]) + 1);
        write((char *)atVal[i], strlen(atVal[i]) + 1);
    }
}

/****** int writeGeom()
{
   geo = (coDoGeometry *)data_obj;
   do1=geo->getGeometry();
   do2=geo->get_colors();
   do3=geo->get_normals();
   t1=geo->getGeometry_type();
   t2=geo->get_color_attr();
   t3=geo->get_normal_attr();
   write(fp,gtype,6);
   write(fp,&do1,sizeof(int));
write(fp,&do2,sizeof(int));
write(fp,&do3,sizeof(int));
write(fp,&t1,sizeof(int));
write(fp,&t2,sizeof(int));
write(fp,&t3,sizeof(int));
if(do1)
writeobj(do1);
if(do2)
writeobj(do2);
if(do3)
writeobj(do3);
writeattrib(data_obj);
delete geo;
} ******/

void covOutFile::writeUSG(int numElem, int numConn, int numVert,
                          const int *el, const int *cl, const int *tl,
                          const float *x, const float *y, const float *z,
                          const char **atNam, const char **atVal)
{
    write("UNSGRD", 6);
    write((char *)&numElem, sizeof(int));
    write((char *)&numConn, sizeof(int));
    write((char *)&numVert, sizeof(int));
    write((char *)el, numElem * sizeof(int));
    write((char *)tl, numElem * sizeof(int));
    write((char *)cl, numConn * sizeof(int));
    write((char *)x, numVert * sizeof(float));
    write((char *)y, numVert * sizeof(float));
    write((char *)z, numVert * sizeof(float));
    writeattrib(atNam, atVal);
}

/*
   else if(strcmp(gtype, "POINTS") == 0)
   {
       pts = (coDoPoints *)data_obj;
       pts->getAddresses(&x_coord,&y_coord,&z_coord);
       write(fp,gtype,6);
       n_elem=pts->getNumPoints();
       write(fp,&n_elem,sizeof(int));
       write(fp,x_coord,n_elem*sizeof(float));
       write(fp,y_coord,n_elem*sizeof(float));
       write(fp,z_coord,n_elem*sizeof(float));
writeattrib(data_obj);
delete pts;
}
else if(strcmp(gtype, "DOTEXT") == 0)
{
char *data;
txt = (coDoText *)data_obj;
txt->getAddress(&data);
write(fp,gtype,6);
n_elem=txt->getTextLength();
write(fp,&n_elem,sizeof(int));
write(fp,data,n_elem);
writeattrib(data_obj);
delete pts;
}
else if(strcmp(gtype, "POLYGN") == 0)
{
pol = (coDoPolygons *)data_obj;
pol->getAddresses(&x_coord,&y_coord,&z_coord, &vl,&el);
usg_h.n_elem=pol->getNumPolygons();
usg_h.n_conn=pol->getNumVertices();
usg_h.n_coord=pol->getNumPoints();
write(fp,gtype,6);
write(fp,&usg_h,sizeof(usg_h));
write(fp,el,usg_h.n_elem*sizeof(int));
write(fp,vl,usg_h.n_conn*sizeof(int));
write(fp,x_coord,usg_h.n_coord*sizeof(float));
write(fp,y_coord,usg_h.n_coord*sizeof(float));
write(fp,z_coord,usg_h.n_coord*sizeof(float));
writeattrib(data_obj);
delete pol;
}
else if(strcmp(gtype, "LINES") == 0)
{
lin = (coDoLines *)data_obj;
lin->getAddresses(&x_coord,&y_coord,&z_coord, &vl,&el);
usg_h.n_elem=lin->getNumLines();
usg_h.n_conn=lin->getNumVertices();
usg_h.n_coord=lin->getNumPoints();
write(fp,gtype,6);
write(fp,&usg_h,sizeof(usg_h));
write(fp,el,usg_h.n_elem*sizeof(int));
write(fp,vl,usg_h.n_conn*sizeof(int));
write(fp,x_coord,usg_h.n_coord*sizeof(float));
write(fp,y_coord,usg_h.n_coord*sizeof(float));
write(fp,z_coord,usg_h.n_coord*sizeof(float));
writeattrib(data_obj);
delete lin;
}
else if(strcmp(gtype, "TRIANG") == 0)
{
tri = (coDoTriangleStrips *)data_obj;
tri->getAddresses(&x_coord,&y_coord,&z_coord, &vl,&el);
usg_h.n_elem=tri->getNumStrips();
usg_h.n_conn=tri->getNumVertices();
usg_h.n_coord=tri->getNumPoints();
write(fp,gtype,6);
write(fp,&usg_h,sizeof(usg_h));
write(fp,el,usg_h.n_elem*sizeof(int));
write(fp,vl,usg_h.n_conn*sizeof(int));
write(fp,x_coord,usg_h.n_coord*sizeof(float));
write(fp,y_coord,usg_h.n_coord*sizeof(float));
write(fp,z_coord,usg_h.n_coord*sizeof(float));
writeattrib(data_obj);
delete tri;
}
else if(strcmp(gtype, "RCTGRD") == 0)
{
rgrid = (coDoRectilinearGrid *)data_obj;
rgrid->getAddresses(&x_coord,&y_coord,&z_coord);
rgrid->getGridSize(&(s_h.xs), &(s_h.ys), &(s_h.zs));
write(fp,gtype,6);
write(fp,&s_h,sizeof(s_h));
write(fp,x_coord,s_h.xs*sizeof(int));
write(fp,y_coord,s_h.ys*sizeof(int));
write(fp,z_coord,s_h.zs*sizeof(int));
writeattrib(data_obj);
delete rgrid;
}
else if(strcmp(gtype, "STRGRD") == 0)
{
sgrid = (coDoStructuredGrid *)data_obj;
sgrid->getAddresses(&x_coord,&y_coord,&z_coord);
sgrid->getGridSize(&(s_h.xs), &(s_h.ys), &(s_h.zs));
write(fp,gtype,6);
write(fp,&s_h,sizeof(s_h));
write(fp,x_coord,s_h.xs*s_h.ys*s_h.zs*sizeof(float));
write(fp,y_coord,s_h.xs*s_h.ys*s_h.zs*sizeof(float));
write(fp,z_coord,s_h.xs*s_h.ys*s_h.zs*sizeof(float));
writeattrib(data_obj);
delete sgrid;
}
else if(strcmp(gtype, "UNIGRD") == 0)
{
float x_min, y_min, z_min, x_max, y_max, z_max;

ugrid = (coDoUniformGrid *)data_obj;
ugrid->getGridSize(&(s_h.xs), &(s_h.ys), &(s_h.zs));
ugrid->getMinMax(&x_min, &x_max, &y_min, &y_max, &z_min, &z_max);

write(fp,gtype,6);
write(fp,&s_h,sizeof(s_h));

write(fp, &x_min, sizeof(float));
write(fp, &x_max, sizeof(float));

write(fp, &y_min, sizeof(float));
write(fp, &y_max, sizeof(float));

write(fp, &z_min, sizeof(float));
write(fp, &z_max, sizeof(float));

writeattrib(data_obj);
delete ugrid;
}

*/
void covOutFile::writeS3D(int numElem, const float *x,
                          const char **atNam, const char **atVal)
{
    write("USTSDT", 6);
    write(&numElem, sizeof(int));
    write(x, numElem * sizeof(float));
    writeattrib(atNam, atVal);
}

/*
   else if(strcmp(gtype, "RGBADT") == 0)
   {
       rgba = (coDoRGBA *)data_obj;
       rgba->getAddress((int **)(&z_coord));
       n_elem=rgba->getNumElements();
       write(fp,gtype,6);
       write(fp,&n_elem,sizeof(int));
       write(fp,z_coord,n_elem*sizeof(int));
       writeattrib(data_obj);
       delete rgba;
}
*/
void covOutFile::writeV3D(int numElem,
                          const float *x, const float *y, const float *z,
                          const char **atNam, const char **atVal)
{
    write("USTSDT", 6);
    write(&numElem, sizeof(int));
    write(x, numElem * sizeof(float));
    write(y, numElem * sizeof(float));
    write(z, numElem * sizeof(float));
    writeattrib(atNam, atVal);
}

/*
   else if(strcmp(gtype, "STRSDT") == 0)
   {
       s3d = (DO_Structured_S3D_Data *)data_obj;
       s3d->getGridSize(&(s_h.xs), &(s_h.ys), &(s_h.zs));
       s3d->getAddress(&z_coord);
       n_elem=s_h.xs*s_h.ys*s_h.zs;
       write(fp,gtype,6);
       write(fp,&s_h,sizeof(s_h));
       write(fp,z_coord,n_elem*sizeof(float));
       writeattrib(data_obj);
delete s3d;
}
else if(strcmp(gtype, "STRVDT") == 0)
{
s3dv = (DO_Structured_V3D_Data *)data_obj;
s3dv->getGridSize(&(s_h.xs), &(s_h.ys), &(s_h.zs));
s3dv->getAddresses(&x_coord,&y_coord,&z_coord);
n_elem=s_h.xs*s_h.ys*s_h.zs;
write(fp,gtype,6);
write(fp,&s_h,sizeof(s_h));
write(fp,x_coord,n_elem*sizeof(float));
write(fp,y_coord,n_elem*sizeof(float));
write(fp,z_coord,n_elem*sizeof(float));
writeattrib(data_obj);
delete s3dv;
}
else
{
Covise::sendError("ERROR: unsupported DataType");
return;
}
}
else
{
Covise::sendError("ERROR: object name not correct for 'mesh_in'");
return;
}
}

inline void swap_int(int &d)
{
unsigned int &data = (unsigned int &) d;
data =    ( (data & 0xff000000) >> 24 )
| ( (data & 0x00ff0000) >>  8 )
| ( (data & 0x0000ff00) <<  8 )
| ( (data & 0x000000ff) << 24 ) ;
}

inline void swap_int(int *d, int num)
{
unsigned int *data = (unsigned int *) d;
int i;
//fprintf(stderr,"swapping %d integers\n", num);
for (i=0;i<num;i++) {
//fprintf(stderr,"data=%d\n", *data);

*data =   ( ((*data) & 0xff000000) >> 24 )
| ( ((*data) & 0x00ff0000) >>  8 )
| ( ((*data) & 0x0000ff00) <<  8 )
| ( ((*data) & 0x000000ff) << 24 ) ;
//fprintf(stderr,"data=%d\n", *data);
data++;
}
}

// Not used
//inline void swap_float(float &d)
//{
//   unsigned int &data = (unsigned int &) d;
//   data =    ( (data & 0xff000000) >> 24 )
//           | ( (data & 0x00ff0000) >>  8 )
//           | ( (data & 0x0000ff00) <<  8 )
//           | ( (data & 0x000000ff) << 24 ) ;
//}

inline void swap_float(float *d, int num)
{
unsigned int *data = (unsigned int *) d;
int i;
for (i=0;i<num;i++) {
*data =   ( ((*data) & 0xff000000) >> 24 )
| ( ((*data) & 0x00ff0000) >>  8 )
| ( ((*data) & 0x0000ff00) <<  8 )
| ( ((*data) & 0x000000ff) << 24 ) ;
data++;
}
}

*/

/*
void Application::readattrib(coDistributedObject *tmp_Object)
{
    int numattrib=0, size=0, i;
    char *an, *at;
    char *buf;

    read(fp,&size,sizeof(int));
    if (byte_swap) swap_int(size);
    size-=sizeof(int);
    read(fp,&numattrib,sizeof(int));
if (byte_swap) swap_int(numattrib);
if(size>0)
{
buf=new char[size];
read(fp, buf, size);
an=buf;
for(i=0;i<numattrib;i++)
{
at=an;
while(*at)
at++;
at++;
tmp_Object->addAttribute(an, at);
an=at;
while(*an)
an++;
an++;
}
delete[] buf;
}
}

coDistributedObject *Application::readData(char *Name)
{
coDoSet*	    set;
coDoGeometry*    geo;
coDoRGBA*   rgba;
coDoLines*	    lin;
coDoPoints*      pts;
coDoText*      txt;
char buf[300],Data_Type[7];
USG_HEADER usg_h;
STR_HEADER s_h;
coDistributedObject **tmp_objs, *do1, *do2, *do3;
int numsets, i, t1, t2, t3;

read(fp,Data_Type,6);
Data_Type[6]='\0';

// MAGIC check

if (strcmp(Data_Type, "COV_BE") == 0) {   // The file is big-endian
byte_swap = 1 - byte_swap;
read(fp,Data_Type,6); // skip magic
}
else if (strcmp(Data_Type, "COV_LE") == 0)  // The file is big-endian
read(fp,Data_Type,6); // skip magic

if(Mesh != NULL)
{
if(strcmp(Data_Type, "SETELE") == 0)
{
read(fp,&numsets,sizeof(int));
if (byte_swap) swap_int(numsets);

tmp_objs=new coDistributedObject *[numsets+1];

for(i=0;i<numsets;i++)
{
sprintf(buf, "%s_%d", Name, i);
tmp_objs[i]=readData(buf);
}
tmp_objs[i]=NULL;
set = new coDoSet(Name, tmp_objs);
if (!(set->objectOk()))
{
Covise::sendError("ERROR: creation of SETELE object 'mesh' failed");
return(NULL);
}
for(i=0;i<numsets;i++)
{
delete tmp_objs[i];
}
delete[] tmp_objs;
readattrib(set);
return(set);
}

else if(strcmp(Data_Type, "GEOMET") == 0)
{
read(fp,&do1,sizeof(int));
read(fp,&do2,sizeof(int));
read(fp,&do3,sizeof(int));
read(fp,&t1,sizeof(int));  if (byte_swap) swap_int(t1);
read(fp,&t2,sizeof(int));  if (byte_swap) swap_int(t2);
read(fp,&t3,sizeof(int));  if (byte_swap) swap_int(t3);
if(do1)
{
sprintf(buf, "%s_Geo", Name);
do1=readData(buf);
}
if(do2)
{
sprintf(buf, "%s_Col", Name);
do2=readData(buf);
}
if(do3)
{
sprintf(buf, "%s_Norm", Name);
do3=readData(buf);
}

if(do1)
{
geo = new coDoGeometry(Name, do1);
if (!(geo->objectOk()))
{
Covise::sendError("ERROR: creation of GEOMET object 'mesh' failed");
return(NULL);
}
//geo->setGeometry(t1, do1);
}
if(do2)
{
geo->setColor(t2, do2);
}
if(do3)
{
geo->setNormal(t3, do3);
}
if(do1)
delete do1;
if(do2)
delete do2;
if(do3)
delete do3;

readattrib(geo);
return(geo);
}
else if(strcmp(Data_Type, "UNSGRD") == 0)
{

//fprintf(stderr,"found USG\n");
//if (byte_swap!=0)
//   fprintf(stderr,"need swap\n");

read(fp,&usg_h,sizeof(usg_h));
if (byte_swap) swap_int((int*)&usg_h,sizeof(usg_h)/sizeof(int));

//fprintf(stderr,"creating object with %d elements %d connections %d coordinates\n", usg_h.n_elem,usg_h.n_conn, usg_h.n_coord);
//fprintf(stderr, "creating covise usg...");

mesh = new coDoUnstructuredGrid(Name, usg_h.n_elem,usg_h.n_conn, usg_h.n_coord, 1);

//fprintf(stderr, "...done\n");

if (mesh->objectOk())
{
mesh->getAddresses(&el,&vl,&x_coord,&y_coord,&z_coord);
mesh->getTypeList(&tl);

read(fp,el,usg_h.n_elem*sizeof(int));
if (byte_swap) swap_int(el,usg_h.n_elem);

read(fp,tl,usg_h.n_elem*sizeof(int));
if (byte_swap) swap_int(tl,usg_h.n_elem);

read(fp,vl,usg_h.n_conn*sizeof(int));
if (byte_swap) swap_int(vl,usg_h.n_conn);

read(fp,x_coord,usg_h.n_coord*sizeof(float));
if (byte_swap) swap_float(x_coord,usg_h.n_coord);

read(fp,y_coord,usg_h.n_coord*sizeof(float));
if (byte_swap) swap_float(y_coord,usg_h.n_coord);

read(fp,z_coord,usg_h.n_coord*sizeof(float));
if (byte_swap) swap_float(z_coord,usg_h.n_coord);

readattrib(mesh);
return(mesh);
}
else
{
Covise::sendError("ERROR: creation of UNSGRD object 'mesh' failed");
return(NULL);
}
}
else if(strcmp(Data_Type, "POLYGN") == 0)
{
read(fp,&usg_h,sizeof(usg_h));
if (byte_swap) swap_int((int*)&usg_h,sizeof(usg_h)/sizeof(int));

pol = new coDoPolygons(Name, usg_h.n_coord,usg_h.n_conn , usg_h.n_elem);
if (pol->objectOk())
{
pol->getAddresses(&x_coord,&y_coord,&z_coord, &vl,&el);

read(fp,el,usg_h.n_elem*sizeof(int));
if (byte_swap) swap_int(el,usg_h.n_elem);

read(fp,vl,usg_h.n_conn*sizeof(int));
if (byte_swap) swap_int(vl,usg_h.n_conn);

read(fp,x_coord,usg_h.n_coord*sizeof(float));
if (byte_swap) swap_float(x_coord,usg_h.n_coord);

read(fp,y_coord,usg_h.n_coord*sizeof(float));
if (byte_swap) swap_float(y_coord,usg_h.n_coord);

read(fp,z_coord,usg_h.n_coord*sizeof(float));
if (byte_swap) swap_float(z_coord,usg_h.n_coord);

readattrib(pol);
return(pol);
}
else
{
Covise::sendError("ERROR: creation of POLYGN object 'mesh' failed");
return(NULL);
}
}

else if(strcmp(Data_Type, "POINTS") == 0)
{
read(fp,&n_elem,sizeof(int));
if (byte_swap) swap_int(n_elem);
pts = new coDoPoints(Name, n_elem);
if (pts->objectOk())
{
pts->getAddresses(&x_coord,&y_coord,&z_coord);
read(fp,x_coord,n_elem*sizeof(float));
if (byte_swap) swap_float(x_coord,usg_h.n_elem);

read(fp,y_coord,n_elem*sizeof(float));
if (byte_swap) swap_float(y_coord,usg_h.n_elem);

read(fp,z_coord,n_elem*sizeof(float));
if (byte_swap) swap_float(z_coord,usg_h.n_elem);

readattrib(pts);
return(pts);
}
else
{
Covise::sendError("ERROR: creation of data object 'mesh' failed");
return(NULL);
}
}

else if(strcmp(Data_Type, "DOTEXT") == 0)
{
char *data;
read(fp,&n_elem,sizeof(int));
if (byte_swap) swap_int(n_elem);
txt = new coDoText(Name, n_elem);
if (txt->objectOk())
{
txt->getAddress(&data);
read(fp,data,n_elem);

readattrib(txt);
return(txt);
}
else
{
Covise::sendError("ERROR: creation of data object 'mesh' failed");
return(NULL);
}
}

else if(strcmp(Data_Type, "LINES") == 0)
{
read(fp,&usg_h,sizeof(usg_h));
if (byte_swap) swap_int((int*)&usg_h,sizeof(usg_h)/sizeof(int));
lin = new coDoLines(Name, usg_h.n_coord,usg_h.n_conn , usg_h.n_elem);
if (lin->objectOk())
{
lin->getAddresses(&x_coord,&y_coord,&z_coord, &vl,&el);

read(fp,el,usg_h.n_elem*sizeof(int));
if (byte_swap) swap_int(el,usg_h.n_elem);

read(fp,vl,usg_h.n_conn*sizeof(int));
if (byte_swap) swap_int(vl,usg_h.n_conn);

read(fp,x_coord,usg_h.n_coord*sizeof(float));
if (byte_swap) swap_float(x_coord,usg_h.n_coord);

read(fp,y_coord,usg_h.n_coord*sizeof(float));
if (byte_swap) swap_float(y_coord,usg_h.n_coord);

read(fp,z_coord,usg_h.n_coord*sizeof(float));
if (byte_swap) swap_float(z_coord,usg_h.n_coord);

readattrib(lin);
return(lin);
}
else
{
Covise::sendError("ERROR: creation of data object 'mesh' failed");
return(NULL);
}
}
else if(strcmp(Data_Type, "TRIANG") == 0)
{
read(fp,&usg_h,sizeof(usg_h));
if (byte_swap) swap_int((int*)&usg_h,sizeof(usg_h)/sizeof(int));

tri = new coDoTriangleStrips(Name, usg_h.n_coord,usg_h.n_conn , usg_h.n_elem);
if (tri->objectOk())
{
tri->getAddresses(&x_coord,&y_coord,&z_coord, &vl,&el);
read(fp,el,usg_h.n_elem*sizeof(int));
if (byte_swap) swap_int(el,usg_h.n_elem);

read(fp,vl,usg_h.n_conn*sizeof(int));
if (byte_swap) swap_int(vl,usg_h.n_conn);

read(fp,x_coord,usg_h.n_coord*sizeof(float));
if (byte_swap) swap_float(x_coord,usg_h.n_coord);

read(fp,y_coord,usg_h.n_coord*sizeof(float));
if (byte_swap) swap_float(y_coord,usg_h.n_coord);

read(fp,z_coord,usg_h.n_coord*sizeof(float));
if (byte_swap) swap_float(z_coord,usg_h.n_coord);

readattrib(tri);
return(tri);
}
else
{
Covise::sendError("ERROR: creation of TRIANG object 'mesh' failed");
return(NULL);
}
}
else if(strcmp(Data_Type, "RCTGRD") == 0)
{
read(fp,&s_h,sizeof(s_h));
if (byte_swap) swap_int((int*)&s_h,sizeof(s_h)/sizeof(int));

rgrid = new coDoRectilinearGrid(Name,s_h.xs,s_h.ys, s_h.zs);
if (rgrid->objectOk())
{
rgrid->getAddresses(&x_coord,&y_coord,&z_coord);
read(fp,x_coord,s_h.xs*sizeof(float));
if (byte_swap) swap_float(x_coord,s_h.xs);

read(fp,y_coord,s_h.ys*sizeof(float));
if (byte_swap) swap_float(y_coord,s_h.ys);

read(fp,z_coord,s_h.zs*sizeof(float));
if (byte_swap) swap_float(z_coord,s_h.zs);

readattrib(rgrid);
return(rgrid);
}
else
{
Covise::sendError("ERROR: creation of RCTGRD object 'mesh' failed");
return(NULL);
}
}

else if(strcmp(Data_Type, "UNIGRD") == 0)
{
float x_min, y_min, z_min, x_max, y_max, z_max;

read(fp,&s_h,sizeof(s_h));
if (byte_swap) swap_int((int*)&s_h,sizeof(s_h)/sizeof(int));

read(fp, &x_min, sizeof(float));
if (byte_swap) swap_float(&x_min, 1);
read(fp, &x_max, sizeof(float));
if (byte_swap) swap_float(&x_max, 1);

read(fp, &y_min, sizeof(float));
if (byte_swap) swap_float(&y_min, 1);
read(fp, &y_max, sizeof(float));
if (byte_swap) swap_float(&y_max, 1);

read(fp, &z_min, sizeof(float));
if (byte_swap) swap_float(&z_min, 1);
read(fp, &z_max, sizeof(float));
if (byte_swap) swap_float(&z_max, 1);

ugrid = new coDoUniformGrid(Name,s_h.xs,s_h.ys, s_h.zs,
x_min, x_max, y_min, y_max, z_min, z_max);

if (ugrid->objectOk())
{
readattrib(ugrid);
return(ugrid);
}
else
{
Covise::sendError("ERROR: creation of UNIGRID object failed");
return(NULL);
}
}
else if(strcmp(Data_Type, "STRGRD") == 0)
{
read(fp,&s_h,sizeof(s_h));
if (byte_swap) swap_int((int*)&s_h,sizeof(s_h)/sizeof(int));

sgrid = new coDoStructuredGrid(Name, s_h.xs, s_h.ys, s_h.zs);
if (sgrid->objectOk())
{
sgrid->getAddresses(&x_coord,&y_coord,&z_coord);

read(fp,x_coord,s_h.xs*s_h.ys*s_h.zs*sizeof(float));
if (byte_swap) swap_float(x_coord,s_h.xs*s_h.ys*s_h.zs);

read(fp,y_coord,s_h.xs*s_h.ys*s_h.zs*sizeof(float));
if (byte_swap) swap_float(y_coord,s_h.xs*s_h.ys*s_h.zs);

read(fp,z_coord,s_h.xs*s_h.ys*s_h.zs*sizeof(float));
if (byte_swap) swap_float(z_coord,s_h.xs*s_h.ys*s_h.zs);

readattrib(sgrid);
return(sgrid);
}
else
{
Covise::sendError("ERROR: creation of STRGRD object 'mesh' failed");
return(NULL);
}
}

else if(strcmp(Data_Type, "USTSDT") == 0)
{
read(fp,&n_elem,sizeof(int));
if (byte_swap) swap_int(n_elem);
us3d = new coDoFloat(Name, n_elem);
if (us3d->objectOk())
{
us3d->getAddress(&x_coord);

read(fp,x_coord,n_elem*sizeof(float));
if (byte_swap) swap_float(x_coord,n_elem);

readattrib(us3d);
return(us3d);
}
else
{
Covise::sendError("ERROR: creation of USTSDT object 'mesh' failed");
return(NULL);
}
}

else if(strcmp(Data_Type, "RGBADT") == 0)
{
read(fp,&n_elem,sizeof(int));
if (byte_swap) swap_int(n_elem);

rgba = new coDoRGBA(Name, n_elem);
if (rgba->objectOk())
{
rgba->getAddress((int **)(&x_coord));
read(fp,x_coord,n_elem*sizeof(int));
if (byte_swap) swap_int((int*)x_coord,n_elem);

readattrib(rgba);
return(rgba);
}
else
{
Covise::sendError("ERROR: creation of RGBADT object 'mesh' failed");
return(NULL);
}
}
else if(strcmp(Data_Type, "USTVDT") == 0)
{
read(fp,&n_elem,sizeof(int));
if (byte_swap) swap_int(n_elem);

us3dv = new coDoVec3(Name, n_elem);
if (us3dv->objectOk())
{
us3dv->getAddresses(&x_coord,&y_coord,&z_coord);
read(fp,x_coord,n_elem*sizeof(float));
if (byte_swap) swap_float(x_coord,n_elem);

read(fp,y_coord,n_elem*sizeof(float));
if (byte_swap) swap_float(y_coord,n_elem);

read(fp,z_coord,n_elem*sizeof(float));
if (byte_swap) swap_float(z_coord,n_elem);

readattrib(us3dv);
return(us3dv);
}
else
{
Covise::sendError("ERROR: creation of USTVDT object 'mesh' failed");
return(NULL);
}
}

else if(strcmp(Data_Type, "STRSDT") == 0)
{
read(fp,&s_h,sizeof(s_h));
if (byte_swap) swap_int((int*)&s_h,sizeof(s_h)/sizeof(int));

n_elem=s_h.xs*s_h.ys*s_h.zs;
s3d = new DO_Structured_S3D_Data(Name,s_h.xs,s_h.ys,s_h.zs);
if (s3d->objectOk())
{
s3d->getAddress(&x_coord);
read(fp,x_coord,n_elem*sizeof(float));
if (byte_swap) swap_float(x_coord,n_elem);

readattrib(s3d);
return(s3d);
}
else
{
Covise::sendError("ERROR: creation of STRSDT object 'mesh' failed");
return(NULL);
}
}
else if(strcmp(Data_Type, "STRVDT") == 0)
{
read(fp,&s_h,sizeof(s_h));
if (byte_swap) swap_int((int*)&s_h,sizeof(s_h)/sizeof(int));

n_elem=s_h.xs*s_h.ys*s_h.zs;
s3dv = new DO_Structured_V3D_Data(Name,s_h.xs,s_h.ys,s_h.zs);

if (s3dv->objectOk())
{
s3dv->getAddresses(&x_coord,&y_coord,&z_coord);

read(fp,x_coord,n_elem*sizeof(float));
if (byte_swap) swap_float(x_coord,n_elem);

read(fp,y_coord,n_elem*sizeof(float));
if (byte_swap) swap_float(y_coord,n_elem);

read(fp,z_coord,n_elem*sizeof(float));
if (byte_swap) swap_float(z_coord,n_elem);

readattrib(s3dv);
return(s3dv);
}
else
{
Covise::sendError("ERROR: creation of STRVDT object 'mesh' failed");
return(NULL);
}
}
else
{
strcpy(buf, "ERROR: Reading file '");
strcat(buf, grid_Path);
strcat(buf, "', File does not seem to be in Covise Format");
Covise::sendError(buf);
return(NULL);
}
}
else
{
Covise::sendError("ERROR: object name not correct ");
return(NULL);
}
return(NULL);
}
*/
