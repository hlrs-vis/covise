#ifndef  GEOMETRY_INCLUDED
#define  GEOMETRY_INCLUDED

#define  GT_TUBE     1
#define  GT_RRUNNER  2
#define  GT_DRUNNER  3
#define  GT_ARUNNER  4
#define  GT_GATE     5

struct geometry
{
   int type;
   float minmax[2];
   struct tube *tu;
   struct axial *ar;
   struct radial *rr;
   struct gate *ga;
};

struct covise_info *CreateGeometry(char *fn);
struct covise_info *CreateGeometry4Covise(struct geometry *g);
struct geometry *ReadGeometry(const char *fn);
int WriteGeometry(struct geometry *g, const char *fn);
char *GT_Type(int ind);
int Num_GT_Type(void);
#endif
