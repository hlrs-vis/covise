#ifndef  GEOMETRY_INCLUDED
#define  GEOMETRY_INCLUDED

#define  GT_TUBE     1
#define  GT_RRUNNER  2
#define  GT_DRUNNER  3
#define  GT_ARUNNER  4
#define  GT_GATE     5

#ifdef   DECLA
char *gt_type[] =
{
   "none",
   "tube",
   "radial runner",
   "diagonal runner",
   "axial runner",
   "gate"
};
#else
extern char *gt_type[];
#endif

struct geometry
{
   int type;
   struct tube *tu;
   struct axial *ar;
   struct radial *rr;
};

struct covise_info *CreateGeometry(char *fn);
struct covise_info *CreateGeometry4Covise(struct geometry *g);
struct geometry *ReadGeometry(const char *fn);
int WriteGeometry(struct geometry *g, const char *fn);
#endif
