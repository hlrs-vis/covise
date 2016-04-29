#ifndef RR_ELEMENT_INCLUDE
#define RR_ELEMENT_INCLUDE

int CreateRR_Elements(struct region **reg, struct Element *e, int *psblade,
int *ssblade, int *psleperiodic, int *ssleperiodic,
int *psteperiodic, int *ssteperiodic, int *inlet,
int *outlet, struct Element *ewall,
struct Element *pseblade, struct Element *sseblade,
struct Element *pseleperiodic,
struct Element *sseleperiodic,
struct Element *pseteperiodic,
struct Element *sseteperiodic, struct Element *einlet,
struct Element *eoutlet,  int reg_num, int offset);

int GetHubElements(struct Element *e, struct Element *wall, struct Element *frictionless,
struct Element *shroudext, struct Nodelist *n, int offset,
float linlet, float lhub);
int   GetShroudElements(struct Element *e, struct Element *shroud, struct Element *shroudext,
struct Nodelist *n, int offset, int be_num, float linlet, float lhub);
int GetRRIONodes(struct Element *e, struct Nodelist *n, struct ge **ge, int ge_num,
struct Element *in, struct Element *out, struct cgrid **cge, int npoin_ext);
///creates a part of all hub elements including the frictless part
int GetAllHubElements(struct Element *e, struct Element *hubAll, struct Nodelist *n, int offset);
///creates a part of all shroud elements
int GetAllShroudElements(struct Element *e, struct Element *shroudAll, struct Nodelist *n, int offset, int ge_num);
#ifdef DEBUG_ELEMENTS
extern int DumpElements(struct Nodelist *n, struct Element *e, int be_num);
#endif
#ifdef DEBUG_BC
int DumpBCElements(struct Element *bcelem, struct Nodelist *n, char *name);
#endif
#ifdef MESH_2DMERIDIAN_OUT
int Write_2DMeridianMesh(struct Nodelist *n, struct Element *e, int ge_num);
#endif
#endif                                            // RR_ELEMENT_INCLUDE
