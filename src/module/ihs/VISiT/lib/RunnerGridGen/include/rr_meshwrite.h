#ifndef RR_MESHWRITE_INCLUDED
#define RR_MESHWRITE_INCLUDED

#ifdef FENFLOSS_OUT
extern int WriteFENFLOSS_Geofile(struct Nodelist *n, struct Element *e);
extern int WriteFENFLOSS_BCfile(struct Nodelist *n, struct Element *e,
struct Element *wall, struct Element *frictless,
struct Element *shroud,
struct Element *shroudext,
struct Element *psblade,
struct Element *ssblade,
struct Element *psleperiodic,
struct Element *ssleperiodic,
struct Element *psteperiodic,
struct Element *ssteperiodic,
struct Element *inlet, struct Element *outlet,
struct Element *rrinlet,
struct Element *rroutlet,
struct Ilist *innodes,
float **bcval, int rot_flag);
extern int WriteFENFLOSS62x_BCfile(struct Nodelist *n, struct Element *e,
struct Element *wall,
struct Element *frictless,
struct Element *shroud,
struct Element *shroudext,
struct Element *psblade,
struct Element *ssblade,
struct Element *psleperiodic,
struct Element *ssleperiodic,
struct Element *psteperiodic,
struct Element *ssteperiodic,
struct Element *inlet,
struct Element *outlet,
struct Element *rrinlet,
struct Element *rroutlet,
struct Ilist *innodes,
float **bcval, int rot_flag);

int CreateInletBCs(struct Ilist *inlet, struct Nodelist *n,struct bc *inbc,
float ***bcval, int alpha_const, int turb_prof);
#endif
#ifdef PATRAN_SES_OUT
extern int WritePATRAN_SESfile(int nnum, int elnum, int ge_num, int nstart,
int elstart, const char *efile, const char *nfile,
const char *egroup, const char *ngroup);
#endif
#endif                                            // RR_MESHWRITE_INCLUDED
