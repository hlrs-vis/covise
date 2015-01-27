/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/******************************************************************************
 *  This file was stolen from fastDNAml, too.                                 *
 ******************************************************************************/

#ifndef _APPL_FASTDNAML_TYPES_H
#define _APPL_FASTDNAML_TYPES_H

#define programName "fastDNAml"
#define programVersion "1.2.2"
#define programVersionInt 10202
#define programDate "January 3, 2000"
#define programDateInt 20000103

/*  Compile time switches for various updates to program:
 *    0 gives original version
 *    1 gives new version
 */

#define ReturnSmoothedView 1 /* Propagate changes back after smooth */
#define BestInsertAverage 1 /* Build three taxon tree analytically */
#define DeleteCheckpointFile 0 /* Remove checkpoint file when done */

#define Debug 0

/*----------------  Program constants and parameters  ------------------------*/

#define maxlogf 1024 /* maximum number of user trees */
#define maxcategories 35 /* maximum number of site types */

#define smoothings 32 /* maximum smoothing passes through tree */
#define iterations 10 /* maximum iterations of makenewz per insert */
#define newzpercycle 1 /* iterations of makenewz per tree traversal */
#define nmlngth 10 /* number of characters in species name */
#define deltaz 0.00001 /* test of net branch length change in update */
#define zmin 1.0E-15 /* max branch prop. to -log(zmin) (= 34) */
#define zmax (1.0 - 1.0E-6) /* min branch prop. to 1.0-zmax (= 1.0E-6) */
#define defaultz 0.9 /* value of z assigned as starting point */
#define unlikely -1.0E300 /* low likelihood for initialization */

/*  These values are used to rescale the lilelihoods at a given site so that
 *  there is no floating point underflow.
 */
#define twotothe256 115792089237316195423570985008687907853269984665640564039457584007913129639936.0
/*  2**256 (exactly)  */
#define minlikelihood (1.0 / twotothe256) /*  2**(-256)  */
#define log_minlikelihood (-177.445678223345993274) /* log(1.0/twotothe256) */

/*  The next two values are used for scaling the tree that is sketched in the
 *  output file.
 */
#define down 2
#define over 60

#define checkpointname "checkpoint"

#define badEval 1.0
#define badZ 0.0
#define badRear -1
#define badSigma -1.0

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#define treeNone 0
#define treeNewick 1
#define treeProlog 2
#define treePHYLIP 3
#define treeMaxType 3
#define treeDefType treePHYLIP

#define ABS(x) (((x) < 0) ? (-(x)) : (x))
#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif

#define LOG(x) (((x) > 0) ? log(x) : hang("log domain error"))
#define NINT(x) ((int)((x) > 0 ? ((x) + 0.5) : ((x)-0.5)))

/* Program types */
#define DNAML_MONITOR 0
#define DNAML_FOREMAN 1
#define DNAML_MASTER 2
#define DNAML_WORKER 3

/* Message types, Process states */
#define ANY_SOURCE -1
#define ANY_TAG -2
#define INVALID_ID -100

#define DNAML_WORK 1
#define DNAML_WORKER_READY 2
#define DNAML_RESULT 3
#define DNAML_DONE 4
#define DNAML_SPAWNED 5
#define DNAML_AWOL 6
#define DNAML_ADD_SPECS 7
#define DNAML_SEND_TREE 8
#define DNAML_RECV_TREE 9
#define DNAML_STEP_TIME 10
#define DNAML_NUM_TREE 11
#define DNAML_TID_LIST 12
#define DNAML_NOMSG 13
#define DNAML_SEQ_DATA 14
#define DNAML_SEQ_FILE 15
#define DNAML_QUIT 16
#define DNAML_INPUT_TIME 17
#define DNAML_SEQ_DATA_REQUEST 18
#define DNAML_SEQ_DATA_SIZE 19
#define DNAML_IDLING 20
#define DNAML_TASK_ADDED 21
#define DNAML_STATS_REQUEST 22
#define DNAML_STATS 23
#define DNAML_ADD_TASK 24
#define DNAML_KILL_TASK 25

/* Error types   */
#define ERR_SEQFILE 101
#define ERR_OUTFILE 102
#define ERR_LOGFILE 103
#define ERR_DEBUGFILE 104
#define ERR_BAD_MSG_TAG 105
#define ERR_NO_MASTER 106
#define ERR_NO_FOREMAN 107
#define ERR_NO_WORKERS 108
#define ERR_SEQDATA 109
#define ERR_TIMEOUT 110
#define ERR_BADTREE 111
#define ERR_BADEVAL 112
#define ERR_MALLOC 113
#define ERR_GENERIC 114

#define DNAML_STEP_TIME_COUNT 5
#define DNAML_CHAR_COUNT 50

/*
 * ------------------------------  Typedefs  -----------------------------------
 */

#ifndef Vectorize
typedef char appl_yType;
#else
typedef int appl_yType;
#endif

typedef int appl_boolean;
typedef double appl_xtype;

typedef struct appl_likelihood_vector
{
    appl_xtype a, c, g, t;
    long exp;
} appl_likelivector;

typedef struct appl_xmantyp
{
    struct appl_xmantyp *prev;
    struct appl_xmantyp *next;
    struct appl_noderec *owner;
    appl_likelivector *lv;
} appl_xarray;

typedef struct appl_noderec
{
    double z, z0;
    struct appl_noderec *next;
    struct appl_noderec *back;
    int number;
    appl_xarray *x;
    int xcoord, ycoord, ymin, ymax;
    char name[nmlngth + 1]; /*  Space for null termination  */
    appl_yType *tip; /*  Pointer to sequence data  */
} appl_node, *appl_nodeptr;

typedef struct
{
    int numsp; /* number of species (also tr->mxtips) */
    int sites; /* number of input sequence positions */
    appl_yType **y; /* sequence data array */
    appl_boolean freqread; /* user base frequencies have been read */
    /* To do: DNA specific values should get packaged into structure */
    double freqa, freqc, freqg, freqt, /* base frequencies */
        freqr, freqy, invfreqr, invfreqy,
        freqar, freqcy, freqgr, freqty;
    double ttratio, xi, xv, fracchange; /* transition/transversion */
    /* End of DNA specific values */
    int *wgt; /* weight per sequence pos */
    int *wgt2; /* weight per pos (booted) */
    int categs; /* number of rate categories */
    double catrat[maxcategories + 1]; /* rates per categories */
    int *sitecat; /* category per sequence pos */
} appl_rawdata;

typedef struct
{
    int *alias; /* site representing a pattern */
    int *aliaswgt; /* weight by pattern */
    int endsite; /* # of sequence patterns */
    int wgtsum; /* sum of weights of positions */
    int *patcat; /* category per pattern */
    double *patrat; /* rates per pattern */
    double *wr; /* weighted rate per pattern */
    double *wr2; /* weight*rate**2 per pattern */
} appl_cruncheddata;

typedef struct
{
    double likelihood;
    double *log_f; /* info for signif. of trees */
    appl_node **nodep;
    appl_node *start;
    appl_node *outgrnode;
    int mxtips;
    int ntips;
    int nextnode;
    int opt_level;
    int log_f_valid; /* log_f value sites */
    int global; /* branches to cross in full tree */
    int partswap; /* branches to cross in partial tree */
    int outgr; /* sequence number to use in rooting tree */
    appl_boolean prelabeled; /* the possible tip names are known */
    appl_boolean smoothed;
    appl_boolean rooted;
    appl_boolean userlen; /* use user-supplied branch lengths */
    appl_rawdata *rdta; /* raw data structure */
    appl_cruncheddata *cdta; /* crunched data structure */
} appl_tree;

typedef struct appl_conntyp
{
    double z; /* branch length */
    appl_node *p, *q; /* parent and child sectors */
    void *valptr; /* pointer to value of subtree */
    int descend; /* pointer to first connect of child */
    int sibling; /* next connect from same parent */
} appl_connect, *appl_connptr;

typedef struct
{
    double likelihood;
    double *log_f; /* info for signif. of trees */
    appl_connect *links; /* pointer to first connect (start) */
    appl_node *start;
    int nextlink; /* index of next available connect */
    /* tr->start = tpl->links->p */
    int ntips;
    int nextnode;
    int opt_level; /* degree of branch swapping explored */
    int scrNum; /* position in sorted list of scores */
    int tplNum; /* position in sorted list of trees */
    int log_f_valid; /* log_f value sites */
    appl_boolean prelabeled; /* the possible tip names are known */
    appl_boolean smoothed; /* branch optimization converged? */
} appl_topol;

typedef struct
{
    double best; /* highest score saved */
    double worst; /* lowest score saved */
    appl_topol *start; /* starting tree for optimization */
    appl_topol **byScore;
    appl_topol **byTopol;
    int nkeep; /* maximum topologies to save */
    int nvalid; /* number of topologies saved */
    int ninit; /* number of topologies initialized */
    int numtrees; /* number of alternatives tested */
    appl_boolean improved;
} appl_bestlist;

typedef struct
{
    long boot; /* bootstrap random number seed */
    int extra; /* extra output information switch */
    appl_boolean empf; /* use empirical base frequencies */
    appl_boolean interleaved; /* input data are in interleaved format */
    long jumble; /* jumble random number seed */
    int nkeep; /* number of best trees to keep */
    int numutrees; /* number of user trees to read */
    appl_boolean prdata; /* echo data to output stream */
    appl_boolean qadd; /* test addition without full smoothing */
    appl_boolean restart; /* resume addition to partial tree */
    appl_boolean root; /* use user-supplied outgroup */
    appl_boolean trprint; /* print tree to output stream */
    int trout; /* write tree to "treefile" */
    appl_boolean usertree; /* use user-supplied trees */
    appl_boolean userwgt; /* use user-supplied position weight mask */
} appl_analdef;

typedef struct
{
    double tipmax;
    int tipy;
} appl_drawdata;

typedef struct
{
    double tstart; /* process start time */
    double tinput;
    double t0;
    double t1;
    double utime; /* user time for process */
    double stime; /* system time for process */
    int ntrees; /* number of trees evaluated */
} appl_stat_data;

#define HOST_NAME_LEN 80
typedef struct appl_proc_d
{
    char hostname[HOST_NAME_LEN]; /* host name */
    int progtype; /* program type (master,foreman,worker) */
    int tid; /* MPI rank, PVM tid, etc. */
    int state; /* what state the process is in */
    double t0; /* process start time */
    appl_stat_data stats;
} appl_proc_data;

#endif /* FASTDNAML_TYPES_H */
