/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* =====================================================================  */
/* =====================================================================  */
/* =====================================================================  */

FILE *myfopen(const char *file, const char *mode)
{
    char buf[800], *dirname, *covisepath;
    FILE *fp;
    int i;

    fp = fopen(file, mode);
    if (fp != NULL)
        return (fp);

    if ((covisepath = getenv("COVISE_PATH")) == NULL)
    {
        fprintf(stderr, "ERROR: COVISE_PATH not defined!\n");
        return (NULL);
    };

    dirname = strtok(strdup(covisepath), ":");
    while (dirname != NULL)
    {
        sprintf(buf, "%s/%s", dirname, file);
        fp = fopen(buf, mode);
        if (fp != NULL)
            return (fp);
        for (i = strlen(dirname) - 2; i > 0; i--)
        {
            if (dirname[i] == '/')
            {
                dirname[i] = '\0';
                break;
            }
        }
        sprintf(buf, "%s/%s", dirname, file);
        fp = fopen(buf, mode);
        if (fp != NULL)
            return (fp);
        dirname = strtok(NULL, ":");
    }
    return (NULL);
}

/* first some user specific defines and macros */

/* max lengths of some strings and main transfer data structures */

#define FALSE 0
#define TRUE 1

#define FSV_MAXSTRINGLEN 256
#define FSV_MAXTEXTLEN 256
#define FSV_MAXTYPLEN 20
#define MAXTIMESTEPS 5000
#define MAXQUANT 20

typedef char FSV_RESULT_STRING[FSV_MAXSTRINGLEN];

struct FSV_RESULT
{
    int server_ret;
    struct
    {
        unsigned int i_len;
        int *i_val;
    } i;
    struct
    {
        unsigned int f_len;
        float *f_val;
    } f;
    struct
    {
        unsigned int s_len;
        FSV_RESULT_STRING *s_val;
    } s;
};
typedef struct FSV_RESULT FSV_RESULT;

void ReadFIRE(char *dataset, char *dataname, float timestep, int scale,
              int firstStep,
              int *nSteps, float times[],
              int *nScalars, int *nVectors,
              int pScalar[], int pVector[][3],
              char scalarName[][FSV_MAXTYPLEN], char vectorName[][FSV_MAXTYPLEN],
              int *geo_change_flag, char *quantities);

FSV_RESULT *fire_call_slave(char *command_text);
FSV_RESULT *read_fire_timesteps(char *dataset);
FSV_RESULT *read_fire_geom(char *dataset, float timestep);
FSV_RESULT *read_fire_data(char *dataset, float timestep,
                           char *quantities);
void scan_timesteps(FSV_RESULT *result, int *nTimesteps);
void scan_dims(FSV_RESULT *result, int *nNodes, int *nBricks);
void write_coords(FSV_RESULT *result, FILE *geofile);
void write_links(FSV_RESULT *result, FILE *geofile, int **lcv, int **lcc);
void write_data(FSV_RESULT *result, int nNodes, int nBricks,
                int *lcv, int *lcc, int scale, char *dataname, int nSteps,
                int nScalars, int nVectors, int pScalar[], int pVector[][3],
                char scalarName[][FSV_MAXTYPLEN], char vectorName[][FSV_MAXTYPLEN]);
void fire_cells_to_nodes(int nNodes, int nBricks, int nCells, int nQuant,
                         int *lcv, float *cellVal, float *val, int *nVal, int *lcc, int scale);
int fsv_timeofstep(
    FSV_RESULT *result,
    float time_step,
    float *time);
int fsv_stepoftime(
    FSV_RESULT *result,
    float time,
    float *time_step);
int fsv_changeflags(
    FSV_RESULT *result,
    float time_step_1,
    float time_step_2,
    int *geo_changed,
    int *bnd_changed,
    int *rezone);
int decode_flags(
    int change_flags,
    int *geo,
    int *bnd,
    int *out,
    int *lnk,
    int *spray,
    int *comb,
    int *wallf);
int fsv_nextoutput(
    FSV_RESULT *result,
    float time_step,
    int *out_step);
int fsv_lastoutput(
    FSV_RESULT *result,
    float time_step,
    int *out_step);
int fsv_decodeflags(
    FSV_RESULT *result,
    float time_step,
    int *geo,
    int *bnd,
    int *out,
    int *lnk,
    int *spray,
    int *comb,
    int *wallf);
int fsv_printflags(FSV_RESULT *result);

/* server error codes */

#define FSV_OK 0
#define FSV_RPCERR 1
#define FSV_CANTOPENSERVER 2
#define FSV_BUSY 3
#define FSV_CANTFREEMEM 4
#define FSV_INVCMDMODE 5
#define FSV_INVCMDTEXT 6
#define FSV_INVPROB 7
#define FSV_INVTIMESTEP 8
#define FSV_OUTOFMEM 9
#define FSV_REQNOTREGISTERED 10
#define FSV_REQDELETED 11
#define FSV_RESNOTREADY 12
#define FSV_INVCELLSELECT 13
#define FSV_INVRESTYPE 14
#define FSV_TOOMANYRESTYPES 15
#define FSV_ERRINFIRE 16
#define FSV_GEONOTFOUND 17
#define FSV_LNKNOTFOUND 18
#define FSV_FLONOTFOUND 19
#define FSV_FILNOTFOUND 20
#define FSV_INVFIREVER 21
#define FSV_MAXTIMEXC 22
#define FSV_PIPEERROR 23

/*result handling macros*/

#define FSVret_ (result->server_ret)
#define FSVi_len_ (result->i.i_len)
#define FSVf_len_ (result->f.f_len)
#define FSVs_len_ (result->s.s_len)
#define FSVi_val_ (result->i.i_val)
#define FSVf_val_ (result->f.f_val)
#define FSVs_val_ (result->s.s_val)
#define FSVi_(j) (*(result->i.i_val + (j)))
#define FSVf_(j) (*(result->f.f_val + (j)))
#define FSVs_(j) (result->s.s_val + (j))

#define FSVret(result) ((result)->server_ret)
#define FSVi_len(result) ((result)->i.i_len)
#define FSVf_len(result) ((result)->f.f_len)
#define FSVs_len(result) ((result)->s.s_len)
#define FSVi_val(result) ((result)->i.i_val)
#define FSVf_val(result) ((result)->f.f_val)
#define FSVs_val(result) ((result)->s.s_val)
#define FSVi(j, result) (*((result)->i.i_val + (j)))
#define FSVf(j, result) (*((result)->f.f_val + (j)))
#define FSVs(j, result) ((result)->s.s_val + (j))

/* macro to free FSV_RESULT memory */

#define FSV_free               \
    if (result != NULL)        \
    {                          \
        if (FSVi_val_ != NULL) \
            free(FSVi_val_);   \
        if (FSVf_val_ != NULL) \
            free(FSVf_val_);   \
        if (FSVs_val_ != NULL) \
            free(FSVs_val_);   \
        free(result);          \
    }

/* macro to check whether memory allocation failed: */

#define CHECK_MEM(pointer)                  \
    if ((pointer) == NULL)                  \
    {                                       \
        printf("Can't allocate memory!\n"); \
        pclose(pipe_slave);                 \
        return NULL;                        \
    }

/* macro to handle pipe error */

#define PIPE_ERR                         \
    {                                    \
        printf("Error reading pipe!\n"); \
        pclose(pipe_slave);              \
        FSV_free return NULL;            \
    }

/* usage - error message */

#define ARG_ERR                                                                \
    {                                                                          \
        printf("usage:\n%s <dataset> <quantity1> [<quantity2> ...]", argv[0]); \
        printf(" <timestep1> <timestep2> <delta-timestep>\n");                 \
        printf("or:\n%s <dataset> <quantity1> [<quantity2> ...]", argv[0]);    \
        printf(" <timestep>\n");                                               \
        return;                                                                \
    }

/* file open error message */

#define FOPEN_ERR                                     \
    {                                                 \
        printf("\n can't open file %s!\n", filename); \
        return;                                       \
    }

/* cell properties */

#define NODES 8
#define NEIGHBORS 6
#define NODESPERFACE 4

int nelemg;
int nconng;
int ncoordg;
int *elg;
int *clg;
float *xcg;
float *ycg;
float *zcg;
float *ug;
float *vg;
float *wg;
float *scg;

/* main program */

void fire2covise(int *nelem, int *nconn, int *ncoord,
                 int **el, int **cl, float **xc, float **yc, float **zc,
                 float **u, float **v, float **w, float **sc,
                 char *fn, char *qu, int ts)
{
    int argc = 4;
    char *argv[4];
    char dataset[FSV_MAXSTRINGLEN], quantities[FSV_MAXSTRINGLEN],
        dataname[FSV_MAXSTRINGLEN], *slash,
        aux[FSV_MAXSTRINGLEN];
    float timestep1, timestep2, deltatimestep, timestep;
    int scale = FALSE, firstStep = TRUE;
    int nScalars = 0, nVectors = 0, geo_change_flag = FALSE,
        nSteps = 0;
    float times[MAXTIMESTEPS];
    char scalarName[MAXQUANT][FSV_MAXTYPLEN],
        vectorName[MAXQUANT][FSV_MAXTYPLEN];
    static int pScalar[MAXQUANT], pVector[MAXQUANT][3];
    int flo = FALSE, spray = FALSE;
    int nScalarsSpray = 0, nVectorsSpray = 0;
    char ctimestep[20];

    FILE *resfile;
    char filename[FSV_MAXSTRINGLEN], num[4];
    int i = 1, p = 0;

    argv[0] = "fire2covise";
    argv[1] = fn;
    argv[2] = qu;
    sprintf(ctimestep, "%d", ts);
    argv[3] = ctimestep;
    /* decode and check arguments */

    printf("argv[3] (time_step): %s <- %d\n", argv[3], ts);

    if (argc < 4)
        ARG_ERR

    /* scan dataset */

    if (sscanf(argv[i++], "%s", dataset) != 1)
        ARG_ERR

    /* scan quantities */

    quantities[0] = 0;
    while (i < argc && sscanf(argv[i], "%f", &timestep1) != 1 && sscanf(argv[i], "%s", aux) == 1)
    {
        if (!strncmp(argv[i], "drp_", 4) && !spray)
        {
            spray = TRUE;
            strcat(quantities, "drp_x drp_y drp_z ");
            pVector[nVectors][0] = p++;
            pVector[nVectors][1] = p++;
            pVector[nVectors][2] = p++;
            strcpy(vectorName[nVectors++], "drp_loc");
            nVectorsSpray++;
        }
        else
            flo = TRUE;
        if (!strcmp(argv[i], "vel"))
        {
            strcat(quantities, "u_vel v_vel w_vel ");
            pVector[nVectors][0] = p++;
            pVector[nVectors][1] = p++;
            pVector[nVectors][2] = p++;
            strcpy(vectorName[nVectors++], argv[i++]);
        }
        else if (!strcmp(argv[i], "sys_vel"))
        {
            strcat(quantities, "sys_u_vel sys_v_vel sys_w_vel ");
            pVector[nVectors][0] = p++;
            pVector[nVectors][1] = p++;
            pVector[nVectors][2] = p++;
            strcpy(vectorName[nVectors++], argv[i++]);
        }
        else if (!strcmp(argv[i], "p2_vel"))
        {
            strcat(quantities, "p2_u_vel p2_v_vel p2_w_vel ");
            pVector[nVectors][0] = p++;
            pVector[nVectors][1] = p++;
            pVector[nVectors][2] = p++;
            strcpy(vectorName[nVectors++], argv[i++]);
        }
        else if (!strcmp(argv[i], "drp_vel"))
        {
            strcat(quantities, "drp_u drp_v drp_w ");
            pVector[nVectors][0] = p++;
            pVector[nVectors][1] = p++;
            pVector[nVectors][2] = p++;
            strcpy(vectorName[nVectors++], argv[i++]);
            nVectorsSpray++;
        }
        else
        {
            if (!strncmp(argv[i], "drp_", 4))
                nScalarsSpray++;
            strcat(quantities, argv[i]);
            strcat(quantities, " ");
            pScalar[nScalars] = p++;
            strcpy(scalarName[nScalars++], argv[i++]);
        }
    }
    if (!(nScalars || nVectors))
        ARG_ERR

    /* scan timestep(s) */

    if (sscanf(argv[i++], "%f", &timestep1) != 1)
        ARG_ERR
    if (i < argc)
    {
        if (sscanf(argv[i++], "%f", &timestep2) != 1)
            ARG_ERR
        if (sscanf(argv[i++], "%f", &deltatimestep) != 1)
            ARG_ERR
    }
    else
    {
        timestep2 = timestep1;
        deltatimestep = 1.0;
    }

    /* get dataname (without path) */

    slash = strrchr(dataset, '/');
    if (slash != NULL)
        strcpy(dataname, ++slash);
    else
        strcpy(dataname, dataset);

    printf("\n\tImporting data files from\n\tFIRE .tim, .geo, .lnk"
           " and .flo files\n\tfor dataset %s\n\n",
           dataname);

    /* read geometry and result values for all specified time steps
and store values in files */

    for (timestep = timestep1; timestep <= timestep2;
         timestep += deltatimestep)
    {
        ReadFIRE(dataset, dataname, timestep, scale, firstStep, &nSteps,
                 times, &nScalars, &nVectors, pScalar, pVector,
                 scalarName, vectorName,
                 &geo_change_flag, quantities);
        if (nSteps)
            firstStep = FALSE;
    }

    *nelem = nelemg;
    *nconn = nconng;
    *ncoord = ncoordg;
    *el = elg;
    *cl = clg;
    *xc = xcg;
    *yc = ycg;
    *zc = zcg;

    return;

    /* write EnSight result file */

    if (flo)
    {

        sprintf(filename, "EnSight_%s.res", dataname);
        resfile = myfopen(filename, "w");
        if (resfile == NULL)
            FOPEN_ERR

        printf("Writing EnSight result file %s\n", filename);

        fprintf(resfile, "%d %d %d\n", nScalars - nScalarsSpray,
                nVectors - nVectorsSpray, geo_change_flag);
        fprintf(resfile, "%d\n", nSteps);
        for (i = 0; i < nSteps; i++)
            fprintf(resfile, "%f ", times[i]);
        fprintf(resfile, "\n");
        if (nSteps > 1)
            fprintf(resfile, "0 1\n");
        if (geo_change_flag)
            fprintf(resfile, "EnSight_%s_geom.****\n", dataname);
        if (nSteps > 1)
            strcpy(num, "****");
        else
            strcpy(num, "0000");
        for (i = 0; i < nScalars; i++)
            if (strncmp(scalarName[i], "drp_", 4))
                fprintf(resfile, "EnSight_%s_scl%02d.%s %s\n",
                        dataname, i, num, scalarName[i]);
        for (i = 0; i < nVectors; i++)
            if (strncmp(vectorName[i], "drp_", 4))
                fprintf(resfile, "EnSight_%s_vec%02d.%s %s\n",
                        dataname, i, num, vectorName[i]);
        fclose(resfile);
    }
    if (spray)
    {

        sprintf(filename, "EnSight_p_%s.res", dataname);
        resfile = myfopen(filename, "w");
        if (resfile == NULL)
            FOPEN_ERR

        printf("Writing EnSight particle result file %s\n", filename);

        fprintf(resfile, "%d %d %d\n", nScalarsSpray,
                nVectorsSpray - 1, 1);
        fprintf(resfile, "%d\n", nSteps);
        for (i = 0; i < nSteps; i++)
            fprintf(resfile, "%f ", times[i]);
        fprintf(resfile, "\n");
        if (nSteps > 1)
            fprintf(resfile, "0 1\n");
        fprintf(resfile, "EnSight_p_%s_geom.****\n", dataname);
        if (nSteps > 1)
            strcpy(num, "****");
        else
            strcpy(num, "0000");
        for (i = 0; i < nScalars; i++)
            if (!strncmp(scalarName[i], "drp_", 4))
                fprintf(resfile, "EnSight_%s_scl%02d.%s %s\n",
                        dataname, i, num, scalarName[i]);
        for (i = 0; i < nVectors; i++)
            if (!strncmp(vectorName[i], "drp_", 4) && strcmp(vectorName[i], "drp_loc"))
                fprintf(resfile, "EnSight_%s_vec%02d.%s %s\n",
                        dataname, i, num, vectorName[i]);
        fclose(resfile);
    }
}

/* module user function */

void ReadFIRE(
    char *dataset,
    char *dataname,
    float timestep,
    int scale,
    int firstStep,
    int *nSteps,
    float times[],
    int *nScalars,
    int *nVectors,
    int pScalar[],
    int pVector[][3],
    char scalarName[][FSV_MAXTYPLEN],
    char vectorName[][FSV_MAXTYPLEN],
    int *geo_change_flag,
    char *quantities)
{
    int i, j;
    static int *lcv, *lcc;

    float time;

    static int nTimesteps, nNodes, nBricks;
    static float timestepGeo;

    FSV_RESULT *result = NULL;
    static FSV_RESULT *resultTime = NULL;

    int newGeo, newBnd, rezone;

    char filename[FSV_MAXSTRINGLEN];
    FILE *geofile;

    /* check if dataset is entered */

    if (!dataset || dataset[0] == NULL)
    {
        printf("error: no dataset entered!\n");
        return;
    }

    /* read time steps and available quantities or check if timestep
valid and whether geometry changed */

    if (firstStep)
    {
        if (resultTime != NULL)
            free(resultTime);
        resultTime = read_fire_timesteps(dataset);
        if (resultTime == NULL)
            return;
        scan_timesteps(resultTime, &nTimesteps);
        newGeo = TRUE;
        /*fsv_printflags(resultTime);*/
    }
    else
    {
        if (fsv_changeflags(resultTime, timestep, timestepGeo,
                            &newGeo, &newBnd, &rezone) != FSV_OK)
        {
            printf("%d: Invalid timestep!\n", (int)timestep);
            return;
        }
        if (newGeo)
        {
            *geo_change_flag = TRUE;
            printf("Geometry changes between timesteps %d and %d\n",
                   (int)timestepGeo, (int)timestep);
        }
    }

    /* check timestep */

    if (fsv_timeofstep(resultTime, timestep, &time) != FSV_OK)
    {
        printf("%d: Invalid timestep!\n", (int)timestep);
        return;
    }
    else
        printf("Time[%d]: %f s\n", (int)timestep, time);

    /* read geometry data if necessary */

    if (newGeo)
    {
        printf("Reading the geometry of %s at %d\n", dataset,
               (int)timestep);

        /* read FIRE geometry information and store it in result */

        result = read_fire_geom(dataset, timestep);
        if (result == NULL)
            return;
        scan_dims(result, &nNodes, &nBricks);

        /* open EnSight geo file */

        sprintf(filename, "EnSight_%s_geom.%04d", dataname,
                *nSteps);
        geofile = myfopen(filename, "w");
        if (geofile == NULL)
            FOPEN_ERR

        /* take care of saving the coords info  */

        write_coords(result, geofile);

        /* take care of the connections list */

        write_links(result, geofile, &lcv, &lcc);

        /* free fire specific memory */

        FSV_free
            timestepGeo = timestep;
    }

    return;

    /* Data values */

    if (!quantities || quantities[0] == NULL)
        return;

    /* read FIRE data values */

    result = read_fire_data(dataset, timestep, quantities);
    if (result == NULL)
        return;
    write_data(result, nNodes, nBricks, lcv, lcc, scale,
               dataname, *nSteps, *nScalars, *nVectors, pScalar, pVector,
               scalarName, vectorName);
    times[(*nSteps)++] = time;

    /* free fire specific memory */

    FSV_free
}
/*ReadFIRE*/

FSV_RESULT *read_fire_timesteps(
    char *dataset)
{
    char command_text[FSV_MAXSTRINGLEN];

    /* set up command text */

    sprintf(command_text, "%s %s %f", "get_time_steps", dataset);

    /* call fire_server_slave */

    return fire_call_slave(command_text);
}

FSV_RESULT *read_fire_geom(
    char *dataset,
    float timestep)
{
    char command_text[FSV_MAXSTRINGLEN];

    /* set up command text */

    printf("Reading FIRE geometry...\n");
    sprintf(command_text, "%s %s %f", "get_geom6",
            dataset, timestep);

    /* call fire_server_slave */

    return fire_call_slave(command_text);
}

FSV_RESULT *read_fire_data(
    char *dataset,
    float timestep,
    char *quantities)
{
    char command_text[FSV_MAXSTRINGLEN];

    /* set up command text */

    printf("Reading FIRE data...\n");
    sprintf(command_text, "%s %s %f %s", "get_result",
            dataset, timestep, quantities);

    /* call fire_server_slave */

    return fire_call_slave(command_text);
}

void scan_timesteps(
    FSV_RESULT *result,
    int *nTimesteps)
{
    char *q;
    int p = 0, dummy;

    *nTimesteps = FSVi_len_;

    printf("Timesteps ranging from 0 to %d\n", *nTimesteps - 1);
}

void scan_dims(
    FSV_RESULT *result,
    int *nNodes,
    int *nBricks)
{
    *nNodes = FSVi_(0);
    *nBricks = FSVi_(1);

    printf("Number of nodes is %d, number of cells is %d\n",
           *nNodes, *nBricks);
}

void write_coords(
    FSV_RESULT *result,
    FILE *geofile)
{
    int i, n = FSVi_(0);
    float *x, *y, *z;

    printf("Writing coordinates...\n");
    /*
	fprintf(geofile, "coordinates id off\n%8d\n", n);
        for(x = FSVf_val_ + 1, y = x + n, z = y + n,
                i = 0; i < n; i++){
                fprintf(geofile, "%12.5e%12.5e%12.5e\n",
			*x++, *y++, *z++);
        }
*/
    ncoordg = n;
    xcg = (float *)malloc(sizeof(float) * n);
    ycg = (float *)malloc(sizeof(float) * n);
    zcg = (float *)malloc(sizeof(float) * n);

    for (x = FSVf_val_ + 1, y = x + n, z = y + n,
        i = n - 1;
         i >= 0; i--)
    {
        xcg[i] = *x++;
        ycg[i] = *y++;
        zcg[i] = *z++;
    }

    return;
}

void write_links(
    FSV_RESULT *result,
    FILE *geofile,
    int **lcv,
    int **lcc)
{

    int i, j, n = FSVi_(1);
    int *cv = FSVi_val_ + 3 + NODES;
    int *cc = FSVi_val_ + 3 + NODES * (FSVi_(1) + 1) + NEIGHBORS;
    int *c, *v;
    int ei, first_zero;

    nelemg = n;
    nconng = n * NODES;

    elg = (int *)malloc(sizeof(int) * nelemg);
    clg = (int *)malloc(sizeof(int) * nconng);

    printf("Writing link information...\n");

    for (ei = 0, i = 0; i < n; i++)
    {
        for (j = 0; j < NODES; j++)
        {
            clg[ei * NODES + j] = *cv++ - 1;
        }
        first_zero = 0;
        for (j = 0; j < NODES; j++)
        {
            if (xcg[clg[ei * NODES + j]] == 0.0 && ycg[clg[ei * NODES + j]] == 0.0 && zcg[clg[ei * NODES + j]] == 0.0)
                if (first_zero == 0)
                    first_zero = 1;
                else
                    break;
        }
        if (j == NODES)
        {
            ei++;
            elg[ei] = ei * NODES;
        }
        else
            nelemg--;

        /*
		for(j = 0; j < NEIGHBORS; j++)
			*c++ = *cc++ - 1;
*/
    }

    /*
	int	i, j, n = FSVi_(1);
	int	*cv = FSVi_val_ + 3 + NODES;
	int	*cc = FSVi_val_ + 3 + NODES * (FSVi_(1) + 1) + NEIGHBORS;
	int	*c, *v;

	printf("Writing link information...\n");

	fprintf(geofile, "part 1\nFlow Field\nhexa8 id off\n%8d\n", n);

	if(*lcv != NULL) free(*lcv);
	if(*lcc != NULL) free(*lcc);
	*lcv = (int *) calloc(n * NODES, sizeof(int));
	*lcc = (int *) calloc(n * NEIGHBORS, sizeof(int));
	
	for(v = *lcv, c = *lcc, i = 0; i < n; i++){
		for(j = 0; j < NODES; j++){
			fprintf(geofile, "%8d", *cv);
			*v++ = *cv++ - 1;
		}
		fprintf(geofile, "\n");
		for(j = 0; j < NEIGHBORS; j++)
			*c++ = *cc++ - 1;
	}
*/
}

void write_data(
    FSV_RESULT *result,
    int nNodes,
    int nBricks,
    int *lcv,
    int *lcc,
    int scale,
    char *dataname,
    int nSteps,
    int nScalars,
    int nVectors,
    int pScalar[],
    int pVector[][3],
    char scalarName[][FSV_MAXTYPLEN],
    char vectorName[][FSV_MAXTYPLEN])
{
    int i, j, k, n, nQuant, *nVal, nCells;
    float *val, *cellVal, *cval, *valp[3];

    char filename[FSV_MAXSTRINGLEN];
    FILE *varfile;

    /* number of quantities found */

    nQuant = FSVi_len_ / 2;
    printf("write_data: nNodes is %d\n", nNodes);
    printf("write_data: nBricks is %d\n", nBricks);
    printf("write_data: nQuant is %d\n", nQuant);

    /* auxiliary array for interpolation */

    nVal = (int *)calloc(nNodes, sizeof(int));
    if (nVal == NULL)
    {
        printf("Can't allocate memory for nVal!\n");
        return;
    }
    val = (float *)calloc(3 * nNodes, sizeof(float));
    if (val == NULL)
    {
        printf("Can't allocate memory for val!\n");
        return;
    }

    /* interpolation cells -> vertices for all quantities */

    printf("Writing data/Interpolating data values from cells to nodes...\n");

    /* Scalars: */

    for (i = 0; i < nScalars; i++)
    {

        printf("scalar %s:", scalarName[i]);

        /* interpolate values if not particle data */

        cellVal = FSVf_val_ + FSVi_(pScalar[i]);
        nCells = FSVi_(nQuant + pScalar[i]);

        if (!strncmp(scalarName[i], "drp_", 4))
        {
            printf("no interpolation for particle data\n");
        }
        else if (nCells > 1)
        {
            fire_cells_to_nodes(nNodes, nBricks, nCells, nQuant,
                                lcv, cellVal, val, nVal, lcc, scale);
        }
        else
        {
            printf("set to zero\n");
            for (j = 0; j < nNodes; j++)
                val[j] = 0.0;
        }

        /* open variable file */

        sprintf(filename, "EnSight_%s_scl%02d.%04d",
                dataname, i, nSteps);
        varfile = myfopen(filename, "w");
        if (varfile == NULL)
            FOPEN_ERR

        /* file header */

        fprintf(varfile,
                "EnSight scalar file for dataset %s, scalar %d\n",
                dataname, i);

        /* write values */

        if (strncmp(scalarName[i], "drp_", 4))
        {
            for (j = 0; j < nNodes;)
            {
                fprintf(varfile, "%12.5e", val[j]);
                if (!(++j % 6))
                    fprintf(varfile, "\n");
            }
        }
        else
        {
            for (j = 0; j < nCells;)
            {
                fprintf(varfile, "%12.5e", cellVal[j]);
                if (!(++j % 6))
                    fprintf(varfile, "\n");
            }
        }

        /* close file */

        fprintf(varfile, "\n");
        fclose(varfile);
    }
    printf("scalars done.\n");

    /* Vectors: */

    for (i = 0; i < nVectors; i++)
    {

        printf("vector %s:\n", vectorName[i]);

        /* loop over 3 vector components for interpolation/faking */

        for (cval = val, j = 0; j < 3; j++, cval += nNodes)
        {

            printf("\tcomponent %d:", j);

            /* if number of values = 0 fake zero values */
            /* else interpolate values if not particle data */

            cellVal = FSVf_val_ + FSVi_(pVector[i][j]);
            nCells = FSVi_(nQuant + pVector[i][j]);

            if (nCells == 0)
            {
                for (k = 0; k < nNodes; cval[k++] = 0.0)
                    ;
                printf("values set to 0.0\n");
            }
            else if (strncmp(vectorName[i], "drp_", 4))
            {
                fire_cells_to_nodes(nNodes, nBricks, nCells,
                                    nQuant, lcv, cellVal, cval, nVal,
                                    lcc, scale);
            }
            else
            {
                valp[j] = cellVal;
                printf("no interpolation for particle data\n");
                /*for(k = 0; k < nCells; k+=1000)
					printf("%d %f\n", k, cellVal[k]);*/
            }
        }

        /* open file */

        if (!strcmp(vectorName[i], "drp_loc"))
        {
            sprintf(filename, "EnSight_p_%s_geom.%04d",
                    dataname, nSteps);
        }
        else
        {
            sprintf(filename, "EnSight_%s_vec%02d.%04d",
                    dataname, i, nSteps);
        }

        varfile = myfopen(filename, "w");
        if (varfile == NULL)
            FOPEN_ERR

        /* file header */

        if (!strcmp(vectorName[i], "drp_loc"))
        {
            fprintf(varfile,
                    "EnSight particle geometry file for dataset %s\n",
                    dataname);
            fprintf(varfile, "particle coordinates\n%8d\n", nCells);
        }
        else
        {
            fprintf(varfile,
                    "EnSight vector file for dataset %s, vector %d\n",
                    dataname, i);
        }

        /* write variable file */

        if (!strcmp(vectorName[i], "drp_loc"))
        {
            for (j = 0; j < nCells; j++)
            {
                fprintf(varfile, "%8d", j + 1);
                /*if(!(j % 1000))
					printf("%d %f %f %f\n", j, *valp[0],
						*valp[1], *valp[2]);*/
                for (k = 0; k < 3; k++)
                {
                    fprintf(varfile, "%12.5e",
                            *(valp[k]++));
                }
                fprintf(varfile, "\n");
            }
        }
        else if (strncmp(vectorName[i], "drp_", 4))
        {
            for (j = 0, n = 0; j < nNodes; j++)
                for (k = 0; k < 3; k++)
                {
                    fprintf(varfile, "%12.5e",
                            val[j + k * nNodes]);
                    if (!(++n % 6))
                        fprintf(varfile, "\n");
                }
        }
        else
        {
            for (j = 0, n = 0; j < nCells; j++)
                for (k = 0; k < 3; k++)
                {
                    fprintf(varfile, "%12.5e",
                            *(valp[k]++));
                    if (!(++n % 6))
                        fprintf(varfile, "\n");
                }
        }

        /* close file */

        fprintf(varfile, "\n");
        fclose(varfile);
    }
    printf("vectors done.\n");

    /* free memory */

    free(val);
    free(nVal);

    /* return */

    return;
}

/* routine to call fire_server_slave process */

FSV_RESULT *fire_call_slave(
    char *command_text)
{
    FILE *pipe_slave;

    char cmd[FSV_MAXSTRINGLEN], c[10];
    int i;
    FSV_RESULT *result;

    /* complete command to call slave */

    sprintf(cmd, "/mnt/raid/cc/users/awi_te/covise/sgi/bin/IO/fire_server_slave %d %s",
            -getpid(), command_text);
    /*printf("starting slave:%s\n", cmd);*/

    printf("%s\n", cmd);
    /* call slave via pipe */

    if ((pipe_slave = popen(cmd, "r")) == NULL)
    {
        printf("Can't open pipe to slave via\n");
        printf(" %s\n", cmd);
        return NULL;
    }

    /* initialize result */

    result = (FSV_RESULT *)malloc(sizeof(FSV_RESULT));
    CHECK_MEM(result)
    FSVret_ = FSV_OK;
    FSVi_len_ = 0;
    FSVf_len_ = 0;
    FSVs_len_ = 0;
    FSVi_val_ = NULL;
    FSVf_val_ = NULL;
    FSVs_val_ = NULL;

    /* read slave pipe and allocate memory */

    printf("Waiting for slave...\n");

    if (fscanf(pipe_slave, ">%d,", &FSVret_) != 1)
        PIPE_ERR

    if (FSVret_ != FSV_OK && FSVret_ != FSV_ERRINFIRE)
    {
        FSV_free return NULL;
    }
    else if (fscanf(pipe_slave, "%d,%d,%d;BeginOfData>",
                    &FSVi_len_, &FSVf_len_, &FSVs_len_) != 3)
        PIPE_ERR

    printf("FSVret_ %d, FSVi_len_ %d, FSVf_len_ %d, FSVs_len_ %d\n",
           FSVret_, FSVi_len_, FSVf_len_, FSVs_len_);

    if (FSVi_len_ > 0)
    {
        FSVi_val_ = (int *)
            calloc(FSVi_len_, sizeof(int));
        CHECK_MEM(FSVi_val_)
    }

    if (FSVf_len_ > 0)
    {
        FSVf_val_ = (float *)
            calloc(FSVf_len_, sizeof(float));
        CHECK_MEM(FSVf_val_)
    }

    if (FSVs_len_ > 0)
    {
        FSVs_val_ = (FSV_RESULT_STRING *)
            calloc(FSVs_len_, sizeof(FSV_RESULT_STRING));
        CHECK_MEM(FSVs_val_)
    }

    if (fread((void *)FSVi_val_, sizeof(int), FSVi_len_,
              pipe_slave) != FSVi_len_)
        PIPE_ERR
    if (fread((void *)FSVf_val_, sizeof(float), FSVf_len_,
              pipe_slave) != FSVf_len_)
        PIPE_ERR
    if (fread((void *)FSVs_val_, sizeof(FSV_RESULT_STRING), FSVs_len_,
              pipe_slave) != FSVs_len_)
        PIPE_ERR

    /*
	printf("FSVi_len_:%d\n", FSVi_len_);
	for(i = 0; i < FSVi_len_; i+=200)
		printf("FSVi_(%d):%d\n", i, FSVi_(i));
	printf("FSVf_len_:%d\n", FSVf_len_);
	for(i = 0; i < FSVf_len_; i+=10)
		printf("FSVf_(%d):%f\n", i, FSVf_(i));
	printf("FSVs_len_:%d\n", FSVs_len_);
*/
    for (i = 0; i < FSVs_len_; i++)
        printf("FSVs_(%d):%s\n", i, FSVs_(i));

    pclose(pipe_slave);

    if (FSVret_ == FSV_ERRINFIRE)
    {
        printf("Error in FIRE-Slave:\nFIRE error:%s\nFIRE error:%s\n",
               FSVs_(0), FSVs_(1));
        FSV_free return NULL;
    }
    else
        printf("Data read successfully.\n");

    return result;

} /*fire_call_slave*/

/* routine to interpolate from cell to node values */
/* (node == vertex!) */

void
fire_cells_to_nodes(
    int nNodes,
    int nBricks,
    int nCells,
    int nQuant,
    int *lcv,
    float *cellVal,
    float *val,
    int *nVal,
    int *lcc,
    int scale)
{
    int vert_ext[NEIGHBORS][NODESPERFACE] = {
        0, 1, 5, 4,
        1, 2, 6, 5,
        2, 3, 7, 6,
        0, 4, 7, 3,
        0, 3, 2, 1,
        4, 5, 6, 7
    };

    int i, j, k, cell, vert, *cc, *cv;
    float *v;

    float min, max, div;

    if (nCells < nBricks)
        return;

    /* initilize Val and nVal */

    for (i = 0; i < nNodes; val[i] = 0.0, nVal[i++] = 0)
        ;

    /* add internal cell values */

    for (i = 0, v = cellVal, cv = lcv; i < nBricks; i++, v++)
    {
        for (j = 0; j < NODES; j++)
            if ((k = *cv++) >= 0)
            {
                val[k] += *v;
                nVal[k]++;
            }
    }

    /* add external cell values if available */

    if (nCells > nBricks)
        for (cv = lcv, cc = lcc,
            i = 0;
             i < nBricks; i++, cv += NODES)
        {
            /*printf("internal cell #%d:\n", i);*/
            for (j = 0; j < NEIGHBORS; j++)
            {
                cell = *cc++;
                if (cell >= nBricks)
                {
                    /*printf("add value of %d to vertices ",
                                               cell);*/
                    v = cellVal + cell;
                    for (k = 0; k < NODESPERFACE; k++)
                    {
                        vert = *(cv + vert_ext[j][k]);
                        /*printf("%d ", vert);*/
                        val[vert] += *v;
                        nVal[vert]++;
                    }
                    /*printf("\n");*/
                }
            }
        }

    /* divide by number of values for each node */

    min = 100000.0;
    max = -100000.0;
    for (i = j = 0; i < nNodes; i++)
    {
        if (nVal[i])
            val[i] /= nVal[i];
        if (val[i] > max)
            max = val[i];
        if (val[i] < min)
            min = val[i];
        /*printf("val[%d]=%f, n was %d\n", i, val[i], nVal[i]);*/
    }
    printf("min is %f, max is %f\n", min, max);

#define EPS 0.000001

    if (scale)
    {
        div = max - min;
        if (div < EPS)
            div = EPS;
        for (i = 0; i < nNodes; i++)
        {
            val[i] -= min;
            val[i] /= div;
        }
    }
}

/*
	FIRE SERVER auxiliary routines; Sampl, 8/93;
 */

int fsv_timeofstep(
    FSV_RESULT *result,
    float time_step,
    float *time)
{
    /* get time [s] of corresponding time_step */

    int n_time_steps = FSVi_len_;
    float *times = FSVf_val_;
    int i;

    if (time_step < 0)
        return FSV_INVTIMESTEP;
    if (time_step > n_time_steps)
        return FSV_INVTIMESTEP;
    i = time_step;
    *time = times[i] + (time_step - i) * (times[i] - times[i - 1]);
    return FSV_OK;
} /*fsv_timeofstep*/

int fsv_stepoftime(
    FSV_RESULT *result,
    float time,
    float *time_step)
{
    /* get corresponding time_step of time [s] */

    int n_time_steps = FSVi_len_;
    float *times = FSVf_val_;
    int i;

    for (i = 0; i < n_time_steps; i++)
        if (time < times[i])
        {
            if (i == 0)
                return FSV_INVTIMESTEP;
            *time_step = i - 1 + (time - times[i - 1]) / (times[i] - times[i - 1]);
            return FSV_OK;
        }
    return FSV_INVTIMESTEP;
} /*fsv_stepoftime*/

int fsv_changeflags(
    FSV_RESULT *result,
    float time_step_1,
    float time_step_2,
    int *geo_changed,
    int *bnd_changed,
    int *rezone)
{
    /* inquire changes of geometry between time steps 1 and 2 */

    int n_time_steps = FSVi_len_, *change_flags = FSVi_val_;
    int i1, i2, i, geo, bnd, out, rez, spray, comb, wallf;
    float tmp;

    *geo_changed = FALSE;
    *bnd_changed = FALSE;
    *rezone = FALSE;

    if (time_step_1 == time_step_2)
        return FSV_OK;
    else if (time_step_1 > time_step_2)
    {
        tmp = time_step_2;
        time_step_2 = time_step_1;
        time_step_1 = tmp;
    }

    if (time_step_1 < 0 || time_step_2 >= n_time_steps)
        return FSV_INVTIMESTEP;

    i1 = ++time_step_1;
    i2 = ++time_step_2;
    if (i2 >= n_time_steps)
        i2 = n_time_steps - 1;

    for (i = i1; i <= i2; i++)
    {
        decode_flags(change_flags[i], &geo, &bnd, &out, &rez,
                     &spray, &comb, &wallf);
        if (geo == TRUE)
            *geo_changed = TRUE;
        if (bnd == TRUE)
            *bnd_changed = TRUE;
        if (rez == TRUE)
            *rezone = TRUE;
    }
    if (*rezone == TRUE)
    {
        *geo_changed = TRUE;
        *bnd_changed = TRUE;
    }
    return FSV_OK;
} /*fsv_changeflags*/

int decode_flags(
    int change_flags,
    int *geo,
    int *bnd,
    int *out,
    int *lnk,
    int *spray,
    int *comb,
    int *wallf)
{
    *geo = change_flags % 2;
    change_flags = (change_flags - *geo) / 2;
    *bnd = change_flags % 2;
    change_flags = (change_flags - *bnd) / 2;
    *out = change_flags % 2;
    change_flags = (change_flags - *out) / 2;
    *lnk = change_flags % 2;
    change_flags = (change_flags - *lnk) / 2;
    *spray = change_flags % 2;
    change_flags = (change_flags - *spray) / 2;
    *comb = change_flags % 2;
    change_flags = (change_flags - *comb) / 2;
    *wallf = change_flags % 2;

} /*decode_flags*/

int fsv_nextoutput(
    FSV_RESULT *result,
    float time_step,
    int *out_step)
{
    /* get next output time step to time_step */

    int n_time_steps = FSVi_len_, *change_flags = FSVi_val_;
    int geo, bnd, out, lnk, spray, comb, wallf;

    if (time_step < 0 || time_step >= n_time_steps)
        return FSV_INVTIMESTEP;
    for (*out_step = ++time_step; *out_step < n_time_steps;
         (*out_step)++)
    {
        decode_flags(change_flags[*out_step],
                     &geo, &bnd, &out, &lnk,
                     &spray, &comb, &wallf);
        if (out == TRUE)
            return FSV_OK;
    }
    return FSV_INVTIMESTEP;
} /*fsv_nextoutput*/

int fsv_lastoutput(
    FSV_RESULT *result,
    float time_step,
    int *out_step)
{
    /* get last output time step previous to time_step */

    int n_time_steps = FSVi_len_, *change_flags = FSVi_val_;
    int geo, bnd, out, lnk, spray, comb, wallf;

    if (time_step <= 0 || time_step > n_time_steps)
        return FSV_INVTIMESTEP;
    for (*out_step = time_step; *out_step >= 0;
         (*out_step)--)
    {
        decode_flags(change_flags[*out_step],
                     &geo, &bnd, &out, &lnk,
                     &spray, &comb, &wallf);
        if (out == TRUE)
            return FSV_OK;
    }
    return FSV_INVTIMESTEP;
} /*fsv_lastoutput*/

int fsv_decodeflags(
    FSV_RESULT *result,
    float time_step,
    int *geo,
    int *bnd,
    int *out,
    int *lnk,
    int *spray,
    int *comb,
    int *wallf)
{
    /* get all flags for time_step */

    int n_time_steps = FSVi_len_, *change_flags = FSVi_val_;
    int i = time_step;

    if (time_step <= 0 || time_step > n_time_steps)
        return FSV_INVTIMESTEP;

    decode_flags(change_flags[i], geo, bnd, out, lnk,
                 spray, comb, wallf);
    return FSV_OK;
} /*fsv_decodeflags*/

int fsv_printflags(FSV_RESULT *result)
{
    /* inquire changes of geometry between time steps 1 and 2 */

    int n_time_steps = FSVi_len_, *change_flags = FSVi_val_;
    int i, geo, bnd, out, rez, spray, comb, wallf;

    printf("FIRE tim flags:\n");
    for (i = 0; i < n_time_steps; i++)
    {
        decode_flags(change_flags[i], &geo, &bnd, &out, &rez,
                     &spray, &comb, &wallf);
        if (geo || bnd || out || rez || spray || comb || wallf)
        {
            printf("%4d::", i);
            printf("geo:%2d, ", geo);
            printf("bnd:%2d, ", bnd);
            printf("out:%d, ", out);
            printf("rez:%d, ", rez);
            printf("spray:%d, ", spray);
            printf("comb:%d, ", comb);
            printf("wallf:%d\n", wallf);
        }
    }
    return FSV_OK;
} /*fsv_changeflags*/
