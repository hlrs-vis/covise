/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*RICARDO SQA =========================================================
 * Module Name   : RUproto
 * Subject       : Public prototypes for RUtil routines.
 * Language      : C
 * Requires      :
 * Documentation : RUtil program file.  Software.
 * Filename      : RUproto.h
 * Author        : F R Jeske
 * Creation Date : 23-Jun-97
 * Last Modified : $Date: 24-Mar-2001 13:07:47 $ by $Author: jwc $
 * Version       : $Revision: /main/30 $
 * Status        : Current
 * Modified      : Revision, date, author, reason.
 * See ClearCase history.
 *======================================================================
 */

#ifndef _RUPROTO_H
#define _RUPROTO_H

#if defined(WIN32) && defined(__cplusplus)
extern "C" {
#endif

/************/
/* rumath.h */
/************/

double RU_atan2(double, double);

/**************/
/* rumemory.c */
/**************/

#if defined(DEBUG)
void *RU_allocMemDebug(size_t, char *, char *, int);
#define RU_allocMem(s, n) RU_allocMemDebug(s, n, __FILE__, __LINE__)

void *RU_allocMemNoZeroDebug(size_t, char *, char *, int);
#define RU_allocMemNoZero(s, n) RU_allocMemNoZeroDebug(s, n, __FILE__, __LINE__)

void *RU_reallocMemDebug(void *, size_t, char *, int);
#define RU_reallocMem(s, n) RU_reallocMemDebug(s, n, __FILE__, __LINE__)

char *RU_strDupDebug(const char *, char *, int);
#define RU_strDup(s) RU_strDupDebug(s, __FILE__, __LINE__)

void *RU_allocBlockDebug(MemoryBlock *, size_t, char *, int);
#define RU_allocBlock(p, n) RU_allocBlockDebug(p, n, __FILE__, __LINE__)
#else
void *RU_allocMem(size_t, char *);
void *RU_allocMemNoZero(size_t, char *);
void *RU_reallocMem(void *, size_t);
char *RU_strDup(const char *);
void *RU_allocBlock(MemoryBlock *, size_t);
#endif
void RU_freeMem(void *);
void RU_initMemoryBlock(MemoryBlock *, size_t);
void RU_freeBlocks(MemoryBlock *);
char *RU_strDupBlock(MemoryBlock *, const char *);
double RU_totalMem(void);

/*************/
/* rulists.c */
/*************/

void RU_initList(List *);
void RU_insertAfter(Node *, Node *);
void RU_insertBefore(Node *, Node *);
void RU_addHead(List *, Node *);
Node *RU_removeHead(List *);
void RU_addTail(List *, Node *);
Node *RU_removeTail(List *);
void RU_removeNode(Node *);
int RU_isEmptyList(List *);
int RU_numNodes(List *);
Node *RU_accessNode(List *, int);

/***********/
/* rutil.c */
/***********/

void RU_initialize(int autoshutdown);
void RU_shutdown(void);

void RU_setError(int, char *, int, int);
int RU_getError(char *, int *);
int RU_getErrorCode(void);

int RU_isInt(char *);
int RU_isNum(char *);
void RU_sortData(float *, float *, int);
void RU_sortData3(float *, float *, float *, int);
void RU_findMaxMin(int, float *, float *, float *);
double RU_aToNum(char *);
int RU_isFile(char *);
int RU_isDir(char *);
time_t RU_getFileTimestamp(char *);
void RU_getTimeDate(char *, char *);

int RU_numDigits(float, int);
int RU_fontSize(char *, int);

char *RU_strLwr(char *);

size_t RU_lenTrim(const char *, size_t);
char *RU_strnCpy(char *, const char *, size_t);
char *RU_strc2fCpy(char *, const char *, size_t);
char *RU_strf2cCpy(char *, const char *, size_t);

/***************/
/* rucvtunit.c */
/***************/

int RU_cvtToSI(const char *, char *, double *, double *);
int RU_cvtUnit(const char *, const char *, double *, double *, double *);
int RU_getUnits(RU_Cpair **, int *);
int RU_parseUnit(const char *, char *);
int RU_setUnits(const char *);

void RU_freeUnits(void);
int RU_convertUnits(const char *, const char *, double *, double *);

/*************/
/* ruerror.c */
/*************/

void RU_errSet(const char *, int, int, const char *);
RU_Err *RU_errGet(const char *, int *);
int RU_errSev(const char *, int *);
void RU_errClr(const char *, int);
void RU_errDel(RU_Err *);
char *RU_errStr(const char *, const RU_Err *);
void RU_errOut(const char *, const char *, int, int *);
void RU_errEnq(const char *, int, int *, int *);
void RU_freeErr(void);

/*************/
/* rumatch.c */
/*************/

int RU_setPat(const char *, char *, int);
int RU_matchPat(const char *, const char *);
int RU_strMatch(const char *, const char *);

/*************/
/* rupopen.c */
/*************/

FILE *RU_popen(const char *, const char *);
int RU_pclose(FILE *);

/***************/
/* rufortran.c */
/***************/

#if defined(DEBUG)
char *RU_strf2cDupDebug(const char *str, size_t n, char *file, int line);
#define RU_strf2cDup(s, n) RU_strf2cDupDebug(s, n, __FILE__, __LINE__)
#else
char *RU_strf2cDup(const char *str, size_t n);
#endif

size_t RU_lenTrim(const char *str, size_t len);
char *RU_strnCpy(char *s1, const char *s2, size_t n);
char *RU_strc2fCpy(char *s1, const char *s2, size_t n);
char *RU_strf2cCpy(char *s1, const char *s2, size_t n);

/***********/
/* rusex.c */
/***********/

int RU_getSex(void);
int RU_oppositeSex(int sex);
void RU_changeSex(const void *in, void *out, int size, int many);

/**************/
/* rudecode.c */
/**************/

int RU_decodeString(const char *str,
                    RU_Item *items[], int maxitems, int maxlen);
void RU_setDecodeComments(const char *comments);

/****************/
/* rufilelist.c */
/****************/

int RU_fileList(const char *dirpath,
                const char *tmplate, char **files[]);
int RU_fileListR(const char *dirpath,
                 const char *tmplate, char **files[]);
int RU_dirList(const char *dirpath, const char *tmplate, char **dirs[]);

/*************/
/* ruspawn.c */
/*************/

int RU_spawn(const char *cmd, const char *args);

/************/
/* rupipe.c */
/************/

int RU_pipeCreate(const char *name, int *handle, char *path, int flags);
int RU_pipeClose(int handle);
int RU_pipeOpen(int *handle, const char *path, int flags);
int RU_pipeWrite(int handle, const void *buf, int count);
int RU_pipeRead(int handle, void *buf, int bufsize, int *bytesRead);

/***************/
/* ruversion.c */
/***************/

char *RU_version(void);

/**********/
/* others */
/**********/

#if defined(__sgi) || defined(WIN32)
int strncasecmp(const char *, const char *, size_t);
int strcasecmp(const char *, const char *);
#endif

#if defined(WIN32) && defined(__cplusplus)
}
#endif

#if defined(__hp9000s700) || defined(WIN32)
char *dirname(char *path);
char *basename(char *path);
#elif !defined(__alpha)
#include <libgen.h>
#endif
#endif /* _RUPROTO_H */
