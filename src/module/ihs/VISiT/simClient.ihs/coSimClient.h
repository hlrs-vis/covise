#ifndef _CO_SIMLIB_H_
#define _CO_SIMLIB_H_

/**************************************************************************\ 
 **                                                           (C)1999 RUS  **
 **                                                                        **
 ** Description: Covise Simulation Library, Simulation side                **
 **                                                                        **
 **     When linking C++, C or FORTRAN simulations with Covise, this       **
 **     File describes the functionality available to the Simulation.      **
 **                                                                        **
 ** Author:                     Andreas Werner                             **
 **                Computing Center University of Stuttgart                **
 **                            Allmandring 30a                             **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  08.06.99  V1.0                                                  **
\**************************************************************************/

/* Set CO_SIMLIB_VERBOSE to one of the following to switch verbose levels:
 * 0: no messages (default)
 * 1: low message level
 * 2: high message level
 * 3: ultra-high message level, data verbose
 */

/* To be specified in covise.config:
 * <Modulname>
 * {
 *   PORTS    <first>-<last>     # allowed ports range,    default: 31000-31999
 *   SERVER   Module|Simulation  # who is TCP server,      default: Module
 *   LOCAL    <address>          # use this local address, default: gethostname()
 *   TIMEOUT  <seconds>          # timeout for network,    default: 60 sec
 *   STARTUP  string to start application, can use:
 *               %e           ->  replaced by contents of CO_SIMLIB_CONN
 *               %h           ->  replaced by target host, default localhost
 *               %1, %2, ...  ->  user-definable targets
 *               %%           ->  the '%' char
 * }
 * example:
 *
 * Star
 * {
 *    PORT      31000 31999
 *    SERVER    Module
 *    Startup   rsh -l %1 %h "env CO_SIMLIB_CONN=%e ; cd %2 ; echo %3 | star"
 * }
 */

#ifdef __cplusplus
extern "C"
{
#endif

   /* All commands return an exit code: =0 ok, =1 error ****/

   /* Startup : read connectivity parameters but do NOT connect
      return -1 on Error, 0 if ok */
   int coInitConnect();                           /* Fortran 77: COVINI() */

   /* Check, whether we are still connected: <0 if non- */
   int coNotConnected();                          /* Fortran 77: CONOCO() */

   /* Logic send/receive calls ******/

   /* Read a slider Parameter of the module */
   int coGetParaSlider(const char *name,  float *min, float *max, float *val);

   /* Read a scalar Parameter of the module */
                                                  /* Fortran 77: COGPFL() */
   int coGetParaFloatScalar(const char *name, float *data);

   /* Read a String Parameter of the module */
                                                  /* Fortran 77: COGPIN() */
   int coGetParaIntScalar(const char *name, int *data);

   /* Read a choice Parameter of the module */
                                                  /* Fortran 77: COGPCH() */
   int coGetParaChoice(const char *name, int *data);

   /* Read a Boolean Parameter of the module */
   int coGetParaBool(const char *name, int *data);/* Fortran 77: COGPBO() */

   /* Read a Boolean Parameter of the module */
                                                  /* Fortran 77: COGPTX() */
   int coGetParaText(const char *name, char *data);

   /* Read a Filename Parameter of the module */
   int coGetParaFile(const char *name, int *data);/* Fortran 77: COGPFI() */

   /* Send an Unstructured Grid, Covise format */
   int coSendUSG(const char *portName,
      int numElem, int numConn, int numCoord,     /* Fortran 77: COSUGC */
      int *elemList, int *connList,
      float *xCoord, float *yCoord,float *zCoord);

   /* Send a structured Grid, Covise format */
   int coSendUSG(const char *portName,
      int numElem, int numConn, int numCoord,     /* Fortran 77: COSUGC */
      int *elemList, int *connList,
      float *xCoord, float *yCoord,float *zCoord);

   /* Send an Unstructured Grid, xyz coordinate fields,
      8-elem conn List with multiple point for non-hex elements (e.g. STAR) */
   int coSendUSGhex(const char *portName,
      int numElem, int numCoord,                  /* Fortran 77: COSUSG */
      int *elemList, float *coord);

   /* Send an USG scalar data field */
   int coSend1Data(const char *portName,
      int numElem, float *data);                  /* Fortran 77: COSU1D */

   /* Send an USG vector data field */
   int coSend3Data(const char *portName,
      int numElem, float *data);                  /* Fortran 77: COSU3D */

   /* Attach attribute to object at port */
   int coAddAttribute(const char *portName,       /* Fortran 77: COATTR */
      const char *attrName,
      const char *attrVal);

   /* Execute the Covise Module now : required for continuously running simulations*/
   int coExecModule();                            /* Fortran 77: COEXEC */

   /* End Server in module and let pipeline run */
   int coFinished();                              /* Fortran 77: COFINI */

   /* Send a message (up to 64 char) to covise message output */

   /* ++++++++++++++++ Parallel Simulation support ++++++++++++++++ */

   /* Begin Parallel Ports definition                   F77: COPAIN */
   int coParallelInit(int numParts, int numPorts);

   /* Declare this port as parallel output port         F77: COPAPO */
   /* and tell whether this is cell- or vertex-based data           */
   int coParallelPort(const char *portname, int isCellData);

   /* Send a cell mapping local -> global               F77: COPACM */
   int coParallelCellMap(int node, int numCells, const int *localToGlobal);

   /* Send a vertex mapping local -> global             F77: COPAVM */
   int coParallelVertexMap(int node, int numCells, const int *localToGlobal);

   /* Next data sent is from node #                     F77: COPANO */
   /* WARNING: Covise always expects nodes 0..numParts-1            */
   int coParallelNode(int node);

   /******************************************************************
    *****                                                        *****
    ***** Binary send/receive: use this only if NOT using logics *****
    *****                                                        *****
    ******************************************************************/

   /* Send a certain amount of data to the module      F77: COSEND */
   int sendData(const void *buffer, size_t length, int swaptype);

   /* Receive a certain amount of data from the module F77: CORECV */
   int recvData(void *buffer, size_t length, int swaptype);

   /******************************************************************
    ******************************************************************/

   /* get the verbose level                            F77: COVERB */
   int getVerboseLevel();

#ifdef __cplusplus
}                                                 //    extern "C" {
#endif
#endif
