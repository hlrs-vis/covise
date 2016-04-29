#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>



#if defined(__linux) || defined(__hpux) || defined(CO_t3e)
#  include <string.h>
#else
#  include <bstring.h>
#endif 

#if defined(CO_t3e)
#  include <fortran.h>
#endif

#include <sys/time.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#ifndef _SX
#  include <netinet/in.h>
#else
#  include <sys/socket.h>
#  include <sys/select.h>
#endif

#include <netdb.h>
#include <arpa/inet.h>
#include <errno.h>
#include <ctype.h>
#include "coSimClient.h"
#include "coSimLibComm.h"
#include <signal.h>
#include <assert.h>
#include <sys/fcntl.h>

/* SGI uses lowercase and trailing _ for FORTRAN */
#ifdef __sgi
#  define COVINI covini_
#  define COFINI cofini_
#  define CONOCO conoco_
#  define COGPSL cogpsl_
#  define COGPFL cogpfl_
#  define COGPIN cogpin_
#  define COGPTX cogptx_
#  define COGPFI cogpfi_
#  define COGPBO cogpbo_
#  define COGPCH cogpch_
#  define COSU1D cosu1d_
#  define COSU3D cosu3d_
#  define COEXEC coexec_
#  define COPAIN copain_
#  define COPAPO copapo_
#  define COPACM copacm_
#  define COPAVM copavm_
#  define COPANO copano_
#  define CORECV corecv_
#  define COSEND cosend_
#  define COVERB coverb_
#  define COATTR coattr_
#else
#  ifdef __hpux
#    define COVINI covini
#    define COFINI cofini
#    define CONOCO conoco
#    define COGPSL cogpsl
#    define COGPFL cogpfl
#    define COGPIN cogpin
#    define COGPTX cogptx
#    define COGPFI cogpfi
#    define COGPBO cogpbo
#    define COGPCH cogpch
#    define COSU1D cosu1d
#    define COSU3D cosu3d
#    define COEXEC coexec
#    define COPAIN copain
#    define COPAPO copapo
#    define COPACM copacm
#    define COPAVM copavm
#    define COPANO copano
#    define CORECV corecv
#    define COSEND cosend
#    define COVERB coverb
#    define COATTR coattr
#  endif
#endif

#ifdef CO_t3e
  typedef short int32;
#else
  typedef int int32;
#endif

/**** All commands return an exit code: =0 ok, =1 error ****/
/* Startup : read connectivity parameters but do NOT connect
   return -1 on Error, 0 if ok */


/************ LOCAL STATIC DATA ******************/

static struct
{
   int soc;
   int verbose;
}  
   coSimLibData = { 0,0 } ;

/************ Utilities ******************/

static int openServer(int minPort, int maxPort);
static int openClient(unsigned long ip, int port, float timeout);
static int acceptServer(float wait);

/************ ESTABLISH CONNECTION ******************/

int COVINI(void)
{ 
   return coInitConnect(); 
}

int coInitConnect()
{
   const char *envStr;
   char env[256],*portStr;
   int minPort,maxPort;
   struct in_addr ip;
   float timeout;
   
   coSimLibData.soc = -1;
   
   /* we do our own handling of broken pipes */
   signal(SIGPIPE,SIG_IGN);
  
   /* get environment: if variable not set, exit with error-code */
   envStr=getenv("CO_SIMLIB_CONN");
   if ((!envStr)||(strlen(envStr)>255)) 
      return -1;
   
   /* Client connection */
   strcpy(env,envStr);
   if (*env=='C')
   {
      /* get adress and port */
      portStr  = strchr(env,'/');
      *portStr = '\0';
      portStr++;
      sscanf(portStr,"%d,%f,%d",&minPort,&timeout,&coSimLibData.verbose);
      if ( minPort<1024 || minPort>32767 )  return -1;
      if (!inet_aton(env+2,&ip)) return -1;
      
      if (coSimLibData.verbose>0)
         fprintf(stderr," Starting Client to %s Port %d with %f sec timeout\n",
                         inet_ntoa(ip),minPort,timeout);
      
      /* we try to connect now */
      if (openClient(ip.s_addr,minPort,timeout))
         return -1;
   }
   
   /* Server connection */
   else if (*env=='S')
   {
      /* get adress and port */
      portStr  = strchr(env,'/');
      *portStr = '\0';
      portStr++;
      sscanf(portStr,"%d-%d,%f,%d",&minPort,&maxPort,&timeout,&coSimLibData.verbose);
      if (    minPort<1024 || minPort>32767
           || maxPort<1024 || maxPort>32767 )  return -1;
      if (!inet_aton(env+2,&ip)) return -1;

      if (coSimLibData.verbose>0)
         fprintf(stderr," Starting Server to %s Port %d-%d with %f sec timeout\n",
                         inet_ntoa(ip),minPort,maxPort,timeout);
      
      /* we open and wait for the other side to connect */
      if ( openServer(minPort,maxPort) || acceptServer(timeout) )
         return -1;
      
   }
   
   /* Neither Client nor Server = ERROR */
   else
      return -1;
   
   return 0;
}


int CONOCO() {  return coNotConnected(); }
int coNotConnected()
{
   /* try to send: if not possible, thenm it's not connected */
   int32 testdata=TEST;
   if (    coSimLibData.soc < 0
        || sendData((void*)&testdata,sizeof(int32)) != sizeof(int32) )
      return -1;
   else
      return 0;
}



/****** Logic send/receive calls ******/

/* Utilities for Parameter requests */
static int coSendFTN(int32 type,const char *name,int length)
{
   char buffer[64];
   int i;
   
   /* check length */
   if (length>63) return -1;
   
   
   /* Send request */
   if (sendData((void*)&type,sizeof(int32)) != sizeof(int32)) 
      return -1;
   
   /* Send name with fixed length of 64 bytes */
   if (length>63) 
      length=63;
   strncpy(buffer,name,length);
   buffer[length]='\0';

   /* remove everything after first blank and make sute the string is terminated */
   i=0;
   while ( i<63 && !isspace(buffer[i]) )
      i++;
   buffer[i] = '\0';

   if (sendData((void*)buffer,64) != 64) 
      return -1;
   else
      return 0;
}

/* Utilities for Parameter requests */
static int coSendC(int32 type, const char *name)
{
   char buffer[64];
   
   /* check length */
   int length=strlen(name);
   if (length>63) return -1;
   
   /* Send request */
   if (sendData((void*)&type,sizeof(int32)) != sizeof(int32)) 
      return -1;
   
   /* Send name with fixed length of 64 bytes */
   strcpy(buffer,name);
   if (sendData((void*)buffer,64) != 64) 
      return -1;
   else
      return 0;
}

/***********************************************************************/
/**************** Read a slider Parameter of the module ****************/

/* Common Slider code for C and Fortran */
static int coGetParaSli(float *min, float *max, float *val)       
{
   /* Receive result */
   struct { float min,max,val ; int32 error; } ret;
   if (   recvData((void*)&ret,sizeof(ret)) != sizeof(ret)
       || ret.error ) 
      return -1;
    
   *min = ret.min; 
   *max = ret.max;
   *val = ret.val;
      
   return 0;
}

/* Fortran API */
#ifdef CO_t3e
int COGPSL(_fcd name, float *min, float *max, float *val)
{
if (coSendFTN(GET_SLI_PARA,_fcdtocp(name),_fcdlen(name))) 
	 return -1;
   return coGetParaSli(min,max,val);
}
#else
int COGPSL(char *name, float *min, float *max, float *val, int length)
{
   if (coSendFTN(GET_SLI_PARA,name,length)) 
      return -1;
   return coGetParaSli(min,max,val);
}
#endif

/* C API */
int coGetParaSlider(const char *name,  float *min, float *max, float *val)       
{
   if (coSendC(GET_SLI_PARA,name)) 
      return -1;
   return coGetParaSli(min,max,val);
}   


/*******************************************************************************/
/****** Read a flaot scalar Parameter of the module  Fortran 77: COGPFL() ******/

/* Common Slider code for C and Fortran */
static int coGetParaScaFlo(float *val)       
{
   /* Receive result */
   struct { float val ; int32 error; } ret;
   if (   recvData((void*)&ret,sizeof(ret)) != sizeof(ret)
       || ret.error ) 
      return -1;
    
   *val = ret.val;
   return 0;
}
/* Fortran API */
#ifdef CO_t3e
int COGPFL(_fcd name, float *val)
{
   if (coSendFTN(GET_SC_PARA_FLO,_fcdtocp(name),_fcdlen(name)))
		 return -1;
   return coGetParaScaFlo(val);
}
#else
int COGPFL(char *name, float *val, int length)
{
   if (coSendFTN(GET_SC_PARA_FLO,name,length)) 
      return -1;
   return coGetParaScaFlo(val);
}
#endif


int coGetParaFloatScalar(const char *name, float *val)
{
   if (coSendC(GET_SC_PARA_FLO,name)) 
      return -1;
   return coGetParaScaFlo(val);
}

/*******************************************************************************/
/****** Read a int scalar Parameter of the module  Fortran 77: COGPIN() ******/

/* Common Slider code for C and Fortran */
static int coGetParaScaInt(int *val)   /* Receive result */
{
   struct { int32 val ; int32 error; } ret;
   if (   recvData((void*)&ret,sizeof(ret)) != sizeof(ret)
       || ret.error ) 
      return -1;
    
   *val = ret.val;
   return 0;
}
/* Fortran API */

#ifdef CO_t3e
int COGPIN(_fcd name,int *val)
{
   if (coSendFTN(GET_SC_PARA_INT,_fcdtocp(name),_fcdlen(name)))
		 return -1;
   return coGetParaScaInt(val);

}
#else
int COGPIN(char *name, int *val, int length)
{
   if (coSendFTN(GET_SC_PARA_INT,name,length)) 
      return -1;
   return coGetParaScaInt(val);
}
#endif

int coGetParaIntScalar(const char *name, int *val)
{
   if (coSendC(GET_SC_PARA_INT,name)) 
      return -1;
   return coGetParaScaInt(val);
}

/*******************************************************************************/
/****** Read a choice Parameter of the module  Fortran 77: COGPCH ******/

/* Common Slider code for C and Fortran */
static int coGetParaCh(int *val)   /* Receive result */
{
   struct { int32 val ; int32 error; } ret;
   if (   recvData((void*)&ret,sizeof(ret)) != sizeof(ret)
       || ret.error ) 
      return -1;
    
   *val = ret.val;
   return 0;
}

/* Fortran API */
#ifdef CO_t3e
int COGPCH(_fcd name,int *val)
{
   if (coSendFTN(GET_CHOICE_PARA,_fcdtocp(name),_fcdlen(name)))
		 return -1;
   return coGetParaCh(val);
}
#else
int COGPCH(char *name, int *val, int length)
{
   if (coSendFTN(GET_CHOICE_PARA,name,length)) 
      return -1;
   return coGetParaCh(val);
}
#endif

/* C API */
int coGetParaChoice(const char *name, int *val)
{
   if (coSendC(GET_CHOICE_PARA,name))
      return -1;
   return coGetParaCh(val);
}

/*******************************************************************************/
/****** Read a boolean Parameter of the module      Fortran 77: COGPBO    ******/

/* Common Slider code for C and Fortran */
static int coGetParaBo(int *val)   /* Receive result */
{
   struct { int32 val ; int32 error; } ret;
   if (   recvData((void*)&ret,sizeof(ret)) != sizeof(ret)
       || ret.error ) 
      return -1;
    
   *val = ret.val;
   return 0;
}

/* Fortran API */
#ifdef CO_t3e
int COGPBO(_fcd name, int *val)
{
 if (coSendFTN(GET_BOOL_PARA,_fcdtocp(name),_fcdlen(name)))
	   return -1;
  return coGetParaBo(val);
}

#else
int COGPBO(char *name, int *val, int length)
{
   if (coSendFTN(GET_BOOL_PARA,name,length)) 
      return -1;
   return coGetParaBo(val);
}
#endif

/* C API */
int coGetParaBool(const char *name, int *val)
{
   if (coSendC(GET_BOOL_PARA,name))
      return -1;
   return coGetParaBo(val);
}

/*******************************************************************************/
/****** Read a Text Parameter of the module         Fortran 77: COGPTX    ******/

/* Read a Text Parameter of the module */
int coGetParaText(const char *name, char *data)
{
   if (coSendC(GET_TEXT_PARA,name))
      return -1;;
   if (   recvData((void*)&data,256) != 256 ) 
      return -1;
   return 0;
}

#ifdef CO_t3e
int COGPTX(_fcd name,_fcd strdata)
{
 char buffer[256],*data;
 int i;
 data = _fcdtocp(strdata) ;
 if (coSendFTN(GET_TEXT_PARA,_fcdtocp(name),_fcdlen(name)))
   return -1;

 if (   recvData((void*)&buffer,256) != 256 )
	return -1;

 strcpy(data,buffer);
 for (i=strlen(buffer);i<256;i++)     /* FORTRAN is blank padded  */
  data[i]=' '; 
 return 0;
}
#else
int COGPTX(const char *name, char *data, int lenNane, int lenData)
{
   char buffer[256];
   int i;
   if (coSendFTN(GET_TEXT_PARA,name,lenNane)) 
      return -1;
   if (   recvData((void*)&buffer,256) != 256 ) 
      return -1;
   strcpy(data,buffer);
   for (i=strlen(buffer);i<256;i++)     /* FORTRAN is blank padded */
      data[i]=' ';
   return 0;
}
#endif

/*******************************************************************************/

/* Read a Filename Parameter of the module */
int coGetParaFile(const char *name, int *data)            /* Fortran 77: COGPFI() */
{
   if (coSendC(GET_FILE_PARA,name))
      return -1;;
   if (   recvData((void*)&data,256) != 256 ) 
      return -1;
   return 0;
}

#ifdef CO_t3e
int COGOFI(_fcd name,_fcd strdata)
{
 char buffer[256],*data;
 int i;
 data = _fcdtocp(strdata) ;
 if (coSendFTN(GET_FILE_PARA,_fcdtocp(name),_fcdlen(name)))
   return -1;

 if (   recvData((void*)&buffer,256) != 256 )
	return -1;

 strcpy(data,buffer);
 for (i=strlen(buffer);i<256;i++)     /* FORTRAN is blank padded  */
  data[i]=' '; 
 return 0;
}
#else
int COGPFI(const char *name, char *data, int lenNane, int lenData)
{
   char buffer[256];
   int i;
   if (coSendFTN(GET_FILE_PARA,name,lenNane)) 
      return -1;
   if (   recvData((void*)&buffer,256) != 256 ) 
      return -1;
   strcpy(data,buffer);
   for (i=strlen(buffer);i<256;i++)     /* FORTRAN is blank padded */
      data[i]=' ';
   return 0;
}
#endif

/* Send an Unstructured Grid, Covise format */
int coSendUSGcov(const char *portName,
                 int numElem, int numConn, int numCoord, /* Fortran 77: COSUGC */
                 int *elemList, int *connList, 
                 float *xCoord, float *yCoord,float *zCoord)
{
   return 0;
}


/* Send an Unstructured Grid, xyz coordinate fields, 
   8-elem conn List with multiple point for non-hex elements (e.g. STAR) */
int coSendUSGhex(const char *portName,
                 int numElem, int numCoord,                /* Fortran 77: COSUSG */
                 int *elemList, float *coord)
{
   return 0;
}

/********* Send an USG vector data field    Fortran 77: COSU3D ********/
int coSend1DataCommon(int numElem, float *data)
{
   int32 num = numElem;
   if ( sendData((void*)&num,sizeof(int32)) != sizeof(int32) )
      return -1;
   if ( sendData((void*)data,numElem*sizeof(float)),numElem*sizeof(float) 
                                                        != numElem*sizeof(float))
      return -1;
   return 0;
}
#ifdef CO_t3e
int COSU1D(_fcd portName, int *numElem, float *data)
{
   if (coSendFTN(SEND_1DATA,_fcdtocp(portName),_fcdlen(portName)))
	  return -1;
   return coSend1DataCommon(*numElem,data);
}
#else
int COSU1D(const char *portName, int *numElem, float *data, int length)
{
   if (coSendFTN(SEND_1DATA,portName,length)) 
      return -1;
   return coSend1DataCommon(*numElem,data);
}
#endif
int coSend1Data(const char *portName, int numElem, float *data)
{
   if (coSendC(SEND_1DATA,portName)) 
      return -1;
   return coSend1DataCommon(numElem,data);
}



/********* Send an USG vector data field    Fortran 77: COSU3D ********/
int coSend3DataCommon(int numElem, float *data0, float *data1, float *data2)
{
   int32 num = numElem;
   if ( sendData((void*)&num,sizeof(int32)) != sizeof(int32) )
      return -1;
   if ( sendData((void*)data0,numElem*sizeof(float)) != numElem*sizeof(float) )
      return -1;
   if ( sendData((void*)data1,numElem*sizeof(float)) != numElem*sizeof(float) )
      return -1;
   if ( sendData((void*)data2,numElem*sizeof(float)) != numElem*sizeof(float) )
      return -1;
   return 0;
}
#ifdef CO_t3e
int COSU3D(_fcd portName,int *numElem, float *data0, float *data1, float *data2)
{
   if (coSendFTN(SEND_3DATA,_fcdtocp(portName),_fcdlen(portName)))
		 return -1;
	return coSend3DataCommon(*numElem,data0,data1,data2);
}
#else
int COSU3D(const char *portName, int *numElem, float *data0, float *data1, float *data2, int length)
{
   if (coSendFTN(SEND_3DATA,portName,length)) 
      return -1;
   return coSend3DataCommon(*numElem,data0,data1,data2);
}
#endif

int coSend3Data(const char *portName, int numElem, float *data)
{
   if (coSendC(SEND_3DATA,portName)) 
      return -1;
   /* return coSend3DataCommon(numElem,data,);@@@@@@@@@@@@@@@@@@@@@@@@@@@@ */
   return 0;
}

/* End Server in module and let pipeline run */
int COFINI() { return coFinished(); }
int coFinished()                                          /* Fortran 77: COWAIT */
{
   int32 testdata=QUIT;
   if (    coSimLibData.soc < 0
        || sendData((void*)&testdata,sizeof(int32)) != sizeof(int32) )
      return -1;
   else
      return 0;
}

/* Execute the Covise Module now :  Fortran 77: COEXEC */
int COEXEC() { return coExecModule(); }
int coExecModule()
{
   int32 execCommand=EXEC_COVISE;
   if (sendData((void*)&execCommand,sizeof(int32)) != sizeof(int32) )
      return -1;
   else
      return 0;
}                                        



/******************************************************************
 ******************************************************************

      PARALLEL SIMULATION SUPPORT

 ******************************************************************
 ******************************************************************/ 

/* --------------------------------------------------------------*/
/* Begin Parallel Ports definition                   F77: COPAIN */
int COPAIN(const int *numParts, const int *numPorts)  
{ return coParallelInit(*numParts,*numPorts); }

int coParallelInit(int numParts, int numPorts)
{
   struct { int32 type,parts,ports; } data;
   data.type  = PARA_INIT;
   data.parts = numParts;
   data.ports = numPorts;
   if (sendData((void*)&data,sizeof(data)) != sizeof(data) )
      return -1;
   else
      return 0;
}

/* --------------------------------------------------------------*/
/* Declare this port as parallel output port         F77: COPAPO */
#ifdef CO_t3e
int COPAPO(_fcd portName, const int *isCellData)
{
   int32 data = *isCellData;
   if (coSendFTN(PARA_PORT,_fcdtocp(portName),_fcdlen(portName)))
	  return -1;  
   if (sendData((void*)&data,sizeof(data)) != sizeof(data) )
     return -1;
   else
	return 0;
}
#else
int COPAPO(const char *portName, const int *isCellData, int length)
{ 
   int32 data = *isCellData; 
   if (coSendFTN(PARA_PORT,portName,length))                          return -1;
   
   if (sendData((void*)&data,sizeof(data)) != sizeof(data) )
      return -1;
   else
      return 0;
}
#endif

int coParallelPort(const char *portName, int isCellData)
{ 
   int32 data = isCellData; 
   if ( coSendC(PARA_PORT,portName))                                  return -1;
   
   if (sendData((void*)&data,sizeof(data)) != sizeof(data) )
      return -1;
   else
      return 0;
}

/* --------------------------------------------------------------*/
/* Attach attribute to object at port                            */
#ifdef CO_t3e
int COATTR(_fcd pn,_fcd an, _fcd av)
{
   char buf[1024],*bPtr;
   int32 i; 

   const char *portName = _fcdtocp(pn);
   int         poLen    = _fcdlen(pn);
   const char *attrName = _fcdtocp(an);
   int         naLen    = _fcdlen(an);
   const char *attrVal = _fcdtocp(av);
   int         vaLen    = _fcdlen(av);


#else
int COATTR(const char *portName, const char *attrName, const char *attrVal,
           int poLen, int naLen, int vaLen)
{ 
   char buf[1024];
   int32 i; 
#endif

   if (coSendFTN(ATTRIBUTE,portName,poLen))                          return -1;
   if (naLen>1023 || vaLen> 1023)                                    return -1;
   
   /* copy name to buffer and remove blanks */
   strncpy(buf,attrName,naLen);
   buf[naLen]='\0';
   i=naLen-1;
   while (i>=0 && buf[i]==' ')   /*  not used isspace() : allow \t */
   {
      buf[i] = '\0';
      i--;
   }
   if (sendData((void*)&buf,1024) != 1024 )                          return -1;

   /* copy name to buffer and remove blanks */
   strncpy(buf,attrVal,vaLen);
   buf[vaLen]='\0';
   i=vaLen-1;
   while (i>=0 && buf[i]==' ')   /*  not used isspace() : allow \t */
   {
      buf[i] = '\0';
      i--;
   }
   if (sendData((void*)&buf,1024) != 1024 )                          return -1;


   return 0;
}

int coAddAttribute(const char *portName,
                   const char *attrName,
                   const char *attrVal)
{ 
   char buf[1024];

   if ( coSendC(ATTRIBUTE,portName))                                  return -1;
   
   strncpy(buf,attrName,1023);
   buf[1023]='\0';
   if (sendData(buf,1024) != 1024 )                                   return -1;

   strncpy(buf,attrVal,1023);
   buf[1023]='\0';
   if (sendData(buf,1024) != 1024 )                                   return -1;

   return 0;
}

/* --------------------------------------------------------------*/
/* common code for cell and vertex mapping                       */
static int sendMapping(int type, int fortrani, int node, int length, const int *field)
{
   struct { int32 type,fortrani,node,length; } data;
   data.type    = type;
   data.fortrani = fortrani;  /* =1 for FORTRAN, =0 for C : for array indices */
   data.node    = node;
   data.length  = length;
   if (sendData((void*)&data,sizeof(data)) != sizeof(data) )          return -1;
   
   assert(sizeof(int)==sizeof(int32));
   if (sendData((void*)field,sizeof(int)*length) != sizeof(int)*length )
      return -1;
   else
      return 0;
}

/* Send a cell mapping local -> global               F77: COPACM */
int COPACM(int *node, int *numCells, int *localToGlobal)
{  return sendMapping(PARA_CELL_MAP,1,*node,*numCells,localToGlobal); }

int coParallelCellMap(int node, int numCells, const int *localToGlobal)
{  return sendMapping(PARA_CELL_MAP,0,node,numCells,localToGlobal); }

/* Send a vertex mapping local -> global             F77: COPAVM */
int COPAVM(int *node, int *numCells, int *localToGlobal)
{  return sendMapping(PARA_VERTEX_MAP,1,*node,*numCells,localToGlobal); }

int coParallelVertexMap(int node, int numCells, const int *localToGlobal)
{  return sendMapping(PARA_VERTEX_MAP,0,node,numCells,localToGlobal); }

/* --------------------------------------------------------------*/
/* Next data sent is from node #                     F77: COPANO */
int COPANO(int *node) 
{ return coParallelNode(*node); }

int coParallelNode(int node)
{
   struct { int32 id ; int32 node; } data;
   data.id     = PARA_NODE;
   data.node   = node;
   if (sendData((void*)&data,sizeof(data)) != sizeof(data) )
      return -1;
   else
      return 0;
}



/******************************************************************
 *****                                                        *****
 ***** Binary send/receive: use this only if NOT using logics *****
 *****                                                        *****
 ******************************************************************/ 

/***********************************************      
 * Send a certain amount of data to the module *    FORTRAN: COSEND
 ***********************************************/
int COSEND(int *data, int *length)
{
   int size=*length;
   return sendData((const void *)data, size);
}

int sendData(const void *buffer, size_t length)
{
   register char *bptr=(char *)buffer;
   register int written;
   register int nbytes=length;
   if (coSimLibData.verbose>3)
      fprintf(stderr,"coSimClient sending %d Bytes to Socket %d\n", 
                       (int)length, coSimLibData.soc);
  
   while (nbytes>0) 
   {
      written = write(coSimLibData.soc,(void *) bptr,nbytes);
      if (written < 0) 
      {
         fprintf(stderr,"coSimClient error: write returned %d\n",written);
         perror("coSimClient sendData error: ");
         return -1;
      }
      nbytes-=written;
      bptr+=written;
      if (written==0) return -2;
   }
   if (coSimLibData.verbose>3)
      fprintf(stderr,"coSimClient sent %d Bytes\n",(int)length);
   return length;
}

/****************************************************
 * Receive a certain amount of data from the module *    FORTRAN: CORECV
 ****************************************************/

int CORECV(int *data, int *length)
{
   int size=*length;
   return recvData((void *)data, size);
}

int recvData(void *buffer, size_t length)  
{
   register char *bptr=(char*) buffer;
   register int nread;
   register int nbytes=length;
   if (coSimLibData.verbose>3)
      fprintf(stderr," coSimClient waiting for %d Bytes from Socket %d\n",
                     (int)length,coSimLibData.soc);
  
   while (nbytes>0) 
   {
      nread = read(coSimLibData.soc, (void*)bptr, nbytes);
      if (nread < 0) 
      {
         fprintf(stderr,"coSimClient error: received %d Bytes\n",nread);
         perror("coSimClient recvData error: ");
         assert(-1);
         return -1; 
      }
      nbytes-=nread;
      bptr+=nread;
      if (nread==0) break; 
   }
   if (nbytes) 
   {
      fprintf(stderr," error: received 0 Bytes while %d left\n",nbytes);
      perror("coSimClient recvData error: ");
      return -2;
   }
   else 
   {
      if (coSimLibData.verbose>3)
         fprintf(stderr,"coSimClient received %d Bytes\n",(int)length);
      return length;
   }
}

/****************************************************************************/
static int openServer(int minPort, int maxPort)
{
   int port;
   struct sockaddr_in addr_in;
   
   /* open the socket: if not possible, return -1 */
   coSimLibData.soc = socket(AF_INET, SOCK_STREAM, 0);
   if (coSimLibData.soc < 0) {
      return -1;
   } 

   port=minPort;

   /* Assign an address to this socket */
   memset((char *)&addr_in, 0, sizeof(addr_in));
   addr_in.sin_family = AF_INET;

   addr_in.sin_addr.s_addr = INADDR_ANY;
   addr_in.sin_port = htons(port);

   /* bind with changing port# until unused port found */
   while ( (port<=maxPort)   
        && (bind(coSimLibData.soc,(struct sockaddr*)&addr_in,sizeof(addr_in)) < 0)
          )  
   {
#ifndef _WIN32
      if (errno == EADDRINUSE)               /* if port is used (UNIX) */
#else
      if (GetLastError() == WSAEADDRINUSE)   /*                 (WIN32) */
#endif
      {
         port++;        /* try next port */
         addr_in.sin_port = htons(port);
      }
      else                                  /* other errors : ERROR, leave loop */
          port=maxPort+1;
   }
   
   /* we didn't find an empty one OR could not bind */
   if (port>maxPort)
   {
      close(coSimLibData.soc);
      coSimLibData.soc = -1;
      return -1;
   }
   else
   {
      listen(coSimLibData.soc, 20);
      return 0;
   }
}


/****************************************************************************/
static int openClient(unsigned long ip, int port, float timeout)
{
   int connectStatus=0;
   int numConnectTries = 0;
   struct sockaddr_in s_addr_in;
   do 
   {
      /* open the socket: if not possible, return -1 */
      coSimLibData.soc = socket(AF_INET, SOCK_STREAM, 0);
      if (coSimLibData.soc < 0) {
         return -1; 
      } 

      /* set s_addr structure */
      s_addr_in.sin_addr.s_addr = htonl(ip); 
      s_addr_in.sin_port = htons(port);
      s_addr_in.sin_family = AF_INET;

       /* Try connecting */
      connectStatus=connect(coSimLibData.soc,(struct sockaddr*)&s_addr_in,
                            sizeof(s_addr_in));
      
     /* didn't connect */
      if (connectStatus<0) 
      {    /* -> next port */
         numConnectTries++;
         sleep(1);
      }
   }
   while ( (connectStatus<0) && (numConnectTries<=((int)timeout) ) ); 

   if (connectStatus==0) {
      return 0;
    }
   else
   {
      return -1; 
   }
}
 
/****************************************************************************/
static int acceptServer(float wait)
{
   int tmp_soc;
   struct timeval timeout;
   fd_set fdread;
   int i;
   struct sockaddr_in s_addr_in;
   int length;

   /* prepare for select(2) call and wait for incoming connection */
   timeout.tv_sec = (int)wait;
   timeout.tv_usec = (int)((wait-timeout.tv_sec)*1000000);
   FD_ZERO(&fdread);
   FD_SET(coSimLibData.soc, &fdread);
   if (wait>=0)                  /* wait period was specified */
      i=select(coSimLibData.soc+1, &fdread, NULL, NULL, &timeout);
   else                          /* wait infinitly */
      i=select(coSimLibData.soc+1, &fdread, NULL, NULL, NULL);
 
   if (i == 0) {     /* nothing happened: return -1 */
      close(coSimLibData.soc);
      coSimLibData.soc=-1;
      return -1;
   }

   /* now accepting the connection */
   length = sizeof(s_addr_in);
   tmp_soc = accept(coSimLibData.soc, (struct sockaddr *)&s_addr_in, &length);
    
   if (tmp_soc < 0) {
      close(coSimLibData.soc);
      coSimLibData.soc=-1;
      return -1;
   }

#ifdef DEBUG
   fprintf(stderr,"Connection from host %s, port %u\n",
        Xinet_ntoa(s_addr_in.sin_addr), ntohs(s_addr_in.sin_port));
   fflush(stdout);
#endif

   close(coSimLibData.soc);

   /* use the socket 'accept' delivered */
   coSimLibData.soc = tmp_soc;

   /* END_CRITICAL(this,"accept") */
   return 0;
}

/****************************************************************************/

/* get the verbose level                            F77: COVERB */
int getVerboseLevel()
{
   return coSimLibData.verbose;
}  

int COVERB()
{
    return coSimLibData.verbose;
}
