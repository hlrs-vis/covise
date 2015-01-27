/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !(defined(WIN32) || defined(WIN64))
#include <unistd.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/fcntl.h>
#else
#include <winsock2.h>
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#ifdef _SX
#include <sys/select.h>
#endif

#if defined(CO_t3e)
#include <fortran.h>
#endif

#include <ctype.h>
#include "coSimClient.h"
#include "coSimLibComm.h"
#include <assert.h>

#ifdef HAVE_GLOBUS
#undef IOV_MAX
#include <SimulationService_client.h>
#include "globus_common.h"
#include "globus_io.h"
#include <pwd.h>
int simulationID;
SimulationService_client_handle_t client_handle;
wsa_EndpointReferenceType *epr;
globus_soap_message_attr_t message_attr;
#endif

/* SGI uses lowercase and trailing _ for FORTRAN */
#if defined(__sgi) || defined(__linux) || defined(_SX)
#define COVINI covini_
#define COVWSI covwsi_
#define COVWSE covwse_
#define COVINI covini_
#define COFINI cofini_
#define CONOCO conoco_
#define COGPSL cogpsl_
#define COGPFL cogpfl_
#define COGVFL cogvfl_
#define COGPIN cogpin_
#define COGPTX cogptx_
#define COGPFI cogpfi_
#define COGPBO cogpbo_
#define COGPCH cogpch_
#define COSU1D cosu1d_
#define COSU3D cosu3d_
#define COEXEC coexec_
#define COPAIN copain_
#define COPAPO copapo_
#define COPACM copacm_
#define COPAVM copavm_
#define COPANO copano_
#define CORECV corecv_
#define CORRCV corrcv_
#define COSEND cosend_
#define COVERB coverb_
#define COATTR coattr_
#define COGDIM cogdim_
#define COBDIM cobdim_
#define CORGEO corgeo_
#define ATTACH attach_
#define DETACH detach_
#define COSIPD cosipd_
#define COEXIT coexit_
#define COSLEP coslep_
#else
#ifdef __hpux
#define COVINI covini
#define COVWSI covwsi
#define COVWSE covwse
#define COFINI cofini
#define CONOCO conoco
#define COGPSL cogpsl
#define COGPFL cogpfl
#define COGVFL cogvfl
#define COGPIN cogpin
#define COGPTX cogptx
#define COGPFI cogpfi
#define COGPBO cogpbo
#define COGPCH cogpch
#define COSU1D cosu1d
#define COSU3D cosu3d
#define COEXEC coexec
#define COPAIN copain
#define COPAPO copapo
#define COPACM copacm
#define COPAVM copavm
#define COPANO copano
#define CORECV corecv
#define CORRCV corrcv
#define COSEND cosend
#define COVERB coverb
#define COATTR coattr
#define COGDIM cogdim
#define COBDIM cobdim
#define CORGEO corgeo
#define ATTACH attach
#define DETACH detach
#define COEXIT coexit
#define COSIPD cosipd
#endif
#endif
#ifdef __cplusplus
extern "C" {
#endif

extern int COVWSI(void);
extern int COVWSE(void);
extern int COVINI(void);
extern int CONOCO();
extern int COEXEC();
extern int COFINI();
extern int COSIPD();
extern int COPAIN(const int *numParts, const int *numPorts);
extern int COEXIT();
extern int DETACH();
extern int ATTACH();
extern int COPACM(int *node, int *numCells, int *localToGlobal);
extern int COPAVM(int *node, int *numCells, int *localToGlobal);
extern int COPANO(int *node);
extern int COSEND(int *data, int *length);
extern int CORECV(int *data, int *length);
extern int CORRCV(float *data, int *length);
extern int COVERB();
extern int COGDIM(int *npoin_ges, int *nelem_ges, int *knmax_num, int *elmax_num,
                  int *npoin_geb, int *nelem_geb);
extern int COBDIM(int *nrbpoi_geb, int *nwand_geb, int *npres_geb, int *nsyme_geb,
                  int *nconv_geb);
extern int CORGEO();
extern int COSLEP(int time);
#ifndef MIXED_STR_LEN
extern int COGPSL(char *name, float *min, float *max, float *val, int length);
extern int COGPFL(char *name, float *val, int length);
extern int COGVFL(char *name, float *val, int length);
extern int COGPIN(char *name, int *val, int length);
extern int COGPCH(char *name, int *val, int length);
extern int COGPBO(char *name, int *val, int length);
extern int COGPTX(const char *name, char *data, int lenNane, int lenData);
extern int COGPFI(const char *name, char *data, int lenNane, int lenData);
extern int COSU1D(const char *portName, int *numElem, float *data, int length);
extern int COSU3D(const char *portName, int *numElem, float *data0, float *data1, float *data2, int length);
extern int COPAPO(const char *portName, const int *isCellData, int length);
extern int COATTR(const char *portName, const char *attrName, const char *attrVal, int poLen, int naLen, int vaLen);
#else
extern int COGPSL(char *name, int length, float *min, float *max, float *val);
extern int COGPFL(char *name, int length, float *val);
extern int COGVFL(char *name, int length, float *val);
extern int COGPIN(char *name, int length, int *val);
extern int COGPCH(char *name, int length, int *val);
extern int COGPBO(char *name, int length, int *val);
extern int COGPTX(const char *name, int lenName, char *data, int lenData);
extern int COGPFI(const char *name, int lenName, char *data, int lenData);
extern int COSU1D(const char *portName, int length, int *numElem, float *data);
extern int COSU3D(const char *portName, int length, int *numElem, float *data0, float *data1, float *data2);
extern int COPAPO(const char *portName, int length, const int *isCellData);
extern int COATTR(const char *portName, int poLen, const char *attrName, int naLen, const char *attrVal, int vaLen);
#endif

#ifdef __cplusplus
}
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
    int serv;
    int verbose;
} coSimLibData = { -1, -1, 0 };

/************ Utilities ******************/

static int openServer(int minPort, int maxPort);
static int openClient(unsigned long ip, int port, float timeout);
static int acceptServer(float wait);

/************ ESTABLISH CONNECTION ******************/

int COVWSI(void)
{
    return coWSAInit();
}

int coWSAInit()
{
    int iResult = 0;
#ifdef _WIN32
    WSADATA wsaData;
    iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (iResult != NO_ERROR)
        fprintf(stderr, "Error: WSAStartup() - failed!\n");
#endif
    return iResult;
}

int COVWSE(void)
{
    return coWSAEnd();
}

int coWSAEnd()
{
    int iResult = 0;
#ifdef _WIN32
    iResult = WSACleanup();
#endif
    return iResult;
}

int COVINI(void)
{
    return coInitConnect();
}

int coInitConnect()
{
    const char *envStr;
    char env[256], *portStr;
    float timeout;
    int32 handshake;
    int port;

    coSimLibData.soc = -1;

#if !(defined(WIN32) || defined(WIN64))
    /* we do our own handling of broken pipes */
    signal(SIGPIPE, SIG_IGN);
#endif

    /* get environment: if variable not set, exit with error-code */
    envStr = getenv("CO_SIMLIB_CONN");
    fprintf(stdout, "SIMLIB: %s\n", envStr);
    if ((!envStr) || (strlen(envStr) > 255))
        return -1;

    /* Client connection */
    strcpy(env, envStr);
    if (*env == 'C')
    {
        size_t retval;
        isServer = 0;
        /* get adress and port */
        portStr = strchr(env, '/');
        if (!portStr)
        {
            fprintf(stderr, "error parsing environment variable [%s]\n", env);
            return -1;
        }
        *portStr = '\0';
        portStr++;
        retval = sscanf(portStr, "%d_%f_%d", &minPort, &timeout, &coSimLibData.verbose);
        if (retval != 3)
        {
            fprintf(stderr, "coInitConnect: sscanf failed\n");
            return -1;
        }
        if (minPort < 1024 || minPort > 32767)
            return -1;

#ifndef WIN32
        if (!inet_aton(env + 2, &ip))
            return -1;
#else
        ip.s_addr = inet_addr(env + 2);
        if (ip.s_addr == -1)
            return -1;
#endif
        if (coSimLibData.verbose > 0)
        {
            fprintf(stderr, " Starting Client to %s Port %d with %f sec timeout\n",
                    inet_ntoa(ip), minPort, timeout);
        }

        /* we try to connect now */
        if ((port = openClient(ip.s_addr, minPort, timeout)) < 0)
            return -1;
    }

    /* Server connection */
    else if (*env == 'S')
    {
        size_t retval;
        isServer = 1;
        /* get adress and port */
        portStr = strchr(env, ':');
        if (!portStr)
        {
            fprintf(stderr, "error parsing environment variable [%s]\n", env);
            return -1;
        }
        *portStr = '\0';
        portStr++;
        retval = sscanf(portStr, "%d-%d_%f_%d", &minPort, &maxPort, &timeout, &coSimLibData.verbose);
        if (retval != 4)
        {
            fprintf(stderr, "coInitConnect: sscanf failed\n");
            return -1;
        }

        if (minPort < 1024 || minPort > 32767 || maxPort < 1024 || maxPort > 32767)
            return -1;

        /* we open and wait for the other side to connect */
        if ((port = openServer(minPort, maxPort)) < 0)
        {
            fprintf(stderr, "could not open server\n");
            return -1;
        }

        if (acceptServer(timeout) < 0)
        {
            fprintf(stderr, "could not accept server\n");
            return -1;
        }
    }
    /* Neither Client nor Server = ERROR */
    else
        return -1;

    /* Handshake: send 12345 to other side, so they might determine byte-swapping */
    handshake = 12345;
    sendData(&handshake, sizeof(int32));

#ifdef HAVE_GLOBUS
    {
        globus_result_t result = GLOBUS_SUCCESS;
        xsd_any *fault;
        int err, fault_type;
        fprintf(stderr, "activate globus modules\n");
        globus_module_activate(GLOBUS_COMMON_MODULE);
        globus_module_activate(GLOBUS_SOAP_MESSAGE_MODULE);
        registerSimulationType regSimulation;
        registerSimulationResponseType *regResponse;
        registerSimulationResponseType_init(&regResponse);
        globus_soap_message_attr_init(&message_attr);
        globus_soap_message_attr_set(message_attr, GLOBUS_SOAP_MESSAGE_AUTHZ_METHOD_KEY,
                                     NULL, NULL,
                                     (void *)GLOBUS_SOAP_MESSAGE_AUTHZ_HOST);
        globus_soap_message_attr_set(message_attr, GLOBUS_SOAP_MESSAGE_AUTH_PROTECTION_KEY,
                                     NULL, NULL,
                                     (void *)GLOBUS_SOAP_MESSAGE_AUTH_PROTECTION_PRIVACY);
        if ((result = SimulationService_client_init(&client_handle, message_attr, NULL)) == GLOBUS_SUCCESS)
        {
            char hostname[128];
            struct passwd *user = getpwuid(getuid());

            gethostname(hostname, 127);
            fprintf(stderr, "globus simulation client initialized\n");

            regSimulation.user = user->pw_name;
            regSimulation.host = hostname;
            regSimulation.port = port;
            regSimulation.name = "Fenfloss";
            fprintf(stderr, "globus regSimulation: [%s] [%s] [%d] [%s]\n", regSimulation.user, regSimulation.host, regSimulation.port, regSimulation.name);
            wsa_EndpointReferenceType_init(&epr);
            wsa_AttributedURI_init_contents(&epr->Address);
            xsd_anyURI_init_contents_cstr(&epr->Address.base_value,
                                          globus_common_create_string(getenv("GLOBUS_SIMULATIONSERVICE")));
            fprintf(stderr, " [%s]\n", getenv("GLOBUS_SIMULATIONSERVICE"));
            if ((err = SimulationPortType_registerSimulation_epr(client_handle,
                                                                 epr,
                                                                 &regSimulation,
                                                                 &regResponse,
                                                                 (SimulationPortType_registerSimulation_fault_t *)&fault_type,
                                                                 &fault)) == GLOBUS_SUCCESS)
            {
                SimulationType r = regResponse->result;

                simulationID = r.id;
            }
            else
            {
                fprintf(stderr, "globus error %d: [%s] \n", err, globus_object_printable_to_string(globus_error_get(err)));
            }
        }
        else
        {
            fprintf(stderr, "globus error %d: [%s] \n", result, globus_object_printable_to_string(globus_error_get(result)));
        }
        registerSimulationResponseType_destroy(regResponse);
    }
#endif
    return 0;
}

int CONOCO()
{
    return coNotConnected();
}

int coNotConnected()
{
    /* try to send: if not possible, thenm it's not connected */
    int32 testdata = COMM_TEST;
    if (coSimLibData.soc < 0
        || sendData((void *)&testdata, sizeof(int32)) != sizeof(int32))
        return -1;
    else
        return 0;
}

/****** Logic send/receive calls ******/

static int coSendCommand(int32 command)
{

    int n;
    if ((n = sendData(&command, sizeof(int32))) != sizeof(int32))
        return -1;
    return n;
}

/* Utilities for Parameter requests */
static int coSendFTN(int32 type, const char *name, int length)
{
    char buffer[64];
    int i;

    /* check length */
    if (length > 63)
        return -1;

    /* Send request */
    if (sendData((void *)&type, sizeof(int32)) != sizeof(int32))
        return -1;

    /* Send name with fixed length of 64 bytes */
    if (length > 63)
        length = 63;
    strncpy(buffer, name, length);
    buffer[length] = '\0';

    /* remove everything after first blank and make sute the string is terminated */
    i = 0;
    while (buffer[i] && i < 63 && !isspace(buffer[i]))
        i++;
    buffer[i] = '\0';

    if (sendData((void *)buffer, 64) != 64)
        return -1;
    else
        return 0;
}

/* Utilities for Parameter requests */
static int coSendC(int32 type, const char *name)
{
    char buffer[64];

    /* check length */
    int length = strlen(name);
    if (length > 63)
        return -1;

    /* Send request */
    if (sendData((void *)&type, sizeof(int32)) != sizeof(int32))
        return -1;

    /* Send name with fixed length of 64 bytes */
    strcpy(buffer, name);
    if (sendData((void *)buffer, 64) != 64)
        return -1;
    else
        return 0;
}

int coSendParaDone()
{

    return coSendCommand(GET_INITIAL_PARA_DONE);
}

int COSIPD()
{
    return coSendParaDone();
}

/***********************************************************************/
/**************** Read a slider Parameter of the module ****************/

/* Common Slider code for C and Fortran */
static int coGetParaSli(float *min, float *max, float *val)
{
    /* Receive result */
    struct
    {
        float min, max, val;
        int32 error;
    } ret;
    if (recvData((void *)&ret, sizeof(ret)) != sizeof(ret)
        || ret.error)
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
    if (coSendFTN(GET_SLI_PARA, _fcdtocp(name), _fcdlen(name)))
        return -1;
    return coGetParaSli(min, max, val);
}

#else
#ifdef MIXED_STR_LEN
int COGPSL(char *name, int length, float *min, float *max, float *val)
#else
int COGPSL(char *name, float *min, float *max, float *val, int length)
#endif
{
    if (coSendFTN(GET_SLI_PARA, name, length))
        return -1;
    return coGetParaSli(min, max, val);
}
#endif

/* C API */
int coGetParaSlider(const char *name, float *min, float *max, float *val)
{
    if (coSendC(GET_SLI_PARA, name))
        return -1;
    return coGetParaSli(min, max, val);
}

/*******************************************************************************/
/****** Read a flaot scalar Parameter of the module  Fortran 77: COGPFL() ******/

/* Common Slider code for C and Fortran */
static int coGetParaScaFlo(float *val)
{
    /* Receive result */
    int n;
    struct
    {
        float val;
        int32 error;
    } ret;
    if ((n = recvData((void *)&ret, sizeof(ret))) != sizeof(ret)
        || ret.error)
        return -1;

    *val = ret.val;
    return 0;
}

/* Fortran API */
#ifdef CO_t3e
int COGPFL(_fcd name, float *val)
{
    if (coSendFTN(GET_SC_PARA_FLO, _fcdtocp(name), _fcdlen(name)))
        return -1;
    return coGetParaScaFlo(val);
}

#else
#ifdef MIXED_STR_LEN
int COGPFL(char *name, int length, float *val)
#else
int COGPFL(char *name, float *val, int length)
#endif
{
    if (coSendFTN(GET_SC_PARA_FLO, name, length))
    {
        return -1;
    }
    return coGetParaScaFlo(val);
}
#endif

int coGetParaFloatScalar(const char *name, float *val)
{
    if (coSendC(GET_SC_PARA_FLO, name))
        return -1;
    return coGetParaScaFlo(val);
}

/*******************************************************************************/
/**** Read a float 3D vector Parameter of the module  Fortran 77: COGPFL()  ****/

/* Common Slider code for C and Fortran */
static int coGetParaVecFlo(float *val)
{
    /* Receive result */
    int n, i;
    struct
    {
        float val[3];
        int32 error;
    } ret;
    if ((n = recvData((void *)&ret, sizeof(ret))) != sizeof(ret)
        || ret.error)
        return -1;

    for (i = 0; i < 3; i++)
        val[i] = ret.val[i];
    return 0;
}

/* Fortran API */
#ifdef CO_t3e
int COGVFL(_fcd name, float *val)
{
    if (coSendFTN(GET_V3_PARA_FLO, _fcdtocp(name), _fcdlen(name)))
        return -1;
    return coGetParaVecFlo(val);
}

#else
#ifdef MIXED_STR_LEN
int COGVFL(char *name, int length, float *val)
#else
int COGVFL(char *name, float *val, int length)
#endif
{
    if (coSendFTN(GET_V3_PARA_FLO, name, length))
    {
        return -1;
    }
    return coGetParaVecFlo(val);
}
#endif

int coGetParaFloatVector(const char *name, float *val)
{
    if (coSendC(GET_V3_PARA_FLO, name))
        return -1;
    return coGetParaVecFlo(val);
}

/*******************************************************************************/
/****** Read a int scalar Parameter of the module  Fortran 77: COGPIN() ******/

/* Common Slider code for C and Fortran */
static int coGetParaScaInt(int *val) /* Receive result */
{
    struct
    {
        int32 val;
        int32 error;
    } ret;
    if (recvData((void *)&ret, sizeof(ret)) != sizeof(ret)
        || ret.error)
        return -1;

    *val = ret.val;
    return 0;
}

/* Fortran API */

#ifdef CO_t3e
int COGPIN(_fcd name, int *val)
{
    if (coSendFTN(GET_SC_PARA_INT, _fcdtocp(name), _fcdlen(name)))
        return -1;
    return coGetParaScaInt(val);
}

#else
#ifdef MIXED_STR_LEN
int COGPIN(char *name, int length, int *val)
#else
int COGPIN(char *name, int *val, int length)
#endif
{
    if (coSendFTN(GET_SC_PARA_INT, name, length))
        return -1;
    return coGetParaScaInt(val);
}
#endif

int coGetParaIntScalar(const char *name, int *val)
{
    if (coSendC(GET_SC_PARA_INT, name))
        return -1;
    return coGetParaScaInt(val);
}

/*******************************************************************************/
/****** Read a choice Parameter of the module  Fortran 77: COGPCH ******/

/* Common Slider code for C and Fortran */
static int coGetParaCh(int *val) /* Receive result */
{
    struct
    {
        int32 val;
        int32 error;
    } ret;
    if (recvData((void *)&ret, sizeof(ret)) != sizeof(ret)
        || ret.error)
        return -1;

    *val = ret.val;
    return 0;
}

/* Fortran API */
#ifdef CO_t3e
int COGPCH(_fcd name, int *val)
{
    if (coSendFTN(GET_CHOICE_PARA, _fcdtocp(name), _fcdlen(name)))
        return -1;
    return coGetParaCh(val);
}

#else
#ifdef MIXED_STR_LEN
int COGPCH(char *name, int length, int *val)
#else
int COGPCH(char *name, int *val, int length)
#endif
{
    if (coSendFTN(GET_CHOICE_PARA, name, length))
        return -1;
    return coGetParaCh(val);
}
#endif

/* C API */
int coGetParaChoice(const char *name, int *val)
{
    if (coSendC(GET_CHOICE_PARA, name))
        return -1;
    return coGetParaCh(val);
}

/*******************************************************************************/
/****** Read a boolean Parameter of the module      Fortran 77: COGPBO    ******/

/* Common Slider code for C and Fortran */
static int coGetParaBo(int *val) /* Receive result */
{
    struct
    {
        int32 val;
        int32 error;
    } ret;

    if (recvData((void *)&ret, sizeof(ret)) != sizeof(ret) || ret.error)
        return -1;

    *val = ret.val;
    return 0;
}

/* Fortran API */
#ifdef CO_t3e
int COGPBO(_fcd name, int *val)
{
    if (coSendFTN(GET_BOOL_PARA, _fcdtocp(name), _fcdlen(name)))
        return -1;
    return coGetParaBo(val);
}

#else
#ifdef MIXED_STR_LEN
int COGPBO(char *name, int length, int *val)
#else
int COGPBO(char *name, int *val, int length)
#endif
{
    if (coSendFTN(GET_BOOL_PARA, name, length))
        return -1;
    return coGetParaBo(val);
}
#endif

/* C API */
int coGetParaBool(const char *name, int *val)
{
    if (coSendC(GET_BOOL_PARA, name))
        return -1;
    return coGetParaBo(val);
}

/*******************************************************************************/
/****** Read a Text Parameter of the module         Fortran 77: COGPTX    ******/

/* Read a Text Parameter of the module */
int coGetParaText(const char *name, char *data)
{
    char buffer[256];
    if (coSendC(GET_TEXT_PARA, name))
        return -1;
    ;
    if (recvData((void *)&buffer, 256) != 256)
    {
        fprintf(stdout, "cogetparatext():%s", buffer);
        return -1;
    }
    strcpy(data, buffer);
    if (strlen(data) == 0)
        return -1;
    return 0;
}

#ifdef CO_t3e
int COGPTX(_fcd name, _fcd strdata)
{
    char buffer[256], *data;
    int i;
    data = _fcdtocp(strdata);
    if (coSendFTN(GET_TEXT_PARA, _fcdtocp(name), _fcdlen(name)))
        return -1;

    if (recvData((void *)&buffer, 256) != 256)
        return -1;

    strcpy(data, buffer);
    for (i = strlen(buffer); i < 256; i++) /* FORTRAN is blank padded  */
        data[i] = ' ';
    return 0;
}

#else
#ifdef MIXED_STR_LEN
int COGPTX(const char *name, int lenName, char *data, int lenData)
#else
int COGPTX(const char *name, char *data, int lenName, int lenData)
#endif
{
    char buffer[256];
    int i;
    if (lenData > 256)
        lenData = 256;

    if (coSendFTN(GET_TEXT_PARA, name, lenName))
        return -1;
    if (recvData((void *)&buffer, 256) != 256)
        return -1;
    strncpy(data, buffer, lenData);
    for (i = strlen(buffer); i < lenData; i++) /* FORTRAN is blank padded */
        data[i] = ' ';
    return 0;
}
#endif

/*******************************************************************************/

/* Read a Filename Parameter of the module */
int coGetParaFile(const char *name, int *data) /* Fortran 77: COGPFI() */
{
    if (coSendC(GET_FILE_PARA, name))
        return -1;
    ;
    if (recvData((void *)&data, 256) != 256)
        return -1;
    return 0;
}

#ifdef CO_t3e
int COGOFI(_fcd name, _fcd strdata)
{
    char buffer[256], *data;
    int i;
    data = _fcdtocp(strdata);
    if (coSendFTN(GET_FILE_PARA, _fcdtocp(name), _fcdlen(name)))
        return -1;

    if (recvData((void *)&buffer, 256) != 256)
        return -1;

    strcpy(data, buffer);
    for (i = strlen(buffer); i < 256; i++) /* FORTRAN is blank padded  */
        data[i] = ' ';
    return 0;
}

#else
#ifdef MIXED_STR_LEN
int COGPFI(const char *name, int lenName, char *data, int lenData)
#else
int COGPFI(const char *name, char *data, int lenName, int lenData)
#endif
{
    char buffer[256];
    int i;
    (void)lenData;
    if (coSendFTN(GET_FILE_PARA, name, lenName))
        return -1;
    if (recvData((void *)&buffer, 256) != 256)
        return -1;
    strcpy(data, buffer);
    for (i = strlen(buffer); i < 256; i++) /* FORTRAN is blank padded */
        data[i] = ' ';
    return 0;
}
#endif

/* Send an Unstructured Grid, Covise format */
int coSendUSGcov(const char *portName,
                 int numElem, int numConn, int numCoord, /* Fortran 77: COSUGC */
                 int *elemList, int *connList,
                 float *xCoord, float *yCoord, float *zCoord)
{
    (void)portName;
    (void)numElem;
    (void)numConn;
    (void)numCoord;
    (void)elemList;
    (void)connList;
    (void)xCoord;
    (void)yCoord;
    (void)zCoord;
    return 0;
}

/* Send an Unstructured Grid, xyz coordinate fields,
   8-elem conn List with multiple point for non-hex elements (e.g. STAR) */
int coSendUSGhex(const char *portName,
                 int numElem, int numCoord, /* Fortran 77: COSUSG */
                 int *elemList, float *coord)
{
    (void)portName;
    (void)numElem;
    (void)numCoord;
    (void)elemList;
    (void)coord;
    return 0;
}

/********* Send an USG vector data field    Fortran 77: COSU3D ********/
int coSend1DataCommon(int numElem, float *data)
{
    int32 num = numElem;
    if (sendData((void *)&num, sizeof(int32)) != sizeof(int32))
        return -1;
    if (sendData((void *)data, numElem * sizeof(float)), numElem * sizeof(float)
                                                         != numElem * sizeof(float))
        return -1;
    return 0;
}

#ifdef CO_t3e
int COSU1D(_fcd portName, int *numElem, float *data)
{
    if (coSendFTN(SEND_1DATA, _fcdtocp(portName), _fcdlen(portName)))
        return -1;
    return coSend1DataCommon(*numElem, data);
}

#else
#ifdef MIXED_STR_LEN
int COSU1D(const char *portName, int length, int *numElem, float *data)
#else
int COSU1D(const char *portName, int *numElem, float *data, int length)
#endif
{
    if (coSendFTN(SEND_1DATA, portName, length))
        return -1;
    return coSend1DataCommon(*numElem, data);
}
#endif
int coSend1Data(const char *portName, int numElem, float *data)
{
    if (coSendC(SEND_1DATA, portName))
        return -1;
    return coSend1DataCommon(numElem, data);
}

/********* Send an USG vector data field    Fortran 77: COSU3D ********/
int coSend3DataCommon(int numElem, float *data0, float *data1, float *data2)
{
    int32 num = numElem;
    if (sendData((void *)&num, sizeof(int32)) != sizeof(int32))
        return -1;
    if (sendData((void *)data0, numElem * sizeof(float)) != numElem * sizeof(float))
        return -1;
    if (sendData((void *)data1, numElem * sizeof(float)) != numElem * sizeof(float))
        return -1;
    if (sendData((void *)data2, numElem * sizeof(float)) != numElem * sizeof(float))
        return -1;
    return 0;
}

#ifdef CO_t3e
int COSU3D(_fcd portName, int *numElem, float *data0, float *data1, float *data2)
{
    if (coSendFTN(SEND_3DATA, _fcdtocp(portName), _fcdlen(portName)))
        return -1;
    return coSend3DataCommon(*numElem, data0, data1, data2);
}

#else
#ifdef MIXED_STR_LEN
int COSU3D(const char *portName, int length, int *numElem, float *data0, float *data1, float *data2)
#else
int COSU3D(const char *portName, int *numElem, float *data0, float *data1, float *data2, int length)
#endif
{
    if (coSendFTN(SEND_3DATA, portName, length))
        return -1;
    return coSend3DataCommon(*numElem, data0, data1, data2);
}
#endif

int coSend3Data(const char *portName, int numElem, float *data0,
                float *data1, float *data2)
{
    if (coSendC(SEND_3DATA, portName))
        return -1;
    return coSend3DataCommon(numElem, data0, data1, data2);
}

/* End Server in module and let pipeline run */
int COFINI()
{
    return coFinished();
}

int coFinished() /* Fortran 77: COWAIT */
{
    int32 testdata = COMM_QUIT;
    if (/*coSimLibData.soc < 0 || */
        sendData((void *)&testdata, sizeof(int32)) != sizeof(int32))
        return -1;
    else
        return 0;
}

/* Execute the Covise Module now :  Fortran 77: COEXEC */
int COEXEC()
{
    return coExecModule();
}
int coExecModule()
{
    int32 execCommand = EXEC_COVISE;
    if (sendData((void *)&execCommand, sizeof(int32)) != sizeof(int32))
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
{
    return coParallelInit(*numParts, *numPorts);
}

int coParallelInit(int numParts, int numPorts)
{
    struct
    {
        int32 type, parts, ports;
    } data;
    data.type = PARA_INIT;
    data.parts = numParts;
    data.ports = numPorts;
    if (sendData((void *)&data, sizeof(data)) != sizeof(data))
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
    if (coSendFTN(PARA_PORT, _fcdtocp(portName), _fcdlen(portName)))
        return -1;
    if (sendData((void *)&data, sizeof(data)) != sizeof(data))
        return -1;
    else
        return 0;
}

#else
#ifdef MIXED_STR_LEN
int COPAPO(const char *portName, int length, const int *isCellData)
#else
int COPAPO(const char *portName, const int *isCellData, int length)
#endif
{
    int32 data = *isCellData;
    if (coSendFTN(PARA_PORT, portName, length))
        return -1;

    if (sendData((void *)&data, sizeof(data)) != sizeof(data))
        return -1;
    else
        return 0;
}
#endif

int coParallelPort(const char *portName, int isCellData)
{
    int32 data = isCellData;
    if (coSendC(PARA_PORT, portName))
        return -1;

    if (sendData((void *)&data, sizeof(data)) != sizeof(data))
        return -1;
    else
        return 0;
}

int COEXIT()
{
    int32 data = COMM_EXIT;

#ifdef HAVE_GLOBUS
    xsd_any *fault;
    int fault_type;
    int err;

    unregisterSimulationType unregSimulation;
    unregisterSimulationResponseType *unregResponse;
    unregisterSimulationResponseType_init(&unregResponse);

    unregSimulation.user = getenv("USER");
    unregSimulation.id = simulationID;

    if ((err = SimulationPortType_unregisterSimulation_epr(client_handle,
                                                           epr,
                                                           &unregSimulation,
                                                           &unregResponse,
                                                           (SimulationPortType_unregisterSimulation_fault_t *)&fault_type,
                                                           &fault)) == GLOBUS_SUCCESS)
    {
        fprintf(stderr, "globus unregistered: %d\n", unregResponse->result);
    }
    else
    {
        fprintf(stderr, "globus error %d: [%s] \n", err, globus_object_printable_to_string(globus_error_get(err)));
    }

    unregisterSimulationResponseType_destroy(unregResponse);
#endif

    sendData((void *)&data, sizeof(int32));

    close(coSimLibData.soc);
    coSimLibData.soc = -1;

    return 0;
}

int DETACH()
{
    return coDetach();
}

int coDetach(void)
{
    int32 data = COMM_DETACH;
    sendData((void *)&data, sizeof(int32));

    close(coSimLibData.soc);
    coSimLibData.soc = -1;

    fprintf(stderr, "Simulation detached from COVISE\n");

#ifdef HAVE_GLOBUS
    {
        xsd_any *fault;
        int fault_type;
        int err;

        setStatusType setStatus;
        setStatusResponseType *statusResponse;
        setStatusResponseType_init(&statusResponse);

        setStatus.user = getenv("USER");
        setStatus.id = simulationID;
        setStatus.status = 1; // detach

        if ((err = SimulationPortType_setStatus_epr(client_handle,
                                                    epr,
                                                    &setStatus,
                                                    &statusResponse,
                                                    (SimulationPortType_setStatus_fault_t *)&fault_type,
                                                    &fault)) == GLOBUS_SUCCESS)
        {
            fprintf(stderr, "globus set status: %d\n", statusResponse->result);
        }
        else
        {
            fprintf(stderr, "globus error %d: [%s] \n", err, globus_object_printable_to_string(globus_error_get(err)));
        }
        setStatusResponseType_destroy(statusResponse);
    }
#endif

    return 0;
}

int ATTACH()
{
    return coAttach();
}

int coAttach(void)
{
    int result;
    if (coSimLibData.soc == -1)
    {
        fprintf(stderr, "trying to reattach to covise...");
        if (isServer)
        {
            result = acceptServer(0.05f);
        }
        else
        {
            result = openClient(ip.s_addr, minPort, 1);
        }

        if (result != -1)
        {
            int32 handshake = 12345;
            fprintf(stderr, "attached\n");
            sendData(&handshake, sizeof(int32));

#ifdef HAVE_GLOBUS
            {
                xsd_any *fault;
                int err, fault_type;
                getSimulationType getSimulation;
                getSimulationResponseType *getResponse;
                getSimulationResponseType_init(&getResponse);

                getSimulation.user = getenv("USER");
                getSimulation.host = getenv("HOSTNAME");
                getSimulation.port = result;
                if ((err = SimulationPortType_getSimulation_epr(client_handle,
                                                                epr,
                                                                &getSimulation,
                                                                &getResponse,
                                                                (SimulationPortType_getSimulation_fault_t *)&fault_type,
                                                                &fault)) == GLOBUS_SUCCESS)
                {
                    if (getResponse != 0)
                    {
                        SimulationType sim = getResponse->result;

                        setStatusType setStatus;
                        setStatusResponseType *statusResponse;
                        setStatusResponseType_init(&statusResponse);

                        setStatus.user = sim.user;
                        setStatus.id = sim.id;
                        setStatus.status = 0; // attach

                        if ((err = SimulationPortType_setStatus_epr(client_handle,
                                                                    epr,
                                                                    &setStatus,
                                                                    &statusResponse,
                                                                    (SimulationPortType_setStatus_fault_t *)&fault_type,
                                                                    &fault)) == GLOBUS_SUCCESS)
                        {
                            fprintf(stderr, "globus set status: %d\n", statusResponse->result);
                        }
                        else
                        {
                            fprintf(stderr, "globus error %d: [%s] \n", err, globus_object_printable_to_string(globus_error_get(err)));
                        }
                        setStatusResponseType_destroy(statusResponse);
                    }
                }
                else
                {
                    fprintf(stderr, "globus error %d: [%s] \n", err, globus_object_printable_to_string(globus_error_get(err)));
                }
                getSimulationResponseType_destroy(getResponse);
            }
#endif

            return 1;
        }
        else
        {
            fprintf(stderr, "failed\n");
        }
    }
    return 0;
}

/* --------------------------------------------------------------*/
/* Attach attribute to object at port                            */
#ifdef CO_t3e
int COATTR(_fcd pn, _fcd an, _fcd av)
{
    char buf[1024], *bPtr;
    int32 i;

    const char *portName = _fcdtocp(pn);
    int poLen = _fcdlen(pn);
    const char *attrName = _fcdtocp(an);
    int naLen = _fcdlen(an);
    const char *attrVal = _fcdtocp(av);
    int vaLen = _fcdlen(av);

#else

#ifdef MIXED_STR_LEN
int COATTR(const char *portName, int poLen,
           const char *attrName, int naLen,
           const char *attrVal, int vaLen)
#else
int COATTR(const char *portName, const char *attrName, const char *attrVal,
           int poLen, int naLen, int vaLen)
#endif
{
    char buf[1024];
    int32 i;
#endif

    if (coSendFTN(ATTRIBUTE, portName, poLen))
        return -1;
    if (naLen > 1023 || vaLen > 1023)
        return -1;

    /* copy name to buffer and remove blanks */
    strncpy(buf, attrName, naLen);
    buf[naLen] = '\0';
    i = naLen - 1;
    while (i >= 0 && buf[i] == ' ') /*  not used isspace() : allow \t */
    {
        buf[i] = '\0';
        i--;
    }
    if (sendData((void *)&buf, 1024) != 1024)
        return -1;

    /* copy name to buffer and remove blanks */
    strncpy(buf, attrVal, vaLen);
    buf[vaLen] = '\0';
    i = vaLen - 1;
    while (i >= 0 && buf[i] == ' ') /*  not used isspace() : allow \t */
    {
        buf[i] = '\0';
        i--;
    }
    if (sendData((void *)&buf, 1024) != 1024)
        return -1;

    return 0;
}

int coAddAttribute(const char *portName,
                   const char *attrName,
                   const char *attrVal)
{
    char buf[1024];

    if (coSendC(ATTRIBUTE, portName))
        return -1;

    strncpy(buf, attrName, 1023);
    buf[1023] = '\0';
    if (sendData(buf, 1024) != 1024)
        return -1;

    strncpy(buf, attrVal, 1023);
    buf[1023] = '\0';
    if (sendData(buf, 1024) != 1024)
        return -1;

    return 0;
}

/* --------------------------------------------------------------*/
/* common code for cell and vertex mapping                       */
static int sendMapping(int type, int fortrani, int node, int length, const int *field)
{
    struct
    {
        int32 type, fortrani, node, length;
    } data;
    data.type = type;
    data.fortrani = fortrani; /* =1 for FORTRAN, =0 for C : for array indices */
    data.node = node;
    data.length = length;
    if (sendData((void *)&data, sizeof(data)) != sizeof(data))
        return -1;

    assert(sizeof(int) == sizeof(int32));
    if (sendData((void *)field, sizeof(int) * length) != sizeof(int) * length)
        return -1;
    else
        return 0;
}

/* Send a cell mapping local -> global               F77: COPACM */
int COPACM(int *node, int *numCells, int *localToGlobal)
{
    return sendMapping(PARA_CELL_MAP, 1, *node, *numCells, localToGlobal);
}

int coParallelCellMap(int node, int numCells, const int *localToGlobal)
{
    return sendMapping(PARA_CELL_MAP, 0, node, numCells, localToGlobal);
}

/* Send a vertex mapping local -> global             F77: COPAVM */
int COPAVM(int *node, int *numCells, int *localToGlobal)
{
    return sendMapping(PARA_VERTEX_MAP, 1, *node, *numCells, localToGlobal);
}

int coParallelVertexMap(int node, int numCells, const int *localToGlobal)
{
    return sendMapping(PARA_VERTEX_MAP, 0, node, numCells, localToGlobal);
}

/* --------------------------------------------------------------*/
/* Next data sent is from node #                     F77: COPANO */
int COPANO(int *node)
{
    return coParallelNode(*node);
}

int coParallelNode(int node)
{
    struct
    {
        int32 id;
        int32 node;
    } data;
    data.id = PARA_NODE;
    data.node = node;
    if (sendData((void *)&data, sizeof(data)) != sizeof(data))
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
    int size = *length;
    return sendData((const void *)data, size);
}

int sendData(const void *buffer, size_t length)
{
    register char *bptr = (char *)buffer;
    register int written;
    register int nbytes = length;

    if (coSimLibData.soc == -1)
    {
        fprintf(stderr, "socket is closed\n");
        return -1;
    }

    if (coSimLibData.verbose > 3)
        fprintf(stderr, "coSimClient sending %d Bytes to Socket %d\n",
                (int)length, coSimLibData.soc);

    while (nbytes > 0)
    {
        do
        {
#if !(defined(WIN32) || defined(WIN64))
            written = write(coSimLibData.soc, (void *)bptr, nbytes);
#else
            written = send(coSimLibData.soc, bptr, nbytes, 0);
#endif
#if defined(WIN32) || defined(WIN64)
        } while ((written < 0) && ((WSAGetLastError() == WSAEINPROGRESS) || (WSAGetLastError() == WSAEINTR) || (WSAGetLastError() == WSAEWOULDBLOCK)));
#else
        } while ((written < 0) && ((errno == EAGAIN) || (errno == EINTR)));
#endif
        if (written < 0)
        {
            fprintf(stderr, "coSimClient error: write returned %d\n", written);
            close(coSimLibData.soc);
            coSimLibData.soc = -1;
            return -1;
        }
        nbytes -= written;
        bptr += written;
        if (written == 0)
        {
            close(coSimLibData.soc);
            coSimLibData.soc = -1;
            return -1;
        }
    }
    if (coSimLibData.verbose > 3)
        fprintf(stderr, "coSimClient sent %d Bytes\n", (int)length);
    return length;
}

/****************************************************
 * Receive a certain amount of data from the module *    FORTRAN: CORECV
 ****************************************************/

int CORECV(int *data, int *length)
{
    int size = *length;
    return recvData((void *)data, size);
}

int CORRCV(float *data, int *length)
{
    int size = *length;
    return recvData((void *)data, size);
}

int recvData(void *buffer, size_t length)
{
    register char *bptr = (char *)buffer;
    register int nread;
    register int nbytes = length;

    if (coSimLibData.soc == -1)
    {
        memset(buffer, 0, length);
        return -1;
    }

    if (coSimLibData.verbose > 3)
        fprintf(stderr, " coSimClient waiting for %d Bytes from Socket %d\n",
                (int)length, coSimLibData.soc);

    while (nbytes > 0)
    {
        do
        {
#if !(defined(WIN32) || defined(WIN64))
            nread = read(coSimLibData.soc, (void *)bptr, nbytes);
#else
            nread = recv(coSimLibData.soc, bptr, nbytes, 0);
#endif
#if defined(WIN32) || defined(WIN64)
        } while ((nread < 0) && ((WSAGetLastError() == WSAEINPROGRESS) || (WSAGetLastError() == WSAEINTR) || (WSAGetLastError() == WSAEWOULDBLOCK)));
#else
        } while ((nread < 0) && ((errno == EAGAIN) || (errno == EINTR)));
#endif
        if (nread < 0)
        {
            fprintf(stderr, "coSimClient error: received %d Bytes\n", nread);
            close(coSimLibData.soc);
            coSimLibData.soc = -1;
            return -1;
        }
        nbytes -= nread;
        bptr += nread;
        if (nread == 0)
            break;
    }
    if (nbytes)
    {
        fprintf(stderr, " error: received 0 Bytes while %d left\n", nbytes);
        close(coSimLibData.soc);
        coSimLibData.soc = -1;
        return -1;
    }
    else
    {
        if (coSimLibData.verbose > 3)
            fprintf(stderr, "coSimClient received %d Bytes\n", (int)length);
        return length;
    }
}

/****************************************************************************/
static int openServer(int minPort, int maxPort)
{
    int port;
    struct sockaddr_in addr_in;

    /* open the socket: if not possible, return -1 */
    coSimLibData.serv = socket(AF_INET, SOCK_STREAM, 0);
    if (coSimLibData.serv < 0)
    {
        return -1;
    }

    port = minPort;

    /* Assign an address to this socket */
    memset((char *)&addr_in, 0, sizeof(addr_in));
    addr_in.sin_family = AF_INET;

    addr_in.sin_addr.s_addr = INADDR_ANY;
    addr_in.sin_port = htons(port);

    /* bind with changing port# until unused port found */
    while ((port <= maxPort) && (bind(coSimLibData.serv, (struct sockaddr *)&addr_in, sizeof(addr_in)) < 0))
    {
#ifndef _WIN32
        if (errno == EADDRINUSE) /* if port is used (UNIX) */
#else
        if (GetLastError() == WSAEADDRINUSE) /*                 (WIN32) */
#endif
        {
            port++; /* try next port */
            addr_in.sin_port = htons(port);
        }
        else /* other errors : ERROR, leave loop */
            port = maxPort + 1;
    }

    /* we didn't find an empty one OR could not bind */
    if (port > maxPort)
    {
        fprintf(stderr, "coSimClient: opening ports failed\n");
        close(coSimLibData.serv);
        coSimLibData.serv = -1;
        return -1;
    }
    else
    {
        fprintf(stderr, "coSimClient: listening on port %d\n", port);
        listen(coSimLibData.serv, 20);
        return port;
    }
}

/****************************************************************************/
static int openClient(unsigned long ip, int port, float timeout)
{
    int connectStatus = 0;
    int numConnectTries = 0;
    struct sockaddr_in s_addr_in;

    do
    {
        if (numConnectTries > 0)
        {
#if !(defined(WIN32) || defined(WIN64))
            sleep(1);
#else
            Sleep(1000);
#endif
        }

        /* open the socket: if not possible, return -1 */
        coSimLibData.soc = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (coSimLibData.soc < 0)
        {
            fprintf(stderr, "openClient() - socket() call failed\n");
            coSimLibData.soc = -1;
            return -1;
        }

        /* set s_addr structure */
        s_addr_in.sin_addr.s_addr = ip;
        s_addr_in.sin_port = htons(port);
        s_addr_in.sin_family = AF_INET;

        /* Try connecting */
        connectStatus = connect(coSimLibData.soc, (struct sockaddr *)&s_addr_in, sizeof(s_addr_in));
        if (connectStatus == -1)
        {
            fprintf(stderr, "openClient() - connect() call failed: %s\n", strerror(errno));
        }
        /* didn't connect */
        if (connectStatus < 0)
            numConnectTries++;
    } while ((connectStatus < 0) && (numConnectTries <= ((int)timeout)));

    if (connectStatus == 0)
        return port;
    else
        return -1;
}

/****************************************************************************/
static int acceptServer(float wait)
{
    int tmp_soc;
    struct timeval timeout;
    fd_set fdread;
    int i;
    struct sockaddr_in s_addr_in;
#if !(defined(WIN32) || defined(WIN64) || defined(_SX))
    socklen_t length;
#else
    unsigned int length;
#endif

    /* prepare for select(2) call and wait for incoming connection */
    timeout.tv_sec = (int)wait;
    timeout.tv_usec = (int)((wait - timeout.tv_sec) * 1000000);
    FD_ZERO(&fdread);
    FD_SET(coSimLibData.serv, &fdread);
    if (wait >= 0)
    { /* wait period was specified */
        i = select(coSimLibData.serv + 1, &fdread, NULL, NULL, &timeout);
    }
    else
    { /* wait infinitly */
        i = select(coSimLibData.serv + 1, &fdread, NULL, NULL, NULL);
    }

    if (i == 0) /* nothing happened: return -1 */
    {
        close(coSimLibData.soc);
        coSimLibData.soc = -1;
        return -1;
    }

    /* now accepting the connection */
    length = sizeof(s_addr_in);
    tmp_soc = accept(coSimLibData.serv, (struct sockaddr *)&s_addr_in, &length);
    if (tmp_soc < 0)
    {
        close(coSimLibData.soc);
        coSimLibData.soc = -1;
        return -1;
    }

#ifdef DEBUG
    fprintf(stderr, "Connection from host %s, port %u\n",
            Xinet_ntoa(s_addr_in.sin_addr), ntohs(s_addr_in.sin_port));
    fflush(stdout);
#endif

    close(coSimLibData.soc);

    /* use the socket 'accept' delivered */
    coSimLibData.soc = tmp_soc;

    fprintf(stderr, "new coSimLibData.soc: %d\n", coSimLibData.soc);

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

/* read grid dimensions from covise */
int COGDIM(int *npoin_ges, int *nelem_ges, int *knmax_num, int *elmax_num,
           int *npoin_geb, int *nelem_geb)
{

    int data[2], r;
    int dim[9];
    r = recvData(data, 2 * sizeof(int));

    if (data[0] != GEO_DIM)
    {
        fprintf(stderr, "Data sent from COVISE is of wrong type %d, expected type %d (GEO_DIM)\n", data[0], GEO_DIM);
        return -1;
    }
    if (data[1] != 36)
    {
        fprintf(stderr, "Grid dimensions sent from COVISE has wrong length %d, expected length 36\n", data[1]);
        return -1;
    }

    recvData(dim, 9 * sizeof(int));
    *npoin_ges = dim[1];
    *nelem_ges = dim[2];
    *knmax_num = dim[3];
    *elmax_num = dim[4];
    *npoin_geb = dim[5];
    *nelem_geb = dim[6];
    return 0;
}

/* read boco dimensions from covise */
int COBDIM(int *nrbpoi_geb, int *nwand_geb, int *npres_geb, int *nsyme_geb,
           int *nconv_geb)
{

    int data[2], r;
    int dim[6];
    r = recvData(data, 2 * sizeof(int));

    if (data[0] != BOCO_DIM)
    {
        fprintf(stderr, "Data sent from COVISE is of wrong type %d, expected type %d (BOCO_DIM)\n", data[0], BOCO_DIM);
        return -1;
    }
    if (data[1] != 24)
    {
        fprintf(stderr, "Boco dimensions sent from COVISE has wrong length %d, expected length 24\n", data[1]);
        return -1;
    }

    recvData(dim, 6 * sizeof(int));
    *nrbpoi_geb = dim[0];
    *nwand_geb = dim[1];
    *npres_geb = dim[2];
    *nsyme_geb = dim[3];
    *nconv_geb = dim[4];
    return 0;
}

int CORGEO()
{

    int data[2], r;
    r = recvData(data, 2 * sizeof(int));

    if (data[0] != SEND_GEO)
    {
        fprintf(stderr, "Data sent from COVISE is of wrong type %d, expected type %d (SEND_GEO)\n", data[0], SEND_GEO);
        return -1;
    }

    return 0;
};

int COSLEP(int time)
{
    return coSleep(time);
}

int coSleep(int time)
{
#if !(defined(WIN32) || defined(WIN64))
    sleep(time);
#else
    Sleep(1000 * time);
#endif
    return 0;
}
