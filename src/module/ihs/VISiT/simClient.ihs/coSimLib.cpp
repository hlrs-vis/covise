#include "covise_securestring.h"
#include <sys/time.h>
#include <unistd.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <ctype.h>

#if defined(__linux) || defined(__hpux) || defined(__sun)
#include <string.h>
#else
#include <bstring.h>
#endif

#include "coSimLib.h"

/// Copy-Constructor: NOT  IMPLEMENTED
coSimLib::coSimLib(const coSimLib &)
{ assert(1); }

/// Assignment operator: NOT  IMPLEMENTED
coSimLib &coSimLib::operator =(const coSimLib &)
{ assert(1); return *this; }

/// Default constructor: NOT  IMPLEMENTED
coSimLib::coSimLib()
{ assert(1); }

////////////////////////////////////////////////////////////////////
/// reset all member fields for startup and re-start
void coSimLib::resetSimLib()
{

   if (d_socket>1)
   {
      close(d_socket);
      removeSocket(d_socket);
   }
   d_socket = -1;

   // if we had user args : erase it
   int i;
   for (i=0;i<10;i++)
   {
      delete [] d_userArg[i];
      d_userArg[i]=NULL;
   }

   // no command pending
   d_command = 0;

   // Parallel distribution Maps
   if (d_cellMap)
   {
      for (i=0;i<d_numNodes;i++)
         delete [] d_cellMap[i];
      delete [] d_cellMap;
      d_cellMap  = NULL;
   }
   d_numCells = 0;

   if (d_vertMap)
   {
      for (i=0;i<d_numNodes;i++)
         delete [] d_vertMap[i];
      delete [] d_vertMap;
      d_vertMap = NULL;
   }
   d_numVert = 0;

   // we are not parallel ... ANY MORE
   d_numNodes = 0;

   // kill all members of the port list after the dummy
   PortListElem *portPtr = d_portList->next;
   while (portPtr)
   {
      PortListElem *nextPort = portPtr->next;
      delete portPtr;
      portPtr = nextPort;
   }
   d_portList->next = NULL;

   // not yet parallel - no active number
   d_actNode = -1;

   // sim hasn't requested exec yet
   d_simExec = 0;
}


////////////////////////////////////////////////////////////////////
/// constructor
coSimLib::coSimLib(const char *name, const char *desc)
:coModule(desc)
{
   // not yet connected
   d_socket = -1;

   // save then name
   d_name = strcpy(new char[strlen(name)+1],name);

   const char *confData;
   int i;

   // no user-defined arguments so far
   for (i=0;i<10;i++)
      d_userArg[i]=NULL;

   // typically we use the default interface of the machine,
   // but user may specify a different, e.g. for routing reasons
   char buffer[128];
   d_localIP=0;
   confData = CoviseConfig::getScopeEntry(d_name,"LOCAL");
   if (confData)
   {
      d_localIP = nslookup(confData);
      if (!d_localIP)
         cerr << "did not find host '" << confData
            << "'specified in covise.config, using hostname instead"
            << endl;
   }

   // either no LOCAL string or host not found
   if (!d_localIP)
   {
      gethostname(buffer,128);
      buffer[127]='\0';                           // just in case... ;-)
      confData = buffer;
      d_localIP     = nslookup(buffer);
   }

   // still nothing fond -> misconfigured system
   if (!d_localIP)
   {
      cerr << "Mis-configured system, could not find IP of '"
         << buffer << "', the configured hostname,"
         << endl;
   }

   // and, if user doesn't change it, we start the simulation locally
   d_targetIP    = nslookup("localhost");

   if (d_localIP==0 || d_targetIP==0)
   {
      d_socket=1;
      return;
   }

   // get port numbers
   const char *ports = CoviseConfig::getScopeEntry(d_name,"PORTS");
   if (!ports)
      ports="31000 31999";                        // default covise ports
   sscanf(ports,"%d %d",&d_minPort,&d_maxPort);

   // no command pending
   d_command = 0;

   // we are not parallel ... YET
   d_numNodes = 0;

   // but we declare a dummy port for easier searching
   static const char *empty = "";
   d_portList = new PortListElem;
   d_portList->name     = empty;
   d_portList->map      = NULL;
   d_portList->numParts = 0;
   d_portList->openObj  = NULL;
   d_portList->next     = NULL;

   // we do not know yet, how big the fields will become
   d_numCells = d_numVert = 0;

   // analyse all STARTUP lines
   const char **entry;
   const char **ePtr;
   entry = CoviseConfig::getScopeEntries(name,"STARTUP");

   ePtr = entry;
   d_numStartup = 0;
   while (ePtr && *ePtr)
   {
      d_numStartup++;
      ePtr++;
   }
   d_numStartup /= 2;

   // copy all STARTUP lines into buffers
   if (d_numStartup)
   {
      d_startup_line  = new const char *[d_numStartup];
      d_startup_label = new char       *[d_numStartup];
      for (i=0;i<d_numStartup;i++)
      {
         // skip leading blanks in field
         const char *actEntry = entry[2*i+1];
         while (*actEntry && isspace(*actEntry))
            actEntry++;

         // copy complete string to names field
         d_startup_label[i] = new char[ strlen(actEntry)+1 ];
         strcpy(d_startup_label[i],actEntry);
         char *cPtr = d_startup_label[i];

         // find next space
         while (*cPtr && !isspace(*cPtr))
            cPtr++;

         // terminate first string here: if this is only a label,
         // correctly terminate the 'line' string
         if (*cPtr)
         {
            *cPtr = '\0';
            cPtr++;
            while (*cPtr && isspace(*cPtr))       // skip spaces
               cPtr++;
            d_startup_line[i] = cPtr;

         }
         else
         {
            cerr << "only label, but no text in startup line:\n    \""
               << entry[2*i+1] << '"' << endl;
            d_startup_line[i] = actEntry;
         }
      }
   }
   else
   {
      static const char *dummy[1] = { "echo \"startup sequence not specified\"" };

      d_startup_label = new char*[1];
      d_startup_label[0] = strcpy( new char[9] , "no label");
      d_startup_line  = dummy;
      d_numStartup    = 1;
   }

   p_StartupSwitch = addChoiceParam("Startup","Switch startup messages");
   p_StartupSwitch->setValue(d_numStartup,d_startup_label,1);

   // sim hasn't requested exec yet
   d_simExec = 0;
}


// destructor
coSimLib::~coSimLib()
{
   int i;
   for (i=0;i<10;i++)
      delete [] d_userArg[i];
}


// start the user's application
int coSimLib::startSim()
{
   // sim hasn't requested exec yet
   d_simExec = 0;

   const char *confData;

   // Check, who is server: default is Module
   confData = CoviseConfig::getScopeEntry(d_name,"SERVER");
   int modIsServer = (!confData || *confData=='M' || *confData=='m');

   // now get the timeout, default = 1min
   float timeout=60.0;
   confData=CoviseConfig::getScopeEntry(d_name,"TIMEOUT");
   if (confData)
   {
      sscanf(confData,"%f",&timeout);
      //cerr << " Timeout: " << timeout << endl;
   }

   // get the verbose level, default = 0
   d_verbose=0;
   confData=CoviseConfig::getScopeEntry(d_name,"VERBOSE");
   if (confData)
   {
      sscanf(confData,"%d",&d_verbose);
      //cerr << " Verbose: level=" << d_verbose << endl;
      switch (d_verbose)
      {
         case 4:  cerr << "VERBOSE  - Log all binary read/write" << endl;
         case 3:  cerr << "VERBOSE  - Log data creation details" << endl;
         case 2:  cerr << "VERBOSE  - Write Mapping files" << endl;
         case 1:  cerr << "VERBOSE  - Protocol Object creations and Port requests" << endl;
      }
   }

   // if we are a server, start the server
   if (modIsServer)
      if (openServer())
   {
      d_socket=1;
      return -1;
   }

   // if we don't have the start line, forget it...
   //const char *configLine=d_config->get_scope_entry(d_name,"STARTUP");
   const char *configLine = d_startup_line[ p_StartupSwitch->getValue() ];
   if (!configLine)
   {
      sendError("Could not find section %s with STARTUP line in covise.config",
         d_name);
      d_socket=1;
      return -1;
   }
   // we need an additional '\0' after the termination: makes treating
   // a '%' as the last char easier
   char *startLine = strcpy(new char[strlen(configLine)+2],configLine);
   startLine[strlen(configLine)+1]='\0';

   if (d_verbose>0)
      cerr << "Startup Line: " << startLine << endl;

   // build the CO_SIMLIB_CONN variable
   char envVar[64];
   if (modIsServer)
      sprintf(envVar,"C:%s/%d,%f,%d",inet_ntoa(*(in_addr*)&d_localIP),
         d_usePort,timeout,d_verbose);
   else
      sprintf(envVar,"S:%d-%d,%f,%d",d_minPort,d_maxPort,timeout,d_verbose);

   //cerr << "Env var: " << envVar << endl;

   // now: build the command line
   char command[4096];
   strcpy(command, "( ");
   char *startPtr=startLine;

   char *nextTok = strchr(startLine,'%');
   while (nextTok)
   {
      //cerr << "NextTok = " << nextTok << endl;
      // copy everything before the '%'
      *nextTok = '\0';
      strcat(command,startPtr);
      nextTok++;
      switch (*nextTok)
      {
         case '%' :  strcat(command,"%");                                break;
         case 'e' :  strcat(command,envVar);                             break;
         case 'h' :  strcat(command,inet_ntoa(*(in_addr*)&d_targetIP));  break;
         case '0' :  if (d_userArg[0]) strcat(command,d_userArg[0]);     break;
         case '1' :  if (d_userArg[1]) strcat(command,d_userArg[1]);     break;
         case '2' :  if (d_userArg[2]) strcat(command,d_userArg[2]);     break;
         case '3' :  if (d_userArg[3]) strcat(command,d_userArg[3]);     break;
         case '4' :  if (d_userArg[4]) strcat(command,d_userArg[4]);     break;
         case '5' :  if (d_userArg[5]) strcat(command,d_userArg[5]);     break;
         case '6' :  if (d_userArg[6]) strcat(command,d_userArg[6]);     break;
         case '7' :  if (d_userArg[7]) strcat(command,d_userArg[7]);     break;
         case '8' :  if (d_userArg[8]) strcat(command,d_userArg[8]);     break;
         case '9' :  if (d_userArg[9]) strcat(command,d_userArg[9]);     break;
      }

      startPtr = nextTok+1;
      nextTok  = strchr(startPtr,'%');
   }

   // copy the rest
   strcat(command,startPtr);

   // the line is nearly ready, just make sure we start into background
   strcat(command, " ) &");

   sendInfo("Starting simulation: '%s'",command);

   delete [] startLine;

   // now ..... GO!
   if (system(command))
   {
      ::close(d_socket);
      d_socket = -1;
      return -1;
   }

   // if we are a server, accept, else start the client connection
   if (modIsServer)
   {
      if (acceptServer(timeout))
      {
         d_socket = -1;
         return -1;
      }
   }
   else
   {
      if (openClient())
      {
         d_socket = 1;
         return -1;
      }
   }

   // now: register this port at coModule
   addSocket(d_socket);

   return 0;

}


/// set target host: return -1 on error
int coSimLib::setTargetHost(const char *hostname)
{
   d_targetIP    = nslookup(hostname);
   if (d_targetIP)
      return 0;
   else
      return -1;

}


/// set local host: return -1 on error
int coSimLib::setLocalHost(const char *hostname)
{
   d_localIP    = nslookup(hostname);
   if (d_localIP)
      return 0;
   else
      return -1;

}


/// set a user startup argument
int coSimLib::setUserArg(int num, const char *data)
{
   // max. 10 user arguments
   if ( (num<0) || (num>9) || (!data) ) return -1;

   // delete old and set new arg
   delete [] d_userArg[num];
   d_userArg[num] = strcpy(new char[strlen(data)+1],data);

   if (d_verbose>0)
      cerr << "Set user[" << num << "] = '" << data << "'" << endl;

   return 0;
}


////////////////////////////////////////////////////////////////////////////////

int coSimLib::serverMode()
{
   // handles data now - serverMode is always in compute() CB
   d_simExec = 0;

   if (d_socket<0) return -1;

   int res;

   // read commands until we get a QUIT (with return) or run into trouble
   do
   {
      if ( !d_command
         && recvData((void*)&d_command,sizeof(d_command)) != sizeof(d_command)
         )
         return -1;

      res = handleCommand(COMPUTE);

   }
   while ( res!=QUIT && res!=ERROR );

   if (res==QUIT)
      return 0;
   else
      return -1;
}


////////////////////////////////////////////////////////////////////////
//// Utility functions
////////////////////////////////////////////////////////////////////////

///// Nameserver request

#ifdef _CRAY
#define CONV (char *)
#else
#define CONV
#endif

uint32_t coSimLib::nslookup(const char *name)
{
   // try whether this is already ###.###.###.### IP adress
   unsigned long addr = inet_addr(name);

   if (addr && addr!= INADDR_NONE)
      return (uint32_t) addr;

   // not yet a numerical adress, try a nameserver look-up
   struct hostent *hostinfo;
   hostinfo=gethostbyname( CONV name);            /* Hack for Cray */
   if (hostinfo)
   {
#ifndef _CRAY
      return *(uint32_t *)*hostinfo->h_addr_list;
#else
      unsigned char *x = (unsigned char *) *hostinfo->h_addr_list;
      return  ((*x)<<24) | (*(x+1)<<16) | (*(x+2)<<8) | *(x+3) ;
#endif
   }
   else
      sendError("Could find IP adress for hostname '%s'",name);

   return 0;

}


///// Open a TCP server

int coSimLib::openServer()
{
   int port;

   // open the socket: if not possible, return -1
   d_socket = socket(AF_INET, SOCK_STREAM, 0);
   if (d_socket < 0)
   {
      d_socket=-1;
      return -1;
   }

   // Find a port to start with
   port=d_minPort;

   // Assign an address to this socket
   struct sockaddr_in addr_in;
   memset((char *)&addr_in, 0, sizeof(addr_in));
   addr_in.sin_family = AF_INET;

   addr_in.sin_addr.s_addr = INADDR_ANY;
   addr_in.sin_port = htons(port);

   // bind with changing port# until unused port found
   while ( (port<=d_maxPort)
      && (bind(d_socket,(sockaddr*)&addr_in,sizeof(addr_in)) < 0)
      )
   {
#ifndef _WIN32
      if (errno == EADDRINUSE)                    // if port is used (UNIX)
#else
         if (GetLastError() == WSAEADDRINUSE)     //                 (WIN32)
#endif
      {
         port++;                               // try next port
         addr_in.sin_port = htons(port);
      }
      else                                        // other errors : ERROR, leave loop
         port=d_maxPort+1;
   }

   // we didn't find an empty one OR could not bind
   if (port>d_maxPort)
   {
      ::close(d_socket);
      d_socket = -1;
      return -1;
   }
   else
   {
      ::listen(d_socket, 5);                      // start listening
      d_usePort = port;
      return 0;
   }
}


///// Open a TCP client ////////////////////////////////////////////////////

int coSimLib::openClient()
{
   int connectStatus=0;
   int numConnectTries = 0;
   int port=d_minPort;
   do
   {
      // open the socket: if not possible, return -1
      d_socket = socket(AF_INET, SOCK_STREAM, 0);
      if (d_socket < 0)
      {
         d_socket=-1;
         return -1;
      }

      // set s_addr structure
      struct sockaddr_in s_addr_in;
                                                  // inet_addr delivers
      s_addr_in.sin_addr.s_addr = htonl(d_targetIP);
      s_addr_in.sin_port = htons(port);           // network byte order
      s_addr_in.sin_family = AF_INET;

      // Try connecting
      connectStatus=connect(d_socket,(sockaddr*)&s_addr_in,sizeof(s_addr_in));

      // didn't connect
      if (connectStatus<0)                        // -> next port
      {
         port++;

         if (port>d_maxPort)                      // last Port failed -> wait & start over
         {
            port=d_minPort;
            numConnectTries++;
            sleep(2);
         }
      }
   }
                                                  // try 5 rounds
   while ( (connectStatus<0) && (numConnectTries<=5) );

   if (connectStatus==0)
   {
      return 0;
   }
   else
   {
      d_socket=1;
      return -1;
   }
}


int coSimLib::acceptServer(float wait)
{
   int tmp_soc;
   struct timeval timeout;
   fd_set fdread;
   int i;

   // prepare for select(2) call and wait for incoming connection
   timeout.tv_sec = (int)wait;
   timeout.tv_usec = (int)((wait-timeout.tv_sec)*1000000);
   FD_ZERO(&fdread);
   FD_SET(d_socket, &fdread);
   if (wait>=0)                                   // wait period was specified
      i=select(d_socket+1, &fdread, NULL, NULL, &timeout);
   else                                           // wait infinitly
      i=select(d_socket+1, &fdread, NULL, NULL, NULL);

   if (i == 0)                                    // nothing happened: return -1
   {
      ::close(d_socket);
      d_socket=-1;
      return -1;
   }

   // now accepting the connection
   struct sockaddr_in s_addr_in;

#ifdef CO_linux
   size_t length;
#else
   int length;
#endif

   length = sizeof(s_addr_in);
   tmp_soc = accept(d_socket, (sockaddr *)&s_addr_in, &length);

   if (tmp_soc < 0)
   {
      ::close(d_socket);
      d_socket=-1;
      return -1;
   }

   ::close(d_socket);

   // use the socket 'accept' delivered
   d_socket = tmp_soc;

   if (d_verbose>0)
      cerr << "Accepted connection from "
         << inet_ntoa(s_addr_in.sin_addr)
         << " to socket " << d_socket
         << endl;
   return 0;
}


/***************************************************
 * Send a certain amount of data to the simulation *
 ***************************************************/
int coSimLib::sendData(const void *buffer, size_t _length)
{
   unsigned long length = _length;                // make 64-bit proof printf
   register char *bptr=(char *)buffer;
   register int written;
   register int nbytes=length;
   if (d_verbose>3)
      fprintf(stderr,"coSimLib sending %ld Bytes to Socket %d\n",
         length, d_socket);

   while (nbytes>0)
   {
      written = write(d_socket,(void *) bptr,nbytes);
      if (written < 0)
      {
         fprintf(stderr,"coSimLib error: write returned %d\n",written);
         return -1;
      }
      nbytes-=written;
      bptr+=written;
      if (written==0) return -2;
   }
   if (d_verbose>3)
      fprintf(stderr,"coSimLib sent %ld Bytes\n",length);
   return length;
}


/********************************************************
 * Receive a certain amount of data from the simulation *
 ********************************************************/

int coSimLib::recvData(void *buffer, size_t _length)
{
   unsigned long length = _length;                // make 64-bit proof printf
   register char *bptr=(char*) buffer;
   register int nread;
   register int nbytes=length;
   if (d_verbose>3)
      fprintf(stderr," coSimLib waiting for %ld Bytes from Socket %d\n",
         length,d_socket);

   while (nbytes>0)
   {
      nread = read(d_socket, (void*)bptr, nbytes);
      if (nread < 0)
      {
         fprintf(stderr,"coSimLib error: received %d Bytes\n",nread);
         return -1;
      }
      nbytes-=nread;
      bptr+=nread;
      if (nread==0) break;
   }
   if (nbytes)
   {
      fprintf(stderr,"coSimLib error: received 0 Bytes while %d left\n",nbytes);
      sleep(1);
      return -2;
   }
   else
   {
      if (d_verbose>3)
         fprintf(stderr,"coSimLib received %ld Bytes\n",length);
      return length;
   }
}


///// handle events sent outside ServerMode

void coSimLib::sockData(int sockNo)
{
   // if we have another command in the queue: just return and do nothing
   if (d_command)
      return;

   // if this is not our socket: throw message and return
   if (sockNo!=d_socket)
   {
      Covise::sendError("Overloading of sockData() not allowed in coSimLib");
      return;
   }

   // receive the command ID from the socket
   int recvSize=recvData((void*)&d_command,sizeof(d_command));

   // if we get -1 here, something went wrong
   if (recvSize<0)
   {
      sendError("Simulation socket crashed: closed connection");
      close(d_socket);
      d_socket=-1;
   }

   if ( recvSize < sizeof(d_command))
      return;

   switch (d_command)
   {
      // commands handled here
      case EXEC_COVISE:
         d_simExec = 1;                           // set flag that sim requested exec

      case TEST:
      case GET_SLI_PARA:                          // handle immediate here
      case GET_SC_PARA_FLO:                       // allow user parameter request
      case GET_SC_PARA_INT:
      case GET_CHOICE_PARA:
      case GET_BOOL_PARA:
      case GET_TEXT_PARA:

         handleCommand(MAIN_LOOP);
         break;

   }

   // do NOT call selfExec() when receiving data - loops when the
   // application has called COEXEC before

   // All other commands ignored here.

   return;
}


///// handle all kinds of commands from the simulation
///// caller must check, whether we are allowed to do this NOW

///// return last command, set active command to 0
int coSimLib::handleCommand(int fromWhere)
{
   // memorise the last command, we have to return it...
   CommandType actComm = (CommandType)d_command;
   d_command = NONE;

   /// we re-use this buffer frequently...
   char buffer[128];

   // common segments for some Command types
   switch (actComm)
   {
      // all these send a portname first...
      case GET_SLI_PARA:
      case GET_SC_PARA_FLO:
      case GET_SC_PARA_INT:
      case GET_CHOICE_PARA:
      case GET_BOOL_PARA:
      case GET_TEXT_PARA:
      case SEND_USG:
      case SEND_1DATA:
      case SEND_3DATA:
      case PARA_PORT:
      case ATTRIBUTE:
      {
         // receive Parameter name
         if (recvData((void*)buffer,64) != 64 ) return -1;
         break;
      }
   }

   switch (actComm)
   {
      // ###########################################################
      // TEST is always simply ignored
      // ###########################################################
      case TEST:
         if (d_verbose>1)
         {
            cerr << "coSimLib Client called TEST" << endl;
         }
         break;

         // ###########################################################
         // EXEC sends a callback message : only set self-exec flag
         // ###########################################################
      case EXEC_COVISE:
      {
         if (d_verbose>0)
         {
            cerr << "coSimLib Client called EXEC" << endl;
         }
         selfExec();
         break;
      }

      case ATTRIBUTE:
      {
         char name[1024],value[1024];

         // receive the content
         if (recvData(name,1024) != 1024 ) return -1;
         if (recvData(value,1024) != 1024 ) return -1;

         if (d_verbose>0)
         {
            cerr << "coSimLib Client called ATTRIBUTE" << endl;
            cerr << "   port: '" << buffer << "'" << endl;
            cerr << "   name: '" << name   << "'" << endl;
            cerr << "   val : '" << value  << "'" << endl;
         }

         // ###########################################################
         // we need the port to get the data object

         coUifElem *portElem = findElem(buffer);

         if (    !portElem
            || portElem->kind() != coUifElem::OUTPORT)
         {
            sendWarning("Simulation sent attribs for '%s': not an output data port",
               buffer);
            break;
         }

         coOutputPort *outPort     = (coOutputPort *)portElem;
         coDistributedObject *obj = outPort->getCurrentObject();

         if (obj)
            obj->addAttribute(name,value);
         else
         {
            sendWarning("Simulation sent attribs for '%s': no object at port",
               buffer);
            break;
         }
         break;
      }

      // ###########################################################
      // Slider Parameter request
      // ###########################################################
      case GET_TEXT_PARA:
      {
         if (d_verbose>0)
         {
            cerr << "coSimLib Client called GET_TEXT_PARA" << endl;
         }
         // get this parameter
         char res[256];

         coUifElem *para = findElem(buffer);
         if ( (para) && (para->kind() == coUifPara::PARAM)
            && (((coUifPara*)para)->isOfType(coStringParam::getType()))
            )
         {
            const char *val = ((coStringParam*)para) -> getValue();
            strncpy(res,val,255);
            res[255]='\0';
            if (strlen(val)>255)
               sendWarning("coSimLib: Truncated parameter %s when sending",buffer);
         }

         // send the answer back to the client
         if ( sendData(res,256) != 256 )    return -1;
         break;
      }

      // ###########################################################
      // Slider Parameter request
      // ###########################################################
      case GET_SLI_PARA :
      {
         if (d_verbose>0)
         {
            cerr << "coSimLib Client called GET_SLI_PARA" << endl;
         }
         // get this parameter
         struct { float min,max,value ; int32 error; }
         ret;

         coUifElem *para = findElem(buffer);
         if ( (para) && (para->kind() == coUifPara::PARAM)
            && (((coUifPara*)para)->isOfType(coFloatSliderParam::getType()))
            )
         {
            ((coFloatSliderParam*)para) -> getValue(ret.min,ret.max,ret.value);
            ret.error=0;
         }
         else
            ret.error=-1;

         // send the answer back to the client
         if ( sendData((void*)&ret,sizeof(ret)) !=sizeof(ret) )    return -1;
         break;
      }
      // ###########################################################
      // Float scalar Parameter request
      // ###########################################################
      case GET_SC_PARA_FLO :
      {
         if (d_verbose>0)
         {
            cerr << "coSimLib Client called GET_SC_PARA_FLO" << endl;
         }
         // get this parameter
         struct { float val ; int32 error; }
         ret;

         coUifElem *para = findElem(buffer);
         if ( (para) && (para->kind() == coUifPara::PARAM)
            && (((coUifPara*)para)->isOfType(coFloatParam::getType()))
            )
         {
            ret.val = ((coFloatParam*)para) -> getValue();
            ret.error=0;
         }
         else
            ret.error=-1;

         // send the answer back to the client
         if ( sendData((void*)&ret,sizeof(ret)) !=sizeof(ret) )    return -1;
         break;
      }

      // ###########################################################
      // All these get exactly one integer, just the appearance is different
      // ###########################################################
      case GET_SC_PARA_INT :
      case GET_CHOICE_PARA :
      case GET_BOOL_PARA :
      {
         // get this parameter
         struct { int32 val ; int32 error; }
         ret;

         coUifElem *para = findElem(buffer);
         if ( (para) && (para->kind() == coUifPara::PARAM) )
         {
            if (    (actComm==GET_SC_PARA_INT)
               && ((coUifPara*)para)->isOfType(coIntScalarParam::getType()) )
            {
               if (d_verbose>0)
               {
                  cerr << "coSimLib Client called GET_SC_PARA_INT" << endl;
               }
               ret.val = ((coIntScalarParam*)para) -> getValue();
               ret.error=0;
            }
            else if (    (actComm==GET_CHOICE_PARA)
               && ((coUifPara*)para)->isOfType(coChoiceParam::getType()) )
            {
               if (d_verbose>0)
               {
                  cerr << "coSimLib Client called GET_CHOICE_PARA" << endl;
               }
               ret.val = ((coChoiceParam*)para) -> getValue();
               ret.error=0;
            }
            else if (    (actComm==GET_BOOL_PARA)
               && ((coUifPara*)para)->isOfType(coBooleanParam::getType()) )
            {
               if (d_verbose>0)
               {
                  cerr << "coSimLib Client called GET_BOOL_PARA" << endl;
               }
               ret.val = ((coBooleanParam*)para) -> getValue();
               ret.error=0;
            }
            else
               ret.error=-1;
         }
         else
            ret.error=-1;
         // send the answer back to the client
         if ( sendData((void*)&ret,sizeof(ret)) !=sizeof(ret) )    return -1;
         break;
      }

      // ###########################################################
      //  Client creates 1D or 3D data field
      // ###########################################################
      case SEND_1DATA :
      case SEND_3DATA :
      {
         // number of components in the data set
         int numComp = (actComm==SEND_1DATA) ? 1 : 3;

         if (d_verbose>0)
         {
            cerr << "coSimLib Client called SEND_" << numComp << "DATA" << endl;
         }
         // Check: we MUST be in the compute callback now
         if (fromWhere!=COMPUTE)
         {
            Covise::sendError("SimLib: Only send data in Compute callback !!!");
            break;
         }

         int32 length;
         float *dataPtr[3];

         // receive length
         if (recvData((void*)&length,sizeof(int32)) != sizeof(int32) )
            return -1;

         if (d_verbose>0)
         {
            cerr << "  : Length = " << length << endl;
         }

         // ###########################################################
         // parallel applications may have non-parallel data...

         PortListElem *port = NULL;
         if (d_numNodes)
         {
            // find the port's record
            port = d_portList->next;              // skip dummy
            while (port && (strcmp(port->name,buffer)) )
               port = port->next;
         }

         // ###########################################################
         // we need to use the ports if we want to access the data at
         // the port after leaving serverMode()

         coUifElem *portElem = findElem(buffer);
         coOutputPort *outPort = (coOutputPort *)portElem;

         if (!portElem || portElem->kind() != coUifElem::OUTPORT)
         {
            sendError("Simulation sent data for %s: not an output data port",
               buffer);
            // receive to dummy field
            float *dummy = new float[length*numComp];

            // read node data into temporary field and discard
            //                                    -> keep message queue
            int local = length*numComp*sizeof(float);
            recvData((void*)dummy,local);
            delete [] dummy;
            return -1;
         }

         // ###########################################################
         // ok, this is really parallel : distributed data collection
         if (port)
         {
            // set our translation table to an easier pointer and check
            int *global = (*(port->map))[d_actNode];
            if (!global)
            {
               Covise::sendError("No translation map for node found");
               return -1;
            }

            if (d_verbose>2)
            {
               cerr << "Recv Data from node " << d_actNode << endl;
               cerr << "Conversion table starts : "
                  << global[0] << "," << global[1] << "," << global[2] << endl;
            }
            // if we haven't created an object right now, we do it
            int globLen = ( port->map == &d_cellMap) ? d_numCells : d_numVert;
            char *name;
            if (!port->openObj)
               name = get_object_name(buffer);

            if (numComp==1)
            {
               coDoFloat *data;
               if (!port->openObj)
               {
                  data = new coDoFloat(name,globLen);
                  if (d_verbose>2)
                  {
                     cerr << "Open Scalar data object '" << name
                        << "' for " << globLen << " Elem" << endl;
                  }
               }
               else
                  data = (coDoFloat*) port->openObj;

               data->get_adress(&dataPtr[0]);
               port->openObj = data;
               outPort->setCurrentObject(data);             // assign non-ready obj to port...
            }
            else
            {
               coDoVec3 *data;
               if (!port->openObj)
               {
                  data = new coDoVec3(name,globLen);
                  if (d_verbose>2)
                  {
                     cerr << "Open Vector data object '" << name
                        << "' for "
                        << globLen << " Elem" << endl;
                  }
               }
               else
                  data = (coDoVec3*) port->openObj;

               data->get_adresses(&dataPtr[0],&dataPtr[1],&dataPtr[2]);
               port->openObj = data;

               outPort->setCurrentObject(data);             // assign non-ready obj to port...
            }

            // receive to dummy field now @@@@@@@@@@ do better !!
            int fieldNo;
            float *dummy = new float[length*numComp];

            // read node data into temporary field
            int local = length*numComp*sizeof(float);
            if (recvData((void*)dummy,local) != local )
               return -1;

            if (d_verbose>2)
            {
               cerr << "received data: " << length << " values"
                  << dummy[0] << "," << dummy[1] << "," << dummy[2] << endl;
            }

            float *dPtr = dummy;
            // sort into global field

            // sort into global array
            for (fieldNo=0 ; fieldNo<numComp ; fieldNo++)
            {
               for (local=0;local<length;local++)
               {
                  // just make sure indexing table is ok...
                  if ( global[local]>=globLen || global[local]<0 )
                     cerr << "@@@@@@@@ illegal: accessing field "
                        << global[local] << " on field size "
                        << globLen << endl;
                  else
                     dataPtr[fieldNo][global[local]] = *dPtr;
                  dPtr++;

               }
            }
            delete [] dummy;
         }

         // ###########################################################
         // if we are not parallel, it is easy
         else
         {
            // create new data object
            char *name = get_object_name(buffer);

            if (d_verbose>1)
            {
               cerr << "  : non-MPP for Port " << buffer
                  << " -> " << name << endl;
            }

            if (!name)
            {
               cerr << "Could not create object name for port '"
                  << buffer << "'" << endl;
               break;
            }

            coDistributedObject *distrObj;

            if (numComp==1)
            {
               if (d_verbose>2)
               {
                  cerr << "Open Scalar data object '" << name
                     << "' for " << length << " Elem (non-parallel)" << endl;
               }
               coDoFloat *data
                  =  new coDoFloat(name,length);
               data->get_adress(&dataPtr[0]);
               distrObj = data;
            }
            else
            {
               if (d_verbose>2)
               {
                  cerr << "Open Vector data object '" << name
                     << "' for " << length << " Elem (non-parallel)" << endl;
               }
               coDoVec3 *data
                  =  new coDoVec3(name,length);
               data->get_adresses(&dataPtr[0],&dataPtr[1],&dataPtr[2]);
               distrObj = data;
            }
            length *= sizeof(float);

            // receive contents
            int i;
            for (i=0;i<numComp;i++)
            {
               if (recvData(dataPtr[i],length) != length )
                  return -1;
               if (d_verbose>2)
               {
                  cerr << "  : received data chunk of "<<length<<" bytes " << endl;
               }
            }
            outPort->setCurrentObject(distrObj);
         }

         break;
      }

      // ###########################################################
      // Initialisation of a parallel data
      // ###########################################################
      case PARA_INIT:
      {
         if (d_verbose>0)
         {
            cerr << "coSimLib Client called PARA_INIT" << endl;
         }

         // the information, but type has been read !!
         struct { int32 partitions,ports; }
         data;
         if (recvData(&data,sizeof(data)) != sizeof(data) )           return -1;

         // we have got another PARA_INIT call before
         if (d_numNodes)
            Covise::sendWarning("Multi-parallelism not implemented YET: Expect coredump");

         coParallelInit(data.partitions,data.ports);

         break;
      }

      // ###########################################################
      // Definition of a parallel data port
      // ###########################################################
      case PARA_PORT:                             // portname is read before
      {
         if (d_verbose>0)
         {
            cerr << "coSimLib Client called PARA_PORT" << endl;
         }

         int32 data;
         if (recvData(&data,sizeof(data)) != sizeof(data) )           return -1;

         // buffer holds the port name
         coParallelPort(buffer,data);

         break;
      }

      // ###########################################################
      // We get a mapping: either vertex or cell
      // ###########################################################
      case PARA_CELL_MAP:
      case PARA_VERTEX_MAP:
      {
         if (d_verbose>0)
            if (actComm==PARA_CELL_MAP)
               cerr << "coSimLib Client PARA_CELL_MAP" << endl;
         else
            cerr << "coSimLib Client PARA_VERTEX_MAP" << endl;

         // the information, but type has been read !!
         struct { int32 fortran,node,length; }
         data;
         if (recvData(&data,sizeof(data)) != sizeof(data) )           return -1;

         // whether this is a valid node number
         if ( data.node>d_numNodes || data.node<0)
         {
            sprintf(buffer,"%s: illegal node number: %d",d_name,data.node);
            Covise::sendError(buffer);                               return -1;
         }

         // read binary -> client has to convert if necessary
         int32 *actMap = new int32[data.length];
         int bytes = sizeof(int32) * data.length;
         if (recvData(actMap,bytes) != bytes )                return -1;

         // and now set the map
         if (setParaMap( (actComm==PARA_CELL_MAP),
            data.fortran, data.node, data.length,
            actMap) <0) return -1;

         break;
      }

      // ###########################################################
      // We get a new active node number
      // ###########################################################
      case PARA_NODE:
      {
         if (d_verbose>0)
            cerr << "coSimLib Client PARA_NODE :";

         // recv node number
         if (recvData(&d_actNode,sizeof(d_actNode))!=sizeof(d_actNode))  return -1;

         if (d_verbose>0)
            cerr << " Node = " << d_actNode << endl;

         // check
         if (d_actNode<0 || d_actNode>=d_numNodes)
         {
            sprintf(buffer,"%s: illegal node number: %d",d_name,d_actNode);
            Covise::sendError(buffer);                               return -1;
         }

         break;
      }

      // ###########################################################
      // QUIT : exit from server mode - close all open object
      // ###########################################################
      case QUIT:
      {
         if (d_verbose>0)
            cerr << "coSimLib Client QUIT" << endl;

         PortListElem *port = d_portList->next;
         while (port)
         {
            if (port->openObj)
            {
               port->openObj = NULL;              // we don't delete it -> the port will do
            }

            port = port->next;
         }

         break;
      }

      ////// default: kill the command from the Queue
      default:
      {
         d_command=NONE;
         break;
      }

   }

   // command was handled
   return actComm;
}


int coSimLib::isConnected()
{
   return (d_socket>1);
}


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
/////
/////   Module using client commands

///////////////////////////////////////////////////////////////////////////
/// set the mapping from the module

int coSimLib::coParallelInit(int numNodes, int numPorts)
{
   if (d_verbose>0)
   {
      cerr << "coParallelInit( numNodes=" << numNodes << ", numPorts="
         << numPorts << " )" << endl;
   }

   d_numNodes = numNodes;

   // prepate the lists
   d_cellMap = new int *[d_numNodes];
   d_vertMap = new int *[d_numNodes];

   // set zero
   int i;
   for (i=0;i<d_numNodes;i++)
      d_cellMap[i] = d_vertMap[i] = NULL;
   return 0;
}


///////////////////////////////////////////////////////////////////////////
/// initialize Parallel ports: local COPAPO

int coSimLib::coParallelPort(const char *portName, int isCellData)
{
   if (d_verbose>0)
   {
      cerr << "coParallelPort( portName='" << portName << "', isCellData="
         << isCellData << " )" << endl;
   }

   // create port at the end of the list: we have a dummy at the start!
   PortListElem *port = d_portList;
   while (port->next)
      port=port->next;
   port->next = new PortListElem;
   port = port->next;

   port->name     = strcpy(new char[strlen(portName)+1],portName);
   port->openObj  = NULL;
   port->numParts = 0;
   port->next     = NULL;
   if (isCellData)
      port->map = &d_cellMap;
   else
      port->map = &d_vertMap;
   return 0;
}


///////////////////////////////////////////////////////////////////////////
/// set the mapping from the module

int coSimLib::setParaMap(int isCell, int isFortran, int nodeNo, int length,
int32 *nodeMap)
{
   if (d_verbose>0)
   {
      cerr << "setParaMap( isCell=" << isCell
         << ", isFortran=" << isFortran
         << ", nodeNo=" << nodeNo
         << ", length=" << length
         << ", Map=("  <<  nodeMap[0] << "," << nodeMap[1] << "..."
         <<  nodeMap[length-2] << "," << nodeMap[length-1]
         << " )" << endl;
   }
   int i;
   char buffer[128];
   int **map = (isCell) ? d_cellMap : d_vertMap;

   // write Mappings to files
   if (d_verbose>1)
   {
      sprintf(buffer,"coSimlib.%d.Map",nodeNo);
      FILE *outFile = fopen(buffer,"w");

      fprintf(outFile,"Node mapping file for %s(%d) for Proc #%d\n\n",
         d_name,getpid(),nodeNo);
      fprintf(outFile,"%6s ; %6s\n\n","node","proc");
      if (outFile)
      {
         for (i=0;i<length;i++)
            fprintf(outFile,"%6d ; %6d\n",nodeMap[i],nodeNo);
         fclose(outFile);
      }
      else
         cerr << "ERROR: coSimLib requested output file '"
            << buffer << "' could not be created." << endl;
   }

   // whether this is a valid node number
   if ( nodeNo>d_numNodes || nodeNo<0)
   {
      sprintf(buffer,"%s: illegal node number: %d",d_name,nodeNo);
      Covise::sendError(buffer);                               return -1;
   }

   map[nodeNo] = nodeMap;

   // decrement 1 for fortran counting
   if (isFortran)
      for (i=0;i<length;i++)
         (nodeMap[i])--;

   // now find the max. Index
   if (isCell)
   {
      for (i=0;i<length;i++)
         if (nodeMap[i] >= d_numCells)
            d_numCells = nodeMap[i]+1;
   }
   else
   {
      for (i=0;i<length;i++)
         if (nodeMap[i] >= d_numVert)
            d_numVert = nodeMap[i]+1;
   }

   return 0;
}
