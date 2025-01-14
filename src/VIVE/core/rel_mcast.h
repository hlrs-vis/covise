/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** Reliable Multicast via NORM (NACK-Oriented Reliable Multicast) ******
 *									*
 * Filename: rel_mcast.h						*
 * Author:   Alex Velazquez						*
 * Created:  2011-03-31							*
 ************************************************************************/

/** Adjustable settings of the NORM protocol: ***************************
 * 									*
 * Description:		Var.name:	Set with:			*
 *                     (coconfig + .cpp)				*
 * ---------------------------------------------------------------	*
 * Debug level		debugLevel	setDebugLevel(int)		*
 * Multicast address	mcastAddr	constructor (has default)	*
 * Multicast port	mcastPort	constructor (has default)	*
 * Multicast interface	mcastIface	constructor			*
 * MTU 			mtu		setMTU(int)			*
 * TTL			ttl		setTTL(int)			*
 * Loopback behavior	lback		setLoopback(bool)		*
 * Group size		groupSize	server's constructor		*
 * Send buffer space	sndBufferSpace	setBufferSpace(unsigned int)	*
 * Receive buffer space	rcvBufferSpace	setBufferSpace(unsigned int)	*
 * Block size		blockSize	setBlocksAndParity(int,int)	*
 * Parity		numParity	setBlocksAndParity(int,int)	*
 * Max cache bytes	txCacheSize	setTxCacheBounds(int,int,int)	*
 * Min cache num	txCacheMin	setTxCacheBounds(int,int,int)	*
 * Max cache num	txCacheMax	setTxCacheBounds(int,int,int)	*
 * TX Rate		txRate		setTransferRate(int)		*
 * Backoff factor	backoffFactor	setBackoffFactor(double)	*
 * UDP sock.buff.size	sockBufferSize	setSockBufferSize(int)		*
 * Read timeout	(sec)	readTimeoutSec	setTimeout(int)			*
 * Write timeout (ms)	writeTimeoutMsec setTimeout(int)		*
 * Retry timeout (us)	retryTimeout	setRetryTimeout(int)		*
 * Max obj length (B)	maxLength	setMaxLength(int)		*
 ************************************************************************/

/** Example covise configuration file: **********************************

 <?xml version="1.0"?>

 <COCONFIG version="1" >
  <GLOBAL>
   <COVER>
    <MultiPC>

     <SyncMode value="MULTICAST" />

     <Multicast>
      <debugLevel value="0" />
      <mcastAddr value="224.223.222.221" />
      <mcastPort value="23232" />
      <mtu value="1500" />
      <ttl value="1" />
      <lback value="off" />
      <sndBufferSpace value="1000000" />
      <rcvBufferSpace value="1000000" />
      <blockSize value="4" />
      <numParity value="0" />
      <txCacheSize value="100000000" />
      <txCacheMin value="1" />
      <txCacheMax value="128" />
      <txRate value="1000" />
      <backoffFactor value="0.0" />
      <sockBufferSize value="512000" />
      <readTimeoutSec value="30" />
      <writeTimeoutMsec value="500" />
      <retryTimeout value="100" />
      <maxLength value="1000000" />
     </Multicast>

    </MultiPC>
   </COVER>
  </GLOBAL>
 </COCONFIG>

 ************************************************************************/

/** Creating server/client instances: ***********************************

 1) Server (master):
   
   // Determine server's parameters
   int numClients = 1;			// Number of clients
   char *addr = "224.223.222.221";	// Any Class-D IP address
   int port = 23232; 			// Any valid port number
   
   // Call server's constructor
   // (could also call Rel_Mcast(true, numClients) for default addr/port)
   multicast = new Rel_Mcast(true, numClients, addr, port);
   
   // Set any non-default settings (advanced)
   //e.g. multicast->setTimeout(1000);	// Write timeout
   //etc....
   
   // Initialize the server
   // (may fail if the addr/port is invalid or in use)
   if (multicast->init() != Rel_Mcast::RM_OK) delete multicast;
   
   // Write a multicast message
   multicast->write_mcast("Testing 123", 12);
   
   // Clean up by calling destructor
   sleep(10);
   delete multicast;
 
 
 2) Clients (slaves):
   
   // Determine client's parameters
   int clientID = 1;			// Any integer greater than 0
   					//   and unique among clients
   char *addr = "224.223.222.221";	// Any Class-D IP address
   int port = 23232; 			// Any valid port number
   
   // Call client's constructor
   // (could also call Rel_Mcast(clientID) to choose default addr/port)
   multicast = new Rel_Mcast(clientID, addr, port);
   
   // Set any non-default settings (advanced)
   //e.g. multicast->setTimeout(20);	// Read timeout
   //etc....
   
   // Initialize the client
   // (may fail if the addr/port is invalid or in use)
   if ( multicast->init() != Rel_Mcast::RM_OK ) delete multicast;
   
   // Receive an incoming multicast
   char *buffer = new char[12];
   multicast->read_mcast(buffer, 12);
   printf("Received: %d\n", buffer);
   
   // Clean up by calling destructor
   delete multicast;

 ************************************************************************/

// Default debug mode (0=errors, 1=info, 2=status, 3=all(trace) )
#define DEBUG_LVL 0

// Default address/port and interface to be used for multicast
#define MCAST_ADDR "224.223.222.221"
#define MCAST_PORT 23232
#define MCAST_IFACE "eth0"

// Default values of various NORM settings
#define SND_BUFFER_SPACE 1000000
#define RCV_BUFFER_SPACE 1000000
#define MTU_SIZE 1500
#define BLOCK_SIZE 4
#define NUM_PARITY 0

// Default TTL (for communication on a LAN, this can be 1)
#define NORM_TTL 1

// Default cache bounds
#define CACHE_MAX_SIZE 100000000
#define CACHE_MIN_NUM 1
#define CACHE_MAX_NUM 128

// Default transmit rate in Mbps (1000 = 1Gbps)
#define TX_MBPS 1000

// Default backoff factor (low value yields low latency)
#define BACKOFF_FACTOR 0.0

// Default buffer size (bytes) of the sending+receiving UDP sockets (increase to prevent socket bottleneck due to high Tx rate)
#define SOCK_BUFFER_SIZE 512000

// Default maximum amount of time in sec to wait for select()
#define READ_TIMEOUT_SEC 30
#define WRITE_TIMEOUT_MSEC 500
#define RETRY_TIMEOUT 100

// Define max object length for read/write
#define MAX_LENGTH 1000000

// Include the NORM API
#include "normApi.h"
#include "protoDefs.h"
#include "protoDebug.h"
// Include necessary header files
#include <stdio.h> // for printf(), stderr, etc.
#include <string.h> // for strcmp()
#include <stdlib.h> // for rand()
#include <unistd.h> // for usleep()
#include <stdarg.h> // for variable argument list
#include <util/coTypes.h> // for COVISE

using namespace std;

namespace opencover
{
// Class for multicast
class Rel_Mcast
{
public:
    // Error types
    enum RM_Error_Type
    {
        RM_OK, // No error
        RM_INIT_ERROR, // Error during init() phase
        RM_SETTING_ERROR, // Error when changing a setting (i.e. parameter out of bounds)
        RM_WRITE_ERROR, // Error while writing a multicast message
        RM_READ_ERROR // Error while reading a multicast message
    };

    // Constructor for server
    Rel_Mcast(bool server, int numClients, const char *addr = MCAST_ADDR, int port = MCAST_PORT, const char *interface = MCAST_IFACE);
    // Constructor for client
    Rel_Mcast(int clientNum, const char *addr = MCAST_ADDR, int port = MCAST_PORT, const char *interface = MCAST_IFACE);
    // Destructor
    ~Rel_Mcast();
    // Initialize Rel_Mcast instance
    RM_Error_Type init();
    // Send a multicast message to all clients
    RM_Error_Type write_mcast(const void *data, int length);
    // Read a multicast message sent from the server
    RM_Error_Type read_mcast(void *dest, int length);
    // Change settings anytime
    RM_Error_Type setDebugLevel(int lvl);
    RM_Error_Type setLoopback(bool lb);
    RM_Error_Type setTxRate(int Mbps);
    RM_Error_Type setTimeout(int t);
    RM_Error_Type setRetryTimeout(int u);
    RM_Error_Type setNumClients(int n);
    RM_Error_Type setTxCacheBounds(unsigned int bytes, int min, int max);
    RM_Error_Type setBackoffFactor(double factor);
    // Change settings before init()
    RM_Error_Type setSockBufferSize(int bytes);
    RM_Error_Type setInterface(const char *iface);
    RM_Error_Type setTTL(int t);
    RM_Error_Type setBufferSpace(unsigned int bytes);
    RM_Error_Type setMTU(int bytes);
    RM_Error_Type setBlocksAndParity(int b, int p);
    RM_Error_Type setMaxLength(int m);

private:
    // Set up server node upon a call to init()
    RM_Error_Type init_server();
    // Set up client node upon a call to init()
    RM_Error_Type init_client();
    // Event handlers for write_mcast()
    void waitForVacancy();
    void waitForSend();
    // Event handler for read_mcast()
    void waitForRead(int sec, int msec, void *dest, int length);
    // Function to print error/debug messages: lvl=0 -> error, lvl>=1 -> info_msg
    void printMsg(int lvl, const char *msg, ...);

    // Instance variables
    int debugLevel; // (0=errors, 1=basic info, 2=detailed info,
    //    3=per-message info [warning: many print statements], 4=all)
    const char *mcastAddr; // Multicast address, i.e. Class-D IP address
    //    in range 224.0.0.0 - 239.255.255.255
    int mcastPort; // Port that communication will take place on
    const char *mcastIface; // Interface to use for multicast (e.g. eth0)
    int nodeId; // Unique identifier for each node taking part in the communication
    //    (i.e. server and each client)
    bool isServer; // Should be 'true' for one node-- the master
    bool lback; // If 'true', multicast packets will loopback to the sender
    int ttl; // Max number of routing hops a packet can make before being discarded
    NormInstanceHandle normInstance;
    NormSessionHandle normSession;
    NormSessionId normSessionId;
    NormDescriptor normFD; // Norm file descriptor for pselect() call
    fd_set fdSet; // File descriptor set for pselect() call
    struct timespec tv; // timeout for pselect() call
    int retval; // Return value of pselect() call
    bool isRunning; // Whether the node is currently running
    bool quitting; // Whether the node is due to quit
    int numObjPending; // Sender's count of the number of pending transmissions
    //    in the transmit queue (keep low to avoid flooding receivers)
    int numPurged; // Count of the number of purged items
    unsigned int sndBufferSpace; // How much history (bytes) is kept on sender-side
    unsigned int rcvBufferSpace; // How much history (bytes) is kept on receiver-side
    int mtu; // Size of NORM segments (includes protocol headers)
    int blockSize; // Size of NORM blocks (used for FEC error-correction)
    int numParity; // Number of parity symbol segments sender will calculate per FEC block
    int txRate; // Sender's max transmission rate in Mbps (e.g. 1000=>1Gbps)
    NormSize txCacheSize; // Size of server's cache containing pending data items
    int txCacheMin; // Minumum number of data items cached by server (overrides txCacheSize)
    int txCacheMax; // Maximum number of data items cached by server (overrides txCacheSize)
    int sockBufferSize; // Size of the UDP socket's buffer (should be increased for fast communication)
    double backoffFactor; // Time (sec) to scale NACK-related repairs (0.0 for lowest latency)
    int groupSize; // Expected number of clients, does not need to be exact
    //    (within an order of magnitude)
    bool dataSent; // For server to keep track the "send" status of each multicast
    long sentCounter; // Counter for the number of sent multicasts
    bool gotData; // True if a multicast has been previously received
    long readCounter; // Counter for the number of received multicasts
    int readTimeoutSec; // Max #seconds for a client to wait for a read
    int retryTimeout; // Microseconds to wait before retrying a failed data enqueue (due to full queue)
    int writeTimeoutSec; // Max #seconds to wait for a write (in addition to writeTimeoutMsec)
    int writeTimeoutMsec; // Max #msecs to wait for a write (in addition to writeTimeoutSec)
    int maxLength; // Max length (bytes) of a multicast
};
}
