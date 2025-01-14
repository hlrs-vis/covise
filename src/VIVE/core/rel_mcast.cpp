/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** Reliable Multicast via NORM (NACK-Oriented Reliable Multicast) ******
 *									*
 * Filename: rel_mcast.cpp						*
 * Author:   Alex Velazquez						*
 * Created:  2011-03-31							*
 ************************************************************************/

#ifndef NOMCAST

#include "rel_mcast.h"

using namespace opencover;

/**
 * Rel_Mcast()
 * Constructor for multicast-server (master/sender)
 */
Rel_Mcast::Rel_Mcast(bool, int numClients, const char *addr, int port,
                     const char *interface)
{
    // Initialize instance variables to given (or default) values
    isRunning = false;
    quitting = false;
    isServer = true;
    nodeId = 1000; // unique id for server
    groupSize = numClients;
    mcastAddr = addr;
    mcastPort = port;
    mcastIface = interface;
    dataSent = false;
    sentCounter = 0;
    numObjPending = 0;
    numPurged = 0;

    // Various settings (for server)
    debugLevel = DEBUG_LVL;
    lback = false;
    ttl = NORM_TTL;
    normSessionId = (NormSessionId)rand();
    sndBufferSpace = SND_BUFFER_SPACE;
    mtu = MTU_SIZE;
    blockSize = BLOCK_SIZE;
    numParity = NUM_PARITY;
    txRate = TX_MBPS;
    txCacheSize = CACHE_MAX_SIZE;
    txCacheMin = CACHE_MIN_NUM;
    txCacheMax = CACHE_MAX_NUM;
    sockBufferSize = SOCK_BUFFER_SIZE;
    backoffFactor = BACKOFF_FACTOR;
    writeTimeoutSec = (int)(WRITE_TIMEOUT_MSEC / 1000);
    writeTimeoutMsec = (int)(WRITE_TIMEOUT_MSEC % 1000);
    retryTimeout = RETRY_TIMEOUT;
    maxLength = MAX_LENGTH;

    // DEBUG: instance created
    printMsg(2, "Rel_Mcast instance created.");
}

/**
 * Rel_Mcast()
 * Constructor for multicast-client (slave/receiver)
 */
Rel_Mcast::Rel_Mcast(int clientNum, const char *addr, int port,
                     const char *interface)
{
    // Initialize instance variables to given (or default) values
    isRunning = false;
    quitting = false;
    isServer = false;
    nodeId = 1000 + clientNum; // unique id for this client
    mcastAddr = addr;
    mcastPort = port;
    mcastIface = interface;
    gotData = false;
    readCounter = 0;

    // Various settings (for client)
    debugLevel = DEBUG_LVL;
    lback = false;
    ttl = NORM_TTL;
    rcvBufferSpace = RCV_BUFFER_SPACE;
    sockBufferSize = SOCK_BUFFER_SIZE;
    readTimeoutSec = READ_TIMEOUT_SEC;
    maxLength = MAX_LENGTH;

    // DEBUG: instance created
    printMsg(2, "Rel_Mcast instance created.");
}

/**
 * ~Rel_Mcast()
 * Destructor
 */
Rel_Mcast::~Rel_Mcast()
{
    if (isServer)
    {
        // Send all clients a "quit" message
        write_mcast(NULL, -1);

        // Brief pause before finishing cleanup
        usleep(2000); // 2 msec

        // Stop the server
        NormStopSender(normSession);
        isRunning = false;
        // DEBUG: notify of server stop
        printMsg(1, "Server successfully stopped.");
    }
    else
    {
        // Stop the client
        NormStopReceiver(normSession);
        isRunning = false;
        // DEBUG: notify of client stop
        printMsg(1, "Client successfully stopped.");
    }

    // Destroy the NORM session
    NormDestroySession(normSession);
    // DEBUG: session destroyed
    printMsg(2, "NORM Session destroyed.");

    // Stop and then destroy the NORM API instance
    NormStopInstance(normInstance);
    NormDestroyInstance(normInstance);
    // DEBUG: instance stopped/destroyed, now exiting
    printMsg(2, "NORM Instance stopped and destroyed.");
}

/**
 * init()
 * Perform operations common to both server+client, and then choose proper helper function
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::init()
{
    // Create an instance of the NORM API; exit upon failure
    normInstance = NormCreateInstance();
    if (normInstance == NORM_INSTANCE_INVALID)
    {
        printMsg(0, "NORM Instance could not be created.");
        return RM_INIT_ERROR;
    }
    // DEBUG: normInstance created
    printMsg(2, "NORM Instance successfully created.");

    // Create a new NORM session; exit upon failure
    normSession = NormCreateSession(normInstance, mcastAddr, mcastPort, nodeId);
    NormSetCongestionControl(normSession, false);

    // Get integer file descriptor for this NORM instance
    normFD = NormGetDescriptor(normInstance);

    if (normSession == NORM_SESSION_INVALID)
    {
        printMsg(0, "NORM Session could not be created with addr='%s/%d'.", mcastAddr, mcastPort);
        return RM_INIT_ERROR;
    }
    // DEBUG: normSession created
    printMsg(2, "NORM Session successfully created with addr='%s/%d'.", mcastAddr, mcastPort);

    // Set the multicast interface
    if (NormSetMulticastInterface(normSession, mcastIface) == false)
        printMsg(0, "Multicast interface could not be set to '%s'.", mcastIface);
    // DEBUG: Multicast set
    printMsg(2, "Interface = '%s'", mcastIface);

    // Set the TTL of this NORM session (1 to stay local)
    if (NormSetTTL(normSession, ttl) == false)
        printMsg(0, "TTL could not be set to %d.", ttl);
    // DEBUG: TTL set
    printMsg(2, "TTL = %d", ttl);

    // Set the multicast loopback behavior
    NormSetLoopback(normSession, lback);
    // DEBUG: loopback set to false
    printMsg(2, "Loopback = '%s'", (lback) ? "TRUE" : "FALSE");

    // Pass control to appropriate helper function
    if (isServer)
        return init_server();
    else
        return init_client();
}

/**
 * init_server()
 * Start running in server mode
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::init_server()
{
    // DEBUG: Print status message
    printMsg(2, "Running in Server mode.");

    // Start the server (sender); return upon failure
    if (!NormStartSender(normSession, normSessionId, sndBufferSpace, mtu, blockSize, numParity))
    {
        printMsg(0, "Server could not be started.");
        return RM_INIT_ERROR;
    }
    else
    {
        isRunning = true;
    }
    // DEBUG: sender started
    printMsg(2, "Server successfully started:\n                            > SessionId   = %d\n                            > BufferSpace = %d\n                            > SegmentSize = %d\n                            > BlockSize   = %d\n                            > NumParity   = %d", normSessionId, sndBufferSpace, mtu, blockSize, numParity);

    // Set additional settings of the NORM API (v1.4b3)
    NormSetTransmitRate(normSession, (double)1000000 * txRate);
    NormSetTransmitCacheBounds(normSession, txCacheSize, txCacheMin, txCacheMax);
    // Disable the prior two lines and enable the following two lines for NORM API v1.4b4
    //NormSetTxRate (normSession, (double) 1000000 * txRate);
    //NormSetTxCacheBounds (normSession, txCacheSize, txCacheMin, txCacheMax);

    // Set transmit socket buffer size to keep up with fast transmission
    if (!NormSetTxSocketBuffer(normSession, sockBufferSize))
        printMsg(0, "Transmit socket size could not be set to %d.", sockBufferSize);
    // Set a low backoff factor for low-latency communication
    NormSetBackoffFactor(normSession, backoffFactor);
    NormSetGroupSize(normSession, groupSize);
    // DEBUG: server settings set
    printMsg(1, "Server settings successfully set, init() complete.");

    // Return an appropriate response code
    return RM_OK;
}

/**
 * init_client()
 * Start running in client mode
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::init_client()
{
    // DEBUG: Print status message
    printMsg(2, "Running in Client mode.");

    // Start the client (receiver); return upon failure
    if (!NormStartReceiver(normSession, rcvBufferSpace))
    {
        printMsg(0, "Client could not be started.");
        return RM_INIT_ERROR;
    }
    else
        isRunning = true;
    // DEBUG: receiver started
    printMsg(2, "Client successfully started.");

    // Set additional settings of the NORM Receiver API
    // Set receive socket buffer size to keep up with fast transmission
    if (!NormSetRxSocketBuffer(normSession, sockBufferSize))
        printMsg(0, "Receive socket size could not be set to %d.", sockBufferSize);
    // DEBUG: client settings set
    printMsg(1, "Client settings successfully set, init() complete.");

    // init_client() completed successfully
    return RM_OK;
}

/**
 * write_mcast()
 * Send a multicast message to all clients
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::write_mcast(const void *data, int length)
{
    // DEBUG: Rel_Mcast::write_mcast()
    printMsg(3, "Rel_Mcast::write_mcast() [Sent=%d/Pending=%d/Purged=%d]", sentCounter, numObjPending, numPurged);

    if (length > 0 && isServer)
    {
        NormObjectHandle obj;

        // Enqueue the data to be sent
        obj = NormDataEnqueue(normSession, (char *)data, length);

        // Check that it was properly enqueued
        if (obj == NORM_OBJECT_INVALID)
        {
            // If the data could not be enqueued, try again until it succeeds
            do
            {
                // DEBUG: enqueue failed
                printMsg(2, "Data could not be enqueued. Retrying...");

                // Sleep briefly
                //usleep(retryTimeout);

                // Wait for room the queue
                waitForVacancy();

                // Try again to enqueue
                obj = NormDataEnqueue(normSession, (char *)data, length);
            } while (obj == NORM_OBJECT_INVALID);
        }

        // DEBUG: enqueue successful
        printMsg(3, "Data successfully enqueued. Waiting for send...");
        // Block and wait for data to be sent
        waitForSend();

        // Data has been sent
        if (dataSent)
        {
            // DEBUG: data sent
            printMsg(3, "Data successfully sent.");

            // Reset 'dataSent' for next multicast
            dataSent = false;

            // Before returning successfully, check to make sure that the counters are consistent
            if (sentCounter == numPurged + numObjPending)
                return RM_OK;
            else
                return RM_WRITE_ERROR;
        }
        // Otherwise return an error
        else
        {
            printMsg(0, "Data could not be sent.");
            return RM_WRITE_ERROR;
        }
    }
    else if (!isServer)
    {
        printMsg(0, "Client cannot write messages via multicast.");
        return RM_WRITE_ERROR;
    }
    else if (length == 0)
    {
        printMsg(2, "Rel_Mcast::write_mcast [message length is 0]");
        sentCounter++;
        numPurged++;
        return RM_OK; //Nothing needs to be done, so OK (this coordinates with read_mcast() )
    }
    else
    {
        // DEBUG: quit and tell clients
        printMsg(2, "Server is quitting and notifying clients.");
        quitting = true;
        NormObjectHandle obj;

        // Enqueue the data to be sent
        obj = NormDataEnqueue(normSession, "QUIT", 5, "QUIT", 5);

        // Check that it was properly enqueued
        if (obj == NORM_OBJECT_INVALID)
        {
            // DEBUG: enqueue failed
            printMsg(2, "Data could not be enqueued. Waiting for spot in queue...");
            // Block and wait for a vacancy in the queue
            waitForVacancy();
            // Then requeue and exit if it fails (shouldn't fail)
            obj = NormDataEnqueue(normSession, "QUIT", 5, "QUIT", 5);
            if (obj == NORM_OBJECT_INVALID)
                return RM_WRITE_ERROR;
        }
        return RM_OK;
    }
}

/**
 * read_mcast()
 * Read a multicast message sent by the server
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::read_mcast(void *dest, int length)
{
    // DEBUG: Rel_Mcast::read_mcast()
    printMsg(3, "Rel_Mcast::read_mcast() [Read=%d]", readCounter);

    if (length > 0 && !isServer)
    {
        // Handle any pending events
        waitForRead(readTimeoutSec, 0, dest, length);

        // Data has been copied to its destination
        if (gotData)
        {
            // DEBUG: data read
            printMsg(3, "Data successfully read.");

            // Reset 'gotData' for next multicast
            gotData = false;
            return RM_OK;
        }
        else
        {
            printMsg(0, "Client could not read data.");
            return RM_READ_ERROR;
        }
    }
    else if (isServer)
    {
        printMsg(0, "Server cannot read messages.");
        return RM_READ_ERROR;
    }
    else if (length == 0)
    {
        printMsg(2, "Rel_Mcast::read_mcast [message length is 0]");
        readCounter++;
        return RM_OK; // Nothing needs to be done, so OK (this coordinates with write_mcast() )
    }
    else
    {
        printMsg(0, "Rel_Mcast::read_mcast [negative message length]");
        return RM_READ_ERROR;
    }
}

/**
 * waitForVacancy()
 * Block and wait for a NORM_TX_QUEUE_VACANCY event (handle other events as they are received)
 */
void Rel_Mcast::waitForVacancy()
{
    // DEBUG: Rel_Mcast::waitForVacancy()
    printMsg(2, "Rel_Mcast::waitForVacancy()");

    // Block and wait for vacancy event
    NormEvent normEvent;
    bool gotVacancy = false;
    while (!gotVacancy)
    {
        // Block and get next event
        NormGetNextEvent(normInstance, &normEvent);

        // Handle the NORM event
        switch (normEvent.type)
        {
        case NORM_TX_QUEUE_EMPTY:
        {
            gotVacancy = true;

            // DEBUG: empty queue
            printMsg(2, "NORM_TX_QUEUE_EMPTY: no items waiting to be transmitted.");
            break;
        }
        case NORM_TX_QUEUE_VACANCY:
        {
            gotVacancy = true;

            // DEBUG: new vacancy in queue
            printMsg(2, "NORM_TX_QUEUE_VACANCY: ready to enqueue another object.");
            break;
        }
        case NORM_TX_OBJECT_PURGED:
        {
            numObjPending--;
            numPurged++;

            // DEBUG: object purged
            printMsg(2, "NORM_TX_OBJECT_PURGED: %d items still pending.", numObjPending);
            break;
        }
        case NORM_TX_OBJECT_SENT:
        {
            dataSent = true;
            numObjPending++;
            sentCounter++;

            // DEBUG: object sent and pending
            printMsg(2, "NORM_TX_OBJECT_SENT #%d: %d items now pending.", sentCounter, numObjPending);
            break;
        }
        default:
            break;
        }
    }
}

/**
 * waitForSend()
 * Block and wait for a NORM_TX_OBJECT_SENT event (handle other events as they are received)
 */
void Rel_Mcast::waitForSend()
{
    // DEBUG: Rel_Mcast::waitForSend()
    printMsg(3, "Rel_Mcast::waitForSend()");

    // Block and wait for vacancy event
    NormEvent normEvent;
    while (!dataSent)
    {
        // Block and get next event
        NormGetNextEvent(normInstance, &normEvent);

        // Handle the NORM event
        switch (normEvent.type)
        {
        case NORM_TX_QUEUE_EMPTY:
        {
            // DEBUG: new vacancy in queue
            printMsg(4, "NORM_TX_QUEUE_EMPTY: no items waiting to be transmitted.");
            break;
        }
        case NORM_TX_QUEUE_VACANCY:
        {
            // DEBUG: new vacancy in queue
            printMsg(4, "NORM_TX_QUEUE_VACANCY: ready to enqueue another object.");
            break;
        }
        case NORM_TX_OBJECT_PURGED:
        {
            numObjPending--;
            numPurged++;

            // DEBUG: object purged
            printMsg(4, "NORM_TX_OBJECT_PURGED: %d items still pending.", numObjPending);
            break;
        }
        case NORM_TX_OBJECT_SENT:
        {
            dataSent = true;
            numObjPending++;
            sentCounter++;

            // DEBUG: object sent and pending
            printMsg(4, "NORM_TX_OBJECT_SENT #%d: %d items now pending.", sentCounter, numObjPending);
            break;
        }
        default:
            break;
        }
    }
}

/**
 * waitForRead()
 * Wait for a NORM_RX_OBJECT_COMPLETED event until the timeout is reached
 */
void Rel_Mcast::waitForRead(int sec, int msec, void *dest, int length)
{
    // DEBUG: Rel_Mcast::waitForRead()
    printMsg(3, "Rel_Mcast::waitForRead()");

    // Add normFD to the file descriptor set
    FD_ZERO(&fdSet);
    FD_SET(normFD, &fdSet);
    // Wait up to sec+msec seconds
    tv.tv_sec = sec;
    tv.tv_nsec = (long)1000000 * msec; // convert milliseconds to nanoseconds for pselect()

    // If quitting, wait for a very short time
    if (quitting)
    {
        tv.tv_sec = 0;
        tv.tv_nsec = 2000000; // wait 2 msec
    }

    // Watch the file descriptor for events to be read
    bool keepGoing = true;
    while (keepGoing)
    {
        retval = pselect((int)normFD + 1, &fdSet, NULL, NULL, &tv, NULL);

        // Handle pselect()
        if (retval == -1)
        {
            // Return error
            printMsg(0, "Error during pselect().");
        }
        else if (retval)
        {
            // Read the new event
            NormEvent normEvent;
            NormGetNextEvent(normInstance, &normEvent);

            // Handle client events
            if (normEvent.type == NORM_RX_OBJECT_COMPLETED)
            {
                // Detach the received data
                gotData = true;
                readCounter++;
                const char *receivedData = NormDataAccessData(normEvent.object);

                // Copy data to destination
                memcpy(dest, receivedData, length);

                // DEBUG: data read
                printMsg(4, "NORM_RX_OBJECT_COMPLETED #%d: data successfully copied.", readCounter);
            }
            else if (normEvent.type == NORM_RX_OBJECT_INFO)
            {
                // If "QUIT" was received as info, clean up and exit
                int infoLen = NormObjectGetInfoLength(normEvent.object);
                char *infoBuf = new char[infoLen];
                if (infoLen == NormObjectGetInfo(normEvent.object, infoBuf, infoLen))
                {
                    if (strcmp(infoBuf, "QUIT") == 0)
                    {
                        quitting = true;
                        // DEBUG: quit read
                        printMsg(2, "Client received notification to quit from info.");
                    }
                }
                // DEBUG: data read
                printMsg(4, "NORM_RX_OBJECT_INFO: object with info.", ++readCounter);
            }
        }
        else
        {
            // DEBUG: Reached the timeout with no new events available
            if (quitting)
                printMsg(2, "Quitting after timeout.");
            else
                printMsg(2, "Timeout reached-- no event in %d s, %d ms.", sec, msec);
        }

        // Check whether to continue or not
        // If there were no more events (timeout or error), stop checking
        if (retval == 0 || retval == -1)
            keepGoing = false;
        // If the client received its data, stop checking
        if (gotData || quitting)
            keepGoing = false;
    }
}

/**
 * printMsg()
 * Print a variable-length debug or error message
 */
void Rel_Mcast::printMsg(int lvl, const char *msg, ...)
{

    // If this message is at/below the current debug level, print it
    if (lvl <= debugLevel)
    {
        // Print header
        if (lvl == 0)
        {
            // Level 0  is an error
            fprintf(stderr, "REL_MCAST>%s%d (error): ",
                    (isServer) ? "master" : "slave", (nodeId - 1000));
        }
        else
        {
            // Level x  is an informational message
            printf("REL_MCAST>%s%d (info%d): ",
                   (isServer) ? "master" : "slave", (nodeId - 1000), lvl);
        }

        // Print passed message and newline
        va_list list;
        va_start(list, msg);
        if (lvl == 0)
        {
            vfprintf(stderr, msg, list);
            va_end(list);
            fprintf(stderr, "\n");
        }
        else
        {
            vprintf(msg, list);
            va_end(list);
            fprintf(stdout, "\n");
        }
    }
}

/********* Functions to set various client/server parameters ***********/

/**
 * setDebugLevel()
 * Set the debug value to a given level (1, 2, or 3)
 * Can be changed anytime
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::setDebugLevel(int lvl)
{
    if (lvl > 0)
    {
        debugLevel = lvl;
        printMsg(2, "DebugLevel = %d", debugLevel);
        return RM_OK;
    }
    else
    {
        printMsg(0, "DebugLevel must be non-negative.");
        return RM_SETTING_ERROR;
    }
}

/**
 * setLoopback()
 * Set whether the multicast messages loop back to the server
 * Can be changed anytime
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::setLoopback(bool lb)
{
    if (!isServer)
    {
        printMsg(0, "Loopback only set by server.");
        return RM_SETTING_ERROR; // Only makes sense for the server
    }
    else
    {
        lback = lb;
        if (isRunning)
            NormSetLoopback(normSession, lback);
        printMsg(2, "Loopback = '%s'", (lback) ? "TRUE" : "FALSE");
        return RM_OK;
    }
}

/**
 * setTxRate()
 * Set the sender's max rate of transfer in Mbps (e.g. 1000 = 1Gbps)
 * Can be changed anytime
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::setTxRate(int Mbps)
{
    if (!isServer || Mbps < 0)
    {
        printMsg(0, "Transmit rate only applies to server, and must be non-negative.");
        return RM_SETTING_ERROR;
    }
    else
    {
        txRate = Mbps;
        if (isRunning)
            NormSetTransmitRate(normSession, (double)1000000 * txRate);
        // Disable the prior line and enable the following line for NORM API v1.4b4
        //if (isRunning) NormSetTxRate (normSession, (double) 1000000 * txRate);
        printMsg(2, "TxRate = %d Mbps", txRate);
        return RM_OK;
    }
}

/**
 * setTimeout()
 * Set the time to wait for an event
 * Can be changed anytime
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::setTimeout(int t)
{
    if (t < 0)
    {
        printMsg(0, "Timeout must be non-negative.");
        return RM_SETTING_ERROR;
    }
    else if (!isServer)
    {
        readTimeoutSec = t;
        printMsg(2, "ReadTimeout = %d sec", readTimeoutSec);
        return RM_OK;
    }
    else
    {
        writeTimeoutSec = (int)(t / 1000);
        writeTimeoutMsec = (int)(t % 1000);
        printMsg(2, "WriteTimeout = %d sec, %d msec", writeTimeoutSec, writeTimeoutMsec);
        return RM_OK;
    }
}

/**
 * setRetryTimeout()
 * Set the time to wait before retrying a failed data enqueue
 * Can be changed anytime
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::setRetryTimeout(int u)
{
    if (!isServer || u <= 0 || u >= 1000000)
    {
        printMsg(0, "Retry timeout only applies to server, and must be between 0 and 1000000 usec.");
        return RM_SETTING_ERROR;
    }
    else
    {
        retryTimeout = u;
        printMsg(2, "RetryTimeout = %d usec", retryTimeout);
        return RM_OK;
    }
}

/**
 * setNumClients()
 * Set the number of clients
 * Can be changed anytime
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::setNumClients(int n)
{
    if (n < 0 || !isServer)
    {
        printMsg(0, "Number of clients only applies to server, and must be non-negative.");
        return RM_SETTING_ERROR;
    }
    else
    {
        groupSize = n;
        if (isRunning)
            NormSetGroupSize(normSession, groupSize);
        printMsg(2, "GroupSize = %d", groupSize);
        return RM_OK;
    }
}

/**
 * setTxCacheBounds()
 * 
 * Can be changed anytime
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::setTxCacheBounds(unsigned int bytes, int min, int max)
{
    if (!isServer || max < min || min < 1)
    {
        printMsg(0, "Cache bounds only apply to server. Min must be less than max and greater than zero.");
        return RM_SETTING_ERROR;
    }
    else
    {
        txCacheSize = bytes;
        txCacheMin = min;
        txCacheMax = max;
        if (isRunning)
            NormSetTransmitCacheBounds(normSession, txCacheSize, txCacheMin, txCacheMax);
        // Disable the prior line and enable the following line for NORM API v1.4b4
        //if (isRunning) NormSetTxCacheBounds (normSession, txCacheSize, txCacheMin, txCacheMax);
        printMsg(2, "CacheBounds = %d bytes, %d min, %d max", txCacheSize, txCacheMin, txCacheMax);
        return RM_OK;
    }
}

/**
 * setBackoffFactor()
 * 
 * Can be changed anytime
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::setBackoffFactor(double factor)
{
    if (factor < 0)
    {
        printMsg(0, "Backoff factor must be non-negative.");
        return RM_SETTING_ERROR;
    }
    else
    {
        backoffFactor = factor;
        if (isRunning)
            NormSetBackoffFactor(normSession, backoffFactor);
        printMsg(2, "BackoffFactor = %L", backoffFactor);
        return RM_OK;
    }
}

/**
 * setSockBufferSize()
 * Set the size of UDP socket's buffer in bytes
 * Can be set after init()
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::setSockBufferSize(int bytes)
{
    if (bytes < 0)
    {
        printMsg(0, "Socket buffer size must be non-negative.");
        return RM_SETTING_ERROR;
    }
    else
    {
        sockBufferSize = bytes;
        if (isRunning)
        {
            if (isServer)
                NormSetTxSocketBuffer(normSession, sockBufferSize);
            else
                NormSetRxSocketBuffer(normSession, sockBufferSize);
        }
        printMsg(2, "SockBufferSize = %d", sockBufferSize);
        return RM_OK;
    }
}

/**
 * setInterface()
 * Set whether the multicast messages loop back to the server
 * Must be set before init()
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::setInterface(const char *iface)
{
    if (isRunning)
    {
        printMsg(0, "Interface must be set before call to init().");
        return RM_SETTING_ERROR;
    }
    else
    {
        mcastIface = iface;
        printMsg(2, "Interface = '%s'", mcastIface);
        return RM_OK;
    }
}

/**
 * setTTL()
 * Set the TTL for multicast messages (1 = stay on LAN, 255 = global)
 * Must be set before init()
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::setTTL(int t)
{
    if (!isServer || t < 0 || t > 255 || isRunning)
    {
        printMsg(0, "TTL only applies to server, must be between 0 and 255, and must be set before call to init().");
        return RM_SETTING_ERROR;
    }
    else
    {
        ttl = t;
        printMsg(2, "TTL = %d", ttl);
        return RM_OK;
    }
}

/**
 * setBufferSpace()
 * Set the number of bytes available to the NORM protocol as a send/receive buffer
 * Must be set before init()
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::setBufferSpace(unsigned int bytes)
{
    if (isRunning)
    {
        printMsg(0, "Buffer space must be set before call to init().");
        return RM_SETTING_ERROR;
    }
    else
    {
        if (isServer)
        {
            sndBufferSpace = bytes;
            printMsg(2, "SendBufferSpace = %d", sndBufferSpace);
        }
        else
        {
            rcvBufferSpace = bytes;
            printMsg(2, "ReceiveBufferSpace = %d", rcvBufferSpace);
        }
        return RM_OK;
    }
}

/**
 * setMTU()
 * Set the MTU (max transfer unit) size
 * Must be set before init()
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::setMTU(int bytes)
{
    if (!isServer || bytes - 20 - 48 - 8 < 0 || isRunning)
    {
        printMsg(0, "MTU only applies to server, must be greater than 76 bytes, and must be set before call to init().");
        return RM_SETTING_ERROR;
    }
    else
    {
        // Account for the size of the NORM header field (48-byte max) IP header (20 bytes) and UDP header (8 bytes)
        mtu = bytes - 20 - 48 - 8;
        printMsg(2, "MTU = %d / SegmentSize = %d", bytes, mtu);
        return RM_OK;
    }
}

/**
 * setBlocksAndParity()
 * 
 * Must be set before init()
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::setBlocksAndParity(int b, int p)
{
    if (!isServer || b < 0 || p < 0 || b + p > 255 || isRunning)
    {
        printMsg(0, "Blocks/parity only apply to server, must both be non-negative, must total between 0 and 255, and must be set before call to init().");
        return RM_SETTING_ERROR;
    }
    else
    {
        blockSize = b;
        numParity = p;
        printMsg(2, "BlockSize = %d / NumParity = %d", blockSize, numParity);
        return RM_OK;
    }
}

/**
 * setMaxLength()
 * 
 * Must be set before init()
 */
Rel_Mcast::RM_Error_Type Rel_Mcast::setMaxLength(int m)
{
    if (m <= mtu || isRunning)
    {
        printMsg(0, "MaxLength must be greater than MTU [%d] and must be set before call to init().", mtu);
        return RM_SETTING_ERROR;
    }
    else
    {
        maxLength = m;
        printMsg(2, "MaxLength = %d", maxLength);
        return RM_OK;
    }
}

#endif
