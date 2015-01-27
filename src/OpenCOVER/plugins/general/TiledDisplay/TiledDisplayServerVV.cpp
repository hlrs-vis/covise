/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include "TiledDisplayServerVV.h"

#include <cover/coVRPluginSupport.h>
#include <virvo/vvtcpserver.h>

#include "TiledDisplayOGLTexQuadCompositor.h"
#include "TiledDisplayOSGTexQuadCompositor.h"

TiledDisplayServerVV::TiledDisplayServerVV(int number)
    : TiledDisplayServer(number)
{
    this->socket = 0;
    //vvDebugMsg::setDebugLevel(3);
}

TiledDisplayServerVV::~TiledDisplayServerVV()
{
    delete socket;
}

bool TiledDisplayServerVV::accept()
{
    vvTcpServer server = vvTcpServer(32340 + number);

    cerr << "TiledDisplayServerVV::connect info: waiting for client " << number << endl;

    vvTcpSocket *sock = server.nextConnection(60.0);

    if (sock != NULL)
    {
        cerr << "TiledDisplayServerVV::connect info: connection to client " << number << " established" << endl;
        this->socket = new vvSocketIO(sock);
        sock->setParameter(vvSocket::VV_BUFFSIZE, 65535);
        return true;
    }
    else
    {
        cerr << "TiledDisplayServerVV::connect err: connection to client " << number << " failed" << endl;
        return false;
    }
}

//#define TILED_DISPLAY_SERVER_TIME_RUN

void TiledDisplayServerVV::run()
{

#ifdef TILED_DISPLAY_SERVER_TIME_RUN
    Timer runTimer;
    bool startTimer = true;
#endif

    cerr << "TiledDisplayServerVV::run info: server " << number << " start server" << endl;
#ifdef TILE_ENCODE_JPEG
    cerr << "TiledDisplayServerVV::run info: server " << number << " uses JPEG encoding" << endl;
#endif
    isRunning = true;
    accept();

    char frameTimeBuffer[sizeof(opencover::cover->frameTime())];

    while (keepRunning)
    {
        if (bufferAvailable)
        {
#ifdef TILED_DISPLAY_SERVER_TIME_RUN
            if (startTimer)
            {
                runTimer.start(number);
                startTimer = false;
            }
            else
            {
                runTimer.restart(number);
            }
#endif

//cerr << "TiledDisplayServerVV::run info: server " << number << " before lock" << endl;
#ifndef TILED_DISPLAY_SYNC
            sendLock.lock();
#endif
            //cerr << "TiledDisplayServerVV::run info: server " << number << " receive image" << endl;

            socket->getData(frameTimeBuffer, sizeof(double), vvSocketIO::VV_UCHAR);
            memcpy(&frameTime, frameTimeBuffer, sizeof(double));
/*
         if (frameTime == cover->frameTime())
         {
            cerr << "TiledDisplayServerVV::run info: server" << number
                  << ": local time = " << cover->frameTime() - 1.15333e+09
                  << ", remote time = " << frameTime - 1.15333e+09
                  << ", equals: " << (frameTime == cover->frameTime())
                  << endl;
         }
         else
         {
            cerr << "TiledDisplayServerVV::run info: server" << number
                  << ": local time = " << cover->frameTime() - 1.15333e+09
                  << ", remote time = " << frameTime - 1.15333e+09
                  << ", equals: " << (frameTime == cover->frameTime())
                  << endl;
         }
*/

#ifdef TILE_ENCODE_JPEG

            unsigned int compressedDataSize = jpegImage.compressedDataSize;

            socket->getInt32(jpegImage.compressedDataSize);
            if (jpegImage.compressedDataSize == 0)
                continue;

            if (compressedDataSize < jpegImage.compressedDataSize)
            {
                delete[] jpegImage.compressedData;
                jpegImage.compressedData = new unsigned char[jpegImage.compressedDataSize];
                compressedDataSize = jpegImage.compressedDataSize;
            }

            vvSocket::ErrorType err;
            err = socket->getData(jpegImage.compressedData, jpegImage.compressedDataSize, vvSocketIO::VV_UCHAR);
            if (err != vvSocket::VV_OK)
            {
                cerr << "TiledDisplayServerVV::run err: comm error " << err
                     << " for server " << number
                     << " reading " << jpegImage.compressedDataSize << " bytes"
                     << endl;
            }
            else
            {
                //cerr << "TiledDisplayServerVV::run info: got " << jpegImage.compressedDataSize
                //      << " bytes from server " << number << endl;
            }

            if (!jpegImage.compressedData)
                continue;

            jpegDecoder.decode(jpegImage);

            int w = jpegImage.width;
            int h = jpegImage.height;

            if (dimension.width != w || dimension.height != h)
            {
                cerr << "TiledDisplayServerVV::run info: resizing pixelarray from ["
                     << dimension.width << "|" << dimension.height << "] to [" << w << "|" << h << "]"
                     << endl;
                dimension.width = w;
                dimension.height = h;

                delete[] pixels;
                pixels = new unsigned char[w * h * 3];
            }

            memcpy(pixels, jpegImage.data, w * h * 3);

#else
            int w;
            int h;

            socket->getInt32(w);
            socket->getInt32(h);

            if (dimension.width != w || dimension.height != h)
            {
                cerr << "TiledDisplayServerVV::run info: resizing pixelarray from ["
                     << dimension.width << "|" << dimension.height << "] to [" << w << "|" << h << "]"
                     << endl;
                dimension.width = w;
                dimension.height = h;

                delete[] pixels;
                pixels = new unsigned char[w * h * 4];
            }
            vvSocket::ErrorType err = socket->getData((void *)pixels, w * h * 4, vvSocketIO::VV_UCHAR);

            if (err == vvSocket::VV_OK)
            {
                //cerr << "TiledDisplayServerVV::run info: received " << w*h*4 << " bytes from client " << number << endl;
            }
            else
            {
                cerr << "TiledDisplayServerVV::run err: comm error " << err << endl;
            }

#endif // TILE_ENCODE_JPEG

            bufferAvailable = false;
            dataAvailable = true;
#ifdef TILED_DISPLAY_SYNC
            sendBarrier.block(2);
#else
            sendLock.unlock();
#endif

#ifdef TILED_DISPLAY_SERVER_TIME_RUN
            cerr << "TiledDisplayServerVV::run info: server " << number << " recv. data in " << runTimer.elapsed(number)
                 << " usec, avg. cps = " << runTimer.cps(number) << endl;
#endif
        }
        microSleep(10000);
    }

    isRunning = false;

    cerr << "TiledDisplayServerVV::run info: server " << number << " stop server" << endl;
}
