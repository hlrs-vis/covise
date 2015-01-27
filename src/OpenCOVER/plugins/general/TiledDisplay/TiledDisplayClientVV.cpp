/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef _WIN32
#include <winsock2.h>
#endif

#include <util/unixcompat.h>
#include <virvo/vvtcpsocket.h>

#include <cover/coVRPluginSupport.h>

#include "TiledDisplayClientVV.h"

TiledDisplayClientVV::TiledDisplayClientVV(int number, const std::string &compositor)
    : TiledDisplayClient(number, compositor)
{

    cerr << "TiledDisplayClientVV::<init> info: creating client " << number
         << " for compositor " << compositor << endl;
    this->socket = 0;

#ifdef TILE_ENCODE_JPEG
    this->externalPixelFormat = GL_RGB;
#else
    this->externalPixelFormat = GL_BGRA;
#endif
}

TiledDisplayClientVV::~TiledDisplayClientVV()
{
    delete socket;
}

bool TiledDisplayClientVV::connect()
{
    // FIXME Ugly Hack....
    sleep(5);

    cerr << "TiledDisplayClientVV::connect info: connecting to compositor '"
         << compositor << "' as server " << number << endl;
    vvTcpSocket *sock = new vvTcpSocket();
    this->socket = new vvSocketIO(sock);
    sock->setParameter(vvSocket::VV_BUFFSIZE, 65535);

    for (int ctr = 0; ctr < 10; ++ctr)
    {
        if (sock->connectToHost(const_cast<char *>(compositor.c_str()), 32340 + number) == vvSocket::VV_OK)
        {
            cerr << "TiledDisplayClientVV::connect info: connection for client " << number << " established" << endl;
            return true;
        }
        else
        {
            cerr << "TiledDisplayClientVV::connect err: connection for client " << number << " failed" << endl;
            sleep(3);
        }
    }

    return false;
}

void TiledDisplayClientVV::run()
{

    cerr << "TiledDisplayClientVV::run info: starting client " << number << endl;

    connect();

    vvSocket::ErrorType err;
    int bytesWritten = 0;
    (void)bytesWritten;

    while (keepRunning)
    {

        if (dataAvailable)
        {
            fillLock.lock();

            //cerr << "TiledDisplayClientVV::run info: sending" << endl;

            double ft = opencover::cover->frameTime();
            socket->putData(reinterpret_cast<unsigned char *>(&ft), sizeof(double), vvSocketIO::VV_UCHAR);

#ifdef TILE_ENCODE_JPEG
            jpegImage.data = image->data();
            jpegImage.width = width;
            jpegImage.height = height;
            jpegImage.deleteCompressedData = true;

            encoder.encode(jpegImage);

            socket->putInt32(jpegImage.compressedDataSize);
            err = socket->putData(jpegImage.compressedData, jpegImage.compressedDataSize, vvSocketIO::VV_UCHAR);
            bytesWritten = jpegImage.compressedDataSize; // Not really bytes written, just for debug output
#else
            socket->putInt32(width);
            socket->putInt32(height);
            err = socket->putData((void *)image, width * height * 4, vvSocketIO::VV_UCHAR);
            bytesWritten = width * height * 4; // Not really bytes written, just for debug output
#endif

            if (err == vvSocket::VV_OK)
            {
                //cerr << "TiledDisplayClientVV::run info: sent " << bytesWritten << " bytes for client " << number << endl;
            }
            else
            {
                cerr << "TiledDisplayClientVV::run err: comm error " << err << " for client " << number << endl;
            }
            dataAvailable = false;
            bufferAvailable = true;
            fillLock.unlock();
        }
        microSleep(10000);
    }
}
