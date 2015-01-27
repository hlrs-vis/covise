/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 * WebGL renderer                                                           *
 *                                                                          *
 * Author: Florian Niebling                                                 *
 ****************************************************************************/

/*
 * WebGL renderer features an HTTP server listening for requests for
 * COVISE objects and sending back XML CDATA consisting of javascript code 
 * that creates WebGL objects on the client.
 *
 * The client polls for new COVISE data objects on the server, receives and
 * executes javascript code containing data arrays, generates VBOs and renders.
 */
#include <errno.h>

#include <libpng12/png.h>

#include <string>
#include <fstream>
#include <algorithm>
#include <vector>
#include <iostream>
#include <iterator>

#include <stdarg.h>

#include <sys/types.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <appl/RenderInterface.h>
#include <do/coDoGeometry.h>
#include <do/coDoSet.h>
#include <do/coDoLines.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoPolygons.h>
#include <do/coDoData.h>
#include <do/coDoPixelImage.h>
#include <do/coDoTexture.h>
#include <config/CoviseConfig.h>

#include <util/coVector.h>
#include <util/coMatrix.h>
#include "WebGLRenderer.h"

#include <microhttpd.h>

#include <QResource>

#include "ft.h"

#include "debug.h"

using namespace covise;
using namespace std;

float mvm[16];
float viewIndex = 0;

int msglevel = 4;

unsigned char *createPNG1D(unsigned char *data, unsigned int *size);
unsigned char *createPNG2D(unsigned char *data, unsigned int width,
                           unsigned int height, unsigned int stride,
                           unsigned int *size);

unsigned long upperPow2(unsigned long v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

/*
 * extract substring from string after delimiter ": " (HTTP headers)
 */
std::string extractValue(std::string line)
{

    size_t start = line.find(": ");
    std::string value = line.substr(start + 2);
    return value;
}

/*
 * extract numerical ascii characters from a string and divide by
 * number of blanks inside the string (WebSocket handshake)
 */
unsigned int extractKey(std::string key)
{

    size_t start = key.find(": ");

    if (start == std::string::npos)
        return 0;

    unsigned long int space = 0;
    unsigned long int num = 0;
    for (unsigned int index = start + 2; index < key.size(); index++)
    {
        if (key.at(index) == ' ')
            space++;
        if (key.at(index) >= '0' && key.at(index) <= '9')
            num = num * 10 + (key.at(index) - '0');
    }
    return (num / space);
}

void tokenize(const string &str, vector<string> &tokens,
              const string &delimiters = " ")
{
    string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    string::size_type pos = str.find_first_of(delimiters, lastPos);

    while (string::npos != pos || string::npos != lastPos)
    {
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = str.find_first_not_of(delimiters, pos);
        pos = str.find_first_of(delimiters, lastPos);
    }
}

void WebSocketServer::run()
{
    int client_sock, sock_opt = 1;
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char *)&sock_opt, sizeof(sock_opt));

    struct sockaddr_in server_addr;

    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(32082);
    int rc;
    int first = true;
    do
    {
        rc = ::bind(sock, (sockaddr *)&server_addr, sizeof(server_addr));
        if (rc == -1)
        {
            if (first)
            {
                pmesg(1, "WebSocketServer could not bind to port\n");
                first = false;
            }
            sleep(1);
        }
    } while (rc != 0);

    rc = listen(sock, 5);
    pmesg(1, "WebSocketServer listening\n");

    while (!finished)
    {
        client_sock = accept(sock, NULL, NULL);
        WebSocketThread *t = new WebSocketThread(client_sock, renderer);
        threads.push_back(t);
    }

    close(sock);
}

WebSocketServer::WebSocketServer(WebGLRenderer *r)
    : renderer(r)
    , finished(false)
{
    start();
}

WebSocketServer::~WebSocketServer()
{
    finished = true;

    std::vector<WebSocketThread *>::iterator i;
    for (i = threads.begin(); i < threads.end(); i++)
    {
        delete *i;
    }
}

WebSocketThread::WebSocketThread(int s, WebGLRenderer *r)
    : socket(s)
    , renderer(r)
    , finished(false)
{
    start();
}

WebSocketThread::~WebSocketThread()
{
    finished = true;
}

void WebSocketThread::run()
{
    // websocket handshake
    char buf[1024];
    int r = read(socket, buf, 1024);
    (void)r;
    std::vector<std::string> headers;
    tokenize(std::string(buf), headers, std::string("\r\n"));

    unsigned int key1, key2;
    std::string origin, host;

    for (int index = 0; index < headers.size(); index++)
    {

        if (headers[index].find("Sec-WebSocket-Key1: ") == 0)
            key1 = extractKey(headers[index]);

        if (headers[index].find("Sec-WebSocket-Key2: ") == 0)
            key2 = extractKey(headers[index]);

        if (headers[index].find("Host: ") == 0)
            host = extractValue(headers[index]);

        if (headers[index].find("Origin: ") == 0)
            origin = extractValue(headers[index]);
    }

    // handshake reply must be the 16 byte md5sum of the 16 bytes
    //   key1(4byte) key2(4byte) key3(8byte after end of headers)
    // key1 and key2 are big-endian

    unsigned char keys[16];
    unsigned char keys_md5[16];

    keys[0] = (unsigned char)(0xff & (key1 >> 24));
    keys[1] = (unsigned char)(0xff & (key1 >> 16));
    keys[2] = (unsigned char)(0xff & (key1 >> 8));
    keys[3] = (unsigned char)(0xff & (key1));

    keys[4] = (unsigned char)(0xff & (key2 >> 24));
    keys[5] = (unsigned char)(0xff & (key2 >> 16));
    keys[6] = (unsigned char)(0xff & (key2 >> 8));
    keys[7] = (unsigned char)(0xff & (key2));

    for (int index = 0; index < 8; index++)
        keys[8 + index] = headers[headers.size() - 1].at(index);

    md5_state_t md5state;
    md5_init(&md5state);
    md5_append(&md5state, keys, sizeof(keys));
    md5_finish(&md5state, keys_md5);

    stringstream reply;
    reply << "HTTP/1.1 101 WebSocket Protocol Handshake\r\n"
             "Upgrade: WebSocket\r\n"
             "Connection: Upgrade\r\n"
             "Sec-WebSocket-Location: ws://" << host << "/\r\n"
                                                        "Sec-WebSocket-Origin: " << origin << "\r\n\r\n" << keys_md5;

    std::string rep = reply.str();
    write(socket, rep.c_str(), rep.size());
    fsync(socket);
    // WebSocket handshake done

    // receive "Hello" message and send it back
    r = read(socket, buf, 1024);
    write(socket, buf, r);
    fsync(socket);

    const unsigned char frame_start = 0;
    const unsigned char frame_end = 255;

    list<string> tokens;

    // data exchange
    while (!finished)
    {

        bool changed = false;

        map<const char *, Object *, ltstr>::iterator oi;

        ostringstream stream;
        list<string> remove;
        renderer->objectMutex.lock();
        list<string>::iterator ri;
        // delete objects that the client has that are no longer on the server
        for (ri = tokens.begin(); ri != tokens.end(); ri++)
            if (renderer->objects.find((*ri).c_str()) == renderer->objects.end())
            {
                remove.push_back(*ri);
                ri = tokens.erase(ri);
            }
        stream << "<objects tMin='" << renderer->tMin << "' "
               << "tMax='" << renderer->tMax << "' "
               << "view='" << viewIndex << "'>";
        for (ri = remove.begin(); ri != remove.end(); ri++)
        {
            stream << "<obj name='" << *ri << "'><![CDATA[deleteObject('"
                   << *ri << "')]]></obj>";
            changed = true;
        }

        // add objects that the client does not already have
        for (oi = renderer->objects.begin(); oi != renderer->objects.end(); oi++)
        {
            bool have = false;
            list<string>::iterator havei;
            for (havei = tokens.begin(); havei != tokens.end(); havei++)
                if (!strcmp((*havei).c_str(), oi->first))
                    have = true;

            if (!have)
            {
                if (oi->second->stream)
                {
                    pmesg(3, "WebSocketThread sending object [%s]\n",
                          oi->first);
                    stream << "<obj name='dummy'><![CDATA[deleteObject('" << oi->first << "')]]></obj>";
                    stream << oi->second->stream->str();
                    tokens.push_back(oi->second->name);
                    changed = true;
                }
                else
                    pmesg(3, "WebSocketThread could not send object [%s]\n",
                          oi->first);
            }
        }
        renderer->objectMutex.unlock();

        stream << "</objects>" << endl;
        string s = stream.str();

        if (changed)
        {
            write(socket, &frame_start, 1);
            write(socket, s.c_str(), s.size());
            write(socket, &frame_end, 1);
            fsync(socket);
        }
        renderer->changeCondition->wait(renderer->changeMutex, 1000);
        renderer->changeMutex->unlock();
    }
    close(socket);
}

/*
 * Make the contents of a file available via the webserver under the names
 * stored in varargs.
 */
void WebGLRenderer::registerFile(const char *fileName,
                                 const char *mimeType, ...)
{
    pmesg(4, "WebGLRenderer::registerFile [%s]", fileName);
    va_list l;
    struct file f;

    QResource resource(fileName);
    if (resource.isValid())
    {
        f.buf = (unsigned char *)resource.data();
        f.size = resource.size();
        f.mimeType = mimeType;

        va_start(l, mimeType);
        for (;;)
        {
            const char *name = va_arg(l, const char *);
            if (!name)
                break;
            files[name] = f;
            pmesg(4, " -> [%s] ", name);
        }
        va_end(l);
    }
    else
        pmesg(4, " does not exist.");

    pmesg(4, "\n");
}

void WebGLRenderer::registerBuffer(struct file f, const char *mimeType, ...)
{
    pmesg(4, "WebGLRenderer::registerBuffer [%s]", f.mimeType);
    va_list l;
    va_start(l, mimeType);
    for (;;)
    {
        const char *name = va_arg(l, const char *);
        if (!name)
            break;
        files[name] = f;
        pmesg(4, " -> [%s] ", name);
    }
    va_end(l);
    pmesg(4, "\n");
}

void WebGLRenderer::addFileBufferToObject(const char *name, unsigned char *buf)
{
    pmesg(5, "WebGLRenderer::addFileBufferToObject [%s]\n", name);
    std::vector<unsigned char *> *v = NULL;
    map<const char *, std::vector<unsigned char *> *, ltstr>::iterator i = objectFiles.find(name);
    if (i == objectFiles.end())
    {
        v = new std::vector<unsigned char *>;
        objectFiles[name] = v;
    }
    v->push_back(buf);
}

void WebGLRenderer::deleteFileBuffers(const char *name)
{
    pmesg(5, "WebGLRenderer::deleteFileBuffers [%s]\n", name);
    std::vector<unsigned char *> *v = NULL;
    map<const char *, std::vector<unsigned char *> *, ltstr>::iterator i = objectFiles.find(name);
    if (i != objectFiles.end())
    {
        v = i->second;
        std::vector<unsigned char *>::iterator vi;
        for (vi = v->begin(); vi != v->end(); vi++)
            delete (*vi);

        objectFiles.erase(name);
        delete v;
    }
}

/*
 * HTTP handler function
 * handles requests for: - files registered in WebGLRenderer files
 *                       - requests for COVISE objects ( /getdata )
 *
 * getdata requests look like this: /getdata?objects=o1,o2,o3 where
 * o1, o2 and o3 are the names of the COVISE objects that the client already
 * has. handler answers with an xml message:
 * <objects ts="number of timesteps">
 *   <obj name="COVISE object name"><![CDATA[...javascript code...]]/>
 * </objects>
 * 
 * The javascript code that represents COVISE objects is built in 
 * WebGLRenderer::addGeometry
 */
static int handler(void *cls, struct MHD_Connection *connection,
                   const char *url, const char * /*method*/,
                   const char * /*version*/,
                   const char * /*upload_data*/,
#if MHD_VERSION > 0x00040001
                   size_t * /*upload_data_size*/,
#else
                   unsigned int * /*upload_data_size*/,
#endif
                   void **ptr)
{
    WebGLRenderer *renderer = (WebGLRenderer *)cls;
    struct MHD_Response *response = NULL;
    int ret = MHD_NO;
    static int dummy;

    pmesg(6, "WebGLRenderer::handler [%s]\n", url);

    if (&dummy != *ptr)
    {
        // The first time only the headers are valid
        // do not respond in the first round...
        *ptr = &dummy;
        return MHD_YES;
    }
    *ptr = NULL; // clear context pointer

    if (!strcmp(url, "/view"))
    {
        const char *view = MHD_lookup_connection_value(connection,
                                                       MHD_GET_ARGUMENT_KIND,
                                                       "mvm");
        float v[16];
        if (view && sscanf(view, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8], &v[9], &v[10], &v[11], &v[12], &v[13], &v[14], &v[15]) == 16)
        {
            memcpy(mvm, v, 16 * sizeof(float));
            printf("view: %f %f %f\n", mvm[0], mvm[1], mvm[2]);
            viewIndex++;
        }
        ostringstream stream;

        stream << "<mvm index='" << viewIndex << "' view='view=$M([["
               << mvm[0] << "," << mvm[1] << "," << mvm[2] << "," << mvm[3] << "],["
               << mvm[4] << "," << mvm[5] << "," << mvm[6] << "," << mvm[7] << "],["
               << mvm[8] << "," << mvm[9] << "," << mvm[10] << "," << mvm[11] << "],["
               << mvm[12] << "," << mvm[13] << "," << mvm[14] << "," << mvm[15] << "]])'/>";

        string s = stream.str();
        response = MHD_create_response_from_data(s.length(),
                                                 (void *)s.c_str(),
                                                 MHD_NO,
                                                 MHD_YES);
    }
    else if (!strcmp(url, "/interactor"))
    {
        const char *name = MHD_lookup_connection_value(connection,
                                                       MHD_GET_ARGUMENT_KIND,
                                                       "name");

        const char *value = MHD_lookup_connection_value(connection,
                                                        MHD_GET_ARGUMENT_KIND,
                                                        "value");

        pmesg(3, "WebGLRenderer::handler /interactor [%s] [%s]\n", name, value);

        if (name)
        {
            // name!param
            vector<string> feed;
            tokenize(name, feed, "!");

            std::map<string, string>::iterator i = feedbackinfo.find(feed[0]);
            if (value && i != feedbackinfo.end())
            {
                ostringstream stream;
                stream << feed[1] << "\nFloatVector\n" << value;

                CoviseRender::set_feedback_info(feedbackinfo[feed[0]].c_str());
                CoviseRender::send_feedback_message("PARAM", stream.str().c_str());

                CoviseRender::set_feedback_info(feedbackinfo[feed[0]].c_str());
                char c = '\0';
                CoviseRender::send_feedback_message("EXEC", &c);
            }
            else
            {
                // labels are currently deleted & generated again when their
                // interactor is moved !
                renderer->objectMutex.lock();

                int len = feed[0].size() + 2;
                char *labelName = new char[len];
                string labelText;

                memset(labelName, 0, len);
                int index = 0;

                std::map<const char *, Object *, ltstr>::iterator ri = renderer->objects.find(feed[0].c_str());
                if (ri != renderer->objects.end())
                {
                    std::string s(feed[0]);
                    size_t pos = s.rfind("_");
                    if (pos != std::string::npos)
                    {
                        sscanf(feed[0].c_str() + pos + 1, "%d", &index);
                        index++;
                        memcpy(labelName, feed[0].c_str(), pos);
                        labelText = labelName;
                        sprintf(labelName + pos, "_%d", index);
                    }
                    else
                    {
                        labelText = feed[0];
                        sprintf(labelName, "%s_0", feed[0].c_str());
                    }
                }
                else
                {
                    labelText = feed[0];
                    sprintf(labelName, "%s_0", feed[0].c_str());
                }

                std::map<const char *, Object *, ltstr>::iterator i = renderer->objects.find(feed[0].c_str());

                if (i != renderer->objects.end())
                {
                    renderer->objects.erase(i);
                    delete i->first;
                    delete i->second;
                }

                ostringstream *label = new ostringstream;

                FT ft;
                ft_pixmap *pix = ft.createPixmap();
                ft_string *string = ft.createString(pix, labelText.c_str());
                ft.drawString(pix, string);
                float w = pix->width / 200.0;
                float h = pix->height / 200.0;
                int width = upperPow2(pix->width);
                int height = upperPow2(pix->height);

                *label << "<obj name=\"" << labelName << "\" type=\"polygons\">";
                *label << "<![CDATA[var a = new Object(); a['type'] = 'triangles';"
                          " a['timestep'] = -1; a['vertices'] = [ "
                       << -w << ", " << -h << ", 0.0, "
                       << w << ", " << -h << ", 0.0, "
                       << w << ", " << h << ", 0.0, "
                       << -w << ", " << h << ", 0.0 ]; "
                                             " a['indices'] = [ 0, 1, 3, 1, 2, 3 ]; a['normals'] = [ 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0 ]; a['name'] = \"" << labelName << "\"; a['alpha'] = \"1\"; ";

                *label << " gl.enable(gl.TEXTURE_2D); a['texture'] = gl.createTexture();"
                          " gl.bindTexture(gl.TEXTURE_2D, a['texture']);"
                          " gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);"
                          " gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);"
                          " gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP);"
                          " gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP);"
                          " a['texdata'] = [ ";

                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++)
                    {
                        if (x >= pix->width || y >= pix->height)
                            *label << " 0, 0, 0, 0, ";
                        else
                            *label << (int)pix->buffer[x * 4 + y * pix->rowstride]
                                   << ", "
                                   << (int)pix->buffer[x * 4 + y * pix->rowstride + 1]
                                   << ", "
                                   << (int)pix->buffer[x * 4 + y * pix->rowstride + 2]
                                   << ", "
                                   << (int)(pix->buffer[x * 4 + y * pix->rowstride + 3] / 2.0) << ", ";
                    }

                float wr = pix->width / (float)width;
                float hr = pix->height / (float)height;

                *label << "]; a['texcoords'] = [ 0.0, " << hr << ", " << wr << ", " << hr << ", " << wr << ", 0.0, 0.0, 0.0 ]; "
                                                                                                           "gl.bindTexture(gl.TEXTURE_2D, a['texture']); "
                                                                                                           "gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, " << width << ", " << height << ", 0, gl.RGBA, gl.UNSIGNED_BYTE, new WebGLByteArray(a['texdata'])); objects['" << labelName << "'] = a; ";

                float x = 0, y = 0, z = 0;
                if (value)
                    sscanf(value, "%f %f %f", &x, &y, &z);

                *label << "addInteractor(\"XLabel\", \"" << labelName << "\", ['point', 'FLOVEC', '3 " << x << " " << y << " " << z << "']); ";
                *label << "]]></obj>";

                delete string;
                delete pix;

                Object *o = new Object(labelName, label, true, -1);
                renderer->objects[labelName] = o;

                renderer->objectMutex.unlock();
            }
            response = MHD_create_response_from_data(5, (void *)"<ok/>",
                                                     MHD_NO, MHD_NO);
        }
    }
    else if (!strcmp(url, "/getdata"))
    {
        // request for data objects
        const char *obj = MHD_lookup_connection_value(connection,
                                                      MHD_GET_ARGUMENT_KIND,
                                                      "objects");

        ostringstream stream;

        vector<string> remove;
        vector<string> tokens;

        if (obj)
            tokenize(obj, tokens, ",");

        // check if there are objects that the client does not yet have
        // or objects that the clients should delete. if the client is
        // up-to-date, wait for changes on renderer->condition.
        bool changed = false;
        renderer->objectMutex.lock();

        // deleted objects?
        vector<string>::iterator ri;
        for (ri = tokens.begin(); ri != tokens.end(); ri++)
            if (renderer->objects.find((*ri).c_str()) == renderer->objects.end())
            {
                changed = true;
                break;
            }

        map<const char *, Object *, ltstr>::iterator oi;

        for (oi = renderer->objects.begin(); oi != renderer->objects.end(); oi++)
        {
            bool have = false;
            vector<string>::iterator havei;
            for (havei = tokens.begin(); havei != tokens.end(); havei++)
            {
                if (!strcmp((*havei).c_str(), oi->first))
                    have = true;
            }
            if (!have)
                changed = true;
        }
        /*
      renderer->objectMutex.unlock();


      // if there are no changes, wait
      if (!changed)
         renderer->changeCondition->wait(renderer->changeMutex);
      renderer->changeMutex->unlock();

      renderer->objectMutex.lock();
      */
        // delete objects that the client has that are no longer on the server
        for (ri = tokens.begin(); ri != tokens.end(); ri++)
            if (renderer->objects.find((*ri).c_str()) == renderer->objects.end())
                remove.push_back(*ri);
        stream << "<objects tMin='" << renderer->tMin << "' "
               << "tMax='" << renderer->tMax << "' "
               << "view='" << viewIndex << "'>";

        /*
      stream << " mvm='view=$M([["
        << mvm[0] << "," << mvm[1] << "," << mvm[2] << "," << mvm[3] << "],["
        << mvm[4] << "," << mvm[5] << "," << mvm[6] << "," << mvm[7] << "],["
        << mvm[8] << "," << mvm[9] << "," << mvm[10] << "," << mvm[11] << "],["
        << mvm[12] << "," << mvm[13] << "," << mvm[14] << "," << mvm[15] << "]])'>";
*/
        for (ri = remove.begin(); ri != remove.end(); ri++)
            stream << "<obj name='" << *ri << "'><![CDATA[deleteObject('"
                   << *ri << "')]]></obj>";

        // add objects that the client does not already have
        for (oi = renderer->objects.begin(); oi != renderer->objects.end(); oi++)
        {
            bool have = false;
            vector<string>::iterator havei;
            for (havei = tokens.begin(); havei != tokens.end(); havei++)
                if (!strcmp((*havei).c_str(), oi->first))
                    have = true;

            if (!have)
            {
                if (oi->second->stream)
                {
                    pmesg(3, "WebGLRenderer::handler sending object [%s]\n",
                          oi->first);
                    stream << "<obj name='dummy'><![CDATA[deleteObject('" << oi->first << "')]]></obj>";
                    stream << oi->second->stream->str();
                }
                else
                    pmesg(3, "WebGLRenderer::handler could not send object [%s]\n",
                          oi->first);
            }
        }
        renderer->objectMutex.unlock();

        stream << "</objects>" << endl;
        string s = stream.str();

        response = MHD_create_response_from_data(s.length(),
                                                 (void *)s.c_str(),
                                                 MHD_NO,
                                                 MHD_YES);
    }
    else
    {
        // request for files
        map<const char *, struct file>::iterator fi;

        if ((fi = renderer->files.find(url)) != renderer->files.end())
        {
            pmesg(3, "WebGLRenderer::handler sent file [%s]\n", url);
            response = MHD_create_response_from_data(fi->second.size,
                                                     (void *)fi->second.buf,
                                                     MHD_NO, MHD_NO);
            MHD_add_response_header(response, "Content-Type", fi->second.mimeType);
        }
        else
            pmesg(3, "WebGLRenderer::handler could not send file [%s]\n", url);
    }

    if (response)
    {
        ret = MHD_queue_response(connection, MHD_HTTP_OK, response);
        MHD_destroy_response(response);
    }
    return ret;
}

int main(int argc, char *argv[])
{
    WebGLRenderer *renderer = new WebGLRenderer(argc, argv);
    renderer->run();

    return 0;
}

WebGLRenderer::WebGLRenderer(int argc, char *argv[])
{
    CoviseRender::set_module_description("WebGL");
    CoviseRender::add_port(INPUT_PORT, "RenderData",
                           "Geometry|Polygons|Lines|TriangleStrips",
                           "render geometry");

    CoviseRender::add_port(PARIN, "filename", "Browser", "Output file name");
    CoviseRender::add_port(PARIN, "renderToFile", "Boolean", "Render the scene");

    CoviseRender::set_render_callback(WebGLRenderer::renderCallback, this);
    CoviseRender::set_master_switch_callback(WebGLRenderer::masterSwitchCallback, this);
    CoviseRender::set_quit_callback(WebGLRenderer::quitCallback, this);
    CoviseRender::set_add_object_callback(WebGLRenderer::addObjectCallback, this);
    CoviseRender::set_delete_object_callback(WebGLRenderer::deleteObjectCallback, this);
    CoviseRender::set_param_callback(WebGLRenderer::paramCallback, this);
    CoviseRender::set_custom_callback(WebGLRenderer::doCustomCallback, this);

    static char defVal[256];
    sprintf(defVal, "data.xml");
    CoviseRender::set_port_default("filename", defVal);
    CoviseRender::set_port_default("renderToFile", "FALSE");

    CoviseRender::init(argc, argv);

    revisionID = 1;

    registerFile(":/index.html", "text/html",
                 "/index.html", "/", 0);
    registerFile(":/control.js", "application/x-javascript",
                 "/control.js", 0);
    registerFile(":/sylvester.js", "application/x-javascript",
                 "/sylvester.js", 0);
    registerFile(":/glUtils.js", "application/x-javascript",
                 "/glUtils.js", 0);
    registerFile(":/firebug.jgz", "application/x-javascript",
                 "/firebug.jgz", 0);
    registerFile(":/logo.png", "image/png",
                 "/logo.png", 0);
    registerFile(":/yui/fonts/fonts-min.css", "text/css",
                 "/yui/fonts/fonts-min.css", 0);
    registerFile(":/yui/slider/assets/skins/sam/slider.css", "text/css",
                 "/yui/slider/assets/skins/sam/slider.css", 0);
    registerFile(":/yui/yahoo-dom-event/yahoo-dom-event.js", "application/x-javascript",
                 "/yui/yahoo-dom-event/yahoo-dom-event.js", 0);
    registerFile(":/yui/dragdrop/dragdrop-min.js", "application/x-javascript",
                 "/yui/dragdrop/dragdrop-min.js", 0);
    registerFile(":/yui/slider/slider-min.js", "application/x-javascript",
                 "/yui/slider/slider-min.js", 0);
    registerFile(":/yui/slider/assets/bg-h.gif", "image/gif",
                 "/yui/slider/assets/bg-h.gif", "/yui/slider/assets/skins/sam/bg-h.gif", 0);
    registerFile(":/yui/slider/assets/thumb-n.gif", "image/gif",
                 "/yui/slider/assets/thumb-n.gif", 0);

    tMin = 0;
    tMax = 0;

    int port = coCoviseConfig::getInt("Module.WebGL.Port", 32080);

    mvm[0] = 1;
    mvm[5] = 1;
    mvm[10] = 1;
    mvm[15] = 1;

    changeCondition = new OpenThreads::Condition;
    changeMutex = new OpenThreads::Mutex;

    daemon = MHD_start_daemon(MHD_USE_SELECT_INTERNALLY,
                              port, NULL, NULL, &handler, this,
                              MHD_OPTION_END);

    new WebSocketServer(this);
}

WebGLRenderer::~WebGLRenderer()
{
    std::map<const char *, std::vector<unsigned char *> *>::iterator i;
    for (i = objectFiles.begin(); i != objectFiles.end(); i++)
        deleteFileBuffers(i->first);
}

void WebGLRenderer::run()
{
    CoviseRender::main_loop();
}

//
// static stub callback functions calling the real class
// member functions
//
void WebGLRenderer::quitCallback(void *userData, void *callbackData)
{
    WebGLRenderer *thisRenderer = (WebGLRenderer *)userData;
    thisRenderer->quit(callbackData);
}

void WebGLRenderer::addObjectCallback(void *userData, void *callbackData)
{
    WebGLRenderer *thisRenderer = (WebGLRenderer *)userData;
    thisRenderer->addObject(callbackData);
}

void WebGLRenderer::deleteObjectCallback(void *userData, void *callbackData)
{
    WebGLRenderer *thisRenderer = (WebGLRenderer *)userData;
    thisRenderer->deleteObject(callbackData);
}

void WebGLRenderer::masterSwitchCallback(void *userData, void *callbackData)
{
    WebGLRenderer *thisRenderer = (WebGLRenderer *)userData;
    thisRenderer->masterSwitch(callbackData);
}

void WebGLRenderer::renderCallback(void *userData, void *callbackData)
{
    WebGLRenderer *thisRenderer = (WebGLRenderer *)userData;
    thisRenderer->render(callbackData);
}

void WebGLRenderer::paramCallback(bool inMapLoading, void *userData, void *callbackData)
{
    WebGLRenderer *thisRenderer = (WebGLRenderer *)userData;
    thisRenderer->param(inMapLoading, CoviseRender::get_reply_param_name(), callbackData);
}

void WebGLRenderer::doCustomCallback(void *userData, void *callbackData)
{
    WebGLRenderer *thisRenderer = (WebGLRenderer *)userData;
    thisRenderer->doCustom(callbackData);
}

void WebGLRenderer::quit(void * /*callbackData*/)
{
    CoviseRender::sendInfo("Quitting now");
}

void WebGLRenderer::param(bool inMapLoading, const char *paraName,
                          void * /*callbackData*/)
{
    if (inMapLoading)
        return;

    int render = 0;
    const char *filename = NULL;

    if (!strcmp(paraName, "renderToFile"))
        CoviseBase::get_reply_boolean(&render);

    if (!strcmp(paraName, "filename"))
    {
        CoviseBase::get_reply_string(&filename);
        fileName = string(filename);
    }

    if (render)
    {
        pmesg(4, "WebGLRenderer::param savefile [%s]\n", fileName.c_str());
        ofstream file(fileName.c_str());

        objectMutex.lock();
        map<const char *, Object *, ltstr>::iterator obji;
        file << "<objects>";
        for (obji = objects.begin(); obji != objects.end(); obji++)
        {
            if (obji->second->stream)
            {
                pmesg(2, "WebGLRenderer::param writing object [%s]\n", obji->first);
                file << obji->second->stream->str() << endl;
            }
            else
                pmesg(2, "WebGLRenderer::param could not write object [%s]\n",
                      obji->first);
        }
        file << "</objects>";
        objectMutex.unlock();

        file.close();
    }
}

void WebGLRenderer::addObject(void * /*callbackData*/)
{
    objectMutex.lock();
    revisionID++;

    CoviseRender::sendInfo("Adding object %s", CoviseRender::get_object_name());

    const coDistributedObject *data_obj = coDistributedObject::createFromShm(CoviseRender::get_object_name());
    if (data_obj != NULL)
        addObject(data_obj, NULL, NULL, NULL);

    std::map<const char *, Object *, ltstr>::iterator i;
    for (i = objects.begin(); i != objects.end(); i++)
    {
        RenderObject *o = dynamic_cast<RenderObject *>(i->second);
        if (o && !o->added)
        {
            o->stream = addGeometry(o->geometry, o->normals, o->colors,
                                    o->texture, i->first, o->timeStep);
            o->added = true;
        }
    }
    objectMutex.unlock();
}

void WebGLRenderer::addObject(const coDistributedObject *geo,
                              const coDistributedObject *col,
                              const coDistributedObject *norm,
                              const coDistributedObject *tex,
                              int timeStep)
{
    pmesg(3, "WebGLRenderer::addObject %p %p %p %p (%d)\n",
          geo, col, norm, tex, timeStep);

    const coDoGeometry *geometry;
    const coDoSet *set;

    const coDoPolygons *polygons;
    const coDoLines *lines;
    const coDoTriangleStrips *strips;

    if ((set = dynamic_cast<const coDoSet *>(geo)))
    {
        timeStep = -1;
        const char *tstep_attrib = geo->getAttribute("TIMESTEP");
        const coDoSet *cset = dynamic_cast<const coDoSet *>(col);
        const coDoSet *nset = dynamic_cast<const coDoSet *>(norm);
        const coDoSet *tset = dynamic_cast<const coDoSet *>(tex);

        int min, max;
        if (tstep_attrib && sscanf(tstep_attrib, "%d %d", &min, &max) == 2)
        {
            if (tMin == 0 || tMin < min)
                tMin = min;
            if (tMax == 0 || tMax > max)
                tMax = max;

            timeStep = min;
        }
        pmesg(3, "WebGLRenderer::addObject timestep [%s]\n", tstep_attrib);

        int num = set->getNumElements();
        for (int index = 0; index < num; index++)
        {
            const coDistributedObject *c = NULL;
            const coDistributedObject *n = NULL;
            const coDistributedObject *t = NULL;
            if (cset)
                c = cset->getElement(index);
            if (nset)
                n = nset->getElement(index);
            if (tset)
                t = tset->getElement(index);

            addObject(set->getElement(index), c, n, t, timeStep);
            if (timeStep != -1)
                timeStep++;
        }
    }
    else if ((geometry = dynamic_cast<const coDoGeometry *>(geo)))
    {
        if (geometry->objectOk())
        {
            if ((set = dynamic_cast<const coDoSet *>(geometry->getGeometry())))
            {
                const coDoSet *c = dynamic_cast<const coDoSet *>(geometry->getColors());
                const coDoSet *n = dynamic_cast<const coDoSet *>(geometry->getNormals());
                const coDoSet *t = dynamic_cast<const coDoSet *>(geometry->getTexture());
                addObject(set, c, n, t, timeStep);
            }
            else
            {
                char *name = new char[strlen(geometry->getGeometry()->getName()) + 13];
                sprintf(name, "%s_%d", geometry->getGeometry()->getName(),
                        revisionID);

                RenderObject *o = new RenderObject(name,
                                                   geometry->getGeometry(),
                                                   geometry->getColors(),
                                                   geometry->getNormals(),
                                                   geometry->getTexture(),
                                                   timeStep);

                objects[name] = o;
                revName[geometry->getGeometry()->getName()] = name;

                addToGroup(CoviseRender::get_object_name(), name);
            }
        }
    }
    else if ((strips = dynamic_cast<const coDoTriangleStrips *>(geo)))
    {
        char *name = new char[strlen(strips->getName()) + 13];
        sprintf(name, "%s_%d", strips->getName(), revisionID);
        RenderObject *o = new RenderObject(name, strips, col, norm, tex, timeStep);
        objects[name] = o;
        revName[strips->getName()] = name;
        addToGroup(CoviseRender::get_object_name(), name);
    }
    else if ((polygons = dynamic_cast<const coDoPolygons *>(geo)))
    {
        char *name = new char[strlen(polygons->getName()) + 13];
        sprintf(name, "%s_%d", polygons->getName(), revisionID);
        RenderObject *o = new RenderObject(name, polygons, col, norm, tex, timeStep);
        objects[name] = o;
        revName[polygons->getName()] = name;
        addToGroup(CoviseRender::get_object_name(), name);
    }
    else if ((lines = dynamic_cast<const coDoLines *>(geo)))
    {
        char *name = new char[strlen(lines->getName()) + 13];
        sprintf(name, "%s_%d", lines->getName(), revisionID);
        RenderObject *o = new RenderObject(name, lines, col, norm, tex, timeStep);
        objects[name] = o;
        revName[lines->getName()] = name;
        addToGroup(CoviseRender::get_object_name(), name);
    }
    // wakeup all clients waiting for changes
    changeCondition->broadcast();
}

void WebGLRenderer::addToGroup(const char *gname, const char *oname)
{
    vector<string> *v = NULL;
    std::map<const char *, vector<string> *, ltstr>::iterator i = groups.find(gname);

    if (i == groups.end())
    {
        v = new vector<string>;
        groups[strdup(gname)] = v;
        pmesg(5, "WebGLRenderer::addToGroup new group [%s] [%p]\n", gname, v);
    }
    else
    {
        v = (*i).second;
        pmesg(5, "WebGLRenderer::addToGroup using group [%p] [%s]\n", v, gname);
    }
    v->push_back(strdup(oname));
    pmesg(5, "WebGLRenderer::addToGroup added [%s] to [%s]\n", oname, gname);
}

void WebGLRenderer::deleteObject(void *callbackData)
{
    const coDistributedObject *data_obj = coDistributedObject::createFromShm(CoviseRender::get_object_name());

    if (data_obj != NULL)
        deleteObject(data_obj);
    else if (callbackData)
    {
        objectMutex.lock();
        map<const char *, const char *, ltstr>::iterator i = revName.find((char *)callbackData);
        if (i != revName.end())
        {
            map<const char *, Object *, ltstr>::iterator oi = objects.find(i->second);
            if (oi != objects.end())
            {
                deleteFileBuffers(i->second);
                delete (oi->first);
                delete (oi->second);
                objects.erase(oi);
            }
        }
        else
        {
            map<const char *, vector<string> *, ltstr>::iterator vi = groups.find((char *)callbackData);
            if (vi != groups.end())
            {
                vector<string>::iterator i;
                for (i = vi->second->begin(); i != vi->second->end(); i++)
                {
                    map<const char *, Object *, ltstr>::iterator oi = objects.find(i->c_str());
                    if (oi != objects.end())
                    {
                        deleteFileBuffers(i->c_str());
                        delete (oi->first);
                        delete (oi->second);
                        objects.erase(oi);
                    }
                }
            }
        }
        objectMutex.unlock();
    }
    // wakeup all clients waiting for changes
    changeCondition->broadcast();
}

void WebGLRenderer::deleteObject(const coDistributedObject *object)
{
    const coDoSet *set;
    const coDoGeometry *geometry;

    if ((set = dynamic_cast<const coDoSet *>(object)))
    {
        int num = set->getNumElements();
        for (int index = 0; index < num; index++)
            deleteObject(set->getElement(index));
    }
    else if ((geometry = dynamic_cast<const coDoGeometry *>(object)))
    {
        if (geometry->objectOk())
            deleteObject(geometry->getGeometry());
    }
    else
    {
        const char *name = revName[object->getName()];
        map<const char *, Object *, ltstr>::iterator oi = objects.find(name);
        if (oi != objects.end())
        {
            deleteFileBuffers(name);
            feedbackinfo.erase(name);
            pmesg(4, "WebGLRenderer::deleteObject [%s]\n", object->getName());
            delete (oi->first);
            delete (oi->second);
            objects.erase(oi);
        }
    }

    // wakeup all clients waiting for changes
    changeCondition->broadcast();
}

void WebGLRenderer::render(void * /*callbackData*/)
{
}

void WebGLRenderer::masterSwitch(void * /*callbackData*/)
{
}

void WebGLRenderer::doCustom(void * /*callbackData*/)
{
}

/*
 * create javascript code that represents COVISE objects.
 * objects look like this in javascript:
 *   objects["COVISE object name"]["type"] = "triangles" or "lines"
 *   objects["COVISE object name"]["timestep"] = -1 or number of timestep
 *   objects["COVISE object name"]["texture"] = gl.createTexture(); ...
 *   objects["COVISE object name"]["texcoords"] = [ ... ];
 *   objects["COVISE object name"]["colors"] = [ ... ];
 *   objects["COVISE object name"]["vertices"] = [ ... ];
 *   objects["COVISE object name"]["indices"] = [ ... ];
 */
ostringstream *WebGLRenderer::addGeometry(const coDistributedObject *geometry,
                                          const coDistributedObject *colors,
                                          const coDistributedObject *normals,
                                          const coDistributedObject *text,
                                          const char *name,
                                          int timeStep)
{
    ostringstream *str = NULL;
    ostringstream color;
    ostringstream tex;
    ostringstream interactor;

    const coDoPolygons *polygons = NULL;
    const coDoTriangleStrips *triangles = NULL;
    const coDoLines *lines = NULL;

    int numPoints = -1;

    if ((polygons = dynamic_cast<const coDoPolygons *>(geometry)))
        numPoints = polygons->getNumPoints();
    else if ((triangles = dynamic_cast<const coDoTriangleStrips *>(geometry)))
        numPoints = triangles->getNumPoints();
    else if ((lines = dynamic_cast<const coDoLines *>(geometry)))
        numPoints = lines->getNumPoints();

    if (numPoints > USHRT_MAX)
    {
        pmesg(1, "WebGLRenderer::addGeometry ERROR in object [%s]"
                 ": more than USHRT_MAX indices\n",
              name);
        return NULL;
    }

    const coDoRGBA *rgba = dynamic_cast<const coDoRGBA *>(colors);
    const coDoTexture *texture = dynamic_cast<const coDoTexture *>(text);

    const char *feedback = geometry->getAttribute("INTERACTOR");
    printf("feedback\n%s\n", feedback);

    if (feedback)
    {
        pmesg(6, "WebGLRenderer::addGeometry feedback [%s]\n", feedback);
        vector<string> l;
        vector<string> p;

        tokenize(feedback, l, "\n");

        if (l.size() >= 5 && l[4].find("coFeedback:") == 0)
        {
            string f = l[0] + "\n" + l[1] + "\n" + l[2] + "\n";
            feedbackinfo[name] = f;

            istringstream str(l[4]);
            int numPara, numUser;

            l[4].erase(0, strlen("coFeedback:"));
            str >> numPara >> numUser;
            tokenize(l[4], p, "!");

            interactor << "addInteractor(\"" << l[0] << "\", \""
                       << name << "\"";

            for (int index = 1; index < (p.size() - 2); index += 3)
            {
                interactor << ", ['" << p[index]
                           << "', '" << p[index + 1]
                           << "', '" << p[index + 2] << "'] ";
            }
            interactor << "); ";

            //pmesg(2, "WebGLRenderer::addGeometry interactor [%s]\n", interactor.str().c_str());
        }
    }

    int colorBinding = CO_NONE;

    if (rgba)
    {
        if (rgba->getNumPoints() == 1)
            colorBinding = CO_OVERALL;
        else if (rgba->getNumPoints() == numPoints)
            colorBinding = CO_PER_VERTEX;
        else
            colorBinding = CO_PER_FACE;

        switch (colorBinding)
        {
        case CO_PER_VERTEX:
            color << " a['colors'] = [ ";
            for (int index = 0; index < rgba->getNumPoints(); index++)
            {
                float r, g, b, a;
                rgba->getFloatRGBA(index, &r, &g, &b, &a);
                color << r << ", " << g << ", " << b << ", " << a << ", ";
            }

            color << "]; ";

            const char *cmap = rgba->getAttribute("COLORMAP");
            if (cmap)
            {
                istringstream str(cmap);
                string cname, ctype;
                float min, max;
                int num, dummy;
                str >> cname >> ctype >> min >> max >> num >> dummy;

                unsigned char *data = new unsigned char[num * 4];
                float val;
                for (int index = 0; index < num * 4; index++)
                {
                    str >> val;
                    data[index] = (unsigned char)(255.0 * val);
                }

                unsigned int size;
                unsigned char *png = createPNG1D(data, &size);
                struct file f = { png, "image/png", size };
                char *n = new char[strlen(name) + 6];
                sprintf(n, "/%s.png", name);
                registerBuffer(f, "image/png", n, 0);

                addFileBufferToObject(name, png);

                delete data;
                color << "addColorMap('" << name << "', '" << n << "', " << min << ", " << max << "); ";
            }

            pmesg(4, "WebGLRenderer::addGeometry colors [%s]\n", name);
            break;
        }
    }

    if (texture)
    {
        int numCoords = texture->getNumCoordinates();
        int width = texture->getBuffer()->getWidth();

        float **coords = texture->getCoordinates();

        tex << " gl.enable(gl.TEXTURE_2D); a['texture'] = gl.createTexture();"
               " gl.bindTexture(gl.TEXTURE_2D, a['texture']);"
               " gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);"
               " gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);"
               " gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP);"
               " gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP);"
               " a['texcoords'] = [ ";
        for (int index = 0; index < numCoords; index++)
            tex << coords[0][index] << ", 0.0, "; // 2D texture coordinates
        tex << "]; a['texdata'] = [ ";

        unsigned char *data = (unsigned char *)texture->getBuffer()->getPixels();
        for (int index = 0; index < width * 4; index++)
            tex << (int)data[index] << ", ";

        tex << "]; gl.bindTexture(gl.TEXTURE_2D, a['texture']);"
               " gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new WebGLByteArray(a['texdata'])); ";

        const char *cmap = texture->getAttribute("COLORMAP");

        if (cmap)
        {
            istringstream str(cmap);
            string cname, ctype;
            float min, max;
            str >> cname >> ctype >> min >> max;

            unsigned int size;
            unsigned char *png = createPNG1D(data, &size);
            struct file f = { png, "image/png", size };
            char *n = new char[strlen(name) + 6];
            sprintf(n, "/%s.png", name);
            registerBuffer(f, "image/png", n, 0);

            addFileBufferToObject(name, png);

            tex << "addColorMap('" << name << "', '" << n << "', " << min << ", " << max << "); ";
        }
    }

    if (polygons)
    {
        pmesg(3, "WebGLRenderer::addGeometry polygons [%s] [%p] [%p] [%p] (%d)\n", polygons->getName(), colors, normals, text, timeStep);
        str = new ostringstream();
        ostringstream normal;

        *str << "<obj name=\"" << polygons->getName() << "\" type=\"polygons\">";
        *str << "<![CDATA[var a = new Object(); a['type'] = 'triangles';"
                " a['timestep'] = " << timeStep << "; ";

        float *x, *y, *z;
        int *corners, *polys;

        polygons->getAddresses(&x, &y, &z, &corners, &polys);

        int numFaces = 0;
        for (int p = 0; p < polygons->getNumPolygons(); p++)
        {
            int first = polys[p];
            int last;
            if (p == polygons->getNumPolygons() - 1)
                last = polygons->getNumVertices() - 1;
            else
                last = polys[p + 1] - 1;

            int num = last - first + 1;
            if (num == 3)
                numFaces++;
            else if (num == 4)
                numFaces += 2;
            else if (num > 2)
                numFaces += (num - 2);
        }

        if (const coDoVec3 *n = dynamic_cast<const coDoVec3 *>(normals))
        {
            float *x, *y, *z;
            n->getAddresses(&x, &y, &z);
            normal << " a['normals'] = [ ";
            for (int index = 0; index < n->getNumPoints(); index++)
                normal << x[index] << ", " << y[index] << ", " << z[index] << ", ";
            normal << "]; ";
            pmesg(4, "WebGLRenderer::addGeometry normals [%s]\n", name);
        }

        if (numPoints > 0)
            *str << " a['vertices'] = [ ";

        for (int index = 0; index < numPoints; index++)
            *str << x[index] << ", " << y[index] << ", " << z[index] << ", ";

        if (numPoints > 0)
            *str << "]; ";

        int face = 0;

        if (polygons->getNumPolygons())
            *str << " a['indices'] = [ ";

        for (int p = 0; p < polygons->getNumPolygons(); p++)
        {
            int first = polys[p];
            int last;
            if (p == polygons->getNumPolygons() - 1)
                last = polygons->getNumVertices() - 1;
            else
                last = polys[p + 1] - 1;

            if (last - first == 2)
            {
                // triangle
                pmesg(6, "t");
                for (int index = 0; index < 3; index++)
                    *str << corners[first + index] << ", ";

                face++;
            }
            else if (last - first == 3)
            {
                // quad, triangulate it
                pmesg(6, "q");
                float x31 = x[corners[first + 3]] - x[corners[first + 1]];
                float y31 = y[corners[first + 3]] - y[corners[first + 1]];
                float z31 = z[corners[first + 3]] - z[corners[first + 1]];
                float x20 = x[corners[first + 2]] - x[corners[first]];
                float y20 = y[corners[first + 2]] - y[corners[first]];
                float z20 = z[corners[first + 2]] - z[corners[first]];

                if (sqrt(x31 * x31 + y31 * y31 + z31 * z31) < sqrt(x20 * x20 + y20 * y20 + z20 * z20))
                {
                    int tri[2][3] = { { first, first + 1, first + 3 },
                                      { first + 1, first + 2, first + 3 } };

                    for (int t = 0; t < 2; t++)
                        for (int index = 0; index < 3; index++)
                            *str << corners[tri[t][index]] << ", ";
                }
                else
                {
                    int tri[2][3] = { { first, first + 1, first + 2 },
                                      { first, first + 2, first + 3 } };
                    for (int t = 0; t < 2; t++)
                        for (int index = 0; index < 3; index++)
                            *str << corners[tri[t][index]] << ", ";
                }
                face++;
            }
            else
            {
                int start = 1;

                int faces = last - first - 1;
                pmesg(6, "p%d", last - first + 1);
                for (int index = 0; index < faces; index++)
                {
                    *str << corners[first] << ", ";
                    *str << corners[first + start] << ", ";
                    *str << corners[first + start + 1] << ", ";
                    start++;
                }
                face++;
            }
        }
        if (polygons->getNumPolygons())
        {
            *str << "]; " << color.str() << normal.str() << tex.str();
            *str << "a[\"name\"] = \"" << name << "\";"
                                                  " objects['" << name << "'] = a; " << interactor.str();
            *str << "]]></obj>";
            pmesg(6, "\n");
        }
    }
    else if (triangles)
    {
        pmesg(3, "WebGLRenderer::addGeometry trianglestrips [%s] [%p] [%p] [%p] (%d)\n", triangles->getName(), colors, normals, text, timeStep);
        str = new ostringstream();
        *str << "<obj name=\"" << triangles->getName() << "\" type=\"triangles\">";
        *str << "<![CDATA[var a = new Object(); a['type'] = 'triangles';"
                " a['timestep'] = " << timeStep << "; ";

        float *x, *y, *z;
        int *corners, *strips;

        triangles->getAddresses(&x, &y, &z, &corners, &strips);

        // vertices
        if (numPoints > 0)
            *str << " a['vertices'] = [ ";
        for (int index = 0; index < numPoints; index++)
            *str << x[index] << ", " << y[index] << ", " << z[index] << ", ";
        if (numPoints > 0)
            *str << "]; ";

        // indices
        if (triangles->getNumStrips())
            *str << " a['indices'] = [ ";
        for (int s = 0; s < triangles->getNumStrips(); s++)
        {
            int first = strips[s];
            int last;
            if (s == triangles->getNumStrips() - 1)
                last = triangles->getNumVertices() - 1;
            else
                last = strips[s + 1] - 1;

            while (first <= last - 2)
            {
                *str << corners[first] << ", " << corners[first + 1] << ", "
                     << corners[first + 2] << ", ";
                first++;
            }
        }

        if (triangles->getNumStrips())
        {
            *str << "]; " << color.str() << tex.str();
            *str << " a[\"name\"] = \"" << name << "\";"
                                                   " objects['" << name << "'] = a; " << interactor.str();
            *str << "]]></obj>";
        }
    }
    else if (lines)
    {
        pmesg(3, "WebGLRenderer::addGeometry lines [%s] [%p] [%p] [%p]\n", lines->getName(), colors, normals, text);
        str = new ostringstream();
        float *x, *y, *z;
        int *corners, *line;

        *str << "<obj name=\"" << lines->getName() << "\" type=\"lines\">";
        *str << "<![CDATA[var a = new Object(); a['type'] = 'lines';"
                " a['timestep'] = " << timeStep << "; ";

        lines->getAddresses(&x, &y, &z, &corners, &line);

        // vertices
        if (numPoints > 0)
            *str << " a['vertices'] = [ ";
        for (int index = 0; index < numPoints; index++)
            *str << x[index] << ", " << y[index] << ", " << z[index] << ", ";
        if (numPoints > 0)
            *str << "]; ";

        // indices
        if (lines->getNumLines())
            *str << " a['indices'] = [ ";
        for (int l = 0; l < lines->getNumLines(); l++)
        {
            int first = line[l];
            int last;
            if (l == lines->getNumLines() - 1)
                last = lines->getNumVertices() - 1;
            else
                last = line[l + 1] - 1;

            while (first < last)
            {
                for (int index = 0; index < 2; index++)
                    *str << corners[first + index] << ", ";
                first++;
            }
        }
        if (lines->getNumLines())
        {
            *str << "]; " << color.str() << tex.str();
            *str << " a[\"name\"] = \"" << name << "\";"
                                                   " objects['" << name << "'] = a; " << interactor.str();
            *str << "]]></obj>";
        }
    }
    else
    {
        pmesg(1, "WebGLRenderer::addGeometry geometry type [%s] not supported\n",
              geometry->getType());
    }

    return str;
}

void WebGLRenderer::createImage(const char *name, const char *text)
{
    FT ft;
    ft_pixmap *pix = ft.createPixmap();
    ft_string *str = ft.createString(pix, text);
    ft.drawString(pix, str);
    delete str;
    unsigned int size;
    unsigned char *png = createPNG2D(pix->buffer, pix->width, pix->height,
                                     pix->rowstride, &size);
    struct file f = { png, "image/png", size };
    registerBuffer(f, "image/png", name, 0);
    delete pix;
}

struct png_mem
{
    unsigned char *buf;
    size_t size;
};

void png_data_fn(png_struct *png, png_byte *data, size_t num_bytes)
{
    struct png_mem *p = (struct png_mem *)png->io_ptr;

    if (p->buf)
        p->buf = (unsigned char *)realloc(p->buf, p->size + num_bytes);
    else
        p->buf = (unsigned char *)malloc(num_bytes);

    memcpy(p->buf + p->size, data, num_bytes);
    p->size += num_bytes;
}

unsigned char *createPNG1D(unsigned char *data, unsigned int *size)
{
    const int width = 1, height = 256, bit_depth = 8;

    png_structp png_ptr;
    png_infop info_ptr;

    struct png_mem mem = { NULL, 0 };

    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING,
                                      (void *)NULL, NULL, NULL);

    if (png_ptr == NULL)
        return NULL;

    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL)
    {
        png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
        return NULL;
    }

    if (setjmp(png_ptr->jmpbuf))
    {
        png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
        return NULL;
    }

    png_set_write_fn(png_ptr, &mem, &png_data_fn, NULL);

    png_set_IHDR(png_ptr, info_ptr, width, height, bit_depth, PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    png_byte **row_pointers = new png_byte *[height];
    for (int index = 0; index < height; index++)
        row_pointers[index] = data + 4 * (height - 1 - index);

    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, info_ptr);

    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);

    delete[] row_pointers;

    *size = mem.size;
    unsigned char *buf = new unsigned char[*size];
    memcpy(buf, mem.buf, *size);
    free(mem.buf);

    return buf;
}

unsigned char *createPNG2D(unsigned char *data, unsigned int width,
                           unsigned int height, unsigned int stride,
                           unsigned int *size)
{
    const int bit_depth = 8;

    png_structp png_ptr;
    png_infop info_ptr;

    struct png_mem mem = { NULL, 0 };

    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING,
                                      (void *)NULL, NULL, NULL);

    if (png_ptr == NULL)
        return NULL;

    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL)
    {
        png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
        return NULL;
    }

    if (setjmp(png_ptr->jmpbuf))
    {
        png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
        return NULL;
    }

    png_set_write_fn(png_ptr, &mem, &png_data_fn, NULL);

    png_set_IHDR(png_ptr, info_ptr, width, height, bit_depth, PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    png_byte **row_pointers = new png_byte *[height];
    for (int index = 0; index < height; index++)
        row_pointers[index] = data + stride * index;

    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, info_ptr);

    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);

    delete[] row_pointers;

    *size = mem.size;
    unsigned char *buf = new unsigned char[*size];
    memcpy(buf, mem.buf, *size);
    free(mem.buf);

    return buf;
}
