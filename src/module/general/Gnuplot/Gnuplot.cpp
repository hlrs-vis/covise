/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                    (C) 2010 HLRS       **
 **                                                                        **
 ** Description: 2D Gnuplot Modul                                          **
 **                                                                        **
 ** Name:        Gnuplot                                                   **
 **                                                                        **
 ** v1.0 12/2010                                                           **
 **                                                                        **
\****************************************************************************/

#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#else
#include <io.h>
#endif
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>

#include <stdlib.h>

#include <string>
#include <fstream>
#include <png.h>

#include "Gnuplot.h"

struct mempng
{

    const char *data;
    size_t loc;
};

static void png_read(png_structp png_ptr, png_bytep data, png_size_t length)
{

    struct mempng *mem = (struct mempng *)png_get_io_ptr(png_ptr);
    memcpy(data, mem->data + mem->loc, length);
    mem->loc += length;
}

Gnuplot::Gnuplot(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Gnuplot 2D plot module")
    , gnuplot_running(false)
{
#ifdef WIN32
    g_hChildStd_IN_Rd = NULL;
    g_hChildStd_IN_Wr = NULL;
    g_hChildStd_OUT_Rd = NULL;
    g_hChildStd_OUT_Wr = NULL;
#endif
    p_data = addInputPort("data", "Vec2", "data to plot");

    p_geom = addOutputPort("geo", "Polygons", "geometry for mapping the"
                                              "plot texture");
    p_texture = addOutputPort("tex", "Texture", "the plot texture");

    windowed = addBooleanParam("windowed", "show gnuplot window");
}

#ifdef WIN32

void Gnuplot::CreateChildProcess()
// Create a child process that uses the previously created pipes for STDIN and STDOUT.
{
    SECURITY_ATTRIBUTES saAttr;

    // Set the bInheritHandle flag so pipe handles are inherited.

    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE;
    saAttr.lpSecurityDescriptor = NULL;

    // Create a pipe for the child process's STDOUT.

    if (!CreatePipe(&g_hChildStd_OUT_Rd, &g_hChildStd_OUT_Wr, &saAttr, 0))
        ErrorExit(TEXT("StdoutRd CreatePipe"));

    // Ensure the read handle to the pipe for STDOUT is not inherited.

    if (!SetHandleInformation(g_hChildStd_OUT_Rd, HANDLE_FLAG_INHERIT, 0))
        ErrorExit(TEXT("Stdout SetHandleInformation"));

    // Create a pipe for the child process's STDIN.

    if (!CreatePipe(&g_hChildStd_IN_Rd, &g_hChildStd_IN_Wr, &saAttr, 0))
        ErrorExit(TEXT("Stdin CreatePipe"));

    // Ensure the write handle to the pipe for STDIN is not inherited.

    if (!SetHandleInformation(g_hChildStd_IN_Wr, HANDLE_FLAG_INHERIT, 0))
        ErrorExit(TEXT("Stdin SetHandleInformation"));

    // Create the child process.

    TCHAR szCmdline[] = TEXT("pgnuplot.exe"); // pgnuplot should be in the PATH
    PROCESS_INFORMATION piProcInfo;
    STARTUPINFO siStartInfo;
    BOOL bSuccess = FALSE;

    // Set up members of the PROCESS_INFORMATION structure.

    ZeroMemory(&piProcInfo, sizeof(PROCESS_INFORMATION));

    // Set up members of the STARTUPINFO structure.
    // This structure specifies the STDIN and STDOUT handles for redirection.

    ZeroMemory(&siStartInfo, sizeof(STARTUPINFO));
    siStartInfo.cb = sizeof(STARTUPINFO);
    siStartInfo.hStdError = g_hChildStd_OUT_Wr;
    siStartInfo.hStdOutput = g_hChildStd_OUT_Wr;
    siStartInfo.hStdInput = g_hChildStd_IN_Rd;
    siStartInfo.dwFlags |= STARTF_USESTDHANDLES;

    // Create the child process.

    bSuccess = CreateProcess(NULL,
                             szCmdline, // command line
                             NULL, // process security attributes
                             NULL, // primary thread security attributes
                             TRUE, // handles are inherited
                             0, // creation flags
                             NULL, // use parent's environment
                             NULL, // use parent's current directory
                             &siStartInfo, // STARTUPINFO pointer
                             &piProcInfo); // receives PROCESS_INFORMATION

    // If an error occurs, exit the application.
    if (!bSuccess)
        ErrorExit(TEXT("CreateProcess"));
    else
    {
        // Close handles to the child process and its primary thread.
        // Some applications might keep these handles to monitor the status
        // of the child process, for example.

        CloseHandle(piProcInfo.hProcess);
        CloseHandle(piProcInfo.hThread);
    }
}

/*
void Gnuplot::WriteToPipe(void)

// Read from a file and write its contents to the pipe for the child's STDIN.
// Stop when there is no more data.
{
DWORD dwRead, dwWritten;
CHAR chBuf[BUFSIZE];
BOOL bSuccess = FALSE;

for (;;)
{
bSuccess = ReadFile(g_hInputFile, chBuf, BUFSIZE, &dwRead, NULL);
if ( ! bSuccess || dwRead == 0 ) break;

bSuccess = WriteFile(g_hChildStd_IN_Wr, chBuf, dwRead, &dwWritten, NULL);
if ( ! bSuccess ) break;
}

// Close the pipe handle so the child process stops reading.

if ( ! CloseHandle(g_hChildStd_IN_Wr) )
ErrorExit(TEXT("StdInWr CloseHandle"));
}

void Gnuplot::ReadFromPipe(void)

// Read output from the child process's pipe for STDOUT
// and write to the parent process's pipe for STDOUT.
// Stop when there is no more data.
{
DWORD dwRead, dwWritten;
CHAR chBuf[BUFSIZE];
BOOL bSuccess = FALSE;
HANDLE hParentStdOut = GetStdHandle(STD_OUTPUT_HANDLE);

// Close the write end of the pipe before reading from the
// read end of the pipe, to control child process execution.
// The pipe is assumed to have enough buffer space to hold the
// data the child process has already written to it.

if (!CloseHandle(g_hChildStd_OUT_Wr))
ErrorExit(TEXT("StdOutWr CloseHandle"));

for (;;)
{
bSuccess = ReadFile( g_hChildStd_OUT_Rd, chBuf, BUFSIZE, &dwRead, NULL);
if( ! bSuccess || dwRead == 0 ) break;

bSuccess = WriteFile(hParentStdOut, chBuf,
dwRead, &dwWritten, NULL);
if (! bSuccess ) break;
}
}
*/
void Gnuplot::ErrorExit(const TCHAR *lpszFunction)

// Format a readable error message, display a message box,
// and exit from the application.
{
    LPVOID lpMsgBuf;
    LPVOID lpDisplayBuf;
    DWORD dw = GetLastError();

    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        dw,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR)&lpMsgBuf,
        0, NULL);

    lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT,
                                      (lstrlen((LPCTSTR)lpMsgBuf) + lstrlen((LPCTSTR)lpszFunction) + 40) * sizeof(TCHAR));
    StringCchPrintf((LPTSTR)lpDisplayBuf,
                    LocalSize(lpDisplayBuf) / sizeof(TCHAR),
                    TEXT("%s failed with error %d: %s"),
                    lpszFunction, dw, lpMsgBuf);
    MessageBox(NULL, (LPCTSTR)lpDisplayBuf, TEXT("Error"), MB_OK);

    LocalFree(lpMsgBuf);
    LocalFree(lpDisplayBuf);
    ExitProcess(1);
}
#endif

int Gnuplot::writeStdIn(const void *_Buf, unsigned int _MaxCharCount)
{
#ifdef WIN32

    BOOL bSuccess = FALSE;
    DWORD dwWritten;
    bSuccess = WriteFile(g_hChildStd_IN_Wr, _Buf, _MaxCharCount, &dwWritten, NULL);
    if (!bSuccess)
        return -1;
    return dwWritten;
#else
    return write(fdpc[1], _Buf, _MaxCharCount);
#endif
}

int Gnuplot::readStdOut(void *_Buf, unsigned int _MaxCharCount)
{
#ifdef WIN32
    BOOL bSuccess = FALSE;
    DWORD dwRead;
    bSuccess = ReadFile(g_hChildStd_OUT_Rd, _Buf, _MaxCharCount, &dwRead, NULL);
    if (!bSuccess || dwRead == 0)
        return -1;
    return dwRead;
#else
    return read(fdcp[0], _Buf, _MaxCharCount);
#endif
}

int Gnuplot::compute(const char *)
{
#ifndef WIN32
    pid_t pid;
#endif

    if (!gnuplot_running)
    {
#ifdef WIN32
        CreateChildProcess();
#else
        if (!pipe(fdpc) && !pipe(fdcp))
        {
            pid = fork();

            if (pid == (pid_t)0)
            {
                // child
                close(fdcp[0]);
                close(fdpc[1]);
                dup2(fdcp[1], 1);
                dup2(fdpc[0], 0);
                execlp("gnuplot", "gnuplot", (char *)NULL);
            }
            else
            {
                // parent
                close(fdpc[0]);
                close(fdcp[1]);
            }
        }
#endif
    }

    float *x_data, *y_data;
    int numData = 0;

    const coDoVec2 *data = dynamic_cast<const coDoVec2 *>(p_data->getCurrentObject());
    if (data)
    {
        data->getAddresses(&x_data, &y_data);
        numData = data->getNumPoints();
    }

    const char *command = p_command->getValue();
    const char *blocks = p_blocks->getValue();
    std::vector<int> dataBlockSize;

    if (data)
    {
        if (const char *c = data->getAttribute("GNUPLOT_COMMAND"))
            command = c;
        if (const char *b = data->getAttribute("GNUPLOT_DATABLOCKS"))
            blocks = b;

        if (blocks)
        {
            stringstream ss(blocks);
            int n;
            while (!(ss >> n).fail())
                dataBlockSize.push_back(n);
        }
    }

    if (!data || !command || !blocks)
        return FAIL;

    if (windowed->getValue())
    {
        writeStdIn(command, (unsigned int)strlen(command));
        int block = 0;
        int blockLine = 0;
        int blockLength = dataBlockSize[block];

        char line[40];

        for (int index = 0; index < numData; index++)
        {

            if (blockLine == blockLength - 1)
            {
                block++;
                blockLine = 0;
                blockLength = dataBlockSize[block];
                //printf("new block length %d\n", blockLength);
                snprintf(line, 40, "%f %f\ne\n", x_data[index], y_data[index]);
            }
            else
            {
                blockLine++;
                snprintf(line, 40, "%f %f\n", x_data[index], y_data[index]);
            }
            writeStdIn(line, (int)strlen(line));
        }
        gnuplot_running = true;
    }
    else
    {
#ifdef WIN32
        int fd = _open("schnubb.gp", _O_WRONLY | _O_CREAT);
#else
        int fd = open("schnubb.gp", O_WRONLY | O_CREAT);
#endif
        const char *w = "set terminal png size 1024, 1024; ";
        writeStdIn(w, (unsigned int)strlen(w));
        if (write(fd, w, (unsigned int)strlen(w)) == -1)
        {
            fprintf(stderr, "write error 1: %s\n", strerror(errno));
        }
        int block = 0;
        int blockLine = 0;
        int blockLength = dataBlockSize[block];

        char line[40];

        writeStdIn(command, (unsigned int)strlen(command));
        if (write(fd, command, (unsigned int)strlen(command)) == -1)
        {
            fprintf(stderr, "write error 2: %s\n", strerror(errno));
        }

        for (int index = 0; index < numData; index++)
        {

            if (blockLine == blockLength - 1)
            {
                block++;
                blockLine = 0;
                blockLength = dataBlockSize[block];
                printf("new block length %d\n", blockLength);
                snprintf(line, 40, "%f %f\ne\n", x_data[index], y_data[index]);
            }
            else
            {
                blockLine++;
                snprintf(line, 40, "%f %f\n", x_data[index], y_data[index]);
            }
            writeStdIn(line, (unsigned int)strlen(line));
            if (write(fd, line, (unsigned int)strlen(line)) == -1)
            {
                fprintf(stderr, "write error 3: %s\n", strerror(errno));
            }
        }
        close(fd);
        gnuplot_running = true;

#ifdef WIN32
        char cbuf[1000];
        char *tmpDir = getenv("TEMP");
        char *tempName = new char[strlen(tmpDir) + 1000];
        sprintf(tempName, "%s\\%s", tmpDir, "test.png");
        int slen = (int)strlen(tempName);
        for (int i = 0; i < slen; i++)
        {
            if (tempName[i] == '\\')
                tempName[i] = '/';
        }
        snprintf(cbuf, 1000, "set output \"%s\"; ", tempName);
        writeStdIn(cbuf, (unsigned int)strlen(cbuf));
#endif
        writeStdIn(command, (unsigned int)strlen(command));
        command = "\n";
        writeStdIn(command, (unsigned int)strlen(command));
#ifndef WIN32
        close(fdpc[1]);
#else
        CloseHandle(g_hChildStd_OUT_Rd);
        CloseHandle(g_hChildStd_OUT_Wr);
        CloseHandle(g_hChildStd_IN_Rd);
        CloseHandle(g_hChildStd_IN_Wr);
#endif
        gnuplot_running = false;

		int bufSize = 4096;
		size_t length = 0;
		size_t s;
        char *buf = (char *)malloc(bufSize);
#ifdef WIN32
        FILE *fp;
        int tries = 0;
        do
        {
            fp = fopen(tempName, "rb");
            usleep(100000);
            tries++;

        } while ((fp == NULL) && (tries < 100));
        if (fp == NULL)
            return FAIL;
        while ((s = fread(buf + length, 1, 4096, fp)) > 0)
#else
        while ((s = readStdOut(buf + length, 4096)) > 0)
#endif
        {
            length += s;
            if (bufSize < length + 4096)
            {
                bufSize += 4096;
                buf = (char *)realloc(buf, bufSize);
            }
        }
#ifdef WIN32
        fclose(fp);
        remove(tempName);
        delete[] tempName;
#endif

        if (length)
        {
            png_structp png_ptr = NULL;
            png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
                                             NULL, NULL, NULL);
            png_infop info_ptr = NULL;
            info_ptr = png_create_info_struct(png_ptr);
            struct mempng p = { buf, 0 };

            png_set_read_fn(png_ptr, &p, png_read);
            png_read_info(png_ptr, info_ptr);

            png_uint_32 width = 0;
            png_uint_32 height = 0;
            int bitDepth = 0;
            int colorType = -1;
            png_get_IHDR(png_ptr, info_ptr, &width, &height, &bitDepth,
                         &colorType, NULL, NULL, NULL);

            const png_byte color_type = png_get_color_type(png_ptr, info_ptr);
            if (color_type == PNG_COLOR_TYPE_PALETTE)
                png_set_expand(png_ptr);
            if (color_type == PNG_COLOR_TYPE_GRAY && png_get_bit_depth(png_ptr, info_ptr) < 8)
                png_set_expand(png_ptr);
            if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
                png_set_expand(png_ptr);
            png_read_update_info(png_ptr, info_ptr);

            const char *texName = p_texture->getObjName();

            stringstream pName;
            stringstream tName;
            pName << texName << "_img";
            tName << texName << "_tex";

            coDoPixelImage *img = new coDoPixelImage(pName.str().c_str(),
                                                     width, height, 3, 3);

            char *pixels = img->getPixels();

            png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);

            for (unsigned int y = 0; y < height; y++)
                row_pointers[y] = (png_byte *)malloc(png_get_rowbytes(png_ptr, info_ptr));
            png_read_image(png_ptr, row_pointers);

            for (unsigned int y = 0; y < height; y++)
            {
                png_byte *row = row_pointers[y];
                for (unsigned int x = 0; x < width; x++)
                {
                    png_byte *ptr = &(row[x * 3]);
                    pixels[(x + y * width) * 3 + 0] = ptr[0];
                    pixels[(x + y * width) * 3 + 1] = ptr[1];
                    pixels[(x + y * width) * 3 + 2] = ptr[2];
                }
            }

            for (unsigned int y = 0; y < height; y++)
                free(row_pointers[y]);
            free(row_pointers);

            png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
            free(buf);

            int *txIndex = new int[4];
            txIndex[0] = 0;
            txIndex[1] = 1;
            txIndex[2] = 2;
            txIndex[3] = 3;

            float **txCoord = new float *[2];
            txCoord[0] = new float[4];
            txCoord[1] = new float[4];

            txCoord[0][0] = 0.0;
            txCoord[1][0] = 1.0;
            txCoord[0][1] = 1.0;
            txCoord[1][1] = 1.0;
            txCoord[0][2] = 1.0;
            txCoord[1][2] = 0.0;
            txCoord[0][3] = 0.0;
            txCoord[1][3] = 0.0;

            coDoTexture *tex = new coDoTexture(p_texture->getObjName(), img,
                                               0, 3, 0, 4, txIndex, 4, txCoord);

            tex->addAttribute("MIN_FILTER", "LINEAR_MIPMAP_LINEAR");
            tex->addAttribute("MAG_FILTER", "LINEAR_MIPMAP_LINEAR");

            p_texture->setCurrentObject(tex);

            float xc[] = { -0.5, 0.5, 0.5, -0.5 };
            float yc[] = { -0.5, -0.5, 0.5, 0.5 };
            float zc[] = { 0.0, 0.0, 0.0, 0.0 };

            int cl[] = { 0, 1, 2, 3 };
            int pl[] = { 0 };

            coDoPolygons *poly = new coDoPolygons(p_geom->getObjName(),
                                                  4, xc, yc, zc, 4, cl, 1, pl);
            poly->addAttribute("MENU_TEXTURE", "MyTitle(TODO)");
            p_geom->setCurrentObject(poly);

            delete[] txCoord[0];
            delete[] txCoord[1];
            delete[] txCoord;
            delete[] txIndex;
        }
    }

    return SUCCESS;
}

MODULE_MAIN(Tools, Gnuplot)
