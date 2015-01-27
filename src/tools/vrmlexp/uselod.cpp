/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#ifdef WIN32
#include <io.h>
#elif WIN64
#include <io.h>
#else
#include <unistd.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>

#include "defuse.h"
class lodGroup
{
public:
    lodGroup(const char *name, int numLevels);
    char *name;
    char *defName[100];
    int numLevels;
};

lodGroup *lodGroups[10000];
int numlodGroups;
extern char *buf, *buf2;
extern int size;
extern char *readpos;
extern char *writepos;
char *currentLODname;
int currentNumLevels;
#define add(x)         \
    {                  \
        *writepos = x; \
        writepos++;    \
    }
#define copy()                \
    {                         \
        *writepos = *readpos; \
        writepos++;           \
        readpos++;            \
    }

int copySpaces()
{
    while (readpos - buf < size)
    {
        if ((*readpos != '\n') && (*readpos != '\r') && (*readpos != ',') && (*readpos != '\t') && (*readpos != ' '))
        {
            return 1;
        }
        copy();
    }
    return 0;
}

int copyTo(char c)
{
    while (readpos - buf < size)
    {
        if (*readpos == c)
        {
            copy();
            return 1;
        }
        copy();
    }
    return 0;
}

lodGroup::lodGroup(const char *myName, int numL)
{
    name = new char[strlen(myName) + 1];
    strcpy(name, myName);
    numLevels = numL;
    int i;
    for (i = 0; i < numLevels; i++)
    {
        defName[i] = readpos + 4;
        if (strncmp(readpos, "DEF", 3) != 0)
        {
            fprintf(stderr, "wrong LOD\n");
            defName[i] = "noname";
        }
        else
        {
            copyTo(' ');
            copyTo(' ');
            *(readpos - 1) = '\0';
            copyTo('{');

            // skip input to matching ]
            int numOpen = 0;
            while ((numOpen > 0) || (*readpos != '}'))
            {
                if (*readpos == '{')
                    numOpen++;
                if (*readpos == '}')
                    numOpen--;
                copy();
            }
            copy();
            copySpaces();
        }
    }
    fprintf(stderr, "new LOD: %s\n", myName);
}

lodGroup *getLOD(const char *name)
{
    int i;
    for (i = 0; i < numlodGroups; i++)
    {
        if (strcmp(name, lodGroups[i]->name) == 0)
            return lodGroups[i];
    }
    return NULL;
}

void addLOD(lodGroup *lod)
{
    lodGroups[numlodGroups] = lod;
    numlodGroups++;
}

int findLOD()
{
    char *tmpPos;
    char *stringEnd;
    currentNumLevels = 1;
    while (readpos - buf < size)
    {
        if (strncmp(readpos, "LOD", 3) == 0)
        { // this is probably an LOD

            tmpPos = readpos;
            // get LODName
            while (tmpPos > buf)
            {
                if (strncmp(tmpPos, "DEF", 3) == 0)
                { // we found the corresponding DEF

                    currentLODname = tmpPos + 4;
                    tmpPos += 4;
                    break;
                }
                if (*tmpPos == '}')
                {
                    return 0; // oops, this was not a DEF xxx LOD
                }
                tmpPos--;
            }
            if (tmpPos == buf)
            {
                return 0; // oops, this was not a DEF xxx Transform
            }
            while (tmpPos - buf < size)
            {
                if (*tmpPos == ' ')
                { // we found the end of name
                    // now get rid of _%d if there is one
                    if (strncmp(tmpPos - 5, "_LoD_", 5) == 0)
                    {
                        tmpPos -= 5;
                    }
                    if (strncmp(tmpPos - 4, "_LOD", 4) == 0)
                    {
                        tmpPos -= 4;
                    }
                    stringEnd = tmpPos;
                    while ((*(tmpPos - 1) >= '0') && (*(tmpPos - 1) <= '9'))
                        tmpPos--;
                    stringEnd = tmpPos;
                    break;
                }
                tmpPos++;
            }
            if (tmpPos - buf == size)
            {
                return 0; // oops, this was not a DEF xxx Transform
            }

            tmpPos = readpos;
            int num = 0;
            // get LODName
            while (tmpPos - buf < size)
            {
                if (strncmp(tmpPos, "level", 5) == 0)
                { // the level field
                    break;
                }
                if (*tmpPos == '{')
                    num++;
                if (num > 1)
                {
                    return 0; // oops, this was not a DEF xxx LOD
                }
                tmpPos++;
            }
            // now, we are really sure
            //go to beginning of this transform Node
            // while(strncmp(readpos,"DEF",3)!=0)
            // {
            //    readpos--;
            //    writepos--;
            // }
            /*//remove all previous nodes in this group
             int numOpen = 0;
             while((numOpen > 0) || (*writepos!='['))
             {
                 if(*writepos == '[')
                     numOpen--;
                 if(*writepos == ']')
                     numOpen++;
                 writepos--;
             }
             writepos++;*/
            // sprintf(writepos," \n ");
            // writepos += strlen(writepos);

            copyTo('{');

            *stringEnd = '\0';

            copyTo('['); // we are here: range [ 15, 30 ]

            if (copySpaces() == 0)
                return 0; //                     ^
            while (*readpos != ']')
            {
                if (*readpos == ' ')
                    currentNumLevels++;
                copy();
            }

            copyTo('['); // we are here: level [
            if (copySpaces() == 0)
                return 0;

            return 1;
        }
        copy();
    }
    return 0;
}

int uselod(const char *fileName)
{
    numlodGroups = 0;

    int fd, i;
    fd = _open(fileName, O_RDONLY | O_BINARY);

    if (fd <= 0)
    {
        fprintf(stderr, "could not open %s for reading\n", fileName);
        return 0;
    }

    fprintf(stderr, "converting %s\n", fileName);
    struct stat statbuf;
    fstat(fd, &statbuf);
    size = statbuf.st_size;

    /*	  
	 {
		RECT rcClient;  // client area of parent window 
		int cyVScroll;  // height of a scroll bar arrow 
		hWndPDlg = CreateDialog(hInstance, MAKEINTRESOURCE(IDD_PROGRESSDLG),
			GetActiveWindow(), ProgressDlgProc);
		GetClientRect(hWndPDlg, &rcClient); 
		cyVScroll = GetSystemMetrics(SM_CYVSCROLL); 
		ShowWindow(hWndPDlg, SW_SHOW);
		hWndPB = CreateWindow(PROGRESS_CLASS, (LPSTR) NULL, 
			WS_CHILD | WS_VISIBLE, rcClient.left, 
			rcClient.bottom - cyVScroll, 
			rcClient.right, cyVScroll, 
			hWndPDlg, (HMENU) 0, hInstance, NULL); 
		// Set the range and increment of the progress bar. 
		SendMessage(hWndPB, PBM_SETRANGE, 0, MAKELPARAM(0,
			size/1000 + 1));
		SendMessage(hWndPB, PBM_SETSTEP, (WPARAM) 1, 0); 
	}*/

    buf = new char[size + 1000];
    buf2 = new char[size + 100000];
    readpos = buf;
    writepos = buf2;
    memset(buf, 0, size + 1000);
    if ((buf2 == NULL) || (buf == NULL))
    {
        fprintf(stderr, "out of memory\n");
        return -1;
    }
    _read(fd, buf, size);
    int numref = 0;
    numlodGroups = 0;
    while (findLOD())
    {
        lodGroup *currentLOD = getLOD(currentLODname);
        if (currentLOD)
        {
            // already defined lodGroup

            // skip input to matching ]
            int numOpen = 0;
            while ((numOpen > 0) || (*readpos != ']'))
            {
                if (*readpos == '[')
                    numOpen++;
                if (*readpos == ']')
                    numOpen--;
                readpos++;
            }
            for (i = 0; i < currentLOD->numLevels; i++)
            {
                sprintf(writepos, " USE %s \n", currentLOD->defName[i]);
                writepos += strlen(writepos);
            }
            numref++;
        }
        else
        {
            addLOD(new lodGroup(currentLODname, currentNumLevels));
        }
        // go to end of LOD
        int numOpen = 0;
        while ((numOpen > 0) || (*readpos != ']'))
        {
            if (*readpos == '[')
                numOpen++;
            if (*readpos == ']')
                numOpen--;
            copy();
        }
        copyTo('}');
        copy();
        copyTo('}');
        // skip all nodes after this one
        /*  numOpen = 0;
         while((numOpen > 0) || (*readpos!=']'))
         {
            if(*readpos == '[')
               numOpen++;
            if(*readpos == ']')
               numOpen--;
            readpos++;
         }*/
    }

    _close(fd);
    fprintf(stderr, "found %d lods, %d references\n", numlodGroups, numref);
    if (numref == 0)
    {
        fprintf(stderr, "%s unchanged\n", fileName);
    }
    else
    {

        fd = _open(fileName, O_WRONLY | O_CREAT | O_BINARY | O_TRUNC, 0777);
        if (fd <= 0)
        {
            fprintf(stderr, "could not open %s for writing\n", fileName);
            return 0;
        }
        _write(fd, buf2, (unsigned int)(writepos - buf2));
        _close(fd);
    }
    delete[] buf;
    delete[] buf2;
    return 0;
}
