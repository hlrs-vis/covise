/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#ifdef WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

class lodGroup
{
public:
    lodGroup(const char *name, int numLevels);
    char *name;
    const char *defName[100];
    int numLevels;
};

lodGroup *lodGroups[10000];
int numlodGroups;
char *buf, *buf2;
int size;
char *readpos;
char *writepos;
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
    for (int i = 0; i < numLevels; i++)
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
    char *stringEnd = NULL;
    currentNumLevels = 1;
    while (readpos - buf < size)
    {
        if (strncmp(readpos, "_LoD_", 5) == 0)
        { // this is probably a LOD Group

            while (readpos - buf < size)
            {
                copy();
                if (*readpos == '[') // we are here: children [
                {
                    copy();
                    break;
                }
            }
            tmpPos = readpos;
            // get GroupName
            while (tmpPos > buf)
            {
                if (strncmp(tmpPos, "DEF", 3) == 0)
                { // we found the corresponding DEF

                    currentLODname = tmpPos + 4;
                    tmpPos += 4;
                    break;
                }
                tmpPos--;
            }
            if (tmpPos == buf)
            {
                return 2; // oops, this was not a DEF xxx Transform
            }
            while (tmpPos - buf < size)
            {
                if (*tmpPos == ' ')
                { // we found the end of name
                    // now get rid of %d_LoD_
                    stringEnd = tmpPos;
                    while (*(tmpPos - 1) != '_')
                        tmpPos--;
                    tmpPos--;
                    while (*(tmpPos - 1) != '_')
                        tmpPos--;
                    tmpPos--;
                    while ((*(tmpPos - 1) >= '0') && (*(tmpPos - 1) <= '9'))
                        tmpPos--;
                    /* war mal das:  if(*(tmpPos-1) == '_')
                   stringEnd = tmpPos-1;*/
                    if (strncmp(tmpPos - 1, "_0_", 3) == 0)
                    {
                        tmpPos--;
                        while ((*(tmpPos - 1) >= '0') && (*(tmpPos - 1) <= '9'))
                            tmpPos--;
                    }
                    if (*(tmpPos - 1) != ' ')
                        stringEnd = tmpPos;
                    break;
                }
                tmpPos++;
            }
            if (tmpPos - buf == size)
            {
                return 0; // oops, this was not a DEF xxx Transform
            }

            if (copySpaces() == 0)
                return 2;
            while (readpos - buf < size)
            {
                copy();
                if (*readpos == ' ') // we are here: children [
                {
                    copy();
                    break;
                }
            }
            if (copySpaces() == 0)
                return 0;
            while (readpos - buf < size)
            {
                copy();
                if (*readpos == ' ') // we are here: children [
                {
                    copy();
                    break;
                }
            }
            if (copySpaces() == 0)
                return 0;
            if (strncmp(readpos, "LOD {", 5) != 0)
            {
                return 2; // this was not a LOD
            }
            // now, we are really sure
            //go to beginning of this transform Node
            while (strncmp(readpos, "DEF", 3) != 0)
            {
                readpos--;
                writepos--;
            }
            readpos--;
            writepos--;
            while (strncmp(readpos, "DEF", 3) != 0)
            {
                readpos--;
                writepos--;
            }
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
            sprintf(writepos, " \n ");
            writepos += strlen(writepos);

            copyTo('{');
            copyTo('{');

            *stringEnd = '\0';

            sprintf(writepos, "\n      center 0 0 0\n");
            writepos += strlen(writepos);
            while ((*readpos != '\r') && (*readpos != '\n'))
                readpos++;
            while ((*readpos == '\r') || (*readpos == '\n'))
                readpos++;
            while ((*readpos != '\r') && (*readpos != '\n'))
                readpos++;
            while ((*readpos == '\r') || (*readpos == '\n'))
                readpos++;

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

int main(int argc, char **argv)
{
    numlodGroups = 0;
    if (argc < 2)
    {
        fprintf(stderr, "Usage: defuse file.wrl \n");
        return -1;
    }
    int filenum;
    int i;
    for (filenum = 1; filenum < argc; filenum++)
    {

        int fd;
        fd = open(argv[filenum], O_RDONLY);
        char *command = new char[2 * (strlen(argv[filenum])) + 500];
        sprintf(command, "cp -f %s %s.bak", argv[filenum], argv[filenum]);

        int retval;
        retval = system(command);
        if (retval == -1)
        {
            std::cerr << "main: system failed" << std::endl;
            delete[] command;
            return -1;
        }

        delete[] command;
        if (fd <= 0)
        {
            fprintf(stderr, "could not open %s for reading\n", argv[filenum]);
            continue;
        }
        fprintf(stderr, "converting %s\n", argv[1]);
        struct stat statbuf;
        fstat(fd, &statbuf);
        size = statbuf.st_size;
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
        int numread;
        int todo = size;
        while (todo > 0)
        {
            numread = read(fd, buf, todo);
            if (numread < 0)
            {
                fprintf(stderr, "read error %s\n", argv[filenum]);
            }
            todo -= numread;
            if (todo > 0)
            {
                fprintf(stderr, "short read %d\n", numread);
            }
        }
        int numref = 0;
        numlodGroups = 0;
        int found = 0;
        while ((found = findLOD()))
        {
            if (found == 1)
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
                numOpen = 0;
                while ((numOpen > 0) || (*readpos != ']'))
                {
                    if (*readpos == '[')
                        numOpen++;
                    if (*readpos == ']')
                        numOpen--;
                    readpos++;
                }
            }
        }

        close(fd);
        fprintf(stderr, "found %d lodGroups, %d references\n", numlodGroups, numref);
        if (numref == 0)
        {
            fprintf(stderr, "%s unchanged\n", argv[filenum]);
        }
        else
        {

            char *filename;
            filename = argv[filenum];
            fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0777);
            if (fd <= 0)
            {
                fprintf(stderr, "could not open %s for writing\n", filename);
                continue;
            }
            size = writepos - buf2;
            int numread;
            int todo = size;
            while (todo > 0)
            {
                numread = write(fd, buf2, todo);
                if (numread < 0)
                {
                    fprintf(stderr, "write error %s\n", argv[filenum]);
                }
                todo -= numread;
                if (todo > 0)
                {
                    fprintf(stderr, "short write %d\n", numread);
                }
            }
            close(fd);
        }
        delete[] buf;
        delete[] buf2;
    }
    return 0;
}
