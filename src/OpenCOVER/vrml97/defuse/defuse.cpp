/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

class texture
{
public:
    texture(const char *name, bool environment);
    char *name;
    char *defName;
    bool env;
};

texture::texture(const char *newname, bool environment)
{
    name = new char[strlen(newname) + 1];
    defName = new char[strlen(newname) + 2];
    strcpy(name, newname);
    if (newname[0] >= '0' && newname[0] <= '9')
        sprintf(defName, "x%s", newname);
    else
        strcpy(defName, newname);
    unsigned int i;
    for (i = 0; i < strlen(defName); i++)
    {
        if (defName[i] == '.')
            defName[i] = '_';
        if (defName[i] == '/')
            defName[i] = '_';
        if (defName[i] == ' ')
            defName[i] = '_';
        if (defName[i] == '\\')
            defName[i] = '_';
    }
    env = environment;
}

texture *textures[10000];
int numtextures;
char *buf, *buf2;
int size;
char *readpos;
char *writepos;
char *currentTexturename;
bool currentEnvironment = false;
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

texture *getTexture(const char *name)
{
    int i;
    for (i = 0; i < numtextures; i++)
    {
        if (strcmp(name, textures[i]->name) == 0)
            return textures[i];
    }
    return NULL;
}

void addTexture(texture *tex)
{
    textures[numtextures] = tex;
    numtextures++;
}

int findTexture()
{
    int i;
    while (readpos - buf < size)
    {
        if (strncmp(readpos, "texture", 7) == 0)
        { // this is probably a Image Texture
            if (strncmp(readpos, "texture2", 8) == 0)
            {
                for (i = 0; i < 8; i++)
                    copy();
            }
            else
            {
                for (i = 0; i < 7; i++)
                    copy();
            }
            char *texture;
            texture = readpos;
            while (texture - buf < size)
            {
                if (strncmp(texture, "ImageTexture", 12) == 0)
                {
                    currentEnvironment = false;
                    while ((*texture) && (*texture != '"'))
                    {
                        texture++;
                    }
                    texture++;
                    currentTexturename = texture;
                    while ((*texture) && (*texture != '"'))
                    {
                        texture++;
                    }
                    *texture = '\0';
                    texture++;
                    while ((*texture) && (*texture != '}'))
                    {
                        if (strncmp(texture, "environment", 11) == 0)
                        {
                            while ((*texture) && (*texture != '}') && (*texture != '\n'))
                            {
                                if (strncmp(texture, "TRUE", 4) == 0)
                                {

                                    currentEnvironment = true;
                                }
                                texture++;
                            }
                        }
                        if (*texture != '}')
                            texture++;
                    }
                    texture++;
                    readpos = texture;
                    return 1;
                }
                if (strncmp(texture, "DEF", 3) == 0)
                {
                    //fprintf(stderr,"already has DEF\n");
                    break;
                }
                if (strncmp(texture, "USE", 3) == 0)
                {
                    //fprintf(stderr,"already has USE\n");
                    break;
                }
                if (*texture == '{')
                {
                    //fprintf(stderr,"oops\n");
                    break;
                }
                texture++;
            }
        }

        copy();
    }
    return 0;
}

int main(int argc, char **argv)
{
    numtextures = 0;
    if (argc < 2)
    {
        fprintf(stderr, "Usage: defuse file.wrl \n");
        return -1;
    }
    int filenum;
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
            return -1;
        }

        delete[] command;
        if (fd <= 0)
        {
            fprintf(stderr, "could not open %s for reading\n", argv[filenum]);
            continue;
        }
        fprintf(stderr, "converting %s\n", argv[filenum]);
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
        retval = read(fd, buf, size);
        if (retval == -1)
        {
            std::cerr << "main: read failed" << std::endl;
            return -1;
        }
        int numref = 0;
        numtextures = 0;
        while (findTexture())
        {
            texture *currentTex = getTexture(currentTexturename);
            if (currentTex)
            {
                // already defined texture
                sprintf(writepos, " USE %s \n", currentTex->defName);
                writepos += strlen(writepos);
                numref++;
            }
            else
            {
                // new texture
                currentTex = new texture(currentTexturename, currentEnvironment);
                //fprintf(stderr,"currentEnvironment %d\n" , currentEnvironment);
                addTexture(currentTex);
                if (currentTex->env)
                    sprintf(writepos, " DEF %s ImageTexture{ url \"%s\" environment TRUE}\n", currentTex->defName, currentTex->name);
                else
                    sprintf(writepos, " DEF %s ImageTexture{ url \"%s\"}\n", currentTex->defName, currentTex->name);
                writepos += strlen(writepos);
            }
        }

        close(fd);
        fprintf(stderr, "found %d textures, %d references\n", numtextures, numref);
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
            retval = write(fd, buf2, writepos - buf2);
            if (retval == -1)
            {
                std::cerr << "main: fread failed" << std::endl;
                return -1;
            }
            close(fd);
        }
        delete[] buf;
        delete[] buf2;
    }
    return 0;
}
