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

class texture
{
public:
    texture(const char *name, bool environment, int bm);
    char *name;
    char *defName;
    bool env;
    int blendMode;
};

texture::texture(const char *newname, bool environment, int bm)
{
    blendMode = bm;
    env = environment;
    name = new char[strlen(newname) + 1];
    defName = new char[strlen(newname) + 100];
    strcpy(name, newname);
    if (newname[0] >= '0' && newname[0] <= '9')
        sprintf(defName, "x%s", newname);
    else
        strcpy(defName, newname);
    int i;
    for (i = 0; i < (int)strlen(defName); i++)
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
    if (env)
        strcat(defName, "_environment");
    if (blendMode)
    {
        char num[100];
        sprintf(num, "_blendMode%d", blendMode);
        strcat(defName, num);
    }
}

texture *textures[10000];
int numtextures;
char *buf, *buf2;
int size;
char *readpos;
char *writepos;
char *currentTexturename;
bool currentEnvironment = false;
int currentBlendMode = 0;
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

texture *getTexture(const char *name, bool environment, int bm)
{
    int i;
    for (i = 0; i < numtextures; i++)
    {
        if ((strcmp(name, textures[i]->name) == 0) && (textures[i]->env == environment) && (textures[i]->blendMode == bm))
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
            for (i = 0; i < 7; i++)
                copy();
            while (*readpos >= '0' && *readpos <= '9')
                copy();
            char *texture;
            texture = readpos;
            while (texture - buf < size)
            {
                //if( strncmp(texture, "ImageTexture",12)==0 || strncmp(texture, "CubeTexture",11)==0)
                // TODO CubeTexture save all urls
                if (strncmp(texture, "ImageTexture", 12) == 0)
                {
                    currentEnvironment = false;
                    currentBlendMode = 0;
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
                        if (strncmp(texture, "blendMode", 9) == 0)
                        {
                            texture += 9;
                            while ((*texture) && (*texture != '}') && (*texture != '\n') && ((*texture == ' ') || (*texture == '\t')))
                            {
                                texture++;
                            }
                            if (*texture >= '0' && *texture <= '9')
                            {
                                sscanf(texture, "%d", &currentBlendMode);
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

int defuse(const char *fileName)
{
    numtextures = 0;

    int fd;
    fd = _open(fileName, O_RDONLY | O_BINARY);
    if (fd <= 0)
    {
        fprintf(stderr, "could not open %s for reading\n", fileName);
        return -1;
    }
    //fprintf(stderr,"converting %s\n",argv[filenum]);
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
    _read(fd, buf, size);
    int numref = 0;
    numtextures = 0;
    while (findTexture())
    {
        texture *currentTex = getTexture(currentTexturename, currentEnvironment, currentBlendMode);
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
            currentTex = new texture(currentTexturename, currentEnvironment, currentBlendMode);
            //fprintf(stderr,"currentEnvironment %d\n" , currentEnvironment);
            addTexture(currentTex);
            sprintf(writepos, " DEF %s ImageTexture{ url \"%s\"\n", currentTex->defName, currentTex->name);
            writepos += strlen(writepos);
            if (currentTex->env)
            {
                sprintf(writepos, " environment TRUE\n");
                writepos += strlen(writepos);
            }
            if (currentTex->blendMode > 0)
            {
                sprintf(writepos, " blendMode %d\n", currentTex->blendMode);
                writepos += strlen(writepos);
            }
            sprintf(writepos, "}\n");
            writepos += strlen(writepos);
        }
    }

    _close(fd);
    //fprintf(stderr,"found %d textures, %d references\n",numtextures, numref);
    if (numref == 0)
    {
        //fprintf(stderr,"%s unchanged\n",argv[filenum]);
    }
    else
    {

        fd = _open(fileName, O_WRONLY | O_CREAT | O_BINARY | O_TRUNC, 0777);
        if (fd <= 0)
        {
            fprintf(stderr, "could not open %s for writing\n", fileName);
            return -1;
        }
        _write(fd, buf2, (int)(writepos - buf2));
        _close(fd);
    }
    delete[] buf;
    delete[] buf2;
    return 0;
}
