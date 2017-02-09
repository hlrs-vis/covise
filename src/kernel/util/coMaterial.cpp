/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coMaterial.h"

#include "coFileUtil.h"
#include "unixcompat.h"
#include <stdio.h>

#ifdef _WIN32
#include <process.h>
#include <io.h>
#include <direct.h>
#endif

using namespace covise;

coMaterial::coMaterial(const char *n, float *a, float *d, float *s, float *e, float sh, float tr)
{
    name = new char[strlen(n) + 1];
    strcpy(name, n);
    ambientColor[0] = a[0];
    ambientColor[1] = a[1];
    ambientColor[2] = a[2];
    diffuseColor[0] = d[0];
    diffuseColor[1] = d[1];
    diffuseColor[2] = d[2];
    specularColor[0] = s[0];
    specularColor[1] = s[1];
    specularColor[2] = s[2];
    emissiveColor[0] = e[0];
    emissiveColor[1] = e[1];
    emissiveColor[2] = e[2];
    shininess = sh;
    transparency = tr;
}

coMaterial::~coMaterial()
{
    delete[] name;
}

coMaterial::coMaterial(const char *n, const char *filename)
{
    name = new char[strlen(n) + 1];
    strcpy(name, n);
    ambientColor[0] = 1;
    ambientColor[1] = 0;
    ambientColor[2] = 0;
    diffuseColor[0] = 1;
    diffuseColor[1] = 0;
    diffuseColor[2] = 0;
    specularColor[0] = 1;
    specularColor[1] = 0;
    specularColor[2] = 0;
    emissiveColor[0] = 1;
    emissiveColor[1] = 0;
    emissiveColor[2] = 0;
    shininess = 0;
    transparency = 0;
    char buf[300], *b;
    FILE *fp = fopen(filename, "r");
    char *retval_fgets;
    size_t retval_sscanf;
    if (fp)
    {
        while (!feof(fp))
        {
            retval_fgets = fgets(buf, 300, fp);
            if (retval_fgets == NULL)
            {
                // this also happens on EOF
                //std::cerr<<"coMaterial::coMaterial: fgets from " << filename << " failed"<<std::endl;
                break;
            }
            b = buf;
            while ((*b != '\0') && ((*b == ' ') || (*b == '\t')))
                b++;
            if (strncasecmp(b, "ambientColor", 12) == 0)
            {
                retval_sscanf = sscanf(b + 12, "%f %f %f", ambientColor, ambientColor + 1, ambientColor + 2);
                if (retval_sscanf != 3)
                {
                    std::cerr << "coMaterial::coMaterial: sscanf failed" << std::endl;
                    break;
                }
            }
            else if (strncasecmp(b, "diffuseColor", 12) == 0)
            {
                retval_sscanf = sscanf(b + 12, "%f %f %f", diffuseColor, diffuseColor + 1, diffuseColor + 2);
                if (retval_sscanf != 3)
                {
                    std::cerr << "coMaterial::coMaterial: sscanf failed" << std::endl;
                    break;
                }
            }
            else if (strncasecmp(b, "specularColor", 13) == 0)
            {
                retval_sscanf = sscanf(b + 13, "%f %f %f", specularColor, specularColor + 1, specularColor + 2);
                if (retval_sscanf != 3)
                {
                    std::cerr << "coMaterial::coMaterial: sscanf failed" << std::endl;
                    break;
                }
            }
            else if (strncasecmp(b, "emissiveColor", 13) == 0)
            {
                retval_sscanf = sscanf(b + 13, "%f %f %f", emissiveColor, emissiveColor + 1, emissiveColor + 2);
                if (retval_sscanf != 3)
                {
                    std::cerr << "coMaterial::coMaterial: sscanf failed" << std::endl;
                    break;
                }
            }
            else if (strncasecmp(b, "shininess", 9) == 0)
            {
                retval_sscanf = sscanf(b + 9, "%f", &shininess);
                if (retval_sscanf != 1)
                {
                    std::cerr << "coMaterial::coMaterial: sscanf failed" << std::endl;
                    break;
                }
            }
            else if (strncasecmp(b, "transparency", 12) == 0)
            {
                retval_sscanf = sscanf(b + 12, "%f", &transparency);
                if (retval_sscanf != 1)
                {
                    std::cerr << "coMaterial::coMaterial: sscanf failed" << std::endl;
                    break;
                }
            }
        }
        fclose(fp);
    }
}

coMaterialList::coMaterialList(const char *dirname)
{
    add(dirname);
}

void coMaterialList::add(const char *dirname)
{
    char *coviseDir;

    if ((coviseDir = getenv("COVISEDIR")) == NULL)
    {
        cerr << "*                                                             *" << endl;
        cerr << "* COVISEDIR variable not set !!!                            *" << endl;
        cerr << "*                                                             *" << endl;
        return;
    }
    char tmp[1000];
    strcpy(tmp, coviseDir);
#ifdef _WIN32
    strcat(tmp, "\\share\\covise\\materials\\");
#else
    strcat(tmp, "/share/covise/materials/");
#endif
    strcat(tmp, dirname);

#ifdef _WIN32
    char olddir[5002];
    char newpath[5002];

    getcwd(olddir, 5000);
    if (chdir(tmp + 2) >= 0)
    {
        getcwd(newpath, 5000);
        struct _finddata_t file;
        intptr_t dir_handle = _findfirst("*.*", &file);
        if (dir_handle != -1)
        {
            int find = 0;
            while (find == 0)
            {

                if (file.attrib != _A_SUBDIR)
                {
                    char buf[300];
                    char *tmp2 = new char[strlen(tmp) + strlen(file.name) + 10];
                    sprintf(tmp2, "%s\\%s", tmp, file.name);
                    sprintf(buf, "%s %s", dirname, file.name);
                    append(new coMaterial(buf, tmp2));
                }

                find = _findnext(dir_handle, &file);
            }
            _findclose(dir_handle);
        }
    }
    chdir(olddir);

#else
    coDirectory *mdir = coDirectory::open(tmp);
    if (mdir)
    {
        for (int n = 0; n < mdir->count(); n++)
        {
            if (!(mdir->is_directory(n)))
            {
                char *tmp2 = mdir->full_name(n);
                char buf[300];
                sprintf(buf, "%s %s", dirname, mdir->name(n));
                append(new coMaterial(buf, tmp2));
                delete[] tmp2;
            }
        }
        delete mdir;
    }
#endif
}

/*
coMaterialList::coMaterialList(const char *dirname)
{
    int i,n;
   Directory *dir=Directory::open("/usr/share/data/materials");
   if(dir)
   {
      for(i=0;i<dir->count();i++)
      {
         if((dir->name(i))[0]!='.')
         {
if(dir->is_directory(i))
{
char *tmp=dir->full_name(i);
Directory *mdir=Directory::open(tmp);
for(n=0;n<mdir->count();n++)
{
if(!(mdir->is_directory(n)))
{
char *tmp2=mdir->full_name(n);
char buf[300];
sprintf(buf,"%s %s",dir->name(i),mdir->name(n));
append(new coMaterial(buf,tmp2));
delete[] tmp2;
}
}
delete mdir;
delete[] tmp;
}
}
}
}
delete dir;
}*/

coMaterial *coMaterialList::get(const char *n)
{
    reset();
    while (current())
    {
        if (strcasecmp(current()->name, n) == 0)
            return (current());
        next();
    }
    return (NULL);
}
