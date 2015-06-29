/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++
// MODULE ReadNas
//
// This module interpolates data values from Cell to Vertex
// based data representation
//
// Initial version: 2002-07-17 we
// +++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++
// Changes:
// 18.05.2004 sl: triangles may be separated when their orientation
//                is too different. This makes it possible
//                to produce a better normal interpolation.
//                At the moment it is assumed a consistent orientation
//                of the triangles with that of their neighbours.
//                This assumption
//                is known to be violated by some exotic stl datasets.

#include "ReadNas.h"
#include <do/coDoData.h>
#ifndef _WIN32
#include <inttypes.h>
#endif
#include <functional>

#include <iterator>

#include <sys/types.h>
#include <sys/stat.h>

#include <errno.h>
//#include <util/coviseCompat.h>


// remove trailing path from filename
inline const char *coBasename(const char *str)
{
    const char *lastslash = strrchr(str, '/');
    if (lastslash)
        return lastslash + 1;
    else
        return str;
}

// Module set-up in Constructor
ReadNas::ReadNas(int argc, char *argv[])
    : coModule(argc, argv, "Read NAS")
{
    // file browser parameter
    p_filename = addFileBrowserParam("file_path", "Data file path");
    p_filename->setValue("data/nofile.nas", "*.nas;*.NAS;*");

    // Output ports
    p_polyOut = addOutputPort("mesh", "Polygons", "Polygons");

    d_file = NULL;
}

ReadNas::~ReadNas()
{
}

// param callback read header again after all changes
void
ReadNas::param(const char *paraName, bool inMapLoading)
{
    if (inMapLoading)
        return;

}

/**
 * Put the respective objects to the ports.
 *
 * We want to lump as much as possible covise calls together.
 */
void
ReadNas::outputObjects(
    vector<float> &x, vector<float> &y, vector<float> &z,
    vector<int> &connList, vector<int> &elemList)
{
    coDoPolygons *poly = new coDoPolygons(p_polyOut->getObjName(),
                                          x.size(),
                                          (x.size() > 0) ? &x[0] : NULL,
                                          (y.size() > 0) ? &y[0] : NULL,
                                          (z.size() > 0) ? &z[0] : NULL,
                                          connList.size(),
                                          (connList.size() > 0) ? &connList[0] : NULL,
                                          elemList.size(),
                                          (elemList.size() > 0) ? &elemList[0] : NULL);
    poly->addAttribute("vertexOrder", "2");
    poly->addAttribute("COLOR", "Grey");

    p_polyOut->setCurrentObject(poly);
}
float ReadNas::readFloat(char *buf, int pos)
{
    char fbuf[100];
    char exp[100];
    char *tmp;
    int e=0;
    int sign=1;
    tmp = buf+pos;
    int n=0;
    for(int i=0;i<16;i++)
    {
        if(i==0 || (*tmp!='-' && *tmp !='+'))
        {
            fbuf[i]=*tmp;
            fbuf[i+1]='\0';
        }
        else
        {
            if(*tmp=='-')
            {
                sign = -1;
            }
            i++;
            tmp++;
            while(i<16)
            {
                i++;
                exp[n]=*tmp;
                n++;
                exp[n]='\0';

            }
            sscanf(exp,"%d",&e);
            break;
        }
        tmp++;
    }
    float num=0.0;
    sscanf(fbuf,"%f",&num);
    if(e>0)
    {
        num = num * pow(10,sign * e);
    }
    return num;
}
// taken from old ReadNas module: 2-Pass reading
int ReadNas::readASCII()
{
    char buf[600], *cbuf, tb1[600];

    int n_coord = 0;
    int n_elem = 0;
    int n_normals = 0;
    vector<float> x_coord, y_coord, z_coord;
    vector<int> vl, el;
    std::map<int,int> nnToCoord;

    while (!feof(d_file))
    {
        if (fgets(buf, sizeof(buf), d_file) == NULL)
        {
            cerr << "ReadNas::readASCII: fgets1 failed" << endl;
        }
        cbuf = buf+9;
        while (*cbuf != '\0')
        {
            if(*cbuf == '*')
            {
                *cbuf='\0';
                if (fgets(tb1, sizeof(tb1), d_file) == NULL)
                {
                    cerr << "ReadNas::readASCII: fgets1 failed" << endl;
                }
                strcpy(cbuf,tb1+8);
                break;
            }
            cbuf++;
        }
        if(strncmp(buf,"GRID*",5)==0)
        {
            float x, y, z; 
            int nn;
            if (sscanf(buf+8, "%16d", &nn) != 1)
            {
                cerr << "ReadNas::readASCII: sscanf1 failed" << endl;
            }
            nnToCoord[nn]=x_coord.size();
            x=readFloat(buf,40);
            y=readFloat(buf,56);
            z=readFloat(buf,72);
            x_coord.push_back(x);
            y_coord.push_back(y);
            z_coord.push_back(z);
            n_coord++;
        }
        if(strncmp(buf,"CTRIA3*",7)==0)
        {
            
            int v1,v2,v3;
            if (sscanf(buf+40, "%16d%16d%16d", &v1, &v2, &v3) != 3)
            {
                cerr << "ReadNas::readASCII: sscanf1 failed" << endl;
            }
            el.push_back(vl.size());
            vl.push_back(nnToCoord[v1]);
            vl.push_back(nnToCoord[v2]);
            vl.push_back(nnToCoord[v3]);
            n_elem++;
        }
    }

    outputObjects(x_coord, y_coord, z_coord, vl, el);

    rewind(d_file);
    return CONTINUE_PIPELINE;
}

int ReadNas::compute(const char *)
{
    const char *fileName = p_filename->getValue();

    // Try to open file
    d_file = fopen(fileName, "r");
    if (!d_file)
    {
        sendError("Could not read %s: %s", fileName, strerror(errno));
        return STOP_PIPELINE;
    }
    // Now, this must be an error:
    //     No message, readHeader already cries if problems occur
    if (!d_file)
        return STOP_PIPELINE;


    int result;
    result = readASCII();
    // add filename as an attribute
    coDistributedObject *obj = p_polyOut->getCurrentObject();
    if (obj)
        obj->addAttribute("OBJECTNAME", p_filename->getValue());

    return result;
}

MODULE_MAIN(IO, ReadNas)
