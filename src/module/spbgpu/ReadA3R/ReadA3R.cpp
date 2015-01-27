/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 * ReadA3R module
 *
\****************************************************************************/

#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

#include "ReadA3R.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <api/coFeedback.h>
#include <do/coDoPolygons.h>
#include <do/coDoSet.h>
#include <fstream>

using namespace std;

A3R_HEADER::A3R_HEADER()
    : count(0)
    , r(0)
    , count_1(0)
{
    strcpy(file_type, "a3r");
    strcpy(version, "a");
    data_start = sizeof(A3R_HEADER);
}

string int2str(int n)
{
    std::ostringstream o;
    o << n;
    return o.str();
}

ReadA3R::ReadA3R(int argc, char *argv[])
    : coModule(argc, argv, "Simple ReadA3R Generation Module")
{
    p_ptsOut = addOutputPort("points", "Points", "points");

    p_file = addFileBrowserParam("File_name", "Enter file name");
    p_use_timesteps = addBooleanParam("Use_timesteps", "Check it if you use timesteps");
    p_first_step = addInt32Param("First_step", "First step");
    p_last_step = addInt32Param("Last_step", "Last step");
    p_inc = addInt32Param("Step_increment", "Step increment");
}

void ReadA3R::postInst()
{
    //p_center->show();
    //p_cusize->show();
}

int ReadA3R::read_a3r(const char *fname, vector<float> &x, vector<float> &y, vector<float> &z)
{
    filebuf file;

    if (!file.open(fname, ios::in))
    {
        sendError("read_a3r: Can not open file %s", fname);
        return 0;
    }

    A3R_HEADER header;

    file.sgetn((char *)&header, sizeof(A3R_HEADER));
    if (strcmp(header.file_type, "a3r"))
    {
        sendError("read_a3r: wrong file format: %s", fname);
        return 0;
    }
    int n = header.count;

    x.reserve(n);
    y.reserve(n);
    z.reserve(n);

    //float r = header.r;
    Vect3D::SFLOAT *buf = new Vect3D::SFLOAT[n];
    //Vect3D::SFLOAT* j = buf;

    file.sgetn((char *)buf, sizeof(Vect3D::SFLOAT) * n);
    /*
	Vect3D* start = new Vect3D[n];
	Vect3D* stop = start + n;
	for (Vect3D* i = start; i != stop; i++) *i = *j++;*/

    /*
	Atom3D* start = new Atom3D[n];
	Atom3D* stop = start + n;
	Atom3D* i;

	for (i = start; i != stop; i++)
	{
		i->r.Set(j->x, j->y, j->z); j++;
		i->v.SetRand(v_rand);
	}
	*/

    for (int i = 0; i < n; ++i)
    {
        x.push_back(buf[i].x);
        y.push_back(buf[i].y);
        z.push_back(buf[i].z);
        //sendInfo("i=%d xyz[i]=(%g %g %g )",i,x[i],y[i],z[i]);
    }
    sendInfo("%d points read, file=%s", n, fname);
    delete[] buf;
    file.close();
    return n;
}

coDoPoints *ReadA3R::create_do_points(const char *fname, const char *do_name)
{
    vector<float> x;
    vector<float> y;
    vector<float> z;

    int n = read_a3r(fname, x, y, z);

    if (n == 0)
    {
        sendError("Can not readA3R file");
    }

    coDoPoints *pts = NULL;
    pts = new coDoPoints(do_name, n, &x[0], &y[0], &z[0]);
    return pts;
}

int ReadA3R::compute(const char *port)
{
    (void)port;

    if (!p_use_timesteps->getValue())
    {
        ///////////////////
        coDoPoints *pts = NULL;
        const char *pname = p_ptsOut->getObjName();
        if (pname)
        {
            pts = create_do_points(p_file->getValue(), pname);
            if (pts == NULL)
            {
                sendError("Can not create coDoPoints");
                return FAIL;
            }
        }

        p_ptsOut->setCurrentObject(pts);
        return SUCCESS;
    }
    else
    {
        int step = p_first_step->getValue();

        int numsteps = (p_last_step->getValue() - p_first_step->getValue()) / p_inc->getValue() + 1;

        coDistributedObject **dobjs = new coDistributedObject *[numsteps + 1];

        const char *pname = p_ptsOut->getObjName();
        //for(int i=0;i<numsteps;i+=p_inc->getValue())
        for (int i = 0; i < numsteps; i++)
        {
            char fname[300];
            indexed_file_name(fname, step);

            string do_name;
            do_name.append(p_ptsOut->getObjName());
            do_name.append(string("_") + int2str(i));

            dobjs[i] = create_do_points(fname, do_name.c_str());

            step += p_inc->getValue();
        }
        dobjs[numsteps] = 0;
        string attrval = string("0 ") + int2str(numsteps - 1);
        coDistributedObject *doset = new coDoSet(pname, dobjs);
        doset->addAttribute("TIMESTEP", attrval.c_str());
        p_ptsOut->setCurrentObject(doset);
        delete[] dobjs;
    }

    return SUCCESS;
}

bool ReadA3R::indexed_file_name(char *s, int n)
{
    if (sprintf(s, p_file->getValue(), n) < 0)
        return false;
    return true;
}

ReadA3R::~ReadA3R()
{
}

void ReadA3R::param(const char *name, bool /*inMapLoading*/)
{
    sendInfo("%s", name);
}

MODULE_MAIN(Examples, ReadA3R)
