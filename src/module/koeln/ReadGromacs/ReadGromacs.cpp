/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ReadGomacs.h"
#include <iostream>
#include <fstream>
//extern "C" {
//#include "xtcio.h"
//#include "txtdump.h"
//#include "vec.h"
//}
#include <string>
#include <do/coDoData.h>
//#include <sstream>

ReadGromacs::ReadGromacs(int argc, char *argv[])
    : coModule(argc, argv, "Gromacs File Reader")
{
    // Output- Ports
    pointsOutput = addOutputPort("outPort", "Points", "Points output");
    elementout = addOutputPort("outP", "Int", "Ordnungszahl output");
    // pointsAnimationOutput = addOutputPort("outAnimation","Points","Animationoutput");

    // Parameters
    groFileParam = addFileBrowserParam("grofile", "Gromacs sturcture file");
    groFileParam->setValue("data/gromacs", "*.gro");

    // xtcFileParam=addFileBrowserParam("xtcfile","Gromacs tractory file");
    // xtcFileParam->setValue("data/gromacs","*.xtc");
}

int ReadGromacs::compute(const char *port)
{
    (void)port;
    readFileGro();
    //readFileXtc();
    return SUCCESS;
}

void ReadGromacs::readFileGro()
{
    fname = groFileParam->getValue();
    ifstream file;
    file.open(fname, ios::in);
    string s;
    getline(file, s);

    int atomcount;

    file >> atomcount;
    string trash;
    float *a;
    float *b;
    float *c;
    int *d;

    coDoPoints *outpoints = new coDoPoints(pointsOutput->getObjName(), atomcount);
    outpoints->getAddresses(&a, &b, &c);
    pointsOutput->setCurrentObject(outpoints);

    coDoInt *elementnumber = new coDoInt(elementout->getObjName(), atomcount);
    elementnumber->getAddress(&d);
    elementout->setCurrentObject(elementnumber);

    for (int i = 0; i < atomcount; i++)
    {
        string help;
        file >> trash;

        file >> help;
        help.erase(1);

        if (help == "O")
        {
            d[i] = 1;
        }
        if (help == "H")
        {
            d[i] = 2;
        }
        if (help == "C")
        {
            d[i] = 3;
        }
        if (help == "P")
        {
            d[i] = 4;
        }
        if (help == "N")
        {
            d[i] = 5;
        }
        if (help == "S")
        {
            d[i] = 6;
        }

        file >> trash;
        file >> a[i];
        file >> b[i];
        file >> c[i];
    }

    file.close();
}

void ReadGromacs::readFileXtc()
{
    int xd, indent;
    char buf[256];
    rvec *x;
    matrix box;
    int nframe, natoms, step;
    real prec, time;
    bool bOK;
    float *d;
    float *e;
    float *f;

    coDoPoints **outanimationpoints = NULL;
    coDoSet *setpoints = NULL;

    const char *filename = xtcFileParam->getValue();

    xd = open_xtc((char *)"traj.xtc", (char *)"r");
    read_first_xtc(xd, &natoms, &step, &time, box, &x, &prec, &bOK);

    outanimationpoints = new coDoPoints *[100];

    for (int h = 0; h < 100; h++)
    {
        outanimationpoints[h] = new coDoPoints(pointsAnimationOutput->getObjName(), natoms);
    }

    nframe = 0;

    do
    {
        indent = 0;
        outanimationpoints[nframe]->getAddresses(&d, &e, &f);
        for (int i = 0; i < natoms; i++)
        {
            d[i] = x[i][0];
            e[i] = x[i][1];
            f[i] = x[i][2];
        }
        nframe++;
    } while (read_next_xtc(xd, natoms, &step, &time, box, x, &prec, &bOK));

    setpoints = new coDoSet(pointsAnimationOutput->getObjName(), (coDistributedObject **)outanimationpoints);
    pointsAnimationOutput->setCurrentObject(setpoints);
}
a[i].wert = while (int = 0000; for (int i) i++)
{
    write_xtc(&prev, &x)
        i++;
    d[i];
outputanimationsoutput,marauder )
Test.812.e) zur werden per e-Mail in dem Berlauf (int z)#;
L *
}

xd = open_xtc("traj.xtc", "r");
read_first_xtc(xd, &natoms, &step, &time, box, &x, &prec, &bOK);

MODULE_MAIN(IO, ReadGromacs)
