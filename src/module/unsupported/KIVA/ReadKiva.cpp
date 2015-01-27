/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for Ihs data         	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Uwe Woessner                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  17.03.95  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "ReadKiva.h"
void main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();
}

//
// static stub callback functions calling the real class
// member functions
//

void Application::quitCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->quit(callbackData);
}

void Application::computeCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->compute(callbackData);
}

//
//
//..........................................................................
//
void Application::quit(void *)
{
    //
    // ...... delete your data here .....
    //
}

int Application::beginRead(int fd)
{
    // skip fortran block header returns -1 on error
    int ret;
    ret = read(fd, &blockLen, sizeof(int));
    if (ret <= 0)
    {
        return (-1);
    }
    return (0);
}

int Application::endRead(int fd)
{
    // skip fortran block header returns -1 on error
    int header, ret;
    ret = read(fd, &header, sizeof(int));
    if (ret <= 0)
    {
        Covise::sendError("KIVA: unexpected end of file");
        return (-1);
    }
    if (header != blockLen)
    {
        Covise::sendError("KIVA: Block marker expected but not found");
        return (-1);
    }
    return (0);
}

int Application::skipBlocks(int fd, int numBlocks)
{
    // skip fortran block header returns -1 on error
    int header, ret, i;
    for (i = 0; i < numBlocks; i++)
    {
        if (read(fd, &blockLen, sizeof(int)) < 0)
        {
            return (-2);
        }
        lseek(fd, blockLen, SEEK_CUR);
        ret = read(fd, &header, sizeof(int));
        if (ret <= 0)
        {
            Covise::sendError("KIVA: unexpected end of file");
            return (-1);
        }
        if (header != blockLen)
        {
            Covise::sendError("KIVA: Block marker expected but not found");
            return (-1);
        }
    }
    return (0);
}

int Application::readFloat(int fd, float &f)
{
    if (readDouble)
    {
        double d;
        if (read(fd, &d, sizeof(double)) < sizeof(double))
        {
            Covise::sendError("unexpected end of file");
            return (-1);
        }
        f = (float)d;
    }
    else
    {
        if (read(fd, &f, sizeof(float)) < sizeof(float))
        {
            Covise::sendError("unexpected end of file");
            return (-1);
        }
    }
    return (0);
}

int Application::readInt(int fd, int &integer)
{
    if (read(fd, &integer, sizeof(int)) < sizeof(int))
    {
        Covise::sendError("unexpected end of file");
        return (-1);
    }
    return (0);
}

int Application::readFloat(int fd, float *f, int num)
{
    if (readDouble)
    {
        double d[8000];
        int numread = 0, toread = num, i;
        while (toread)
        {
            if (toread > 8000)
                toread = 8000;
            if (read(fd, d, toread * sizeof(double)) < toread * sizeof(double))
            {
                Covise::sendError("unexpected end of file");
                return (-1);
            }
            for (i = 0; i < toread; i++)
            {
                f[i] = (float)d[i];
            }
            numread += toread;
            toread = num - numread;
        }
    }
    else
    {
        if (read(fd, f, num * sizeof(float)) < (num * sizeof(float)))
        {
            Covise::sendError("unexpected end of file");
            return (-1);
        }
    }
    return (0);
}

int Application::readHeader(int fd) //returns <0 on Error or FORMAT_ASCII or FORMAT_BINARY
{
    if (beginRead(fd))
    {
        lseek(fd, 0, SEEK_SET);
        return (-2);
    }
    if (read(fd, header.dataDate, 24) < 24)
    {
        Covise::sendError("KIVA: unexpected end of file in header");
        return (-1);
    }
    if (read(fd, header.dataName, 80) < 80)
    {
        Covise::sendError("KIVA: unexpected end of file in header");
        return (-1);
    }
    header.dataName[80] = '\0';
    header.dataDate[24] = '\0';
    //Covise::sendInfo(header.dataName);
    //Covise::sendInfo(header.dataDate);
    if (readFloat(fd, header.time) < 0)
    {
        return (-1);
    }
    if (readInt(fd, header.ncyc) < 0)
    {
        return (-1);
    }
    if (readFloat(fd, header.crank) < 0)
    {
        return (-1);
    }
    if (read(fd, header.jnm, 8) < 8)
    {
        Covise::sendError("KIVA: unexpected end of file in header");
        return (-1);
    }
    header.jnm[8] = '\0';
    if (readInt(fd, header.ifirst) < 0)
    {
        return (-1);
    }
    if (readInt(fd, header.ncells) < 0)
    {
        return (-1);
    }
    if (readInt(fd, header.nverts) < 0)
    {
        return (-1);
    }
    if (readFloat(fd, header.cylrad) < 0)
    {
        return (-1);
    }
    if (readFloat(fd, header.zpistn) < 0)
    {
        return (-1);
    }
    if (readFloat(fd, header.zhead) < 0)
    {
        return (-1);
    }
    if (readInt(fd, header.np) < 0)
    {
        return (-1);
    }
    if (readInt(fd, header.nrk) < 0)
    {
        return (-1);
    }
    if (readInt(fd, header.nsp) < 0)
    {
        return (-1);
    }
    if (readInt(fd, header.irez) < 0)
    {
        return (-1);
    }
    if (readInt(fd, header.numBoundaryVertices) < 0)
    {
        return (-1);
    }
    header.boundaryVertices = new int[header.numBoundaryVertices];
    if (read(fd, header.boundaryVertices, header.numBoundaryVertices * sizeof(int)) < header.numBoundaryVertices * sizeof(int))
    {
        Covise::sendError("KIVA: unexpected end of file in header");
        return (-1);
    }
    if (readInt(fd, header.iper) < 0)
    {
        return (-1);
    }
    if (readFloat(fd, header.rhop) < 0)
    {
        return (-1);
    }
    if (readFloat(fd, header.cmueps) < 0)
    {
        return (-1);
    }
    if (readInt(fd, header.naxisj) < 0)
    {
        return (-1);
    }
    if (readInt(fd, header.nregions) < 0)
    {
        return (-1);
    }
    if (endRead(fd))
    {
        return (-1);
    }
    header.numCoords = (header.nverts - header.ifirst) + 1;
    header.numElem = (header.ncells - header.ifirst) + 1;
    // sk 21.06.2001
    //header.print();
    return (FORMAT_BINARY);
}

int Application::readParticles(int fd) //returns <0 on Error
{
    int i; //,nskip;
    if (beginRead(fd))
    {
        return (-1);
    }
    rb.init(blockLen, header.np, fd);
    for (i = 0; i < (header.np / pfactor); i++)
    {
        rb.skip(pfactor - 1); //skip some particles
        if (rb.read() < 0)
            return (-1);
        rb.readFloat(px[i]);
        rb.readFloat(py[i]);
        rb.readFloat(pz[i]);
        rb.skip(2, 2); // skip to floats and two ints
        rb.readFloat(pu[i]);
        rb.readFloat(pv[i]);
        rb.readFloat(pw[i]);
        rb.readFloat(ptemp[i]);
    }
    if ((header.np - (header.np / pfactor) * pfactor))
        rb.skip((header.np - (header.np / pfactor) * pfactor));
    if (endRead(fd))
    {
        return (-1);
    }
    return (0);
}

int Application::readData(int fd) //returns <0 on Error
{
    int i;
    if (beginRead(fd))
    {
        return (-1);
    }
    rb.init(blockLen, header.numCoords + header.ifirst - 1, fd);
    rb.skip(header.ifirst - 1); //skip some unused Vertices
    for (i = 0; i < header.numCoords; i++)
    {
        if (rb.read() < 0)
            return (-1);
        rb.skip(0, 2); // skip to floats (f +fv) and no ints
        if (i < header.numElem)
        {
            rb.readFloat(x_coord[i]);
            rb.readFloat(y_coord[i]);
            rb.readFloat(z_coord[i]);
            rb.readFloat(u[i]);
            rb.readFloat(v[i]);
            rb.readFloat(w[i]);
            rb.readFloat(p[i]);
            rb.readFloat(rho[i]);
            rb.readFloat(vol[i]);
            rb.readFloat(temp[i]);
            rb.readFloat(amu[i]);
            rb.readFloat(tke[i]);
            rb.readFloat(eps[i]);
        }
        else
        {
            rb.readFloat(x_coord[i]);
            rb.readFloat(y_coord[i]);
            rb.readFloat(z_coord[i]);
            rb.readFloat(u[i]);
            rb.readFloat(v[i]);
            rb.readFloat(w[i]);
        }
    }
    /*
   if(readDouble)
       lseek(fd,((header.ifirst-1)*15*8),SEEK_CUR);
   else
       lseek(fd,((header.ifirst-1)*15*4),SEEK_CUR);
   for(i=0;i<header.numCoords;i++)
   {
       if(i<header.numElem)
       {
        if(readDouble)
        {
   lseek(fd,(2*8),SEEK_CUR); //f + fv
   }
   else
   {
   lseek(fd,(2*4),SEEK_CUR); //f + fv
   }
   readFloat(fd,x_coord[i]);
   readFloat(fd,y_coord[i]);
   readFloat(fd,z_coord[i]);
   readFloat(fd,u[i]);
   readFloat(fd,v[i]);
   readFloat(fd,w[i]);
   readFloat(fd,p[i]);
   readFloat(fd,rho[i]);
   readFloat(fd,vol[i]);
   readFloat(fd,temp[i]);
   readFloat(fd,amu[i]);
   readFloat(fd,tke[i]);
   readFloat(fd,eps[i]);
   }
   else
   {
   if(readDouble)
   {
   lseek(fd,(2*8),SEEK_CUR); //f + fv
   }
   else
   {
   lseek(fd,(2*4),SEEK_CUR); //f + fv
   }
   readFloat(fd,x_coord[i]);
   readFloat(fd,y_coord[i]);
   readFloat(fd,z_coord[i]);
   readFloat(fd,u[i]);
   readFloat(fd,v[i]);
   readFloat(fd,w[i]);
   if(readDouble)
   {
   lseek(fd,(7*8),SEEK_CUR);
   }
   else
   {
   lseek(fd,(7*4),SEEK_CUR);
   }
   }
   }*/
    if (endRead(fd))
    {
        return (-1);
    }
    return (0);
}

int Application::readConn(int fd) //returns <0 on Error
{
    int i, i1, i2, i3, i4, i5, i6, i7, i8, *vlist, *elist, *tlist;
    if (beginRead(fd))
    {
        return (-1);
    }
    rb.init(blockLen, header.numCoords + header.ifirst - 1, fd);
    rb.skip(header.ifirst - 1); //skip some unused Vertices
    /*if(readDouble)
   {
       lseek(fd,((header.ifirst-1)*(7*4+3*8)),SEEK_CUR);
   }
   else
   {
       lseek(fd,((header.ifirst-1)*10*4),SEEK_CUR);
   }*/
    int *i1tab = new int[header.numCoords];
    int *i3tab = new int[header.numCoords];
    int *i8tab = new int[header.numCoords];

    for (i = 0; i < header.numCoords; i++)
    {
        if (rb.read() < 0)
            return (-1);
        rb.readInt(i1tab[i]);
        rb.readInt(i3tab[i]);
        rb.readInt(i8tab[i]);
        /*
      readInt(fd,i1tab[i]);
      readInt(fd,i3tab[i]);
      readInt(fd,i8tab[i]);
       if(readDouble)
       {
         lseek(fd,(4*4+3*8),SEEK_CUR);
       }
       else
       {
         lseek(fd,(7*4),SEEK_CUR);
      }*/
    }
    vlist = vl;
    elist = el;
    tlist = tl;
    for (i = 0; i < header.numElem; i++)
    {
        i1 = i1tab[i] - header.ifirst;
        i2 = i3tab[i1] - header.ifirst;
        i3 = i3tab[i] - header.ifirst;
        i4 = i;
        i5 = i8tab[i1] - header.ifirst;
        i6 = i8tab[i2] - header.ifirst;
        i7 = i8tab[i3] - header.ifirst;
        i8 = i8tab[i] - header.ifirst;
        *vlist = i1;
        vlist++;
        *vlist = i2;
        vlist++;
        *vlist = i3;
        vlist++;
        *vlist = i4;
        vlist++;
        *vlist = i5;
        vlist++;
        *vlist = i6;
        vlist++;
        *vlist = i7;
        vlist++;
        *vlist = i8;
        vlist++;
        *elist = i * 8;
        *tlist = TYPE_HEXAEDER;
        elist++;
        tlist++;
    }
    delete[] i1tab;
    delete[] i3tab;
    delete[] i8tab;

    if (endRead(fd))
    {
        return (-1);
    }
    return (0);
}

void fileHeader::print()
{
    Covise::sendInfo(dataName);
    Covise::sendInfo(dataDate);
    Covise::sendInfo(jnm);
    char buf[600];
    sprintf(buf, "time: %f ncyc: %d crank: %f ifirst: %d ncells: %d nverts: %d", time, ncyc, crank, ifirst, ncells, nverts);
    Covise::sendInfo(buf);
    sprintf(buf, "cylrad: %f zpistn: %f zhead: %f np: %d nrk: %d nsp: %d", cylrad, zpistn, zhead, np, nrk, nsp);
    Covise::sendInfo(buf);
    sprintf(buf, "irez: %d numBoundaryVertices: %d iper: %d rhop: %f cmueps: %f naxisj: %d nregions: %d", irez, numBoundaryVertices, iper, rhop, cmueps, naxisj, nregions);
    Covise::sendInfo(buf);
}

void Application::compute(void *)
{
    int fd, numt, skip, ret, i, n;
    char buf[500];

    // read input parameters and data object name

    Covise::get_browser_param("path", &dataPath);
    Covise::get_scalar_param("numt", &numt);
    Covise::get_scalar_param("skip", &skip);
    Covise::get_choice_param("format", &i);
    if (i == 1)
        readDouble = 1;
    else
        readDouble = 0;
    pfactor = 1;
    Covise::get_scalar_param("pfactor", &pfactor);

    coDistributedObject **grids;
    coDistributedObject **DOSveloc;
    coDistributedObject **DOSpress;
    coDistributedObject **DOSrho;
    coDistributedObject **DOSvol;
    coDistributedObject **DOStemperature;
    coDistributedObject **DOSamu;
    coDistributedObject **DOStke;
    coDistributedObject **DOSeps;
    coDistributedObject **DOSparticles;
    coDistributedObject **DOSpveloc;
    coDistributedObject **DOSptemperature;
    grids = new coDistributedObject *[numt + 1];
    DOSveloc = new coDistributedObject *[numt + 1];
    DOSpress = new coDistributedObject *[numt + 1];
    DOSrho = new coDistributedObject *[numt + 1];
    DOSvol = new coDistributedObject *[numt + 1];
    DOStemperature = new coDistributedObject *[numt + 1];
    DOSamu = new coDistributedObject *[numt + 1];
    DOStke = new coDistributedObject *[numt + 1];
    DOSeps = new coDistributedObject *[numt + 1];
    DOSparticles = new coDistributedObject *[numt + 1];
    DOSpveloc = new coDistributedObject *[numt + 1];
    DOSptemperature = new coDistributedObject *[numt + 1];
    for (i = 0; i < numt + 1; i++)
    {
        grids[i] = NULL;
        DOSveloc[i] = NULL;
        DOSpress[i] = NULL;
        DOSrho[i] = NULL;
        DOSvol[i] = NULL;
        DOStemperature[i] = NULL;
        DOSamu[i] = NULL;
        DOStke[i] = NULL;
        DOSeps[i] = NULL;
        DOSparticles[i] = NULL;
        DOSpveloc[i] = NULL;
        DOSptemperature[i] = NULL;
    }

    Mesh = Covise::get_object_name("mesh");
    Veloc = Covise::get_object_name("velocity");
    Press = Covise::get_object_name("pressure");
    rho_name = Covise::get_object_name("rho");
    vol_name = Covise::get_object_name("vol");
    temp_name = Covise::get_object_name("temperature");
    amu_name = Covise::get_object_name("amu");
    tke_name = Covise::get_object_name("tke");
    eps_name = Covise::get_object_name("eps");
    p_name = Covise::get_object_name("particles");
    pv_name = Covise::get_object_name("pvelocity");
    pt_name = Covise::get_object_name("ptemperature");

    if ((fd = Covise::open(dataPath, O_RDONLY)) < 0)
    {
        strcpy(buf, "ERROR: Can't open file >> ");
        strcat(buf, dataPath);
        Covise::sendError(buf);
        return;
    }
    for (n = 0; n < numt; n++)
    {
        ret = readHeader(fd);
        if (ret < 0)
        {
            close(fd);
            break;
        }
        sprintf(buf, "%s_%d", Mesh, n);
        mesh = new coDoUnstructuredGrid(buf, header.numElem, header.numElem * 8, header.numCoords, 1);
        if (!mesh->objectOk())
        {
            Covise::sendError("could not create output object:");
            break;
        }
        sprintf(buf, "%s_%d", Veloc, n);
        DOveloc = new coDoVec3(buf, header.numCoords);
        if (!DOveloc->objectOk())
        {
            Covise::sendError("could not create output object:");
            break;
        }
        sprintf(buf, "%s_%d", Press, n);
        DOpress = new coDoFloat(buf, header.numElem);
        if (!DOpress->objectOk())
        {
            Covise::sendError("could not create output object:");
            break;
        }
        sprintf(buf, "%s_%d", rho_name, n);
        DOrho = new coDoFloat(buf, header.numElem);
        if (!DOrho->objectOk())
        {
            Covise::sendError("could not create output object:");
            break;
        }
        sprintf(buf, "%s_%d", vol_name, n);
        DOvol = new coDoFloat(buf, header.numElem);
        if (!DOvol->objectOk())
        {
            Covise::sendError("could not create output object:");
            break;
        }
        sprintf(buf, "%s_%d", temp_name, n);
        DOtemperature = new coDoFloat(buf, header.numElem);
        if (!DOtemperature->objectOk())
        {
            Covise::sendError("could not create output object:");
            break;
        }
        sprintf(buf, "%s_%d", amu_name, n);
        DOamu = new coDoFloat(buf, header.numElem);
        if (!DOamu->objectOk())
        {
            Covise::sendError("could not create output object:");
            break;
        }
        sprintf(buf, "%s_%d", tke_name, n);
        DOtke = new coDoFloat(buf, header.numElem);
        if (!DOtke->objectOk())
        {
            Covise::sendError("could not create output object:");
            break;
        }
        sprintf(buf, "%s_%d", eps_name, n);
        DOeps = new coDoFloat(buf, header.numElem);
        if (!DOeps->objectOk())
        {
            Covise::sendError("could not create output object:");
            break;
        }
        if (header.np > 0)
        {
            sprintf(buf, "%s_%d", p_name, n);
            DOparticles = new coDoPoints(buf, header.np / pfactor);
            if (!DOeps->objectOk())
            {
                Covise::sendError("could not create output object:");
                break;
            }
            sprintf(buf, "%s_%d", pv_name, n);
            DOpveloc = new coDoVec3(buf, header.np / pfactor);
            if (!DOeps->objectOk())
            {
                Covise::sendError("could not create output object:");
                break;
            }
            sprintf(buf, "%s_%d", pt_name, n);
            DOptemperature = new coDoFloat(buf, header.np / pfactor);
            if (!DOeps->objectOk())
            {
                Covise::sendError("could not create output object:");
                break;
            }
        }
        mesh->getAddresses(&el, &vl, &x_coord, &y_coord, &z_coord);
        mesh->getTypeList(&tl);
        DOveloc->getAddresses(&u, &v, &w);
        DOpress->getAddress(&p);
        DOrho->getAddress(&rho);
        DOvol->getAddress(&vol);
        DOtemperature->getAddress(&temp);
        DOamu->getAddress(&amu);
        DOtke->getAddress(&tke);
        DOeps->getAddress(&eps);
        if (header.np > 0)
        {
            DOparticles->getAddresses(&px, &py, &pz);
            DOpveloc->getAddresses(&pu, &pv, &pw);
            DOptemperature->getAddress(&ptemp);
        }

        if (readData(fd) < 0)
        {
            close(fd);
            break;
        }
        skipBlocks(fd, 1);
        if (readConn(fd) < 0)
        {
            close(fd);
            break;
        }
        if (header.iper > 0)
            skipBlocks(fd, 1);
        if (header.np > 0)
            readParticles(fd);

        grids[n] = mesh;
        DOSveloc[n] = DOveloc;
        DOSpress[n] = DOpress;
        DOSrho[n] = DOrho;
        DOSvol[n] = DOvol;
        DOStemperature[n] = DOtemperature;
        DOSamu[n] = DOamu;
        DOStke[n] = DOtke;
        DOSeps[n] = DOeps;
        if (header.np > 0)
        {
            DOSparticles[n] = DOparticles;
            DOSpveloc[n] = DOpveloc;
            DOSptemperature[n] = DOptemperature;
        }

        if (header.iper > 0)
        {
            if (header.np > 0)
                skipBlocks(fd, skip * 5);
            else
                skipBlocks(fd, skip * 4);
        }
        else if (header.np > 0)
            skipBlocks(fd, skip * 4);
        else
            skipBlocks(fd, skip * 3);
    }

    coDoSet *set = new coDoSet(Mesh, grids);
    set->addAttribute("TIMESTEP", "1 1000");
    delete set;
    set = new coDoSet(Veloc, DOSveloc);
    set->addAttribute("TIMESTEP", "1 1000");
    delete set;
    set = new coDoSet(Press, DOSpress);
    set->addAttribute("TIMESTEP", "1 1000");
    delete set;
    set = new coDoSet(rho_name, DOSrho);
    set->addAttribute("TIMESTEP", "1 1000");
    delete set;
    set = new coDoSet(vol_name, DOSvol);
    set->addAttribute("TIMESTEP", "1 1000");
    delete set;
    set = new coDoSet(temp_name, DOStemperature);
    set->addAttribute("TIMESTEP", "1 1000");
    delete set;
    set = new coDoSet(amu_name, DOSamu);
    set->addAttribute("TIMESTEP", "1 1000");
    delete set;
    set = new coDoSet(tke_name, DOStke);
    set->addAttribute("TIMESTEP", "1 1000");
    delete set;
    set = new coDoSet(eps_name, DOSeps);
    set->addAttribute("TIMESTEP", "1 1000");
    delete set;
    if (header.np > 0)
    {
        set = new coDoSet(p_name, DOSparticles);
        set->addAttribute("TIMESTEP", "1 1000");
        delete set;
        set = new coDoSet(pv_name, DOSpveloc);
        delete set;
        set = new coDoSet(pt_name, DOSptemperature);
        delete set;
    }

    for (i = 0; i < n; i++)
    {
        delete grids[i];
        delete DOSveloc[i];
        delete DOSpress[i];
        delete DOSrho[i];
        delete DOSvol[i];
        delete DOStemperature[i];
        delete DOSamu[i];
        delete DOStke[i];
        delete DOSeps[i];
        delete DOSparticles[i];
        delete DOSpveloc[i];
        delete DOSptemperature[i];
    }
    delete[] grids;
    delete[] DOSveloc;
    delete[] DOSpress;
    delete[] DOSrho;
    delete[] DOSvol;
    delete[] DOStemperature;
    delete[] DOSamu;
    delete[] DOStke;
    delete[] DOSeps;
    delete[] DOSparticles;
    delete[] DOSpveloc;
    delete[] DOSptemperature;
    close(fd);
}
