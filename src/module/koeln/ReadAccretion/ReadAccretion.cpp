/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2002 RUS  **
 **                                                                        **
 ** Description: Read IMD checkpoint files from ITAP.                      **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                     Juergen Schulze-Doebold                            **
 **     High Performance Computing Center University of Stuttgart          **
 **                         Allmandring 30                                 **
 **                         70550 Stuttgart                                **
 **                                                                        **
 ** Cration Date: 03.09.2002                                               **
\**************************************************************************/

#include <do/coDoSet.h>
#include <do/coDoData.h>
#include <do/coDoPoints.h>
#include <api/coModule.h>
#include <alg/coChemicalElement.h>
#include <limits.h>
#include <float.h>
#include <cassert>
#include "ReadAccretion.h"

#if defined(_WIN32) && !defined(PATH_MAX)
#define PATH_MAX MAX_PATH
#endif

/// Constructor
coReadAccretion::coReadAccretion(int argc, char *argv[])
    : coModule(argc, argv, "Read XYZ files containing lists of atom positions and their element types for several time steps.")
{

    // Create ports:
    poPoint = addOutputPort("Location", "Points", "Particle location");
    poVel = addOutputPort("Velocity", "Vec3", "Particle speed");
    poMass = addOutputPort("Mass", "Float", "Particle mass");
    poInitialMass = addOutputPort("InitialMass", "Float", "Initial particle mass");
    poMask = addOutputPort("Mask", "Int", "Particle mask");
    poId = addOutputPort("Id", "Int", "Particle id");
    poAttr = addOutputPort("Attribute", "Int", "Particle attribute");

    poRStars = addOutputPort("RStars", "Points", "Star location");
    poVStars = addOutputPort("VStars", "Vec3", "Star velocity");
    poMassStars = addOutputPort("MStars", "Float", "Star mass");

    // Create parameters:
    pbrFilename = addFileBrowserParam("Filename", "Input file pattern");
    pbrFilename->setValue("data/", "*.dat/*");

    pStartTimestep = addInt32Param("StartTimestep", "Minimum timesteps to read");
    pStartTimestep->setValue(1);

    pLimitTimesteps = addInt32Param("LimitTimestep", "Maximum number of timesteps to read (0 = all)");
    pLimitTimesteps->setValue(0);
}

static bool readStarStep(FILE *fp, float x[3][2], float v[3][2])
{
    bool ok = true;

    ok &= (fscanf(fp, "%f %f %f\n", &x[0][0], &x[1][0], &x[2][0]) == 3);
    ok &= (fscanf(fp, "%f %f %f\n", &v[0][0], &v[1][0], &v[2][0]) == 3);
    ok &= (fscanf(fp, "%f %f %f\n", &x[0][1], &x[1][1], &x[2][1]) == 3);
    ok &= (fscanf(fp, "%f %f %f\n", &v[0][1], &v[1][1], &v[2][1]) == 3);
    ok &= (fscanf(fp, "\n") == 0);

    return ok;
}

static const char *objName(coOutputPort *port, int timestep)
{
    static char name[10000];
    snprintf(name, sizeof(name), "%s_%d", port->getObjName(), timestep);
    //fprintf(stderr, "objname: %s\n", name);
    return name;
}

/// Compute routine: load checkpoint file
int coReadAccretion::compute(const char *)
{
    static const int BUFLEN = 2048;
    int timestepLimit = pLimitTimesteps->getValue();
    int timestepStart = pStartTimestep->getValue();

    const char *starfile = pbrFilename->getValue();
    FILE *starfp = fopen(starfile, "r");
    if (!starfp)
    {
        sendError("failed to open star file %s for reading.", starfile);
        return STOP_PIPELINE;
    }

    char initialfile[PATH_MAX];
    strcpy(initialfile, starfile);
    char *basename = strrchr(initialfile, '/');
    if (basename)
    {
        strcpy(basename, "/initial.dat");
    }
    else
    {
        strcpy(initialfile, "initial.dat");
    }
    std::vector<float> initialMass;
    FILE *initialfp = fopen(initialfile, "r");
    if (!initialfp)
    {
        sendWarning("failed to open initial.dat file %s for reading, no initial masses available.", initialfile);
    }
    else
    {
        int lineno = 0;
        int emptylines = 0;
        while (!feof(initialfp) && !ferror(initialfp))
        {
            ++lineno;
            char linebuf[BUFLEN];
            if (!fgets(linebuf, sizeof(linebuf), initialfp))
            {
                //fprintf(stderr, "stopping reading after %d lines, %d empty\n", lineno, emptylines);
                break;
            }
            //fprintf(stderr, "line %d, len %d\n", lineno, (int)strlen(linebuf));

            // skip header
            if (!strcmp(linebuf, " \n"))
            {
                ++emptylines;
                continue;
            }
            if (emptylines < 2)
                continue;

            // parse
            int id;
            float dist, mass;
            int numparsed = sscanf(linebuf, "%i %f %f\n", &id, &mass, &dist);
            if (numparsed != 3)
            {
                sendWarning("parse error in %s, line %d.", initialfile, lineno);
                break;
            }

            while (initialMass.size() <= id)
                initialMass.push_back(0.f);
            initialMass[id] = mass;
        }
        fclose(initialfp);
    }
    //fprintf(stderr, "initial masses for %d particles\n", (int)initialMass.size());
    std::vector<int> accreted;

    // Open first checkpoint file:
    char pathformat[PATH_MAX];
    char trailingletter[2] = " ";
    for (trailingletter[0] = 'a'; trailingletter[0] < 'z'; ++trailingletter[0])
    {
        strncpy(pathformat, starfile, sizeof(pathformat));
        pathformat[sizeof(pathformat) - 2] = '\0';
        char *p = strrchr(pathformat, '.');
        if (p)
        {
            *p = '\0';
            strcat(pathformat, "%04d");
            strcat(pathformat, trailingletter);
        }
        if (!strstr(pathformat, "%"))
        {
            sendWarning("no %% in format, assuming single timestep");
            timestepLimit = 1;
        }

        {
            char path[PATH_MAX];
            snprintf(path, sizeof(path), pathformat, timestepStart);
            FILE *fp = fopen(path, "r");
            if (fp)
            {
                fclose(fp);
                break;
            }
            else
            {
                strncpy(pathformat, starfile, sizeof(pathformat));
                pathformat[sizeof(pathformat) - 2] = '\0';
                char *p = strrchr(pathformat, '.');
                if (p)
                {
                    *p = '\0';
                }
                p = strrchr(pathformat, '/');
                if (p)
                {
                    *p = '\0';
                    ++p;
                    char basename[PATH_MAX];
                    strcpy(basename, p);

                    p = strrchr(pathformat, '/');
                    if (p)
                    {
                        *p = '\0';
                        strcat(pathformat, "/");
                        strcat(pathformat, basename);
                        strcat(pathformat, "%04d");
                        strcat(pathformat, trailingletter);
                        snprintf(path, sizeof(path), pathformat, timestepStart);
                    }
                }
                snprintf(path, sizeof(path), pathformat, timestepStart);
                FILE *fp = fopen(path, "r");
                if (fp)
                {
                    fclose(fp);
                    break;
                }
            }
        }
    }
    sendInfo("trying to read filenames of type %s", pathformat);

    std::vector<coDoPoints *> points;
    std::vector<coDoVec3 *> vels;
    std::vector<coDoFloat *> masses;
    std::vector<coDoFloat *> initialMasses;
    std::vector<coDoInt *> mask;
    std::vector<coDoInt *> id;
    std::vector<coDoInt *> attrs;

    std::vector<coDoPoints *> rcenter;
    std::vector<coDoVec3 *> vcenter;
    // Read time steps one by one:
    int timestep = 0;
    int totalAtoms = 0;

    float xc[3][2], vc[3][2];
    for (timestep = 1; timestep < timestepStart; ++timestep)
    {
        if (!readStarStep(starfp, xc, vc))
        {
            fclose(starfp);
            sendError("failed to open star file %s for reading.", starfile);
            return STOP_PIPELINE;
        }
    }

    std::vector<float> xx, yy, zz, vxx, vyy, vzz, mass;
    std::vector<int> vmask, vid, vattr;
    for (timestep = timestepStart; timestepLimit == 0 || timestep < timestepLimit; ++timestep)
    {
        char path[PATH_MAX];
        snprintf(path, sizeof(path), pathformat, timestep);
        FILE *fp = fopen(path, "r");
        if (!fp)
        {
            if (timestep == timestepStart)
            {
                sendError("failed to open file %s for reading.", path);
                return STOP_PIPELINE;
            }
            else
            {
                sendInfo("read %d timesteps.", timestep - timestepStart + 1);
                break;
            }
        }

        if (!readStarStep(starfp, xc, vc))
        {
        }
        rcenter.push_back(new coDoPoints(objName(poRStars, timestep), 2, &xc[0][0], &xc[1][0], &xc[2][0]));
        vcenter.push_back(new coDoVec3(objName(poVStars, timestep), 2, &vc[0][0], &vc[1][0], &vc[2][0]));

        int numAtoms = 0;

        xx.clear();
        yy.clear();
        zz.clear();
        vxx.clear();
        vyy.clear();
        vzz.clear();
        mass.clear();
        vmask.clear();
        vid.clear();
        vattr.clear();

        bool haveAttr = true;
        (void)haveAttr;

        int lineno = 0;
        while (!feof(fp) && !ferror(fp))
        {
            ++lineno;
            ++numAtoms;
            ++totalAtoms;

            char linebuf[BUFLEN];
            if (!fgets(linebuf, sizeof(linebuf), fp))
            {
                if (feof(fp))
                    break;
                sendError("Failed to read data for particle from file=%s, line=%d.", path, lineno);
                fclose(fp);
                return STOP_PIPELINE;
            }

            int i;
            float x, y, z, vx, vy, vz, m;
            int attr;
            int numparsed = sscanf(linebuf, "%i %f %f %f %f %f %f %f %d", &i, &x, &y, &z, &vx, &vy, &vz, &m, &attr);
            if (numparsed != 9)
                haveAttr = false;
            if (numparsed < 8 || numparsed > 9)
            {
                sendError("Failed to parse data for particle: file=%s, particle=%d, line=%d.", path, i, lineno);
                fclose(fp);
                return STOP_PIPELINE;
            }

            while (accreted.size() <= i)
                accreted.push_back(0);

            while (vmask.size() <= i)
                vmask.push_back(0);
            while (vid.size() <= i)
                vid.push_back(0);

            while (mass.size() <= i)
                mass.push_back(0.f);

            for (size_t j = xx.size(); j < i; ++j)
            {
                if (j == 0)
                    continue;

                if (accreted[j] == 0 && !points.empty())
                {
                    float *prev[3];
#define dist2(a0, a1, a2, b0, b1, b2) (a0 - b0) * (a0 - b0) + (a1 - b1) * (a1 - b1) + (a2 - b2) * (a2 - b2)
                    points.back()->getAddresses(&prev[0], &prev[1], &prev[2]);
                    float dc2 = dist2(xc[0][0], xc[1][0], xc[2][0], prev[0][j - 1], prev[1][j - 1], prev[2][j - 1]);
                    float dp2 = dist2(xc[0][1], xc[1][1], xc[2][1], prev[0][j - 1], prev[1][j - 1], prev[2][j - 1]);
#undef dist2
                    if (dc2 < dp2)
                        accreted[j] = 1;
                    else
                        accreted[j] = 2;
                    //fprintf(stderr, "particle %d accreted onto %d (%f vs %f)\n", j, accreted[j], dc2, dp2);
                }

                if (accreted[j] == 1)
                {
                    xx.push_back(xc[0][0]);
                    yy.push_back(xc[1][0]);
                    zz.push_back(xc[2][0]);

                    vxx.push_back(vc[0][0]);
                    vyy.push_back(vc[1][0]);
                    vzz.push_back(vc[2][0]);
                }
                else
                {
                    xx.push_back(xc[0][1]);
                    yy.push_back(xc[1][1]);
                    zz.push_back(xc[2][1]);

                    vxx.push_back(vc[0][1]);
                    vyy.push_back(vc[1][1]);
                    vzz.push_back(vc[2][1]);
                }
            }

            while (xx.size() <= i)
                xx.push_back(0.);
            while (yy.size() <= i)
                yy.push_back(0.);
            while (zz.size() <= i)
                zz.push_back(0.);

            while (vxx.size() <= i)
                vxx.push_back(0.);
            while (vyy.size() <= i)
                vyy.push_back(0.);
            while (vzz.size() <= i)
                vzz.push_back(0.);

            while (vattr.size() <= i)
                vattr.push_back(4); // accreted - hopefully onto perturber

            accreted[i] = 0;
            xx[i] = x;
            yy[i] = y;
            zz[i] = z;
            vxx[i] = vx;
            vyy[i] = vy;
            vzz[i] = vz;

            mass[i] = m;
            vmask[i] = 1;
            vid[i] = i;
            if (numparsed == 9)
                vattr[i] = attr;
        }
        fclose(fp);

        // no particle with id 0
        points.push_back(new coDoPoints(objName(poPoint, timestep), (int)xx.size() - 1, &xx[1], &yy[1], &zz[1]));
        vels.push_back(new coDoVec3(objName(poVel, timestep), (int)vxx.size() - 1, &vxx[1], &vyy[1], &vzz[1]));

        masses.push_back(new coDoFloat(objName(poMass, timestep), (int)mass.size() - 1, &mass[1]));
        // only copy remaining particles
        while (initialMass.size() < mass.size())
            initialMass.push_back(0.f);
        initialMasses.push_back(new coDoFloat(objName(poInitialMass, timestep), (int)mass.size() - 1, &initialMass[1]));

        mask.push_back(new coDoInt(objName(poMask, timestep), (int)vmask.size() - 1, &vmask[1]));
        id.push_back(new coDoInt(objName(poId, timestep), (int)vid.size() - 1, &vid[1]));
        attrs.push_back(new coDoInt(objName(poAttr, timestep), (int)vattr.size() - 1, &vattr[1]));
    }
    float starmass[2];
    int nvals = fscanf(starfp, "M1 = %f Msun M2 = %f Msun\n", &starmass[0], &starmass[1]);
    if (nvals != 2)
    {
        sendWarning("failed to parse star masses");
    }
    fclose(starfp);

    coDoFloat *starMass = new coDoFloat(poMassStars->getObjName(), 2, &starmass[0]);

    // Create set objects:
    coDoSet *setPoints = new coDoSet(poPoint->getObjName(), (int)points.size(), (coDistributedObject **)&points[0]);
    coDoSet *setVels = new coDoSet(poVel->getObjName(), (int)vels.size(), (coDistributedObject **)&vels[0]);
    coDoSet *setMasses = new coDoSet(poMass->getObjName(), (int)masses.size(), (coDistributedObject **)&masses[0]);
    coDoSet *setInitialMasses = new coDoSet(poInitialMass->getObjName(), (int)initialMasses.size(), (coDistributedObject **)&initialMasses[0]);
    coDoSet *setMasks = new coDoSet(poMask->getObjName(), (int)mask.size(), (coDistributedObject **)&mask[0]);
    coDoSet *setIds = new coDoSet(poId->getObjName(), (int)id.size(), (coDistributedObject **)&id[0]);
    coDoSet *setAttrs = new coDoSet(poAttr->getObjName(), (int)attrs.size(), (coDistributedObject **)&attrs[0]);
    coDoSet *setRStars = new coDoSet(poRStars->getObjName(), (int)rcenter.size(), (coDistributedObject **)&rcenter[0]);
    coDoSet *setVStars = new coDoSet(poVStars->getObjName(), (int)vcenter.size(), (coDistributedObject **)&vcenter[0]);

    // Set timestep attribute:
    if (timestep - timestepStart + 1 > 1)
    {
        char buf[1024], scalebuf[1024], basebuf[1024], unitbuf[1024];
        snprintf(buf, sizeof(buf), "%d %d", 0, timestep - 1);
        snprintf(scalebuf, sizeof(scalebuf), "%i", 12);
        snprintf(basebuf, sizeof(basebuf), "0");
        snprintf(unitbuf, sizeof(unitbuf), "years");

#define ADDATTRS(codo)                             \
    codo->addAttribute("TIMESTEP", buf);           \
    codo->addAttribute("TIMESTEPBASE", basebuf);   \
    codo->addAttribute("TIMESTEPSCALE", scalebuf); \
    codo->addAttribute("TIMESTEPUNIT", unitbuf);

        ADDATTRS(setPoints);
        ADDATTRS(setVels);
        ADDATTRS(setMasses);
        ADDATTRS(setInitialMasses);
        ADDATTRS(setMasks);
        ADDATTRS(setIds);
        ADDATTRS(setAttrs);
        ADDATTRS(setVStars);
        ADDATTRS(setRStars);
#undef ADDATTRS
    }

    // Assign sets to output ports:
    poPoint->setCurrentObject(setPoints);
    poVel->setCurrentObject(setVels);
    poMass->setCurrentObject(setMasses);
    poInitialMass->setCurrentObject(setInitialMasses);
    poMask->setCurrentObject(setMasks);
    poId->setCurrentObject(setIds);
    poAttr->setCurrentObject(setAttrs);
    poRStars->setCurrentObject(setRStars);
    poVStars->setCurrentObject(setVStars);
    poMassStars->setCurrentObject(starMass);

    sendInfo("Timesteps loaded: %d (%d atoms)", timestep, totalAtoms);

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(IO, coReadAccretion)
