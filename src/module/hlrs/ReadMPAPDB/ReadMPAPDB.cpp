/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                   	                  **
 **                                                                        **
 ** Description: READ MPA pdb / rasmol files                               **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: Uwe Woessner / Martin Becker                                   **
 **                                                                        **
\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <map>
#include <iostream>
#include <iomanip>

#include <do/coDoSet.h>
#include <do/coDoPoints.h>
#include <do/coDoData.h>
#include "ReadMPAPDB.h"
#include <config/CoviseConfig.h>

ReadMPAPDB::ReadMPAPDB(int argc, char *argv[])
    : coModule(argc, argv, "Reader for MPA PDB files")
{

    m_pParamFile = addFileBrowserParam("filename", "name of first PDB file to read");
    m_pParamFile->setValue("", "*.pdb;*.rasmol/*");

    m_pUseIDFromFile = addBooleanParam("use_ID_from_file", "use atom ID from file (otherwise use atom name)");
    m_pUseIDFromFile->setValue(true);

    m_pNTimesteps = addInt32Param("n_timesteps", "number of timesteps to read");
    m_pNTimesteps->setValue(1);

    m_pStepTimesteps = addInt32Param("Step_timesteps", "read every n-th timestep (Step)");
    m_pStepTimesteps->setValue(1);

    m_portPoints = addOutputPort("points", "Points", "points Output");
    m_portPoints->setInfo("points Output");

    m_portAtomType = addOutputPort("AtomType", "Int", "atom type");
    m_portAtomType->setInfo("Atom type");

    m_portAtomID = addOutputPort("AtomID", "Int", "ID of each atom");
    m_portAtomID->setInfo("Atom ID");

    // try to add local atommapping.xml to current coviseconfig
    m_mapConfig = new coConfigGroup("Module.AtomColors");
    m_mapConfig->addConfig(coConfigDefaultPaths::getDefaultLocalConfigFilePath() + "atommapping.xml", "local", true);
    coConfig::getInstance()->addConfig(m_mapConfig);

    coCoviseConfig::ScopeEntries mappingEntries = coCoviseConfig::getScopeEntries("Module.AtomMapping");
    if (mappingEntries.empty())
    {
        // add global atommapping.xml to current coviseconfig
        m_mapConfig->addConfig(coConfigDefaultPaths::getDefaultGlobalConfigFilePath() + "atommapping.xml", "global", true);
        coConfig::getInstance()->addConfig(m_mapConfig);
        mappingEntries = coCoviseConfig::getScopeEntries("Module.AtomMapping");

        // retrieve the values of atommapping.xml and build the GUI
    }

    if (mappingEntries.empty())
        std::cout << "The scope Module.AtomMapping is not available in your covise.config file!" << std::endl;
    int iNrCurrent = 0;
    float radius;
    char cAtomName[256];
    char cAtomType[TYPELENGTH + 1];

    for (const auto &entry : mappingEntries)
    {
        AtomColor ac;
        int iScanResult = sscanf(entry.second.c_str(), "%3s %s %f %f %f %f %f", cAtomType, cAtomName, &radius, &ac.color[0], &ac.color[1], &ac.color[2], &ac.color[3]);

        if (iScanResult == 7)
        {
            m_rgb.push_back(ac);
            if (radius < 0.)
                radius = 0.;
            m_radius.push_back(radius);
            m_atomtype.push_back(ac.type);

            // convert to lower case (we want to be case-insensitive)
            for (int i = 0; i < TYPELENGTH; i++)
            {
                cAtomType[i] = tolower(cAtomType[i]);
            }
            AtomID[cAtomType] = iNrCurrent;

            //fprintf(stderr, "%d: name=%s (%s)\n", iNrCurrent+1, cAtomName, ac.type);
            if (iNrCurrent + 1 != std::stoi(entry.first))
            {
                std::cout << "Your atommapping.xml is garbled" << std::endl;
            }
        }
        iNrCurrent++;
    }
}

ReadMPAPDB::~ReadMPAPDB()
{
}

// =======================================================

int ReadMPAPDB::compute(const char *)
{

    coDistributedObject **pAtomPoints;
    coDistributedObject **pAtomTypes;
    coDistributedObject **pAtomID;

    int nTimestepsToRead;
    if (m_pStepTimesteps->getValue() == 1)
    {
        nTimestepsToRead = m_pNTimesteps->getValue();
    }
    else if (m_pStepTimesteps->getValue())
    {
        nTimestepsToRead = (m_pNTimesteps->getValue() - 1) / m_pStepTimesteps->getValue() + 1;
        cerr << "nTimestepsToRead=" << nTimestepsToRead << endl;
    }
    else
    {
        sendError("Step_timesteps should never be 0, resetting to 1");
        m_pStepTimesteps->setValue(1);
        nTimestepsToRead = 1;
    }

    if (m_pStepTimesteps->getValue() > m_pNTimesteps->getValue())
    {
        sendError("Step_timesteps > n_timesteps. stopping.");
        return STOP_PIPELINE;
    }

    if (nTimestepsToRead > 1)
    {
        pAtomPoints = new coDistributedObject *[nTimestepsToRead + 1];
        pAtomPoints[nTimestepsToRead] = NULL;

        pAtomTypes = new coDistributedObject *[nTimestepsToRead + 1];
        pAtomTypes[nTimestepsToRead] = NULL;

        pAtomID = new coDistributedObject *[nTimestepsToRead + 1];
        pAtomID[nTimestepsToRead] = NULL;
    }

    FILE *pFile;

    // construct filenames
    std::vector<string> filenames;
    int ndigits = 0;
    char *file_basename = NULL;
    char *file_number = NULL;
    int firstnumber;
    char *file_ending = NULL;

    if (m_pNTimesteps->getValue() > 1)
    {
        const char *filename;
        filename = m_pParamFile->getValue();

        const char *s1 = strrchr(filename, '.');
        if (s1 == NULL)
        {
            sendError("filename contains no '.', please rename! stopping");
            return STOP_PIPELINE;
        }
        int n = (int)(s1 - filename);

        int startdigit = n - 1;
        while (isdigit(filename[startdigit]))
        {
            startdigit--;
            if (startdigit > n)
            {
                sendError("filename contains no digits! Probably no transient simulation? stopping");
                return STOP_PIPELINE;
            }
        }
        startdigit++;

        ndigits = n - startdigit;
        //cerr << "ndigits=" << ndigits << endl;

        file_basename = strdup(filename);
        file_basename[n - ndigits] = '\0';
        //cerr << "file_basename=" << file_basename << endl;

        file_number = strdup(filename + n - ndigits);
        file_number[ndigits] = '\0';
        //cerr << "file_number=" << file_number << endl;
        firstnumber = atoi(file_number);

        file_ending = strdup(filename + n + 1);
        //cerr << "file_ending=" << file_ending << endl;

        char *filetoread = new char[strlen(file_basename) + strlen(file_ending) + ndigits + 2];
        for (int i = 0; i < m_pNTimesteps->getValue(); i++)
        {
            char *number = new char[ndigits + 1];
            char format[100];
            sprintf(format, "%%0%dd", ndigits);
            sprintf(number, format, i + firstnumber);
            //cerr << "number=" << number << endl;

            sprintf(filetoread, "%s%s.%s", file_basename, number, file_ending);
            filenames.push_back(filetoread);
            delete[] number;
            //cerr << filenames[i] << endl;
        }
        delete[] filetoread;
    }
    else
    {
        filenames.push_back(m_pParamFile->getValue());
    }

    // check number of atoms (count lines)
    // !number of atoms is not the same in all timesteps!
    // could be done faster if file we read the file only once
    // feel free to implement this ...
    int *numAtoms = new int[nTimestepsToRead];
    char buf[1025];
    buf[1024] = '\0';

    int readTimesteps = 0; // remember the number of timesteps we really have read (we can Step)

    cerr << "counting number of atoms in all files ..." << endl;
    for (int timestep = 0; timestep < m_pNTimesteps->getValue(); timestep += m_pStepTimesteps->getValue())
    {
        cerr << "timestep=" << timestep << endl;
        int nAtoms = 0;
        pFile = fopen(filenames[timestep].c_str(), "r");
        if (!pFile)
        {
            sendError("ERROR: can't open file %s", filenames[timestep].c_str());
            return STOP_PIPELINE;
        }
        while (!feof(pFile))
        {
            fgets(buf, 1024, pFile);
            nAtoms++;
        }
        numAtoms[readTimesteps] = nAtoms;

        readTimesteps++;

        //cerr << "numAtoms[" << timestep << "]=" << numAtoms[timestep] << endl;
        fclose(pFile);
    }

    // read file(s)
    char atomName[1025];
    int *at;
    int *aID;
    float *xc, *yc, *zc;
    readTimesteps = 0;

    for (int timestep = 0; timestep < m_pNTimesteps->getValue(); timestep += m_pStepTimesteps->getValue())
    {
        pFile = fopen(filenames[timestep].c_str(), "r");
        if (!pFile)
        {
            sendError("ERROR: can't open file %s", m_pParamFile->getValue());
            return STOP_PIPELINE;
        }

        cerr << "reading file " << filenames[timestep] << endl;

        if (numAtoms[readTimesteps] > 0)
        {
            char pointsname[100];
            char typesname[100];
            char idname[100];
            if (nTimestepsToRead > 1)
            {
                sprintf(pointsname, "%s_%d", m_portPoints->getObjName(), readTimesteps);
                sprintf(typesname, "%s_%d", m_portAtomType->getObjName(), readTimesteps);
                sprintf(idname, "%s_%d", m_portAtomID->getObjName(), readTimesteps);
            }
            else
            {
                sprintf(pointsname, "%s", m_portPoints->getObjName());
                sprintf(typesname, "%s", m_portAtomType->getObjName());
                sprintf(idname, "%s", m_portAtomID->getObjName());
            }
            coDoPoints *pCovisePoints = new coDoPoints(pointsname, numAtoms[readTimesteps]);
            coDoInt *pCoviseAtomType = new coDoInt(typesname, numAtoms[readTimesteps]);
            coDoInt *pCoviseAtomID = new coDoInt(idname, numAtoms[readTimesteps]);

            pCovisePoints->getAddresses(&xc, &yc, &zc);
            pCoviseAtomType->getAddress(&at);
            pCoviseAtomID->getAddress(&aID);

            // read file
            if (m_pUseIDFromFile->getValue())
            {
                while (!feof(pFile))
                {
                    // ATOM      7  NA          2       2.862   4.299   0.145  1.00  0.00
                    fread(buf, 1, 66, pFile);
                    buf[66] = '\0';
                    if (buf[13] == '\0')
                        buf[13] = 'X';
                    if (buf[14] == '\0')
                        buf[14] = 'X';
                    sscanf(buf + 5, "%d %s %d %f %f %f", aID++, atomName, at++, xc++, yc++, zc++);

                    fgets(buf, 1024, pFile);
                }
            }
            else
            {
                int id; // not used in this case, we use the atom name
                int nAtoms = 0;
                while (!feof(pFile))
                {
                    // ATOM      7  NA          2       2.862   4.299   0.145  1.00  0.00
                    fread(buf, 1, 66, pFile);
                    buf[66] = '\0';
                    if (buf[13] == '\0')
                        buf[13] = 'X';
                    if (buf[14] == '\0')
                        buf[14] = 'X';
                    sscanf(buf + 5, "%d %s %d %f %f %f", aID++, atomName, &id, xc++, yc++, zc++);

                    for (int i = 0; i < TYPELENGTH; i++)
                    {
                        atomName[i] = tolower(atomName[i]);
                    }
                    it = AtomID.find(atomName);
                    if (it == AtomID.end())
                    {
                        sendError("Could not find element %s in your config, stopping", atomName);
                        return STOP_PIPELINE;
                    }
                    at[nAtoms] = it->second + 1;
                    nAtoms++;

                    fgets(buf, 1024, pFile);
                }
            }

            if (nTimestepsToRead == 1)
            {
                m_portPoints->setCurrentObject(pCovisePoints);
                m_portAtomType->setCurrentObject(pCoviseAtomType);
                m_portAtomID->setCurrentObject(pCoviseAtomID);
            }
            else
            {
                pAtomPoints[readTimesteps] = pCovisePoints;
                pAtomTypes[readTimesteps] = pCoviseAtomType;
                pAtomID[readTimesteps] = pCoviseAtomID;
            }
        }
        else
        {
            return FAIL;
        }

        readTimesteps++;
        fclose(pFile);
    }
    delete[] numAtoms;

    if (nTimestepsToRead > 1)
    {
        coDoSet *AtomPointSet = new coDoSet(coObjInfo(m_portPoints->getObjName()), pAtomPoints);
        coDoSet *AtomTypeSet = new coDoSet(coObjInfo(m_portAtomType->getObjName()), pAtomTypes);
        coDoSet *AtomIDSet = new coDoSet(coObjInfo(m_portAtomID->getObjName()), pAtomID);

        char ts[100];
        sprintf(ts, "1 %d", (int)m_pNTimesteps->getValue());
        AtomPointSet->addAttribute("TIMESTEP", ts);
        AtomTypeSet->addAttribute("TIMESTEP", ts);
        AtomIDSet->addAttribute("TIMESTEP", ts);

        m_portPoints->setCurrentObject(AtomPointSet);
        m_portAtomType->setCurrentObject(AtomTypeSet);
        m_portAtomID->setCurrentObject(AtomIDSet);
    }

    return SUCCESS;
}

MODULE_MAIN(IO, ReadMPAPDB)
