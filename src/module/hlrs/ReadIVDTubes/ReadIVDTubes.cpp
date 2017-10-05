/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                      (C)2005 HLRS   ++
// ++ Description: ReadIVDTubes module                                      ++
// ++                                                                     ++
// ++ Author:  Uwe                                                        ++
// ++                                                                     ++
// ++                                                                     ++
// ++ Date:  2.2006                                                      ++
// ++**********************************************************************/

#include <do/coDoLines.h>
#include <do/coDoData.h>
#include <stdio.h>
#include "ReadIVDTubes.h"

#include <float.h>
#include <limits.h>
#include <string.h>

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

ReadIVDTubes::ReadIVDTubes(int argc, char *argv[])
    : coModule(argc, argv, "Read IVD lines in DX format's")
{
    // module parameters

    m_pParamFile = addFileBrowserParam("Filename", "dummy");

    m_pParamFile->setValue("./", "*.dx");

    // Output ports
    m_portLines = addOutputPort("lines", "Lines", "center lines");
    m_portLines->setInfo("Center Lines");

    m_varPorts = new coOutputPort *[NUM_SCALAR];
    m_varPorts[0] = addOutputPort("DELTAT", "Float", "Darstellung Abweichung von der mittleren Segmentemperatur AS");
    m_varPorts[1] = addOutputPort("TAS", "Float", "Temperaturdarstellung AS");
    m_varPorts[2] = addOutputPort("FAS", "Float", "Massenstrom");
    m_varPorts[3] = addOutputPort("PAS", "Float", "Druck");
    m_varPorts[4] = addOutputPort("VAS", "Float", "spez. Volumen");
    m_varPorts[5] = addOutputPort("XAS", "Float", "Dampfgehalt");
    m_varPorts[6] = addOutputPort("CAS", "Float", "cp");
    m_varPorts[7] = addOutputPort("HAS", "Float", "Enthalpie");
    m_varPorts[8] = addOutputPort("TW", "Float", "Wandtemperatur");
    m_varPorts[9] = addOutputPort("TRG", "Float", "Temperatur Rauchgas");
    m_varPorts[10] = addOutputPort("QMS", "Float", "Waermestromdichte");
    m_varPorts[11] = addOutputPort("QMR", "Float", "Waermestromdichte Rauchgas");
    m_varPorts[12] = addOutputPort("Velo", "Float", "Geschw. Wasserdampf");
    m_varPorts[13] = addOutputPort("TWRG", "Float", "Wandtemperatur Rauchgas");
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int ReadIVDTubes::compute(const char *)
{

    // get parameters
    m_filename = new char[strlen(m_pParamFile->getValue()) + 1];
    strcpy(m_filename, m_pParamFile->getValue());

    // compute parameters
    file = Covise::fopen(m_filename, "r");
    if (!file)
    {
        Covise::sendError("ERROR: can't open file %s", m_filename);
        return FAIL;
    }
    numLines = 0;
    numPoints = 0;

    while (!feof(file))
    {
        if (fgets(line, LINE_SIZE, file) == NULL)
        {
            fprintf(stderr, "ReadIVDTubes::compute(const char *): fgets failed\n");
        }

        char *c = line;
        while (*c != '\0' && isspace(*c))
        {
            c++;
        }
        if (strncmp(c, "object", 6) == 0)
        { // new line segment
            int rank = 0, shape = 0, numP = 0; //, number=0

            while (*c != '\0' && !((*c) >= '0' && (*c) <= '9'))
                c++;
            //number = atoi(c);
            while (*c != '\0' && ((*c) >= '0' && (*c) <= '9'))
                c++;
            if (strstr(c, "class array"))
            {
                while (*c != '\0' && !((*c) >= '0' && (*c) <= '9'))
                    c++;
                rank = atoi(c);
                while (*c != '\0' && ((*c) >= '0' && (*c) <= '9'))
                    c++;
                while (*c != '\0' && !((*c) >= '0' && (*c) <= '9'))
                    c++;
                shape = atoi(c);
                while (*c != '\0' && ((*c) >= '0' && (*c) <= '9'))
                    c++;
                while (*c != '\0' && !((*c) >= '0' && (*c) <= '9'))
                    c++;
                numP = atoi(c);
                while (*c != '\0' && ((*c) >= '0' && (*c) <= '9'))
                    c++;

                // data follows?

                //int num = sscanf(c,"*s %d *s *s *s *s *s %d *s %d *s %d",&number,&rank, &shape, &numP);
                if (rank == 1 && shape == 3)
                {
                    numPoints += numP;
                    numLines++;
                }
                else if (rank == 0)
                {
                    break; // we are done, no more lines
                }

                if (!strstr(c, "data follows"))
                {
                    if (fgets(line, LINE_SIZE, file) == NULL)
                    {
                        fprintf(stderr, "ReadIVDTubes::compute(const char *): fgets failed\n");
                        return FAIL;
                    }
                    if (!strstr(line, "data follows"))
                    {
                        fprintf(stderr, "ReadIVDTubes::compute(const char *): expected 'data follows'\n");
                        return FAIL;
                    }
                }

                int n;
                for (n = 0; n < numP; n++)
                {
                    if (fgets(line, LINE_SIZE, file) == NULL)
                    {
                        fprintf(stderr, "ReadIVDTubes::compute(const char *): fgets failed\n");
                        return FAIL;
                    }
                }
            }
        }
    }

    if (numLines && numPoints)
    {
        // construct the output objects

        const char *objNameLines;
        const char **objNameVars;
        int i;
        objNameLines = m_portLines->getObjName();
        objNameVars = new const char *[NUM_SCALAR];
        for (i = 0; i < NUM_SCALAR; i++)
        {
            objNameVars[i] = m_varPorts[i]->getObjName();
        }
        int *ll, *vl;
        float *xc, *yc, *zc, *scalarVals[NUM_SCALAR];

        coDoLines *linesObj = new coDoLines(objNameLines, numPoints, numPoints, numLines);
        linesObj->getAddresses(&xc, &yc, &zc, &vl, &ll);
        coDoFloat **dataObjs = new coDoFloat *[NUM_SCALAR];
        for (i = 0; i < NUM_SCALAR; i++)
        {
            dataObjs[i] = new coDoFloat(objNameVars[i], numPoints);
            dataObjs[i]->getAddress(&scalarVals[i]);
        }
        dataObjs[0]->addAttribute("SPECIES", "Darstellung Abweichung von der mittleren Segmentemperatur AS");
        dataObjs[1]->addAttribute("SPECIES", "Temperaturdarstellung AS");
        dataObjs[2]->addAttribute("SPECIES", "Massenstrom");
        dataObjs[3]->addAttribute("SPECIES", "Druck");
        dataObjs[4]->addAttribute("SPECIES", "spez. Volumen");
        dataObjs[5]->addAttribute("SPECIES", "Dampfgehalt");
        dataObjs[6]->addAttribute("SPECIES", "cp");
        dataObjs[7]->addAttribute("SPECIES", "Enthalpie");
        dataObjs[8]->addAttribute("SPECIES", "Wandtemperatur");
        dataObjs[9]->addAttribute("SPECIES", "Temperatur Rauchgas");
        dataObjs[10]->addAttribute("SPECIES", "Waermestromdichte");
        dataObjs[11]->addAttribute("SPECIES", "Waermestromdichte Rauchgas");
        dataObjs[12]->addAttribute("SPECIES", "Wandtemperatur Rauchgas");

        // Assign sets to output ports:
        m_portLines->setCurrentObject(linesObj);
        for (i = 0; i < NUM_SCALAR; i++)
        {
            m_varPorts[i]->setCurrentObject(dataObjs[i]);
        }

        // copy data into shared memory
        rewind(file);
        numPoints = 0;
        int maxLines = numLines;
        numLines = 0;

        int numVarsRead = -1;
        while (!feof(file))
        {
            if (fgets(line, LINE_SIZE, file) == NULL)
            {
                fprintf(stderr, "ReadIVDTubes::compute(const char *): fgets failed\n");
            }

            char *c = line;
            while (*c != '\0' && isspace(*c))
            {
                c++;
            }
            if (strncmp(c, "object", 6) == 0)
            { // new line segment
                int rank = 0, shape = 0, numP = 0; //, number=0

                while (*c != '\0' && !((*c) >= '0' && (*c) <= '9'))
                    c++;
                //number = atoi(c);
                while (*c != '\0' && ((*c) >= '0' && (*c) <= '9'))
                    c++;
                if (strstr(c, "class array"))
                {
                    while (*c != '\0' && !((*c) >= '0' && (*c) <= '9'))
                        c++;
                    rank = atoi(c);
                    if (rank == 1) // line coords
                    {
                        while (*c != '\0' && ((*c) >= '0' && (*c) <= '9'))
                            c++;
                        while (*c != '\0' && !((*c) >= '0' && (*c) <= '9'))
                            c++;
                        shape = atoi(c);
                        while (*c != '\0' && ((*c) >= '0' && (*c) <= '9'))
                            c++;
                        while (*c != '\0' && !((*c) >= '0' && (*c) <= '9'))
                            c++;
                        numP = atoi(c);
                        while (*c != '\0' && ((*c) >= '0' && (*c) <= '9'))
                            c++;
                        //int num = sscanf(c,"*s %d *s *s *s *s *s %d *s %d *s %d",&number,&rank, &shape, &numP);

                        if (!strstr(c, "data follows"))
                        {
                            if (fgets(line, LINE_SIZE, file) == NULL)
                            {
                                fprintf(stderr, "ReadIVDTubes::compute(const char *): fgets failed\n");
                                return FAIL;
                            }
                            if (!strstr(line, "data follows"))
                            {
                                fprintf(stderr, "ReadIVDTubes::compute(const char *): expected 'data follows'\n");
                                return FAIL;
                            }
                        }

                        if (shape == 3) // lines
                        {
                            ll[numLines] = numPoints;
                            numLines++;
                        }
                        else
                        {
                            Covise::sendError("expected shape == 3\n");
                            return FAIL;
                        }

                        int n;
                        for (n = 0; n < numP; n++)
                        {
                            if (fgets(line, LINE_SIZE, file) == NULL)
                            {
                                fprintf(stderr, "ReadIVDTubes::compute(const char *): fgets failed\n");
                                return FAIL;
                            }
                            vl[numPoints] = numPoints;
                            if (sscanf(line, "%f %f %f", &xc[numPoints], &yc[numPoints], &zc[numPoints]) != 3)
                                fprintf(stderr, "ReadIVDTubes::compute(const char *): sscanf 3 failed \n");
                            numPoints++;
                        }
                    }
                    else if (rank == 0) // scalar vars
                    {
                        while (*c != '\0' && ((*c) >= '0' && (*c) <= '9'))
                            c++;
                        while (*c != '\0' && !((*c) >= '0' && (*c) <= '9'))
                            c++;
                        numP = atoi(c);
                        while (*c != '\0' && ((*c) >= '0' && (*c) <= '9'))
                            c++;

                        if (numLines == maxLines)
                        {
                            numLines = 0;
                            numPoints = 0;
                            numVarsRead++;
                        }

                        if (!strstr(c, "data follows"))
                        {
                            if (fgets(line, LINE_SIZE, file) == NULL)
                            {
                                fprintf(stderr, "ReadIVDTubes::compute(const char *): fgets failed\n");
                                return FAIL;
                            }
                            if (!strstr(line, "data follows"))
                            {
                                fprintf(stderr, "ReadIVDTubes::compute(const char *): expected 'data follows' [%s]\n", c);
                                return FAIL;
                            }
                        }
                        int n;
                        for (n = 0; n < numP; n++)
                        {
                            if (fgets(line, LINE_SIZE, file) == NULL)
                            {
                                fprintf(stderr, "ReadIVDTubes::compute(const char *): fgets failed\n");
                                return FAIL;
                            }
                            if (sscanf(line, "%f", &scalarVals[numVarsRead][numPoints]) != 1)
                                fprintf(stderr, "ReadIVDTubes::compute(const char *): sscanf 1 failed\n");
                            numPoints++;
                        }
                        numLines++;
                    }
                }
            }
        }
    }
    fclose(file);
    return SUCCESS;
}

MODULE_MAIN(IO, ReadIVDTubes)
