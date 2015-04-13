/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "NodalFile.h"
#include "AscStream.h"
#include "istreamFTN.h"
#include <util/coviseCompat.h>

NodalFile::NodalFile(const char *filename, int filetype)
    : dataTab(NULL)
{
    int i, j;
    if (filetype == NBINARY)
    {
        int dataSize;
        istreamFTN input(::open(filename, O_RDONLY));
        if (input.fail())
        {
            nnodes = 0;
            return;
        }

        // Read Record 1
        input.readFTN((void *)&header, sizeof(Record1));
        header.title[319] = '\0';
        //printf("HEADER TITLE: %s\n", header.title);
        nnodes = header.nnodes;

        // cerr << "Number of nodes in nodal results file: " << nnodes << endl;

        if (header.nwidth > NMAXEL)
        {
            cerr << "illegal: cannot read because NMAXEL Error"
                 << endl;
            nnodes = 0;
            return;
        }

        // Read Record 2 and 3
        input.readFTN(subtitle1, 80 * sizeof(int));
        input.readFTN(subtitle2, 80 * sizeof(int));
        subtitle1[319] = '\0';
        subtitle2[319] = '\0';

        // initialize Data Record Table
        dataTab = new DataRecord[nnodes];
        for (i = 0; i < nnodes; i++)
            for (j = 0; j < NMAXEL; j++)
                dataTab[i].data[j] = 0.0;

        // Read Data Record Table
        dataSize = sizeof(int) + header.nwidth * sizeof(float);
        for (i = 0; i < nnodes; i++)
        {
            input.readFTN((void *)(dataTab + i), (size_t)dataSize);
            /*
         cerr << dataTab[i].nodid << endl;
         cerr << dataTab[i].data[0] << "\t" << dataTab[i].data[1] << "\t"
         << dataTab[i].data[2] << "\t" << dataTab[i].data[3] << "\t"
         << dataTab[i].data[4] << "\t" << dataTab[i].data[5] << "\t"
         << dataTab[i].data[6] << "\t" << dataTab[i].data[7] << "\t"
         << dataTab[i].data[8] << "\t" << dataTab[i].data[9] << "\t"
         << dataTab[i].data[10] << "\t" << dataTab[i].data[11] << endl;
         */
        }
    }
    else if (filetype == NASCII)
    {
        ifstream infile(filename);
        if (infile.fail())
        {
            nnodes = 0;
            return;
        }

        AscStream infile_line(&infile);
        char buffer[NMAXCOL];
        int skip_lines;

        // Read Record 1
        infile_line.getline(buffer, NMAXCOL);
        infile_line.getline(buffer, NMAXCOL);
        if (sscanf(buffer, "%d %d %f %d %d",
                   &(header.nnodes),
                   &(header.maxnod),
                   &(header.defmax),
                   &(header.ndmax),
                   &(header.nwidth)) != 5)
        {
	  header.defmax = 0;
	  header.ndmax  = 0;
	  if (sscanf(buffer, "%d %d %d",
		     &(header.nnodes),
		     &(header.maxnod),
		     &(header.nwidth)) != 3)
	    {
	      
	      fprintf(stderr, "NodalFile::NodalFile: sscanf failed\n");
	    }
	}
        /* input.parseString( buffer, 0, 4, &(header.nnodes) );
       input.parseString( buffer, 5, 9, &(header.maxnod) );
       input.parseString( buffer, 10, 25, &(header.defmax) );
       input.parseString( buffer, 26, 30, &(header.ndmax) );
       input.parseString( buffer, 31, 36, &(header.nwidth) );*/
        if (header.nwidth > NMAXEL)
        {
            Covise::sendError("Too many columns in result file");
            nnodes = 0;
            return;
        }

        infile_line.getline(header.title, NMAXCOL);
        header.title[319] = '\0';
        infile_line.getline(buffer, NMAXCOL);
        nnodes = header.nnodes;

        dataTab = new DataRecord[nnodes];
        i = 0;
        while (infile_line.getline(buffer, NMAXCOL))
        {
            input.parseString(buffer, 0, 7, &(dataTab[i].nodid));
            input.parseString(buffer, 8, 20, &(dataTab[i].data[0]));
            input.parseString(buffer, 21, 33, &(dataTab[i].data[1]));
            input.parseString(buffer, 34, 46, &(dataTab[i].data[2]));
            input.parseString(buffer, 47, 59, &(dataTab[i].data[3]));
            input.parseString(buffer, 60, 72, &(dataTab[i].data[4]));
            skip_lines = (header.nwidth % 5 == 0) ? header.nwidth / 5 - 1 : header.nwidth / 5;
            for (j = 1; j <= skip_lines; j++)
                infile_line.getline(buffer, NMAXCOL);
            i++;
        }
    }
}

NodalFile::~NodalFile()
{
    delete[] dataTab;
}

int NodalFile::getDataField(int fieldFlag, const int *nodeMapping,
                            float *f1, float *f2, float *f3, int diff, const int maxnid)
{
    int i;
    int nodeNo; // COVISE vertex number
    int nodeId;

    if (nodeMapping == NULL)
        return -1;

    // Displacements
    if (fieldFlag == NDISPLACEMENTS)
    {
        // initializing nodal displacements data (also from missing nodes)
        for (i = 0; i < nnodes + diff; i++)
        {
            f1[i] = 0.0;
            f2[i] = 0.0;
            f3[i] = 0.0;
        }

        // ( unused nodes allowed )
        for (i = 0; i < nnodes; i++)
        {
            nodeId = dataTab[i].nodid;
            // omit unised nodes
            if (nodeId >= 0 && nodeId <= maxnid)
            {
                nodeNo = nodeMapping[nodeId];
                if (nodeNo != -1) // node used in FE model
                {
                    f1[nodeNo] = dataTab[i].data[0];
                    f2[nodeNo] = dataTab[i].data[1];
                    f3[nodeNo] = dataTab[i].data[2];
                }
            }
        }
    }
    return 0;
}

int NodalFile::getDataField(int fieldFlag, const int *nodeMapping,
                            int col, float *f1, int diff, const int maxnid)
{
    int i;
    int nodeId;
    int nodeNo; // COVISE vertex number

    if (nodeMapping == NULL)
        return -1;

    // Nodal stress
    if (fieldFlag == NNODALSTRESS)
    {

        // initializing nodal data (also from missing nodes)
        for (i = 0; i < nnodes + diff; i++)
            f1[i] = 0.0;

        // ( unused nodes allowed )
        for (i = 0; i < nnodes; i++)
        {
            nodeId = dataTab[i].nodid;
            // omit unused nodes
            if (nodeId >= 0 && nodeId <= maxnid)
            {
                nodeNo = nodeMapping[nodeId];
                if (nodeId != -1) // node used in FE model
                {
                    if (col < 1 || col > header.nwidth) // wrong column number
                        return -1;

                    f1[nodeNo] = dataTab[i].data[col - 1];
                    // cerr << nodeId << "\t" << nodeNo << "\t" << f1[nodeNo] << endl;
                }
            }
        }
    }
    return 0;
}
