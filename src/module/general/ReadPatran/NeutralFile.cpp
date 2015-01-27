/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "NeutralFile.h"
#include <util/coviseCompat.h>
#include <api/coModule.h>
#include "AscStream.h"

using namespace covise;

const int NeutralFile::numVert[9] = { 0, 0, 2, 3, 4, 4, 0, 6, 8 };

NeutralFile::NeutralFile(const char *filename)
    : nodeTab(NULL)
    , elemTab(NULL)
    , compTab(NULL)
    , nodeMap(NULL)
    , elemMap(NULL)
{
    ifstream input(filename); // input file stream
    if (input.fail())
    {
        num_nodes = 0;
        num_elements = 0;
        num_components = 0;
        return;
    }

    AscStream input_line(&input);

    char buffer[MAXLINE];
    int i;
    int nn, ne, nc;
    //int tmp_int;
    int lc;
    int np;
    //float tmp_float;

    //======================= parsing =======================
    num_connections = 0;

    num_nodes = 0;
    num_elements = 0;
    num_components = 0;
    maxnode = 0;
    maxelem = 0;
    numTemperatures = 0;
    maxTemperatures = 0;

    while (input_line.getline(buffer, MAXLINE))
    {
        // exploit header card
        /*    sscanf(buffer, "%d %d %d %d %d %d %d %d %d",
                   &(header.it),
                   &(header.id),
                   &(header.iv),
                   &(header.kc),
                   &(header.n1),
                   &(header.n2),
                   &(header.n3),
                   &(header.n4),
                   &(header.n5)
                ); */
        in.parseString(buffer, 0, 1, &(header.it));
        in.parseString(buffer, 2, 9, &(header.id));
        in.parseString(buffer, 10, 17, &(header.iv));
        in.parseString(buffer, 18, 25, &(header.kc));
        in.parseString(buffer, 26, 33, &(header.n1));
        in.parseString(buffer, 34, 41, &(header.n2));
        in.parseString(buffer, 42, 49, &(header.n3));
        in.parseString(buffer, 50, 57, &(header.n4));
        in.parseString(buffer, 58, 65, &(header.n5));

        switch (header.it)
        {

        case 1:
            num_nodes++;
            maxnode = Max(maxnode, header.id);
            skip_lines(input, header.kc);
            break;

        case 2:
            num_elements++;
            maxelem = Max(maxelem, header.id);
            skip_lines(input, header.kc);
            break;
        case 3:
            Covise::sendWarning("Material information ignored");
            break;
        case 7:
            Covise::sendWarning("Node forces ignored");
            break;

        case 10:
            numTemperatures++;
            maxTemperatures = Max(maxTemperatures, header.id);
            skip_lines(input, header.kc);
            break;

        case 11:
            Covise::sendWarning("Element temperature ignored");
            break;

        case 21:
            num_components++;
            skip_lines(input, header.kc);
            break;

        case 25:
            input_line.getline(title, MAXLINE);
            skip_lines(input, header.kc - 1, true);
            break;

        case 26:
            summary.nnodes = header.n1;
            summary.nelements = header.n2;
            summary.nm = header.n3;
            summary.np = header.n4;
            summary.ncf = header.n5;
            input_line.getline(summary.subtitle, MAXLINE);
            skip_lines(input, header.kc - 1, true);
            break;

        case 31:
            Covise::sendWarning("Grid Data information not implemented yet");
            break;
        case 32:
            Covise::sendWarning("Line Data information not implemented yet");
            break;
        case 33:
            Covise::sendWarning("Patch Data information not implemented yet");
            break;
        case 99:
            break;

        default: // skip unsupported data cards
            skip_lines(input, header.kc);
            break;
        }
    }

    nodeTab = new Node[num_nodes];
    assert(nodeTab != 0);
    elemTab = new Element[num_elements];
    assert(elemTab != 0);
    ElemIdRef = new int[num_elements];
    assert(ElemIdRef != 0);
    compTab = new Component[num_components];
    assert(compTab != 0);
    if (numTemperatures)
    {
        cerr << numTemperatures << " temperature values found\n";
    }

    nodeMap = new int[maxnode + 1];
    assert(nodeMap != 0);
    for (i = 0; i < maxnode + 1; i++)
        nodeMap[i] = -1; // -1 == unused

    elemMap = new int[maxelem + 1];
    assert(elemMap != 0);
    for (i = 0; i < maxelem + 1; i++)
        elemMap[i] = -1; // -1 == unused

    //======================= reading grid data =======================
    input.clear(); // cancel status flags
    input.seekg(0, ios::beg); // go to the beginning of the input file

    nn = ne = nc = 0;

    maxId = 0;
    while (input_line.getline(buffer, MAXLINE))
    {
        // exploit header card
        /* sscanf(buffer, "%d %d %d %d %d %d %d %d %d",
                &(header.it),
                &(header.id),
                &(header.iv),
                &(header.kc),
                &(header.n1),
                &(header.n2),
                &(header.n3),
                &(header.n4),
                &(header.n5)
             ); */
        in.parseString(buffer, 0, 1, &(header.it));
        in.parseString(buffer, 2, 9, &(header.id));
        in.parseString(buffer, 10, 17, &(header.iv));
        in.parseString(buffer, 18, 25, &(header.kc));
        in.parseString(buffer, 26, 33, &(header.n1));
        in.parseString(buffer, 34, 41, &(header.n2));
        in.parseString(buffer, 42, 49, &(header.n3));
        in.parseString(buffer, 50, 57, &(header.n4));
        in.parseString(buffer, 58, 65, &(header.n5));

        switch (header.it)
        {

        case 1: // read node table
            nodeTab[nn].set(header.id);
            nodeMap[header.id] = nn;
            if (header.kc != 2)
            {
                cerr << "error: wrong card counter in  Packet Type 1" << endl;
                break;
            }
            input_line.getline(buffer, MAXLINE);
            /*sscanf(buffer, "%f %f %f",
                            nodeTab[nn].coord,
                            nodeTab[nn].coord+1,
                            nodeTab[nn].coord+2
                  );*/
            in.parseString(buffer, 0, 15, nodeTab[nn].coord);
            in.parseString(buffer, 16, 31, nodeTab[nn].coord + 1);
            in.parseString(buffer, 32, 47, nodeTab[nn].coord + 2);
            skip_lines(input, 1);
            nn++;
            break;
        case 10: // read node temperatures
            input_line.getline(buffer, MAXLINE);
            if (nodeMap[header.id] >= num_nodes || nodeMap[header.id] < 0)
            {
            }
            else
            {
                if (sscanf(buffer, "%f", &(nodeTab[nodeMap[header.id]].temperature)) != 1)
                {
                    fprintf(stderr, "NeutralFile::NeutralFile:: sscanf1 failed\n");
                }
            }

            break;

        case 2: // read element table
            elemTab[ne].set(header.id, (SHAPE)header.iv);
            if (header.iv > 8)
            {
                cerr << "Error in el " << ne << " " << header.id << " " << header.iv << endl;
            }
            ElemIdRef[ne] = header.id;
            ElemIdRef[ne]--;
            if (ElemIdRef[ne] > maxId)
                maxId = ElemIdRef[ne];
            elemMap[header.id] = ne;
            input_line.getline(buffer, MAXLINE);
            /* sscanf(buffer, "%d %d %d %d %f %f %f",
                            &tmp_int,
                            &tmp_int,
                            &(elemTab[ne].pid),
                            &tmp_int,
                            &tmp_float,
                            &tmp_float,
                            &tmp_float
                  ); */
            in.parseString(buffer, 9, 16, &(elemTab[ne].pid));

            input_line.getline(buffer, MAXLINE);
            /* sscanf(buffer, "%d %d %d %d %d %d %d %d",
                            elemTab[ne].lnodes,
                            elemTab[ne].lnodes+1,
                            elemTab[ne].lnodes+2,
                            elemTab[ne].lnodes+3,
                            elemTab[ne].lnodes+4,
                            elemTab[ne].lnodes+5,
                            elemTab[ne].lnodes+6,
                            elemTab[ne].lnodes+7
                  );    */
            in.parseString(buffer, 0, 7, elemTab[ne].lnodes);
            in.parseString(buffer, 8, 15, elemTab[ne].lnodes + 1);
            in.parseString(buffer, 16, 23, elemTab[ne].lnodes + 2);
            in.parseString(buffer, 24, 31, elemTab[ne].lnodes + 3);
            in.parseString(buffer, 32, 39, elemTab[ne].lnodes + 4);
            in.parseString(buffer, 40, 47, elemTab[ne].lnodes + 5);
            in.parseString(buffer, 48, 55, elemTab[ne].lnodes + 6);
            in.parseString(buffer, 56, 63, elemTab[ne].lnodes + 7);
            /*for( i=0; i<header.kc-2; i++ ) {
               input_line.getline(buffer, MAXLINE);
            }*/
            skip_lines(input, header.kc - 2);
            ne++;
            break;

        case 21: // read component table
            compTab[nc].set(header.id, header.iv);
            compTab[nc].tab = new Component::dataPair[(header.iv) / 2];
            input_line.getline(compTab[nc].name, MAXLINE);
            lc = 0; // line counter
            for (i = 0; i < header.kc - 2; i++)
            {
                input_line.getline(buffer, MAXLINE);
                /*sscanf(buffer,"%d %d %d %d %d %d %d %d %d %d",
                              &(compTab[nc].tab[5*i].ntype),
                              &(compTab[nc].tab[5*i].id),
                              &(compTab[nc].tab[5*i+1].ntype),
                              &(compTab[nc].tab[5*i+1].id),
                              &(compTab[nc].tab[5*i+2].ntype),
                              &(compTab[nc].tab[5*i+2].id),
                              &(compTab[nc].tab[5*i+3].ntype),
                              &(compTab[nc].tab[5*i+3].id),
                              &(compTab[nc].tab[5*i+4].ntype),
                              &(compTab[nc].tab[5*i+4].id)
               ); */
                in.parseString(buffer, 0, 7, (int *)(void *)&(compTab[nc].tab[5 * i].ntype));
                in.parseString(buffer, 8, 15, &(compTab[nc].tab[5 * i].id));
                in.parseString(buffer, 16, 23, (int *)(void *)&(compTab[nc].tab[5 * i + 1].ntype));
                in.parseString(buffer, 24, 31, &(compTab[nc].tab[5 * i + 1].id));
                in.parseString(buffer, 32, 39, (int *)(void *)&(compTab[nc].tab[5 * i + 2].ntype));
                in.parseString(buffer, 40, 47, &(compTab[nc].tab[5 * i + 2].id));
                in.parseString(buffer, 48, 55, (int *)(void *)&(compTab[nc].tab[5 * i + 3].ntype));
                in.parseString(buffer, 56, 63, &(compTab[nc].tab[5 * i + 3].id));
                in.parseString(buffer, 64, 71, (int *)(void *)&(compTab[nc].tab[5 * i + 4].ntype));
                in.parseString(buffer, 72, 79, &(compTab[nc].tab[5 * i + 4].id));
                lc++;
            }
            // last line of Packet Type 21
            input_line.getline(buffer, MAXLINE);
            np = (compTab[nc].num) / 2 - 5 * lc; // remaining data pairs
            switch (np)
            {
            case 1:
                /* sscanf(buffer,"%d %d",
                                                                &(compTab[nc].tab[5*lc].ntype),
                                                                &(compTab[nc].tab[5*lc].id)
                                                       ); */
                in.parseString(buffer, 0, 7, (int *)(void *)&(compTab[nc].tab[5 * lc].ntype));
                in.parseString(buffer, 8, 15, &(compTab[nc].tab[5 * lc].id));
                break;
            case 2:
                /*sscanf(buffer,"%d %d %d %d",
                                                                &(compTab[nc].tab[5*lc].ntype),
                                                                &(compTab[nc].tab[5*lc].id),
                                                                &(compTab[nc].tab[5*lc+1].ntype),
                                                                &(compTab[nc].tab[5*lc+1].id)
                                                       ); */
                in.parseString(buffer, 0, 7, (int *)(void *)&(compTab[nc].tab[5 * lc].ntype));
                in.parseString(buffer, 8, 15, &(compTab[nc].tab[5 * lc].id));
                in.parseString(buffer, 16, 23, (int *)(void *)&(compTab[nc].tab[5 * lc + 1].ntype));
                in.parseString(buffer, 24, 31, &(compTab[nc].tab[5 * lc + 1].id));

                break;
            case 3:
                /*sscanf(buffer,"%d %d %d %d %d %d",
                                                                &(compTab[nc].tab[5*lc].ntype),
                                                                &(compTab[nc].tab[5*lc].id),
                                                                &(compTab[nc].tab[5*lc+1].ntype),
                                                                &(compTab[nc].tab[5*lc+1].id),
                                                                &(compTab[nc].tab[5*lc+2].ntype),
                                                                &(compTab[nc].tab[5*lc+2].id)
                                                       ); */
                in.parseString(buffer, 0, 7, (int *)(void *)&(compTab[nc].tab[5 * lc].ntype));
                in.parseString(buffer, 8, 15, &(compTab[nc].tab[5 * lc].id));
                in.parseString(buffer, 16, 23, (int *)(void *)&(compTab[nc].tab[5 * lc + 1].ntype));
                in.parseString(buffer, 24, 31, &(compTab[nc].tab[5 * lc + 1].id));
                in.parseString(buffer, 32, 39, (int *)(void *)&(compTab[nc].tab[5 * lc + 2].ntype));
                in.parseString(buffer, 40, 47, &(compTab[nc].tab[5 * lc + 2].id));
                break;
            case 4:
                /*sscanf(buffer,"%d %d %d %d %d %d %d %d",
                                                                &(compTab[nc].tab[5*lc].ntype),
                                                                &(compTab[nc].tab[5*lc].id),
                                                                &(compTab[nc].tab[5*lc+1].ntype),
                                                                &(compTab[nc].tab[5*lc+1].id),
                                                                &(compTab[nc].tab[5*lc+2].ntype),
                                                                &(compTab[nc].tab[5*lc+2].id),
                                                                &(compTab[nc].tab[5*lc+3].ntype),
                                                                &(compTab[nc].tab[5*lc+3].id)
                                                       ); */
                in.parseString(buffer, 0, 7, (int *)(void *)&(compTab[nc].tab[5 * lc].ntype));
                in.parseString(buffer, 8, 15, &(compTab[nc].tab[5 * lc].id));
                in.parseString(buffer, 16, 23, (int *)(void *)&(compTab[nc].tab[5 * lc + 1].ntype));
                in.parseString(buffer, 24, 31, &(compTab[nc].tab[5 * lc + 1].id));
                in.parseString(buffer, 32, 39, (int *)(void *)&(compTab[nc].tab[5 * lc + 2].ntype));
                in.parseString(buffer, 40, 47, &(compTab[nc].tab[5 * lc + 2].id));
                in.parseString(buffer, 48, 55, (int *)(void *)&(compTab[nc].tab[5 * lc + 3].ntype));
                in.parseString(buffer, 56, 63, &(compTab[nc].tab[5 * lc + 3].id));
                break;
            case 5:
                /*sscanf(buffer,"%d %d %d %d %d %d %d %d %d %d",
                                                                &(compTab[nc].tab[5*lc].ntype),
                                                                &(compTab[nc].tab[5*lc].id),
                                                                &(compTab[nc].tab[5*lc+1].ntype),
                                                                &(compTab[nc].tab[5*lc+1].id),
                                                                &(compTab[nc].tab[5*lc+2].ntype),
                                                                &(compTab[nc].tab[5*lc+2].id),
                                                                &(compTab[nc].tab[5*lc+3].ntype),
                                                                &(compTab[nc].tab[5*lc+3].id),
                                                                &(compTab[nc].tab[5*lc+4].ntype),
                                                                &(compTab[nc].tab[5*lc+4].id)
                  );  */
                in.parseString(buffer, 0, 7, (int *)(void *)&(compTab[nc].tab[5 * lc].ntype));
                in.parseString(buffer, 8, 15, &(compTab[nc].tab[5 * lc].id));
                in.parseString(buffer, 16, 23, (int *)(void *)&(compTab[nc].tab[5 * lc + 1].ntype));
                in.parseString(buffer, 24, 31, &(compTab[nc].tab[5 * lc + 1].id));
                in.parseString(buffer, 32, 39, (int *)(void *)&(compTab[nc].tab[5 * lc + 2].ntype));
                in.parseString(buffer, 40, 47, &(compTab[nc].tab[5 * lc + 2].id));
                in.parseString(buffer, 48, 55, (int *)(void *)&(compTab[nc].tab[5 * lc + 3].ntype));
                in.parseString(buffer, 56, 63, &(compTab[nc].tab[5 * lc + 3].id));
                in.parseString(buffer, 64, 71, (int *)(void *)&(compTab[nc].tab[5 * lc + 4].ntype));
                in.parseString(buffer, 72, 79, &(compTab[nc].tab[5 * lc + 4].id));
                break;
            }
            nc++;
            break;

        case 25:
            skip_lines(input, header.kc, true);
            break;

        case 26:
            skip_lines(input, header.kc, true);
            break;

        case 99:
            break;

        default: // skip unsupported data cards
            skip_lines(input, header.kc);
            break;
        }
    }

    IdElemRef = new int[maxId + 1];

    for (i = 0; i < maxId + 1; i++)
        IdElemRef[i] = -1;

    for (i = 0; i < ne; i++)
    {
        IdElemRef[ElemIdRef[i]] = i;
    }

    /*
     // verify input
     for (i=0; i<num_nodes; i++)
       {
         cerr << i << "\t" << nodeTab[i].id
              << "\t" << nodeTab[i].coord[0]
              << "\t" << nodeTab[i].coord[1]
              << "\t" << nodeTab[i].coord[2] << endl;
       }

     for (i=0; i<num_elements; i++)
   {
   cerr << i << "\t" << elemTab[i].id
   << "\t" << elemTab[i].shape << "\t" << elemTab[i].pid  << endl;
   cerr << "\t" << elemTab[i].lnodes[0]
   << "\t" << elemTab[i].lnodes[1]
   << "\t" << elemTab[i].lnodes[2]
   << "\t" << elemTab[i].lnodes[3]
   << "\t" << elemTab[i].lnodes[4]
   << "\t" << elemTab[i].lnodes[5]
   << "\t" << elemTab[i].lnodes[6]
   << "\t" << elemTab[i].lnodes[7] << endl;
   }

   for (i=0; i<num_components; i++)
   {
   cerr << i << "\t" << compTab[i].id
   << "\t" << compTab[i].num << endl;
   cerr << compTab[i].name << endl;
   for (j=0; j<(compTab[i].num)/2; j++)
   cerr << compTab[i].tab[j].ntype
   << "\t" << compTab[i].tab[j].id << "\t";
   cerr << endl;

   }
   */
    /*
    for (i=0; i<maxelem; i++)
      {
         cerr << i << "\t" << elemMap[i] << endl;
      }
    cerr << endl;
   */
}

NeutralFile::~NeutralFile()
{
    delete[] nodeTab;
    delete[] elemTab;
    delete[] compTab;
    delete[] nodeMap;
    delete[] elemMap;
}

void NeutralFile::eval_num_connections()
{
    num_connections = 0;
    for (int i = 0; i < num_elements; i++)
    {
        num_connections += numVert[getShape(i)];
    }
}

void NeutralFile::skip_lines(ifstream &in, int numlines, bool may_be_empty)
{
    char buf[MAXLINE];
    for (int i = 0; i < numlines; i++)
    {
        buf[0] = '\0';
        while (buf[0] == '\0')
        {
            in.getline(buf, MAXLINE);
            if (!in || may_be_empty) // leave this function on eof
                break;
        }
    }
}

Node *NeutralFile::findNodeID(int id)
{
    int i = 0;
    while ((i < num_nodes) && (nodeTab[i].id != id))
        i++;
    if (i < num_nodes)
        return &nodeTab[i];
    else
        return NULL;
}

Element *NeutralFile::findElementID(int id)
{
    int i = 0;
    while ((i < num_elements) && (elemTab[i].id != id))
        i++;
    if (i < num_elements)
        return &elemTab[i];
    else
        return NULL;
}

int NeutralFile::getMaxnode()
{
    return maxnode;
}

int NeutralFile::getMaxelem()
{
    return maxelem;
}

inline SHAPE NeutralFile::getShape(int elemNo)
{
    return (elemTab[elemNo].getShape());
}

void NeutralFile::getTemperatures(float *temp)
{
    int i;
    for (i = 0; i < num_nodes; i++)
    {
        *temp++ = nodeTab[i].temperature;
    }
}

void NeutralFile::getMesh(int *elPtr, int *clPtr, int *tlPtr,
                          float *xPtr, float *yPtr, float *zPtr,
                          int *typPtr)
{
    int i, j, k;
    int n;
    int num;
    int id;
    int elemNo = 0;
    int found;
    int *vertex;
    int **dataPairId;
    size_t size = sizeof(int);
    SHAPE shape;

    // initialize Type-Array
    for (i = 0; i < 3 * num_elements; i++)
        typPtr[i] = -1;

    // allocate and initialize array of IDs for each component
    dataPairId = new int *[num_components];
    for (j = 0; j < num_components; j++)
    {

        num = (compTab[j].num) / 2;
        dataPairId[j] = new int[num];
        for (k = 0; k < num; k++)
        {
            n = compTab[j].tab[k].ntype;
            if (n >= 6 && n <= 12)
                // valid type of IDs
                dataPairId[j][k] = compTab[j].tab[k].id;
            else
                dataPairId[j][k] = -1; // no element IDs
        }
        // sorting arrays in increasing order
        qsort(dataPairId[j], num, size, idcmp);
    }

    for (i = 0; i < num_elements; i++)
    {

        // read type info ( i.d. Element ID, Property ID, Component ID)
        id = elemTab[i].id;
        typPtr[i] = id;
        typPtr[i + num_elements] = elemTab[i].pid;

        // element lying in a component?
        for (j = 0; j < num_components; j++)
        {
            num = (compTab[j].num) / 2;
            found = binsearch(id, dataPairId[j], num);
            if (found != -1)
            {
                typPtr[i + 2 * num_elements] = compTab[j].id;
            }
        }

        /*
              for (j=0; j<num_components; j++){
           for (k=0; k<(compTab[j].num)/2; k++){
                // verify type of each item in this component and ID number
                     n = compTab[j].tab[k].ntype;
                     if( n>= 6 && n<=12 &&  compTab[j].tab[k].id == elemTab[i].id)
                         typPtr[i+2*num_elements] = compTab[j].id;
                }
              }
      */

        shape = getShape(i);
        // set COVISE grid types
        switch (shape)
        {

        case BAR:
            *tlPtr = TYPE_BAR;
            break;
        case TRI:
            *tlPtr = TYPE_TRIANGLE;
            break;
        case QUAD:
            *tlPtr = TYPE_QUAD;
            break;
        case TET:
            *tlPtr = TYPE_TETRAHEDER;
            break;
        case WEDGE:
            *tlPtr = TYPE_PRISM;
            break;
        case HEX:
            *tlPtr = TYPE_HEXAEDER;
            break;
        }
        tlPtr++;

        *elPtr++ = elemNo;

        vertex = elemTab[i].lnodes;

        switch (shape)
        {
        case BAR:
            *clPtr = nodeMap[vertex[0]];
            clPtr++;
            *clPtr = nodeMap[vertex[1]];
            clPtr++;
            elemNo += 2;
            break;

        case TRI:
            *clPtr = nodeMap[vertex[0]];
            clPtr++;
            *clPtr = nodeMap[vertex[1]];
            clPtr++;
            *clPtr = nodeMap[vertex[2]];
            clPtr++;
            elemNo += 3;
            break;

        case QUAD:
            *clPtr = nodeMap[vertex[0]];
            clPtr++;
            *clPtr = nodeMap[vertex[1]];
            clPtr++;
            *clPtr = nodeMap[vertex[2]];
            clPtr++;
            *clPtr = nodeMap[vertex[3]];
            clPtr++;
            elemNo += 4;
            break;

        case TET:
            *clPtr = nodeMap[vertex[0]];
            clPtr++;
            *clPtr = nodeMap[vertex[1]];
            clPtr++;
            *clPtr = nodeMap[vertex[2]];
            clPtr++;
            *clPtr = nodeMap[vertex[3]];
            clPtr++;
            elemNo += 4;
            break;

        case WEDGE:
            *clPtr = nodeMap[vertex[0]];
            clPtr++;
            *clPtr = nodeMap[vertex[1]];
            clPtr++;
            *clPtr = nodeMap[vertex[2]];
            clPtr++;
            *clPtr = nodeMap[vertex[3]];
            clPtr++;
            *clPtr = nodeMap[vertex[4]];
            clPtr++;
            *clPtr = nodeMap[vertex[5]];
            clPtr++;
            elemNo += 6;
            break;

        case HEX:
            *clPtr = nodeMap[vertex[0]];
            clPtr++;
            *clPtr = nodeMap[vertex[1]];
            clPtr++;
            *clPtr = nodeMap[vertex[2]];
            clPtr++;
            *clPtr = nodeMap[vertex[3]];
            clPtr++;
            *clPtr = nodeMap[vertex[4]];
            clPtr++;
            *clPtr = nodeMap[vertex[5]];
            clPtr++;
            *clPtr = nodeMap[vertex[6]];
            clPtr++;
            *clPtr = nodeMap[vertex[7]];
            clPtr++;
            elemNo += 8;
            break;
        }
    }

    for (i = 0; i < num_nodes; i++)
    {
        *xPtr++ = nodeTab[i].coord[0];
        *yPtr++ = nodeTab[i].coord[1];
        *zPtr++ = nodeTab[i].coord[2];
    }

    for (j = 0; j < num_components; j++)
        delete[] dataPairId[j];
    delete[] dataPairId;
}

void Node::set(int identifier)
{
    id = identifier;
}

Element::Element()
{
    // initialize
    pid = 0;
    shape = HEX;
    for (int i = 0; i < 8; i++)
        lnodes[i] = -1; // -1 : flag for unused storage
}

void Element::set(int identifier, SHAPE shp)
{
    id = identifier;
    shape = shp;
}

inline SHAPE Element::getShape()
{
    return shape;
}

Component::Component()
    : tab(NULL)
{
    id = -1;
    num = 0;
}

Component::Component(int i, int n)
{
    id = i;
    num = n;
    tab = new dataPair[num / 2];
    assert(tab != 0);
}

Component::~Component()
{
    delete[] tab;
}

Component &Component::operator=(const Component &comp)
{
    if (this == &comp)
        return *this;

    delete[] tab;
    id = comp.id;
    num = comp.num;
    tab = new dataPair[num / 2];
    assert(tab != 0);
    memcpy(tab, comp.tab, (size_t)(sizeof(dataPair) * num / 2));

    return *this;
}

void Component::set(int i, int n)
{
    id = i;
    num = n;
}

// help functions
int Max(int v1, int v2)
{
    return (v1 >= v2 ? v1 : v2);
}

int idcmp(const void *id1, const void *id2)
{
    // explicit casting
    const int *i1 = (int *)id1;
    const int *i2 = (int *)id2;

    if (*i1 < *i2)
        return -1;
    else if (*i1 > *i2)
        return 1;
    else
        return 0;
}

/* binsearch: find x in v[0] <= v[1] <= ... <= v[n-1] */
int binsearch(int x, int *v, int n)
{
    int low, high, mid;

    low = 0;
    high = n - 1;
    while (low <= high)
    {
        mid = (low + high) / 2;
        if (x < v[mid])
            high = mid - 1;
        else if (x > v[mid])
            low = mid + 1;
        else /* found */
            return mid;
    }

    return -1; /* not found */
}
