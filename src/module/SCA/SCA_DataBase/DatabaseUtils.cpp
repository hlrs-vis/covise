/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "DatabaseUtils.h"
#include <appl/ApplInterface.h>

using namespace covise;

int
DatabaseUtils::AddDisplacements(vector<float> &exc, vector<float> &eyc, vector<float> &ezc,
                                vector<float> &pxc, vector<float> &pyc, vector<float> &pzc,
                                std::vector<int> &mark)
{
    int node, count = 0;
    for (node = 0; node < mark.size(); ++node)
    {
        if (mark[node] != 0)
        {
            ++count;
        }
    }
    if (count != exc.size())
    {
        Covise::sendWarning("Displacement arrays and mark array have an incompatible state");
        return -1;
    }
    for (node = 0, count = 0; node < mark.size(); ++node)
    {
        if (mark[node] != 0)
        {
            exc[count] += pxc[node];
            eyc[count] += pyc[node];
            ezc[count] += pzc[node];
            ++count;
        }
    }
    float z_level = 0.0;
    float xy_rand = -FLT_MAX;
    int punkt;
    for (punkt = 0; punkt < ezc.size(); ++punkt)
    {
        float x = exc[punkt];
        float y = eyc[punkt];
        float z = ezc[punkt];
        if (x + y > xy_rand)
        {
            xy_rand = x + y;
            z_level = z;
        }
    }
    // shift z
    for (punkt = 0; punkt < ezc.size(); ++punkt)
    {
        ezc[punkt] -= z_level;
    }
    return 0;
}

int
DatabaseUtils::readEmbossingResults(ifstream &emb_conn, ifstream &emb_displ,
                                    ifstream &emb_displ_exp,
                                    ifstream &emb_thick,
                                    vector<int> &epl, vector<int> &cpl,
                                    vector<float> &exc,
                                    vector<float> &eyc,
                                    vector<float> &ezc, vector<float> &dicke)
{
    vector<float> pxc;
    vector<float> pyc;
    vector<float> pzc;

    // std::vector<float> pdicke;
    float one_thickness;
    while (emb_thick >> one_thickness)
    {
        if (one_thickness == 0.0)
        {
            continue;
        }
        dicke.push_back(one_thickness);
    }

    char buffer[1024];
    bool found = false;
    while (emb_conn >> buffer)
    {
        if (strcmp(buffer, "*NODE") == 0)
        {
            found = true;
            break;
        }
    }
    if (!found)
    {
        Covise::sendWarning("Could not found node coordinate section");
        return -1;
    }
    char retChar;
    emb_conn.get(retChar);
    if (retChar != '\r')
    {
        if (retChar != '\n')
        {
            emb_conn.putback(retChar);
        }
    }
    else
    {
        emb_conn.get(retChar); // read new line
    }
    while (emb_conn.getline(buffer, 1024))
    {
        int node;
        if (sscanf(buffer, "%d", &node) != 1)
        {
            // we are done with the node definition
            break;
        }
        char numbers[64];
        strncpy(numbers, buffer + 8, 16);
        numbers[16] = ' ';
        strncpy(numbers + 16 + 1, buffer + 8 + 16, 16);
        numbers[16 + 1 + 16] = ' ';
        strncpy(numbers + 16 + 1 + 16 + 1, buffer + 8 + 16 + 16, 16);
        numbers[16 + 1 + 16 + 1 + 16] = '\0';
        istringstream floats(numbers);
        float x, y, z;
        floats.setf(floats.flags() | ios::uppercase);
        if (!(floats >> x >> y >> z))
        {
            Covise::sendWarning("Could not read coordinates for a node");
            return -1;
        }
        pxc.push_back(x);
        pyc.push_back(y);
        pzc.push_back(z);
    }

    // now follows the connectivity
    if (strncmp(buffer, "*ELEMENT_SHELL", 14) != 0)
    {
        found = false;
        while (emb_conn >> buffer)
        {
            if (strcmp(buffer, "*ELEMENT_SHELL") == 0)
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            Covise::sendWarning("Could not find element shell section");
            return -1;
        }
    }
    emb_conn.get(retChar);
    if (retChar != '\r')
    {
        if (retChar != '\n')
        {
            emb_conn.putback(retChar);
        }
    }
    else
    {
        emb_conn.get(retChar); // read new line
    }
    while (emb_conn.getline(buffer, 1024))
    {
        int element, material;
        int n0, n1, n2, n3;
        if (sscanf(buffer, "%d %d %d %d %d %d", &element, &material,
                   &n0, &n1, &n2, &n3) != 6)
        {
            // we are done with the shell section
            break;
        }
        if (material != 2)
        {
            continue; // this is not paper
        }
        epl.push_back(cpl.size());
        cpl.push_back(n0 - 1);
        cpl.push_back(n1 - 1);
        cpl.push_back(n2 - 1);
        cpl.push_back(n3 - 1);
    }
    // eliminate unused nodes
    std::vector<int> mark;
    mark.resize(pxc.size());
    int paperNodes = MarkCoords(cpl, mark);
    if (paperNodes != dicke.size())
    {
        Covise::sendWarning("Mesh data is not compatible with thickness file");
        return -1;
    }

    // read displacements
    if (readDisplacements(emb_displ,
                          mark, exc, eyc, ezc, true, dicke.size()) != 0)
    {
        return -1;
    }
    if (readDisplacements(emb_displ_exp, mark, exc, eyc, ezc, false, -1) != 0)
    {
        return -1;
    }
    // add displacements and positions
    if (AddDisplacements(exc, eyc, ezc, pxc, pyc, pzc, mark) != 0)
    {
        return -1;
    }

    return 0;
}

// mark has already the correct size
int
DatabaseUtils::MarkCoords(vector<int> &cpl, std::vector<int> &mark)
{
    int numini = mark.size();
    memset(&mark[0], 0, numini * sizeof(int));
    int vert;
    for (vert = 0; vert < cpl.size(); ++vert)
    {
        if (cpl[vert] >= numini)
        {
            Covise::sendWarning("MarkCoords: connectivity refers non-existing nodes");
            return -1; // error
        }
        if (cpl[vert] < 0)
        {
            Covise::sendWarning("MarkCoords: connectivity refers node 0 or negative");
            return -1; // error
        }
        ++mark[cpl[vert]];
    }
    std::vector<int> compressedPoints;
    int point, count = 0;
    for (point = 0; point < numini; ++point)
    {
        if (mark[point] != 0)
        {
            compressedPoints.push_back(count);
            ++count;
        }
        else
        {
            compressedPoints.push_back(-1);
        }
    }
    for (vert = 0; vert < cpl.size(); ++vert)
    {
        cpl[vert] = compressedPoints[cpl[vert]];
        if (cpl[vert] < 0)
        {
            Covise::sendWarning("MarkCoords: This is a bug");
            return -1;
        }
    }
    return count;
}

inline float
CrazyFormat(char number[8])
{
    char normal[16];
    int pos;
    normal[0] = number[0];
    for (pos = 1; pos < 6; ++pos)
    {
        normal[pos] = (number[pos] == ' ') ? '0' : number[pos];
    }
    if (number[6] == '-' || number[6] == '+' || number[6] == ' ')
    {
        normal[6] = 'e';
        normal[7] = number[6];
        if (normal[7] == ' ')
        {
            normal[7] = ' ';
        }
        normal[8] = number[7];
        if (normal[8] == ' ')
        {
            normal[6] = '\0';
        }
    }
    else
    {
        normal[6] = '\0';
    }
    float ret;
    sscanf(normal, "%f", &ret);
    return ret;
}

int
DatabaseUtils::readDisplacements(ifstream &emb_displ,
                                 std::vector<int> &mark,
                                 vector<float> &exc,
                                 vector<float> &eyc,
                                 vector<float> &ezc,
                                 bool newFormat,
                                 int dickeSize)
{
    // jump over first reasults
    char buf[1024];
    int count = 0;
    while (emb_displ.getline(buf, 1024))
    {
        ++count;
    }
    if (!newFormat && count % (1 + mark.size()) != 0)
    {
        cerr << "Explicit " << count << ' ' << mark.size() << endl;
    }
    if (newFormat && count % (1 + dickeSize) != 0)
    {
        cerr << "Implicit " << count << ' ' << 1 + dickeSize << endl;
    }
    if ((!newFormat && count % (1 + mark.size()) != 0)
        || (newFormat && count % (1 + dickeSize) != 0))
    {
        Covise::sendWarning("The displacements file has a wrong number of lines");
        return -1;
    }
    emb_displ.clear();
    emb_displ.seekg(0, ios::beg); // rewind
    int recount;
    int lastLines;
    if (newFormat)
    {
        lastLines = dickeSize;
    }
    else
    {
        lastLines = mark.size();
    }
    for (recount = 0; recount < count - lastLines; ++recount)
    {
        emb_displ.getline(buf, 1024);
    }
    // now read displacements of marked nodes
    int node;
    for (node = 0, count = 0; node < lastLines; ++node)
    {
        emb_displ.getline(buf, 1024);
        if (!newFormat && mark[node] == 0)
        {
            continue;
        }
        // check node number
        char number[16];
        strncpy(number, buf, 8);
        number[8] = '\0';
        int checkNode;
        sscanf(number, "%d", &checkNode);
        if (!newFormat)
        {
            if (checkNode != node + 1)
            {
                Covise::sendWarning("Node numbers in displacements file are not correct");
                return -1;
            }
        }
        // read X position
        strncpy(number, buf + 8, 8);
        number[8] = '\0';
        if (newFormat)
        {
            exc.push_back(CrazyFormat(number));
        }
        else
        {
            exc[count] += CrazyFormat(number);
        }
        // read Y position
        strncpy(number, buf + 8 + 8, 8);
        number[8] = '\0';
        if (newFormat)
        {
            eyc.push_back(CrazyFormat(number));
        }
        else
        {
            eyc[count] += CrazyFormat(number);
        }
        // read Z position
        strncpy(number, buf + 8 + 8 + 8, 8);
        number[8] = '\0';
        if (newFormat)
        {
            ezc.push_back(CrazyFormat(number));
        }
        else
        {
            ezc[count] += CrazyFormat(number);
        }
        ++count;
    }
    return 0;
}
