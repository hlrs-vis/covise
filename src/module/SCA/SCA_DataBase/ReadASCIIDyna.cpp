/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ReadASCIIDyna.h"
#include <appl/ApplInterface.h>
#include <config/CoviseConfig.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
using namespace covise;

const float ReadASCIIDyna::CONVERSION_FACTOR = 1.0;

const char *ReadASCIIDyna::tissueTypes[] = { "TAD_Toipa", "Slush", "DoubleVelvet", "HHTConvent", "HHT", "BSQ_A" };
const float ReadASCIIDyna::tissueThickness[] = { 0.870e-01f, 0.870e-01f, 0.870e-01f, 0.870e-01f, 0.870e-01f, 0.870e-01f };

const char *ReadASCIIDyna::noppenFormChoices[] = { "Diamant", "Ellipse", "Circle" };

const float ReadASCIIDyna::noppenHoeheLimits_[] = { 0.4f, 5.0f, 1.5f };
const float ReadASCIIDyna::ausrundungsRadiusLimits_[] = { 0.0f, 0.75f, 0.2f };
const float ReadASCIIDyna::abnutzungsRadiusLimits_[] = { 0.0f, 0.75f, 0.2f };
const float ReadASCIIDyna::noppenWinkelLimits_[] = { 1.0f, 25.0f, 20.0f };
const float ReadASCIIDyna::laenge1Limits_[] = { 0.4f, 5.0f, 1.5f };
const float ReadASCIIDyna::laenge2Limits_[] = { 0.4f, 5.0f, 1.5f };
const float ReadASCIIDyna::gummiHaerteLimits_[] = // FIXME
    {
      0.1f, 1.0f, 0.5f
    };
const float ReadASCIIDyna::anpressDruckLimits_[] = // FIXME
    {
      0.1f, 1.0f, 0.5f
    };

const float *
ReadASCIIDyna::getMinMax(int level)
{
    switch (level)
    {
    case 0:
        return noppenHoeheLimits_;
        break;
    case 1:
        return ausrundungsRadiusLimits_;
        break;
    case 2:
        return abnutzungsRadiusLimits_;
        break;
    case 3:
        return noppenWinkelLimits_;
        break;
    case 5:
        return laenge1Limits_;
        break;
    case 6:
        return laenge2Limits_;
        break;
    case 8:
        return gummiHaerteLimits_;
        break;
    case 9:
        return anpressDruckLimits_;
        break;
    }
    return NULL;
}

//#ifdef __sgi
//   #include <sys/dir.h>
//#else
#include <dirent.h>
//#endif

int
ReadASCIIDyna::cleanFiles(const vector<string> &spurs)
{
    DIR *dirp;
    //#ifdef __sgi
    //   struct direct *entry;
    //#else
    struct dirent *entry;
    //#endif
    dirp = opendir(".");
    if (dirp == NULL)
    {
        cerr << "ReadASCIIDyna::cleanFiles: could not open actual directory" << endl;
    }
    vector<string *> toBeCleaned;
    while ((entry = readdir(dirp)) != NULL)
    {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
        {
            continue;
        }
        bool loesch = true;
        int spurCount;
        for (spurCount = 0; spurCount < spurs.size(); ++spurCount)
        {
            if (strstr(entry->d_name, spurs[spurCount].c_str()) != NULL)
            {
                loesch = false;
                break;
            }
        }
        if (loesch)
        {
            toBeCleaned.push_back(new string(entry->d_name));
        }
    }
    int i;
    for (i = 0; i < toBeCleaned.size(); ++i)
    {
        cerr << "Removing " << toBeCleaned[i]->c_str() << endl;
        unlink(toBeCleaned[i]->c_str());
        delete toBeCleaned[i];
    }
    return 0;
}

int
ReadASCIIDyna::readEmbossingResults(ifstream &emb_conn, ifstream &emb_displ,
                                    ifstream &emb_thick,
                                    std::vector<int> &epl, std::vector<int> &cpl,
                                    std::vector<float> &exc,
                                    std::vector<float> &eyc,
                                    std::vector<float> &ezc, std::vector<float> &dicke)
{
    std::vector<float> pxc;
    std::vector<float> pyc;
    std::vector<float> pzc;

    std::vector<float> pdicke;
    float one_thickness;
    while (emb_thick >> one_thickness)
    {
        pdicke.push_back(one_thickness);
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
        Covise::sendWarning("Could not found element shell section");
        return -1;
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
    if (MarkCoords(cpl, mark) != 0)
    {
        return -1;
    }
    cerr << mark.size() << ' ' << pdicke.size() << endl;
    if (mark.size() != pdicke.size())
    {
        Covise::sendWarning("Mesh data is not compatible with thickness file");
        return -1;
    }
    int node;
    for (node = 0; node < mark.size(); ++node)
    {
        if (mark[node] > 0)
        {
            dicke.push_back(pdicke[node]);
            if (dicke[dicke.size() - 1] <= 0.0)
            {
                Covise::sendWarning("Thickness is not overall positive");
                return -1;
            }
        }
    }
    // read displacements
    if (readDisplacements(emb_displ, mark, exc, eyc, ezc) != 0)
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

int
ReadASCIIDyna::MarkCoords(std::vector<int> &cpl, std::vector<int> &mark)
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
    return 0;
}

int
ReadASCIIDyna::readDisplacements(ifstream &emb_displ,
                                 std::vector<int> &mark,
                                 std::vector<float> &exc,
                                 std::vector<float> &eyc,
                                 std::vector<float> &ezc)
{
    // jump over first reasults
    char buf[1024];
    int count = 0;
    while (emb_displ.getline(buf, 1024))
    {
        ++count;
    }
    if (count % (1 + mark.size()) != 0)
    {
        Covise::sendWarning("The displacements file has a wrong number of lines");
        return -1;
    }
    emb_displ.clear();
    emb_displ.seekg(0, ios::beg); // rewind
    int recount;
    for (recount = 0; recount < count - mark.size(); ++recount)
    {
        emb_displ.getline(buf, 1024);
    }
    // now read displacements of marked nodes
    int node;
    for (node = 0; node < mark.size(); ++node)
    {
        emb_displ.getline(buf, 1024);
        if (mark[node] == 0)
        {
            continue;
        }
        // check node number
        char number[16];
        strncpy(number, buf, 8);
        number[8] = '\0';
        int checkNode;
        sscanf(number, "%d", &checkNode);
        if (checkNode != node + 1)
        {
            Covise::sendWarning("Node numbers in displacements file are not correct");
            return -1;
        }
        // read X position
        strncpy(number, buf + 8, 8);
        number[8] = '\0';
        exc.push_back(CrazyFormat(number));
        // read Y position
        strncpy(number, buf + 8 + 8, 8);
        number[8] = '\0';
        eyc.push_back(CrazyFormat(number));
        // read Z position
        strncpy(number, buf + 8 + 8 + 8, 8);
        number[8] = '\0';
        ezc.push_back(CrazyFormat(number));
    }
    return 0;
}

int
ReadASCIIDyna::AddDisplacements(std::vector<float> &exc, std::vector<float> &eyc, std::vector<float> &ezc,
                                std::vector<float> &pxc, std::vector<float> &pyc, std::vector<float> &pzc,
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

float
ReadASCIIDyna::CrazyFormat(char number[8])
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
ReadASCIIDyna::readIntSlider(istringstream &strText, int *addr, int maxLen)
{
    std::vector<char> Value(maxLen);
    strText >> &Value[0];
    strText >> &Value[0];
    if (!(strText >> *addr))
    {
        Covise::sendWarning("Error on reading IntSlider from istringstream");
        return -1;
    }
    return 0;
}

int
ReadASCIIDyna::readFloatSlider(istringstream &strText, float *addr, int maxLen)
{
    std::vector<char> Value(maxLen);
    strText >> &Value[0];
    strText >> &Value[0];
    if (!(strText >> *addr))
    {
        Covise::sendWarning("Error on reading FloatSlider from istringstream");
        return -1;
    }
    return 0;
}

int
ReadASCIIDyna::readBoolean(istringstream &strText, int *addr, int maxLen)
{
    std::vector<char> Value(maxLen);
    /*
      strText >> &Value[0];
      strText >> &Value[0];
   */

    if (!(strText >> &Value[0]))
    {
        Covise::sendWarning("Error on reading FloatSlider from istringstream");
        return -1;
    }
    if (strcmp(&Value[0], "FALSE") == 0)
    {
        *addr = 0;
    }
    else
    {
        *addr = 1;
    }
    return 0;
}

int
ReadASCIIDyna::readChoice(istringstream &strText, int *addr, int maxLen)
{
    std::vector<char> Value(maxLen);
    strText >> &Value[0];
    if (!(strText >> *addr))
    {
        Covise::sendWarning("Error on reading IntSlider from istringstream");
        return -1;
    }
    strText >> &Value[0];
    return 0;
}

void
ReadASCIIDyna::Mirror2(vector<float> &exc, vector<float> &eyc, vector<float> &ezc,
                       vector<int> &ecl, vector<int> &epl, Direction d)
{
    int numpoints = exc.size();
    int point;
    for (point = 0; point < numpoints; ++point)
    {
        float tmpx = exc[point];
        float tmpy = eyc[point];
        float tmpz = ezc[point];
        exc.push_back(tmpx);
        eyc.push_back(tmpy);
        ezc.push_back(tmpz);
    }
    if (d == X)
    {
        for (point = 0; point < numpoints; ++point)
        {
            exc[numpoints + point] *= -1.0;
        }
    }
    else
    {
        for (point = 0; point < numpoints; ++point)
        {
            eyc[numpoints + point] *= -1.0;
        }
    }
    int vertex;
    int numvertices = ecl.size();
    // eeeeeps, there is a problem with the orientation!!!!
    for (vertex = 0; vertex < numvertices; ++vertex)
    {
        int base = vertex % 2;
        int shift = 1 - 2 * base;
        ecl.push_back(ecl[vertex + shift] + numpoints);
    }
    int count, element;
    int numelems = epl.size();
    for (element = 0, count = numvertices; element < numelems; ++element, count += 4)
    {
        epl.push_back(count);
    }
}

void
ReadASCIIDyna::loadFileNames(string &conn, string &disp, string &thick)
{
    conn += "topology.k";
    disp += "defgeo";
    thick += "movie100.s30";
}

#include <string.h>

void
ReadASCIIDyna::SCA_Calls(const char *var, vector<string> &arguments)
{
    arguments.clear();
    string scope_var = "SCA_Calls.";
    scope_var += var;
    std::string entry = coCoviseConfig::getEntry(scope_var);
    if (!entry.empty())
    {
        char *buf = new char[entry.length() + 1];
        strcpy(buf, entry.c_str());
        char *base = buf;
        // substitute nils for spaces
        vector<const char *> words;
        bool WordBegins = true;
        while (*base != '\0')
        {
            int zeichen = *base;
            if (isspace(zeichen))
            {
                *base = '\0';
                WordBegins = true;
            }
            else if (WordBegins)
            {
                words.push_back(base);
                WordBegins = false;
            }
            ++base;
        }
        int word;
        for (word = 0; word < words.size(); ++word)
        {
            arguments.push_back(string(words[word]));
        }
        delete[] buf;
    }
}
