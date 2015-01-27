/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <vector>
#include <string>
#include <algorithm>

static const int HOWFAR = 8;
static const int SUBSETDIFF = 5;

class CSequence
{
public:
    int num;
    std::string aminoacid;
    float x;
    float y;
    float z;
    void SetData(std::string, int);
    void SetData(std::string, int, float, float, float);
};

class CChain
{
public:
    std::string name;
    int num;
    std::vector<CSequence> chainsequence;
    void SetData(std::string, int);
};

//  This class handles retrieval and printing functions.

class CChainMatrix
{
public:
    float x;
    float y;
    float z;
    float distance;
    std::string chain;
    int seqno;
};

class CProtein
{
public:
    void PrintChain(std::vector<CChain> userChain);
    void PrintChain(CChain userChain);
    void PrintMatrix(std::vector<CChainMatrix> userMatrix);
    int RetrievePositions(std::vector<CChain> &userChain, std::string strFileName);
    int RetrieveSubset(CChain &userChain, std::vector<CChain> &tempChain, std::string strChain, int startPos, int endPos);
    int RetrieveSubset(CChain &userChain, std::string strFileName, std::string strChain, int startPos, int endPos);
    int ClosestAminoAcid(std::vector<CChain> &userChain, float xpos, float ypos, float zpos, std::string &smallestChain, int &smallestChainPos);
    int ReturnChainNumber(std::vector<CChain> &userChain, std::string strChainName);
};
