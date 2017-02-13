/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TopsanViewer.h"
#include <util/unixcompat.h>

#define TOPSAN_INTRO "The Open Protein Structure Annotation Network (TOPSAN) is a wiki designed to collect, share and distribute information about protein three-dimensional structures, and to advance it towards knowledge about functions and roles of these proteins in their respective organisms. TOPSAN will serve as a portal for the scientific community to learn about protein structures and also to contribute their expertise in annotating protein function.\n\nThe premise of the TOPSAN project is that, no matter how much any individual knows about a particular protein, there are other members of the scientific community who know more about certain aspects of the same protein, and that the collective analysis from experts will be far more informative than any local group, let alone individual, could contribute.\n\nWe believe that, if the members of the biological community are given the opportunity, authorship incentives, and an easy way to contribute their knowledge to the structure annotation, they will do so."

using namespace std;

TopsanViewer::TopsanViewer()
{
    _name = "";
    visible = false;
    topsanHandle = new coPopupHandle(std::string("TOPSAN"));
    topsanFrame = new coFrame();
    topsanPanel = new TabbedDialogPanel(new coFlatPanelGeometry(coUIElement::BLACK));
    topsanPanel->setWidth(600);
    topsanPanel->setTabSize(150, 40);
    //topsanPanel->setFontSize(25);
    topsanHandle->addElement(topsanFrame);
    topsanFrame->addElement(topsanPanel);
    topsanPanel->addTab(TOPSAN_INTRO, "Intro");
    topsanHandle->setVisible(false);
    // topsanPanel->setPopupScale(topsanHandle->getScale());
}

TopsanViewer::~TopsanViewer()
{
}

TopsanViewer::nontopsanData *TopsanViewer::parsePDB(string name)
{
    string pdbpath = coCoviseConfig::getEntry("COVER.Plugin.PDB.PDBDir");
    if (pdbpath.empty())
    {
        pdbpath = "/var/tmp/pdb";
    }

    ifstream infile;
    infile.open((pdbpath + "/" + name + ".pdb").c_str(), ios::in);

    //cerr << "parsing " << pdbpath + "/" + name + ".pdb" << endl;

    if (infile.fail())
    {
        cerr << "Problem opening pdb file: " << pdbpath + name + ".pdb" << endl;
        return NULL;
    }

    TopsanViewer::nontopsanData *ntd = new TopsanViewer::nontopsanData;

    string fline;

    string::size_type pos;

    vector<string> sourcevec;

    while (!infile.eof())
    {
        pos = 0;

        getline(infile, fline);

        pos = fline.find("TITLE", 0);

        if (pos == 0)
        {
            int end = fline.find_last_not_of(" ;");
            if (end != string::npos && end >= 10)
            {
                ntd->title += fline.substr(10, end - 9);
            }
            continue;
        }

        pos = fline.find("SOURCE", 0);

        if (pos == 0)
        {
            int end = fline.find_last_not_of(" ;");
            int start = 10;
            if (fline[start] == ' ')
            {
                start++;
            }
            if (end != string::npos && end >= start)
            {
                string temps = (string("\n    ") + fline.substr(start, (end - start) + 1));
                int test = temps.find("MOL_ID:", 0);
                if (test == 5)
                {
                    continue;
                }
                bool invec = false;
                for (int i = 0; i < sourcevec.size(); i++)
                {
                    if (temps == sourcevec[i])
                    {
                        invec = true;
                        break;
                    }
                }
                if (!invec)
                {
                    sourcevec.push_back(temps);
                    ntd->source += temps;
                }
            }
            continue;
        }

        pos = fline.find("AUTHOR", 0);

        if (pos == 0)
        {
            if (ntd->author != "")
            {
                continue;
            }
            int end = fline.find_last_not_of(" ;");
            int start = 10;
            if (fline[start] == ' ')
            {
                start++;
            }
            if (end != string::npos && end >= start)
            {
                ntd->author += (fline.substr(start, (end - start) + 1));
            }
            continue;
        }

        pos = fline.find("REMARK   2", 0);

        if (pos == 0)
        {
            int end = fline.find_last_not_of(" ;");
            if (end != string::npos && end >= 11)
            {
                ntd->resolution += fline.substr(11, end - 10);
            }
            continue;
        }

        pos = fline.find("ATOM", 0);

        if (pos == 0)
        {
            break;
        }
    }
    infile.close();

    for (int i = 0; i < ntd->author.size(); i++)
    {
        if (ntd->author[i] == ',' && i + 1 < ntd->author.size())
        {
            ntd->author.insert(i + 1, 1, ' ');
        }
    }

    return ntd;
}

void TopsanViewer::load(string name)
{
    //cerr << "TopsanViewer::load >> " << name << endl;

    if (loadedData.find(name) != loadedData.end())
    {
        loadedData[name].second++;
        return;
    }

    if (nottopsanData.find(name) != nottopsanData.end())
    {
        nottopsanData[name].second++;
        return;
    }

    //cerr << "Getting config entry: COVER.Plugin.PDB.TopsanDir" << endl;
    _tsDir = coCoviseConfig::getEntry("COVER.Plugin.PDB.TopsanDir");

    char *tempath = getcwd(NULL, 0);

    chdir(_tsDir.c_str());
    int status;
    if (coVRMSController::instance()->isMaster())
    {
        string cmdInput = getenv("COVISEDIR");
        cmdInput.append("/scripts/pdb/topsan.pl");
        cmdInput.append(" ").append(name);
        system(cmdInput.c_str());
        status = 1;
        coVRMSController::instance()->sendSlaves((char *)&status, sizeof(int));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&status, sizeof(int));
    }

    ifstream infile;
    infile.open((_tsDir + "topsan.dat").c_str(), ios::in);

    chdir(tempath);

    if (infile.bad())
    {
        cerr << "Problem opening file topsan.dat" << endl;
        infile.close();
        return;
    }

    char *buffer = new char[25000];

    infile.getline(buffer, 25000);

    string temp;
    temp = string(buffer);

    if (temp == "IN TOPSAN")
    {
        //cerr << "IN TOPSAN" << endl;
    }
    else
    {
        //cerr << "NOT IN TOPSAN" << endl;
        delete[] buffer;
        infile.close();

        nottopsanData[name] = pair<TopsanViewer::nontopsanData *, int>(parsePDB(name), 1);
        return;
    }

    struct topsanData *data = new struct topsanData;

    if (!infile.eof())
    {
        infile.getline(buffer, 25000);
        data->title = string(buffer) + "\n";
    }
    if (!infile.eof())
    {
        infile.getline(buffer, 25000);
        data->site = string(buffer) + "\n";
    }
    if (!infile.eof())
    {
        infile.getline(buffer, 25000);
        data->id = string(buffer) + "\n";
    }
    if (!infile.eof())
    {
        infile.getline(buffer, 25000);
        data->name = string(buffer) + "\n";
    }
    if (!infile.eof())
    {
        infile.getline(buffer, 25000);
        data->source = string(buffer) + "\n";
    }
    if (!infile.eof())
    {
        infile.getline(buffer, 25000);
        data->refid = string(buffer) + "\n";
    }
    if (!infile.eof())
    {
        infile.getline(buffer, 25000);
        data->weight = string(buffer) + "\n";
    }
    if (!infile.eof())
    {
        infile.getline(buffer, 25000);
        data->residues = string(buffer) + "\n";
    }
    if (!infile.eof())
    {
        infile.getline(buffer, 25000);
        data->isoelec = string(buffer) + "\n";
    }
    if (!infile.eof())
    {
        infile.getline(buffer, 25000);
        data->seq = string(buffer) + "\n";
    }
    if (!infile.eof())
    {
        infile.getline(buffer, 25000);
        data->ligands = string(buffer) + "\n";
    }
    if (!infile.eof())
    {
        infile.getline(buffer, 25000);
        data->metals = string(buffer) + "\n";
    }
    if (!infile.eof())
    {
        infile.getline(buffer, 25000);
        while (string(buffer) != "IMAGE END")
        {
            _imagemap[name].push_back(string(buffer));
            if (!infile.eof())
            {
                infile.getline(buffer, 25000);
            }
            else
            {
                break;
            }
        }
    }
    data->summary = "";
    while (!infile.eof())
    {
        infile.getline(buffer, 25000);
        data->summary += string(buffer) + "\n";
    }

    if (data->summary.size() > 0)
    {
        data->summary = data->summary.substr(0, data->summary.size() - 1);
    }

    loadedData[name] = pair<struct topsanData *, int>(data, 1);

    delete[] buffer;
    infile.close();
}

void TopsanViewer::set(string name)
{
    //cerr << "TopsanViewer::set >> " << name << endl;

    if (visible == false)
    {
        if (topsanPanel->getTabString(0) != TOPSAN_INTRO)
        {
            topsanPanel->removeAll();
            topsanPanel->addTab(TOPSAN_INTRO, "Intro");
            _name = "";
        }
        return;
    }

    if (name == _name)
    {
        return;
    }

    if (loadedData.find(name) == loadedData.end())
    {
        topsanPanel->removeAll();
        //topsanPanel->addTab(string("Protein ") + name + " is not in Topsan." ,"Status");
        topsanPanel->addTab(string("Title: ") + nottopsanData[name].first->title + "\nPDB ID: " + name + "\nSource: " + nottopsanData[name].first->source + "\nAuthors: " + nottopsanData[name].first->author + "\n" + nottopsanData[name].first->resolution, "PDB Info");
        _name = name;
        return;
    }

    topsanPanel->removeAll();

    topsanPanel->addTab(string("Title: ") + loadedData[name].first->title + "Site: " + loadedData[name].first->site + "PDB ID: " + loadedData[name].first->id + "Name: " + loadedData[name].first->name.substr(0, loadedData[name].first->name.size() - 1), "General Information");

    topsanPanel->addTab(string("Source: ") + loadedData[name].first->source + "Reference IDs: " + loadedData[name].first->refid + "Molecular Weight (Da): " + loadedData[name].first->weight + "Residues: " + loadedData[name].first->residues + "Isoelectric Point: " + loadedData[name].first->isoelec + "Sequence: " + loadedData[name].first->seq.substr(0, loadedData[name].first->seq.size() - 1), "Molecular Characteristics");

    topsanPanel->addTab(string("Ligands: ") + loadedData[name].first->ligands + "Metals: " + loadedData[name].first->metals.substr(0, loadedData[name].first->metals.size() - 1), "Ligand Information");

    string remainder;

    topsanPanel->addTab(loadedData[name].first->summary, "Summary", remainder, 250);
    int snum = 2;
    char snumbuf[15];
    while (!remainder.empty())
    {
        sprintf(snumbuf, "%d", snum);
        topsanPanel->addTab(remainder, string("Summary") + string(snumbuf), remainder, 250);
        snum++;
    }

    //topsanPanel->addTab(loadedData[name].first->summary.substr(loadedData[name].first->summary.find_first_not_of('\n'), loadedData[name].first->summary.find_last_not_of('\n')), "Summary");
    char fnum[15];
    for (int i = 0; i < _imagemap[name].size(); i++)
    {
        sprintf(fnum, "%d", i + 1);
        topsanPanel->addTextureTab(_tsDir + "/" + _imagemap[name][i], string("Figure ") + string(fnum));
    }

    _name = name;
}

void TopsanViewer::remove(string name)
{
    if (name == _name || _name == "")
    {
        //cerr << "clearing display for " << name << endl;
        topsanPanel->removeAll();
        topsanPanel->addTab(TOPSAN_INTRO, "Intro");
        _name = "";
    }

    if (loadedData.find(name) != loadedData.end())
    {

        if (loadedData[name].second > 1)
        {
            loadedData[name].second--;
            //cerr << "Dec count for " << name << endl;
        }
        else
        {
            delete loadedData[name].first;
            loadedData.erase(name);
            _imagemap.erase(name);
            //cerr << "Removed " << name << endl;
        }
    }
    else
    {
        if (nottopsanData[name].second > 1)
        {
            nottopsanData[name].second--;
        }
        else
        {
            delete nottopsanData[name].first;
            nottopsanData.erase(name);
        }
    }
}

void TopsanViewer::clear()
{
    //cerr << "clearing topsan display." << endl;
    _name = "";

    topsanPanel->removeAll();
    topsanPanel->addTab(TOPSAN_INTRO, "Intro");

    for (map<std::string, std::pair<struct topsanData *, int> >::iterator it = loadedData.begin(); it != loadedData.end(); it++)
    {
        delete (*it).second.first;
    }

    for (map<std::string, std::pair<struct nontopsanData *, int> >::iterator it = nottopsanData.begin(); it != nottopsanData.end(); it++)
    {
        delete (*it).second.first;
    }

    loadedData.clear();
    nottopsanData.clear();
    _imagemap.clear();
}

void TopsanViewer::setVisible(bool v)
{
    topsanHandle->setVisible(v);
    visible = v;
}
