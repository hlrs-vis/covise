/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SequenceViewer.h"
#include <util/unixcompat.h>

using namespace std;
using namespace osg;

SequenceViewer::SequenceViewer(coVRPlugin *p)
{
    _name = "";
    pwidth = 908;
    handle = new coPopupHandle(std::string("SequenceViewer"));
    frame = new coFrame();
    panel = new TabbedDialogPanel(new coFlatPanelGeometry(coUIElement::BLACK));
    panel->setFontSize(1);
    panel->setWidth(pwidth);
    panel->setTabSize(150, 40);
    handle->addElement(frame);
    frame->addElement(panel);
    handle->setVisible(false);
    bwidth = 50;
    bheight = 40;

    visiblechain = -1;
    panel->addTab(" ", "Chains");
    offset = -5;
    space = 3;
    selectedc = selecteda = -1;
    PDBptr = p;
}

SequenceViewer::~SequenceViewer()
{
    clear();
}

void SequenceViewer::buttonEvent(coButton *b)
{
    //cerr << "Sview: Getting button event." << endl;
    for (int i = 0; i < cbuttons[_name].size(); i++)
    {
        if (cbuttons[_name][i] == b)
        {
            cbuttons[_name][i]->setState(true, false);
            setChain(i);
            return;
        }
    }

    if (visiblechain == -1)
    {
        return;
    }

    for (int i = 0; i < buttons[_name][visiblechain].size(); i++)
    {
        if (buttons[_name][visiblechain][i] == b)
        {
            if (selectedc != -1 && selecteda != -1)
            {
                buttons[_name][selectedc][selecteda]->setState(false, false);
            }
            buttons[_name][visiblechain][i]->setState(true, false);
            selectedc = visiblechain;
            selecteda = i;
            struct SequenceMessage sm;
            sm.x = locations[_name][visiblechain][i].first.x;
            sm.y = locations[_name][visiblechain][i].first.y;
            sm.z = locations[_name][visiblechain][i].first.z;
            sm.on = true;
            cerr << "Sending: x: " << sm.x << " y: " << sm.y << " z: " << sm.z << endl;
            strcpy(sm.filename, _name.c_str());
            cover->sendMessage(PDBptr, coVRPluginSupport::TO_SAME,
                               MOVE_MARK, sizeof(struct SequenceMessage), &sm);
        }
    }
}

void SequenceViewer::menuUpdate(coButton *b)
{
    for (int i = 0; i < buttons[_name][visiblechain].size(); i++)
    {
        if (buttons[_name][visiblechain][i] == b)
        {
            if (selectedc != -1 && selecteda != -1)
            {
                buttons[_name][selectedc][selecteda]->setState(false, false);
            }
            buttons[_name][visiblechain][i]->setState(true, false);
            selectedc = visiblechain;
            selecteda = i;
        }
    }
}

void SequenceViewer::load(std::string name)
{
    //cerr << "SView: load called with " << name << endl;
    if (pcount.find(name) != pcount.end())
    {
        //cerr << "loading again.\n";
        pcount[name]++;
        return;
    }

    pdbpath = coCoviseConfig::getEntry("COVER.Plugin.PDB.PDBDir");
    if (pdbpath.empty())
        pdbpath = "/var/tmp/pdb";

    char *tempath = getcwd(NULL, 0);

    chdir(pdbpath.c_str());

    ifstream infile;
    infile.open((pdbpath + "/" + name + ".pdb").c_str(), ios::in);

    chdir(tempath);

    if (infile.fail())
    {
        cerr << "SequenceViewer: Problem opening file pdb file " << (pdbpath + "/" + name + ".pdb") << endl;
        infile.close();
        return;
    }

    std::vector<std::vector<std::pair<struct markpoint, std::string> > > mol;

    string fline;

    Matrix scale, ctor;
    Matrix Cctor, Cscale; //column-major transforms
    int mrow = 0;
    string::size_type pos = 0;
    string chainid = "", acid = "";
    int acidnum;
    int currentchain = 0, currentacid = -1;
    //char cid[10], aid[10];
    float acount = 0;

    while (!infile.eof())
    {
        pos = 0;
        getline(infile, fline);

        pos = fline.find("CRYST1", 0);
        if (pos == 0)
        {
            float Dim[3];
            float Angle[3];

            sscanf(fline.c_str(), "%*s %f %f %f %f %f %f", Dim, Dim + 1, Dim + 2, Angle, Angle + 1, Angle + 2);
            //cerr << "Reading Crystal:\nDim: " << Dim[0] << " " << Dim[1] << " " << Dim[2] << endl;
            //cerr << "Angle: " << Angle[0] << " " << Angle[1] << " " << Angle[2] << endl;

            float cabg[3];
            float sabg[3];
            float cabgs[3];
            float sabgs1;
            int i;

            for (i = 0; i < 3; i++)
            {
                cabg[i] = (float)cos(Angle[i] * PI / 180.0);
                sabg[i] = (float)sin(Angle[i] * PI / 180.0);
            }

            cabgs[0] = (cabg[1] * cabg[2] - cabg[0]) / (sabg[1] * sabg[2]);
            cabgs[1] = (cabg[2] * cabg[0] - cabg[1]) / (sabg[2] * sabg[0]);
            cabgs[2] = (cabg[0] * cabg[1] - cabg[2]) / (sabg[0] * sabg[1]);

            float UnitCellVolume = (float)(Dim[0] * Dim[1] * Dim[2] * sqrt(1.0 + (double)2.0 * cabg[0] * cabg[1] * cabg[2] - (double)(cabg[0] * cabg[0] + (double)cabg[1] * cabg[1] + (double)cabg[2] * cabg[2])));

            float RecipDim[3];

            RecipDim[0] = Dim[1] * Dim[2] * sabg[0] / UnitCellVolume;
            RecipDim[1] = Dim[0] * Dim[2] * sabg[1] / UnitCellVolume;
            RecipDim[2] = Dim[0] * Dim[1] * sabg[2] / UnitCellVolume;

            sabgs1 = (float)sqrt(1.0 - cabgs[0] * cabgs[0]);

            ctor(0, 0) = Dim[0];
            ctor(0, 1) = cabg[2] * Dim[1];
            ctor(0, 2) = cabg[1] * Dim[2];
            ctor(1, 1) = sabg[2] * Dim[1];
            ctor(1, 2) = -sabg[1] * cabgs[0] * Dim[2];
            ctor(2, 2) = sabg[1] * sabgs1 * Dim[2];
            Cctor(0, 0) = Dim[0];
            Cctor(1, 0) = cabg[2] * Dim[1];
            Cctor(2, 0) = cabg[1] * Dim[2];
            Cctor(1, 1) = sabg[2] * Dim[1];
            Cctor(2, 1) = -sabg[1] * cabgs[0] * Dim[2];
            Cctor(2, 2) = sabg[1] * sabgs1 * Dim[2];

            continue;

            /*cerr << "Matrix cm\n";

                    for(int k = 0; k < 4; k++)
                    {
                        
                        for(int j = 0; j < 4; j++)
                        {
                            cerr << cm(k,j) << " ";
                        }
                        cerr << endl;
                    }*/
        }

        pos = fline.find("SCALE", 0);
        if (pos == 0)
        {
            float mx, my, mz, mo;
            sscanf(fline.c_str(), "%*s %f %f %f %f", &mx, &my, &mz, &mo);
            scale(mrow, 0) = mx;
            scale(mrow, 1) = my;
            scale(mrow, 2) = mz;
            scale(mrow, 3) = mo;
            Cscale(0, mrow) = mx;
            Cscale(1, mrow) = my;
            Cscale(2, mrow) = mz;
            Cscale(3, mrow) = mo;
            mrow++;
            continue;
        }

        pos = fline.find("TER", 0);
        if (pos == 0)
        {
            struct markpoint mp = mol[currentchain][currentacid].first;
            mp.x = mp.x / acount;
            mp.y = mp.y / acount;
            mp.z = mp.z / acount;
            Vec4 newval = scale * Vec4(mp.x, mp.y, mp.z, 1.0);
            newval = ctor * newval;
            mp.x = newval.x();
            mp.y = newval.y();
            mp.z = newval.z();
            mol[currentchain][currentacid].first = mp;
            mol[currentchain][currentacid].second = acid;
            currentchain++;
            currentacid = -1;
            acount = 0;
        }

        pos = fline.find("ATOM", 0);
        if (pos == 0)
        {
            float ax, ay, az;
            int num;
            //sscanf(fline.c_str(),"%*s %*s %*s %s %s %d %f %f %f", aid, cid, &num, &ax, &ay, &az);
            string temp = fline.substr(22, 4);
            num = atoi(temp.c_str());
            temp = fline.substr(30, 8);
            ax = atof(temp.c_str());
            temp = fline.substr(38, 8);
            ay = atof(temp.c_str());
            temp = fline.substr(46, 8);
            az = atof(temp.c_str());
            temp = fline.substr(17, 3);
            if (temp.find(" ", 0) == 0)
            {
                temp = temp.substr(1, temp.size() - 1);
            }

            if (currentacid == -1)
            {
                mol.push_back(std::vector<std::pair<struct markpoint, std::string> >());
                mol[currentchain].push_back(std::pair<struct markpoint, std::string>());
                currentacid = 0;
                acidnum = num;
            }

            if (num != acidnum)
            {
                struct markpoint mp = mol[currentchain][currentacid].first;
                mp.x = mp.x / acount;
                mp.y = mp.y / acount;
                mp.z = mp.z / acount;
                Vec4 newval = scale * Vec4(mp.x, mp.y, mp.z, 1.0);
                newval = ctor * newval;
                mp.x = newval.x();
                mp.y = newval.y();
                mp.z = newval.z();
                mol[currentchain][currentacid].first = mp;
                mol[currentchain][currentacid].second = acid;
                acount = 0;
                mol[currentchain].push_back(std::pair<struct markpoint, std::string>());
                currentacid++;
                acidnum = num;
            }

            acid = temp;
            mol[currentchain][currentacid].first.x += ax;
            mol[currentchain][currentacid].first.y += ay;
            mol[currentchain][currentacid].first.z += az;
            acount++;
        }
    }

    inversemap[name] = pair<Matrix, Matrix>(Matrix::inverse(Cscale), Matrix::inverse(Cctor));

    locations[name] = mol;
    for (int i = 0; i < mol.size(); i++)
    {
        coTextButtonGeometry *tb = new coTextButtonGeometry(bwidth, bheight, string(1, (char)('A' + i)));
        tb->setColors(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        cbuttons[name].push_back(new coToggleButton(tb, this));
        buttons[name].push_back(std::vector<coToggleButton *>());
        for (int j = 0; j < mol[i].size(); j++)
        {
            coTextButtonGeometry *tbg = new coTextButtonGeometry(bwidth, bheight, mol[i][j].second);
            tbg->setColors(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0);
            buttons[name][i].push_back(new coToggleButton(tbg, this));
        }
    }
    pcount[name]++;
}

pair<Matrix, Matrix> &SequenceViewer::getInverse(string name)
{
    return inversemap[name];
}

void SequenceViewer::set(std::string name)
{
    if (name == _name)
    {
        return;
    }

    if (_name != "")
    {
        if (selectedc != -1 && selecteda != -1)
        {
            buttons[_name][selectedc][selecteda]->setState(false, false);
        }

        for (int i = 0; i < cbuttons[_name].size(); i++)
        {
            cbuttons[_name][i]->setState(false, false);
            panel->removeElement(cbuttons[_name][i]);
        }
        if (visiblechain != -1)
        {
            for (int i = 0; i < buttons[_name][visiblechain].size(); i++)
            {
                panel->removeElement(buttons[_name][visiblechain][i]);
            }
        }
    }

    int buttonsPerRow = (int)(pwidth / bwidth);
    int rows = cbuttons[name].size() / buttonsPerRow;
    if (cbuttons[name].size() % buttonsPerRow != 0)
    {
        rows++;
    }

    offset = -((rows - 1) * bheight) - 5;

    for (int i = 0; i < cbuttons[name].size(); i++)
    {
        panel->addElement(cbuttons[name][i]);
        cbuttons[name][i]->setPos((i % buttonsPerRow) * bwidth, (((rows - (i / buttonsPerRow)) - 2) * bheight) + offset, 0);
    }

    panel->resize();
    _name = name;
    visiblechain = -1;
    selectedc = selecteda = -1;
}

void SequenceViewer::setLoc(float x, float y, float z)
{
    float dist = -1;
    int savei = -1, savej = -1;
    for (int i = 0; i < locations[_name].size(); i++)
    {
        for (int j = 0; j < locations[_name][i].size(); j++)
        {
            float temp = sqrt(pow(x - locations[_name][i][j].first.x, 2) + pow(y - locations[_name][i][j].first.y, 2) + pow(z - locations[_name][i][j].first.z, 2));
            if (dist == -1 || temp < dist)
            {
                dist = temp;
                savei = i;
                savej = j;
            }
        }
    }
    if (savei == -1 || savej == -1)
    {
        return;
    }
    if (visiblechain != savei)
    {
        setChain(savei);
    }
    menuUpdate(buttons[_name][savei][savej]);
}

void SequenceViewer::remove(std::string name)
{
    /*if(name == _name)
	{
		clearDisplay();
	}*/

    if (pcount[name] > 1)
    {
        pcount[name]--;
        return;
    }

    locations.erase(name);
    pcount.erase(name);
    for (int i = 0; i < buttons[name].size(); i++)
    {
        for (int j = 0; j < buttons[name][i].size(); j++)
        {
            delete buttons[name][i][j];
        }
    }
    buttons.erase(name);

    for (int i = 0; i < cbuttons[name].size(); i++)
    {
        delete cbuttons[name][i];
    }
    cbuttons.erase(name);
}

void SequenceViewer::clear()
{
    clearDisplay();
    pcount.clear();
    locations.clear();
    for (std::map<std::string, std::vector<coToggleButton *> >::iterator it = cbuttons.begin(); it != cbuttons.end(); it++)
    {
        for (int i = 0; i < it->second.size(); i++)
        {
            delete it->second[i];
        }
    }
    cbuttons.clear();

    for (std::map<std::string, std::vector<std::vector<coToggleButton *> > >::iterator it = buttons.begin(); it != buttons.end(); it++)
    {
        for (int i = 0; i < it->second.size(); i++)
        {
            for (int j = 0; j < it->second[i].size(); j++)
            {
                delete it->second[i][j];
            }
        }
    }
    buttons.clear();
}

void SequenceViewer::clearDisplay()
{
    if (_name == "")
    {
        return;
    }

    if (selectedc != -1 && selecteda != -1)
    {
        buttons[_name][selectedc][selecteda]->setState(false, false);
    }

    for (int i = 0; i < cbuttons[_name].size(); i++)
    {
        cbuttons[_name][i]->setState(false, false);
        panel->removeElement(cbuttons[_name][i]);
    }
    if (visiblechain != -1)
    {
        for (int i = 0; i < buttons[_name][visiblechain].size(); i++)
        {
            panel->removeElement(buttons[_name][visiblechain][i]);
        }
    }
    _name = "";
    panel->resize();
    visiblechain = selectedc = selecteda = -1;
}

void SequenceViewer::setVisible(bool vis)
{
    handle->setVisible(vis);
    handle->update();
}

void SequenceViewer::setChain(int chain)
{
    if (chain == visiblechain)
    {
        return;
    }

    if (visiblechain != -1)
    {
        for (int i = 0; i < buttons[_name][visiblechain].size(); i++)
        {
            panel->removeElement(buttons[_name][visiblechain][i]);
        }
    }

    int buttonsPerRow = (int)(pwidth / (bwidth + space));

    visiblechain = chain;

    for (int i = 0; i < cbuttons[_name].size(); i++)
    {
        if (i == visiblechain)
        {
            cbuttons[_name][i]->setState(true, false);
        }
        else
        {
            cbuttons[_name][i]->setState(false, false);
        }
    }

    for (int i = 0; i < buttons[_name][visiblechain].size(); i++)
    {
        buttons[_name][visiblechain][i]->setPos((i % buttonsPerRow) * (bwidth + space), offset - (((i / buttonsPerRow) + 2) * (bheight + space)), 0);
        panel->addElement(buttons[_name][visiblechain][i]);
    }
    panel->resize();
}
