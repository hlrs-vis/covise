/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "EmbossingSimulation.h"
#include "ReadASCIIDyna.h"
#include "ResultDataBase.h"
#include <do/coDoText.h>

EmbossingSimulation::EmbossingSimulation(int argc, char *argv[])
    : coModule(argc, argv, "SCA Embossing simulation")
{
    p_PraegeParam_ = addInputPort("PraegeParam", "coDoText", "Embossing parameters");
    p_colors_ = addInputPort("NoppenColors", "coDoFloat", "design rules colors");
    p_Done_ = addOutputPort("SimulationOutcome", "coDoText", "Simulation outcome");

    p_ndivMet_ = addIntSliderParam("ndivMet", "number of metal divisions");
    p_ndivMet_->setValue(10, 30, 15);
    p_ndivPap_ = addIntSliderParam("ndivPap", "number of paper divisions");
    p_ndivPap_->setValue(10, 30, 24);

    const char *sca_path = getenv("SCA_PATH");
    char currdir[1024];
    if (sca_path)
    {
        simDir_ = sca_path;
    }
    else if (getcwd(currdir, 1024))
    {
        simDir_ = currdir;
    }

    int len = simDir_.length();
    if (len > 0 && simDir_[len - 1] != '/')
    {
        simDir_ += '/';
    }
    simDir_ += "LSDYNA/";
}

void
EmbossingSimulation::setBooleanFalse()
{
    readNoppenHoehe_ = false;
    readAusrundungsRadius_ = false;
    readAbnutzungsRadius_ = false;
    readNoppenWinkel_ = false;
    readNoppenForm_ = false;
    readLaenge1_ = false;
    readLaenge2_ = false;
    readGummiHaerte_ = false;
    readAnpressDruck_ = false;
}

EmbossingSimulation::~EmbossingSimulation()
{
}

int
EmbossingSimulation::compute(const char *port)
{
    (void)port; // silence compiler

    // first read params
    setBooleanFalse();
    // get the text object
    const coDistributedObject *inObj = p_PraegeParam_->getCurrentObject();
    if (inObj == NULL || !inObj->objectOk())
    {
        sendWarning("Got NULL pointer or object is not OK");
        return FAIL;
    }
    if (!inObj->isType("DOTEXT"))
    {
        sendWarning("Only coDoText is acceptable for input");
        return FAIL;
    }
    // check design rules
    const coDistributedObject *colors = p_colors_->getCurrentObject();
    if (colors->isType("USTSDT"))
    {
        const coDoFloat *data_colors = (const coDoFloat *)colors;
        int color_count;
        float *design_rules = NULL;
        data_colors->getAddress(&design_rules);
        for (color_count = 0; color_count < data_colors->getNumPoints(); ++color_count)
        {
            if (design_rules[color_count] != 0)
            {
                sendWarning("Actual design violates some design rules");
                return FAIL;
            }
        }
    }

    const coDoText *theText = dynamic_cast<const coDoText *>(inObj);
    int size = theText->getTextLength();
    if (size == 0)
    {
        outputDummies();
        return SUCCESS;
    }
    char *text;
    theText->getAddress(&text);
    istringstream strText;
    strText.str(string(text));
    int maxLen = strlen(text) + 1;
    std::vector<char> name(maxLen);
    while (strText >> &name[0])
    {
        if (strcmp(&name[0], "noppenHoehe") == 0)
        {
            if (ReadASCIIDyna::readFloatSlider(strText, &noppenHoehe_, maxLen) != 0)
            {
                sendWarning("Could not read noppenHoehe");
                return FAIL;
            }
            readNoppenHoehe_ = true;
        }
        else if (strcmp(&name[0], "ausrundungsRadius") == 0)
        {
            if (ReadASCIIDyna::readFloatSlider(strText, &ausrundungsRadius_, maxLen) != 0)
            {
                sendWarning("Could not read rundZellenBreite");
                return FAIL;
            }
            readAusrundungsRadius_ = true;
        }
        else if (strcmp(&name[0], "abnutzungsRadius") == 0)
        {
            if (ReadASCIIDyna::readFloatSlider(strText, &abnutzungsRadius_, maxLen) != 0)
            {
                sendWarning("Could not read abnutzungsRadius");
                return FAIL;
            }
            readAbnutzungsRadius_ = true;
        }
        else if (strcmp(&name[0], "noppenWinkel") == 0)
        {
            if (ReadASCIIDyna::readFloatSlider(strText, &noppenWinkel_, maxLen) != 0)
            {
                sendWarning("Could not read noppenWinkel");
                return FAIL;
            }
            readNoppenWinkel_ = true;
        }
        else if (strcmp(&name[0], "noppenForm") == 0)
        {
            if (ReadASCIIDyna::readChoice(strText, &noppenForm_, maxLen) != 0)
            {
                sendWarning("Could not read noppenForm");
                return FAIL;
            }
            if (noppenForm_ > 2) // Kreis
            {
                noppenForm_ = 2;
            }
            readNoppenForm_ = true;
        }
        else if (strcmp(&name[0], "laenge1") == 0)
        {
            if (ReadASCIIDyna::readFloatSlider(strText, &laenge1_, maxLen) != 0)
            {
                sendWarning("Could not read laenge1");
                return FAIL;
            }
            readLaenge1_ = true;
        }
        else if (strcmp(&name[0], "laenge2") == 0)
        {
            if (ReadASCIIDyna::readFloatSlider(strText, &laenge2_, maxLen) != 0)
            {
                sendWarning("Could not read laenge2");
                return FAIL;
            }
            readLaenge2_ = true;
        }
        else if (strcmp(&name[0], "tissueTyp") == 0)
        {
            if (ReadASCIIDyna::readChoice(strText, &tissueTyp_, maxLen) != 0)
            {
                sendWarning("Could not read tissueTyp");
                return FAIL;
            }
            readTissueTyp_ = true;
        }
        else if (strcmp(&name[0], "gummiHaerte") == 0)
        {
            if (ReadASCIIDyna::readFloatSlider(strText, &gummiHaerte_, maxLen) != 0)
            {
                sendWarning("Could not read gummiHaerte");
                return FAIL;
            }
            readGummiHaerte_ = true;
        }
        else if (strcmp(&name[0], "anpressDruck") == 0)
        {
            if (ReadASCIIDyna::readFloatSlider(strText, &anpressDruck_, maxLen) != 0)
            {
                sendWarning("Could not read anpressDruck");
                return FAIL;
            }
            readAnpressDruck_ = true;
        }
    }
    if (checkReadFlags() != 0)
    {
        return FAIL;
    }

    // check if an ausgabe-datei is available: in this case quit
    string getPath;
    // 0 -> exact hit
    // 1 -> approx.
    // 2 -> not found
    std::vector<Candidate *> FinalCandidates;
    int knob_found = checkKnobPath(getPath, FinalCandidates);
    int cand;
    for (cand = 0; cand < FinalCandidates.size(); ++cand)
    {
        delete FinalCandidates[cand];
    }

    if (knob_found == 0)
    {
        coDoText *Done = new coDoText(p_Done_->getObjName(), "Done");
        p_Done_->setCurrentObject(Done);
        return SUCCESS;
    }

    // otherwise...
    // create a params file for an ANSYS process
    // and launch ANSYS script -> result: LS-DYNA eingabe-datei
    if (ANSYSInputAndLaunch() != 0)
    {
        sendWarning("Could not create input for lsdyna");
        coDoText *Done = new coDoText(p_Done_->getObjName(), "Failed");
        p_Done_->setCurrentObject(Done);
        return FAIL;
    }

    // correct format of .k files for lsdyna input
    if (CorrectLSDynaFormat() != 0)
    {
        sendWarning("Could not transform format of .k files");
        coDoText *Done = new coDoText(p_Done_->getObjName(), "Failed");
        p_Done_->setCurrentObject(Done);
        return FAIL;
    }

    // launch LS-DYNA -> result: LS-DYNA ausgabe-datei in the database
    if (LSDYNALaunch() != 0)
    {
        sendWarning("lsdyna simulation failed");
        coDoText *Done = new coDoText(p_Done_->getObjName(), "Failed");
        p_Done_->setCurrentObject(Done);
        return FAIL;
    }

    coDoText *Done = new coDoText(p_Done_->getObjName(), "Done");
    p_Done_->setCurrentObject(Done);
    return SUCCESS;
}

int
EmbossingSimulation::checkReadFlags()
{
    if (!readNoppenHoehe_)
    {
        sendWarning("Could not read NoppenHoehe");
        return -1;
    }
    if (!readAusrundungsRadius_)
    {
        sendWarning("Could not read AusrundungsRadius");
        return -1;
    }
    if (!readAbnutzungsRadius_)
    {
        sendWarning("Could not read AbnutzungsRadius");
        return -1;
    }
    if (!readNoppenWinkel_)
    {
        sendWarning("Could not read NoppenWinkel");
        return -1;
    }
    if (!readNoppenForm_)
    {
        sendWarning("Could not read NoppenForm");
        return -1;
    }
    if (!readLaenge1_)
    {
        sendWarning("Could not read Laenge1_");
        return -1;
    }
    if (!readLaenge2_)
    {
        sendWarning("Could not read Laenge2_");
        return -1;
    }
    if (!readTissueTyp_)
    {
        sendWarning("Could not read TissueTyp");
        return -1;
    }
    if (!readGummiHaerte_)
    {
        sendWarning("Could not read GummiHaerte");
        return -1;
    }
    if (!readAnpressDruck_)
    {
        sendWarning("Could not read AnpressDruck");
        return -1;
    }
    return 0;
}

#undef yyFlexLexer
#define yyFlexLexer TopoFlexLexer
#include <FlexLexer.h>

#undef yyFlexLexer
#define yyFlexLexer BounFlexLexer
#include <FlexLexer.h>

#undef yyFlexLexer
#define yyFlexLexer HistFlexLexer
#include <FlexLexer.h>

#include "useLexers.h"

int
EmbossingSimulation::CorrectLSDynaFormat()
{
    sendInfo("Changing file formats, please wait");
    useLexers<TopoFlexLexer, Covise> useTopo(simDir_.c_str());
    int problem = useTopo.run("topology.k", "topology_k.k");
    if (problem != 0)
    {
        sendWarning("Could not correct topology.k file format");
        return problem;
    }
    useLexers<BounFlexLexer, Covise> useBoun(simDir_.c_str());
    problem = useBoun.run("boundary.k", "boundary_k.k");
    if (problem != 0)
    {
        sendWarning("Could not correct boundary.k file format");
        return problem;
    }
    useLexers<BounFlexLexer, Covise> useBounP(simDir_.c_str());
    problem = useBounP.run("bPaper.k", "bPaper_k.k");
    if (problem != 0)
    {
        sendWarning("Could not correct bPaper.k file format");
        return problem;
    }
    useLexers<HistFlexLexer, Covise> useHist(simDir_.c_str());
    problem = useHist.run("history.k", "history_k.k");
    if (problem != 0)
    {
        sendWarning("Could not correct history.k file format");
        return problem;
    }
    sendInfo("File formats were successfully changed");
    return 0;
}

int
EmbossingSimulation::ANSYSInputAndLaunch()
{
    string params(simDir_);
    params += "embParams.log";
    ofstream emb_lsdyna(params.c_str());
    if (!emb_lsdyna.rdbuf()->is_open())
    {
        sendWarning("Could not open embParams.log");
        return -1;
    }

    emb_lsdyna << "hoehe = " << noppenHoehe_ << endl;
    emb_lsdyna << "asRadius = " << ausrundungsRadius_ << endl;
    emb_lsdyna << "abRadius = " << abnutzungsRadius_ << endl;
    emb_lsdyna << "winkel = " << noppenWinkel_ << endl;
    emb_lsdyna << "nopForm = " << noppenForm_ << endl;
    emb_lsdyna << "laenge1 = " << laenge1_ << endl;
    emb_lsdyna << "laenge2 = " << laenge2_ << endl;
    emb_lsdyna << "tissue = " << tissueTyp_ << endl;
    emb_lsdyna << "gummiH = " << gummiHaerte_ << endl;
    emb_lsdyna << "anpressD = " << anpressDruck_ << endl;
    emb_lsdyna << "ndivMet = " << p_ndivMet_->getValue() << endl;
    emb_lsdyna << "ndivPap = " << p_ndivPap_->getValue() << endl;

    // create also an anpressD.k file (time versus force)
    string forceFile(simDir_);
    forceFile += "anpressD.k";
    ofstream force_lsdyna(forceFile.c_str());
    force_lsdyna << "$$$$$ Zeit, ms - Kraft, N: Zeilen 2" << endl;
    force_lsdyna << "*DEFINE_CURVE" << endl;
    force_lsdyna << "         1         0     1.000     1.000     0.000     0.000" << endl;
    int step;
    for (step = 0; step <= 20; ++step)
    {
        force_lsdyna.setf(ios::right);
        force_lsdyna.setf(ios::fixed);
        force_lsdyna.precision(8);
        force_lsdyna << setw(20) << 0.1 * step << setw(20) << 0.5 * anpressDruck_ * (-1.0 + cos(step * M_PI / 20.0)) << endl;
    }
    // this file also contains thickness information
    force_lsdyna << "$" << endl;
    force_lsdyna << "$$$$$$ Papierdicke, mm: Zeile 2 Spalten 1-4" << endl;
    force_lsdyna << "*SECTION_SHELL" << endl;
    force_lsdyna << "        1        16    0.8333       5.0       0.0       0.0         0" << endl;
    force_lsdyna.setf(ios::right);
    force_lsdyna.setf(ios::fixed);
    force_lsdyna.precision(3);
    float thickness = ReadASCIIDyna::tissueThickness[tissueTyp_ - 1];
    force_lsdyna << setw(10) << thickness << setw(10) << thickness << setw(10) << thickness << setw(10) << thickness << "      0.00" << endl;
    force_lsdyna << "*END" << endl;

    return LaunchANSYSForLS();
}

#include <sys/types.h>
#include <sys/wait.h>

int
EmbossingSimulation::LaunchANSYSForLS()
{
    pid_t ANSYS_pid;
    sendInfo("Starting preprocessing with ANSYS for LSDYNA, please wait");
    if ((ANSYS_pid = fork()) < 0)
    {
        sendWarning("fork error");
        return -1;
    }
    else if (ANSYS_pid == 0) // child
    {
        // first duplicate "driver.log" onto the standard input...
        if (chdir(simDir_.c_str()) != 0)
        {
            sendWarning("Could not change directory for simulation");
            exit(-1);
        }

        int driverFD = open("ForLS.log", O_RDONLY);
        if (driverFD < 0)
        {
            sendWarning("Open ForLS.log failed");
            exit(-1);
        }
        int dup2Out = dup2(driverFD, STDIN_FILENO);
        if (dup2Out < 0)
        {
            sendWarning("could not duplicate file descriptor of ForLS.log");
            exit(-1);
        }
        // ... and launch ansys
        vector<string> arguments;
        ReadASCIIDyna::SCA_Calls("ANSYS", arguments);
        if (arguments.size() == 0)
        {
            sendWarning("Could not read ANSYS command line from covise.config");
            exit(-1);
        }
        const char **arglist = new const char *[arguments.size() + 1];
        int arg;
        for (arg = 0; arg < arguments.size(); ++arg)
        {
            arglist[arg] = arguments[arg].c_str();
        }
        arglist[arg] = NULL;
        //if(execlp("ansys60","ansys60","-p","ANSYSRF",NULL)<0){
        if (execvp(arglist[0], const_cast<char **>(arglist)) < 0)
        {
            sendWarning("ANSYS execution failed");
            delete[] arglist;
            exit(-1);
        }
        /*
            if(execlp("ansys60","ansys60","-p","ANSYSRF",NULL)<0){
               sendWarning("ANSYS execution failed");
               exit(-1);
            }
      */
    }
    int status = 0;
    if (waitpid(ANSYS_pid, &status, 0) < 0) // parent
    {
        sendWarning("waitpid failed");
        return -1;
    }
    cerr << "ANSYS output status: " << status << endl;
    sendInfo("ANSYS computation finished!!!");
    // check status
    if (status == 2048)
    {
        status = 0;
    }
    return status;
}

int
EmbossingSimulation::createDirectories(string &outpath)
{
    const char *sca_path_c = getenv("SCA_PATH");
    if (!sca_path_c)
    {
        sendWarning("Please, set SCA_PATH environment variable");
        return 2;
    }
    string sca_path(sca_path_c);
    sca_path += "/DATABASE/";
    ResultDataBase dataB(sca_path.c_str());
    std::vector<ResultParam *> list; // prepare a list with knob parameter values:
    // noppenHoehe, ausrundungsRadius, abnutzungsRadius, noppenWinkel,
    // noppenForm, laenge1, laenge2, tissueTyp, gummiHaerte, anpressDruck
    ResultFloatParam v0("noppenHoehe", noppenHoehe_, 3);
    ResultFloatParam v1("ausrundungsRadius", ausrundungsRadius_, 3);
    ResultFloatParam v2("abnutzungsRadius", abnutzungsRadius_, 3);
    ResultFloatParam v3("noppenWinkel", noppenWinkel_, 3);
    // char *labs[] = { "Raute", "Ellipse" };
    int nop_form = noppenForm_;
    if (nop_form == 3) // Kreis
    {
        nop_form = 2;
    }
    ResultEnumParam v4("noppenForm", 2, ReadASCIIDyna::noppenFormChoices, nop_form - 1);
    ResultFloatParam v5("laenge1", laenge1_, 3);
    ResultFloatParam v6("laenge2", laenge2_, 3);
    ResultEnumParam v7("tissueTyp", 6, ReadASCIIDyna::tissueTypes,
                       tissueTyp_ - 1);
    ResultFloatParam v8("gummiHaerte", gummiHaerte_, 3);
    ResultFloatParam v9("anpressDruck", anpressDruck_, 3);

    list.push_back(&v0);
    list.push_back(&v1);
    list.push_back(&v2);
    list.push_back(&v3);
    list.push_back(&v4);
    list.push_back(&v5);
    list.push_back(&v6);
    list.push_back(&v7);
    list.push_back(&v8);
    list.push_back(&v9);

    outpath = dataB.getSaveDirName(list.size(), list);
    return 0;
}

int
EmbossingSimulation::LSDYNALaunch()
{
    int status = 0;

    pid_t LSDYNA_pid;
    sendInfo("Starting LSDYNA embossing simulation, please wait");
    if ((LSDYNA_pid = fork()) < 0)
    {
        sendWarning("fork error");
        return -1;
    }
    else if (LSDYNA_pid == 0) // child
    {
        // change directory to LSDYNA
        if (chdir(simDir_.c_str()) != 0)
        {
            sendWarning("Could not change directory for embossing simulation");
            exit(-1);
        }
        // clean directory
        vector<string> keepFiles;
        string keepLog(".log");
        string keepK(".k");
        keepFiles.push_back(keepLog);
        keepFiles.push_back(keepK);
        ReadASCIIDyna::cleanFiles(keepFiles);
        // start LSDYNA simulation
        vector<string> arguments;
        ReadASCIIDyna::SCA_Calls("LSDYNA", arguments);
        if (arguments.size() == 0)
        {
            sendWarning("Could not read LSDYNA command line from covise.config");
            exit(-1);
        }
        const char **arglist = new const char *[arguments.size() + 2];
        int arg;
        for (arg = 0; arg < arguments.size(); ++arg)
        {
            arglist[arg] = arguments[arg].c_str();
        }
        arglist[arg] = "I=exp_main.k";
        ++arg;
        arglist[arg] = NULL;
        //if(execlp("ansys60","ansys60","-p","ANSYSRF",NULL)<0){
        if (execvp(arglist[0], const_cast<char **>(arglist)) < 0)
        {
            sendWarning("lsdyna execution failed");
            delete[] arglist;
            exit(-1);
        }
        /*
            if(execlp("lsdyna60","lsdyna60","-p","DYNA","I=exp_main.k",NULL)<0){
               sendWarning("lsdyna execution failed");
               exit(-1);
            }
      */
    }
    if (waitpid(LSDYNA_pid, &status, 0) < 0) // parent
    {
        sendWarning("waitpid (LSDYNA--embossing) failed");
        return -1;
    }
    sendInfo("LSDYNA embossing computation finished!!!");

    // now we have to start the spring back computation
    sendInfo("Starting LSDYNA spring-back simulation, please wait");
    if ((LSDYNA_pid = fork()) < 0)
    {
        sendWarning("fork error");
        return -1;
    }
    else if (LSDYNA_pid == 0) // child
    {
        // change directory to LSDYNA
        if (chdir(simDir_.c_str()) != 0)
        {
            sendWarning("Could not change directory for spring-back simulation");
            exit(-1);
        }
        // clean directory
        vector<string> keepFiles;
        string keepLog(".log");
        string keepK(".k");
        string keepSB("dynain");
        string keepDEFGEO("defgeo");
        keepFiles.push_back(keepLog);
        keepFiles.push_back(keepK);
        keepFiles.push_back(keepSB);
        keepFiles.push_back(keepDEFGEO);
        ReadASCIIDyna::cleanFiles(keepFiles);
        // the actual defgeo comes from the explicit simulation,
        // that is why we give it a new name
        int moveError = rename("defgeo", "defgeo_exp");
        if (moveError != 0)
        {
            perror("Error when renaming defgeo to defgeo_exp");
        }
        // start LSDYNA simulation
        vector<string> arguments;
        ReadASCIIDyna::SCA_Calls("LSDYNA", arguments);
        if (arguments.size() == 0)
        {
            sendWarning("Could not read LSDYNA command line from covise.config");
            exit(-1);
        }
        const char **arglist = new const char *[arguments.size() + 2];
        int arg;
        for (arg = 0; arg < arguments.size(); ++arg)
        {
            arglist[arg] = arguments[arg].c_str();
        }
        arglist[arg] = "I=imp_main.k";
        ++arg;
        arglist[arg] = NULL;
        //if(execlp("ansys60","ansys60","-p","ANSYSRF",NULL)<0){
        if (execvp(arglist[0], const_cast<char **>(arglist)) < 0)
        {
            sendWarning("lsdyna execution failed");
            delete[] arglist;
            exit(-1);
        }
        /*
            if(execlp("lsdyna60","lsdyna60","-p","DYNA","I=imp_main.k",NULL)<0){
               sendWarning("lsdyna execution failed");
               exit(-1);
            }
      */
    }
    // int status;
    if (waitpid(LSDYNA_pid, &status, 0) < 0) // parent
    {
        sendWarning("waitpid (LSDYNA--spring back) failed");
        return -1;
    }
    sendInfo("LSDYNA spring back computation finished!!!");

    sendInfo("lsdyna finished, creating entry in DB");

    // create a directory for the simulation output
    string path;
    if (createDirectories(path) != 0)
    {
        sendWarning("Could not create entries in DB");
        return -1;
    }
    cerr << "New directory: " << path.c_str() << endl;
    // move files to path... topology.k defgeo movie...
    if ((LSDYNA_pid = fork()) < 0)
    {
        sendWarning("fork error");
        return -1;
    }
    else if (LSDYNA_pid == 0) // child
    {
        // change directory to LSDYNA
        if (chdir(simDir_.c_str()) != 0)
        {
            sendWarning("Could not change directory for embossing simulation");
            exit(-1);
        }
        // move files...
        string newTopology(path);
        newTopology += "topology.k";
        int moveError = rename("topology.k", newTopology.c_str());
        cerr << "New file: " << newTopology.c_str() << endl;
        if (moveError != 0)
        {
            perror("Error when moving topology.k");
        }
        string newDefgeo(path);
        newDefgeo += "defgeo";
        moveError = rename("defgeo", newDefgeo.c_str());
        cerr << "New file: " << newDefgeo.c_str() << endl;
        if (moveError != 0)
        {
            perror("Error when moving defgeo");
        }
        string newDefgeoExp(path);
        newDefgeoExp += "defgeo_exp";
        moveError = rename("defgeo_exp", newDefgeoExp.c_str());
        cerr << "New file: " << newDefgeoExp.c_str() << endl;
        if (moveError != 0)
        {
            perror("Error when moving defgeo");
        }
        string newMovie(path);
        newMovie += "movie100.s30";
        moveError = rename("movie002.s30", newMovie.c_str());
        cerr << "New file: " << newMovie.c_str() << endl;
        if (moveError != 0)
        {
            perror("Error when moving movie file");
        }
        exit(0);
    }
    if (waitpid(LSDYNA_pid, &status, 0) < 0) // parent
    {
        sendWarning("Could not create database entry");
        return -1;
    }
    sendInfo("New entry in DB was created");
    return 0;
}

// 0 -> exact hit
// 1 -> approx.
// 2 -> not found
int
EmbossingSimulation::checkKnobPath(string &getPath,
                                   std::vector<Candidate *> &FinalCandidates)
{
    const char *sca_path_c = getenv("SCA_PATH");
    if (!sca_path_c)
    {
        sendWarning("Please, set SCA_PATH environment variable");
        return 2;
    }
    string sca_path(sca_path_c);
    sca_path += "/DATABASE/";
    ResultDataBase dataB(sca_path.c_str());
    std::vector<ResultParam *> list; // prepare a list with knob parameter values:
    // noppenHoehe, ausrundungsRadius, abnutzungsRadius, noppenWinkel,
    // noppenForm, laenge1, laenge2, tissueTyp, gummiHaerte, anpressDruck
    ResultFloatParam v0("noppenHoehe", noppenHoehe_, 3);
    ResultFloatParam v1("ausrundungsRadius", ausrundungsRadius_, 3);
    ResultFloatParam v2("abnutzungsRadius", abnutzungsRadius_, 3);
    ResultFloatParam v3("noppenWinkel", noppenWinkel_, 3);
    // char *labs[] = { "Raute", "Ellipse" };
    int nop_form = noppenForm_;
    if (nop_form == 3) // Kreis
    {
        nop_form = 2;
    }
    ResultEnumParam v4("noppenForm", 2, ReadASCIIDyna::noppenFormChoices, nop_form - 1);
    ResultFloatParam v5("laenge1", laenge1_, 3);
    ResultFloatParam v6("laenge2", laenge2_, 3);
    ResultEnumParam v7("tissueTyp", 6, ReadASCIIDyna::tissueTypes,
                       tissueTyp_ - 1);
    ResultFloatParam v8("gummiHaerte", gummiHaerte_, 3);
    ResultFloatParam v9("anpressDruck", anpressDruck_, 3);

    list.push_back(&v0);
    list.push_back(&v1);
    list.push_back(&v2);
    list.push_back(&v3);
    list.push_back(&v4);
    list.push_back(&v5);
    list.push_back(&v6);
    list.push_back(&v7);
    list.push_back(&v8);
    list.push_back(&v9);
    float diff = 0.0;
    const char *path = dataB.searchForResult(diff, list, FinalCandidates, 10);

    if (path == NULL)
    {
        sendWarning("Could not find a knob with the given parameters");
        return 2;
    }
    getPath = path;
    return (diff != 0.0);
}

void
EmbossingSimulation::outputDummies()
{
    coDoText *dummy = new coDoText(p_Done_->getObjName(), "Dummy");
    p_Done_->setCurrentObject(dummy);
}

MODULE_MAIN(SCA, EmbossingSimulation)
