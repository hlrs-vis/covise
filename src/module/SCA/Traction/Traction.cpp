/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Traction.h"
#include <do/coDoText.h>

#include <sys/types.h>
#ifndef WIN32
#include <unistd.h>
#include <sys/wait.h>
#endif
#include <vector>

static int
pr_exit(int status)
{
    char buf[64];
    if (WIFEXITED(status) && WEXITSTATUS(status) != 0)
    {
        Covise::sendWarning("ANSYS abnormal termination, exit status %d", status);
        return -1;
    }
    else if (WIFSIGNALED(status))
    {
        Covise::sendWarning("ANSYS abnormal termination, signal number %d", WTERMSIG(status));
#ifdef WCOREDUMP
        if (WCOREDUMP(status))
        {
            Covise::sendWarning("core file generated");
        }
#endif
        return -1;
    }
    else if (WIFSTOPPED(status))
    {
        std::string ANSYS_Stopped("ANSYS process stopped, signal number ");
        sprintf(buf, "%d", WSTOPSIG(status));
        return -1; // well, this msy be too radical....
    }
    return 0;
}

Traction::Traction(int argc, char *argv[])
    : coModule(argc, argv, "SCA Traction")
{
    p_ZugParam_ = addInputPort("zug_param", "coDoText", "Traction parameters");
    p_mssg_ = addInputPort("mssgFromEmbossing", "coDoText", "message from embossing");
    p_file_ = addOutputPort("file_name", "coDoText", "rst file name");

    maxLen_ = 0;
    // get simulation directory
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

    if (sca_path)
    {
        simDir_ += "TRACTION/";
    }
    simulation_ = false;
}

Traction::~Traction()
{
}

int
Traction::readIntSlider(istringstream &strText, int *addr)
{
    std::vector<char> Value(maxLen_);
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
Traction::readFloatSlider(istringstream &strText, float *addr)
{
    std::vector<char> Value(maxLen_);
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
Traction::readChoice(istringstream &strText, int *addr)
{
    std::vector<char> Value(maxLen_);
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
Traction::setBooleanFalse()
{
    readDisplacement_ = false;
    readNumOutPoints_ = false;
}

bool
Traction::DifferentZugParams()
{
    const coDistributedObject *inObj = p_ZugParam_->getCurrentObject();
    if (inObj == NULL || !inObj->objectOk())
    {
        return true;
    }
    if (!inObj->isType("DOTEXT"))
    {
        return true;
    }
    const coDoText *theText = dynamic_cast<const coDoText *>(inObj);
    int size = theText->getTextLength();
    if (size == 0)
    {
        return !simulation_;
    }

    char *text;
    theText->getAddress(&text);
    maxLen_ = strlen(text) + 1;
    string Text(text, strlen(text));
    istringstream strText(Text);
    std::vector<char> name(maxLen_);
    float Displacement_c = FLT_MAX;
    int NumOutPoints_c = -1;
    while (strText >> &name[0])
    {
        if (strcmp(&name[0], "Displacement") == 0)
        {
            int int_Displacement_c;
            if (readIntSlider(strText, &int_Displacement_c) != 0)
            {
                return true;
            }
            Displacement_c = int_Displacement_c;
        }
        if (strcmp(&name[0], "NumOutPoints") == 0)
        {
            if (readIntSlider(strText, &NumOutPoints_c) != 0)
            {
                return true;
            }
        }
    }
    if (Displacement_ != Displacement_c || NumOutPoints_ != NumOutPoints_c)
        return true;

    return false;
}

int
Traction::compute(const char *port)
{
    (void)port; // silence compiler

    /*
      if(gotDummies()){
         outputDummies();
         return SUCCESS;
      }
   */
    // check, if the user wants to visuallise results,
    // whether we have previously performed a simulation
    if (simulation_ && p_ZugParam_->getCurrentObject()->getAttribute("READ_ANSYS")
        && !DifferentZugParams())
    {
        std::string rst_file_name(simDir_);
        rst_file_name += "zugmesh.rst";
        coDoText *rst_file = new coDoText(p_file_->getObjName(),
                                          rst_file_name.length() + 1);
        char *text;
        rst_file->getAddress(&text);
        strcpy(text, rst_file_name.c_str());
        rst_file->addAttribute("READ_ANSYS",
                               p_ZugParam_->getCurrentObject()->getAttribute("READ_ANSYS"));
        p_file_->setCurrentObject(rst_file);
        return SUCCESS;
    }
    setBooleanFalse();
    // get the text object
    const coDistributedObject *inObj = p_ZugParam_->getCurrentObject();
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
    const coDoText *theText = dynamic_cast<const coDoText *>(inObj);
    int size = theText->getTextLength();
    if (size == 0)
    {
        simulation_ = false; // if we are looking for a new knob, or making a new design or preparing
        // the embossing simulation, then invalidate any previous simulation
        outputDummies();
        return SUCCESS;
    }

    // before starting computation, check that
    // we have a correct "Embossing"
    const coDistributedObject *inMssg = p_mssg_->getCurrentObject();
    if (inMssg == NULL || !inMssg->objectOk())
    {
        sendWarning("Got NULL pointer or object is not OK");
        return FAIL;
    }
    if (!inMssg->isType("DOTEXT"))
    {
        sendWarning("Only coDoText is acceptable for input");
        return FAIL;
    }
    const coDoText *theMssg = dynamic_cast<const coDoText *>(inMssg);
    size = theMssg->getTextLength();

    if (size < 2)
    {
        sendWarning("Got incorrect message from Embossing");
        return FAIL;
    }
    char *the_mssg;
    theMssg->getAddress(&the_mssg);
    switch (the_mssg[0])
    {
    case 'G':
        // OK
        break;
    case 'D':
        sendWarning("Please, execute Embossing first");
        return FAIL;
    default:
        sendWarning("Got incorrect message from Embossing");
        return FAIL;
    }

    simulation_ = false;

    // now start the real thing...
    // first get param
    char *text;
    theText->getAddress(&text);
    maxLen_ = strlen(text) + 1;
    istringstream strText;
    strText.str(string(text));
    std::vector<char> name(maxLen_);
    while (strText >> &name[0])
    {
        if (strcmp(&name[0], "Displacement") == 0)
        {
            int int_Displacement;
            if (readIntSlider(strText, &int_Displacement) != 0)
            {
                sendWarning("Could not read relative traction displacement");
                return FAIL;
            }
            Displacement_ = int_Displacement;
            readDisplacement_ = true;
        }
        if (strcmp(&name[0], "NumOutPoints") == 0)
        {
            if (readIntSlider(strText, &NumOutPoints_) != 0)
            {
                sendWarning("Could not read noppenHoehe");
                return FAIL;
            }
            readNumOutPoints_ = true;
        }
    }
    if (checkReadFlags() != 0)
    {
        return FAIL;
    }

    /*
      float *xc,*yc,*zc;
      int *vl,*pl;
      dynamic_cast<coDoPolygons *>(inPol)->getAddresses(&xc,&yc,&zc,&vl,&pl);
   */

    // create file with kraft info
    std::string tractionFile = simDir_;
    tractionFile += "/zug_kraft.log";
    ofstream tractionParam(tractionFile.c_str());
    if (!tractionParam.rdbuf()->is_open())
    {
        sendWarning("Could not open zug_kraft.log");
        return FAIL;
    }
    tractionParam << "MyDispPar = " << Displacement_ << endl;
    tractionParam << "NumPoints = " << NumOutPoints_ << endl;
    tractionParam.close();

    // start ANSYS process
    sendInfo("Starting ANSYS for traction simulation, please wait");
    if (LaunchANSYS())
    {
        sendWarning("ANSYS process could not be successfully started or finished");
        return FAIL;
    }
    sendInfo("Traction simulation succeeded");
    // read in loss of tensile resistance

    std::string verlustPath(simDir_);
    verlustPath += "MyOutput.dat";

    ifstream verlust(verlustPath.c_str());
    if (!verlust.rdbuf()->is_open())
    {
        sendWarning("Could not open file tensile-resistance loss");
        return FAIL;
    }

    else
    {
        float disp = 0.0, force_embossed = 0.0, force_unembossed = 0.0;
        while (verlust >> disp >> force_embossed >> force_unembossed)
        {
        }
        if (force_unembossed > 0.0)
        {
            sendInfo("Tensile-resistance loss = %d%%",
                     int(rint(100 - 100 * force_embossed / force_unembossed)));
        }
        else
        {
            sendInfo("Could not work out tensile-resistance loss");
            return FAIL;
        }
    }

    simulation_ = true;

    // create output coDoText object with rst file name: zug_results.rst
    // outputDummies();
    std::string rst_file_name(simDir_);
    rst_file_name += "zugmesh.rst";
    coDoText *rst_file = new coDoText(p_file_->getObjName(),
                                      rst_file_name.length() + 1);
    rst_file->getAddress(&text);
    strcpy(text, rst_file_name.c_str());
    rst_file->addAttribute("READ_ANSYS",
                           p_ZugParam_->getCurrentObject()->getAttribute("READ_ANSYS"));
    p_file_->setCurrentObject(rst_file);

    return SUCCESS;
}

#include "ReadASCIIDyna.h"

int
Traction::LaunchANSYS()
{
    pid_t ANSYS_pid;
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
        int driverFD = open("zug_driver.log", O_RDONLY);
        if (driverFD < 0)
        {
            sendWarning("Open zug_driver.log failed");
            exit(-1);
        }
        int dup2Out = dup2(driverFD, STDIN_FILENO);
        if (dup2Out < 0)
        {
            sendWarning("could not duplicate file descriptor of driver.log");
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
            if(execlp("ansys60","ansys60","-b","-p","ANSYSRF","-m","250",NULL)<0){
               sendWarning("ANSYS execution failed");
               exit(-1);
            }
      */
    }
    int status;
    if (waitpid(ANSYS_pid, &status, 0) < 0) // parent
    {
        sendWarning("waitpid failed");
        return -1;
    }
    // check status
    return pr_exit(status);
}

/*
bool
Traction::gotDummies()
{
   coDistributedObject *inPol = p_cell_->getCurrentObject();
   if(inPol==NULL || !inPol->objectOk()){
      sendWarning("Got NULL pointer or polygon is not OK");
      return true;
   }
   if(!inPol->isType("POLYGN")){
      sendWarning("Did not get any coDoPolygons object, rather something different");
return true;
}
if(0 == dynamic_cast<coDoPolygons *>(inPol)->getNumPoints()){
return true;
}
return false;
}
*/
void
Traction::outputDummies()
{
    coDoText *dummy = new coDoText(p_file_->getObjName(), 0);
    p_file_->setCurrentObject(dummy);
}

int
Traction::checkReadFlags()
{
    if (!readDisplacement_)
    {
        sendWarning("Could not read traction displacement");
        return -1;
    }
    if (!readNumOutPoints_)
    {
        sendWarning("Could not read number of steps in traction");
        return -1;
    }
    return 0;
}

MODULE_MAIN(SCA, Traction)
