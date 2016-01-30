#include "mainwindow.h"
#include "ui_mainwindow.h"



MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    
    coviseIsRunning = false;
    ui->setupUi(this);

    ui->textEdit->setReadOnly(true);

    QObject::connect(ui->pushButton_startCovise, SIGNAL(clicked()),this,SLOT(startcovise()));
    QObject::connect(ui->pushButton_stopCovise, SIGNAL(clicked()),this,SLOT(stopcovise()));

    setlocale(LC_NUMERIC,"C");  // be sure to use dot as decimal separator in IO
}

MainWindow::~MainWindow()
{
    void stopcovise();
    delete ui;
}


void MainWindow::startcovise()
{
    char startcommand[255];

    char rendererName[32];
#ifndef WIN32
    strcpy(rendererName, "IvRenderer");
#else
    strcpy(rendererName, "QtRenderer");
#endif


    if (coviseIsRunning == false)
    {
        // we only allow one covise instance to be run at a time ...
        bool startMapeditor = ui->startMapeditor->isChecked();

        if (startMapeditor)
        {
            sprintf(startcommand, "%s", "covise --script --gui");
        }
        else
        {
            sprintf(startcommand, "%s", "covise --script");
        }

        // start covise
#ifdef _WIN32
        // use popen
        fprintf(stderr,"we are on Windows!\n");
        pfp_write = popen(startcommand, "wt"); fflush(pfp_write);
        //Sleep(2000);

        if (pfp_write == NULL)
        {
            fprintf(stderr,"could not start covise (popen-command)!\n");
            return;
        }
#else
        // fork a child process
        // advantage: we can establish a bi-directional communication
        pfp_read = NULL;
        pfp_write = NULL;
        char *argv[] = {(char*)"covise", (char*)"--script", (char*)"--gui", NULL};
        if (!startMapeditor) argv[2] = NULL;

        pipe_open (	argv,
                    &pfp_read,     // read file handle returned
                    &pfp_write);   // write file handle returned
#endif

        // create the Covise map

        // add the modules
        fprintf(pfp_write, "network = net()\n");fflush(pfp_write);

        fprintf(pfp_write, "GenDat_1 = GenDat()\n");fflush(pfp_write);
        fprintf(pfp_write, "network.add( GenDat_1 )\n");fflush(pfp_write);
        fprintf(pfp_write, "GenDat_1.setPos( -144, -183 )\n");fflush(pfp_write);

        fprintf(pfp_write, "CuttingSurface_1 = CuttingSurface()\n");fflush(pfp_write);
        fprintf(pfp_write, "network.add( CuttingSurface_1 )\n");fflush(pfp_write);
        fprintf(pfp_write, "CuttingSurface_1.setPos( -144, -119 )\n");fflush(pfp_write);

        fprintf(pfp_write, "Colors_1 = Colors()\n");fflush(pfp_write);
        fprintf(pfp_write, "network.add( Colors_1 )\n");fflush(pfp_write);
        fprintf(pfp_write, "Colors_1.setPos( -59, -51 )\n");fflush(pfp_write);

        fprintf(pfp_write, "Collect_1 = Collect()\n");fflush(pfp_write);
        fprintf(pfp_write, "network.add( Collect_1 )\n");fflush(pfp_write);
        fprintf(pfp_write, "Collect_1.setPos( -144, 34 )\n");fflush(pfp_write);

        fprintf(pfp_write, "%s_1 = %s()\n", rendererName, rendererName);fflush(pfp_write);
        fprintf(pfp_write, "network.add( %s_1 )\n", rendererName);fflush(pfp_write);
        fprintf(pfp_write, "%s_1.setPos( -178, 119 )\n", rendererName);fflush(pfp_write);

        fprintf(pfp_write, "DomainSurface_1 = DomainSurface()\n");fflush(pfp_write);
        fprintf(pfp_write, "network.add( DomainSurface_1 )\n");fflush(pfp_write);
        fprintf(pfp_write, "DomainSurface_1.setPos( -314, -119 )\n");fflush(pfp_write);

        // set relevant parameters
        fprintf(pfp_write, "CuttingSurface_1.set_scalar( 0.500000 )\n");fflush(pfp_write);
        cuttsurf1_scalar = 0.5f;
        fprintf(pfp_write, "Colors_1.set_numSteps( 16 )\n");fflush(pfp_write);

        // connections
        fprintf(pfp_write, "network.connect( GenDat_1, \"GridOut0\", CuttingSurface_1, \"GridIn0\" )\n");fflush(pfp_write);
        fprintf(pfp_write, "network.connect( GenDat_1, \"GridOut0\", DomainSurface_1, \"GridIn0\" )\n");fflush(pfp_write);
        fprintf(pfp_write, "network.connect( GenDat_1, \"DataOut0\", CuttingSurface_1, \"DataIn0\" )\n");fflush(pfp_write);
        fprintf(pfp_write, "network.connect( CuttingSurface_1, \"GridOut0\", Collect_1, \"GridIn0\" )\n");fflush(pfp_write);
        fprintf(pfp_write, "network.connect( CuttingSurface_1, \"DataOut0\", Colors_1, \"DataIn0\" )\n");fflush(pfp_write);
        fprintf(pfp_write, "network.connect( Colors_1, \"TextureOut0\", Collect_1, \"TextureIn0\" )\n");fflush(pfp_write);
        fprintf(pfp_write, "network.connect( Collect_1, \"GeometryOut0\", %s_1, \"RenderData\" )\n", rendererName);fflush(pfp_write);
        fprintf(pfp_write, "network.connect( DomainSurface_1, \"GridOut1\", %s_1, \"RenderData\" )\n", rendererName);fflush(pfp_write);

        // run map
        fprintf(pfp_write, "runMap()\n");fflush(pfp_write);

        // run the map ...
        //fprintf(pfp_write, "runMap()\n");fflush(pfp_write);

        coviseIsRunning = true;

        ui->textEdit->setText("Covise gestartet!");

        ui->pushButton_startCovise->setText("Update CuttingSurface");
    }
    else
    {
        if (fabs(cuttsurf1_scalar - ui->dsbox_cuttsurf1_scalar->value()) > TOL)
        {
            cuttsurf1_scalar = ui->dsbox_cuttsurf1_scalar->value();
            fprintf(pfp_write, "CuttingSurface_1.set_scalar( %f )\n", cuttsurf1_scalar);
            fprintf(pfp_write, "runMap()\n");fflush(pfp_write);
            ui->textEdit->setText("CuttingSurface scalar geändert");
        }
        else
        {
            ui->textEdit->setText("CuttingSurface x-Wert nicht geändert, Schluri!");
        }
    }

}


void MainWindow::stopcovise()
{
    fprintf(pfp_write, "quit()\n");fflush(pfp_write);

    coviseIsRunning = false;

    ui->textEdit->setText("Covise beendet!");

    ui->pushButton_startCovise->setText("Start Covise");
}


#ifndef WIN32
int MainWindow::pipe_open(	char *argv[],
                            FILE **fp_read,     // read file handle returned
                            FILE **fp_write)    // write file handle returned
{
    int fd_read[2], fd_write[2];

    *fp_read = NULL;
    *fp_write = NULL;

    if (pipe(fd_read)<0)
    {
        return 0;
    }
    if (pipe(fd_write)<0)
    {
        ::close(fd_read[0]);
        ::close(fd_read[1]);
        return 0;
    }

    if(!fork())
    {
        // child process

        ::close(STDOUT_FILENO);
        ::close(STDIN_FILENO);

        dup2(fd_write[0], STDIN_FILENO);	// parent out -> child stdin
        dup2(fd_read[1], STDOUT_FILENO);	// parent in  <- child stdout

        ::close(fd_read[0]);
        ::close(fd_read[1]);
        //::close(fd_write[0]);
        //::close(fd_write[1]);

        execvp(argv[0], argv);
    }
    else
    {
        // parent process

        ::close(fd_read[1]);	// These are being used by the child
        ::close(fd_write[0]);

        *fp_write = fdopen(fd_write[1],"w");
        *fp_read = fdopen(fd_read[0],"r");
    }

    return 0;
}


int MainWindow::pipe_close(FILE *fp_read,               //returned from pfp_read_open()
                           FILE *fp_write)              //returned from pfp_read_open()
{
    if (fp_read)  fclose(fp_read);
    if (fp_write) fclose(fp_write);

    return 1;
}
#endif
