/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <assert.h>
#ifndef WIN32
#include <sys/socket.h>
#endif
#include <iostream>

#include <QMenu>
#include <QDir>
#include <QFileDialog>
#include <QDrag>

#include <QLabel>
#include <QDialog>
#include <QTabWidget>
#include <QFile>
#include <QCursor>
#include <QPoint>
#include <QDragLeaveEvent>
#include <QGridLayout>
#include <QPixmap>
#include <QFrame>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMouseEvent>
#include <QCloseEvent>
#include <QLineEdit>
#include <QSpinBox>
#include <QComboBox>
#include <QTimer>
#include <QDateTime>
#include <QSignalMapper>
#include <QMessageBox>
#include <QMimeData>

#include "TUITab.h"
#include "TUIApplication.h"
#include "TUITextureTab.h"
#if !defined _WIN32_WCE && !defined ANDROID_TUI
#include <net/tokenbuffer.h>
#include <config/CoviseConfig.h>
#include <util/unixcompat.h>
#else
#include <wce_msg.h>
#endif

#ifndef _WIN32
#include <signal.h>
#endif

//Constructor
TUITextureTab::TUITextureTab(int id, int type, QWidget *w, int parent, QString name)
    : TUITab(id, type, w, parent, name)
{
    label = name;

    clientConn = NULL;
    StaticProps::getInstance();

    frame = new QFrame(w);
    frame->setFrameStyle(QFrame::NoFrame);

    auto grid = new QGridLayout(frame);
    layout = grid;
    widget = frame;

    externTexView = new PixScrollPane(frame);
    grid->addWidget(externTexView, 0, 1, Qt::AlignLeft);
    extLayout = new VButtonLayout(externTexView->viewport(), true, 100, 1);
    extLayout->setObjectName("extLayout");
    externTexView->setFixedWidth(95);
    externTexView->add(extLayout);
    connect(extLayout, SIGNAL(buttonPressed(int)), this, SLOT(sendChangeTextureRequest(int)));

    sceneTexView = new PixScrollPane(frame);
    grid->addWidget(sceneTexView, 0, 2, Qt::AlignLeft);
    sceneLayout = new VButtonLayout(sceneTexView->viewport(), false, 100, 1);
    sceneLayout->setObjectName("sceneLayout");
    sceneTexView->add(sceneLayout);
    sceneTexView->setFixedWidth(95);
    connect(sceneLayout, SIGNAL(buttonPressed(int)), this, SLOT(sendChangeTextureRequest(int)));

    searchTexView = new PixScrollPane(frame);
    grid->addWidget(searchTexView, 0, 3, Qt::AlignLeft);
    searchLayout = new VButtonLayout(searchTexView->viewport(), false, 100, 1);
    searchLayout->setObjectName("searchLayout");
    searchTexView->add(searchLayout);
    searchTexView->setFixedWidth(95);
    connect(searchLayout, SIGNAL(buttonPressed(int)), this, SLOT(sendChangeTextureRequest(int)));

    loadTextureButton = new QPushButton(frame);
    loadTextureButton->setText("load");
    grid->addWidget(loadTextureButton, 1, 1, Qt::AlignCenter);
    connect(loadTextureButton, SIGNAL(clicked()), this, SLOT(loadTexture()));

    updateTexturesButton = new QPushButton(frame);
    updateTexturesButton->setText(" update ");
    grid->addWidget(updateTexturesButton, 1, 2, Qt::AlignCenter);
    connect(updateTexturesButton, SIGNAL(clicked()), this, SLOT(updateTextures()));

    fromURLButton = new QPushButton(frame);
    fromURLButton->setText("search");
    fromURLButton->setFixedWidth(55);
    connect(fromURLButton, SIGNAL(clicked()), this, SLOT(doRequest()));

    nextButton = new QPushButton(frame);
    prevButton = new QPushButton(frame);
    nextButton->setText(">");
    prevButton->setText("<");
    connect(prevButton, SIGNAL(clicked()), this, SLOT(decPicNumber()));
    connect(nextButton, SIGNAL(clicked()), this, SLOT(incPicNumber()));

    nextButton->setEnabled(false);
    nextButton->setFixedWidth(20);
    prevButton->setEnabled(false);
    prevButton->setFixedWidth(20);
    fromURLButton->setEnabled(false);

    searchField = new QLineEdit(frame);
    searchField->setFixedWidth(95);
    connect(searchField, SIGNAL(returnPressed()), this, SLOT(doRequest()));
    connect(searchField, SIGNAL(textChanged(const QString &)), this, SLOT(enableButtons(const QString &)));

    searchButtonLayout = new QGridLayout(frame);
    searchButtonLayout->addWidget(prevButton, 0, 0, Qt::AlignCenter);
    searchButtonLayout->addWidget(nextButton, 0, 2, Qt::AlignCenter);
    searchButtonLayout->addWidget(fromURLButton, 0, 1, Qt::AlignCenter);

    grid->addLayout(searchButtonLayout, 1, 3);
    grid->addWidget(searchField, 2, 3);

    textureNumberSpin = new QSpinBox(frame);
    textureNumberSpin->setObjectName("textureNumberSpin");
    textureNumberSpin->setRange(0, 20);
    textureNumberSpin->setPrefix("Texture Number: ");
    textureNumberSpin->setValue(0);
    textureNumberSpin->setWrapping(true);
    connect(textureNumberSpin, SIGNAL(valueChanged(int)), this, SLOT(changeTexMode(int)));

    textureModeComboBox = new QComboBox(frame);
    textureModeComboBox->setEditable(false);
    textureModeComboBox->addItem("TexEnv Mode : DECAL");
    textureModeComboBox->addItem("TexEnv Mode : MODULATE");
    textureModeComboBox->addItem("TexEnv Mode : BLEND");
    textureModeComboBox->addItem("TexEnv Mode : REPLACE");
    textureModeComboBox->addItem("TexEnv Mode : ADD");
    textureModeComboBox->setCurrentIndex(1);
    for (int j = 0; j < 21; j++)
        textureModes[j] = 1;
    connect(textureModeComboBox, SIGNAL(activated(int)), this, SLOT(setTexMode(int)));

    textureTexGenModeComboBox = new QComboBox(frame);
    textureTexGenModeComboBox->setEditable(false);
    textureTexGenModeComboBox->addItem("TexGen Mode : OFF");
    textureTexGenModeComboBox->addItem("TexGen Mode : ObjectLinear");
    textureTexGenModeComboBox->addItem("TexGen Mode : EyeLinear");
    textureTexGenModeComboBox->addItem("TexGen Mode : SphereMap");
    textureTexGenModeComboBox->addItem("TexGen Mode : NormalMap");
    textureTexGenModeComboBox->addItem("TexGen Mode : ReflectionMap");
    textureTexGenModeComboBox->setCurrentIndex(0);
    for (int j = 0; j < 21; j++)
        textureTexGenModes[j] = 0;
    connect(textureTexGenModeComboBox, SIGNAL(activated(int)), this, SLOT(setTexGenMode(int)));

    spinLayout = new QGridLayout(frame);
    spinLayout->addWidget(textureNumberSpin, 0, 0, Qt::AlignCenter);
    spinLayout->addWidget(textureModeComboBox, 1, 0, Qt::AlignCenter);
    spinLayout->addWidget(textureTexGenModeComboBox, 2, 0, Qt::AlignCenter);
    grid->addLayout(spinLayout, 1, 0);

    fileBrowser = new DirView(frame);
    fileBrowser->setObjectName("fileBrowser");
    fileBrowser->setWindowTitle(tr("Browser"));
    fileBrowser->setHeaderLabel(tr("Dir"));
    fileBrowser->setRootIsDecorated(true);
    fileBrowser->setFixedHeight(400);
    fileBrowser->setMinimumWidth(200);
    connect(fileBrowser, SIGNAL(selectionChanged(QTreeWidgetItem *)), this, SLOT(directorySelected(QTreeWidgetItem *)));
    connect(fileBrowser,
            SIGNAL(rightButtonPressed(QTreeWidgetItem *, const QPoint &, int)),
            this,
            SLOT(popupExec(QTreeWidgetItem *)));

    Directory *root = new Directory(fileBrowser, StaticProps::getTextureDir());
    root->setOpen(true);
    fileBrowser->show();
    grid->addWidget(fileBrowser, 0, 0, Qt::AlignCenter);

    menu = new QMenu(frame);
    menu->setObjectName("menu");
    menu->addAction(tr("add Folder"), this, SLOT(addFolder()));
    menu->addAction(tr("rename Folder"), this, SLOT(renameFolder()));
    menu->addAction(tr("clear Folder"), this, SLOT(removeFolder()));

#if 0
   connect(&urlOperator,
      SIGNAL(data(const QByteArray &, Q3NetworkOperation *)),
      this,
      SLOT(newData(const QByteArray &)));

   connect(&urlOperator,
      SIGNAL(dataTransferProgress(int, int, Q3NetworkOperation *)),
      this,
      SLOT(progress(int ,int, Q3NetworkOperation *)));

   connect(&urlOperator,
      SIGNAL(finished ( Q3NetworkOperation *)),
      this,
      SLOT(endRequest(Q3NetworkOperation *)));

   connect(&urlOperator,
      SIGNAL(startedNextCopy(const Q3PtrList<Q3NetworkOperation> &)),
      this,
      SLOT(nextCopy()));
#endif

    //	q3InitNetworkProtocols();
    firstRequest = true;
    currentPicNumber = 0;
    timer = new QTimer(frame);
    lastQuest = "";
    lastPicNumber = 0;
    nextButton->setEnabled(false);
    prevButton->setEnabled(false);
    fromURLButton->setEnabled(false);
    connect(timer, SIGNAL(timeout()), this, SLOT(stopRequest()));
    currentItem = NULL;
#if !defined _WIN32_WCE && !defined ANDROID_TUI
    port = covise::coCoviseConfig::getInt("port", "COVER.TabletPC", 31802);
#else
    port = 31802;
#endif
    port++; // we use the next free port for asynchronous communication
#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN); // otherwise writes to a closed socket kill the application.
#endif
    int retval;
    for (int idx = 0; idx < 10; ++idx)
    {
        // wenn hier irgend ein port aufgemacht wird sollte der cover auch wissen welcher das ist.
        // also gleich mal ne Nachricht schicken.

        covise::TokenBuffer tb;
        tb << ID;
        tb << TABLET_TEX_PORT;
        tb << port;
        TUIMainWindow::getInstance()->send(tb);
        retval = openServer();
        if (retval == -1)
        {
            port++;
        }
        else
        {
            break;
        }
    }
    if (retval == -1)
    {
        std::cerr << "TUITextureTab: openServer() failed!" << std::endl;
        exit(3);
    }
    receivingTextures = false;
    thread = new TextureThread(this);
    thread->start();
    updateTimer = new QTimer(frame);
    connect(updateTimer, SIGNAL(timeout()), this, SLOT(updateTextureButtons()));
    updateTimer->setSingleShot(false);
    updateTimer->start(250);
    buttonList = QStringList();

    /*#ifdef TABLET_PLUGIN
           TUIMainWindow::getInstance()->getStatusBar()->setText("TextureTab initialized");
   #else
           TUIMainWindow::getInstance()->getStatusBar()->message("TextureTab initialized");
   #endif*/
}

/// Destructor
TUITextureTab::~TUITextureTab()
{
    // Remove all Files from temp directory
    QDir temp(StaticProps::getTexturePluginTempDir());
    QStringList fileList = temp.entryList();
    QStringList::Iterator it;
    for (it = fileList.begin(); it != fileList.end(); ++it)
    {
        QFile::remove(StaticProps::getTexturePluginTempDir() + *it);
    }
    //urlOperator.stop();

    thread->terminateTextureThread();
    while (!thread->isFinished())
    {
#if !defined _WIN32_WCE && !defined ANDROID_TUI
        usleep(5);
#endif
    }
    delete StaticProps::getInstance();
    closeServer();
    delete msg;
}

void TUITextureTab::updateTextureButtons()
{
    if (!thread->isSending() && !receivingTextures)
    {
        updateTexturesButton->setEnabled(true);
        return;
    }
    if (receivingTextures)
    {
        //std::cerr << "timeout & receiving\n";
        m_mutex.lock();
        QStringList::Iterator it;
        for (it = buttonList.begin(); it != buttonList.end(); ++it)
        {
            QString fileName = *it;
            //std::cerr << "Button taken : " << fileName << "\n";
            QPixmap map;
            if (map.load(fileName))
            {
                PixButton *button = new PixButton(sceneLayout);
                button->setFilename(fileName);
                button->setIcon(smoothPix(map, 64));
                sceneLayout->add(button, StaticProps::getInstance()->getButtonList().count());
                StaticProps::getInstance()->getButtonList().append(button);
                button->show();
            }
        }
        //std::cerr << "\n";
        buttonList.clear();
        m_mutex.unlock();
    }
}

void TUITextureTab::setValue(int type, covise::TokenBuffer &tb)
{
    int texNumber;
    int width;
    int height;
    int depth;
    int dataLength;

    if (type == TABLET_TRAVERSED_TEXTURES)
    {
        //usleep(1000000);
        receivingTextures = false;
    }
    if (type == TABLET_TEX_CHANGE)
    {
        int buttonNumber;
        uint64_t currentGeode;
        tb >> buttonNumber;
        tb >> currentGeode;
        /*#ifdef TABLET_PLUGIN
                      TUIMainWindow::getInstance()->getStatusBar()->setText(QString("currentGeode : %1").arg(currentGeode));
      #else
                      TUIMainWindow::getInstance()->getStatusBar()->message(QString("currentGeode : %1").arg(currentGeode));
      #endif*/

        thread->enqueueGeode(buttonNumber, currentGeode);
        updateTexturesButton->setEnabled(false);
    }
    else if (type == TABLET_TEX_MODE)
    {
        tb >> texNumber;
        tb >> textureModes[texNumber];
        tb >> textureTexGenModes[texNumber];
        if (texNumber == textureNumberSpin->value())
        {
            textureModeComboBox->setCurrentIndex(textureModes[texNumber]);
            textureTexGenModeComboBox->setCurrentIndex(textureTexGenModes[texNumber]);
        }
    }
    else if (type == TABLET_TEX)
    {
        tb >> height;
        tb >> width;
        tb >> depth;
        tb >> dataLength;

        if (depth == 24)
            dataLength = (dataLength * 4) / 3;

        char *sendData = new char[dataLength];

        for (int i = 0; i < dataLength; i++)
        {
            if ((i % 4) == 3)
                if (depth == 24)
                    sendData[i] = 1;
                else
                    tb >> sendData[i];
            else if ((i % 4) == 2)
                tb >> sendData[i - 2];
            else if ((i % 4) == 1)
                tb >> sendData[i];
            else if ((i % 4) == 0)
                tb >> sendData[i + 2];
        }
        QImage image;
        if (depth == 32)
            image = QImage(reinterpret_cast<unsigned char *>(sendData), width, height, QImage::Format_RGB32);
        else
            image = QImage(reinterpret_cast<unsigned char *>(sendData), width, height, QImage::Format_ARGB32);
        image = image.mirrored();

        QString dateTime = QString("%1_%2").arg(QDateTime::currentDateTime().toTime_t()).arg(StaticProps::getContinousNumber());
        QString fileName = StaticProps::getTexturePluginTempDir() + "texture" + dateTime + ".png";
        if (image.save(fileName, "PNG"))
        {
            m_mutex.lock();
            buttonList.append(fileName);
            //std::cerr << "Button saved : " << fileName << "\n";
            m_mutex.unlock();
        }
        delete[] sendData;
    }
    TUIElement::setValue(type, tb);
}

void TUITextureTab::loadTexture()
{
    QString file = QFileDialog::getOpenFileName(frame, tr("get texture file"),
                                                StaticProps::getTextureDir(), "*.*");
    QFileInfo info(file);
    if (!file.isEmpty())
    {
        QString newName = StaticProps::getCurrentDir() + info.fileName();
        if (newName != file)
        {
            QFile oldFile(file);
            QDataStream inStream(&oldFile);
            QFile newFile(newName);
            QDataStream outStream(&newFile);

            int fileSize = oldFile.size();
            if (oldFile.open(QIODevice::ReadOnly) && oldFile.isOpen() && newFile.open(QIODevice::WriteOnly) && newFile.isOpen())
            {
                char *c = new char[fileSize];
                while (!inStream.atEnd())
                {
                    inStream.readRawData(c, fileSize);
                    outStream.writeRawData(c, fileSize);
                }
                oldFile.close();
                newFile.close();
                //oldFile.remove();
                delete[] c;
            }
        }
        PixButton *button;
        QPixmap map;
        if (map.load(newName))
        {
            button = new PixButton(extLayout);
            button->setFilename(newName);
            button->setIcon(smoothPix(map, 64));
            int buttonNumber = StaticProps::getInstance()->getButtonList().count();
            extLayout->add(button, buttonNumber);
            StaticProps::getInstance()->getButtonList().append(button);
            //changeTexture(buttonNumber);
            button->show();
        }
    }
}

void TUITextureTab::updateTextures()
{
    delete sceneTexView->takeWidget();
    sceneLayout = new VButtonLayout(sceneTexView->viewport(), false, 100, 1);
    sceneLayout->setObjectName("sceneLayout");
    sceneTexView->setWidget(sceneLayout);
    sceneLayout->show();
    connect(sceneLayout, SIGNAL(buttonPressed(int)), this, SLOT(sendChangeTextureRequest(int)));

    covise::TokenBuffer tb;
    tb << ID;
    tb << TABLET_TEX_UPDATE;
    TUIMainWindow::getInstance()->send(tb);
    updateTexturesButton->setEnabled(false);
    receivingTextures = true;
}

void TUITextureTab::sendChangeTextureRequest(int buttonNumber)
{
    //if(!receivingTextures)
    {
        /*#ifdef TABLET_PLUGIN
                      TUIMainWindow::getInstance()->getStatusBar()->setText("texture request");
      #else
                      TUIMainWindow::getInstance()->getStatusBar()->message("texture request");
      #endif*/

        covise::TokenBuffer tb;
        tb << ID;
        tb << TABLET_TEX_CHANGE;
        tb << buttonNumber;
        TUIMainWindow::getInstance()->send(tb);
    }
}

void TUITextureTab::changeTexture(int buttonNumber, uint64_t geode)
{
    QImage image;
    if (image.load(StaticProps::getInstance()->getButtonList().at(buttonNumber)->getFilename()))
    {
        int hasAlpha;
        if (image.format() == QImage::Format_ARGB32)
            hasAlpha = 1;
        else
            hasAlpha = 0;
        int dataLength = (image.height() * image.width() * image.depth()) / 8;
        int texNumber = textureNumberSpin->value();

        covise::TokenBuffer tb;
        tb << ID;
        tb << TABLET_TEX_CHANGE;
        tb << texNumber;
        tb << textureModeComboBox->currentIndex();
        tb << textureTexGenModeComboBox->currentIndex();
        tb << hasAlpha;
        tb << image.height();
        tb << image.width();
        tb << image.depth();
        tb << dataLength;
        tb << geode;
        tb.addBinary(reinterpret_cast<char *>(image.bits()), dataLength);

        send(tb);
        tb.delete_data();
    }
}

const char *TUITextureTab::getClassName() const
{
    return "TUITextureTab";
}

QPixmap TUITextureTab::smoothPix(const QPixmap &pic, int dim)
{
    QImage image = pic.toImage();
    return QPixmap::fromImage(image.scaled(dim, dim));
}

void TUITextureTab::doRequest()
{
    QString quest = searchField->text();
    if (quest.isEmpty())
        return;
    if (quest == lastQuest)
    {
        if (currentPicNumber == lastPicNumber)
            return;
    }
    else
    {
        currentPicNumber = 0;
    }
    lastQuest = quest;
    lastPicNumber = currentPicNumber;
    quest.replace(" ", "+");
    quest.replace("\"", "%22");

    QString address = "http://images.google.de/images?q=" + quest + "&svnum=100&hl=de&start=" + QString("%1").arg(currentPicNumber) + "&sa=N&filter=0";
    //urlOperator = address;
    //urlOperator.get();
    firstRequest = true;
    picList.clear();
    sourceData = "";

    delete searchTexView->takeWidget();
    searchLayout = new VButtonLayout(searchTexView->viewport(), false, 100, 1);
    searchLayout->setObjectName("searchLayout");
    searchTexView->setWidget(searchLayout);
    searchLayout->show();
    connect(searchLayout, SIGNAL(buttonPressed(int)), this, SLOT(sendChangeTextureRequest(int)));
    nextButton->setEnabled(false);
    prevButton->setEnabled(false);
    fromURLButton->setEnabled(false);
    searchField->setEnabled(false);

    timer->setSingleShot(true);
    timer->start(15000);
}

void TUITextureTab::newData(const QByteArray &data)
{
    if (firstRequest)
        sourceData += data;
}

void TUITextureTab::progress(int done, int total, Q3NetworkOperation *)
{
    (void)done;
    (void)total;
}

void TUITextureTab::endRequest(Q3NetworkOperation * /* op */)
{
    if (firstRequest)
    {
        int index = 0;
        QString url;
        firstRequest = false;

        QRegExp exp("imgurl=http://[^&]*&");
        QStringList urls;
        while ((index = exp.indexIn(sourceData, index)) >= 0)
        {
            url = sourceData.mid(index + 7, exp.matchedLength() - 8);

            urls.append(url);
            index += exp.matchedLength();

            QString filename;
            int pos = 0;
            QRegExp e = QRegExp("/[^/]*");
            while ((pos = e.indexIn(url, pos)) >= 0)
            {
                filename = url.mid(pos + 1, e.matchedLength());
                pos += e.matchedLength();
            }
            e = QRegExp("%2520");
            filename.replace(e, " ");
            picList << filename;
        }
        //urlOperator = "";
        //urlOperator.copy(urls,StaticProps::getTexturePluginTempDir());
    }
}

void TUITextureTab::nextCopy()
{
    QPixmap pm;
    PixButton *button;

    QStringList::Iterator it;
    for (it = picList.begin(); it != picList.end(); ++it)
    {
        QString pic = StaticProps::getTexturePluginTempDir() + *it;
        if (QFile::exists(pic))
        {
            QDir dir;
            QString newName = "Tex" + QString("_%1_").arg(StaticProps::getContinousNumber()) + *it;
            QString newPath = StaticProps::getTexturePluginTempDir() + newName;
            dir.rename(pic, newPath);

            if (pm.load(newPath))
            {
                button = new PixButton(searchLayout);
                button->setFilename(newPath);
                button->setIcon(smoothPix(pm, 64));
                searchLayout->add(button, StaticProps::getInstance()->getButtonList().count());
                StaticProps::getInstance()->getButtonList().append(button);
                button->show();
            }
            break;
        }
    }
    picList.erase(it);
}

void TUITextureTab::incPicNumber()
{
    currentPicNumber += 20;
    doRequest();
}

void TUITextureTab::decPicNumber()
{
    if (currentPicNumber >= 20)
    {
        currentPicNumber -= 20;
        doRequest();
    }
}

void TUITextureTab::stopRequest()
{
    //urlOperator.stop();
    nextButton->setEnabled(true);
    fromURLButton->setEnabled(true);
    searchField->setEnabled(true);
    if (currentPicNumber >= 20)
        prevButton->setEnabled(true);
}

void TUITextureTab::enableButtons(const QString &s)
{
    if (s.isEmpty())
    {
        nextButton->setEnabled(false);
        fromURLButton->setEnabled(false);
        prevButton->setEnabled(false);
    }
    else
    {
        nextButton->setEnabled(true);
        fromURLButton->setEnabled(true);
        if (currentPicNumber >= 20)
            prevButton->setEnabled(true);
    }
}

void TUITextureTab::directorySelected(QTreeWidgetItem *item)
{
    delete externTexView->takeWidget();
    extLayout = new VButtonLayout(externTexView->viewport(), true, 100, 1);
    extLayout->setObjectName("extLayout");
    externTexView->add(extLayout);
    extLayout->show();

    connect(extLayout, SIGNAL(buttonPressed(int)), this, SLOT(sendChangeTextureRequest(int)));
    QString path = ((Directory *)item)->fullPath();
    StaticProps::setCurrentDir(path);
    QDir thisDir(path);
    const QFileInfoList &files = thisDir.entryInfoList();

    foreach (const QFileInfo f, files)
    {
        if (!f.isDir() && f.fileName() != "." && f.fileName() != "..")
        {
            QPixmap map;
            PixButton *button;
            QString name = path + f.fileName();
            if (map.load(name))
            {
                button = new PixButton(extLayout);
                button->setFilename(name);
                button->setIcon(smoothPix(map, 64));
                extLayout->add(button, StaticProps::getInstance()->getButtonList().count());
                StaticProps::getInstance()->getButtonList().append(button);
                button->show();
            }
        }
    }
}

void TUITextureTab::changeTexMode(int unit)
{
    textureModeComboBox->setCurrentIndex(textureModes[unit]);
}

void TUITextureTab::setTexMode(int mode)
{
    textureModes[textureNumberSpin->value()] = mode;
}

void TUITextureTab::changeTexGenMode(int unit)
{
    textureTexGenModeComboBox->setCurrentIndex(textureTexGenModes[unit]);
}

void TUITextureTab::setTexGenMode(int mode)
{
    textureTexGenModes[textureNumberSpin->value()] = mode;
}

void TUITextureTab::popupExec(QTreeWidgetItem *item)
{
    currentItem = (Directory *)item;
    if (currentItem)
    {
        menu->exec(QCursor::pos());
    }
}

void TUITextureTab::addFolder()
{
    FilenameDialog addDirDialog(frame, "");
    addDirDialog.setObjectName("addDirDialog");
    addDirDialog.setWindowTitle(tr("Add new Folder"));
    if (addDirDialog.exec() != QDialog::Accepted)
        return;
    QDir dir;
    if (dir.mkdir(currentItem->fullPath() + addDirDialog.getFilename()))
    {
        Directory *d = new Directory(currentItem, addDirDialog.getFilename());
        d->setOpen(true);
    }
}

void TUITextureTab::renameFolder()
{
    QString oldDir = currentItem->name();
    FilenameDialog renameDirDialog(frame, oldDir);
    renameDirDialog.setObjectName("renameDirDialog");
    renameDirDialog.setWindowTitle(tr("rename Folder"));
    if (renameDirDialog.exec() != QDialog::Accepted)
        return;
    QDir dir;
    QString oldPath = currentItem->fullPath();
    QString newDir = renameDirDialog.getFilename();
    QFileInfo info(oldPath);
    QString newPath = info.absoluteFilePath() + "/" + newDir;
    if (dir.rename(oldPath, newPath))
    {
        currentItem->setText(0, newDir);
        currentItem->setName(newDir);
        currentItem->setPath(newPath);
        currentItem->replaceInPath(oldPath, newPath + "/");
    }
}

void TUITextureTab::removeFolder()
{
    if (removeDir(currentItem->fullPath()))
    {
        delete currentItem;
        delete externTexView->takeWidget();
        extLayout = new VButtonLayout(externTexView->viewport(), true, 100, 1);
        extLayout->setObjectName("extLayout");
        externTexView->add(extLayout);
        extLayout->show();
    }
}

bool TUITextureTab::removeDir(QString path)
{
    QDir thisDir(path);

    const QFileInfoList files = thisDir.entryInfoList();
    foreach (QFileInfo f, files)
    {
        if (f.fileName() == "." || f.fileName() == "..")
            ;
        else if (f.isDir())
        {
            if (!removeDir(path + f.fileName() + "/"))
                return false;
        }
        else
        {
            if (!QFile::remove(path + f.fileName()))
                return false;
        }
    }
    if (!thisDir.rmdir(path))
        return false;
    return true;
}

int TUITextureTab::openServer()
{
    sConn = TUIMainWindow::getInstance()->toCOVERSG;
    msg = new covise::Message;
    return 0;
}

void TUITextureTab::closeServer()
{
    // tut sonst nicht (close_socket kommt nicht an)... ich probiers noch mal
    delete sConn;
    sConn = NULL;
    if (clientConn)
    {
        delete clientConn;
        clientConn = NULL;
    }
    //covise::TokenBuffer tb;
    //clients.sendMessage(tb,-2,VRB_CLOSE_VRB_CONNECTION);
}

void TUITextureTab::closeEvent(QCloseEvent *ce)
{
    closeServer();
    ce->accept();
}

void TUITextureTab::send(covise::TokenBuffer &tb)
{
    if (clientConn == NULL)
        return;
    covise::Message m(tb);
    m.type = covise::COVISE_MESSAGE_TABLET_UI;
    clientConn->send_msg(&m);
}

void TUITextureTab::handleClient(covise::Message *msg)
{
    covise::TokenBuffer tb(msg);
    switch (msg->type)
    {
    case covise::COVISE_MESSAGE_SOCKET_CLOSED:
    case covise::COVISE_MESSAGE_CLOSE_SOCKET:
    {
        delete msg->conn;
        msg->conn = NULL;
        clientConn = NULL;
    }
    break;
    case covise::COVISE_MESSAGE_TABLET_UI:
    {
        int tablettype;
        tb >> tablettype;
        int ID;

        switch (tablettype)
        {
        case TABLET_SET_VALUE:
        {
            int type;
            tb >> type;
            tb >> ID;
            this->setValue(type, tb);
        }
        break;
        default:
        {
            std::cerr << "unhandled Message!!" << tablettype << std::endl;
        }
        break;
        }
    }
    break;
    default:
    {
        if (msg->type > 0)
            std::cerr << "unhandled Message!!" << msg->type << covise::covise_msg_types_array[msg->type] << std::endl;
    }
    break;
    }
}

VButtonLayout::VButtonLayout(QWidget *parent, bool dropEnabled, int w, int h)
    : QWidget(parent)
{
    width = w;
    height = h;
    count = 0;
    layout = new QGridLayout(this);
    signalMapper = new QSignalMapper(this);
    connect(signalMapper, SIGNAL(mapped(int)), this, SLOT(buttonSlot(int)));
    setPalette(QPalette(Qt::green));
    this->dropEnabled = dropEnabled;
}

void VButtonLayout::buttonSlot(int number)
{
    /*#ifdef TABLET_PLUGIN
           TUIMainWindow::getInstance()->getStatusBar()->setText(QString("button %1 pressed ").arg(number));
   #else
           TUIMainWindow::getInstance()->getStatusBar()->message(QString("button %1 pressed ").arg(number));
   #endif*/
    emit buttonPressed(number);
}

void VButtonLayout::add(PixButton *button, int number)
{
    this->layout->addWidget(button, count % width, count / width);
    count++;
    signalMapper->setMapping(button, number);
    button->setAcceptDrops(dropEnabled);
    connect(button, SIGNAL(doubleClicked()), signalMapper, SLOT(map()));
    button->show();
}

void VButtonLayout::mousePressEvent(QMouseEvent *e)
{
    if (e->button() == Qt::RightButton)
    {
    }
}

PixButton::PixButton(VButtonLayout *parent)
    : QPushButton(parent)
{
    parentLayout = parent;
}

void PixButton::mousePressEvent(QMouseEvent *e)
{
    if (e->button() == Qt::LeftButton)
    {
        if (!icon().isNull())
        {
            QDrag *drag = new QDrag(this);
            QMimeData *mimeData = new QMimeData;

            mimeData->setText(getFilename());
            drag->setMimeData(mimeData);
            drag->setPixmap(icon().pixmap(32, 32));
            drag->start();

            e->accept();
        }
    }
}

void PixButton::dragEnterEvent(QDragEnterEvent *e)
{
    QWidget *source = dynamic_cast<QWidget *>(e->source());
    if (parentLayout->isDropEnabled()
        && e->mimeData()->hasText()
        && source
        && (source->parentWidget() != parentWidget()))
        e->acceptProposedAction();
}

void PixButton::dragLeaveEvent(QDragLeaveEvent * /*e*/)
{
}

void PixButton::dropEvent(QDropEvent *e)
{
    if (parentLayout->isDropEnabled())
    {
        QString name;
        if (e->mimeData()->hasText())
        {
            QFileInfo fileInfo(e->mimeData()->text());
            if (fileInfo.exists())
            {
                QString actualName = fileInfo.fileName();
                QString newName;
                int messageBoxResult = 3;
                do
                {
                    messageBoxResult = 3;
                    FilenameDialog *fileNameDialog = new FilenameDialog(this->parentWidget()->parentWidget(),
                                                                        actualName);
                    fileNameDialog->setObjectName("fileNameDialog");
                    fileNameDialog->setWindowTitle(tr("copy to ") + StaticProps::getCurrentDir() + " ?");
                    if (fileNameDialog->exec() != QDialog::Accepted)
                        return;
                    actualName = fileNameDialog->getFilename();
                    newName = StaticProps::getCurrentDir() + actualName;
                    delete fileNameDialog;
                    QFileInfo info(newName);

                    if (info.exists())
                    {
                        messageBoxResult = QMessageBox::information(this->parentWidget()->parentWidget(),
                                                                    "File exists",
                                                                    actualName + "\n\n already exists in\n\n" + StaticProps::getCurrentDir(),
                                                                    "Overwrite",
                                                                    "New Name",
                                                                    "Abort");
                        if (messageBoxResult == 0 && info.filePath() == fileInfo.filePath())
                            return;
                        if (messageBoxResult == 2)
                            return;
                    }

                } while (messageBoxResult == 1);

                QFile oldFile(name);
                QDataStream inStream(&oldFile);
                QFile newFile(newName);
                QDataStream outStream(&newFile);

                if (oldFile.open(QIODevice::ReadOnly) && oldFile.isOpen() && newFile.open(QIODevice::WriteOnly) && newFile.isOpen())
                {
                    char *c = new char[100];
                    while (!inStream.atEnd())
                    {
                        inStream.readRawData(c, 100);
                        outStream.writeRawData(c, 100);
                    }
                    oldFile.close();
                    newFile.close();
                    //oldFile.remove();
                    delete[] c;
                }

                QPixmap pm;
                if (pm.load(newName) && messageBoxResult != 0)
                {
                    QImage img(pm.toImage());
                    img = img.scaled(64, 64);
                    PixButton *button = new PixButton(parentLayout);
                    parentLayout->add(button, StaticProps::getInstance()->getButtonList().count());
                    StaticProps::getInstance()->getButtonList().append(button);
                    button->setFilename(newName);
                    button->setIcon(QPixmap::fromImage(img));
                    button->show();
                }
            }
        }
    }
}

void PixButton::mouseDoubleClickEvent(QMouseEvent *e)
{
    (void)e;
    emit doubleClicked();
}

PixScrollPane::PixScrollPane(QWidget *parent)
    : QScrollArea(parent)
{
    setAcceptDrops(true);
    viewport()->setAcceptDrops(true);
    viewport()->setPalette(QPalette(Qt::gray));
    viewport()->setFixedWidth(70);
    layout = NULL;
}

void PixScrollPane::add(VButtonLayout *child)
{
    layout = child;
    setWidgetResizable(true);
    setWidget(child);
    ////widget()->setLayout(layout);
}

void PixScrollPane::contentsDragEnterEvent(QDragEnterEvent *e)
{
    QWidget *source = dynamic_cast<QWidget *>(e->source());
    if (layout->isDropEnabled() && e->mimeData()->hasText() && source && (source->parentWidget() != layout))
    {
        e->accept();
        viewport()->setPalette(QPalette(QColor(128, 128, 128)));
    }
}

void PixScrollPane::contentsDragLeaveEvent(QDragLeaveEvent *e)
{
    (void)e;
    if (layout->isDropEnabled())
        viewport()->setPalette(QPalette(Qt::gray));
}

void PixScrollPane::contentsDropEvent(QDropEvent *e)
{
    if (layout->isDropEnabled())
    {
        QString name;
        if (e->mimeData()->hasText())
        {
            QFileInfo fileInfo(e->mimeData()->text());
            if (fileInfo.exists())
            {
                QString actualName = fileInfo.fileName();
                QString newName;
                int messageBoxResult = 3;
                do
                {
                    messageBoxResult = 3;
                    FilenameDialog *fileNameDialog = new FilenameDialog(this->parentWidget(),
                                                                        actualName);
                    fileNameDialog->setObjectName("fileNameDialog");
                    fileNameDialog->setWindowTitle(tr("copy to ") + StaticProps::getCurrentDir() + " ?");
                    if (fileNameDialog->exec() != QDialog::Accepted)
                        return;
                    actualName = fileNameDialog->getFilename();
                    newName = StaticProps::getCurrentDir() + actualName;
                    delete fileNameDialog;
                    QFileInfo info(newName);

                    if (info.exists())
                    {
                        messageBoxResult = QMessageBox::information(this->parentWidget(),
                                                                    "File exists",
                                                                    actualName + "\n\n already exists in\n\n" + StaticProps::getCurrentDir(),
                                                                    "Overwrite",
                                                                    "New Name",
                                                                    "Abort");
                        if (messageBoxResult == 0 && info.filePath() == fileInfo.filePath())
                            return;
                        if (messageBoxResult == 2)
                            return;
                    }

                } while (messageBoxResult == 1);

                QFile oldFile(name);
                QDataStream inStream(&oldFile);
                QFile newFile(newName);
                QDataStream outStream(&newFile);
                int fileSize = oldFile.size();
                if (oldFile.open(QIODevice::ReadOnly) && oldFile.isOpen() && newFile.open(QIODevice::WriteOnly) && newFile.isOpen())
                {
                    char *c = new char[fileSize];
                    while (!inStream.atEnd())
                    {
                        inStream.readRawData(c, fileSize);
                        outStream.writeRawData(c, fileSize);
                    }
                    oldFile.close();
                    newFile.close();
                    //oldFile.remove();
                    delete[] c;
                }

                QPixmap pm;
                if (pm.load(newName) && messageBoxResult != 0)
                {
                    QImage img(pm.toImage());
                    img = img.scaled(64, 64);
                    PixButton *button = new PixButton(layout);
                    layout->add(button, StaticProps::getInstance()->getButtonList().count());
                    StaticProps::getInstance()->getButtonList().append(button);
                    button->setFilename(newName);
                    button->setIcon(QPixmap::fromImage(img));
                    button->show();
                }
            }
        }
    }
    viewport()->setPalette(QPalette(Qt::gray));
}

StaticProps *StaticProps::instance = 0;
QString StaticProps::texturePluginDir = QDir::homePath() + "/" + ".texturePlugin" + "/";
QString StaticProps::texturePluginTempDir = texturePluginDir + "temp" + "/";
QString StaticProps::textureDir = texturePluginDir + "textures" + "/";
QString StaticProps::currentDir = textureDir;

int StaticProps::continousNumber = 0;

StaticProps *StaticProps::getInstance()
{
    if (instance)
        return instance;
    instance = new StaticProps();
    return instance;
}

QList<PixButton *> &StaticProps::getButtonList()
{
    return buttonList;
}

StaticProps::StaticProps()
{
    instance = this;
    QDir *path = new QDir();

    if (!path->exists(texturePluginDir))
        path->mkdir(texturePluginDir);
    if (!path->exists(texturePluginTempDir))
        path->mkdir(texturePluginTempDir);
    if (!path->exists(textureDir))
        path->mkdir(textureDir);
    delete path;
}

StaticProps::~StaticProps()
{
    instance = NULL;
}

FilenameDialog::FilenameDialog(QWidget *parent, QString filename)
    : QDialog(parent)
{
    setModal(true);
    setWindowTitle(tr("change name???"));

    nameField = new QLineEdit(this);
    nameField->setObjectName("nameField");

    nameField->setGeometry(10, 10, 250, 25);
    nameField->setText(filename);
    connect(nameField, SIGNAL(returnPressed()), this, SLOT(accept()));
    connect(nameField, SIGNAL(returnPressed()), this, SLOT(correctSign()));

    accept = new QPushButton("OK", this);
    accept->setGeometry(265, 10, 30, 25);
    connect(accept, SIGNAL(clicked()), SLOT(accept()));
    connect(accept, SIGNAL(clicked()), SLOT(correctSign()));
}

FilenameDialog::~FilenameDialog()
{
    delete nameField;
    delete accept;
}

void FilenameDialog::correctSign()
{
    QString text = nameField->text();
    text.replace(QRegExp("[^a-zA-Z0-9_-\\.]"), "");
    nameField->setText(text);
}

QString FilenameDialog::getFilename()
{
    return nameField->text();
}

Directory::Directory(QTreeWidget *parent, QString name)
    : QTreeWidgetItem(parent)
{
    if (!name.endsWith("/"))
        path = name + "/";
    else
        path = name;
    dirName = "/";
    p = NULL;
    //istView()->setAcceptDrops(true);
}

Directory::Directory(Directory *parent, QString name)
    : QTreeWidgetItem(parent)
{
    path = parent->fullPath() + name + "/";
    dirName = name;
    p = parent;
}

void Directory::setName(QString name)
{
    dirName = name;
}

void Directory::setPath(QString path)
{
    if (!path.endsWith("/"))
        this->path = path + "/";
    else
        this->path = path;
}

void Directory::replaceInPath(QString oldPath, QString newPath)
{
    if (!parent())
        return;

    int count = parent()->childCount();
    for (int i = 0; i < count; ++i)
    {
        Directory *item = (Directory *)child(i);
        item->setPath(item->fullPath().replace(oldPath, newPath));
        item->replaceInPath(oldPath, newPath);
    }
}

void Directory::setOpen(bool o)
{
    if (o && !childCount())
    {
        QDir thisDir(path);
        setText(0, dirName);

        foreach (QFileInfo f, thisDir.entryInfoList())
        {
            if (f.fileName() == "." || f.fileName() == "..")
                ;
            else if (f.isDir())
            {
                Directory *d = new Directory(this, f.fileName());
                d->setOpen(o);
                //d->setExpandable(true);
            }
        }
    }
    setExpanded(o);
}

QString Directory::text(int /* c */) const
{
    return dirName;
}

DirView::DirView(QWidget *parent)
    : QTreeWidget(parent)
{
    setAcceptDrops(true);
    setSelectionMode(QAbstractItemView::SingleSelection);
    //setColumnWidthMode(0,QTreeWidget::Maximum);
    /*menu = new QPopupMenu(this,"menu");
   menu->addItem("add Folder",this,SLOT(addFolder()));
   menu->addItem("rename Folder",this,SLOT(renameFolder()));
   menu->addItem("clear Folder",this,SLOT(clearFolder()));*/
}

/*void DirView::mousePressEvent(QMouseEvent *e)
{
   if(e->button() == RightButton )
      menu->exec();
   else
      QListView::mousePressEvent(e);

}*/

void DirView::dragEnterEvent(QDragEnterEvent *e)
{
    if (e->mimeData()->hasText())
        e->accept();
    //viewport()->setBackgroundColor(QColor(100,0,0));
}

void DirView::dragLeaveEvent(QDragLeaveEvent *e)
{
    (void)e;
    //viewport()->setBackgroundColor(QColor(255,255,255));
}

void DirView::dropEvent(QDropEvent *e)
{
    QTreeWidgetItem *item = itemAt(e->pos() - QPoint(0, 16));
    QString path = ((Directory *)item)->fullPath();

    if (item)
    {
        QString name;
        if (e->mimeData()->hasText())
        {
            QFileInfo fileInfo(e->mimeData()->text());
            if (fileInfo.exists())
            {
                QString actualName = fileInfo.fileName();
                QString newName;
                int messageBoxResult = 3;
                do
                {
                    messageBoxResult = 3;
                    FilenameDialog *fileNameDialog = new FilenameDialog((QWidget *)parent(), actualName);
                    fileNameDialog->setObjectName("fileNameDialog");
                    fileNameDialog->setWindowTitle("copy to " + path + " ?");
                    if (fileNameDialog->exec() != QDialog::Accepted)
                        return;
                    actualName = fileNameDialog->getFilename();
                    newName = path + actualName;
                    QFileInfo info(newName);
                    delete fileNameDialog;

                    if (info.exists())
                    {
                        messageBoxResult = QMessageBox::information((QWidget *)parent(),
                                                                    "File exists",
                                                                    actualName + "\n\n already exists in\n\n" + path,
                                                                    "Overwrite",
                                                                    "New Name",
                                                                    "Abort");
                        if (messageBoxResult == 0 && info.filePath() == fileInfo.filePath())
                            return;
                        if (messageBoxResult == 2)
                            return;
                    }
                } while (messageBoxResult == 1);

                QFile oldFile(name);
                QDataStream inStream(&oldFile);
                QFile newFile(newName);
                QDataStream outStream(&newFile);

                if (oldFile.open(QIODevice::ReadOnly) && oldFile.isOpen() && newFile.open(QIODevice::WriteOnly) && newFile.isOpen())
                {
                    char *c = new char[1024];
                    while (!inStream.atEnd())
                    {
                        inStream.readRawData(c, 1024);
                        outStream.writeRawData(c, 1024);
                    }
                    oldFile.close();
                    newFile.close();
                    //oldFile.remove();
                    delete[] c;
                }
            }
        }
    }
    //viewport()->setBackgroundColor(QColor(255,255,255));
}

TextureThread::TextureThread(TUITextureTab *tab)
{
    isRunning = true;
    this->tab = tab;
}

void TextureThread::run()
{
    std::cerr << "thread runs \n ";
    while (isRunning)
    {
        /*if(tab->getClient()==NULL)
      {
         if( tab->getServer()->check_for_input(1000))
         {
            std::cerr << "polling \n";
            tab->setClient(tab->getServer()->spawn_connection());
            struct linger linger;
            linger.l_onoff = 0;
            linger.l_linger = 0;
            setsockopt(tab->getClient()->get_id(NULL), SOL_SOCKET, SO_LINGER,(char *) &linger, sizeof(linger));
         }
      }
      else*/
        {
            if (!buttonQueue.empty())
            {
                //std::cerr << "sending texture \n" ;
                tab->lock();
                tab->changeTexture(buttonQueue.front(), geodeQueue.front());
                buttonQueue.pop();
                geodeQueue.pop();
                tab->unlock();
            }
            else if (tab->isReceivingTextures())
            {
                if (tab->getClient()->check_for_input(1))
                {
                    if (tab->getClient()->recv_msg(tab->getMessage()))
                    {
                        if (tab->getMessage())
                        {
                            tab->handleClient(tab->getMessage());
                        }
                    }
                }
            }
            else
            {
                usleep(250000);
            }
        }
    }
}

void TextureThread::enqueueGeode(int number, uint64_t geode)
{
    tab->lock();
    buttonQueue.push(number);
    geodeQueue.push(geode);
    tab->unlock();
}
