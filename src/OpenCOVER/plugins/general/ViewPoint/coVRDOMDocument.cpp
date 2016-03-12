/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdlib.h>
#include <stdio.h>
#include "coVRDOMDocument.h"
#include <cover/coVRPluginSupport.h>
#include <util/covise_version.h>

#include <QDomDocument>
#include <QFile>
#include <QString>
#include <QTextStream>

coVRDOMDocument *coVRDOMDocument::domDoc = NULL;

coVRDOMDocument *coVRDOMDocument::instance()
{
    if (domDoc == NULL)
        domDoc = new coVRDOMDocument();
    return domDoc;
}

coVRDOMDocument::coVRDOMDocument()
{
    if (opencover::cover->debugLevel(2))
        fprintf(stderr, "\nnew coVRDOMDocument\n");
    domDoc = this;
    vwpPath = "";
    readVwpPath = "";
    if (!read())
    {
        ////      if (VRCoviseConnection::covconn)
        ////         CoviseRender::send_info("Could not read xml into DOMdocument");
    }
}

coVRDOMDocument::~coVRDOMDocument()
{
    if (opencover::cover->debugLevel(2))
        fprintf(stderr, "delete coVRDOMDocument\n");

    domDoc = NULL;
}

bool
coVRDOMDocument::read(const char *f)
{

    if (f != NULL)
        vwpPath = f;

    QString errorMessage;
    int errorLine, errorCol;

    if (vwpPath == readVwpPath)
        return false;

    if (opencover::cover->debugLevel(3))
        fprintf(stderr, "---------coVRDOMDocument::read %s old=%s\n", vwpPath, readVwpPath);

    readVwpPath = vwpPath;

    if ((vwpPath == 0) || (vwpPath[0] == '\0'))
    {
        if (opencover::cover->debugLevel(3))
        {
            fprintf(stderr, "INFO: no statefile given\n");
            fprintf(stderr, "dom was not yet created - creating root element\n");
        }
        // create a dom
        rootElement = doc.createElement("COVERSTATE");
        // append something
        QDomElement tag = doc.createElement("VERSION");
        rootElement.appendChild(tag);
        QDomText t = doc.createTextNode(covise::CoviseVersion::longVersion());
        tag.appendChild(t);

        doc.appendChild(rootElement);
    }
    else //statefile was given
    {
        if (opencover::cover->debugLevel(3))
            fprintf(stderr, "filename=[%s]\n", vwpPath);

        QFile file;
        if (file.exists(vwpPath))
        {
            //fprintf(stderr, "file exists\n");
            file.setFileName(vwpPath);
            if (!file.open(QIODevice::ReadOnly))
            {
                fprintf(stderr, "ERROR: could not open file=[%s] for read\n", vwpPath);
                return (false);
            }
            else
            {
                //fprintf(stderr, "could open file for read\n");
                //fprintf(stderr, "created elem\n");

                if (!doc.setContent(&file, true, &errorMessage, &errorLine, &errorCol))
                {
                    fprintf(stderr, "could not set content error message: %s line:%d col:%d\n", errorMessage.toStdString().c_str(), errorLine, errorCol);
                    file.close();
                    return (false);
                }
                else
                {
                    if (opencover::cover->debugLevel(0))
                        fprintf(stderr, "set dom content from file\n");
                    file.close();
                    rootElement = doc.documentElement();
                }
            }
        }
        else // file does not exist
        {
            if (opencover::cover->debugLevel(3))
                fprintf(stderr, "file does not yet exist\n");
            // if the root element was not yet created
            if (!rootElement.isElement())
            {
                if (opencover::cover->debugLevel(3))
                    fprintf(stderr, "dom was not yet created - creating root element\n");
                // create a dom
                rootElement = doc.createElement("COVERSTATE");

                // append something
                QDomElement tag = doc.createElement("VERSION");
                rootElement.appendChild(tag);
                QDomText t = doc.createTextNode(covise::CoviseVersion::longVersion());
                tag.appendChild(t);

                doc.appendChild(rootElement);
            }
            else
            {
                if (opencover::cover->debugLevel(3))
                    fprintf(stderr, "dom exists already\n");
            }
            file.setFileName(vwpPath);
         /*   if (!file.open(QIODevice::WriteOnly))
            {
                fprintf(stderr, "ERROR: could not open file=[%s] for write\n", vwpPath);
                return (false);
            }
            else
            {
                // save the dom to file
                QFile file(vwpPath);
                file.open(QIODevice::WriteOnly);
                QTextStream stream(&file); // we will serialize the data into the file
                //stream.setPrintableData(true);
                //fprintf(stderr, doc.toString().latin1());
                stream << doc.toString();
                file.close();
                //fprintf(stderr,"saved dom to file\n");
            }*/
        }
    }

    //fprintf(stderr,"set rootElement and doc\n");
    domDoc->rootElement = rootElement;
    domDoc->doc = doc;
    return (true);
}
