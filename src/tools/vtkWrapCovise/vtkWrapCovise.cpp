/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>

#include "vtkParse.h"
#include "vtkPrint.h"

//#define DEBUG 1

int findFunction(std::string name, int startIndex = 0)
{
    for (int i = startIndex >= 0 ? startIndex : 0; i < data.NumberOfFunctions; ++i)
    {
        FunctionInfo *f = &data.Functions[i];
        if (!f->IsPublic)
            continue;

        if (!f->Name)
            continue;

        if (!strcmp(f->Name, name.c_str()))
            return i;
    }

    return -1;
}

bool lookForParam(std::string name)
{
    char getname[1000], setname[1000];
    sprintf(getname, "Get%s", name.c_str());
    sprintf(setname, "Set%s", name.c_str());

    int getter = -1, setter = -1;
    for (int i = 0; i < data.NumberOfFunctions; ++i)
    {
        FunctionInfo *f = &data.Functions[i];
        if (!f->IsPublic)
            continue;

        if (!f->Name)
            continue;

        if (!strcmp(f->Name, getname))
        {
            getter = i;
        }
        else if (!strcmp(f->Name, setname))
        {
            setter = i;
        }
    }

    if (setter == -1 || getter == -1)
    {
#ifdef DEBUG
        std::cerr << "param " << name << ": no getter or setter" << std::endl;
#endif
        return false;
    }

    FunctionInfo *s = &data.Functions[setter];
    FunctionInfo *g = &data.Functions[getter];

    if (s->NumberOfArguments != g->NumberOfArguments + 1)
    {
#ifdef DEBUG
        std::cerr << "param " << name << ": getter & setter args don't match" << std::endl;
#endif
        return false;
    }

    for (int i = 0; i < g->NumberOfArguments; ++i)
    {
        if (s->ArgTypes[i] != g->ArgTypes[i])
        {
#ifdef DEBUG
            std::cerr << "param " << name << ": getter & setter args don't match" << std::endl;
#endif
            return false;
        }
    }

    if ((s->ArgTypes[s->NumberOfArguments - 1] & 0xfff) != (g->ReturnType & 0xfff))
    {
#ifdef DEBUG
        std::cerr << "param " << name << ": getter & setter args & return type don't match: " << g->ReturnType << ", " << s->ArgTypes[s->NumberOfArguments - 1] << std::endl;
#endif
        return false;
    }

    if (g->NumberOfArguments > 1)
    {
#ifdef DEBUG
        std::cerr << "param " << name << ": getter has more than one arg" << std::endl;
#endif
        return false;
    }

    Param *p = NULL;
    ParamMap::iterator it = paramMap.find(name);
    if (it != paramMap.end())
    {
        p = it->second;
        if (!p)
            return false;
        if (p->getter && p->setter)
            return false;
    }

    if (g->ReturnType == 0x303)
    {
        if (!p)
            p = new StringParam;
        StringParam *sp = dynamic_cast<StringParam *>(p);
        if (!sp)
            return false;

        if (name.rfind("FileName") == name.length() - 8)
            sp->filename = true;
    }
    else
    {
        if (!p)
            p = new ScalarParam;
        ScalarParam *sp = dynamic_cast<ScalarParam *>(p);
        if (!sp)
            return false;

        switch (g->ReturnType & 0xf)
        {
        case 1:
        case 7:
            sp->isfloat = true;
            break;
        case 4:
        case 5:
        case 6:
            sp->isint = true;
            break;
        default:
            return false;
        }
    }

    p->setter = p->getter = true;
    p->ispublic = false;
    p->name = name;

    if (g->NumberOfArguments == 0)
    {
    }
    else if (g->NumberOfArguments == 1)
    {
        switch (g->ArgTypes[0] & 0xf)
        {
        case 1:
        case 7:
            return false;
            break;
        case 4:
        case 5:
        case 6:
            break;
        default:
            return false;
        }
        p->hasindex = true;
    }
    else
    {
        delete p;
        return false;
    }

    paramMap[name] = p;
    p->ispublic = true;

    return true;
}

std::string trimspace(std::string s)
{
    while (isspace(s[0]))
        s = s.substr(1);

    while (isspace(s[s.length() - 1]))
        s = s.substr(0, s.length() - 1);

    for (int i = 0; i < s.length(); ++i)
    {
        if (isspace(s[i]))
        {
            while (isspace(s[i + 1]))
                s.replace(i + 1, 1, "");
        }
    }

    return s;
}

std::string escape(std::string s)
{
    std::string replace("\n\"");
    std::vector<std::string> replacement;
    replacement.push_back(" ");
    replacement.push_back("\\\"");
    for (int i = 0; i < replace.length(); ++i)
    {
        char search = replace[i];
        size_t pos = 0;
        while ((pos = s.find(search, pos)) != std::string::npos)
        {
            s.replace(pos, 1, replacement[i]);
            pos += replacement[i].length();
        }
    }
    return s;
}

std::string descriptionForParam(std::string name)
{
    char getname[1000], setname[1000];
    sprintf(getname, "Get%s", name.c_str());
    sprintf(setname, "Set%s", name.c_str());

    int getter = -1, setter = -1;
    for (int i = 0; i < data.NumberOfFunctions; ++i)
    {
        FunctionInfo *f = &data.Functions[i];
        if (!f->IsPublic)
            continue;

        if (!f->Name)
            continue;

        if (!strcmp(f->Name, getname))
        {
            getter = i;
        }
        else if (!strcmp(f->Name, setname))
        {
            setter = i;
        }
    }

    if (setter == -1 || getter == -1)
        return false;

    FunctionInfo *s = &data.Functions[setter];
    FunctionInfo *g = &data.Functions[getter];

    std::string desc = name;
    if (s->Comment)
        desc = s->Comment;
    else if (g->Comment)
        desc = g->Comment;

    return escape(trimspace(desc));
}

int main(int argc, char *argv[])
{
    std::string vtkinc = "/usr/include/vtk";
    int ret;

    if (getenv("VTK_INCPATH"))
        vtkinc = getenv("VTK_INCPATH");

    if (argc < 2 || argc > 4)
    {
        fprintf(stderr,
                "Usage: %s class <hint_file> output_file\n", argv[0]);
        exit(1);
    }

    fhint = NULL;
    if (argc > 2)
    {
        if (!(fhint = fopen(argv[2], "r")))
        {
            fprintf(stderr, "Error opening hint file %s\n", argv[2]);
            exit(1);
        }
    }

    std::string classname = argv[1];

    std::string currentclass = classname;

    std::vector<std::string> baseclasses;
    baseclasses.push_back(classname);

    FILE *fp = fopen("vtkparamblacklist.txt", "r");
    if (!fp && getenv("COVISEDIR"))
    {
        std::string path(getenv("COVISEDIR"));
        path += "/src/tools/vtkWrapCovise/vtkparamblacklist.txt";
        fp = fopen(path.c_str(), "r");
    }

    if (fp)
    {
        while (!feof(fp))
        {
            char buf[100000];
            char *line = fgets(buf, sizeof(buf), fp);
            if (line)
            {
                if (strlen(line) > 1)
                    line[strlen(line) - 1] = '\0';
                paramBlackList.push_back(line);
            }
        }
        fclose(fp);
    }

    do
    {
        char input[100000];
        sprintf(input, "%s/%s.h", vtkinc.c_str(), currentclass.c_str());

        FILE *fin = fopen(input, "r");
        if (!fin)
        {
            fprintf(stderr, "Error opening input file %s\n", argv[1]);
            exit(1);
        }

        memset(&data, '\0', sizeof(data));
        data.FileName = strdup(input);
        data.NameComment = NULL;
        data.Description = NULL;
        data.Caveats = NULL;
        data.SeeAlso = NULL;
        CommentState = 0;
        data.IsConcrete = 0;

        currentFunction = data.Functions;
        InitFunction(currentFunction);

        yyin = fin;
        yyout = stdout;
        ret = vtkParseparse();
        if (ret)
        {
            fprintf(stdout,
                    "*** SYNTAX ERROR found in parsing the header file %s before line %d ***\n",
                    argv[1], yylineno);
            return ret;
        }

#ifdef DEBUG
        fprintf(stderr, "super classes: %d\n", data.NumberOfSuperClasses);
#endif
        if (data.NumberOfSuperClasses == 0)
            break;

        if (data.NumberOfSuperClasses > 1)
        {
            fprintf(stderr, "more than one super class (%d)\n", data.NumberOfSuperClasses);
        }

#ifdef DEBUG
        for (int i = 0; i < data.NumberOfSuperClasses; ++i)
        {
            fprintf(stderr, "\t%s\n", data.SuperClasses[i]);
        }
#endif
        currentclass = data.SuperClasses[0];
        baseclasses.push_back(currentclass);
        if (currentclass == "vtkAlgorithm")
            break;
    } while (1);

    memset(&data, '\0', sizeof(data));
    for (int i = baseclasses.size() - 1; i >= 0; --i)
    {
        currentclass = baseclasses[i];
        fprintf(stderr, "processing %s\n", currentclass.c_str());
        char input[100000];
        sprintf(input, "%s/%s.h", vtkinc.c_str(), currentclass.c_str());

        FILE *fin = fopen(input, "r");
        if (!fin)
        {
            fprintf(stderr, "Error opening input file %s\n", argv[1]);
            exit(1);
        }

        /* memset(&data, '\0', sizeof(data)); */
        data.FileName = strdup(input);
        data.NameComment = NULL;
        data.Description = NULL;
        data.Caveats = NULL;
        data.SeeAlso = NULL;
        CommentState = 0;
        data.IsConcrete = 1;

#if 0
      currentFunction = data.Functions;
      InitFunction(currentFunction);
#endif

        yyin = fin;
        yyout = stdout;
        ret = vtkParseparse();
        if (ret)
        {
            fprintf(stdout,
                    "*** SYNTAX ERROR found in parsing the header file %s before line %d ***\n",
                    argv[1], yylineno);
            return ret;
        }
#ifdef DEBUG
        fprintf(stderr, "\n\n");
#endif
    }

    for (int i = 0; i < data.NumberOfFunctions; ++i)
    {
        if (data.Functions[i].IsPureVirtual)
        {
            fprintf(stderr, "%s is pure virtual\n", data.Functions[i].Signature);
            data.IsConcrete = 0;
            break;
        }
    }

    if (!data.IsConcrete)
    {
        fprintf(stderr, "\n%s is abstract\n\n", classname.c_str());
        exit(1);
    }

    bool haveInput = true;
    if (findFunction("SetInput") == -1)
        haveInput = false;

    std::vector<std::string> required_functions;
    if (haveInput)
        required_functions.push_back("GetNumberOfInputPorts");
    required_functions.push_back("Update");
    required_functions.push_back("GetNumberOfOutputPorts");
    required_functions.push_back("GetOutputDataObject");

    for (int i = 0; i < required_functions.size(); ++i)
    {
        if (findFunction(required_functions[i]) == -1)
        {
            fprintf(stderr, "\n%s does not have %s\n\n", classname.c_str(), required_functions[i].c_str());
            exit(1);
        }
    }

#ifdef DEBUG
    vtkParseOutput(stdout, &data);

    lookForParam("Value");
    lookForParam("FileName");
    lookForParam("FilePrefix");
    lookForParam("FilePattern");

    fprintf(stderr, "%d parameters\n", (int)paramMap.size());
    for (ParamMap::iterator it = paramMap.begin();
         it != paramMap.end();
         ++it)
    {
        if (find(paramBlackList.begin(), paramBlackList.end(), it->first) != paramBlackList.end())
            continue;

        fprintf(stderr, "param name: %s %d %d\n", it->first.c_str(),
                (int)(it->second)->getter,
                (int)(it->second)->setter);
    }
#endif

    for (ParamMap::iterator it = paramMap.begin();
         it != paramMap.end();
         ++it)
    {
        Param *p = it->second;

        if (!p->ispublic)
            continue;

        lookForParam(p->name);
    }

    const char *setInputArg = NULL;
    bool hasIndexedSetInput = false;
    bool classmatched = false;
    int setInputIdx = findFunction("SetInput");
    while (setInputIdx != -1)
    {

        FunctionInfo *f = &data.Functions[setInputIdx];
#ifdef DEBUG
        fprintf(stderr, "SetInput: %d args, argclass %s, defined in %s\n",
                f->NumberOfArguments, f->ArgClasses[f->NumberOfArguments - 1], f->ClassName);
#endif
        bool classmatch = false;
        if (!strcmp(f->ClassName, classname.c_str()))
        {
            if (!classmatched)
                hasIndexedSetInput = false;
            classmatch = true;
            classmatched = true;
        }
        if (classmatch || !classmatched)
        {
            if (f->NumberOfArguments == 2)
            {
                hasIndexedSetInput = true;
                if (f->ArgClasses[1])
                    setInputArg = f->ArgClasses[1];
            }
            else if (!setInputArg && f->ArgClasses[0])
                setInputArg = f->ArgClasses[0];
        }

        setInputIdx = findFunction("SetInput", setInputIdx + 1);
    }

    if (!setInputArg || !strcmp(setInputArg, "vtkDataObject"))
        setInputArg = "vtkDataSet";

    char modname[1000], header[10000], cpp[10000], pro[10000];
    strcpy(modname, classname.c_str());
    modname[0] = toupper(modname[0]);
    if (argc == 4)
    {
        strcpy(header, argv[3]);
    }
    else
    {
        strcpy(header, "Wrap");
        strcat(header, modname);
    }

    strcpy(cpp, header);
    strcpy(pro, modname);

    strcat(header, ".h");
    strcat(pro, ".pro");
    strcat(cpp, ".cpp");

    std::string moduletooltip = modname;
    if (data.NameComment)
    {
        moduletooltip = trimspace(data.NameComment);
        if (moduletooltip.find(classname) == 0)
            moduletooltip = moduletooltip.substr(classname.length());
        moduletooltip = trimspace(moduletooltip);
        if (moduletooltip[0] == '-')
            moduletooltip = moduletooltip.substr(1);
        moduletooltip = trimspace(moduletooltip);
        moduletooltip[0] = toupper(moduletooltip[0]);
    }

    FILE *f_h = fopen(header, "w");
    if (!f_h)
    {
        fprintf(stderr, "Error opening output file %s\n", header);
        exit(1);
    }

    FILE *f_cpp = fopen(cpp, "w");
    if (!f_cpp)
    {
        fprintf(stderr, "Error opening output file %s\n", cpp);
        exit(1);
    }

    FILE *f_make = fopen("Makefile", "w");
    if (!f_make)
    {
        fprintf(stderr, "Error opening output file Makefile\n");
        exit(1);
    }
    fprintf(f_make, "include $(COVISEDIR)/src/Makefile.qmake\n");
    fclose(f_make);

    ret = mkdir("doc", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (ret == -1)
    {
        if (errno != EEXIST)
        {
            fprintf(stderr, "Failed to create doc directory: %s\n", strerror(errno));
            exit(1);
        }
    }

    FILE *f_docparam = fopen("doc/parameters.tex.in", "w");
    if (!f_docparam)
    {
        fprintf(stderr, "Error opening output file %s\n", "doc/parameters.tex.in");
        exit(1);
    }
    fprintf(f_docparam, "\\subsubsection{Parameters}\n");
    fprintf(f_docparam, "\n");
    fprintf(f_docparam, "\\begin{longtable}{|p{4cm}|p{4cm}|p|}\n");
    fprintf(f_docparam, "\\hline\n");
    fprintf(f_docparam, "   \\bf{Name} & \\bf{Type} & \\bf{Description} \\endhead\n");
    fprintf(f_docparam, "\\hline\n");

    FILE *f_docmod = fopen("doc/beforetable.tex.in", "w");
    if (!f_docmod)
    {
        fprintf(stderr, "Error opening output file %s\n", "doc/aftertable.tex.in");
        exit(1);
    }
    fprintf(f_docmod, "%s.\n\n", moduletooltip.c_str());
    fprintf(f_docmod, "%s\n", data.Description);
    fprintf(f_docmod, "\n");
    fprintf(f_docmod, "This module (including its documentation) was auto-generated and provides the functionality of the Visualization Toolkit (VTK) class %s.\n", classname.c_str());
    fclose(f_docmod);

    if (data.Caveats || data.SeeAlso)
    {
        f_docmod = fopen("doc/aftertable.tex.in", "w");
        if (!f_docmod)
        {
            fprintf(stderr, "Error opening output file %s\n", "doc/aftertable.tex.in");
            exit(1);
        }
        if (data.Caveats)
        {
            fprintf(f_docmod, "\\subsubsection{Caveats}\n");
            fprintf(f_docmod, "%s", data.Caveats);
            fprintf(f_docmod, "\n");
        }
        if (data.SeeAlso)
        {
            fprintf(f_docmod, "\\subsubsection{See also}\n");
            fprintf(f_docmod, "%s", data.SeeAlso);
            fprintf(f_docmod, "\n");
        }
        fclose(f_docmod);
    }

    FILE *f_pro = fopen(pro, "w");
    if (!f_pro)
    {
        fprintf(stderr, "Error opening output file %s\n", pro);
        exit(1);
    }
    fprintf(f_pro, "!include($$(COVISEDIR)/mkspecs/config-first.pri):error(include of config-first.pri failed)\n");
    fprintf(f_pro, "\n");
    fprintf(f_pro, "TARGET = %s\n", modname);
    fprintf(f_pro, "PROJECT = VTK\n");
    fprintf(f_pro, "CATEGORY = VTK\n");
    fprintf(f_pro, "\n");
    fprintf(f_pro, "TEMPLATE = covisemodule\n");
    fprintf(f_pro, "\n");
    fprintf(f_pro, "CONFIG *= coalg vtk_all wnoerror\n");
    fprintf(f_pro, "\n");
    fprintf(f_pro, "SOURCES = %s\n", cpp);
    fprintf(f_pro, "HEADERS = %s\n", header);
    fprintf(f_pro, "\n");
    fprintf(f_pro, "!include ($$(COVISEDIR)/mkspecs/config-last.pri):error(include of config-last.pri failed)\n");
    fclose(f_pro);

    fprintf(f_h, "#ifndef CO_%s_H\n", modname);
    fprintf(f_h, "#define CO_%s_H 1\n", modname);
    fprintf(f_h, "\n");

    fprintf(f_h, "#include <api/coSimpleModule.h>\n");
    fprintf(f_h, "#include <util/coviseCompat.h>\n");
    fprintf(f_h, "#include <do/coDoGrid.h>\n");
    fprintf(f_h, "#include <%s.h>\n", classname.c_str());

    fprintf(f_h, "using namespace covise;\n");
    fprintf(f_h, "\n");

    fprintf(f_h, "\n");
    fprintf(f_h, "class Wrap%s : public coSimpleModule\n", modname);
    fprintf(f_h, "{\n");
    fprintf(f_h, "   %s *m_%sInstance;\n", classname.c_str(), classname.c_str());
    fprintf(f_h, "   std::vector<coInputPort *> m_inPorts;\n");
    fprintf(f_h, "   std::vector<coOutputPort *> m_outPorts;\n");
    fprintf(f_h, "   virtual int compute(const char *port);\n");
    fprintf(f_h, "   virtual void param(const char *paraName, bool inMapLoading);\n");

    fprintf(f_cpp, "#include <iostream>\n");
    fprintf(f_cpp, "#include <vtkPointSet.h>\n");
    fprintf(f_cpp, "#include <vtkPolyData.h>\n");
    fprintf(f_cpp, "#include <vtkStructuredGrid.h>\n");
    fprintf(f_cpp, "#include <vtkUnstructuredGrid.h>\n");
    fprintf(f_cpp, "#include <vtkDataSet.h>\n");
    fprintf(f_cpp, "#include <vtkDataSetAttributes.h>\n");
    fprintf(f_cpp, "#include <vtkPointData.h>\n");
    fprintf(f_cpp, "#include <vtkImageData.h>\n");
    if (haveInput)
        fprintf(f_cpp, "#include <%s.h>\n", setInputArg);
    fprintf(f_cpp, "#include <vtkInformation.h>\n");
    fprintf(f_cpp, "#include <do/coDoGeometry.h>\n");
    fprintf(f_cpp, "#include <do/coDoPixelImage.h>\n");
    fprintf(f_cpp, "#include <do/coDoTexture.h>\n");
    fprintf(f_cpp, "#include <vtk/coVtk.h>\n");
    fprintf(f_cpp, "#include \"Wrap%s.h\"\n", modname);
    fprintf(f_cpp, "\n");
    fprintf(f_cpp, "const int nports = 5;\n");
    fprintf(f_cpp, "\n");
    fprintf(f_cpp, "Wrap%s::Wrap%s(int argc, char *argv[])\n", modname, modname);
    fprintf(f_cpp, ": coSimpleModule(argc, argv, \"%s\")\n", escape(moduletooltip).c_str());
    fprintf(f_cpp, "{\n");
    fprintf(f_cpp, "   m_%sInstance = dynamic_cast<%s *>(%s::New());\n", classname.c_str(), classname.c_str(), classname.c_str());
    fprintf(f_cpp, "   if(!m_%sInstance)\n", classname.c_str());
    fprintf(f_cpp, "   {\n");
    fprintf(f_cpp, "      sendError(\"failed to create instance of \\\"%s\\\"\");\n", classname.c_str());
    fprintf(f_cpp, "      return;\n");
    fprintf(f_cpp, "   }\n");
    fprintf(f_cpp, "   int numIn = m_%sInstance->GetNumberOfInputPorts();\n", classname.c_str());
    fprintf(f_cpp, "   for(int i=0; i<numIn; ++i)\n");
    fprintf(f_cpp, "   {\n");
    fprintf(f_cpp, "      vtkInformation *info = m_%sInstance->GetInputPortInformation(i);\n", classname.c_str());
    //fprintf(f_cpp, "      info->Print(std::cout);\n");
    fprintf(f_cpp, "      char portname[1000], portdesc[1000];\n");
    fprintf(f_cpp, "      snprintf(portname, sizeof(portname), \"GridIn%%d\", i);\n");
    fprintf(f_cpp, "      snprintf(portdesc, sizeof(portdesc), \"Grid input %%d\", i);\n");
    fprintf(f_cpp, "      m_inPorts.push_back(addInputPort(portname, coVtk::inPortTypeList(info).c_str(), portdesc));\n");
    fprintf(f_cpp, "      m_inPorts[i*nports]->setRequired(coVtk::isPortRequired(info));\n");

    fprintf(f_cpp, "      snprintf(portname, sizeof(portname), \"DataIn%%d\", i*(nports-2));\n");
    fprintf(f_cpp, "      snprintf(portdesc, sizeof(portdesc), \"Scalar data mapped onto grid %%d\", i);\n");
    fprintf(f_cpp, "      m_inPorts.push_back(addInputPort(portname, \"Int|Float|Vec2|Vec3|RGBA\", portdesc));\n");
    fprintf(f_cpp, "      m_inPorts[i*nports+1]->setRequired(false);\n");

    fprintf(f_cpp, "      snprintf(portname, sizeof(portname), \"DataIn%%d\", i*(nports-2)+1);\n");
    fprintf(f_cpp, "      snprintf(portdesc, sizeof(portdesc), \"Vector data mapped onto grid %%d\", i);\n");
    fprintf(f_cpp, "      m_inPorts.push_back(addInputPort(portname, \"Vec3\", portdesc));\n");
    fprintf(f_cpp, "      m_inPorts[i*nports+2]->setRequired(false);\n");

    fprintf(f_cpp, "      snprintf(portname, sizeof(portname), \"DataIn%%d\", i*(nports-2)+2);\n");
    fprintf(f_cpp, "      snprintf(portdesc, sizeof(portdesc), \"Normals for grid %%d\", i);\n");
    fprintf(f_cpp, "      m_inPorts.push_back(addInputPort(portname, \"Vec3\", portdesc));\n");
    fprintf(f_cpp, "      m_inPorts[i*nports+3]->setRequired(false);\n");

    fprintf(f_cpp, "      snprintf(portname, sizeof(portname), \"TextureIn%%d\", i);\n");
    fprintf(f_cpp, "      snprintf(portdesc, sizeof(portdesc), \"Texture for grid %%d\", i);\n");
    fprintf(f_cpp, "      m_inPorts.push_back(addInputPort(portname, \"Texture|PixelImage|Float|Vec2|Vec3\", portdesc));\n");
    fprintf(f_cpp, "      m_inPorts[i*nports+4]->setRequired(false);\n");
    fprintf(f_cpp, "   }\n");
    fprintf(f_cpp, "   int numOut = m_%sInstance->GetNumberOfOutputPorts();\n", classname.c_str());
    fprintf(f_cpp, "   for(int i=0; i<numOut; ++i)\n");
    fprintf(f_cpp, "   {\n");
    fprintf(f_cpp, "      vtkInformation *info = m_%sInstance->GetOutputPortInformation(i);\n", classname.c_str());
    //fprintf(f_cpp, "      info->Print(std::cout);\n");
    fprintf(f_cpp, "      char portname[1000], portdesc[1000];\n");
    fprintf(f_cpp, "      snprintf(portname, sizeof(portname), \"GridOut%%d\", i);\n");
    fprintf(f_cpp, "      snprintf(portdesc, sizeof(portdesc), \"Grid output %%d\", i);\n");
    fprintf(f_cpp, "      m_outPorts.push_back(addOutputPort(portname,coVtk::outPortTypeList(info).c_str(), portdesc));\n");
    fprintf(f_cpp, "      snprintf(portname, sizeof(portname), \"DataOut%%d\", i*(nports-2));\n");
    fprintf(f_cpp, "      snprintf(portdesc, sizeof(portdesc), \"Scalar data for grid %%d\", i);\n");
    fprintf(f_cpp, "      m_outPorts.push_back(addOutputPort(portname,\"Int|Float|Vec2|Vec3|RGBA\",portdesc));\n");
    fprintf(f_cpp, "      snprintf(portname, sizeof(portname), \"DataOut%%d\", i*(nports-2)+1);\n");
    fprintf(f_cpp, "      snprintf(portdesc, sizeof(portdesc), \"Vector data for grid %%d\", i);\n");
    fprintf(f_cpp, "      m_outPorts.push_back(addOutputPort(portname,\"Vec3\", portdesc));\n");

    fprintf(f_cpp, "      snprintf(portname, sizeof(portname), \"DataOut%%d\", i*(nports-2)+2);\n");
    fprintf(f_cpp, "      snprintf(portdesc, sizeof(portdesc), \"Normals for grid %%d\", i);\n");
    fprintf(f_cpp, "      m_outPorts.push_back(addOutputPort(portname,\"Vec3\", portdesc));\n");

    fprintf(f_cpp, "      snprintf(portname, sizeof(portname), \"TextureOut%%d\", i);\n");
    fprintf(f_cpp, "      snprintf(portdesc, sizeof(portdesc), \"Texture for grid %%d\", i);\n");
    fprintf(f_cpp, "      m_outPorts.push_back(addOutputPort(portname,\"Texture|Float|Vec2|Vec3\", portdesc));\n");
    fprintf(f_cpp, "   }\n");

    char pbuf[100000];
    strcpy(pbuf, "");
    std::vector<std::string> paramfunc;
    fprintf(f_h, "\n");
    for (ParamMap::iterator it = paramMap.begin();
         it != paramMap.end();
         ++it)
    {
        if (std::find(paramBlackList.begin(), paramBlackList.end(), it->first) != paramBlackList.end())
            continue;

        Param *p = it->second;

        if (!p->ispublic)
            continue;

        if (!p->getter || !p->setter)
            continue;

        fprintf(f_docparam, "\\hline\n");
        fprintf(f_cpp, "\n");
        snprintf(pbuf, sizeof(pbuf), "if(!strcmp(paramName,\"%s\")) {", p->name.c_str());
        paramfunc.push_back(pbuf);
        std::string description = descriptionForParam(p->name);
        const char *desc = description.c_str();
        if (ScalarParam *sp = dynamic_cast<ScalarParam *>(p))
        {
            bool handled = true;
            bool slider = false;
            bool choice = false;
            if (sp->isbool)
            {
                fprintf(f_docparam, "%s & Boolean & %s \\\\\n", p->name.c_str(), desc);
                fprintf(f_h, "   coBooleanParam *m_%sParam;\n", p->name.c_str());
                fprintf(f_cpp, "   m_%sParam = addBooleanParam(\"%s\", \"%s\");\n", p->name.c_str(), p->name.c_str(), desc);
            }
            else if (sp->isint)
            {
                if (sp->clamped)
                {
                    std::string asname = "Get" + sp->name + "AsString";
                    if (findFunction(asname) != -1)
                        choice = true;
                    else
                        slider = true;
                    if (choice)
                    {
                        fprintf(f_docparam, "%s & Choice & %s \\\\\n", p->name.c_str(), desc);
                        fprintf(f_h, "   coChoiceParam *m_%sParam;\n", p->name.c_str());
                        fprintf(f_cpp, "   m_%sParam = addChoiceParam(\"%s\", \"%s\");\n", p->name.c_str(), p->name.c_str(), desc);
                    }
                    else
                    {
                        fprintf(f_docparam, "%s & IntSlider & %s \\\\\n", p->name.c_str(), desc);
                        fprintf(f_h, "   coIntSliderParam *m_%sParam;\n", p->name.c_str());
                        fprintf(f_cpp, "   m_%sParam = addIntSliderParam(\"%s\", \"%s\");\n", p->name.c_str(), p->name.c_str(), desc);
                    }
                }
                else
                {
                    fprintf(f_docparam, "%s & IntScalar & %s \\\\\n", p->name.c_str(), desc);
                    fprintf(f_h, "   coIntScalarParam *m_%sParam;\n", p->name.c_str());
                    fprintf(f_cpp, "   m_%sParam = addInt32Param(\"%s\", \"%s\");\n", p->name.c_str(), p->name.c_str(), desc);
                }
            }
            else if (sp->isfloat)
            {
                if (sp->clamped)
                {
                    slider = true;
                    fprintf(f_docparam, "%s & FloatSlider & %s \\\\\n", p->name.c_str(), desc);
                    fprintf(f_h, "   coFloatSliderParam *m_%sParam;\n", p->name.c_str());
                    fprintf(f_cpp, "   m_%sParam = addFloatSliderParam(\"%s\", \"%s\");\n", p->name.c_str(), p->name.c_str(), desc);
                }
                else
                {
                    fprintf(f_docparam, "%s & FloatScalar & %s \\\\\n", p->name.c_str(), desc);
                    fprintf(f_h, "   coFloatParam *m_%sParam;\n", p->name.c_str());
                    fprintf(f_cpp, "   m_%sParam = addFloatParam(\"%s\", \"%s\");\n", p->name.c_str(), p->name.c_str(), desc);
                }
            }
            else
            {
                fprintf(stderr, "unsupported parameter type for %s\n", p->name.c_str());
                handled = false;
            }
            if (handled)
            {
                if (sp->hasindex)
                {
                    fprintf(f_cpp, "   m_%sParam->setValue(m_%sInstance->Get%s(0));\n",
                            p->name.c_str(), classname.c_str(), p->name.c_str());
                    snprintf(pbuf, sizeof(pbuf), "   m_%sInstance->Set%s(0, m_%sParam->getValue());",
                             classname.c_str(), p->name.c_str(), p->name.c_str());
                }
                else
                {
                    if (choice)
                    {
                        fprintf(f_cpp, "   {\n");
                        fprintf(f_cpp, "      static std::vector<const char *> %sParamValues;\n", p->name.c_str());
                        fprintf(f_cpp, "      int %sDefault = m_%sInstance->Get%s();\n", p->name.c_str(), classname.c_str(), p->name.c_str());
                        fprintf(f_cpp, "      for(int i=0; i<m_%sInstance->Get%sMinValue(); ++i)\n",
                                classname.c_str(), p->name.c_str());
                        fprintf(f_cpp, "         %sParamValues.push_back(\"(invalid)\");\n", p->name.c_str());
                        fprintf(f_cpp, "      for(int i=m_%sInstance->Get%sMinValue(); i<=m_%sInstance->Get%sMaxValue(); ++i) {\n",
                                classname.c_str(), p->name.c_str(),
                                classname.c_str(), p->name.c_str());
                        fprintf(f_cpp, "         m_%sInstance->Set%s(i);\n", classname.c_str(), p->name.c_str());
                        fprintf(f_cpp, "         %sParamValues.push_back(m_%sInstance->Get%sAsString());\n", p->name.c_str(), classname.c_str(), p->name.c_str());
                        fprintf(f_cpp, "      }\n");
                        fprintf(f_cpp, "      m_%sInstance->Set%s(%sDefault);\n", classname.c_str(), p->name.c_str(), p->name.c_str());
                        fprintf(f_cpp, "      m_%sParam->setValue(%sParamValues.size(), &%sParamValues[0], %sDefault);\n", p->name.c_str(), p->name.c_str(), p->name.c_str(), p->name.c_str());
                        fprintf(f_cpp, "   }\n");
                    }
                    else if (slider)
                        fprintf(f_cpp, "   m_%sParam->setValue(m_%sInstance->Get%sMinValue(), m_%sInstance->Get%sMaxValue(), m_%sInstance->Get%s());\n",
                                p->name.c_str(),
                                classname.c_str(), p->name.c_str(),
                                classname.c_str(), p->name.c_str(),
                                classname.c_str(), p->name.c_str());
                    else
                        fprintf(f_cpp, "   m_%sParam->setValue(m_%sInstance->Get%s());\n",
                                p->name.c_str(), classname.c_str(), p->name.c_str());
                    if (sp->clamped)
                    {
                        const char *format = "%d";
                        if (sp->isfloat)
                            format = "%f";
                        snprintf(pbuf, sizeof(pbuf), "   if(m_%sInstance->Get%sMinValue() > m_%sParam->getValue()) {",
                                 classname.c_str(), p->name.c_str(), p->name.c_str());
                        paramfunc.push_back(pbuf);
                        snprintf(pbuf, sizeof(pbuf), "      sendWarning(\"%s out of range (%s-%s)\", m_%sInstance->Get%sMinValue(), m_%sInstance->Get%sMaxValue());", p->name.c_str(), format, format,
                                 classname.c_str(), p->name.c_str(),
                                 classname.c_str(), p->name.c_str());
                        paramfunc.push_back(pbuf);
                        snprintf(pbuf, sizeof(pbuf), "   m_%sParam->setValue(m_%sInstance->Get%sMinValue());",
                                 p->name.c_str(), classname.c_str(), p->name.c_str());
                        paramfunc.push_back(pbuf);
                        snprintf(pbuf, sizeof(pbuf), "   } else if(m_%sInstance->Get%sMaxValue() < m_%sParam->getValue()) {",
                                 classname.c_str(), p->name.c_str(), p->name.c_str());
                        paramfunc.push_back(pbuf);
                        snprintf(pbuf, sizeof(pbuf), "      sendWarning(\"%s out of range (%s-%s)\", m_%sInstance->Get%sMinValue(), m_%sInstance->Get%sMaxValue());", p->name.c_str(), format, format,
                                 classname.c_str(), p->name.c_str(),
                                 classname.c_str(), p->name.c_str());
                        paramfunc.push_back(pbuf);
                        snprintf(pbuf, sizeof(pbuf), "   m_%sParam->setValue(m_%sInstance->Get%sMaxValue());",
                                 p->name.c_str(), classname.c_str(), p->name.c_str());
                        paramfunc.push_back(pbuf);
                        snprintf(pbuf, sizeof(pbuf), "   }");
                        paramfunc.push_back(pbuf);
                    }
                    snprintf(pbuf, sizeof(pbuf), "   m_%sInstance->Set%s(m_%sParam->getValue());",
                             classname.c_str(), p->name.c_str(), p->name.c_str());
                }
                paramfunc.push_back(pbuf);
            }
        }
        else if (StringParam *sp = dynamic_cast<StringParam *>(p))
        {
            if (sp->filename)
            {
                fprintf(f_docparam, "%s & Browser & %s \\\\\n", p->name.c_str(), desc);

                fprintf(f_h, "   coFileBrowserParam *m_%sParam;\n", p->name.c_str());
                fprintf(f_cpp, "   m_%sParam = addFileBrowserParam(\"%s\", \"%s\");\n", p->name.c_str(), p->name.c_str(), desc);
                fprintf(f_cpp, "   m_%sParam->setValue(m_%sInstance->Get%s()?m_%sInstance->Get%s():\".\", \"*\");\n",
                        p->name.c_str(), classname.c_str(),
                        p->name.c_str(), classname.c_str(),
                        p->name.c_str());
            }
            else
            {
                fprintf(f_docparam, "%s & String & %s \\\\\n", p->name.c_str(), desc);

                fprintf(f_h, "   coStringParam *m_%sParam;\n", p->name.c_str());
                fprintf(f_cpp, "   m_%sParam = addStringParam(\"%s\", \"%s\");\n", p->name.c_str(), p->name.c_str(), desc);
                fprintf(f_cpp, "   m_%sParam->setValue(m_%sInstance->Get%s());\n",
                        p->name.c_str(), classname.c_str(), p->name.c_str());
            }
            snprintf(pbuf, sizeof(pbuf), "   m_%sInstance->Set%s(m_%sParam->getValue());",
                     classname.c_str(), p->name.c_str(), p->name.c_str());
            paramfunc.push_back(pbuf);
        }
        else if (VecParam *vp = dynamic_cast<VecParam *>(p))
        {
            if (vp->isdouble || vp->isfloat)
            {
                const char *type = "double";
                if (vp->isfloat)
                    type = "float";
                fprintf(f_docparam, "%s & FloatVector & %s \\\\\n", p->name.c_str(), desc);

                fprintf(f_h, "   coFloatVectorParam *m_%sParam;\n", p->name.c_str());
                fprintf(f_cpp, "   m_%sParam = addFloatVectorParam(\"%s\", \"%s\", %d);\n", p->name.c_str(), p->name.c_str(), desc, vp->dim);
                fprintf(f_cpp, "   for(int i=0; i<%d; ++i)\n", vp->dim);
                fprintf(f_cpp, "   {\n");
                fprintf(f_cpp, "      %s *v = m_%sInstance->Get%s();\n", type, classname.c_str(), p->name.c_str());
                fprintf(f_cpp, "      m_%sParam->setValue(i, v[i]);\n", p->name.c_str());
                fprintf(f_cpp, "   }\n");

                snprintf(pbuf, sizeof(pbuf), "   {\n");
                paramfunc.push_back(pbuf);
                snprintf(pbuf, sizeof(pbuf), "      %s v[%d];\n", type, vp->dim);
                paramfunc.push_back(pbuf);

                snprintf(pbuf, sizeof(pbuf), "      for(int i=0; i<%d; ++i)\n", vp->dim);
                paramfunc.push_back(pbuf);
                snprintf(pbuf, sizeof(pbuf), "      {\n");
                paramfunc.push_back(pbuf);
                snprintf(pbuf, sizeof(pbuf), "         v[i] = m_%sParam->getValue(i);\n", p->name.c_str());
                paramfunc.push_back(pbuf);
                snprintf(pbuf, sizeof(pbuf), "      }\n");
                paramfunc.push_back(pbuf);
                snprintf(pbuf, sizeof(pbuf), "      m_%sInstance->Set%s(v);\n", classname.c_str(), p->name.c_str());
                paramfunc.push_back(pbuf);
                snprintf(pbuf, sizeof(pbuf), "   }\n");
            }
            else
            {
                const char *type = "int";
                if (vp->ischar)
                {
                    if (vp->isunsigned)
                        type = "unsigned char";
                    else
                        type = "char";
                }
                fprintf(f_docparam, "%s & IntVector & %s \\\\\n", p->name.c_str(), desc);

                fprintf(f_h, "   coIntVectorParam *m_%sParam;\n", p->name.c_str());
                fprintf(f_cpp, "   m_%sParam = addInt32VectorParam(\"%s\", \"%s\", %d);\n", p->name.c_str(), p->name.c_str(), desc, vp->dim);
                fprintf(f_cpp, "   for(int i=0; i<%d; ++i)\n", vp->dim);
                fprintf(f_cpp, "   {\n");
                fprintf(f_cpp, "      %s *v = m_%sInstance->Get%s();\n", type, classname.c_str(), p->name.c_str());
                fprintf(f_cpp, "      m_%sParam->setValue(i, v[i]);\n", p->name.c_str());
                fprintf(f_cpp, "   }\n");

                snprintf(pbuf, sizeof(pbuf), "   {\n");
                paramfunc.push_back(pbuf);
                snprintf(pbuf, sizeof(pbuf), "      %s v[%d];\n", type, vp->dim);
                paramfunc.push_back(pbuf);

                snprintf(pbuf, sizeof(pbuf), "      for(int i=0; i<%d; ++i)\n", vp->dim);
                paramfunc.push_back(pbuf);
                snprintf(pbuf, sizeof(pbuf), "      {\n");
                paramfunc.push_back(pbuf);
                snprintf(pbuf, sizeof(pbuf), "         v[i] = m_%sParam->getValue(i);\n", p->name.c_str());
                paramfunc.push_back(pbuf);
                snprintf(pbuf, sizeof(pbuf), "      }\n");
                paramfunc.push_back(pbuf);
                snprintf(pbuf, sizeof(pbuf), "      m_%sInstance->Set%s(v);\n", classname.c_str(), p->name.c_str());
                paramfunc.push_back(pbuf);
                snprintf(pbuf, sizeof(pbuf), "   }\n");
            }
            paramfunc.push_back(pbuf);
        }
        else
        {
            fprintf(stderr, "Parameter of unsupported type: %s\n", p->name.c_str());
        }
        paramfunc.push_back("}");
    }

    fprintf(f_h, "\n");
    fprintf(f_h, "   public:\n");
    fprintf(f_h, "   Wrap%s(int argc, char *argv[]);\n", modname);
    fprintf(f_h, "   virtual ~Wrap%s();\n", modname);
    fprintf(f_h, "};\n");
    fprintf(f_h, "#endif\n");

    fprintf(f_cpp, "}\n");
    fprintf(f_cpp, "\n");
    fprintf(f_cpp, "Wrap%s::~Wrap%s()\n", modname, modname);
    fprintf(f_cpp, "{\n");
    fprintf(f_cpp, "   m_%sInstance->Delete();\n", classname.c_str());
    fprintf(f_cpp, "}\n");
    fprintf(f_cpp, "\n");

    fprintf(f_cpp, "int Wrap%s::compute(const char *)\n", modname);
    fprintf(f_cpp, "{\n");
    fprintf(f_cpp, "   if(!m_%sInstance)\n", classname.c_str());
    fprintf(f_cpp, "   {\n");
    fprintf(f_cpp, "      sendError(\"failed to create instance of \\\"%s\\\"\");\n", classname.c_str());
    fprintf(f_cpp, "      return STOP_PIPELINE;\n");
    fprintf(f_cpp, "   }\n");
    fprintf(f_cpp, "\n");
    fprintf(f_cpp, "   std::vector<int> inputType, inputTexType;\n");
    if (haveInput)
    {
        fprintf(f_cpp, "   for(int i=0; i<m_%sInstance->GetNumberOfInputPorts(); ++i)\n", classname.c_str());
        fprintf(f_cpp, "   {\n");
        fprintf(f_cpp, "      vtkDataSet *dataset = coVtk::covise2Vtk(m_inPorts[i*nports]->getCurrentObject());\n");
        fprintf(f_cpp, "      coVtk::Flags flags = coVtk::None;\n");
        fprintf(f_cpp, "      if(dynamic_cast<vtkImageData *>(dataset)) flags = coVtk::RequireDouble;\n");
        fprintf(f_cpp, "      int type = 0;\n");
        fprintf(f_cpp, "      if(dynamic_cast<coDoTexture *>(m_inPorts[i*nports]->getCurrentObject())) type=1;\n");
        fprintf(f_cpp, "      if(dynamic_cast<coDoPixelImage *>(m_inPorts[i*nports]->getCurrentObject())) type=2;\n");
        fprintf(f_cpp, "      inputType.push_back(type);\n");
        fprintf(f_cpp, "      if(!dataset && m_inPorts[i*nports]->getCurrentObject())\n");
        fprintf(f_cpp, "      {\n");
        fprintf(f_cpp, "         sendError(\"conversion to VTK format failed for input port %%d\", i);\n");
        fprintf(f_cpp, "         return STOP_PIPELINE;\n");
        fprintf(f_cpp, "      }\n");
        fprintf(f_cpp, "      %s *vtk = dynamic_cast<%s *>(dataset);\n", setInputArg, setInputArg);
        fprintf(f_cpp, "      if(!vtk) {\n");
        fprintf(f_cpp, "         dataset->Delete();\n");
        fprintf(f_cpp, "         sendError(\"conversion to required type %s failed for input port %%d\", i);\n", setInputArg);
        fprintf(f_cpp, "         return STOP_PIPELINE;\n");
        fprintf(f_cpp, "      }\n");
        fprintf(f_cpp, "      vtkDataSetAttributes *vattr = vtk->GetPointData();\n");
        fprintf(f_cpp, "      if(coDoAbstractData *data = dynamic_cast<coDoAbstractData *>(m_inPorts[i*nports+1]->getCurrentObject())) {\n");
        fprintf(f_cpp, "         vtkDataArray *vdata = coVtk::coviseData2Vtk(dynamic_cast<coDoGrid *>(m_inPorts[i*nports]), data, flags);\n");
        fprintf(f_cpp, "         vattr->SetScalars(vdata);\n");
        fprintf(f_cpp, "      }\n");
        fprintf(f_cpp, "      if(coDoAbstractData *data = dynamic_cast<coDoAbstractData *>(m_inPorts[i*nports+2]->getCurrentObject())) {\n");
        fprintf(f_cpp, "         vtkDataArray *vdata = coVtk::coviseData2Vtk(dynamic_cast<coDoGrid *>(m_inPorts[i*nports]), data, flags);\n");
        fprintf(f_cpp, "         vattr->SetVectors(vdata);\n");
        fprintf(f_cpp, "      }\n");
        fprintf(f_cpp, "      if(coDoAbstractData *data = dynamic_cast<coDoAbstractData *>(m_inPorts[i*nports+3]->getCurrentObject())) {\n");
        fprintf(f_cpp, "         vtkDataArray *vdata = coVtk::coviseData2Vtk(dynamic_cast<coDoGrid *>(m_inPorts[i*nports]), data, coVtk::Flags(flags|coVtk::Normalize));\n");
        fprintf(f_cpp, "         vattr->SetNormals(vdata);\n");
        fprintf(f_cpp, "      }\n");
        fprintf(f_cpp, "      coDistributedObject *texobj = m_inPorts[i*nports+4]->getCurrentObject();\n");
        fprintf(f_cpp, "      int textype = 0;\n");
        fprintf(f_cpp, "      if(dynamic_cast<coDoTexture *>(texobj)) textype=1;\n");
        fprintf(f_cpp, "      if(dynamic_cast<coDoPixelImage *>(texobj)) textype=2;\n");
        fprintf(f_cpp, "      inputTexType.push_back(textype);\n");
        fprintf(f_cpp, "      if(vtk && vtk->CheckAttributes())\n");
        fprintf(f_cpp, "         sendInfo(\"VTK object integrity check failed for input port %%d\", i);\n");
        if (hasIndexedSetInput)
            fprintf(f_cpp, "      m_%sInstance->SetInput(i, vtk);\n", classname.c_str());
        else
            fprintf(f_cpp, "      m_%sInstance->SetInput(vtk);\n", classname.c_str());
        fprintf(f_cpp, "   }\n");
        fprintf(f_cpp, "\n");
    }
    fprintf(f_cpp, "   m_%sInstance->Update();\n", classname.c_str());
    fprintf(f_cpp, "   for(int i=0; i<m_%sInstance->GetNumberOfOutputPorts(); ++i)\n", classname.c_str());
    fprintf(f_cpp, "   {\n");
    fprintf(f_cpp, "      vtkDataSet *out = dynamic_cast<vtkDataSet *>(m_%sInstance->GetOutputDataObject(i));\n", classname.c_str());
    fprintf(f_cpp, "      if(inputType.size()>i*nports && (inputType[i*nports]==1 || inputType[i*nports]==2) && dynamic_cast<vtkImageData *>(out)) {\n");
    fprintf(f_cpp, "         vtkImageData *img = static_cast<vtkImageData *>(out);\n");
    fprintf(f_cpp, "         coObjInfo info = m_outPorts[i*nports]->getNewObjectInfo();\n");
    fprintf(f_cpp, "         if(inputType[i*nports]==1)\n");
    fprintf(f_cpp, "            info = coObjInfo(std::string(info.getName())+\"piximg\");\n");
    fprintf(f_cpp, "         coDoPixelImage *pix = coVtk::vtkImage2Covise(info, img);\n");
    fprintf(f_cpp, "         if(inputType[i*nports]==2)\n");
    fprintf(f_cpp, "            m_outPorts[i*nports]->setCurrentObject(pix);\n");
    fprintf(f_cpp, "         else {\n");
    fprintf(f_cpp, "            const coDoTexture *in = static_cast<const coDoTexture *>(m_inPorts[i*nports]->getCurrentObject());\n");
    fprintf(f_cpp, "            coDoTexture *tex = new coDoTexture(m_outPorts[i*nports]->getNewObjectInfo(), pix, in->getBorder(), pix->getPixelsize(), in->getLevel(), in->getNumVertices(), in->getVertices(), in->getNumCoordinates(), in->getCoordinates());\n");
    fprintf(f_cpp, "            m_outPorts[i*nports]->setCurrentObject(tex);\n");
    fprintf(f_cpp, "         }\n");
    fprintf(f_cpp, "         m_outPorts[i*nports+1]->setCurrentObject(NULL);\n");
    fprintf(f_cpp, "         m_outPorts[i*nports+2]->setCurrentObject(NULL);\n");
    fprintf(f_cpp, "         m_outPorts[i*nports+3]->setCurrentObject(NULL);\n");
    fprintf(f_cpp, "      } else {\n");
    fprintf(f_cpp, "         coDistributedObject *grid = coVtk::vtkGrid2Covise(m_outPorts[i*nports]->getNewObjectInfo(), out);\n");
    fprintf(f_cpp, "         m_outPorts[i*nports]->setCurrentObject(grid);\n");
    fprintf(f_cpp, "         coDistributedObject *scalars = coVtk::vtkData2Covise(m_outPorts[i*nports+1]->getNewObjectInfo(), out, coVtk::Scalars);\n");
    fprintf(f_cpp, "         m_outPorts[i*nports+1]->setCurrentObject(scalars);\n");
    fprintf(f_cpp, "         coDistributedObject *vectors = coVtk::vtkData2Covise(m_outPorts[i*nports+2]->getNewObjectInfo(), out, coVtk::Vectors);\n");
    fprintf(f_cpp, "         m_outPorts[i*nports+2]->setCurrentObject(vectors);\n");
    fprintf(f_cpp, "         coDistributedObject *normals = coVtk::vtkData2Covise(m_outPorts[i*nports+3]->getNewObjectInfo(), out, coVtk::Normals);\n");
    fprintf(f_cpp, "         m_outPorts[i*nports+3]->setCurrentObject(normals);\n");

    fprintf(f_cpp, "         if(inputTexType.size() > i*nports && inputTexType[i*nports]>0) {\n");
    fprintf(f_cpp, "            coDoAbstractData *texc = coVtk::vtkData2Covise(std::string(m_outPorts[i*nports+4]->getNewObjectInfo().getName())+\"texcoord\", out, coVtk::TexCoords);\n");
    fprintf(f_cpp, "            coDoTexture *tex = NULL;\n");
    fprintf(f_cpp, "            if(inputTexType[i*nports]==1) {\n");
    fprintf(f_cpp, "               const coDoTexture *in = static_cast<const coDoTexture *>(m_inPorts[i*nports+4]->getCurrentObject());\n");
    fprintf(f_cpp, "               tex = static_cast<coDoTexture *>(in->clone(m_outPorts[i*nports+4]->getNewObjectInfo()));\n");
    fprintf(f_cpp, "            } else if(inputTexType[i*nports]==2) {\n");
    fprintf(f_cpp, "               int n = texc->getNumPoints();\n");
    fprintf(f_cpp, "               float *x=NULL, *y=NULL, *z=NULL;\n");
    fprintf(f_cpp, "               if(coDoFloat *fl = dynamic_cast<coDoFloat *>(texc)) {\n");
    fprintf(f_cpp, "                  x = fl->getAddress();\n");
    fprintf(f_cpp, "               } else if(coDoVec2 *v2 = dynamic_cast<coDoVec2 *>(texc)) {\n");
    fprintf(f_cpp, "                  v2->getAddresses(&x, &y);\n");
    fprintf(f_cpp, "               } else if(coDoVec3 *v3 = dynamic_cast<coDoVec3 *>(texc)) {\n");
    fprintf(f_cpp, "                  v3->getAddresses(&x, &y, &z);\n");
    fprintf(f_cpp, "               }\n");
    fprintf(f_cpp, "               std::vector<float> ycoord;\n");
    fprintf(f_cpp, "               std::vector<int> ind;\n");
    fprintf(f_cpp, "               for(int j=0; j<n; ++j) {\n");
    fprintf(f_cpp, "                  ycoord.push_back(y ? y[j] : 0.f);\n");
    fprintf(f_cpp, "                  ind.push_back(j);\n");
    fprintf(f_cpp, "               }\n");
    fprintf(f_cpp, "               float *coord[2] = { x, &ycoord[0] };\n");
    fprintf(f_cpp, "               tex = new coDoTexture(m_outPorts[i*nports+4]->getNewObjectInfo(), static_cast<coDoPixelImage *>(m_inPorts[i*nports+4]->getCurrentObject()), 0, 4, 0, n, &ind[0], n, coord);\n");
    fprintf(f_cpp, "            }\n");
    fprintf(f_cpp, "            delete texc;\n");
    fprintf(f_cpp, "            m_outPorts[i*nports+4]->setCurrentObject(tex);\n");
    fprintf(f_cpp, "         } else {\n");
    fprintf(f_cpp, "            coDistributedObject *texc = coVtk::vtkData2Covise(m_outPorts[i*nports+4]->getNewObjectInfo(), out, coVtk::TexCoords);\n");
    fprintf(f_cpp, "            m_outPorts[i*nports+4]->setCurrentObject(texc);\n");
    fprintf(f_cpp, "         }\n");
    fprintf(f_cpp, "      }\n");
    fprintf(f_cpp, "   }\n");
    fprintf(f_cpp, "   return CONTINUE_PIPELINE;\n");
    fprintf(f_cpp, "}\n");
    fprintf(f_cpp, "\n");

    fprintf(f_cpp, "void Wrap%s::param(const char *paramName, bool /*inMapLoading*/)\n", modname);
    fprintf(f_cpp, "{\n");
    fprintf(f_cpp, "   if(!m_%sInstance)\n", classname.c_str());
    fprintf(f_cpp, "   {\n");
    fprintf(f_cpp, "      return;\n");
    fprintf(f_cpp, "   }\n");
    fprintf(f_cpp, "\n");
    for (int i = 0; i < paramfunc.size(); ++i)
    {
        fprintf(f_cpp, "   %s\n", paramfunc[i].c_str());
    }
    fprintf(f_cpp, "}\n");
    fprintf(f_cpp, "\n");

    fprintf(f_cpp, "MODULE_MAIN(VTK, Wrap%s)\n", modname);

    fprintf(f_docparam, "\\end{longtable}\n");
    fprintf(f_docparam, "\n");
    fprintf(f_docparam, "\\clearpage\n");
    fprintf(f_docparam, "\n");
    fclose(f_docparam);
    fclose(f_h);
    fclose(f_cpp);

    return 0;
}
