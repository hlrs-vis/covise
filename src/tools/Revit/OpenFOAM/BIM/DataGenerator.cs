/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

using System;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Collections.Generic;
using System.Windows.Media.Media3D;
using System.Linq;
using System.Windows.Forms;
using System.Security;
using System.Text;
using System.Threading.Tasks;
using Autodesk.Revit.DB;
using RevitApplication = Autodesk.Revit.ApplicationServices.Application;
using GeometryElement = Autodesk.Revit.DB.GeometryElement;
using GeometryOptions = Autodesk.Revit.DB.Options;
using GeometryInstance = Autodesk.Revit.DB.GeometryInstance;
using RevitView = Autodesk.Revit.DB.View;
using Material = Autodesk.Revit.DB.Material;
using utils;

namespace OpenFOAMInterface.BIM
{
    using FOAMDict = OpenFOAM.FOAMDict;
    using RunManager = OpenFOAM.RunManager;
    using RunManagerBlueCFD = OpenFOAM.RunManagerBlueCFD;
    using RunManagerSSH = OpenFOAM.RunManagerSSH;
    using RunManagerWSL = OpenFOAM.RunManagerWSL;
    public readonly ref struct FOAMFileData
    {
        public List<string> Param { get; }
        public List<string> Wall { get; }
        public List<string> Inlet { get; }
        public List<string> Outlet { get; }
        public List<string> Slip { get; }

        public FOAMFileData(in List<string> param, in List<string> wall, in List<string> inlet, in List<string> outlet, in List<string> slip)
        {
            Param = param;
            Wall = wall;
            Inlet = inlet;
            Outlet = outlet;
            Slip = slip;
        }
    }

    /// <summary>
    /// Generate triangular data and save them in a temporary file.
    /// </summary>
    public class DataGenerator
    {

        //STL-Exporter objects
        public enum GeneratorStatus { SUCCESS, FAILURE, CANCEL };
        public enum WriteStages { Wall, Inlet, Outlet, MeshResolution, AirTerminal };

        private SaveData m_Writer;
        private readonly RevitApplication m_RevitApp;
        private readonly Document m_ActiveDocument;
        private readonly RevitView m_ActiveView;
        private int m_TriangularNumber;
        private bool singleFile;
        private bool computeBoundingBox = true;
        private WriteStages WriteStage = WriteStages.Wall;

        private readonly Dictionary<string, Type> m_TypeMap = new()
        {
            { "U.", typeof(OpenFOAM.U) },
            { "p.", typeof(OpenFOAM.P) },
            { "epsilon.", typeof(OpenFOAM.Epsilon) },
            { "nut.", typeof(OpenFOAM.Nut) },
            { "k.", typeof(OpenFOAM.K) },
            { "alphat.", typeof(OpenFOAM.Alphat) },
            { "p_rgh.", typeof(OpenFOAM.P_rgh) },
            { "T.", typeof(OpenFOAM.T) }
        };
        private Type[] m_foamParamConstType = new Type[] {
            typeof(OpenFOAM.Version),
            typeof(string),
            typeof(Dictionary<string, object>),
            typeof(SaveFormat),
            typeof(Settings),
            typeof(List<string>),
            typeof(List<string>),
            typeof(List<string>) ,
            typeof(List<string>)
        };

        /// <summary>
        /// Name of the STL.
        /// </summary>
        private string m_STLName;

        /// <summary>
        /// Name of the WallInSTL
        /// </summary>
        private string m_STLWallName;

        /// <summary>
        /// RunManager for OpenFOAM-Environment
        /// </summary>
        /// </summary>
        private RunManager m_RunManager;

        /// <summary>
        /// Vector for boundingbox
        /// </summary>
        private Vector3D m_LowerEdgeVector;

        /// <summary>
        /// Vector for boundingbox
        /// </summary>
        private Vector3D m_UpperEdgeVector;

        /// <summary>
        /// Current View-Options
        /// </summary>
        private readonly GeometryOptions m_ViewOptions;

        /// <summary>
        /// Categories which will be included in STL
        /// </summary>
        private SortedDictionary<string, Category> m_Categories;

        /// <summary>
        /// Cancel GUI
        /// </summary>
        private readonly OpenFOAMCancelForm m_StlCancel = new();

        /// <summary>
        /// Materials from inlet/outlet
        /// </summary>
        private List<ElementId> m_InletOutletMaterials;

        /// <summary>
        /// Duct-Terminals
        /// </summary>
        private List<Element> m_DuctTerminalsInDoc;

        /// <summary>
        /// OpenFOAM-Dictionaries
        /// </summary>
        private List<FOAMDict> m_OpenFOAMDictionaries;

        /// <summary>
        /// Faces of the inlet/outlet for openFOAM-Simulation
        /// </summary>
        private Dictionary<KeyValuePair<string, Document>, KeyValuePair<List<Face>/*Face*/, Transform>> m_FacesInletOutlet;

        /// <summary>
        /// Number of triangles in exported Revit document.
        /// </summary>
        public int TriangularNumber
        {
            get
            {
                return m_TriangularNumber;
            }
        }

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="revit">
        /// The application object for the active instance of Autodesk Revit.
        /// </param>
        public DataGenerator(RevitApplication revitApp, Document doc)
        {
            singleFile = false;
            //initialize the member variable
            if (revitApp != null)
            {
                m_RevitApp = revitApp;
                m_ActiveDocument = doc;
                RevitView simulationView = Exporter.Instance.FindView(doc, "Simulation");
                if (simulationView == null)
                {
                    simulationView = Exporter.Instance.FindView(doc, "{3D}");
                }
                if (simulationView == null)
                {
                    simulationView = doc.ActiveView;
                }
                m_ActiveView = simulationView;

                m_ViewOptions = m_RevitApp.Create.NewGeometryOptions();
                m_ViewOptions.View = m_ActiveView;
            }
        }

        /// <summary>
        /// Initialize Runmanager as RunManagerBlueCFD or RunManagerDocker depending on the WindowsFOAMVersion that is set in Settings.
        /// </summary>
        /// <param name="casePath">Path to openFOAM-Case.</param>
        /// <returns>Status of runmanager after initialization and generation of files.</returns>
        private GeneratorStatus InitRunManager(string casePath)
        {
            switch (Exporter.Instance.settings.OpenFOAMEnvironment)
            {
                case OpenFOAMEnvironment.blueCFD:
                    m_RunManager = new RunManagerBlueCFD(casePath, Exporter.Instance.settings.OpenFOAMEnvironment);
                    break;
                case OpenFOAMEnvironment.ssh:
                    m_RunManager = new RunManagerSSH(casePath, Exporter.Instance.settings.OpenFOAMEnvironment);
                    break;
                case OpenFOAMEnvironment.wsl:
                    m_RunManager = new RunManagerWSL(casePath, Exporter.Instance.settings.OpenFOAMEnvironment);
                    break;
            }
            return m_RunManager.Status;
        }

        /// <summary>
        /// Create OpenFOAM-Folder at given path.
        /// </summary>
        /// <param name="path">location for the OpenFOAM-folder</param>
        /// <returns>Success if generation of foam case went well.</returns>
        public GeneratorStatus CreateOpenFOAMCase(string path)
        {
            List<string> minCaseFolders = new();

            //first level folders
            string constant = Path.Combine(path, "constant");
            string nullFolder = Path.Combine(path, "0");
            string system = Path.Combine(path, "system");
            string log = Path.Combine(path, "log");

            //second level folders
            string polyMesh = Path.Combine(constant, "polyMesh");
            string triSurface = Path.Combine(constant, "triSurface");

            //paths to folders
            minCaseFolders.Add(nullFolder);
            minCaseFolders.Add(log);
            minCaseFolders.Add(system);
            minCaseFolders.Add(polyMesh);
            minCaseFolders.Add(triSurface);

            //create folders
            foreach (string folder in minCaseFolders)
            {
                Directory.CreateDirectory(folder);
            }
            string zipPath = path;

            if (Exporter.Instance.settings.OpenFOAMEnvironment == OpenFOAMEnvironment.ssh)
            {
                zipPath = path + ".zip";
            }

            GeneratorStatus status = InitRunManager(zipPath);
            if (status != GeneratorStatus.SUCCESS)
            {
                return status;
            }

            //generate files
            OpenFOAM.Version version = new();
            m_OpenFOAMDictionaries = new List<FOAMDict>();

            //init folders
            InitSystemFolder(version, system);
            InitConstantFolder(version, constant);
            if (InitNullFolder(version, nullFolder) == GeneratorStatus.FAILURE)
            {
                return GeneratorStatus.FAILURE;
            }

            foreach (FOAMDict openFOAMDictionary in m_OpenFOAMDictionaries)
            {
                openFOAMDictionary.Init();
            }

            List<string> commands = new List<string>();

            //commands
            if (Exporter.Instance.settings.OpenFOAMEnvironment == OpenFOAMEnvironment.ssh)
            {
                SetupLinux(path, commands);
            }
            else
            {
                commands.Add("blockMesh");
                commands.Add("surfaceFeatureExtract");
                commands.Add("snappyHexMesh");
                commands.Add("rm -r processor*");
                commands.Add(Exporter.Instance.settings.AppSolverControlDict.ToString());
                commands.Add("rm -r processor*");
            }

            //zip file before pushing to cluster
            if (File.Exists(zipPath))
            {
                File.Delete(zipPath);
            }
            ZipFile.CreateFromDirectory(path, zipPath);

            //run commands in windows-openfoam-environment
            if (!m_RunManager.RunCommands(commands))
            {
                return GeneratorStatus.FAILURE;
            }
            return GeneratorStatus.SUCCESS;
        }

        /// <summary>
        /// Add allrun and allclean to the case folder and add corresponding command to the list.
        /// </summary>
        /// <param name="path">Path to case folder.</param>
        /// <param name="commands">List of commands.</param>
        private void SetupLinux(string path, List<string> commands)
        {
            string allrun;
            if (Exporter.Instance.settings.NumberOfSubdomains != 1)
            {
                allrun = "#!/bin/sh" +
                "\ncd ${0%/*} || exit 1    # run from this directory" +
                "\n" +
                "\n# Source tutorial run functions" +
                "\n. $WM_PROJECT_DIR/bin/tools/RunFunctions" +
                "\nrunApplication surfaceFeatureExtract" +
                "\n" +
                "\nrunApplication blockMesh" +
                "\n" +
                "\nrunApplication decomposePar -copyZero" +
                "\nrunParallel snappyHexMesh -overwrite" +
                "\n" +
                //Problem with regular allrun => bypass through recontstructParMesh and decompose the case again
                "\nrunApplication reconstructParMesh -constant" +
                "\nrm -r processor*" +
                "\nrm -rf log.decomposePar" +
                "\nrunApplication decomposePar" +
                //"\nrunParallel renumberMesh -overwrite" +
                "\nrunParallel $(getApplication)" +
                "\nrunApplication reconstructPar " + Exporter.Instance.settings.ReconstructParOption +
                "\nrunApplication foamToEnsight " +
                "\n" +
                "\n#------------------------------------------------------------------------------";
            }
            else
            {
                allrun = "#!/bin/sh" +
                "\ncd ${0%/*} || exit 1    # run from this directory" +
                "\n" +
                "\n# Source tutorial run functions" +
                "\n. $WM_PROJECT_DIR/bin/tools/RunFunctions" +
                "\n" +
                "\nrunApplication surfaceFeatureExtract" +
                "\n" +
                "\nrunApplication blockMesh" +
                "\n" +
                "\nrunApplication snappyHexMesh -overwrite" +
                "\n" +
                "\nrunApplication $(getApplication)" +
                "\n#------------------------------------------------------------------------------";
            }

            if (CreateGeneralFile(path, "Allrun.", allrun))
            {
                string allclean = "#!/bin/sh" +
                    "\ncd ${0%/*} || exit 1    # run from this directory" +
                    "\n" +
                    "\n# Source tutorial clean functions" +
                    "\n. $WM_PROJECT_DIR/bin/tools/CleanFunctions" +
                    "\n" +
                    "\ncleanCase" +
                    "\n" +
                    "\nrm -rf constant/extendedFeatureEdgeMesh > /dev/null 2>&1" +
                    "\nrm -f constant/triSurface/buildings.eMesh > /dev/null 2>&1" +
                    "\nrm -f constant/polyMesh/boundary > /dev/null 2>&1" +
                    "\n" +
                    "\n#------------------------------------------------------------------------------";

                CreateGeneralFile(path, "Allclean.", allclean);
            }
        }

        /// <summary>
        /// Creates general file in openfoam case folder.
        /// For example: Allrun, Allclean
        /// </summary>
        /// <paramref name="path"/>Path<param>ref name="path"/>
        /// <paramref name="name"/>Name of the file<paramref name="name"/>
        /// <paramref name="text"/>Text for file.<paramref name="text"/>
        /// <returns>boolean that indicates if the generation of a file succeeded.</returns>
        private bool CreateGeneralFile(string path, string name, string text)
        {
            bool succeed = true;
            string m_Path = Path.Combine(path, name);
            try
            {
                FileAttributes fileAttribute = FileAttributes.Normal;

                if (File.Exists(m_Path))
                {
                    fileAttribute = File.GetAttributes(m_Path);
                    FileAttributes tempAtt = fileAttribute & FileAttributes.ReadOnly;
                    if (FileAttributes.ReadOnly == tempAtt)
                    {
                        OpenFOAMDialogManager.ShowError(OpenFOAMInterfaceResource.ERR_FILE_READONLY);
                        return false;
                    }
                    File.Delete(m_Path);
                }

                //Create File.
                using (StreamWriter sw = new StreamWriter(m_Path))
                {
                    sw.NewLine = "\n";
                    fileAttribute = File.GetAttributes(m_Path) | fileAttribute;
                    File.SetAttributes(m_Path, fileAttribute);

                    // Add information to the file.
                    sw.Write(text);
                }
            }
            catch (Exception e)
            {
                ShowDialog(ref e);
                succeed = false;
            }
            return succeed;
        }

        /// <summary>
        /// Creates all dictionaries in systemFolder.
        /// </summary>
        /// <param name="version">Current version-object.</param>
        /// <param name="system">Path to system folder.</param>
        private void InitSystemFolder(OpenFOAM.Version version, string system)
        {
            //files in the system folder
            string blockMeshDict = Path.Combine(system, "blockMeshDict.");
            string surfaceFeatureExtractDict = Path.Combine(system, "surfaceFeatureExtractDict.");
            string decomposeParDict = Path.Combine(system, "decomposeParDict.");
            string controlDict = Path.Combine(system, "controlDict.");
            string fvSchemes = Path.Combine(system, "fvSchemes.");
            string fvSolution = Path.Combine(system, "fvSolution.");
            string meshDict = "";
            switch (Exporter.Instance.settings.Mesh)
            {
                case MeshType.Snappy:
                    {
                        meshDict = Path.Combine(system, "snappyHexMeshDict.");
                        break;
                    }
                case MeshType.cfMesh:
                    {
                        meshDict = Path.Combine(system, "meshDict.");
                        break;
                    }
            }

            //generate system dictionary objects 
            OpenFOAM.BlockMeshDict blockMeshDictionary = new(version, blockMeshDict, null, SaveFormat.ascii, m_LowerEdgeVector, m_UpperEdgeVector);
            OpenFOAM.ControlDict controlDictionary = new(version, controlDict, null, SaveFormat.ascii, null);
            OpenFOAM.SurfaceFeatureExtract surfaceFeatureExtractDictionary = new(version, surfaceFeatureExtractDict, null, SaveFormat.ascii, m_STLName);
            OpenFOAM.DecomposeParDict decomposeParDictionary = new(version, decomposeParDict, null, SaveFormat.ascii);
            OpenFOAM.FvSchemes fvSchemesDictionary = new(version, fvSchemes, null, SaveFormat.ascii);
            OpenFOAM.FvSolution fvSolutionDictionary = new(version, fvSolution, null, SaveFormat.ascii);
            OpenFOAM.SnappyHexMeshDict snappyHexMeshDictionary = new(version, meshDict, null, SaveFormat.ascii, m_STLName, m_STLWallName, m_FacesInletOutlet);

            //runmanager have to know how much cpu's should be used
            m_RunManager.NumberOfSubdomains = decomposeParDictionary.NumberOfSubdomains;

            m_OpenFOAMDictionaries.Add(blockMeshDictionary);
            m_OpenFOAMDictionaries.Add(controlDictionary);
            m_OpenFOAMDictionaries.Add(surfaceFeatureExtractDictionary);
            m_OpenFOAMDictionaries.Add(decomposeParDictionary);
            m_OpenFOAMDictionaries.Add(fvSchemesDictionary);
            m_OpenFOAMDictionaries.Add(fvSolutionDictionary);
            m_OpenFOAMDictionaries.Add(snappyHexMeshDictionary);
        }

        /// <summary>
        /// Creates all dictionaries in the nullfolder.
        /// </summary>
        /// <param name="version">Current version-object</param>
        /// <param name="nullFolder">Path to nullfolder.</param>
        /// <returns>GeneratorStatus which indicates that nothing went wrong while object instantiation.</returns>
        private GeneratorStatus InitNullFolder(OpenFOAM.Version version, string nullFolder)
        {
            List<string> paramList = new List<string>();

            //Files in nullfolder
            foreach (var param in Exporter.Instance.settings.SimulationDefault["0"] as Dictionary<string, object>)
            {
                paramList.Add(Path.Combine(nullFolder, param.Key + "."));
            }

            //Extract inlet/outlet-names
            List<string> slipNames = new List<string>();
            List<string> wallNames = new List<string>();
            List<string> inletNames = new List<string>();
            List<string> outletNames = new List<string>();
            foreach (var face in m_FacesInletOutlet)
            {
                string name = face.Key.Key.Replace(" ", "_");
                if (name.Contains("Zuluft") || name.Contains("Inlet"))
                {
                    inletNames.Add(name);
                }
                else
                {
                    outletNames.Add(name);
                }
            }
            wallNames.Add("wallSTL");

            // add ComputationalDomain in/and outlets
            if (!Exporter.Instance.settings.DomainX.IsZeroLength()) // ComputationalDomain Family instance exists
            {
                inletNames.Add("inlet");
                outletNames.Add("outlet");
                wallNames.Add("frontAndBack");
                wallNames.Add("lowerWall");
                wallNames.Add("upperWall");
            }
            else
            {
                wallNames.Add("boundingBox");
            }

            InitNullDirList(setList: Exporter.Instance.settings.MeshResolution.Keys.ToList(),
                            output: ref wallNames);
            InitNullDirList(setList: Exporter.Instance.settings.m_InletElements,
                            output: ref inletNames,
                            prefix: "Inlet_");
            InitNullDirList(setList: Exporter.Instance.settings.m_OutletElements,
                            output: ref outletNames,
                            prefix: "Outlet_");

            FOAMFileData nullFOAMFileNames = new(
                paramList,
                wallNames,
                inletNames,
                outletNames,
                slipNames
            );

            //generate Files
            return GenerateNullDirFOAMDictObj(version, nullFOAMFileNames);
        }

        /// <summary>
        /// Fill given output with names generated with element entries from setList.
        /// </summary>
        /// <param name="prefix">Prefix for entry input name.</param>
        /// <param name="postfix">Postfix for entry intput name.</param>
        /// <param name="setList">Reference to elementlist in settings.</param>
        /// <param name="output">Reference to output list.</param>
        private void InitNullDirList(in List<Element> setList, ref List<string> output, in string prefix = "", in string postfix = "")
        {
            foreach (Element entry in setList)
            {
                string name = AutodeskHelperFunctions.GenerateNameFromElement(entry);
                output.Add(prefix + name + postfix);
            }
        }

        /// <summary>
        /// Get corresponding foamfile type for given string.
        /// </summary>
        /// <param name="name">Name of constructor as string.</param>
        /// <returns>Type of object if string is in m_TypeMap else null.</returns>
        private Type GetNullDirFOAMDictType(in string name)
        {
            foreach (var keyconstructor in m_TypeMap)
            {
                if (name.Contains(keyconstructor.Key))
                {
                    return keyconstructor.Value;
                }
            }
            return null;
        }

        /// <summary>
        /// Generate FOAMFile objects.
        /// </summary>
        /// <param name="version">Version.</param>
        /// <param name="ffdn">FOAMFiledata struct that contains all names for foamfile generation.</param>
        /// <returns>Current status of generator that indicates that objects instantiation went well.</returns>
        private GeneratorStatus GenerateNullDirFOAMDictObj(in OpenFOAM.Version version, in FOAMFileData ffdn)
        {
            FOAMDict parameter;

            foreach (string nameParam in ffdn.Param)
            {
                if (nameParam.Contains("p."))
                {
                    parameter = new OpenFOAM.P("p", version, nameParam, null, SaveFormat.ascii, Exporter.Instance.settings, ffdn.Wall, ffdn.Inlet, ffdn.Outlet, ffdn.Slip);
                    m_OpenFOAMDictionaries.Add(parameter);
                    continue;
                }
                try
                {
                    var type = GetNullDirFOAMDictType(nameParam);

                    // Get the public instance constructor for type
                    ConstructorInfo foamParamConstructor = type.GetConstructor(m_foamParamConstType);
                    if (foamParamConstructor != null)
                    {
                        parameter = (FOAMDict)foamParamConstructor.Invoke(new Object[] { version, nameParam, null, SaveFormat.ascii, Exporter.Instance.settings, ffdn.Wall, ffdn.Inlet, ffdn.Outlet, ffdn.Slip });
                        m_OpenFOAMDictionaries.Add(parameter);
                    }
                    else
                    {
                        OpenFOAMDialogManager.ShowError(OpenFOAMInterfaceResource.ERR_OPENFOAM_PARAM_INVALID);
                        return GeneratorStatus.FAILURE;
                    }
                }
                catch (Exception e)
                {
                    ShowDialog(ref e);
                }
            }
            return GeneratorStatus.SUCCESS;
        }

        /// <summary>
        /// Initialize Dictionaries in constant folder.
        /// </summary>
        /// <param name="version">Version-object.</param>
        /// <param name="constantFolder">Path to constant folder.</param>
        private void InitConstantFolder(in OpenFOAM.Version version, in string constantFolder)
        {
            string transportProperties = Path.Combine(constantFolder, "transportProperties.");
            string g = Path.Combine(constantFolder, "g.");
            string turbulenceProperties = Path.Combine(constantFolder, "turbulenceProperties.");

            OpenFOAM.TransportProperties transportPropertiesDictionary = new(version, transportProperties, null, SaveFormat.ascii);
            OpenFOAM.G gDictionary = new(version, g, null, SaveFormat.ascii, Exporter.Instance.settings);
            OpenFOAM.TurbulenceProperties turbulencePropertiesDictionary = new(version, turbulenceProperties, null, SaveFormat.ascii);

            m_OpenFOAMDictionaries.Add(transportPropertiesDictionary);
            m_OpenFOAMDictionaries.Add(gDictionary);
            m_OpenFOAMDictionaries.Add(turbulencePropertiesDictionary);
        }

        /// <summary>
        /// Build a list that contains all elements of the specified category of the given document.
        /// </summary>
        /// <typeparam name="T">Specifies in which class instance should be seperated.</typeparam>
        /// <param name="doc">Active document.</param>
        /// <param name="category">BuiltInCategory from the Autodesk database.</param>
        /// <returns>List of elements with specified category instances.</returns>
        public static List<Element> GetDefaultCategoryListOfClass<T>(in Document document, in BuiltInCategory category, in string viewName)
        {
            // find the view having the same name of ActiveView.Name in active and linked model documents.
            ElementId viewId = Exporter.Instance.FindViewId(document, viewName);

            FilteredElementCollector collector = null;
            if (viewId != ElementId.InvalidElementId)
                collector = new FilteredElementCollector(document, viewId);
            else
                collector = new FilteredElementCollector(document);
            FilteredElementCollector catCollector = collector.OfCategory(category).OfClass(typeof(T));//.OfCategory(category);;
            return catCollector.ToList();
        }


        /// <summary>
        /// Get a list of ElemendId`s that represents the materials in the given element-list
        /// from the Document-Object document.
        /// </summary>
        /// <param name="doc">Document-object which will used for searching in.</param>
        /// <param name="elements">Element-List which will used for searching in.</param>
        /// <returns>List of ElementId's from the materials.</returns>
        public static List<ElementId> GetMaterialList(in List<Element> elements, in List<string> materialNames)
        {
            List<ElementId> materialIds = new List<ElementId>();
            foreach (Element elem in elements)
            {
                ICollection<ElementId> materials = elem.GetMaterialIds(false);
                foreach (ElementId id in materials)
                {
                    Material material = elem.Document.GetElement(id) as Material;
                    if (materialIds.Contains(id))
                    {
                        continue;
                    }

                    //coloring with materials differentiate from the the original materials which 
                    //will be listed in a component list (colored surface with other material)
                    foreach (string matName in materialNames)
                    {
                        if (material.Name.Equals(matName))
                        {
                            materialIds.Add(id);
                        }
                    }
                }
            }

            return materialIds;
        }

        /// <summary>
        /// Save active Revit document as STL file according to customer's settings.
        /// </summary>
        /// <param name="fileName">The name of the STL file to be saved.</param>
        /// <param name="settings">Settings for save operation.</param>
        /// <returns>Successful or failure.</returns>      
        public GeneratorStatus SaveSTLFile(in string fileName)
        {
            try
            {
                computeBoundingBox = false;
                if (Exporter.Instance.settings.DomainX.IsZeroLength())
                {
                    computeBoundingBox = true;
                }

                m_StlCancel.Show();

                // save data in certain STL file
                if (SaveFormat.binary == Exporter.Instance.settings.SaveFormat)
                {
                    m_Writer = new SaveDataAsBinary(fileName, Exporter.Instance.settings.SaveFormat);
                }
                else
                {
                    m_Writer = new SaveDataAsAscII(fileName, Exporter.Instance.settings.SaveFormat);
                }

                m_Writer.CreateFile();

                GeneratorStatus status = ScanElement(Exporter.Instance.settings.ExportRange);

                Application.DoEvents();

                if (status != GeneratorStatus.SUCCESS)
                {
                    m_StlCancel.Close();
                    return status;
                }

                if (m_StlCancel.CancelProcess == true)
                {
                    m_StlCancel.Close();
                    return GeneratorStatus.CANCEL;
                }

                if (0 == m_TriangularNumber)
                {
                    OpenFOAMDialogManager.ShowError(OpenFOAMInterfaceResource.ERR_NOSOLID);

                    m_StlCancel.Close();
                    return GeneratorStatus.FAILURE;
                }

                if (SaveFormat.binary == Exporter.Instance.settings.SaveFormat)
                {
                    // add triangular number to STL file
                    m_Writer.TriangularNumber = m_TriangularNumber;
                    m_Writer.AddTriangularNumberSection();
                }

                m_Writer.CloseFile();
            }
            catch (Exception e)
            {
                ShowDialog(ref e);
                m_StlCancel.Close();
                return GeneratorStatus.FAILURE;
            }

            m_StlCancel.Close();
            return GeneratorStatus.SUCCESS;
        }

        /// <summary>
        /// Scans all elements in the active document and creates a list of
        /// the categories of those elements.
        /// </summary>
        /// <returns>Sorted dictionary of categories.</returns>
        public SortedDictionary<string, Category> ScanCategories()
        {
            m_Categories = new SortedDictionary<string, Category>();

            // get all elements in the active document
            FilteredElementCollector filterCollector = new FilteredElementCollector(m_ActiveDocument);

            filterCollector.WhereElementIsNotElementType();

            FilteredElementIterator iterator = filterCollector.GetElementIterator();

            // create sorted dictionary of the categories of the elements
            while (iterator.MoveNext())
            {
                Element element = iterator.Current;
                if (element.Category != null)
                {
                    if (!m_Categories.ContainsKey(element.Category.Name))
                    {
                        m_Categories.Add(element.Category.Name, element.Category);
                    }
                }
            }

            return m_Categories;
        }

        /// <summary>
        /// Gets all categores in all open documents if allCategories is true
        /// or the categories of the elements in the active document if allCategories
        /// is set to false. 
        /// </summary>
        /// <param name="allCategories">True to get all categores in all open documents, 
        /// false to get the categories of the elements in the active document.</param>
        /// <returns>Sorted dictionary of categories.</returns>
        public SortedDictionary<string, Category> ScanCategories(bool allCategories)
        {
            if (!allCategories)
            {
                return ScanCategories();
            }
            else
            {
                // get and return all categories
                SortedDictionary<string, Category> sortedCategories = new SortedDictionary<string, Category>();

                // scan the active document for categories
                foreach (Category category in m_ActiveDocument.Settings.Categories)
                {

                    if (!sortedCategories.ContainsKey(category.Name))
                        sortedCategories.Add(category.Name, category);
                }

                // if linked models exist scan for categories
                List<Document> linkedDocs = GetLinkedModels();

                foreach (Document linkedDoc in linkedDocs)
                {
                    foreach (Category category in linkedDoc.Settings.Categories)
                    {
                        if (!sortedCategories.ContainsKey(category.Name))
                            sortedCategories.Add(category.Name, category);
                    }
                }
                return sortedCategories;
            }
        }

        /// <summary>
        /// Get every Element in all open documents.
        /// </summary>
        /// <param name="exportRange">
        /// The range of elements to be exported.
        /// </param>
        private GeneratorStatus ScanElement(ElementsExportRange exportRange)
        {
            List<Document> documents = new List<Document>();

            string pathSTL = m_Writer.FileName;
            string stlName = pathSTL.Substring(pathSTL.LastIndexOf("\\") + 1).Split('.')[0];
            m_STLName = stlName;

            string pathFolder = Exporter.Instance.settings.LocalCaseFolder;

            //contains all duct terminals lists of each document
            Dictionary<Document, List<Element>> terminalListOfAllDocuments = new Dictionary<Document, List<Element>>();

            // active document should be the first docuemnt in the list
            documents.Add(m_ActiveDocument);

            // figure out if we need to get linked models
            if (Exporter.Instance.settings.IncludeLinkedModels)
            {
                List<Document> linkedDocList = GetLinkedModels();
                documents.AddRange(linkedDocList);
            }

            stlName = "wallSTL";
            m_STLWallName = stlName;
            m_LowerEdgeVector = new Vector3D(1000000, 1000000, 1000000);
            m_UpperEdgeVector = new Vector3D(-1000000, -1000000, -1000000);

            m_Writer.WriteSolidName(stlName, true);
            Exporter.Instance.settings.MeshResolution.Clear(); // clear all lists before we parse the database
            Exporter.Instance.settings.m_InletElements.Clear(); // clear all lists before we parse the database
            Exporter.Instance.settings.m_OutletElements.Clear(); // clear all lists before we parse the database

            foreach (Document doc in documents)
            {
                //m_Writer.WriteSolidName("wall", true);
                FilteredElementCollector collector = null;

                if (ElementsExportRange.OnlyVisibleOnes == exportRange)
                {
                    // find the view having the same name of ActiveView.Name in active and linked model documents.
                    ElementId viewId = Exporter.Instance.FindViewId(doc, m_ActiveView.Name);

                    if (viewId != ElementId.InvalidElementId)
                        collector = new FilteredElementCollector(doc, viewId);
                    else
                        collector = new FilteredElementCollector(doc);
                }
                else
                {
                    collector = new FilteredElementCollector(doc);
                }

                //get the category list seperated via FamilyInstance in the current document
                m_DuctTerminalsInDoc = GetDefaultCategoryListOfClass<FamilyInstance>(doc, BuiltInCategory.OST_DuctTerminal, m_ActiveView.Name);
                m_InletOutletMaterials = GetMaterialList(m_DuctTerminalsInDoc, new List<string> { "Inlet", "Outlet" });

                collector.WhereElementIsNotElementType();

                //PrintOutElementNames(collector, doc);

                FilteredElementIterator iterator = collector.GetElementIterator();
                WriteStage = WriteStages.Wall;
                while (iterator.MoveNext())
                {
                    Application.DoEvents();

                    if (m_StlCancel.CancelProcess == true)
                        return GeneratorStatus.FAILURE;

                    //Element element = iterator.Current;
                    Element currentElement = iterator.Current;

                    if (currentElement.Name.Contains(Exporter.Instance.settings.OpenFOAMObjectName))
                        continue;

                    // check if element's category is in the list, if it is continue.
                    // if there are no selected categories, take anything.
                    if (Exporter.Instance.settings.SelectedCategories.Count > 0)
                    {
                        if (currentElement.Category == null)
                        {
                            continue;
                        }
                        else
                        {
                            IEnumerable<Category> cats = from cat in Exporter.Instance.settings.SelectedCategories
                                                         where cat.Id == currentElement.Category.Id
                                                         select cat;

                            if (cats.Count() == 0)
                            {
                                continue;
                            }
                        }
                    }

                    // get the GeometryElement of the element
                    GeometryElement geometry = null;
                    geometry = currentElement.get_Geometry(m_ViewOptions);

                    if (null == geometry)
                    {
                        continue;
                    }

                    if (IsGeometryInList(m_DuctTerminalsInDoc, geometry))
                    {
                        continue;
                    }

                    // get the solids in GeometryElement
                    ScanGeomElement(doc, currentElement, geometry, null, false);
                }
                terminalListOfAllDocuments.Add(doc, m_DuctTerminalsInDoc);
            }


            m_InletOutletMaterials.AddRange(GetMaterialList(Exporter.Instance.settings.m_InletElements, new List<string> { "Inlet", "Outlet" }));
            m_InletOutletMaterials.AddRange(GetMaterialList(Exporter.Instance.settings.m_OutletElements, new List<string> { "Inlet", "Outlet" }));
            m_Writer.WriteSolidName(stlName, false);
            if (!singleFile)
            {
                m_Writer.CloseFile();
            }
            WriteStage = WriteStages.AirTerminal;
            WriteAirTerminalsToSTL(terminalListOfAllDocuments, stlName);
            WriteStage = WriteStages.MeshResolution;
            WriteMeshResolutionObjectsToSTL(stlName);
            WriteStage = WriteStages.Inlet;
            WriteObjectListToSTL("Inlet_", Exporter.Instance.settings.m_InletElements);
            WriteStage = WriteStages.Outlet;
            WriteObjectListToSTL("Outlet_", Exporter.Instance.settings.m_OutletElements);
            if (singleFile)
            {
                m_Writer.CloseFile();
            }
            return GeneratorStatus.SUCCESS;
        }

        /// <summary>
        /// Checks if the list contains the given geometry.
        /// </summary>
        /// <param name="list">List of Elements in which the method search for geometry.</param>
        /// <param name="geometry">GeomtryElement that the method is searching for.</param>
        /// <return>If true, geometry is in the list.</return></returns>
        private bool IsGeometryInList(List<Element> list, GeometryElement geometry)
        {
            foreach (Element elem in list)
            {
                GeometryElement geometryDuct = elem.get_Geometry(m_ViewOptions);
                if (geometryDuct.GraphicsStyleId == geometry.GraphicsStyleId)
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Write the air terminals to STL and extract the inlet/outlet faces.
        /// </summary>
        /// <param name="dict">Contains the document as a key and the corresponding list of air terminals as value.</param>
        /// <param name="stlName">String that represents the name of the STL.</param>
        private void WriteAirTerminalsToSTL(Dictionary<Document, List<Element>> terminals, string stlName)
        {
            //close wall section
            m_FacesInletOutlet = new Dictionary<KeyValuePair<string, Document>, KeyValuePair<List<Face>/*Face*/, Transform>>();
            foreach (var elements in terminals)
            {
                foreach (Element elem in elements.Value)
                {

                    string elemName = AutodeskHelperFunctions.GenerateNameFromElement(elem);
                    // save data in certain STL file
                    if (!singleFile)
                    {
                        if (SaveFormat.binary == Exporter.Instance.settings.SaveFormat)
                        {
                            m_Writer = new SaveDataAsBinary(elemName + ".stl", Exporter.Instance.settings.SaveFormat);
                        }
                        else
                        {
                            m_Writer = new SaveDataAsAscII(elemName + ".stl", Exporter.Instance.settings.SaveFormat);
                        }

                        m_Writer.CreateFile();
                    }

                    m_Writer.WriteSolidName(elemName, true);
                    GeometryElement geometry = null;
                    geometry = elem.get_Geometry(m_ViewOptions);
                    if (null == geometry)
                    {
                        continue;
                    }

                    //need transform for inlet/outlet-face and the face itself.
                    KeyValuePair<List<Face>/*Face*/, Transform> inletOutlet = ExtractMaterialFaces(elements.Key, geometry, null, m_InletOutletMaterials);
                    KeyValuePair<string, Document> inletOutletID = new KeyValuePair<string, Document>(
                        AutodeskHelperFunctions.GenerateNameFromElement(elem), elements.Key);
                    m_FacesInletOutlet.Add(inletOutletID, inletOutlet);
                    m_Writer.WriteSolidName(elemName, false);
                    if (!singleFile)
                    {
                        m_Writer.CloseFile();
                    }
                }
            }


            if (m_FacesInletOutlet.Count == 0)
            {
                return;
            }

            //begin to write inlet/oulet-face to stl
            foreach (var face in m_FacesInletOutlet)
            {
                //face.Key.Key = Name + ID
                if (face.Value.Key == null) // skip emptry in/outFaces (element does not contain any Outlet or Inlet materials, the whole Geometry will be used
                    break;

                // save data in certain STL file
                if (!singleFile)
                {
                    if (SaveFormat.binary == Exporter.Instance.settings.SaveFormat)
                    {
                        m_Writer = new SaveDataAsBinary(face.Key.Key + ".stl", Exporter.Instance.settings.SaveFormat);
                    }
                    else
                    {
                        m_Writer = new SaveDataAsAscII(face.Key.Key + ".stl", Exporter.Instance.settings.SaveFormat);
                    }

                    m_Writer.CreateFile();
                }

                m_Writer.WriteSolidName(face.Key.Key, true);
                foreach (Face currentFace in face.Value.Key)
                {
                    Mesh mesh = currentFace.Triangulate();
                    if (mesh == null)
                    {
                        continue;
                    }
                    //face.Key.Value = Document ; face.Value.Value = transform
                    WriteFaceToSTL(face.Key.Value, mesh, currentFace, face.Value.Value);
                }
                m_Writer.WriteSolidName(face.Key.Key, false);
                if (!singleFile)
                {
                    m_Writer.CloseFile();
                }
            }
        }

        /// <summary>
        /// Write objects with parameter "Mesh Resolution" to stl.
        /// </summary>
        /// <param name="stlName">String that represents the name of the STL.</param>
        private void WriteMeshResolutionObjectsToSTL(string stlName)
        {
            WriteStage = WriteStages.MeshResolution;
            foreach (var element in Exporter.Instance.settings.MeshResolution.Keys)
            {
                GeometryElement geometry = null;
                geometry = element.get_Geometry(m_ViewOptions);
                if (null == geometry)
                {
                    continue;
                }
                if (!IsGeometryInList(m_DuctTerminalsInDoc, geometry))
                {
                    string elemName = AutodeskHelperFunctions.GenerateNameFromElement(element);
                    if (!singleFile)
                    {
                        // save data in certain STL file
                        if (SaveFormat.binary == Exporter.Instance.settings.SaveFormat)
                        {
                            m_Writer = new SaveDataAsBinary(elemName + ".stl", Exporter.Instance.settings.SaveFormat);
                        }
                        else
                        {
                            m_Writer = new SaveDataAsAscII(elemName + ".stl", Exporter.Instance.settings.SaveFormat);
                        }

                        m_Writer.CreateFile();
                    }

                    m_Writer.WriteSolidName(elemName, true);

                    //write geometry as faces to stl-file
                    ScanGeomElement(m_ActiveDocument, element, geometry, null, false);
                    m_Writer.WriteSolidName(elemName, false);
                    if (!singleFile)
                    {
                        m_Writer.CloseFile();
                    }
                }
            }
        }
        private void WriteObjectListToSTL(string prefix, List<Element> elements)
        {
            foreach (var element in elements)
            {
                GeometryElement geometry = null;
                geometry = element.get_Geometry(m_ViewOptions);
                if (null == geometry)
                {
                    continue;
                }
                if (!IsGeometryInList(m_DuctTerminalsInDoc, geometry))
                {
                    string elemName = prefix + AutodeskHelperFunctions.GenerateNameFromElement(element);
                    if (!singleFile)
                    {
                        // save data in certain STL file
                        if (SaveFormat.binary == Exporter.Instance.settings.SaveFormat)
                        {
                            m_Writer = new SaveDataAsBinary(elemName + ".stl", Exporter.Instance.settings.SaveFormat);
                        }
                        else
                        {
                            m_Writer = new SaveDataAsAscII(elemName + ".stl", Exporter.Instance.settings.SaveFormat);
                        }

                        m_Writer.CreateFile();
                    }

                    m_Writer.WriteSolidName(elemName, true);

                    //write geometry as faces to stl-file
                    ScanGeomElement(m_ActiveDocument, element, geometry, null, false);
                    m_Writer.WriteSolidName(elemName, false);
                    if (!singleFile)
                    {
                        m_Writer.CloseFile();
                    }
                }
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="document"></param>
        /// <param name="geometry"></param>
        /// <param name="transform"></param>
        /// <param name="materialList"></param>
        /// <returns></returns>
        private KeyValuePair<List<Face>/*Face*/, Transform> ExtractMaterialFaces(Document document, GeometryElement geometry,
            Transform transform, List<ElementId> materialList)
        {
            KeyValuePair<List<Face>/*Face*/, Transform> face = new KeyValuePair<List<Face>/*Face*/, Transform>();
            foreach (GeometryObject gObject in geometry)
            {
                Solid solid = gObject as Solid;
                if (null != solid)
                {
                    KeyValuePair<List<Face>/*Face*/, Transform> keyValuePair = new KeyValuePair<List<Face>/*Face*/, Transform>
                        (ScanForMaterialFace(document, solid, transform, materialList), transform);
                    if (keyValuePair.Key.Count > 0/*!= null*/)
                    {
                        face = keyValuePair;
                        //break;
                    }
                    continue;
                }

                // if the type of the geometric primitive is instance
                GeometryInstance instance = gObject as GeometryInstance;
                if (null != instance)
                {
                    Transform newTransform;
                    if (null == transform)
                    {
                        newTransform = instance.Transform;
                    }
                    else
                    {
                        newTransform = transform.Multiply(instance.Transform);  // get a transformation of the affine 3-space
                    }
                    face = ExtractMaterialFaces(document, instance.SymbolGeometry, newTransform, materialList);
                    break;
                }

                GeometryElement geomElement = gObject as GeometryElement;
                if (null != geomElement)
                {
                    face = ExtractMaterialFaces(document, geomElement, transform, materialList);
                    break;
                }
            }
            return face;
        }

        /// <summary>
        /// Scans for Inlet/Outlet-face and returns it.
        /// </summary>
        /// <param name="document">Current cocument in which the geometry is included.</param>
        /// <param name="solid">Solid object that includes the inlet/outlet.</param>
        /// <param name="transform">The transformation.</param>
        /// <returns>Inlet/Outlet as a Face-object.</returns>
        private List<Face>/*Face*/ ScanForMaterialFace(Document document, Solid solid, Transform transform, List<ElementId> materialList)
        {
            // Face faceItem = null;
            List<Face> faceItems = new List<Face>();
            // a solid has many faces
            FaceArray faces = solid.Faces;
            if (0 == faces.Size)
            {
                //return faceItem;
                return faceItems;
            }
            foreach (Face face in faces)
            {
                if (face == null)
                {
                    continue;
                }
                if (face.Visibility != Visibility.Visible)
                {
                    continue;
                }

                Mesh mesh = face.Triangulate();
                if (null == mesh)
                {
                    continue;
                }

                m_TriangularNumber += mesh.NumTriangles;
                if (materialList.Contains(face.MaterialElementId))
                {
                    //faceItem = face;
                    faceItems.Add(face);
                    continue;
                }

                //if face is not a inlet/outlet write it to the wall section of the stl.
                WriteFaceToSTL(document, mesh, face, transform);
            }
            return faceItems/*faceItem*/;//valuePair;
        }

        /// <summary>
        /// Extract solids of the given geometry. Therefore the GeometryObjects needs to be converted into Solid. 
        /// </summary>
        /// <param name="document">Current document in which the geometry is included.</param>
        /// <param name="geometry">The geometry that contains the inlet/outlet.</param>
        /// <param name="transform">Specifies the transformation of the geometry.</param>
        /// <param name="solids">List solids will be insterted.</param>
        /// <returns>Solid list.</returns>
        public static void ExtractSolidList(Document document, GeometryElement geometry, Transform transform, List<Solid> solids)
        {
            foreach (GeometryObject gObject in geometry)
            {
                Solid solid = gObject as Solid;
                if (null != solid)
                {
                    solids.Add(solid);
                    continue;
                }

                // if the type of the geometric primitive is instance
                GeometryInstance instance = gObject as GeometryInstance;
                if (null != instance)
                {
                    Transform newTransform;
                    if (null == transform)
                    {
                        newTransform = instance.Transform;
                    }
                    else
                    {
                        newTransform = transform.Multiply(instance.Transform);  // get a transformation of the affine 3-space
                    }
                    ExtractSolidList(document, instance.SymbolGeometry, newTransform, solids);
                    continue;
                }

                GeometryElement geomElement = gObject as GeometryElement;
                if (null != geomElement)
                {
                    ExtractSolidList(document, instance.SymbolGeometry, transform, solids);
                    continue;
                }
            }
        }

        /// <summary>
        /// Get the faceNormal of the face from the solid that has a material from the list.
        /// </summary>
        /// <param name="materialIds">MaterialList that will be checked for.</param>
        /// <param name="faceNormal">Reference of the face normal.</param>
        /// <param name="solid">Solid that will be checked.</param>
        /// <returns>Face normal as XYZ object.</returns>
        public static List<Face>/*Face*/ GetFace(List<ElementId> materialIds, Solid solid)
        {
            // a solid has many faces
            FaceArray faces = solid.Faces;
            List<Face> materialFaces = new List<Face>();
            if (0 == faces.Size)
            {
                return /*null*/materialFaces;
            }

            foreach (Face face in faces)
            {
                if (face == null)
                {
                    continue;
                }
                if (face.Visibility != Visibility.Visible)
                {
                    continue;
                }
                if (materialIds.Contains(face.MaterialElementId))
                {
                    //return face;
                    materialFaces.Add(face);
                }
            }
            return materialFaces/*null*/;
        }


        /// <summary>
        /// Scan GeometryElement to collect triangles.
        /// </summary>
        /// <param name="geometry">The geometry element.</param>
        /// <param name="trf">The transformation.</param>
        private void ScanGeomElement(Document document, Element currentElement, GeometryElement geometry, Transform transform, bool hasMeshResolution)
        {
            bool hasInlet = false;
            bool hasOutlet = false;

            FamilyInstance instance = currentElement as FamilyInstance;
            if (instance != null)
            {
                int meshResolution = Settings.getInt(instance, "Mesh Resolution");
                if (meshResolution > 0)
                {
                    if (hasMeshResolution == false) // don't add an instance twice if it is hierarchical
                    {
                        if (WriteStage == WriteStages.Wall) // only in first pass we collect the Mesh Resolution objects
                        {
                            Exporter.Instance.settings.MeshResolution.Add(instance, meshResolution);
                        }
                        hasMeshResolution = true;
                    }
                }
            }

            //get all geometric primitives contained in the GeometryElement
            foreach (GeometryObject gObject in geometry)
            {

                // continue recursing the hierarchy

                // if the type of the geometric primitive is instance
                GeometryInstance geomInstance = gObject as GeometryInstance;
                GeometryElement geomElement = gObject as GeometryElement;
                Solid solid = gObject as Solid;
                if (geomInstance != null)
                {
                    ScanGeometryInstance(document, currentElement, geomInstance, transform, hasMeshResolution);
                }
                else if (null != geomElement)
                {
                    ScanGeomElement(document, currentElement, geomElement, transform, hasMeshResolution);
                }
                else // write out the geometry to the current stl file
                {
                    bool isInlet = false;
                    bool isOutlet = false;
                    bool visible = true;
                    GraphicsStyle graphicsStyle = document.GetElement(gObject.GraphicsStyleId) as GraphicsStyle;
                    if (graphicsStyle != null)
                    {
                        if (graphicsStyle.Name.Length > 6 && graphicsStyle.Name.Substring(graphicsStyle.Name.Length - 6) == "_Inlet")
                        {
                            hasInlet = true;
                            isInlet = true;
                        }
                        else if (graphicsStyle.Name.Length > 7 && graphicsStyle.Name.Substring(graphicsStyle.Name.Length - 7) == "_Outlet")
                        {
                            hasOutlet = true;
                            isOutlet = true;
                        }
                        else if (graphicsStyle.Name == "HelperObject")
                        {
                            visible = false;
                        }
                    }
                    if (visible && (isInlet == true && WriteStage == WriteStages.Inlet || isOutlet == true && WriteStage == WriteStages.Outlet || WriteStage == WriteStages.Wall && !isInlet && !isOutlet && hasMeshResolution == false || WriteStage == WriteStages.MeshResolution && hasMeshResolution == true && !isInlet && !isOutlet))
                    {
                        // if the type of the geometric primitive is Solid
                        if (null != solid)
                        {
                            ScanSolid(document, solid, transform);
                            continue;
                        }
                        else if (gObject is Mesh)
                        {
                            WriteFaceToSTL(document, gObject as Mesh, null, transform);
                        }
                    }
                }


            }
            if (hasInlet)
            {
                if (!Exporter.Instance.settings.m_InletElements.Contains(currentElement))
                    Exporter.Instance.settings.m_InletElements.Add(currentElement);
            }
            if (hasOutlet)
            {
                if (!Exporter.Instance.settings.m_OutletElements.Contains(currentElement))
                    Exporter.Instance.settings.m_OutletElements.Add(currentElement);
            }
        }

        /// <summary>
        /// Scan GeometryInstance to collect triangles.
        /// </summary>
        /// <param name="instance">The geometry instance.</param>
        /// <param name="trf">The transformation.</param>
        private void ScanGeometryInstance(Document document, Element currentElement, GeometryInstance instance, Transform transform, bool hasMeshResolution)
        {
            GeometryElement instanceGeometry = instance.SymbolGeometry;
            if (null == instanceGeometry)
            {
                return;
            }
            Transform newTransform;
            if (null == transform)
            {
                newTransform = instance.Transform;
            }
            else
            {
                newTransform = transform.Multiply(instance.Transform);	// get a transformation of the affine 3-space
            }
            // get all geometric primitives contained in the GeometryElement
            ScanGeomElement(document, currentElement, instanceGeometry, newTransform, hasMeshResolution);
        }

        /// <summary>
        /// Scan Solid to collect triangles.
        /// </summary>
        /// <param name="solid">The solid.</param>
        /// <param name="trf">The transformation.</param>
        private void ScanSolid(Document document, Solid solid, Transform transform)
        {
            GetTriangular(document, solid, transform);	// get triangles in the solid
        }

        /// <summary>
        /// Get triangles in a solid with transform.
        /// </summary>
        /// <param name="solid">The solid contains triangulars</param>
        /// <param name="transform">The transformation.</param>
        private void GetTriangular(Document document, Solid solid, Transform transform)
        {
            // a solid has many faces
            FaceArray faces = solid.Faces;
            //bool hasTransform = (null != transform);
            if (0 == faces.Size)
            {
                return;
            }

            foreach (Face face in faces)
            {
                if (face.Visibility != Visibility.Visible)
                {
                    continue;
                }
                Mesh mesh = face.Triangulate();
                if (null == mesh)
                {
                    continue;
                }
                m_TriangularNumber += mesh.NumTriangles;
                // write face to stl file
                // a face has a mesh, all meshes are made of triangles
                WriteFaceToSTL(document, mesh, face, transform);
            }
        }

        /// <summary>
        /// Write face to stl file, a face has a mesh, all meshes are made of triangles.
        /// </summary>
        /// <param name="document">Document in which the mesh and face is included.</param>
        /// <param name="mesh">Mesh of the face.</param>
        /// <param name="face">optional Face which the method writes to stl.</param>
        /// <param name="transform">Specifies transformation of the face.</param>
        private void WriteFaceToSTL(Document document, Mesh mesh, Face face, Transform transform)
        {
            bool hasTransform = null != transform;
            for (int ii = 0; ii < mesh.NumTriangles; ii++)
            {
                MeshTriangle triangular = mesh.get_Triangle(ii);
                double[] xyz = new double[9];
                XYZ normal = new XYZ();
                try
                {
                    XYZ[] triPnts = new XYZ[3];
                    for (int n = 0; n < 3; ++n)
                    {
                        double x, y, z;
                        XYZ point = triangular.get_Vertex(n);
                        if (hasTransform)
                        {
                            point = transform.OfPoint(point);
                        }
                        if (Exporter.Instance.settings.ExportSharedCoordinates)
                        {
                            ProjectPosition ps = document.ActiveProjectLocation.GetProjectPosition(point);
                            x = ps.EastWest;
                            y = ps.NorthSouth;
                            z = ps.Elevation;
                        }
                        else
                        {
                            x = point.X;
                            y = point.Y;
                            z = point.Z;
                        }

                        xyz[3 * n] = x * 0.304799999536704; //UnitUtils.ConvertFromInternalUnits(x, UnitTypeId.Meters);
                        xyz[3 * n + 1] = y * 0.304799999536704; //UnitUtils.ConvertFromInternalUnits(y, UnitTypeId.Meters);
                        xyz[3 * n + 2] = z * 0.304799999536704; //UnitUtils.ConvertFromInternalUnits(z, UnitTypeId.Meters);
                        /* else
                         {
                             xyz[3 * n] = x;
                             xyz[3 * n + 1] = y;
                             xyz[3 * n + 2] = z;
                         }*/

                        var mypoint = new XYZ(xyz[3 * n], xyz[3 * n + 1], xyz[3 * n + 2]);
                        if (computeBoundingBox)
                        {
                            IsEdgeVectorForBoundary(mypoint);
                        }
                        triPnts[n] = mypoint;
                    }

                    XYZ pnt1 = triPnts[1] - triPnts[0];
                    normal = pnt1.CrossProduct(triPnts[2] - triPnts[1]);
                }
                catch (Exception ex)
                {
                    m_TriangularNumber--;
                    ShowDialog(ref ex);
                    continue;
                }

                if (face != null && m_Writer is SaveDataAsBinary && Exporter.Instance.settings.ExportColor)
                {
                    Material material = document.GetElement(face.MaterialElementId) as Material;
                    if (material != null)
                        ((SaveDataAsBinary)m_Writer).Color = material.Color;
                }
                m_Writer.WriteSection(normal, xyz);

            }
        }

        /// <summary>
        /// Checks if the given vector is bigger or smaller than m_LowerEdgeVector and m_UpperEdgeVector.
        /// </summary>
        /// <param name="xyz">Point in 3d-Space.</param>
        private void IsEdgeVectorForBoundary(XYZ xyz)
        {
            if (xyz.X < m_LowerEdgeVector.X)
            {
                m_LowerEdgeVector.X = xyz.X;
            }
            else if (xyz.Y < m_LowerEdgeVector.Y)
            {
                m_LowerEdgeVector.Y = xyz.Y;
            }
            else if (xyz.Z < m_LowerEdgeVector.Z)
            {
                m_LowerEdgeVector.Z = xyz.Z;
            }

            if (xyz.X > m_UpperEdgeVector.X)
            {
                m_UpperEdgeVector.X = xyz.X;
            }
            else if (xyz.Y > m_UpperEdgeVector.Y)
            {
                m_UpperEdgeVector.Y = xyz.Y;
            }
            else if (xyz.Z > m_UpperEdgeVector.Z)
            {
                m_UpperEdgeVector.Z = xyz.Z;
            }
        }

        /// <summary>
        /// Scans and returns the documents linked to the current model.
        /// </summary>
        /// <returns>List of linked documents.</returns>
        private List<Document> GetLinkedModels()
        {
            List<Document> linkedDocs = new List<Document>();

            try
            {
                // scan the current model looking for Revit links
                List<Element> linkedElements = FindLinkedModelElements();

                foreach (Element linkedElem in linkedElements)
                {
                    RevitLinkType linkType = linkedElem as RevitLinkType;

                    if (linkType != null)
                    {
                        // now look that up in the open documents
                        foreach (Document openedDoc in m_RevitApp.Documents)
                        {
                            if (Path.GetFileNameWithoutExtension(openedDoc.Title).ToUpper() == Path.GetFileNameWithoutExtension(linkType.Name).ToUpper())
                                linkedDocs.Add(openedDoc);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                ShowDialog(ref ex);
            }

            return linkedDocs;

        }

        /// <summary>
        /// Scan model and return linked model elements.
        /// </summary>
        /// <returns>List of linked model elements.</returns>
        private List<Element> FindLinkedModelElements()
        {
            Document doc = m_ActiveDocument;

            FilteredElementCollector linksCollector = new FilteredElementCollector(doc);
            List<Element> linkElements = linksCollector.WherePasses(new ElementCategoryFilter(BuiltInCategory.OST_RvtLinks)).ToList();

            FilteredElementCollector familySymbolCollector = new FilteredElementCollector(doc);
            linkElements.AddRange(familySymbolCollector.OfClass(typeof(FamilySymbol)).ToList());

            return linkElements;
        }

        /// <summary>
        /// Shows error dialog for corresponding exception.
        /// </summary>
        /// <param name="e">Catched exception.</param>
        private static void ShowDialog(ref Exception e)
        {
            switch (e)
            {
                case IOException:
                    OpenFOAMDialogManager.ShowError(OpenFOAMInterfaceResource.ERR_IO_EXCEPTION);
                    break;
                case SecurityException:
                    OpenFOAMDialogManager.ShowError(OpenFOAMInterfaceResource.ERR_SECURITY_EXCEPTION);
                    break;
                default:
                    OpenFOAMDialogManager.ShowError(OpenFOAMInterfaceResource.ERR_EXCEPTION);
                    break;
            }
        }
    }
}

namespace utils
{
    /// <summary>
    /// Unicode utizer.
    /// Source: http://codepad.org/dUMpGlgg
    /// </summary>
    public static class UnicodeNormalizer
    {
        /// <summary>
        /// Map for critical chars.
        /// </summary>
        private static Dictionary<char, string> charmap = new Dictionary<char, string>() {
            {'À', "A"}, {'Á', "A"}, {'Â', "A"}, {'Ã', "A"}, {'Ä', "Ae"}, {'Å', "A"}, {'Æ', "Ae"},
            {'Ç', "C"},
            {'È', "E"}, {'É', "E"}, {'Ê', "E"}, {'Ë', "E"},
            {'Ì', "I"}, {'Í', "I"}, {'Î', "I"}, {'Ï', "I"},
            {'Ð', "Dh"}, {'Þ', "Th"},
            {'Ñ', "N"},
            {'Ò', "O"}, {'Ó', "O"}, {'Ô', "O"}, {'Õ', "O"}, {'Ö', "Oe"}, {'Ø', "Oe"},
            {'Ù', "U"}, {'Ú', "U"}, {'Û', "U"}, {'Ü', "Ue"},
            {'Ý', "Y"},
            {'ß', "ss"},
            {'à', "a"}, {'á', "a"}, {'â', "a"}, {'ã', "a"}, {'ä', "ae"}, {'å', "a"}, {'æ', "ae"},
            {'ç', "c"},
            {'è', "e"}, {'é', "e"}, {'ê', "e"}, {'ë', "e"},
            {'ì', "i"}, {'í', "i"}, {'î', "i"}, {'ï', "i"},
            {'ð', "dh"}, {'þ', "th"},
            {'ñ', "n"},
            {'ò', "o"}, {'ó', "o"}, {'ô', "o"}, {'õ', "o"}, {'ö', "oe"}, {'ø', "oe"},
            {'ù', "u"}, {'ú', "u"}, {'û', "u"}, {'ü', "ue"},
            {'ý', "y"}, {'ÿ', "y"}
        };

        public static Dictionary<char, string> Charmap { get => charmap; set => charmap = value; }

        /// <summary>
        /// Substitute critical chars with unicode conform chars.
        /// </summary>
        /// <param name="text">String that will be used for substitution.</param>
        /// <returns>string with substitute critical chars.</returns>
        public static string Normalize(this string text)
        {
            return text.Aggregate(
              new StringBuilder(),
              (sb, c) =>
              {
                  string r;
                  if (Charmap.TryGetValue(c, out r))
                  {
                      return sb.Append(r);
                  }
                  return sb.Append(c);
              }).ToString();
        }
    }

    /// <summary>
    /// General coverting function.
    /// </summary>
    public static class ConverterUtil
    {
        /// <summary>
        /// Convert-Assist-Function.
        /// </summary>
        /// <typeparam name="T">Type value will be converted in.</typeparam>
        /// <typeparam name="U">Initial type</typeparam>
        /// <param name="value">Value to convert.</param>
        /// <returns>Converted value.</returns>
        public static T ConvertValue<T, U>(U value)
        {
            return (T)Convert.ChangeType(value, typeof(T));
        }
    }

    /// <summary>
    /// Call Window.ShowDialog asynchron extension.
    /// </summary>
    public static class ShowDialogAsyncExt
    {
        /// <summary>
        /// ExtensionMethod for asynchronous use of showDialog().
        /// Source:https://stackoverflow.com/questions/33406939/async-showdialog/43420090#43420090
        /// </summary>
        /// <param name="this">Windows form object.</param>
        /// <returns>DialogResult in Task.</returns>
        public static async Task<DialogResult> ShowDialogAsync(this System.Windows.Forms.Form @this)
        {
            await Task.Yield();
            if (@this.IsDisposed)
                return DialogResult.OK;
            return @this.ShowDialog();
        }
    }

    /// <summary>
    /// Helper functions for this implementation.
    /// </summary>
    public static class AutodeskHelperFunctions
    {
        /// <summary>
        /// Generate a name from given element.
        /// </summary>
        /// <param name="element">Element object.</param>
        /// <returns>String with name as as "familyName + familyInstanceName + elementId".</returns>
        public static string GenerateNameFromElement(Element element)
        {
            FamilyInstance instance = element as FamilyInstance;
            string name = instance.Symbol.Family.Name.Replace(' ', '_') + "_" + instance.Name.Replace(' ', '_') + "_" + element.Id;
            name = UnicodeNormalizer.Normalize(name);
            return "Terminal_" + name;
        }
    }
}
