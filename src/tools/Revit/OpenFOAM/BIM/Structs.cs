using Autodesk.Revit.DB;
using System.Windows.Media.Media3D;
using System.Collections.Generic;
using System.Collections;
using System.Windows;
using System;

namespace OpenFOAMInterface.BIM.Structs
{
    using Enums;

    namespace General
    {
        using FOAM;

        /// <summary>
        /// SSH struct contains all informations about the tunnel-connection.
        /// </summary>
        readonly public struct SSH
        {
            private static SSH def = new SSH();
            public static ref readonly SSH Default => ref def;
            public SSH() : this("username", "hostname", "openfoam alias", "path/to/compute/dir/on/server",
                true, false, true, 22, "eval salloc -n 16")
            { }

            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="_user">The user to login.</param>
            /// <param name="_ip">IP of the server.</param>
            /// <param name="_alias">Alias for starting openfoam.</param>
            /// <param name="_caseFolder">Casefolder on server.</param>
            /// <param name="_download">if true, case folder will be downloaded from server after simulation.</param>
            /// <param name="_delete">if true, case folder will be deleted after simulation.</param>
            /// <param name="_port">SSH Port.</param>
            /// <param name="_slurmCommand">Slurm command specify tags.</param>
            /// <param name="_slurm">use slurm.</param>
            // public SSH(string user, string ip, string alias, string caseFolder, bool download, bool delete, bool slurm, int port, string slurmCommand)
            public SSH(in string user, in string ip, in string alias, in string caseFolder, bool download, bool delete, bool slurm, int port, in string slurmCommand)
            {
                this.User = user;
                this.ServerIP = ip;
                this.OfAlias = alias;
                this.ServerCaseFolder = caseFolder;
                this.Download = download;
                this.Delete = delete;
                this.Slurm = slurm;
                this.Port = port;
                this.SlurmCommand = slurmCommand;
            }

            /// <summary>
            /// Username.
            /// </summary>
            public string User { get; }

            /// <summary>
            /// IP of the server (/local computer-name)
            /// </summary>
            public string ServerIP { get; }

            /// <summary>
            /// Alias to start openFOAM-Environment on the server.
            /// </summary>
            public string OfAlias { get; }

            /// <summary>
            /// Folder on server openfoam case will be copied to.
            /// </summary>
            public string ServerCaseFolder { get; }

            /// <summary>
            /// Threads used.
            /// </summary>
            public string SlurmCommand { get; }

            /// <summary>
            /// Port server.
            /// </summary>
            public int Port { get; }

            /// <summary>
            /// Download after simulation.
            /// </summary>
            public bool Download { get; }

            /// <summary>
            /// Delete after simulation.
            /// </summary>
            public bool Delete { get; }

            /// <summary>
            /// Use slurm.
            /// </summary>
            public bool Slurm { get; }

            /// <summary>
            /// Connection string.
            /// </summary>
            /// <returns>user + @ + serverIP as string.</returns>
            public string ConnectionString() => User + "@" + ServerIP;
        }

        /// <summary>
        /// Struct for intializing Settings variables.
        /// </summary>
        readonly public struct SettingsParameter
        {
            private static SettingsParameter def = new SettingsParameter();
            public static ref readonly SettingsParameter Default => ref def;
            public SettingsParameter() : this(SaveFormat.ascii, ElementsExportRange.OnlyVisibleOnes, ControlDictParameters.Default, true, false, false) { }
            public SettingsParameter(in SaveFormat format, in ElementsExportRange export, in ControlDictParameters control, bool includeLinkedModels, bool exportColor, bool exportSharedCoordinates)
            {
                this.Format = format;
                this.ExportRange = export;
                this.ControlDict = control;
                this.IncludeLinkedModels = includeLinkedModels;
                this.ExportColor = exportColor;
                this.ExportSharedCoordinates = exportSharedCoordinates;
            }

            /// <summary>
            /// Struct ControlDictParamertes.
            /// </summary>
            public ControlDictParameters ControlDict { get; }

            /// <summary>
            /// SaveFormat enum
            /// </summary>
            public SaveFormat Format { get; }

            /// <summary>
            /// ExportRange enum.
            /// </summary>
            public ElementsExportRange ExportRange { get; }

            /// <summary>
            /// IncludedLinkedModels.
            /// </summary>
            public bool IncludeLinkedModels { get; }

            /// <summary>
            /// ExportColor enum.
            /// </summary>
            public bool ExportColor { get; }

            /// <summary>
            /// ExportSharedCoordinater for STL.
            /// </summary>
            public bool ExportSharedCoordinates { get; }

            public readonly override string ToString() => $"{ControlDict.ToString()}, {Format}, {ExportRange}, {IncludeLinkedModels}, {ExportColor}, {ExportSharedCoordinates}";
        }
    }

    namespace FOAM
    {
        using Model;

        /// <summary>
        /// Struct for controldict entries.
        /// </summary>
        public struct ControlDictParameters
        {
            private static ControlDictParameters def = new ControlDictParameters();
            public static ref readonly ControlDictParameters Default => ref def;

            /// <summary>
            /// Default constructor ControlDictParameters.
            /// </summary>
            public ControlDictParameters() : this(
                SolverControlDict.buoyantBoussinesqSimpleFoam,
                StartFrom.latestTime,
                StopAt.endTime,
                WriteControl.timeStep,
                WriteFormat.ascii,
                WriteCompression.off,
                TimeFormat.general, false, 0, 101, 1, 100, 2, 8, 7, 4)
            { }

            public ControlDictParameters(in SolverControlDict solver, in StartFrom from, in StopAt at,
                                         in WriteControl writeControl, in WriteFormat writeFormat, in WriteCompression writeCompression,
                                         in TimeFormat time, bool runTimeModifiable, double start, double end,
                                         double deltaT, double writeInterval, double purgeWrite,
                                         double writePrecision, double timePrecision, int numberOfSubdomains)
            {
                _appControlDictSolver = solver;
                _writeFromat = writeFormat;
                _endTime = end;
                _writeInterval = writeInterval;
                _numberOfSubdomains = numberOfSubdomains;
                this.StartFrom = from;
                this.StopAt = at;
                this.WriteControl = writeControl;
                this.WriteCompression = writeCompression;
                this.TimeFormat = time;
                this.RunTimeModifiable = runTimeModifiable;
                this.StartTime = start;
                this.DeltaT = deltaT;
                this.PurgeWrite = purgeWrite;
                this.WritePrecision = writePrecision;
                this.TimePrecision = timePrecision;
            }

            private SolverControlDict _appControlDictSolver;
            public SolverControlDict AppControlDictSolver { readonly get => _appControlDictSolver; set => _appControlDictSolver = value; }

            private WriteFormat _writeFromat;
            /// <summary>
            /// ASCII or Binary.
            /// </summary>
            public WriteFormat WriteFormat { readonly get => _writeFromat; set => _writeFromat = value; }

            private double _endTime;
            /// <summary>
            /// End time for ControlDict.
            /// </summary>
            public double EndTime { readonly get => _endTime; set => _endTime = value; }

            private double _writeInterval;
            /// <summary>
            /// WriterInterval for ControlDict.
            /// </summary>
            public double WriteInterval { readonly get => _writeInterval; set => _writeInterval = value; }

            private int _numberOfSubdomains;
            /// <summary>
            /// Number of CPU's
            /// </summary>
            public int NumberOfSubdomains { readonly get => _numberOfSubdomains; set => _numberOfSubdomains = value; }

            /// <summary>
            /// Where to start from after rerun of simulation.
            /// </summary>
            public readonly StartFrom StartFrom { get; }

            /// <summary>
            /// Condition for stop.
            /// </summary>
            public readonly StopAt StopAt { get; }

            /// <summary>
            /// Specify control scheme.
            /// </summary>
            public readonly WriteControl WriteControl { get; }

            /// <summary>
            /// Compression on or off. 
            /// </summary>
            public readonly WriteCompression WriteCompression { get; }

            /// <summary>
            /// Formate of timesteps.
            /// </summary>
            public readonly TimeFormat TimeFormat { get; }

            /// <summary>
            /// Bool for ControlDict.
            /// </summary>
            public readonly bool RunTimeModifiable { get; }

            /// <summary>
            /// Start time for ControlDict.
            /// </summary>
            public readonly double StartTime { get; }

            /// <summary>
            /// DeltaT for ControlDict.
            /// </summary>
            public readonly double DeltaT { get; }

            /// <summary>
            /// PurgeWrite for ControlDict.
            /// </summary>
            public readonly double PurgeWrite { get; }

            /// <summary>
            /// WritePrecision for ControlDict.
            /// </summary>
            public readonly double WritePrecision { get; }

            /// <summary>
            /// TimePrecision for ControlDict.
            /// </summary>
            public readonly double TimePrecision { get; }

            public readonly override string ToString() => $"{RunTimeModifiable}, {StartTime}, {EndTime}, {DeltaT}, {WriteInterval}, {PurgeWrite}, {WritePrecision}, {TimePrecision}, {NumberOfSubdomains}";
        }

        /// <summary>
        /// K-Epsilon turbulence model datatype.
        /// </summary>
        public struct KEpsilon
        {
            public KEpsilon() : this(0, 0) { }
            public KEpsilon(double k, double epsilon)
            {
                _k = k;
                _epsilon = epsilon;
            }

            /// <summary>
            ///  Contructor which calculates via CalculateKEpsilon.
            /// </summary>
            /// <param name="area">Area of inlet surface.</param>
            /// <param name="boundary">Boundary of inlet surface.</param>
            /// <param name="meanFlowVelocity">Mean flow velocity through inlet.</param>
            /// <param name="tempInlet"> CUrrent temp on inlet.</param>
            public KEpsilon(in double area, in double boundary, in double meanFlowVelocity, in double tempInlet) : this(0, 0) => CalculateKEpsilon(area, boundary, meanFlowVelocity, tempInlet);

            private double _k;
            /// <summary>
            /// Turbulence energie.
            /// </summary>
            public double K { readonly get => _k; set => _k = value; }

            private double _epsilon;
            /// <summary>
            /// Dissipation rate.
            /// </summary>
            public double Epsilon { readonly get => _epsilon; set => _epsilon = value; }

            /// <summary>
            /// Calculate k and epsilon with OpenFOAMCalculator-class.
            /// </summary>
            /// <param name="area">Area of inlet surface.</param>
            /// <param name="boundary">Boundary of inlet surface.</param>
            /// <param name="meanFlowVelocity">Mean flow velocity through inlet.</param>
            /// <param name="temp"> Current temp.</param>
            private void CalculateKEpsilon(in double area, in double boundary, in double meanFlowVelocity, in double temp)
            {
                OpenFOAM.OpenFOAMCalculator calculator = new();

                double kinematicViscosity = calculator.InterpolateKinematicViscosity(temp - 273.15);
                double characteristicLength = calculator.CalculateHydraulicDiameter(area, boundary);
                double reynoldsNumber = calculator.CalculateReynoldsnumber(Math.Abs(meanFlowVelocity), kinematicViscosity, characteristicLength);
                double turbulenceLengthScale = calculator.EstimateTurbulencLengthScalePipe(characteristicLength);
                double turbulenceIntensity = calculator.EstimateTurbulenceIntensityPipe(reynoldsNumber);

                if (meanFlowVelocity != 0)
                {
                    _k = calculator.CalculateK(Math.Abs(meanFlowVelocity), turbulenceIntensity);
                    _epsilon = calculator.CalculateEpsilon(turbulenceLengthScale, 0);
                }
            }
        }

        public struct CastellatedMeshControls
        {
            private static CastellatedMeshControls def = new CastellatedMeshControls();
            public static ref readonly CastellatedMeshControls Default => ref def;
            public struct MeshCoords
            {
                private static XYZ ZERO = new(0, 0, 0);
                private static MeshCoords def = new MeshCoords();
                public static ref readonly MeshCoords Default => ref def;
                public MeshCoords() : this(ZERO, ZERO, ZERO, ZERO) { }
                public MeshCoords(in XYZ origin, in XYZ x, in XYZ y, in XYZ z)
                {
                    this.Origin = origin;
                    this.X = x;
                    this.Y = y;
                    this.Z = z;
                }
                public XYZ Origin { get; set; }
                public XYZ X { get; set; }
                public XYZ Y { get; set; }
                public XYZ Z { get; set; }
            }

            readonly public struct CellParameter
            {
                private static CellParameter def = new CellParameter();
                public static ref readonly CellParameter Default => ref def;
                public CellParameter() : this(100000, 2000000) { }
                public CellParameter(int maxLocal, int maxGlobal)
                {
                    this.MaxLocal = maxLocal;
                    this.maxGlobal = maxGlobal;
                }
                public int MaxLocal { get; }
                public int maxGlobal { get; }
            }

            readonly public struct MeshLVL
            {
                private static Vector defWall = new(3, 3);
                private static Vector defOutIn = new(4, 4);
                private static MeshLVL def = new MeshLVL();
                public static ref readonly MeshLVL Default => ref def;
                public MeshLVL() : this(defWall, defOutIn, defOutIn) { }
                public MeshLVL(in Vector wall, in Vector outlet, in Vector inlet)
                {
                    this.Wall = wall;
                    this.Outlet = outlet;
                    this.Inlet = inlet;
                }
                public Vector Wall { get; }
                public Vector Outlet { get; }
                public Vector Inlet { get; }
            }

            public CastellatedMeshControls() : this(
                CellParameter.Default, 10, 0.10, 3, new(), MeshLVL.Default,
                180, new(), true, new(65.6, 0, 16.5), MeshCoords.Default, MeshCoords.Default)
            { }

            public CastellatedMeshControls(in CellParameter cell, int minRefinementCalls,
                                           in double maxLoadUnbalance, int nCellsBetweenLevels,
                                           in ArrayList features, in MeshLVL meshLVL,
                                           int resolveFeatureAngle,
                                           in Dictionary<string, object> refinementRegions,
                                           bool allowFreeStandingZoneFaces, in Vector3D locationInMesh,
                                           in MeshCoords domainBox, in MeshCoords refinementBox)
            {
                this.CellParam = cell;
                this.MinRefinementCalls = minRefinementCalls;
                this.MaxLoadUnbalance = maxLoadUnbalance;
                this.NCellsBetweenLevels = nCellsBetweenLevels;
                this.Features = features;
                this.MeshLevel = meshLVL;
                this.ResolveFeatureAngle = resolveFeatureAngle;
                this.RefinementRegions = refinementRegions;
                this.AllowFreeStandigZoneFaces = allowFreeStandingZoneFaces;
                this.LocationInMesh = locationInMesh;
                this.DomainBox = domainBox;
                this.RefinementBox = refinementBox;
            }
            public CellParameter CellParam { get; }
            public MeshLVL MeshLevel { get; }
            public ArrayList Features { get; }
            public Dictionary<string, object> RefinementRegions { get; }
            public Vector3D LocationInMesh { get; }
            public MeshCoords DomainBox { get; }
            public MeshCoords RefinementBox { get; }
            public int MinRefinementCalls { get; }
            public int ResolveFeatureAngle { get; }
            public int NCellsBetweenLevels { get; }
            public double MaxLoadUnbalance { get; }
            public bool AllowFreeStandigZoneFaces { get; }
        }

        readonly public struct SnapControls
        {
            private static SnapControls def = new SnapControls();
            public static ref readonly SnapControls Default => ref def;
            public SnapControls() : this(5, 5, 100, 8, 10, true, true) { }
            public SnapControls(int nSmoothPatch, int tolerance, int nSolverIter, int nRelaxIter,
                                int featureSnapIter, bool implicitFeatureSnap, bool multiRegionFeatureSnap)
            {
                this.NSmoothPatch = nSmoothPatch;
                this.Tolerance = tolerance;
                this.NSolverIter = nSolverIter;
                this.NRelaxIterSnap = nRelaxIter;
                this.NFeatureSnapIter = featureSnapIter;
                this.ImplicitFeatureSnap = implicitFeatureSnap;
                this.MultiRegionFeatureSnap = multiRegionFeatureSnap;
            }
            public int NSmoothPatch { get; }
            public int Tolerance { get; }
            public int NSolverIter { get; }
            public int NRelaxIterSnap { get; }
            public int NFeatureSnapIter { get; }
            public bool ImplicitFeatureSnap { get; }
            public bool MultiRegionFeatureSnap { get; }
        }

        readonly public struct AddLayersControl
        {
            readonly public struct SmoothParameter
            {
                private static SmoothParameter def = new SmoothParameter();
                public static ref readonly SmoothParameter Default => ref def;
                public SmoothParameter() : this(1, 10, 3) { }
                public SmoothParameter(int nSmoothSurfaceNormals, int nSmoothThickness, int nSmoothNormals)
                {
                    this.NSmoothNormals = nSmoothNormals;
                    this.NSmoothSurfaceNormals = nSmoothSurfaceNormals;
                    this.NSmoothThickness = nSmoothThickness;
                }
                public int NSmoothSurfaceNormals { get; }
                public int NSmoothThickness { get; }
                public int NSmoothNormals { get; }
            }
            private static AddLayersControl def = new AddLayersControl();
            public static ref readonly AddLayersControl Default => ref def;
            public AddLayersControl() : this(true, new(), 1.1, 0.7, 0.1, 0, 110, 3, SmoothParameter.Default,
                                             0.5, 0.3, 130, 0, 50, 20)
            { }
            public AddLayersControl(bool relativeSizes, in Dictionary<string, object> layers, in double expansionRatio,
                                    in double finalLayerThickness, in double minThickness, int nGrow, int featureAngle,
                                    int nRelaxIterLayer, in SmoothParameter smoothParam, in double maxFaceThicknessRatio,
                                    in double maxThicknessToMedialRatio, int minMedianAxisAngle, int nBufferCellsNoExtrude,
                                    int nLayerIter, int nRelaxedIterLayer)
            {
                this.RelativeSizes = relativeSizes;
                this.Layers = layers;
                this.ExpansionRatio = expansionRatio;
                this.FinalLayerThickness = finalLayerThickness;
                this.MinThickness = minThickness;
                this.NGrow = nGrow;
                this.FeatureAngle = featureAngle;
                this.NRelaxeIterLayer = nRelaxIterLayer;
                this.SmoothParam = smoothParam;
                this.MaxFaceThicknessRatio = maxFaceThicknessRatio;
                this.MaxThicknessToMeadialRatio = maxThicknessToMedialRatio;
                this.MinMedianAxisAngle = minMedianAxisAngle;
                this.NBufferCellsNoExtrude = nBufferCellsNoExtrude;
                this.NLayerIter = nLayerIter;
                this.NRelaxedIterLayer = nRelaxedIterLayer;
            }
            public bool RelativeSizes { get; }
            public double ExpansionRatio { get; }
            public double FinalLayerThickness { get; }
            public double MinThickness { get; }
            public double MaxFaceThicknessRatio { get; }
            public double MaxThicknessToMeadialRatio { get; }
            public int NGrow { get; }
            public int FeatureAngle { get; }
            public int NRelaxeIterLayer { get; }
            public int NRelaxedIterLayer { get; }
            public int MinMedianAxisAngle { get; }
            public int NBufferCellsNoExtrude { get; }
            public int NLayerIter { get; }
            public SmoothParameter SmoothParam { get; }
            public Dictionary<string, object> Layers { get; }
        }

        readonly public struct MeshQualityControls
        {
            readonly public struct Max
            {
                private static Max def = new Max();
                public static ref readonly Max Default => ref def;
                public Max() : this(60, 20, 4, 80, 75) { }
                public Max(int maxNonOrthoMeshQualtiy, int maxBoundarySkewness, int maxInternalSkewness, int maxConcave, int maxNonOrtho)
                {
                    this.MaxNonOrthoMeshQuality = maxNonOrthoMeshQualtiy;
                    this.MaxBoundarySkewness = maxBoundarySkewness;
                    this.MaxInternalSkewness = maxInternalSkewness;
                    this.MaxConcave = maxConcave;
                    this.MaxNonOrtho = maxNonOrtho;
                }
                public int MaxNonOrthoMeshQuality { get; }
                public int MaxBoundarySkewness { get; }
                public int MaxInternalSkewness { get; }
                public int MaxConcave { get; }
                public int MaxNonOrtho { get; }
            }
            readonly public struct Min
            {
                private static Min def = new Min();
                public static ref readonly Min Default => ref def;
                public Min() : this(0.5, 1e-13, 1e-15, -1, 0.02, 0.01, 0.02, 0.01, -1) { }
                public Min(in double minFlatness, in double minVol, in double minTetQuality, int minArea,
                           in double minTwist, in double minDeterminant, in double minFaceWeight, in double minVolRatio,
                           int minTriangleTwist)
                {
                    this.MinFlatness = minFlatness;
                    this.MinVol = minVol;
                    this.MinTetQuality = minTetQuality;
                    this.MinArea = minArea;
                    this.MinTwist = minTwist;
                    this.MinDeterminant = minDeterminant;
                    this.MinFaceWeight = minFaceWeight;
                    this.MinVolRatio = minVolRatio;
                    this.MinTriangleTwist = minTriangleTwist;
                }
                public double MinFlatness { get; }
                public double MinVol { get; }
                public double MinTetQuality { get; }
                public double MinTwist { get; }
                public double MinDeterminant { get; }
                public double MinFaceWeight { get; }
                public double MinVolRatio { get; }
                public int MinArea { get; }
                public int MinTriangleTwist { get; }
            }
            private static MeshQualityControls def = new MeshQualityControls();
            public static ref readonly MeshQualityControls Default => ref def;
            public MeshQualityControls() : this(Max.Default, Min.Default, 4, 0.75,
                                                new Dictionary<string, object> { { "maxNonOrtho", Max.Default.MaxNonOrtho } })
            { }
            public MeshQualityControls(in Max maxParam, in Min minParam, int nSmoothScale, in double errorReduction, in Dictionary<string, object> relaxed)
            {
                this.MaxParam = maxParam;
                this.MinParam = minParam;
                this.NSmoothScale = nSmoothScale;
                this.ErrorReduction = errorReduction;
                this.Relaxed = relaxed;
            }
            public Max MaxParam { get; }
            public Min MinParam { get; }
            public int NSmoothScale { get; }
            public double ErrorReduction { get; }
            public Dictionary<string, object> Relaxed { get; }
        }

        public struct SnappyHexMeshDict
        {
            private static double KELVIN_ZERO_DEG = 273.15;
            private static SnappyHexMeshDict def = new SnappyHexMeshDict();
            public static ref readonly SnappyHexMeshDict Default => ref def;
            public SnappyHexMeshDict() : this(true, true, false, 0, 1e-6,
                                              CastellatedMeshControls.Default, SnapControls.Default,
                                              AddLayersControl.Default, MeshQualityControls.Default,
                                              KELVIN_ZERO_DEG + 25, KELVIN_ZERO_DEG + 29)
            { }
            public SnappyHexMeshDict(bool castellatedMesh, bool snap, bool addLayers, int debug, in double mergeTolerance,
                                     in CastellatedMeshControls castellatedMeshControls, in SnapControls snapControls,
                                     in AddLayersControl addLayersControl, in MeshQualityControls meshQualityControls,
                                     in double tempWall, in double tempInlet)
            {
                this.CastellatedMesh = castellatedMesh;
                this.Snap = snap;
                this.AddLayers = addLayers;
                this.Debug = debug;
                this.MergeTolerance = mergeTolerance;
                this.CastellatedMeshControls = castellatedMeshControls;
                this.SnapControls = snapControls;
                this.AddLayersControl = addLayersControl;
                this.MeshQualityControls = meshQualityControls;
                this.TempWall = tempWall;
                this.TempInlet = tempInlet;
            }

            public bool CastellatedMesh { get; }
            public bool Snap { get; }
            public bool AddLayers { get; }
            public int Debug { get; }
            public double MergeTolerance { get; }
            public double TempWall { get; }
            public double TempInlet { get; }
            public CastellatedMeshControls CastellatedMeshControls { get; }
            public SnapControls SnapControls { get; }
            public AddLayersControl AddLayersControl { get; }
            public MeshQualityControls MeshQualityControls { get; }
        }

        /// Represents an initial parameter from the null folder.
        /// </summary>
        public struct NullParameter
        {
            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="_name">Name for parameter.</param>
            /// <param name="_internalField">Value for internalfield.</param>
            /// <param name="_turbulenceModel">Turbulence-model.</param>
            /// <param name="_solverInc">Incompressible-Solver</param>
            /// <param name="_stype">Turbulence-type.</param>
            public NullParameter(in string name, in dynamic internalField, in dynamic turbulenceModel,
                in SolverControlDict solverInc = SolverControlDict.simpleFoam, in SimulationType stype = SimulationType.RAS)
            {
                this.Name = name;
                this.InternalField = internalField;
                this.TurbulenceModel = turbulenceModel;
                this.Solver = solverInc;
                this.SimulationType = stype;
                _patches = new Dictionary<string, FOAMParameterPatch<dynamic>>();
            }

            /// <summary>
            /// Name of Parameter.
            /// </summary>
            public readonly string Name { get; }

            /// <summary>
            /// Value of internalField.
            /// </summary>
            public readonly dynamic InternalField { get; }

            private Dictionary<string, FOAMParameterPatch<dynamic>> _patches;
            /// <summary>
            /// List of inlet-, outlet-, wall-patches.
            /// </summary>
            public Dictionary<string, FOAMParameterPatch<dynamic>> Patches { readonly get => _patches; set => _patches = value; }

            /// <summary>
            /// Solver for incompressible CFD.
            /// </summary>
            public readonly SolverControlDict Solver { get; }

            /// <summary>
            /// Turbulence simulationType.
            /// </summary>
            // public SimulationType SimulationType { readonly get => _simulationType; set => _simulationType = value; }
            public readonly SimulationType SimulationType { get; }

            /// <summary>
            /// Turbulence-model.
            /// </summary>
            public readonly dynamic TurbulenceModel { get; }
        }

        /// <summary>
        /// Patch for boundaryField in Parameter-Dictionaries.
        /// </summary>
        /// <typeparam name="T">Type for value.</typeparam>
        readonly public struct FOAMParameterPatch<T>
        {
            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="_type">Type of Patch</param>
            /// <param name="_uniform">uniform or nonuniform.</param>
            /// <param name="_value">Vector3D or double.</param>
            /// <param name="_patchType">Enum pathtype. </param>
            public FOAMParameterPatch(in string _type, in string _uniform, in T _value, in PatchType _patchType)
            {
                this.Type = _patchType;
                if (!_value.Equals(default) && !_uniform.Equals(""))
                {
                    this.Attributes = new Dictionary<string, object>
                    {
                        { "type", _type },
                        { "value " + _uniform, _value}
                    };
                }
                else
                {
                    this.Attributes = new Dictionary<string, object>
                    {
                        { "type", _type }
                    };
                }
            }

            /// <summary>
            /// Getter-Method for patchType.
            /// </summary>
            public PatchType Type { get; }

            /// <summary>
            /// Getter for Attributes
            /// </summary>
            public Dictionary<string, object> Attributes { get; }
        }

        /// <summary>
        /// Coeffs-Parameter for DecomposeParDict.
        /// </summary>
        public struct CoeffsMethod
        {
            private static CoeffsMethod def = new CoeffsMethod();
            public static ref readonly CoeffsMethod Default => ref def;
            public CoeffsMethod() : this(new Vector3D(2, 2, 1), 0.001) { }
            public CoeffsMethod(in Vector3D n, in double delta)
            {
                _n = n;
                this.Delta = delta;
            }

            /// <summary>
            /// Distribution n-Vector in DecomposeParDict.
            /// </summary>
            private Vector3D _n;
            public Vector3D N { readonly get => _n; set => _n = value; }

            /// <summary>
            /// Delta of DecomposeParDict.
            /// </summary>
            public readonly double Delta { get; }

            /// <summary>
            /// Creates Dictionary and adds attributes to it.
            /// </summary>
            /// <returns>Dictionary filled with attributes.</returns>
            public Dictionary<string, object> ToDictionary()
            {
                return new Dictionary<string, object>
                {
                    {"n", N},
                    {"delta", Delta}
                };
            }
        }

        /// <summary>
        /// P-FvSolution.
        /// </summary>
        readonly public struct PFv
        {
            public PFv(in FvSolutionParameter param, in Agglomerator agglomerator, in CacheAgglomeration cache) : this(
                param: param,
                agglomerator: agglomerator,
                cache: cache,
                nCellsInCoarsesLevel: 10,
                nPostSweeps: 2,
                nPreSweepers: 0,
                mergeLevels: 1)
            { }

            public PFv(in FvSolutionParameter param, in Agglomerator agglomerator, in CacheAgglomeration cache,
                       int nCellsInCoarsesLevel, int nPostSweeps, int nPreSweepers, int mergeLevels)
            {
                this.Param = param;
                this.Agglomerator = agglomerator;
                this.CacheAgglomeration = cache;
                this.NCellsInCoarsesLevel = nCellsInCoarsesLevel;
                this.NPostSweeps = nPostSweeps;
                this.NPreSweepers = nPreSweepers;
                this.MergeLevels = mergeLevels;
            }

            /// <summary>
            /// Parameter for the p-Dicitonionary in FvSolutionDicitonary.
            /// </summary>
            public FvSolutionParameter Param { get; }

            /// <summary>
            /// Agglomerator-Enum.
            /// </summary>
            public Agglomerator Agglomerator { get; }

            /// <summary>
            /// CachAgglomeration-Enum.
            /// </summary>
            public CacheAgglomeration CacheAgglomeration { get; }

            /// <summary>
            /// Interger for nCellsInCoarsesLevel.
            /// </summary>
            public int NCellsInCoarsesLevel { get; }

            /// <summary>
            /// Integer for nPostSweeps.
            /// </summary>
            public int NPostSweeps { get; }

            /// <summary>
            /// Integer for nPreSweepsre.
            /// </summary>
            public int NPreSweepers { get; }

            /// <summary>
            /// Integer for mergeLevels.
            /// </summary>
            public int MergeLevels { get; }

            /// <summary>
            /// Creates a Dictionary of data.
            /// </summary>
            /// <returns>Dictionary filled with attributes.</returns>
            public Dictionary<string, object> ToDictionary()
            {
                Dictionary<string, object> pList = new Dictionary<string, object>
                {
                    {"agglomerator" , Agglomerator},
                    {"relTol" , Param.RelTol },
                    {"tolerance" , Param.Tolerance },
                    {"nCellsInCoarsesLevel", NCellsInCoarsesLevel },
                    {"smoother" , Param.Smoother },
                    {"solver" , Param.Solver },
                    {"cacheAgglomeration" , CacheAgglomeration },
                    {"nPostSweeps" , NPostSweeps },
                    {"nPreSweepers" , NPreSweepers },
                    {"mergeLevels", MergeLevels }
                };
                return pList;
            }
        }

        /// <summary>
        /// Fv-SolutionParam
        /// </summary>
        public struct FvSolutionParameter
        {
            public FvSolutionParameter(in Smoother smoother, in SolverFV solver, in Preconditioner precond, in double relTol, in double tolerance, in int nSweeps)
            {
                _smoother = smoother;
                _solver = solver;
                _preconditioner = precond;
                _relTol = relTol;
                _tolerance = tolerance;
                _nSweeps = nSweeps;
            }

            /// <summary>
            /// Smoother-type.
            /// </summary>
            private Smoother _smoother;
            public Smoother Smoother { readonly get => _smoother; set => _smoother = value; }

            /// <summary>
            /// Solver for FvSolutionDict.
            /// </summary>
            private SolverFV _solver;
            public SolverFV Solver { readonly get => _solver; set => _solver = value; }

            /// <summary>
            /// Double for relTol in FvSolutionDict.
            /// </summary>
            private double _relTol;
            public double RelTol { readonly get => _relTol; set => _relTol = value; }

            /// <summary>
            /// Double for tolerance in FvSolutionDict.
            /// </summary>
            private double _tolerance;
            public double Tolerance { readonly get => _tolerance; set => _tolerance = value; }

            /// <summary>
            /// Double for nSweeps in FvSolutionDict.
            /// </summary>
            private int _nSweeps;
            public int NSweeps { readonly get => _nSweeps; set => _nSweeps = value; }

            /// <summary>
            /// Manipulates the matrix equation (AP^(-1))*Px=b to solve it more readily.
            /// </summary>
            private Preconditioner _preconditioner;
            public Preconditioner Preconditioner { readonly get => _preconditioner; set => _preconditioner = value; }

            /// <summary>
            /// Creates a Dictionary of data.
            /// </summary>
            /// <returns>Dictionary filled with attributes.</returns>
            public Dictionary<string, object> ToDictionary()
            {
                Dictionary<string, object> paramList = new Dictionary<string, object>
                {
                    {"relTol" , RelTol },
                    {"tolerance" , Tolerance },
                    {"nSweeps" , NSweeps},
                    {"smoother" , Smoother },
                    {"solver" , Solver },
                    {"preconditioner", Preconditioner }

                };
                return paramList;
            }
        }

        /// <summary>
        /// Turbulence attributes for the openfoam dictionary turbulenceProperties.
        /// </summary>
        readonly public struct TurbulenceParameter
        {
            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="simType">Simulationtype enum.</param>
            /// <param name="simModel">Simulation model enum.</param>
            /// <param name="turbulence">true = on, false = off</param>
            /// <param name="printCoeff">true = on, false = off</param>
            public TurbulenceParameter(SimulationType simType, Enum simModel, bool turbulence = true, bool printCoeff = true)
            {
                this.SimType = simType;
                this.StructModel = null;
                switch (simType)
                {
                    case SimulationType.RAS:
                        this.StructModel = new RAS((RASModel)simModel, turbulence, printCoeff);
                        break;
                    case SimulationType.LES:
                        this.StructModel = new LES((LESModel)simModel);
                        //TO-DO: Implement.
                        break;
                    case SimulationType.laminar:
                        //TO-DO: Implement.
                        break;
                    default:
                        break;
                }
            }

            /// <summary>
            /// Type of simulation.
            /// </summary>
            public SimulationType SimType { get; }

            /// <summary>
            /// Model for simulation
            /// </summary>
            public ValueType StructModel { get; }

            /// <summary>
            /// This methode creates and returns the attributes as dictionary<string, object>.
            /// </summary>
            /// <returns>Dictionary with attributes.</returns>
            public Dictionary<string, object> ToDictionary()
            {
                Dictionary<string, object> dict = new Dictionary<string, object>
                {
                    { "simulationType", SimType }
                };
                switch (SimType)
                {
                    case SimulationType.RAS:
                        dict.Add(SimType.ToString(), ((RAS)StructModel).ToDictionary());
                        break;
                    case SimulationType.LES:
                        //TO-DO: Implement LES.
                        break;
                    case SimulationType.laminar:
                        //TO-DO: Implement Laminar.
                        break;

                }
                return dict;
            }
        }

        namespace Model
        {
            /// <summary>
            /// RAS-Model attributes in turbulenceProperties.
            /// </summary>
            readonly public struct RAS
            {
                //internal enum for on and off
                public enum OnOff
                {
                    on = 0,
                    off
                }

                /// <summary>
                /// Constructor.
                /// </summary>
                /// <param name="model">Enum-Object for simulation model.</param>
                /// <param name="turb">turbulence true = on, false = off</param>
                /// <param name="printCoeff">printCoeef true = on, false = off</param>
                public RAS(RASModel model, bool turb, bool printCoeff)
                {
                    this.RASModel = model;
                    this.Turbulence = turb ? OnOff.on : OnOff.off;
                    this.PrintCoeffs = printCoeff ? OnOff.on : OnOff.off;
                }

                ///<summary>
                /// Enum for model name
                ///</summary> 
                public RASModel RASModel { get; }

                ///<summary>
                /// turbulence on or off
                ///</summary> 
                public OnOff Turbulence { get; }

                ///<summary>
                /// print coefficient on or off
                ///</summary> 
                public OnOff PrintCoeffs { get; }

                /// <summary>
                /// Returns all attributes as Dictionary<string,object>
                /// </summary>
                /// <returns>Dictionary filled with attributes.</returns>
                public Dictionary<string, object> ToDictionary()
                {
                    return new Dictionary<string, object>
                    {
                        { "RASModel", RASModel },
                        { "turbulence", Turbulence },
                        { "printCoeffs", PrintCoeffs}
                    };
                }
            }

            /// <summary>
            /// Simulationmodel LES-Parameter.
            /// </summary>
            readonly public struct LES
            {
                public LES(LESModel lESModel)
                {
                    this.LESModel = lESModel;
                }
                public LESModel LESModel { get; }
                //TO-DO: implement LES
            }

            /// <summary>
            /// Simulationmodel Laminar-Parameter.
            /// </summary>
            public struct Laminar
            {
                //SimulationType typeL;
                //TO-DO: Implement laminar
            }
        }
    }

    namespace Revit
    {
        /// <summary>
        /// Properties of a duct terminal.
        /// </summary>
        readonly public struct DuctProperties
        {
            public DuctProperties(in XYZ faceNormal, int rpm, in double externalPressure, in double area, in double boundary, in double flowRate, in double meanFlowVelocity, in double temp)
            {
                this.FaceNormal = faceNormal;
                this.RPM = rpm;
                this.ExternalPressure = externalPressure;
                this.Area = area;
                this.Boundary = boundary;
                this.FlowRate = flowRate;
                this.MeanFlowVelocity = meanFlowVelocity;
                this.Temperature = temp;
            }

            /// <summary>
            /// RPM for swirl diffuser.
            /// </summary>
            public int RPM { get; }

            /// <summary>
            /// External pressure.
            /// </summary>
            public double ExternalPressure { get; }

            /// <summary>
            /// Area of the surface.
            /// </summary>
            public double Area { get; }

            /// <summary>
            /// Boundary of the surface.
            /// </summary>
            public double Boundary { get; }

            /// <summary>
            /// Air flow rate in m�/s.
            /// </summary>
            public double FlowRate { get; }

            /// <summary>
            /// Mean flow velocity through surface.
            /// </summary>
            public double MeanFlowVelocity { get; }

            /// <summary>
            /// Face normal of the surface.
            /// </summary>
            public XYZ FaceNormal { get; }

            /// <summary>
            /// Flow Temperature.
            /// </summary>
            public double Temperature { get; }
        }
    }
}