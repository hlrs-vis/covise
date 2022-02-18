using Autodesk.Revit.DB;
using System.Windows.Media.Media3D;
using System.Collections.Generic;
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
        public struct SSH
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
            public SSH(in string user, in string ip, in string alias, in string caseFolder, bool download, bool delete, bool slurm, int port, in string slurmCommand)
            {
                _user = user;
                _serverIP = ip;
                _ofAlias = alias;
                _serverCaseFolder = caseFolder;
                _download = download;
                _delete = delete;
                _slurm = slurm;
                _port = port;
                _slurmCommands = slurmCommand;
            }

            /// <summary>
            /// Username.
            /// </summary>
            private string _user;
            public string User { readonly get => _user; set => _user = value; }

            /// <summary>
            /// IP of the server (/local computer-name)
            /// </summary>
            private string _serverIP;
            public string ServerIP { readonly get => _serverIP; set => _serverIP = value; }

            /// <summary>
            /// Alias to start openFOAM-Environment on the server.
            /// </summary>
            private string _ofAlias;
            public string OfAlias { readonly get => _ofAlias; set => _ofAlias = value; }

            /// <summary>
            /// Folder on server openfoam case will be copied to.
            /// </summary>
            private string _serverCaseFolder;
            public string ServerCaseFolder { readonly get => _serverCaseFolder; set => _serverCaseFolder = value; }

            /// <summary>
            /// Threads used.
            /// </summary>
            private string _slurmCommands;
            public string SlurmCommand { readonly get => _slurmCommands; set => _slurmCommands = value; }

            /// <summary>
            /// Port server.
            /// </summary>
            private int _port;
            public int Port { readonly get => _port; set => _port = value; }

            /// <summary>
            /// Download after simulation.
            /// </summary>
            private bool _download;
            public bool Download { readonly get => _download; set => _download = value; }

            /// <summary>
            /// Delete after simulation.
            /// </summary>
            private bool _delete;
            public bool Delete { readonly get => _delete; set => _delete = value; }

            /// <summary>
            /// Use slurm.
            /// </summary>
            private bool _slurm;
            public bool Slurm { readonly get => _slurm; set => _slurm = value; }

            /// <summary>
            /// Connection string.
            /// </summary>
            /// <returns>user + @ + serverIP as string.</returns>
            public string ConnectionString()
            {
                return _user + "@" + _serverIP;
            }
        }

        /// <summary>
        /// Struct for intializing Settings variables.
        /// </summary>
        public struct InitialSettingsParameter
        {
            private static InitialSettingsParameter def = new InitialSettingsParameter();
            public static ref readonly InitialSettingsParameter Default => ref def;
            public InitialSettingsParameter() : this(SaveFormat.ascii, ElementsExportRange.OnlyVisibleOnes, ControlDictParameters.Default, true, false, false) { }
            public InitialSettingsParameter(in SaveFormat format, in ElementsExportRange export, in ControlDictParameters control, bool includeLinkedModels, bool exportColor, bool exportSharedCoordinates)
            {
                _format = format;
                _exportRange = export;
                _controlDict = control;
                _includeLinkedModels = includeLinkedModels;
                _exportColor = exportColor;
                _exportSharedCoordinates = exportSharedCoordinates;
            }

            private ControlDictParameters _controlDict;
            /// <summary>
            /// Struct ControlDictParamertes.
            /// </summary>
            public ControlDictParameters ControlDict { readonly get => _controlDict; set => _controlDict = value; }

            private SaveFormat _format;
            /// <summary>
            /// SaveFormat enum
            /// </summary>
            public SaveFormat Format { readonly get => _format; set => _format = value; }

            private ElementsExportRange _exportRange;
            /// <summary>
            /// ExportRange enum.
            /// </summary>
            public ElementsExportRange ExportRange { readonly get => _exportRange; set => _exportRange = value; }

            private bool _includeLinkedModels;
            /// <summary>
            /// IncludedLinkedModels.
            /// </summary>
            public bool IncludeLinkedModels { readonly get => _includeLinkedModels; set => _includeLinkedModels = value; }

            private bool _exportColor;
            /// <summary>
            /// ExportColor enum.
            /// </summary>
            public bool ExportColor { readonly get => _exportColor; set => _exportColor = value; }

            private bool _exportSharedCoordinates;
            /// <summary>
            /// ExportSharedCoordinater for STL.
            /// </summary>
            public bool ExportSharedCoordinates { readonly get => _exportSharedCoordinates; set => _exportSharedCoordinates = value; }

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
                _startFrom = from;
                _stopAt = at;
                _writeControl = writeControl;
                _writeFromat = writeFormat;
                _writeCompression = writeCompression;
                _timeFormat = time;
                _runTimeModifiable = runTimeModifiable;
                _startTime = start;
                _endTime = end;
                _deltaT = deltaT;
                _writeInterval = writeInterval;
                _purgeWrite = purgeWrite;
                _writePrecision = writePrecision;
                _timePrecision = timePrecision;
                _numberOfSubdomains = numberOfSubdomains;
            }

            private SolverControlDict _appControlDictSolver;
            public SolverControlDict AppControlDictSolver { readonly get => _appControlDictSolver; set => _appControlDictSolver = value; }

            private StartFrom _startFrom;
            /// <summary>
            /// Where to start from after rerun of simulation.
            /// </summary>
            public StartFrom StartFrom { readonly get => _startFrom; set => _startFrom = value; }

            private StopAt _stopAt;
            /// <summary>
            /// Condition for stop.
            /// </summary>
            public StopAt StopAt { readonly get => _stopAt; set => _stopAt = value; }

            private WriteControl _writeControl;
            /// <summary>
            /// Specify control scheme.
            /// </summary>
            public WriteControl WriteControl { readonly get => _writeControl; set => _writeControl = value; }

            private WriteFormat _writeFromat;
            /// <summary>
            /// ASCII or Binary.
            /// </summary>
            public WriteFormat WriteFormat { readonly get => _writeFromat; set => _writeFromat = value; }

            private WriteCompression _writeCompression;
            /// <summary>
            /// Compression on or off. 
            /// </summary>
            public WriteCompression WriteCompression { readonly get => _writeCompression; set => _writeCompression = value; }

            private TimeFormat _timeFormat;
            /// <summary>
            /// Formate of timesteps.
            /// </summary>
            public TimeFormat TimeFormat { readonly get => _timeFormat; set => _timeFormat = value; }

            private bool _runTimeModifiable;
            /// <summary>
            /// Bool for ControlDict.
            /// </summary>
            public bool RunTimeModifiable { readonly get => _runTimeModifiable; set => _runTimeModifiable = value; }

            private double _startTime;
            /// <summary>
            /// Start time for ControlDict.
            /// </summary>
            public double StartTime { readonly get => _startTime; set => _startTime = value; }

            private double _endTime;
            /// <summary>
            /// End time for ControlDict.
            /// </summary>
            public double EndTime { readonly get => _endTime; set => _endTime = value; }

            private double _deltaT;
            /// <summary>
            /// DeltaT for ControlDict.
            /// </summary>
            public double DeltaT { readonly get => _deltaT; set => _deltaT = value; }

            private double _writeInterval;
            /// <summary>
            /// WriterInterval for ControlDict.
            /// </summary>
            public double WriteInterval { readonly get => _writeInterval; set => _writeInterval = value; }

            private double _purgeWrite;
            /// <summary>
            /// PurgeWrite for ControlDict.
            /// </summary>
            public double PurgeWrite { readonly get => _purgeWrite; set => _purgeWrite = value; }

            /// <summary>
            /// WritePrecision for ControlDict.
            /// </summary>
            private double _writePrecision;
            public double WritePrecision { readonly get => _writePrecision; set => _writePrecision = value; }

            private double _timePrecision;
            /// <summary>
            /// TimePrecision for ControlDict.
            /// </summary>
            public double TimePrecision { readonly get => _timePrecision; set => _timePrecision = value; }

            private int _numberOfSubdomains;
            /// <summary>
            /// Number of CPU's
            /// </summary>
            public int NumberOfSubdomains { readonly get => _numberOfSubdomains; set => _numberOfSubdomains = value; }

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
            public KEpsilon(double area, double boundary, double meanFlowVelocity, double tempInlet) : this(0,0) => CalculateKEpsilon(area, boundary, meanFlowVelocity, tempInlet);

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
            // public readonly double Epsilon { get; set; }

            /// <summary>
            /// Calculate k and epsilon with OpenFOAMCalculator-class.
            /// </summary>
            /// <param name="area">Area of inlet surface.</param>
            /// <param name="boundary">Boundary of inlet surface.</param>
            /// <param name="meanFlowVelocity">Mean flow velocity through inlet.</param>
            /// <param name="temp"> Current temp.</param>
            private void CalculateKEpsilon(double area, double boundary, double meanFlowVelocity, double temp)
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

        /// <summary>
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
                _name = name;
                _turbulenceModel = turbulenceModel;
                _simulationType = stype;
                _patches = new Dictionary<string, FOAMParameterPatch<dynamic>>();
                _internalField = internalField;
                _solver = solverInc;
            }
            private string _name;
            /// <summary>
            /// Name of Parameter.
            /// </summary>
            public string Name { readonly get => _name; set => _name = value; }

            private dynamic _internalField;
            /// <summary>
            /// Value of internalField.
            /// </summary>
            public dynamic InternalField { readonly get => _internalField; set => _internalField = value; }

            private Dictionary<string, FOAMParameterPatch<dynamic>> _patches;
            /// <summary>
            /// List of inlet-, outlet-, wall-patches.
            /// </summary>
            public Dictionary<string, FOAMParameterPatch<dynamic>> Patches { readonly get => _patches; set => _patches = value; }

            private SolverControlDict _solver;
            /// <summary>
            /// Solver for incompressible CFD.
            /// </summary>
            public SolverControlDict Solver { readonly get => _solver; set => _solver = value; }

            private SimulationType _simulationType;
            /// <summary>
            /// Turbulence simulationType.
            /// </summary>
            public SimulationType SimulationType { readonly get => _simulationType; set => _simulationType = value; }

            private dynamic _turbulenceModel;
            /// <summary>
            /// Turbulence-model.
            /// </summary>
            public dynamic TurbulenceModel { readonly get => _turbulenceModel; set => _turbulenceModel = value; }
        }
        /// <summary>
        /// Patch for boundaryField in Parameter-Dictionaries.
        /// </summary>
        /// <typeparam name="T">Type for value.</typeparam>
        public struct FOAMParameterPatch<T>
        {
            /// <summary>
            /// Type of patch.
            /// </summary>
            string type;

            /// <summary>
            /// PatchType-Enum: inlet, outlet or wall.
            /// </summary>
            PatchType patchType;

            /// <summary>
            /// Attributes of the patch.
            /// </summary>
            Dictionary<string, object> attributes;

            /// <summary>
            /// Value of the patch.
            /// </summary>
            T value;

            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="_type">Type of Patch</param>
            /// <param name="_uniform">uniform or nonuniform.</param>
            /// <param name="_value">Vector3D or double.</param>
            public FOAMParameterPatch(string _type, string _uniform, T _value, PatchType _patchType)
            {
                value = _value;
                type = _type;
                patchType = _patchType;
                if (!_value.Equals(default) && !_uniform.Equals(""))
                {
                    attributes = new Dictionary<string, object>
                {
                    { "type", type },
                    { "value " + _uniform, value}
                };
                }
                else
                {
                    attributes = new Dictionary<string, object>
                {
                    { "type", type }
                };
                }

            }

            /// <summary>
            /// Getter-Method for patchType.
            /// </summary>
            public PatchType Type { get => patchType; }

            /// <summary>
            /// Getter for Attributes
            /// </summary>
            public Dictionary<string, object> Attributes { get => attributes; }
        }
        /// <summary>
        /// Coeffs-Parameter for DecomposeParDict.
        /// </summary>
        public struct CoeffsMethod
        {
            //Attributes
            /// <summary>
            /// Distribution n-Vector in DecomposeParDict.
            /// </summary>
            Vector3D n;

            /// <summary>
            /// Delta of DecomposeParDict.
            /// </summary>
            double delta;

            /// <summary>
            /// Getter for n-Vector.
            /// </summary>
            public Vector3D N { get => n; }

            /// <summary>
            /// Getter for Delta.
            /// </summary>
            public double Delta { get => delta; set => delta = value; }

            /// <summary>
            /// Initialize Vector N with the number of cpu's.
            /// </summary>
            /// <param name="numberOfSubdomains">Number of physical CPU's.</param>
            public void SetN(int numberOfSubdomains)
            {
                //Algo for subDomains

            }

            /// <summary>
            /// Initialize Vector N with given Vecotr _n.
            /// </summary>
            /// <param name="_n">Explicit Vector for N.</param>
            public void SetN(Vector3D _n)
            {
                n = _n;
            }

            /// <summary>
            /// Creates Dictionary and adds attributes to it.
            /// </summary>
            /// <returns>Dictionary filled with attributes.</returns>
            public Dictionary<string, object> ToDictionary()
            {
                Dictionary<string, object> attributes = new Dictionary<string, object>();
                attributes.Add("n", n);
                attributes.Add("delta", delta);
                return attributes;
            }
        }
        /// <summary>
        /// P-FvSolution.
        /// </summary>
        public struct PFv
        {
            //Parameter for the p-Dictionary in FvSolutionDictionary
            /// <summary>
            /// Parameter for the p-Dicitonionary in FvSolutionDicitonary.
            /// </summary>
            FvSolutionParameter param;

            /// <summary>
            /// Agglomerator-Enum.
            /// </summary>
            Agglomerator agglomerator;

            /// <summary>
            /// CachAgglomeration-Enum.
            /// </summary>
            CacheAgglomeration cacheAgglomeration;

            /// <summary>
            /// Interger for nCellsInCoarsesLevel.
            /// </summary>
            int nCellsInCoarsesLevel;

            /// <summary>
            /// Integer for nPostSweeps.
            /// </summary>
            int nPostSweeps;

            /// <summary>
            /// Integer for nPreSweepsre.
            /// </summary>
            int nPreSweepsre;

            /// <summary>
            /// Integer for mergeLevels.
            /// </summary>
            int mergeLevels;

            //Getter-Setter
            public FvSolutionParameter Param { get => param; set => param = value; }
            public Agglomerator Agglomerator { get => agglomerator; set => agglomerator = value; }
            public CacheAgglomeration CacheAgglomeration { get => cacheAgglomeration; set => cacheAgglomeration = value; }
            public int NCellsInCoarsesLevel { get => nCellsInCoarsesLevel; set => nCellsInCoarsesLevel = value; }
            public int NPostSweeps { get => nPostSweeps; set => nPostSweeps = value; }
            public int NPreSweepsre { get => nPreSweepsre; set => nPreSweepsre = value; }
            public int MergeLevels { get => mergeLevels; set => mergeLevels = value; }

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
                {"nPreSweepsre" , NPreSweepsre },
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
            //Paramter that has to be set in FvSolitonDict
            /// <summary>
            /// Smoother-type.
            /// </summary>
            Smoother smoother;

            /// <summary>
            /// Solver for FvSolutionDict.
            /// </summary>
            SolverFV solver;

            /// <summary>
            /// Double for relTol in FvSolutionDict.
            /// </summary>
            double relTol;

            /// <summary>
            /// Double for tolerance in FvSolutionDict.
            /// </summary>
            double tolerance;

            /// <summary>
            /// Double for nSweeps in FvSolutionDict.
            /// </summary>
            int nSweeps;

            /// <summary>
            /// Manipulates the matrix equation (AP^(-1))*Px=b to solve it more readily.
            /// </summary>
            Preconditioner preconditioner;

            //Getter-Setter for Parameter
            public Smoother Smoother { get => smoother; set => smoother = value; }
            public SolverFV Solver { get => solver; set => solver = value; }
            public double RelTol { get => relTol; set => relTol = value; }
            public double Tolerance { get => tolerance; set => tolerance = value; }
            public int NSweeps { get => nSweeps; set => nSweeps = value; }
            public Preconditioner Preconditioner { get => preconditioner; set => preconditioner = value; }

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
        public struct TurbulenceParameter
        {
            /// <summary>
            /// Type of simulation.
            /// </summary>
            SimulationType simulationType;

            /// <summary>
            /// Model for simulation
            /// </summary>
            ValueType _structModel;

            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="simType">Simulationtype enum.</param>
            /// <param name="simModel">Simulation model enum.</param>
            /// <param name="turbulence">true = on, false = off</param>
            /// <param name="printCoeff">true = on, false = off</param>
            public TurbulenceParameter(SimulationType simType, Enum simModel, bool turbulence = true, bool printCoeff = true)
            {
                simulationType = simType;
                _structModel = null;
                switch (simulationType)
                {
                    case SimulationType.RAS:
                        RAS ras = new RAS((RASModel)simModel, turbulence, printCoeff);
                        _structModel = ras;
                        break;
                    case SimulationType.LES:
                        LES les = new LES((LESModel)simModel);
                        _structModel = les;
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
            /// Getter for simulationType.
            /// </summary>
            public SimulationType SimType { get => simulationType; }

            /// <summary>
            /// Getter for structModel.
            /// </summary>
            public ValueType StructModel { get => _structModel; }

            /// <summary>
            /// This methode creates and returns the attributes as dictionary<string, object>.
            /// </summary>
            /// <returns>Dictionary with attributes.</returns>
            public Dictionary<string, object> ToDictionary()
            {
                Dictionary<string, object> dict = new Dictionary<string, object>
            {
                { "simulationType", simulationType }
            };
                switch (simulationType)
                {
                    case SimulationType.RAS:
                        dict.Add(simulationType.ToString(), ((RAS)_structModel).ToDictionary());
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
            public struct RAS
            {
                //internal enum for on and off
                enum OnOff
                {
                    on = 0,
                    off
                }
                //Enum for model name
                RASModel rasModel;
                //turbulence on or off
                OnOff turbulence;
                //print coefficient on or off
                OnOff printCoeffs;

                /// <summary>
                /// Constructor.
                /// </summary>
                /// <param name="model">Enum-Object for simulation model.</param>
                /// <param name="turb">turbulence true = on, false = off</param>
                /// <param name="printCoeff">printCoeef true = on, false = off</param>
                public RAS(RASModel model, bool turb, bool printCoeff)
                {
                    rasModel = model;
                    if (turb)
                    {
                        turbulence = OnOff.on;
                    }
                    else
                    {
                        turbulence = OnOff.off;
                    }

                    if (printCoeff)
                    {
                        printCoeffs = OnOff.on;
                    }
                    else
                    {
                        printCoeffs = OnOff.off;
                    }
                }

                public RASModel RASModel { get => rasModel; }

                /// <summary>
                /// Returns all attributes as Dictionary<string,object>
                /// </summary>
                /// <returns>Dictionary filled with attributes.</returns>
                public Dictionary<string, object> ToDictionary()
                {
                    Dictionary<string, object> dict = new Dictionary<string, object>
            {
                { "RASModel", rasModel },
                { "turbulence", turbulence },
                { "printCoeffs", printCoeffs}
            };
                    return dict;
                }
            }

            /// <summary>
            /// Simulationmodel LES-Parameter.
            /// </summary>
            public struct LES
            {
                LESModel lesModel;
                public LES(LESModel _lesModel)
                {
                    lesModel = _lesModel;
                }

                public LESModel LESModel { get => lesModel; }
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
        public struct DuctProperties
        {
            /// <summary>
            /// RPM for swirl diffuser.
            /// </summary>
            public int RPM { get; set; }

            /// <summary>
            /// External pressure.
            /// </summary>
            public double ExternalPressure { get; set; }

            /// <summary>
            /// Area of the surface.
            /// </summary>
            public double Area { get; set; }

            /// <summary>
            /// Boundary of the surface.
            /// </summary>
            public double Boundary { get; set; }

            /// <summary>
            /// Air flow rate in m�/s.
            /// </summary>
            public double FlowRate { get; set; }

            /// <summary>
            /// Mean flow velocity through surface.
            /// </summary>
            public double MeanFlowVelocity { get; set; }

            /// <summary>
            /// Face normal of the surface.
            /// </summary>
            public XYZ FaceNormal { get; set; }
            /// <summary>
            /// Flow Temperature.
            /// </summary>
            public double Temperature { get; set; }
        }
    }
}