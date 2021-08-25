/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Security;
using System.Text.RegularExpressions;
using System.Windows.Forms;

namespace OpenFOAMInterface.BIM.OpenFOAM
{
    /// <summary>
    /// Abstract base-class runmanager contains functions which have to be implemented for each OpenFOAM-Run-Environment itselfs.
    /// </summary>
    public abstract class RunManager
    {
        /// <summary>
        /// Status for runmanager.
        /// </summary>
        private DataGenerator.GeneratorStatus m_Status;

        /// <summary>
        /// Path to environment install folder.
        /// </summary>
        private string m_DefaultEnvPath;

        /// <summary>
        /// Environment for simulation.
        /// </summary>
        private string m_EnvTag;

        /// <summary>
        /// Path to config file.
        /// </summary>
        private string m_ConfigPath;

        /// <summary>
        /// Path to openFoam-Case
        /// </summary>
        protected string m_CasePath;

        /// <summary>
        /// Path to install folder of the Environment.
        /// </summary>
        protected string m_FOAMEnvPath;

        /// <summary>
        /// Path to command.bat.
        /// </summary>
        protected string m_CommandBat;

        /// <summary>
        /// Environment selection.
        /// </summary>
        private readonly OpenFOAMEnvironment m_Env;

        /// <summary>
        /// NumberOfSubdomains
        /// </summary>
        private int m_NumberOfSubdomains;

        /// <summary>
        /// TextBox-Form for install folder of simulation environment which will be called if installFolder not default path.
        /// </summary>
        private OpenFOAMUI.OpenFOAMTextBoxForm m_OpenFOAMTxtForm;

        /// <summary>
        /// Getter-Setter for numberOfSubdomains.
        /// </summary>
        public int NumberOfSubdomains { set => m_NumberOfSubdomains = value; get => m_NumberOfSubdomains; }

        /// <summary>
        /// Contructor.
        /// </summary>
        /// <param name="casePath">Path to case.</param>
        public RunManager(string casePath, OpenFOAMEnvironment env)
        {
            m_Env = env;
            m_CasePath = casePath;
            CreateEnvConfig();
        }

        /// <summary>
        /// Interface for initial shell commands for starting openFOAM environment.
        /// </summary>
        /// <returns>List with batch commands.</returns>
        public abstract List<string> InitialEnvRunCommands();

        /// <summary>
        /// Interface for writing command into given streamwriter.
        /// </summary>
        /// <param name="sw">Streamwriter.</param>
        /// <param name="command">Command.</param>
        public abstract void WriteLine(StreamWriter sw, string command);

        /// <summary>
        /// Create bat and write given commands into batch file.
        /// </summary>
        public virtual bool WriteToCommandBat(List<string> command)
        {
            bool succeed = true;

            //create bat
            try
            {
                FileAttributes fileAttribute = FileAttributes.Normal;
                if (File.Exists(m_CommandBat))
                {
                    fileAttribute = File.GetAttributes(m_CommandBat);
                    FileAttributes tempAtt = fileAttribute & FileAttributes.ReadOnly;
                    if (FileAttributes.ReadOnly == tempAtt)
                    {
                        MessageBox.Show(OpenFOAMExportResource.ERR_FILE_READONLY, OpenFOAMExportResource.MESSAGE_BOX_TITLE,
                              MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                        return false;
                    }
                    File.Delete(m_CommandBat);
                }

                using (StreamWriter commandBat = new StreamWriter(m_CommandBat))
                {
                    fileAttribute = File.GetAttributes(m_CommandBat) | fileAttribute;
                    File.SetAttributes(m_CommandBat, fileAttribute);
                    foreach (string com in command)
                    {
                        //diversify between environment
                        WriteLine(commandBat, com);
                    }
                }
            }
            catch (SecurityException)
            {
                MessageBox.Show(OpenFOAMExportResource.ERR_SECURITY_EXCEPTION, OpenFOAMExportResource.MESSAGE_BOX_TITLE,
                            MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                succeed = false;
            }
            catch (IOException)
            {
                MessageBox.Show(OpenFOAMExportResource.ERR_IO_EXCEPTION, OpenFOAMExportResource.MESSAGE_BOX_TITLE,
                            MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                succeed = false;
            }
            catch (Exception)
            {
                MessageBox.Show(OpenFOAMExportResource.ERR_EXCEPTION, OpenFOAMExportResource.MESSAGE_BOX_TITLE,
                            MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                succeed = false;
            }
            return succeed;
        }

        /// <summary>
        /// Running given commands in OpenFOAM-Environment.
        /// </summary>
        /// <param name="commands">Contains commands as string.</param>
        /// <returns>True if suceed and false if not.</returns>
        public virtual bool RunCommands(List<string> commands)
        {
            //create initial Environment commands.
            List<string> runCommands = new List<string>();
            List<string> envCommands = InitialEnvRunCommands();
            foreach (string s in envCommands)
            {
                runCommands.Add(s);
            }
            string log = " | tee " + @"log/";
            foreach (string command in commands)
            {
                if (m_NumberOfSubdomains != 0)
                {
                    if (m_NumberOfSubdomains > 1)
                    {
                        if (command.Equals("snappyHexMesh"))
                        {
                            runCommands.Add("decomposePar" + log + "decomposepar.log");
                            runCommands.Add("mpirun -np " + m_NumberOfSubdomains + " " + command + " -overwrite -parallel " + log + command + ".log");
                            runCommands.Add("reconstructParMesh -constant" + log + "reconstructParMesh.log");
                            continue;
                        }
                        else if (command.Equals("simpleFoam") || command.Equals("buoyantBoussinesqSimpleFoam"))
                        {
                            runCommands.Add("decomposePar");
                            //runCommands.Add("mpirun -n " + DecomposeParDict.NumberOfSubdomains + " renumberMesh -overwrite -parallel");
                            runCommands.Add("mpirun -np " + m_NumberOfSubdomains + " " + command + " -parallel " + log + command + ".log");
                            runCommands.Add("reconstructPar " + Exporter.Instance.settings.ReconstructParOption);
                            continue;
                        }
                    }
                }
                runCommands.Add(command);
            }

            //write commands into batch file
            bool succeed = WriteToCommandBat(runCommands);
            if (succeed)
            {
                ProcessStartInfo startInfo = new ProcessStartInfo
                {
                    FileName = m_CommandBat,
                    WorkingDirectory = m_CasePath
                };

                //start batch
                using (Process process = Process.Start(startInfo))
                {
                    // don't wait
                    //process.WaitForExit();
                    //if(process.ExitCode != 0)
                    //{
                    //    MessageBox.Show("Simulation isn't running properly. Please check the simulation parameter or openfoam environment. If SSH is in use" +
                    //        " check the VPN connection." +
                    //        "\nC#-Process ExitCode: " + process.ExitCode,
                    //        OpenFOAMExportResource.MESSAGE_BOX_TITLE);
                    //    return false;
                    //} 
                }
            }
            else
            {
                return false;
            }
            return succeed;
        }

        /// <summary>
        /// Initialize config.
        /// </summary>
        public virtual void CreateEnvConfig()
        {
            m_DefaultEnvPath = "None";
            m_EnvTag = "<" + m_Env + ">"; ;

            string assemblyDirCorrect = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
            //m_ConfigPath = "c:/tmp/" + "openfoam_env_config.config";
            m_ConfigPath = assemblyDirCorrect + "openfoam_env_config.config";
            switch (m_Env)
            {
                case OpenFOAMEnvironment.blueCFD:
                    {
                        m_DefaultEnvPath = @"C:\Program Files\blueCFD-Core-2017\setvars.bat";
                        break;
                    }
                case OpenFOAMEnvironment.wsl:
                    {
                        m_DefaultEnvPath = @"C:\Windows\System32\bash.exe";
                        break;
                    }
                //case OpenFOAMEnvironment.docker:
                //    {
                //        //implement docker runmanger.
                //        break;
                //    }
                case OpenFOAMEnvironment.ssh:
                    {
                        m_DefaultEnvPath = @"C:\Windows\System32\OpenSSH\ssh.exe";
                        break;
                    }
            }
            CreateConfigEntry();
        }

        /// <summary>
        /// Creates entry for environment based on selected environment.
        /// </summary>
        /// <param name="defaultEnvPath">Default path.</param>
        /// <param name="envTag">Environment enum.</param>
        /// <param name="configPath">Path to config file.</param>
        private void CreateConfigEntry()
        {
            if (!File.Exists(m_ConfigPath))
            {
                NewConfig();
            }
            else
            {
                using (var sr = File.OpenText(m_ConfigPath))
                {
                    var configDone = false;
                    string s;
                    while ((s = sr.ReadLine()) != null)
                    {
                        if (s.Contains(m_EnvTag))
                        {
                            configDone = true;
                            if (!s.Contains(m_DefaultEnvPath))
                            {
                                m_FOAMEnvPath = s.Substring(s.IndexOf(" "));
                            }
                            else
                            {
                                m_FOAMEnvPath = m_DefaultEnvPath;
                            }
                            break;
                        }
                    }
                    sr.Close();
                    if (configDone)
                    {
                        return;
                    }
                    AppendEnvironmentEntryToConfig();
                }
            }
        }

        /// <summary>
        /// Append environment path to existing config.
        /// </summary>
        /// <param name="defaultEnvPath">Default path to install folder of environment.</param>
        /// <param name="envTag">String for environment enum.</param>
        /// <param name="configPath">Path to config file.</param>
        private void AppendEnvironmentEntryToConfig()
        {
            using (var sw = File.AppendText(m_ConfigPath))
            {
                if (File.Exists(m_DefaultEnvPath))
                {
                    sw.WriteLine(m_EnvTag + " " + m_DefaultEnvPath);
                }
                else
                {
                    StartOpenFOAMTextBoxForm();
                    sw.WriteLine(m_EnvTag + " " + m_FOAMEnvPath);
                }
            }
        }

        /// <summary>
        /// Generate new environment entry in config.
        /// </summary>
        /// <param name="defaultEnvPath">default environment path.</param>
        /// <param name="envTag">Environment enum as string.</param>
        /// <param name="configPath">Path to config.</param>
        private void NewConfig()
        {
            StreamWriter sw = new(m_ConfigPath);
            sw.NewLine = "\n";
            sw.WriteLine("**********************Config for OpenFOAM-Environment**********************");

            if (File.Exists(m_DefaultEnvPath))
            {
                sw.WriteLine(m_EnvTag + " " + m_DefaultEnvPath);
                m_FOAMEnvPath = m_DefaultEnvPath;
            }
            else
            {
                StartOpenFOAMTextBoxForm();
                if (m_Status != DataGenerator.GeneratorStatus.SUCCESS)
                    sw.WriteLine(m_EnvTag + " " + m_FOAMEnvPath);
            }
            sw.Close();
        }

        /// <summary>
        /// Initialize OpenFOAMTextBoxForm and show it.
        /// </summary>
        /// <param name="defaultEnvPath">Default simulation environment path.</param>
        private void StartOpenFOAMTextBoxForm()
        {
            Regex reg = new("^\\S+$");
            m_OpenFOAMTxtForm = new(reg, m_DefaultEnvPath, "Searching for " + m_DefaultEnvPath.Substring(m_DefaultEnvPath.LastIndexOf('\\')));

            //set txtBoxForm to active Form
            m_OpenFOAMTxtForm.ShowDialog();

            if (m_OpenFOAMTxtForm.CancelProcess)
            {
                m_Status = DataGenerator.GeneratorStatus.CANCEL;
            }
            else if (reg.IsMatch(m_OpenFOAMTxtForm.Text))
            {
                m_FOAMEnvPath = m_OpenFOAMTxtForm.TxtBox.Text;
                m_Status = DataGenerator.GeneratorStatus.SUCCESS;
            }
            else
            {
                m_Status = DataGenerator.GeneratorStatus.FAILURE;
            }
        }

        /// <summary>
        /// Getter for status of runManager.
        /// </summary>
        public DataGenerator.GeneratorStatus Status
        {
            get
            {
                return m_Status;
            }
        }
    }

    /// <summary>
    /// Runmanager that is used for running OpenFOAM with BlueCFD.
    /// </summary>
    public class RunManagerBlueCFD : RunManager
    {
        /// <summary>
        /// Constructor creates RunManagerBlueCFD-Object.
        /// </summary>
        /// <param name="casePath">Path to case.</param>
        /// <param name="foamPath">Path to DOS_Mode.bat file.</param>
        public RunManagerBlueCFD(string casePath, OpenFOAMEnvironment env)
            : base(casePath, env)
        {
            m_CommandBat = casePath + @"\RunBlueCFD.bat";
        }

        /// <summary>
        /// Initial shell commands for running the blueCFD environment.
        /// </summary>
        /// <returns>List with shell commands as string.</returns>
        public override List<string> InitialEnvRunCommands()
        {
            //if casepath is on another drive than openfoam environment => additional tag /d
            if (m_CasePath.ToCharArray()[0] != m_FOAMEnvPath.ToCharArray()[0])
            {
                m_CasePath = "/d " + m_CasePath;
            }
            List<string> shellCommands = new List<string>
            {
                "call " + "\"" + m_FOAMEnvPath + "\"",
                "set " + @"PATH=%HOME%\msys64\usr\bin;%PATH%",
                "cd " + m_CasePath
            };
            return shellCommands;
        }

        /// <summary>
        /// The method writes command as a line into the streamWriter-object.
        /// </summary>
        /// <param name="sw">StreamWriter object.</param>
        /// <param name="command">Command as string.</param>
        public override void WriteLine(StreamWriter sw, string command)
        {
            sw.WriteLine(command);
        }
    }

    /// <summary>
    /// Runmanager that is used for running OpenFOAM in windows subsystem for linux.
    /// </summary>
    public class RunManagerWSL : RunManager
    {
        /// <summary>
        /// Constructor needs the casePath of the openFoam-case and environment.
        /// </summary>
        /// <param name="casePath">Path to openFfoam-case.</param>
        /// <param name="env">Enum that specifies the environment.</param>
        public RunManagerWSL(string casePath, OpenFOAMEnvironment env)
            : base(casePath, env)
        {
            //windows subsystem for linux
            m_CommandBat = casePath + @"\RunWSL.bat";
        }

        /// <summary>
        /// Initial shell commands for running the windows subsystem for linux environment.
        /// </summary>
        /// <returns>List with shell commands as string.</returns>
        public override List<string> InitialEnvRunCommands()
        {
            List<string> shellCommands = new List<string>
            {
                "bash -c -i \""
            };
            return shellCommands;
        }

        /// <summary>
        /// Adjust commands for running in <see cref="T:RunManager:RunCommands"/> method.
        /// </summary>
        /// <param name="commands">Contains commands as string.</param>
        /// <returns>True if suceed and false if not.</returns>
        public override bool RunCommands(List<string> commands)
        {
            commands.Add("\"");
            bool succeed = base.RunCommands(commands);
            return succeed;
        }

        /// <summary>
        /// The method writes command in one line into the streamWriter-object.
        /// </summary>
        /// <param name="sw">StreamWriter object.</param>
        /// <param name="command">Command as string.</param>
        public override void WriteLine(StreamWriter sw, string command)
        {
            if (command.Equals("\"") || command.Contains("bash") || command.Contains("blockMesh") || command.Contains("All"))
            {
                sw.Write(command);
                return;
            }
            sw.Write(" && " + command);
        }
    }

    /// <summary>
    /// Runmanager that is used for running OpenFOAM in Docker.
    /// </summary>
    public class RunManagerDocker : RunManager
    {
        public RunManagerDocker(string casePath, OpenFOAMEnvironment env)
            : base(casePath, env)
        {

        }

        public override List<string> InitialEnvRunCommands()
        {
            throw new NotImplementedException();
        }


        public override void WriteLine(StreamWriter sw, string command)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// RunManager that is used for running OpenFOAM via SSH.
    /// </summary>
    public class RunManagerSSH : RunManager
    {
        /// <summary>
        /// Default alias strings.
        /// </summary>
        private List<string> m_AliasString = new List<string> { "source", "of", "ofx", /*"bash", */"call" };

        /// <summary>
        /// Constructor needs the casePath of the openFoam-case and environment.
        /// </summary>
        /// <param name="casePath">Path to openFfoam-case.</param>
        /// <param name="env">Enum that specifies the environment.</param>
        public RunManagerSSH(string casePath, OpenFOAMEnvironment env)
            : base(casePath, env)
        {
            m_CommandBat = casePath + @"\RunSSH.bat";
        }

        /// <summary>
        /// Initial shell commands for running OpenFOAM via ssh remote server.
        /// </summary>
        /// <returns>List with shell commands as string.</returns>
        public override List<string> InitialEnvRunCommands()
        {
            string CaseDir = m_CasePath.Substring(m_CasePath.LastIndexOf("\\"));

            //***********************SSH FOR LINUX IMPLEMENTED*******************/
            List<string> shellCommands = new List<string>
            {
                "ssh -p " + Exporter.Instance.settings.SSH.Port + " -t " + Exporter.Instance.settings.SSH.ConnectionString() + " mkdir -p " + Exporter.Instance.settings.SSH.ServerCaseFolder +"\n",
                "scp -P " + Exporter.Instance.settings.SSH.Port + " -r " + m_CasePath + " " + Exporter.Instance.settings.SSH.ConnectionString() + ":" + Exporter.Instance.settings.SSH.ServerCaseFolder,
                "ssh -p " + Exporter.Instance.settings.SSH.Port + " -t " + Exporter.Instance.settings.SSH.ConnectionString() +
                " \"shopt -s expand_aliases ; source ~/.bash_aliases; eval " + Exporter.Instance.settings.SSH.OfAlias +
                "; cd " + Exporter.Instance.settings.SSH.ServerCaseFolder+"/"+CaseDir
            };

            if (Exporter.Instance.settings.SSH.Slurm)
            {
                shellCommands.Add("; chmod +x ./Allrun; chmod +x ./Allclean; ./Allclean; " + Exporter.Instance.settings.SSH.SlurmCommand + " ./Allrun");
            }

            return shellCommands;
        }

        /// <summary>
        /// Adjust commands for running in <see cref="T:RunManager:RunCommands"/> method.
        /// </summary>
        /// <param name="commands">Contains commands as string.</param>
        /// <returns>True if suceed and false if not.</returns>
        public override bool RunCommands(List<string> commands)
        {
            commands.Add("\"");
            string CaseDir = m_CasePath.Substring(m_CasePath.LastIndexOf("\\"));

            //Download directory from Server: scp -r user@ssh.example.com:/path/to/remote/source /path/to/local/destination
            if (Exporter.Instance.settings.SSH.Download)
            {
                string CasePathResults = m_CasePath + "_results";
                if (!Directory.Exists(CasePathResults))
                {
                    Directory.CreateDirectory(CasePathResults);
                }

                commands.Add("scp -P " + Exporter.Instance.settings.SSH.Port + " -r " + Exporter.Instance.settings.SSH.ConnectionString() + ":" + Exporter.Instance.settings.SSH.ServerCaseFolder + "/" + CaseDir + "/* " + CasePathResults);
            }
            if (Exporter.Instance.settings.SSH.Delete)
            {
                commands.Add("ssh -p " + Exporter.Instance.settings.SSH.Port + " -t " + Exporter.Instance.settings.SSH.ConnectionString() + " \"rm -rf " + Exporter.Instance.settings.SSH.ServerCaseFolder);
            }
            bool succeed = base.RunCommands(commands);
            return succeed;
        }

        /// <summary>
        /// The method writes the command into the streamWriter-object.
        /// </summary>
        /// <param name="sw">StreamWriter object.</param>
        /// <param name="command">Command as string.</param>
        public override void WriteLine(StreamWriter sw, string command)
        {
            //add \" as command to show the end of the commands for one bash operation
            if (command.Contains("scp") || command.Equals("\""))
            {
                sw.WriteLine(command);
                return;
            }
            if (command.Contains("ssh") || command.Contains("/bin/bash") || CheckAlias(command))
            {
                sw.Write(command);
                return;
            }

            if (Exporter.Instance.settings.SSH.Slurm)
            {
                //current state => only Allrun
                sw.Write(command + ";");
            }
            else
            {
                sw.Write("; eval " + command);
            }
        }

        /// <summary>
        /// Checks if the command contains alias string from m_AliasString.
        /// </summary>
        /// <param name="command">Command as string.</param>
        /// <returns>True, if the command contains alias.</returns>
        private bool CheckAlias(string command)
        {
            foreach (string alias in m_AliasString)
            {
                if (command.Contains(alias))
                {
                    return true;
                }
            }
            return false;
        }
    }
}
