/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
namespace BIM.OpenFOAMExport
{
    partial class OpenFOAMExportForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(OpenFOAMExportForm));
            this.btnCancel = new System.Windows.Forms.Button();
            this.btnHelp = new System.Windows.Forms.Button();
            this.btnSave = new System.Windows.Forms.Button();
            this.tpCategories = new System.Windows.Forms.TabPage();
            this.tvCategories = new System.Windows.Forms.TreeView();
            this.btnCheckNone = new System.Windows.Forms.Button();
            this.btnCheckAll = new System.Windows.Forms.Button();
            this.tpGeneral = new System.Windows.Forms.TabPage();
            this.cbOpenFOAM = new System.Windows.Forms.CheckBox();
            this.cbExportSharedCoordinates = new System.Windows.Forms.CheckBox();
            this.cbExportColor = new System.Windows.Forms.CheckBox();
            this.comboBox_DUT = new System.Windows.Forms.ComboBox();
            this.label1 = new System.Windows.Forms.Label();
            this.cbIncludeLinked = new System.Windows.Forms.CheckBox();
            this.gbSTLFormat = new System.Windows.Forms.GroupBox();
            this.rbAscii = new System.Windows.Forms.RadioButton();
            this.rbBinary = new System.Windows.Forms.RadioButton();
            this.tabControlExporter = new System.Windows.Forms.TabControl();
            this.tbOpenFOAM = new System.Windows.Forms.TabPage();
            this.gbDefault = new System.Windows.Forms.GroupBox();
            this.gbGeneral = new System.Windows.Forms.GroupBox();
            this.txtBoxLocationInMesh = new System.Windows.Forms.TextBox();
            this.lblLocationInMesh = new System.Windows.Forms.Label();
            this.lblTransportModel = new System.Windows.Forms.Label();
            this.comboBoxTransportModel = new System.Windows.Forms.ComboBox();
            this.textBoxCPU = new System.Windows.Forms.TextBox();
            this.lblCPU = new System.Windows.Forms.Label();
            this.comboBoxSolver = new System.Windows.Forms.ComboBox();
            this.lblSolver = new System.Windows.Forms.Label();
            this.comboBoxEnv = new System.Windows.Forms.ComboBox();
            this.lblEnv = new System.Windows.Forms.Label();
            this.vScrollBar1 = new System.Windows.Forms.VScrollBar();
            this.tbSSH = new System.Windows.Forms.TabPage();
            this.cbSlurm = new System.Windows.Forms.CheckBox();
            this.txtBoxSlurmCmd = new System.Windows.Forms.TextBox();
            this.lblSlurmCmd = new System.Windows.Forms.Label();
            this.txtBoxPort = new System.Windows.Forms.TextBox();
            this.lblPort = new System.Windows.Forms.Label();
            this.cbDelete = new System.Windows.Forms.CheckBox();
            this.cbDownload = new System.Windows.Forms.CheckBox();
            this.txtBoxCaseFolder = new System.Windows.Forms.TextBox();
            this.txtBoxAlias = new System.Windows.Forms.TextBox();
            this.txtBoxUserIP = new System.Windows.Forms.TextBox();
            this.lblCaseFolder = new System.Windows.Forms.Label();
            this.lblAlias = new System.Windows.Forms.Label();
            this.lblUserHost = new System.Windows.Forms.Label();
            this.tpCategories.SuspendLayout();
            this.tpGeneral.SuspendLayout();
            this.gbSTLFormat.SuspendLayout();
            this.tabControlExporter.SuspendLayout();
            this.tbOpenFOAM.SuspendLayout();
            this.gbGeneral.SuspendLayout();
            this.tbSSH.SuspendLayout();
            this.SuspendLayout();
            // 
            // btnCancel
            // 
            resources.ApplyResources(this.btnCancel, "btnCancel");
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.BtnCancel_Click);
            // 
            // btnHelp
            // 
            resources.ApplyResources(this.btnHelp, "btnHelp");
            this.btnHelp.Name = "btnHelp";
            this.btnHelp.UseVisualStyleBackColor = true;
            this.btnHelp.Click += new System.EventHandler(this.BtnHelp_Click);
            // 
            // btnSave
            // 
            resources.ApplyResources(this.btnSave, "btnSave");
            this.btnSave.Name = "btnSave";
            this.btnSave.UseVisualStyleBackColor = true;
            this.btnSave.Click += new System.EventHandler(this.BtnSave_Click);
            // 
            // tpCategories
            // 
            this.tpCategories.BackColor = System.Drawing.SystemColors.Control;
            this.tpCategories.Controls.Add(this.tvCategories);
            this.tpCategories.Controls.Add(this.btnCheckNone);
            this.tpCategories.Controls.Add(this.btnCheckAll);
            resources.ApplyResources(this.tpCategories, "tpCategories");
            this.tpCategories.Name = "tpCategories";
            // 
            // tvCategories
            // 
            this.tvCategories.CheckBoxes = true;
            resources.ApplyResources(this.tvCategories, "tvCategories");
            this.tvCategories.Name = "tvCategories";
            // 
            // btnCheckNone
            // 
            resources.ApplyResources(this.btnCheckNone, "btnCheckNone");
            this.btnCheckNone.Name = "btnCheckNone";
            this.btnCheckNone.UseVisualStyleBackColor = true;
            this.btnCheckNone.Click += new System.EventHandler(this.BtnCheckNone_Click);
            // 
            // btnCheckAll
            // 
            resources.ApplyResources(this.btnCheckAll, "btnCheckAll");
            this.btnCheckAll.Name = "btnCheckAll";
            this.btnCheckAll.UseVisualStyleBackColor = true;
            this.btnCheckAll.Click += new System.EventHandler(this.BtnCheckAll_Click);
            // 
            // tpGeneral
            // 
            this.tpGeneral.BackColor = System.Drawing.SystemColors.Control;
            this.tpGeneral.Controls.Add(this.cbOpenFOAM);
            this.tpGeneral.Controls.Add(this.cbExportSharedCoordinates);
            this.tpGeneral.Controls.Add(this.cbExportColor);
            this.tpGeneral.Controls.Add(this.comboBox_DUT);
            this.tpGeneral.Controls.Add(this.label1);
            this.tpGeneral.Controls.Add(this.cbIncludeLinked);
            this.tpGeneral.Controls.Add(this.gbSTLFormat);
            resources.ApplyResources(this.tpGeneral, "tpGeneral");
            this.tpGeneral.Name = "tpGeneral";
            // 
            // cbOpenFOAM
            // 
            resources.ApplyResources(this.cbOpenFOAM, "cbOpenFOAM");
            this.cbOpenFOAM.Checked = true;
            this.cbOpenFOAM.CheckState = System.Windows.Forms.CheckState.Checked;
            this.cbOpenFOAM.Name = "cbOpenFOAM";
            this.cbOpenFOAM.UseVisualStyleBackColor = true;
            this.cbOpenFOAM.CheckedChanged += new System.EventHandler(this.CbOpenFOAM_CheckedChanged);
            // 
            // cbExportSharedCoordinates
            // 
            resources.ApplyResources(this.cbExportSharedCoordinates, "cbExportSharedCoordinates");
            this.cbExportSharedCoordinates.Name = "cbExportSharedCoordinates";
            this.cbExportSharedCoordinates.UseVisualStyleBackColor = true;
            // 
            // cbExportColor
            // 
            resources.ApplyResources(this.cbExportColor, "cbExportColor");
            this.cbExportColor.Name = "cbExportColor";
            this.cbExportColor.UseVisualStyleBackColor = true;
            // 
            // comboBox_DUT
            // 
            this.comboBox_DUT.FormattingEnabled = true;
            resources.ApplyResources(this.comboBox_DUT, "comboBox_DUT");
            this.comboBox_DUT.Name = "comboBox_DUT";
            // 
            // label1
            // 
            resources.ApplyResources(this.label1, "label1");
            this.label1.Name = "label1";
            // 
            // cbIncludeLinked
            // 
            resources.ApplyResources(this.cbIncludeLinked, "cbIncludeLinked");
            this.cbIncludeLinked.Name = "cbIncludeLinked";
            this.cbIncludeLinked.UseVisualStyleBackColor = true;
            this.cbIncludeLinked.CheckedChanged += new System.EventHandler(this.CbIncludeLinked_CheckedChanged);
            // 
            // gbSTLFormat
            // 
            this.gbSTLFormat.Controls.Add(this.rbAscii);
            this.gbSTLFormat.Controls.Add(this.rbBinary);
            resources.ApplyResources(this.gbSTLFormat, "gbSTLFormat");
            this.gbSTLFormat.Name = "gbSTLFormat";
            this.gbSTLFormat.TabStop = false;
            // 
            // rbAscii
            // 
            this.rbAscii.Checked = true;
            resources.ApplyResources(this.rbAscii, "rbAscii");
            this.rbAscii.Name = "rbAscii";
            this.rbAscii.TabStop = true;
            this.rbAscii.UseVisualStyleBackColor = true;
            // 
            // rbBinary
            // 
            resources.ApplyResources(this.rbBinary, "rbBinary");
            this.rbBinary.Name = "rbBinary";
            this.rbBinary.UseVisualStyleBackColor = true;
            this.rbBinary.CheckedChanged += new System.EventHandler(this.RbExportFormat_CheckedChanged);
            // 
            // tabControlExporter
            // 
            this.tabControlExporter.Controls.Add(this.tpGeneral);
            this.tabControlExporter.Controls.Add(this.tpCategories);
            this.tabControlExporter.Controls.Add(this.tbOpenFOAM);
            this.tabControlExporter.Controls.Add(this.tbSSH);
            resources.ApplyResources(this.tabControlExporter, "tabControlExporter");
            this.tabControlExporter.Name = "tabControlExporter";
            this.tabControlExporter.SelectedIndex = 0;
            // 
            // tbOpenFOAM
            // 
            this.tbOpenFOAM.BackColor = System.Drawing.SystemColors.Control;
            this.tbOpenFOAM.Controls.Add(this.gbDefault);
            this.tbOpenFOAM.Controls.Add(this.gbGeneral);
            resources.ApplyResources(this.tbOpenFOAM, "tbOpenFOAM");
            this.tbOpenFOAM.Name = "tbOpenFOAM";
            // 
            // gbDefault
            // 
            resources.ApplyResources(this.gbDefault, "gbDefault");
            this.gbDefault.Name = "gbDefault";
            this.gbDefault.TabStop = false;
            // 
            // gbGeneral
            // 
            this.gbGeneral.Controls.Add(this.txtBoxLocationInMesh);
            this.gbGeneral.Controls.Add(this.lblLocationInMesh);
            this.gbGeneral.Controls.Add(this.lblTransportModel);
            this.gbGeneral.Controls.Add(this.comboBoxTransportModel);
            this.gbGeneral.Controls.Add(this.textBoxCPU);
            this.gbGeneral.Controls.Add(this.lblCPU);
            this.gbGeneral.Controls.Add(this.comboBoxSolver);
            this.gbGeneral.Controls.Add(this.lblSolver);
            this.gbGeneral.Controls.Add(this.comboBoxEnv);
            this.gbGeneral.Controls.Add(this.lblEnv);
            this.gbGeneral.Controls.Add(this.vScrollBar1);
            resources.ApplyResources(this.gbGeneral, "gbGeneral");
            this.gbGeneral.Name = "gbGeneral";
            this.gbGeneral.TabStop = false;
            // 
            // txtBoxLocationInMesh
            // 
            resources.ApplyResources(this.txtBoxLocationInMesh, "txtBoxLocationInMesh");
            this.txtBoxLocationInMesh.Name = "txtBoxLocationInMesh";
            this.txtBoxLocationInMesh.Click += new System.EventHandler(this.TxtBoxLocationInMesh_Click);
            this.txtBoxLocationInMesh.TextChanged += new System.EventHandler(this.TxtBoxLocationInMesh_ValueChanged);
            this.txtBoxLocationInMesh.KeyPress += new System.Windows.Forms.KeyPressEventHandler(this.TxtBoxLocationInMesh_KeyPress);
            this.txtBoxLocationInMesh.Leave += new System.EventHandler(this.TxtBoxLocationInMesh_Leave);
            // 
            // lblLocationInMesh
            // 
            resources.ApplyResources(this.lblLocationInMesh, "lblLocationInMesh");
            this.lblLocationInMesh.Name = "lblLocationInMesh";
            // 
            // lblTransportModel
            // 
            resources.ApplyResources(this.lblTransportModel, "lblTransportModel");
            this.lblTransportModel.Name = "lblTransportModel";
            // 
            // comboBoxTransportModel
            // 
            this.comboBoxTransportModel.FormattingEnabled = true;
            resources.ApplyResources(this.comboBoxTransportModel, "comboBoxTransportModel");
            this.comboBoxTransportModel.Name = "comboBoxTransportModel";
            this.comboBoxTransportModel.SelectedValueChanged += new System.EventHandler(this.ComboBoxTransport_SelectedValueChanged);
            // 
            // textBoxCPU
            // 
            resources.ApplyResources(this.textBoxCPU, "textBoxCPU");
            this.textBoxCPU.Name = "textBoxCPU";
            // 
            // lblCPU
            // 
            resources.ApplyResources(this.lblCPU, "lblCPU");
            this.lblCPU.Name = "lblCPU";
            // 
            // comboBoxSolver
            // 
            this.comboBoxSolver.FormattingEnabled = true;
            resources.ApplyResources(this.comboBoxSolver, "comboBoxSolver");
            this.comboBoxSolver.Name = "comboBoxSolver";
            this.comboBoxSolver.SelectedValueChanged += new System.EventHandler(this.ComboBoxSolver_SelectedValueChanged);
            // 
            // lblSolver
            // 
            resources.ApplyResources(this.lblSolver, "lblSolver");
            this.lblSolver.Name = "lblSolver";
            // 
            // comboBoxEnv
            // 
            this.comboBoxEnv.FormattingEnabled = true;
            resources.ApplyResources(this.comboBoxEnv, "comboBoxEnv");
            this.comboBoxEnv.Name = "comboBoxEnv";
            this.comboBoxEnv.SelectedValueChanged += new System.EventHandler(this.ComboBoxEnv_SelectedValueChanged);
            // 
            // lblEnv
            // 
            resources.ApplyResources(this.lblEnv, "lblEnv");
            this.lblEnv.Name = "lblEnv";
            // 
            // vScrollBar1
            // 
            resources.ApplyResources(this.vScrollBar1, "vScrollBar1");
            this.vScrollBar1.Name = "vScrollBar1";
            // 
            // tbSSH
            // 
            this.tbSSH.BackColor = System.Drawing.SystemColors.Control;
            this.tbSSH.Controls.Add(this.cbSlurm);
            this.tbSSH.Controls.Add(this.txtBoxSlurmCmd);
            this.tbSSH.Controls.Add(this.lblSlurmCmd);
            this.tbSSH.Controls.Add(this.txtBoxPort);
            this.tbSSH.Controls.Add(this.lblPort);
            this.tbSSH.Controls.Add(this.cbDelete);
            this.tbSSH.Controls.Add(this.cbDownload);
            this.tbSSH.Controls.Add(this.txtBoxCaseFolder);
            this.tbSSH.Controls.Add(this.txtBoxAlias);
            this.tbSSH.Controls.Add(this.txtBoxUserIP);
            this.tbSSH.Controls.Add(this.lblCaseFolder);
            this.tbSSH.Controls.Add(this.lblAlias);
            this.tbSSH.Controls.Add(this.lblUserHost);
            resources.ApplyResources(this.tbSSH, "tbSSH");
            this.tbSSH.Name = "tbSSH";
            // 
            // cbSlurm
            // 
            resources.ApplyResources(this.cbSlurm, "cbSlurm");
            this.cbSlurm.Checked = true;
            this.cbSlurm.CheckState = System.Windows.Forms.CheckState.Checked;
            this.cbSlurm.Name = "cbSlurm";
            this.cbSlurm.UseVisualStyleBackColor = true;
            this.cbSlurm.CheckedChanged += new System.EventHandler(this.CbSlurm_CheckedChanged);
            // 
            // txtBoxSlurmCmd
            // 
            resources.ApplyResources(this.txtBoxSlurmCmd, "txtBoxSlurmCmd");
            this.txtBoxSlurmCmd.Name = "txtBoxSlurmCmd";
            this.txtBoxSlurmCmd.TextChanged += new System.EventHandler(this.TxtBoxSlurmCommands_ValueChanged);
            // 
            // lblSlurmCmd
            // 
            resources.ApplyResources(this.lblSlurmCmd, "lblSlurmCmd");
            this.lblSlurmCmd.Name = "lblSlurmCmd";
            // 
            // txtBoxPort
            // 
            resources.ApplyResources(this.txtBoxPort, "txtBoxPort");
            this.txtBoxPort.Name = "txtBoxPort";
            this.txtBoxPort.TextChanged += new System.EventHandler(this.TxtBoxPort_ValueChanged);
            // 
            // lblPort
            // 
            resources.ApplyResources(this.lblPort, "lblPort");
            this.lblPort.Name = "lblPort";
            // 
            // cbDelete
            // 
            resources.ApplyResources(this.cbDelete, "cbDelete");
            this.cbDelete.Checked = true;
            this.cbDelete.CheckState = System.Windows.Forms.CheckState.Checked;
            this.cbDelete.Name = "cbDelete";
            this.cbDelete.UseVisualStyleBackColor = true;
            this.cbDelete.CheckedChanged += new System.EventHandler(this.CbDelete_CheckedChanged);
            // 
            // cbDownload
            // 
            resources.ApplyResources(this.cbDownload, "cbDownload");
            this.cbDownload.Checked = true;
            this.cbDownload.CheckState = System.Windows.Forms.CheckState.Checked;
            this.cbDownload.Name = "cbDownload";
            this.cbDownload.UseVisualStyleBackColor = true;
            this.cbDownload.CheckedChanged += new System.EventHandler(this.CbDownload_CheckedChanged);
            // 
            // txtBoxCaseFolder
            // 
            resources.ApplyResources(this.txtBoxCaseFolder, "txtBoxCaseFolder");
            this.txtBoxCaseFolder.Name = "txtBoxCaseFolder";
            this.txtBoxCaseFolder.TextChanged += new System.EventHandler(this.TxtBoxServerCaseFolder_ValueChanged);
            // 
            // txtBoxAlias
            // 
            resources.ApplyResources(this.txtBoxAlias, "txtBoxAlias");
            this.txtBoxAlias.Name = "txtBoxAlias";
            this.txtBoxAlias.TextChanged += new System.EventHandler(this.TxtBoxAlias_ValueChanged);
            // 
            // txtBoxUserIP
            // 
            resources.ApplyResources(this.txtBoxUserIP, "txtBoxUserIP");
            this.txtBoxUserIP.Name = "txtBoxUserIP";
            this.txtBoxUserIP.TextChanged += new System.EventHandler(this.TxtBoxUserIP_ValueChanged);
            // 
            // lblCaseFolder
            // 
            resources.ApplyResources(this.lblCaseFolder, "lblCaseFolder");
            this.lblCaseFolder.Name = "lblCaseFolder";
            // 
            // lblAlias
            // 
            resources.ApplyResources(this.lblAlias, "lblAlias");
            this.lblAlias.Name = "lblAlias";
            // 
            // lblUserHost
            // 
            resources.ApplyResources(this.lblUserHost, "lblUserHost");
            this.lblUserHost.Name = "lblUserHost";
            // 
            // OpenFOAMExportForm
            // 
            resources.ApplyResources(this, "$this");
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Dpi;
            this.Controls.Add(this.tabControlExporter);
            this.Controls.Add(this.btnSave);
            this.Controls.Add(this.btnHelp);
            this.Controls.Add(this.btnCancel);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "OpenFOAMExportForm";
            this.tpCategories.ResumeLayout(false);
            this.tpGeneral.ResumeLayout(false);
            this.tpGeneral.PerformLayout();
            this.gbSTLFormat.ResumeLayout(false);
            this.tabControlExporter.ResumeLayout(false);
            this.tbOpenFOAM.ResumeLayout(false);
            this.gbGeneral.ResumeLayout(false);
            this.gbGeneral.PerformLayout();
            this.tbSSH.ResumeLayout(false);
            this.tbSSH.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.Button btnHelp;
        private System.Windows.Forms.Button btnSave;
        private System.Windows.Forms.TabPage tpCategories;
        private System.Windows.Forms.TreeView tvCategories;
        private System.Windows.Forms.Button btnCheckNone;
        private System.Windows.Forms.Button btnCheckAll;
        private System.Windows.Forms.TabPage tpGeneral;
        private System.Windows.Forms.CheckBox cbExportSharedCoordinates;
        private System.Windows.Forms.CheckBox cbExportColor;
        private System.Windows.Forms.ComboBox comboBox_DUT;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.CheckBox cbIncludeLinked;
        private System.Windows.Forms.GroupBox gbSTLFormat;
        private System.Windows.Forms.RadioButton rbAscii;
        private System.Windows.Forms.RadioButton rbBinary;
        private System.Windows.Forms.TabControl tabControlExporter;
        private System.Windows.Forms.CheckBox cbOpenFOAM;
        private System.Windows.Forms.TabPage tbOpenFOAM;
        private System.Windows.Forms.GroupBox gbDefault;
        private System.Windows.Forms.GroupBox gbGeneral;
        private System.Windows.Forms.ComboBox comboBoxEnv;
        private System.Windows.Forms.Label lblEnv;
        private System.Windows.Forms.VScrollBar vScrollBar1;
        private System.Windows.Forms.ComboBox comboBoxSolver;
        private System.Windows.Forms.Label lblSolver;
        private System.Windows.Forms.TextBox textBoxCPU;
        private System.Windows.Forms.Label lblCPU;
        private System.Windows.Forms.Label lblTransportModel;
        private System.Windows.Forms.ComboBox comboBoxTransportModel;
        private System.Windows.Forms.TabPage tbSSH;
        private System.Windows.Forms.CheckBox cbDelete;
        private System.Windows.Forms.CheckBox cbDownload;
        private System.Windows.Forms.TextBox txtBoxCaseFolder;
        private System.Windows.Forms.TextBox txtBoxAlias;
        private System.Windows.Forms.TextBox txtBoxUserIP;
        private System.Windows.Forms.Label lblCaseFolder;
        private System.Windows.Forms.Label lblAlias;
        private System.Windows.Forms.Label lblUserHost;
        private System.Windows.Forms.TextBox txtBoxPort;
        private System.Windows.Forms.Label lblPort;
        private System.Windows.Forms.TextBox txtBoxLocationInMesh;
        private System.Windows.Forms.Label lblLocationInMesh;
        private System.Windows.Forms.TextBox txtBoxSlurmCmd;
        private System.Windows.Forms.Label lblSlurmCmd;
        private System.Windows.Forms.CheckBox cbSlurm;
    }
}
