
// Firewall profiles (XP and Vista/7)
#define NET_FW_PROFILE_DOMAIN 0
#define NET_FW_PROFILE_STANDARD 1
#define NET_FW_PROFILE2_DOMAIN = 1
#define NET_FW_PROFILE2_PRIVATE = 2
#define NET_FW_PROFILE2_PUBLIC = 4

// Firewall IP scopes (subnet or all)
#define NET_FW_SCOPE_ALL = 0

// Firewall Protocol (TCP or UDP)
#define NET_FW_IP_PROTOCOL_TCP = 6
#define NET_FW_IP_PROTOCOL_UDP = 17
  
// Firewall IP scopes (v4 and v6)
#define NET_FW_IP_VERSION_ANY = 2

// Firewall action
#define NET_FW_ACTION_ALLOW = 1

procedure AddProgramToFirewallXP(appname, filename: string);
var
  FirewallObject: Variant;
  FirewallManager: Variant;
  FirewallProfile: Variant;
begin
try
  FirewallObject := CreateOleObject('HNetCfg.FwAuthorizedApplication');
  FirewallObject.Name := appname;
  FirewallObject.Enabled := true;
  FirewallObject.ProcessImageFileName := filename;
  FirewallObject.Scope := {#NET_FW_SCOPE_ALL};
  FirewallObject.IpVersion := {#NET_FW_IP_VERSION_ANY};
  FirewallManager := CreateOleObject('HNetCfg.FwMgr');
  FirewallProfile := FirewallManager.LocalPolicy.GetProfileByType({#NET_FW_PROFILE_DOMAIN});
  FirewallProfile.AuthorizedApplications.Add(FirewallObject);
  FirewallProfile := FirewallManager.LocalPolicy.GetProfileByType({#NET_FW_PROFILE_STANDARD});
  FirewallProfile.AuthorizedApplications.Add(FirewallObject);
except
  end;
end;

procedure AddPortToFirewallXP(portname: string; port, protocol: Integer);
var
  FirewallObject: Variant;
  FirewallManager: Variant;
  FirewallProfile: Variant;
begin
try
  FirewallObject := CreateOleObject('HNetCfg.FwOpenPort');
  FirewallObject.Name := portname;
  FirewallObject.Enabled := True;
  FirewallObject.Port := port;
  FirewallObject.Protocol := protocol;
  FirewallObject.Scope := {#NET_FW_SCOPE_ALL};
  FirewallManager := CreateOleObject('HNetCfg.FwMgr');
  FirewallProfile := FirewallManager.LocalPolicy.GetProfileByType({#NET_FW_PROFILE_DOMAIN});
  FirewallProfile.GloballyOpenPorts.Add(FirewallObject);
  FirewallProfile := FirewallManager.LocalPolicy.GetProfileByType({#NET_FW_PROFILE_STANDARD});
  FirewallProfile.GloballyOpenPorts.Add(FirewallObject);
except
  end;
end;

procedure RemoveProgramFromFirewallXP(filename: string);
var
  FirewallManager: Variant;
  FirewallProfile: Variant;
begin
try
  FirewallManager := CreateOleObject('HNetCfg.FwMgr');
  FirewallProfile := FirewallManager.LocalPolicy.GetProfileByType({#NET_FW_PROFILE_DOMAIN});
  FireWallProfile.AuthorizedApplications.Remove(filename);
  FirewallProfile := FirewallManager.LocalPolicy.GetProfileByType({#NET_FW_PROFILE_STANDARD});
  FireWallProfile.AuthorizedApplications.Remove(filename);
except
  end;
end;

procedure RemovePortFromFirewallXP(port, protocol: Integer);
var
  FirewallManager: Variant;
  FirewallProfile: Variant;
begin
try
  FirewallManager := CreateOleObject('HNetCfg.FwMgr');
  FirewallProfile := FirewallManager.LocalPolicy.GetProfileByType({#NET_FW_PROFILE_DOMAIN});
  FireWallProfile.GloballyOpenPorts.Remove(port, protocol);
  FirewallProfile := FirewallManager.LocalPolicy.GetProfileByType({#NET_FW_PROFILE_STANDARD});
  FireWallProfile.GloballyOpenPorts.Remove(port, protocol);
except
  end;
end;

procedure AddProgramToFirewallVista7(appname, grouping, filename: string);
var
  FirewallRule: Variant;
  FirewallPolicy: Variant;
begin
try
  FirewallRule := CreateOleObject('HNetCfg.FWRule');
  FirewallRule.Name := appname;
  FirewallRule.Grouping := grouping;
  FirewallRule.Enabled := True;
  FirewallRule.ApplicationName := filename;
  FirewallRule.InterfaceTypes := 'All';
  FirewallRule.Profiles := {#NET_FW_PROFILE2_PUBLIC} OR {#NET_FW_PROFILE2_DOMAIN} OR {#NET_FW_PROFILE2_PRIVATE};
  FirewallRule.Action := {#NET_FW_ACTION_ALLOW};
  FirewallPolicy := CreateOleObject('HNetCfg.FwPolicy2');
  FirewallPolicy.Rules.Add(FirewallRule);
except
  end;
end;

procedure AddPortToFirewallVista7(portname, grouping: string; port, protocol: Integer);
var
  FirewallRule: Variant;
  FirewallPolicy: Variant;
begin
try
  FirewallRule := CreateOleObject('HNetCfg.FWRule');
  FirewallRule.Name := portname;
  FirewallRule.Grouping := grouping;
  FirewallRule.Enabled := True;
  FirewallRule.Protocol := protocol;
  FirewallRule.LocalPorts := inttostr(port);
  FirewallRule.InterfaceTypes := 'All';
  FirewallRule.Profiles := {#NET_FW_PROFILE2_PUBLIC} OR {#NET_FW_PROFILE2_DOMAIN} OR {#NET_FW_PROFILE2_PRIVATE};
  FirewallRule.Action := {#NET_FW_ACTION_ALLOW};
  FirewallPolicy := CreateOleObject('HNetCfg.FwPolicy2');
  FirewallPolicy.Rules.Add(FirewallRule);
except
  end;
end;

procedure RemoveRuleFromFirewallVista7(rulename: string);
var
  FirewallPolicy: Variant;
begin
try
  FirewallPolicy := CreateOleObject('HNetCfg.FwPolicy2');
  FirewallPolicy.Rules.Remove(rulename);
except
  end;
end;

function IsVista7(): boolean;
var
  WinVersion: TWindowsVersion;
begin
GetWindowsVersionEx(WinVersion);
result := WinVersion.NTPlatform and (WinVersion.Major >= 6);
end;

procedure AddProgramToFirewall(appname, grouping, filename: string);
begin
try
  if IsVista7() then begin
    RemoveRuleFromFirewallVista7(appname);
    AddProgramToFirewallVista7(appname, grouping, filename)
    end
  else begin
    AddProgramToFirewallXP(appname, filename);
    RemoveProgramFromFirewallXP(filename);
    end;
except
  end;
end;

procedure RemoveProgramFromFirewall(appname, filename: string);
begin
try
  if IsVista7() then
    RemoveRuleFromFirewallVista7(appname)
  else
    RemoveProgramFromFirewallXP(filename);
except
  end;
end;

procedure AddPortToFirewall(portname, grouping: string; port, protocol: Integer);
begin
try
  if IsVista7() then begin
    RemoveRuleFromFirewallVista7(portname);
    AddPortToFirewallVista7(portname, grouping, port, protocol)
    end
  else begin
    RemovePortFromFirewallXP(port, protocol);
    AddPortToFirewallXP(portname, port, protocol);
    end;
except
  end;
end;

procedure RemovePortFromFirewall(portname: string; port, protocol: Integer);
begin
try
  if IsVista7() then
    RemoveRuleFromFirewallVista7(portname)
  else
    RemovePortFromFirewallXP(port, protocol);
except
  end;
end;

