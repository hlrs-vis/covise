proc LICENSEITEM { key name date hostname} {
    global ListForSection
    set pKey [getPrimaryKey]
    lappend ListForSection(License,pKey,$hostname) $pKey
    set ListForSection(License,$pKey,key,$hostname) $key
    set ListForSection(License,$pKey,name,$hostname) $name
    set ListForSection(License,$pKey,date,$hostname) $date
}

proc makeLicenseEntry {frame pKey row hostname} {
    global ListForSection
    entry $frame.key${row} -width 60 -textvariable ListForSection(License,$pKey,key,$hostname)
    grid  $frame.key${row} -row $row -column 0 -sticky w
    entry $frame.name${row} -width 12 -textvariable ListForSection(License,$pKey,name,$hostname)
    grid $frame.name${row} -row $row -column 1 -sticky w
    entry $frame.date${row} -width 12 -textvariable ListForSection(License,$pKey,date,$hostname)
    grid $frame.date${row} -row $row -column 2 -sticky w

    button $frame.delete$row -text delete -command "deleteLicenseEntry $frame $pKey $row $hostname"
    grid $frame.delete${row} -row $row -column 3 -sticky w
}

proc deleteLicenseEntry {frame pKey row hostname} {
    global ListForSection
    set pos [lsearch $ListForSection(License,pKey,$hostname) $pKey]
    set ListForSection(License,$hostname) [lreplace $ListForSection(License,pKey,$hostname) $pos $pos]
    unset ListForSection(License,$pKey,key,$hostname)
    unset ListForSection(License,$pKey,name,$hostname)
    unset ListForSection(License,$pKey,date,$hostname)
    
    destroy $frame.key${row}
    destroy $frame.name${row}
    destroy $frame.date${row}
    destroy $frame.delete${row}
}


proc makeLicenseGUI {hostname pk } {
    set f [getFrame License $pk]
    global ListForSection
    global Counters
    label $f.key -text Key
    label $f.name -text Name
    label $f.date -text "Expiration date"
    grid $f.key -row 0 -column 0 -sticky w
    grid $f.name -row 0 -column 1  -sticky w 
    grid $f.date -row 0 -column 2 -sticky w
    
    set row 1
    
    if [info exists ListForSection(License,pKey,$hostname)] {
        foreach pKey $ListForSection(License,pKey,$hostname) {
            makeLicenseEntry $f $pKey $row $hostname
            incr row
        }
    }
    set Counters(LicenseRows,$hostname) $row


    button $f.add -text "Add" -command "addLicenseEntry $f $hostname"
    grid $f.add -row 1 -column 4
}

proc addLicenseEntry {frame hostname} {
    global Counters
    LICENSEITEM VOID VOID VOID $hostname
    makeLicenseEntry $frame [getLastPrimaryKey] $Counters(LicenseRows,$hostname) $hostname
    incr Counters(LicenseRows,$hostname)
}













