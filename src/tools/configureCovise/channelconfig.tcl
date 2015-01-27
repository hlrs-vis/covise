proc CHANNELCONFIGITEM { channelNo name winNo viewPortXMin viewPortYMin viewPortXMax viewPortYMax hostname } { 
        global ListForSection
        set ListForSection(ChannelConfig,name,$channelNo,$hostname) $name
        set ListForSection(ChannelConfig,winNo,$channelNo,$hostname) $winNo
        set ListForSection(ChannelConfig,viewPortXMin,$channelNo,$hostname) $viewPortXMin
        set ListForSection(ChannelConfig,viewPortYMin,$channelNo,$hostname) $viewPortYMin
        set ListForSection(ChannelConfig,viewPortXMax,$channelNo,$hostname) $viewPortXMax
        set ListForSection(ChannelConfig,viewPortYMax,$channelNo,$hostname) $viewPortYMax
}



proc saveChannelConfig { body hostname } {
    upvar $body b
    global ListForSection
    for { set i 0 } { $i < $ListForSection(COVERConfig,NUM_SCREENS,$hostname) } { incr i } {
        if { [canBeSaved ListForSection(ChannelConfig,name,$i,$hostname)] && 
        [canBeSaved ListForSection(ChannelConfig,winNo,$i,$hostname)] &&
        [canBeSaved ListForSection(ChannelConfig,viewPortXMin,$i,$hostname)] &&
        [canBeSaved ListForSection(ChannelConfig,viewPortYMin,$i,$hostname)] &&
        [canBeSaved ListForSection(ChannelConfig,viewPortXMax,$i,$hostname)] &&
        [canBeSaved ListForSection(ChannelConfig,viewPortYMax,$i,$hostname)] }  {
            set b "$b     $i     \
                $ListForSection(ChannelConfig,name,$i,$hostname) \
                $ListForSection(ChannelConfig,winNo,$i,$hostname) \
                $ListForSection(ChannelConfig,viewPortXMin,$i,$hostname) \
                $ListForSection(ChannelConfig,viewPortYMin,$i,$hostname) \
                $ListForSection(ChannelConfig,viewPortXMax,$i,$hostname) \
                $ListForSection(ChannelConfig,viewPortYMax,$i,$hostname)\n"
        }
    }
}


proc makeChannelConfigGUI { hostname pk } {
     global ListForSection
     set f [getFrame ChannelConfig $pk]
     set i 0
     foreach { l t } {number "Channel Number" name "Channel Name"\
        winNo "Window Number"\
        viewPortXMin "X-Min"\
        viewPortYMin "Y-Min"\
        viewPortXMax "X-Max"\
        viewPortYMax "Y-Max" } {
        label $f.$l -text $t
        bind $f.$l <Button-3> "showHelp ChannelConfig $l"
        grid  $f.$l -row 0 -column $i -sticky w
        incr i
        
     }
     for { set i 0 } { $i < $ListForSection(COVERConfig,NUM_SCREENS,$hostname) } { incr i } {
        label $f.lChannelNo$i -text $i
        grid  $f.lChannelNo$i -row [expr $i+1] -column 0
        set j 1
        foreach e {name winNo viewPortXMin viewPortYMin viewPortXMax viewPortYMax } {
                entry $f.e$e$i -textvariable ListForSection(ChannelConfig,$e,$i,$hostname)
                bind $f <Button-3> "showHelp ChannelConfig $t"
                grid $f.e$e$i -column $j -row [expr $i+1]
                incr j
        }

     }
}