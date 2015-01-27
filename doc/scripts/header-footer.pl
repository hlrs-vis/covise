#! /usr/bin/perl -w

$header="../general/header.tpl";
$footer="../general/footer.tpl";

open(HEADER, "< $header")
	or die "Couldn't open $header for reading: $!\n";
open(FOOTER, "< $footer")
	or die "Couldn't open $footer for reading: $!\n";

while (<HEADER>) {
	print;
}
close(HEADER);

while (<>) {
	if (/^<!--Table of Child-Links-->$/) {
		last;
	}
}

while(<>) {
	if(/<ADDRESS>/) {
		last;
	}
	print;
}

while(<FOOTER>) {
	print;
}
close(FOOTER);

