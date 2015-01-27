#! /usr/bin/perl -w


while (<>) {
	if (/startdocument/) {
		print;
		s/startdocument/tableofchildlinks/;
		print;
	}
	else {
		print;
	}
}



