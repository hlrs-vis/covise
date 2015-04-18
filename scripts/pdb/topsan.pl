#!/usr/bin/perl -w
use strict;

use LWP::Simple;
use LWP::UserAgent;
use HTTP::Request;
use HTTP::Response;
use HTML::LinkExtor;
use Getopt::Std;
#use XML::LibXML;


my $var = $ARGV[0];

my $URL = "http://www.topsan.org/explore?pdbId=$var";

#print "$URL\n";

my $browser = LWP::UserAgent->new();
$browser->timeout(10);

my $request = HTTP::Request->new(GET => $URL);
my $response = $browser->request($request);
if ($response->is_error()) {printf "%s\n", $response->status_line;}

my $TAGRX = "<div id=\"pdbPath\">.*?(http://www.topsan.org/\@api.*?)</div";
#my $TAGRX = "(.*?\&)";
my $contents = $response->content();

my $topsanurl = "";
my $xmlrequest;
my $xmlbrowser;
my $xmlresponse;
my $xmlcontent;

open(OUTFILE, ">topsan.dat");
#print OUTFILE "$contents\n";
#print "$contents\n";
if ( $contents =~ m/$TAGRX/s ) {
   $topsanurl = $1;
   $topsanurl =~ m/(.*?)\&/s;
   print "$1\n";
   $topsanurl = $1;
   if($topsanurl =~ m/^http/ )
   {
      if($topsanurl =~ m/([^\ ]*)/)
      {
         $topsanurl = $1;
      }
      $topsanurl = $topsanurl . "/contents?mode=edit&format=xhtml";
      print "$topsanurl\n";
      print OUTFILE "IN TOPSAN\n";
      $xmlbrowser = LWP::UserAgent->new();
      $xmlbrowser->timeout(10);
      $xmlrequest = HTTP::Request->new(GET => $topsanurl);
      $xmlresponse = $xmlbrowser->request($xmlrequest);
      $xmlcontent = $xmlresponse->content();
      #print "$xmlcontent\n";
      $xmlcontent =~ s/&gt;/>/g;
      $xmlcontent =~ s/&lt;/</g;
      $xmlcontent =~ s/&nbsp;/ /g;
      $xmlcontent =~ s/&amp;/&/g;
      #print "$xmlcontent\n";
      my $title = "";
      if($xmlcontent =~ m/title:\"(.*?)\"/)
      {
        $title = $1;
      }
      print OUTFILE "$title\n";
      my $site = "";
      if($xmlcontent =~ m/site:\'(.*?)\'/)
      {
         $site = $1;
      }
      print OUTFILE "$site\n";
      my $pdbid = "";
      if($xmlcontent =~ m/pdbid:\"(.*?)\"/)
      {
         $pdbid = $1;
      }
      print OUTFILE "$pdbid\n";
      my $name = "";
      if($xmlcontent =~ m/name:\"(.*?)\"/)
      {
         $name = $1;
      }
      print OUTFILE "$name\n";
      my $source = "";
      if($xmlcontent =~ m/source:\"(.*?)\"/)
      {
         $source = $1;
      }
      print OUTFILE "$source\n";
      my $refids = "";
      if($xmlcontent =~ m/refids:\"(.*?)\"/)
      {
         $refids = $1;
      }
      $refids =~ s/ //g;
      print OUTFILE "$refids\n";
      my $weight = "";
      if($xmlcontent =~ m/molwt:\"(.*?)\"/)
      {
         $weight = $1;
      }
      $weight =~ s/ //g;
      print OUTFILE "$weight\n";
      my $residues = "";
      if($xmlcontent =~ m/residues:\"(.*?)\"/)
      {
         $residues = $1;
      }
      print OUTFILE "$residues\n";
      my $isoelec = "";
      if($xmlcontent =~ m/isopoint:\"(.*?)\"/)
      {
         $isoelec = $1;
      }
      $isoelec =~ s/ //g;
      print OUTFILE "$isoelec\n";
      my $sequ = "";
      if($xmlcontent =~ m/sequence:\"(.*?)\"/)
      {
         $sequ = $1;
      }
      $sequ =~ s/ //g;
      print OUTFILE "$sequ\n";
      my $ligands = "";
      if($xmlcontent =~ m/ligands:\"(.*?)\"/)
      {
         $ligands = $1;
      }
      print OUTFILE "$ligands\n";
      my $metals = "";
      if($xmlcontent =~ m/metals:\"(.*?)\"/)
      {
         $metals = $1;
      }
      print OUTFILE "$metals\n";
      $xmlcontent =~ m/Protein Summary<\/h4>(.*?)<h4 class=\"topsan h4/s;
      my $summary = $1;
      my $fnum = 1;
      my $file;
      while($summary =~ m/<img.*?src="(.*?)"/)
      {
	 $file = $1;
	 if($file =~ m/^\/\@api/)
         {
	     $file = "http://proteins.burnham.org" . $file;
	 }
         #print OUTFILE "$file\n";
         my @args = ("wget", "$file", "--output-file=wget.out", "-nc");
         system(@args);
         open(WGETFILE, "<wget.out");
         my @wget = <WGETFILE>;
	 my $saved = 0;
         foreach(@wget)
         {
            if($_ =~ m/Saving to: `(.*?)'/)
            {
               print OUTFILE "$1\n";
	       $saved = 1;
            }
            else 
            {
               if($_ =~ m/File `(.*?)' already there/)
               {
                  print OUTFILE "$1\n";
		  $saved = 1;
               }
	       else
	       {
	       }
            }
         }
	 if($saved)
         {
	     $summary =~ s/<img.*?\/>/[Figure$fnum]/;
	     $fnum = $fnum + 1;
	 }
	 else
	 {
	     $summary =~ s/<img.*?\/>//;
	 }
      }
      print OUTFILE "IMAGE END\n";
      $summary =~ s/<br[^<]*?\/>/\n/g;
      $summary =~ s/<[^<]*?>//g;
      $summary =~ s/&nbsp;/ /g;
      $summary =~ s/&amp;/&/g;
      $summary =~ s/Â|//g;
      $summary =~ s/\240//g;
      while($summary =~ s/\n\n\n/\n\n/)
      {
      }
      $summary =~ /^\n*(.*?)\n*$/s;
      print OUTFILE "$1";
   }
   else
   {
      print OUTFILE "NOT IN TOPSAN";
   }
}
else
{
   print OUTFILE "NOT IN TOPSAN";
}

