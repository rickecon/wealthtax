***********************************************************
* Estimating Earnings Processes Project
* 
***********************************************************
* This program reads in the CPS data.  These
* data are created in merge_hours_test.do
*
* The program then calculates mean hours by age and percentile.
*************************************************************


#delimit ;
set more off;
capture clear all;
capture log close;
set memory 8000m;
*cd "/Users/jasondebacker/Econ/Research/EstimatingEarningsProcesses/data" ;
*log using "/Users/jasondebacker/Econ/Research/EstimatingEarningsProcesses/data/cps_tabs.log", replace ;

*local datapath "/Users/jasondebacker/Econ/Research/EstimatingEarningsProcesses/data" ;
*local outpath "/Users/jasondebacker/Econ/Research/EstimatingEarningsProcesses/data/output" ;

sysdir set PLUS "/home/jdebacker/stata/plus" ;
cd "/home/jdebacker/EarningsProcess" ;
log using "/home/jdebacker/EarningsProcess/cps_hours_tabs.log", replace ;
local datapath "/home/jdebacker/EarningsProcess" ;
local outpath "/home/jdebacker/EarningsProcess/output" ;

set matsize 800 ;

use "`datapath'/cps_est_ability_hours_1992to2013.dta", clear ;

keep year age hours hours_unit wtsupp ;

drop if year < 1992 ;
drop if year > 2013 ;
drop if age < 20 ;
drop if age > 80 ;

/* create percentile of hours worked by age */
egen hours_unit_pct = xtile(hours_unit), by(age) nq(100) weights(wtsupp) ;
egen hours_pct = xtile(hours), by(age) nq(100) weights(wtsupp) ;

save temp1, replace ;
/* collapse data */
collapse (mean) mean_hrs_unit=hours_unit (count) num_obs=hours_unit [iw=wtsupp], by(age hours_unit_pct) ;
format * %23.5f ;
outsheet using "`outpath'/cps_hours_unit_by_age_hourspct.txt", replace ;


use temp1, clear ;
/* collapse data */
collapse (mean) mean_hrs=hours (count) num_obs=hours [iw=wtsupp], by(age hours_pct) ;
format * %23.5f ;
outsheet using "`outpath'/cps_hours_by_age_hourspct.txt", replace ;

capture log close ;