/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef FIX_STAT_DENS_CYL_H
#define FIX_STAT_DENS_CYL_H

#include "fix_stat.h"

namespace LAMMPS_NS {

class FixStatDensCyl : public FixStat {
 public:
  FixStatDensCyl(class LAMMPS *, int, char **);
  ~FixStatDensCyl();
  int setmask();
  void end_of_step();
  void write_stat(int);

 private:
  double ***num, ***nmass;
};

}

#endif
