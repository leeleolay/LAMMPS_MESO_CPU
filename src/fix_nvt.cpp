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

/* ----------------------------------------------------------------------
   Contributing author: Mark Stevens (SNL)
------------------------------------------------------------------------- */

#include "string.h"
#include "stdlib.h"
#include "math.h"
#include "fix_nvt.h"
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "group.h"
#include "update.h"
#include "respa.h"
#include "modify.h"
#include "compute.h"
#include "error.h"

using namespace LAMMPS_NS;

enum{NOBIAS,BIAS};

/* ---------------------------------------------------------------------- */

FixNVT::FixNVT(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 7) error->all("Illegal fix nvt command");

  restart_global = 1;
  time_integrate = 1;
  scalar_flag = 1;
  scalar_vector_freq = 1;
  extscalar = 1;

  t_start = atof(arg[4]);
  t_stop = atof(arg[5]);
  double t_period = atof(arg[6]);

  if (narg == 7) drag = 0.0;
  else if (narg == 9 && strcmp(arg[7],"drag") == 0) drag = atof(arg[8]);
  else error->all("Illegal fix nvt command");
  
  // error checks
  // convert input period to frequency

  if (t_start < 0.0 || t_stop <= 0.0)
    error->all("Target T for fix nvt cannot be 0.0");
  if (t_period <= 0.0) error->all("Fix nvt period must be > 0.0");
  t_freq = 1.0 / t_period;

  // create a new compute temp style
  // id = fix-ID + temp, compute group = fix group

  int n = strlen(id) + 6;
  id_temp = new char[n];
  strcpy(id_temp,id);
  strcat(id_temp,"_temp");

  char **newarg = new char*[3];
  newarg[0] = id_temp;
  newarg[1] = group->names[igroup];
  if (strcmp(style,"nvt") == 0) 
    newarg[2] = (char *) "temp";
  else if (strcmp(style,"nvt/asphere") == 0) 
    newarg[2] = (char *) "temp/asphere";
  else if (strcmp(style,"nvt/sllod") == 0) 
    newarg[2] = (char *) "temp/deform";
  else if (strcmp(style,"nvt/sphere") == 0) 
    newarg[2] = (char *) "temp/sphere";
  modify->add_compute(3,newarg);
  delete [] newarg;
  tflag = 1;

  // Nose/Hoover temp init

  eta = eta_dot = 0.0;
}

/* ---------------------------------------------------------------------- */

FixNVT::~FixNVT()
{
  // delete temperature if fix created it

  if (tflag) modify->delete_compute(id_temp);
  delete [] id_temp;
}

/* ---------------------------------------------------------------------- */

int FixNVT::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= THERMO_ENERGY;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNVT::init()
{
  int icompute = modify->find_compute(id_temp);
  if (icompute < 0) error->all("Temp ID for fix nvt does not exist");
  temperature = modify->compute[icompute];

  if (temperature->tempbias) which = BIAS;
  else which = NOBIAS;

  // set timesteps and frequencies

  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  dthalf = 0.5 * update->dt;
  drag_factor = 1.0 - (update->dt * t_freq * drag);

  if (strcmp(update->integrate_style,"respa") == 0) {
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
    step_respa = ((Respa *) update->integrate)->step;
  }
}

/* ---------------------------------------------------------------------- */

void FixNVT::setup(int vflag)
{
  t_target = t_start;                         // used by compute_scalar()
  t_current = temperature->compute_scalar();
}

/* ---------------------------------------------------------------------- */

void FixNVT::initial_integrate(int vflag)
{
  double dtfm;

  double delta = update->ntimestep - update->beginstep;
  delta /= update->endstep - update->beginstep;
  t_target = t_start + delta * (t_stop-t_start);

  // update eta_dot

  f_eta = t_freq*t_freq * (t_current/t_target - 1.0);
  eta_dot += f_eta*dthalf;
  eta_dot *= drag_factor;
  eta += dtv*eta_dot;
  factor = exp(-dthalf*eta_dot);

  // update v and x of only atoms in group

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  if (rmass) {
    if (which == NOBIAS) {
      for (int i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  dtfm = dtf / rmass[i];
	  v[i][0] = v[i][0]*factor + dtfm*f[i][0];
	  v[i][1] = v[i][1]*factor + dtfm*f[i][1];
	  v[i][2] = v[i][2]*factor + dtfm*f[i][2];
	  x[i][0] += dtv * v[i][0];
	  x[i][1] += dtv * v[i][1];
	  x[i][2] += dtv * v[i][2];
	}
      }

    } else if (which == BIAS) {
      for (int i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  temperature->remove_bias(i,v[i]);
	  dtfm = dtf / rmass[i];
	  v[i][0] = v[i][0]*factor + dtfm*f[i][0];
	  v[i][1] = v[i][1]*factor + dtfm*f[i][1];
	  v[i][2] = v[i][2]*factor + dtfm*f[i][2];
	  temperature->restore_bias(i,v[i]);
	  x[i][0] += dtv * v[i][0];
	  x[i][1] += dtv * v[i][1];
	  x[i][2] += dtv * v[i][2];
	}
      }
    }

  } else {
    if (which == NOBIAS) {
      for (int i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  dtfm = dtf / mass[type[i]];
	  v[i][0] = v[i][0]*factor + dtfm*f[i][0];
	  v[i][1] = v[i][1]*factor + dtfm*f[i][1];
	  v[i][2] = v[i][2]*factor + dtfm*f[i][2];
	  x[i][0] += dtv * v[i][0];
	  x[i][1] += dtv * v[i][1];
	  x[i][2] += dtv * v[i][2];
	}
      }

    } else if (which == BIAS) {
      for (int i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  temperature->remove_bias(i,v[i]);
	  dtfm = dtf / mass[type[i]];
	  v[i][0] = v[i][0]*factor + dtfm*f[i][0];
	  v[i][1] = v[i][1]*factor + dtfm*f[i][1];
	  v[i][2] = v[i][2]*factor + dtfm*f[i][2];
	  temperature->restore_bias(i,v[i]);
	  x[i][0] += dtv * v[i][0];
	  x[i][1] += dtv * v[i][1];
	  x[i][2] += dtv * v[i][2];
	}
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixNVT::final_integrate()
{
  double dtfm;

  // update v of only atoms in group

  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  if (rmass) {
    if (which == NOBIAS) {
      for (int i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  dtfm = dtf / rmass[i] * factor;
	  v[i][0] = v[i][0]*factor + dtfm*f[i][0];
	  v[i][1] = v[i][1]*factor + dtfm*f[i][1];
	  v[i][2] = v[i][2]*factor + dtfm*f[i][2];
	}
      }
      
    } else if (which == BIAS) {
      for (int i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  temperature->remove_bias(i,v[i]);
	  dtfm = dtf / rmass[i] * factor;
	  v[i][0] = v[i][0]*factor + dtfm*f[i][0];
	  v[i][1] = v[i][1]*factor + dtfm*f[i][1];
	  v[i][2] = v[i][2]*factor + dtfm*f[i][2];
	  temperature->restore_bias(i,v[i]);
	}
      }
    }

  } else {
    if (which == NOBIAS) {
      for (int i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  dtfm = dtf / mass[type[i]] * factor;
	  v[i][0] = v[i][0]*factor + dtfm*f[i][0];
	  v[i][1] = v[i][1]*factor + dtfm*f[i][1];
	  v[i][2] = v[i][2]*factor + dtfm*f[i][2];
	}
      }
      
    } else if (which == BIAS) {
      for (int i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  temperature->remove_bias(i,v[i]);
	  dtfm = dtf / mass[type[i]] * factor;
	  v[i][0] = v[i][0]*factor + dtfm*f[i][0];
	  v[i][1] = v[i][1]*factor + dtfm*f[i][1];
	  v[i][2] = v[i][2]*factor + dtfm*f[i][2];
	  temperature->restore_bias(i,v[i]);
	}
      }
    }
  }

  // compute current T

  t_current = temperature->compute_scalar();

  // update eta_dot

  f_eta = t_freq*t_freq * (t_current/t_target - 1.0);
  eta_dot += f_eta*dthalf;
  eta_dot *= drag_factor;
}

/* ---------------------------------------------------------------------- */

void FixNVT::initial_integrate_respa(int vflag, int ilevel, int flag)
{
  if (flag) return;             // only used by NPT,NPH

  // set timesteps by level

  double dtfm;
  dtv = step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  dthalf = 0.5 * step_respa[ilevel];

  // atom quantities

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // outermost level - update eta_dot and apply to v with factor
  // all other levels - NVE update of v (factor = 1)
  // innermost level - also update x

  if (ilevel == nlevels_respa-1) {
    double delta = update->ntimestep - update->beginstep;
    delta /= update->endstep - update->beginstep;
    t_target = t_start + delta * (t_stop-t_start);

    // update eta_dot
    
    f_eta = t_freq*t_freq * (t_current/t_target - 1.0);
    eta_dot += f_eta*dthalf;
    eta_dot *= drag_factor;
    eta += dtv*eta_dot;
    factor = exp(-dthalf*eta_dot);
  } else factor = 1.0;

  if (rmass) {
    if (which == NOBIAS) {
      for (int i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  dtfm = dtf / rmass[i];
	  v[i][0] = v[i][0]*factor + dtfm*f[i][0];
	  v[i][1] = v[i][1]*factor + dtfm*f[i][1];
	  v[i][2] = v[i][2]*factor + dtfm*f[i][2];
	}
      }

    } else if (which == BIAS) {
      for (int i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  temperature->remove_bias(i,v[i]);
	  dtfm = dtf / rmass[i];
	  v[i][0] = v[i][0]*factor + dtfm*f[i][0];
	  v[i][1] = v[i][1]*factor + dtfm*f[i][1];
	  v[i][2] = v[i][2]*factor + dtfm*f[i][2];
	  temperature->restore_bias(i,v[i]);
	}
      }
    }

  } else {
    if (which == NOBIAS) {
      for (int i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  dtfm = dtf / mass[type[i]];
	  v[i][0] = v[i][0]*factor + dtfm*f[i][0];
	  v[i][1] = v[i][1]*factor + dtfm*f[i][1];
	  v[i][2] = v[i][2]*factor + dtfm*f[i][2];
	}
      }

    } else if (which == BIAS) {
      for (int i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  temperature->remove_bias(i,v[i]);
	  dtfm = dtf / mass[type[i]];
	  v[i][0] = v[i][0]*factor + dtfm*f[i][0];
	  v[i][1] = v[i][1]*factor + dtfm*f[i][1];
	  v[i][2] = v[i][2]*factor + dtfm*f[i][2];
	  temperature->restore_bias(i,v[i]);
	}
      }
    }
  }

  if (ilevel == 0) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
	x[i][0] += dtv * v[i][0];
	x[i][1] += dtv * v[i][1];
	x[i][2] += dtv * v[i][2];
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixNVT::final_integrate_respa(int ilevel)
{
  // set timesteps by level

  double dtfm;
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  dthalf = 0.5 * step_respa[ilevel];

  // outermost level - update eta_dot and apply to v via final_integrate()
  // all other levels - NVE update of v

  if (ilevel == nlevels_respa-1) final_integrate();
  else {
    double **v = atom->v;
    double **f = atom->f;
    double *rmass = atom->rmass;
    double *mass = atom->mass;
    int *type = atom->type;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    if (igroup == atom->firstgroup) nlocal = atom->nfirst;

    if (rmass) {
      for (int i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  dtfm = dtf / rmass[i];
	  v[i][0] += dtfm*f[i][0];
	  v[i][1] += dtfm*f[i][1];
	  v[i][2] += dtfm*f[i][2];
	}
      }
      
    } else {
      for (int i = 0; i < nlocal; i++) {
	if (mask[i] & groupbit) {
	  dtfm = dtf / mass[type[i]];
	  v[i][0] += dtfm*f[i][0];
	  v[i][1] += dtfm*f[i][1];
	  v[i][2] += dtfm*f[i][2];
	}
      }
    }
  }
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write 
------------------------------------------------------------------------- */

void FixNVT::write_restart(FILE *fp)
{
  int n = 0;
  double list[2];
  list[n++] = eta;
  list[n++] = eta_dot;

  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(&list,sizeof(double),n,fp);
  }
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix 
------------------------------------------------------------------------- */

void FixNVT::restart(char *buf)
{
  int n = 0;
  double *list = (double *) buf;
  eta = list[n++];
  eta_dot = list[n++];
}

/* ---------------------------------------------------------------------- */

int FixNVT::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"temp") == 0) {
    if (narg < 2) error->all("Illegal fix_modify command");
    if (tflag) {
      modify->delete_compute(id_temp);
      tflag = 0;
    }
    delete [] id_temp;
    int n = strlen(arg[1]) + 1;
    id_temp = new char[n];
    strcpy(id_temp,arg[1]);

    int icompute = modify->find_compute(id_temp);
    if (icompute < 0) error->all("Could not find fix_modify temp ID");
    temperature = modify->compute[icompute];

    if (temperature->tempflag == 0)
      error->all("Fix_modify temp ID does not compute temperature");
    if (temperature->igroup != igroup && comm->me == 0)
      error->warning("Group for fix_modify temp != fix group");
    return 2;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

void FixNVT::reset_target(double t_new)
{
  t_start = t_stop = t_new;
}

/* ---------------------------------------------------------------------- */

void FixNVT::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  dthalf = 0.5 * update->dt;
  drag_factor = 1.0 - (update->dt * t_freq * drag);
}

/* ---------------------------------------------------------------------- */

double FixNVT::compute_scalar()
{
  double ke = temperature->dof * force->boltz * t_target;
  double energy = ke * (eta + 0.5*eta_dot*eta_dot/(t_freq*t_freq));
  return energy;
}
