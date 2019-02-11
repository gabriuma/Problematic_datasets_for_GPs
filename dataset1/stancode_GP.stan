functions {

	# SE kernel
	vector gp_SE(real[] x, real sdgp, real lscale, vector zgp) { 
		int n= size(x);
		matrix[n, n] cov;
		
		cov = cov_exp_quad(x, sdgp, lscale);
		
		for (i in 1:n) {
			cov[i, i] = cov[i, i] + 1e-12;
		}
		return cholesky_decompose(cov) * zgp;
	}
	
	# Mattern kernel
	vector gp_mattern(real[] x, real sdgp, real lscale, vector zgp, int v) {
		int n= size(x);	
		matrix[n, n] cov;
				
		# Mattern 3/2
		if (v==3) {
			for (i in 1:n) {
				for (j in 1:n) {
					cov[i,j] = sdgp^2 * (1 + (sqrt(3)*(fabs(x[i]-x[j])))/lscale) * exp(-sqrt(3)*(fabs(x[i]-x[j]))/lscale);
				}
			}
		}
		
		# Mattern 5/2
		if (v==5) {
			for (i in 1:n) {
				for (j in 1:n) {
					cov[i,j] = sdgp^2 * (1 + (sqrt(3)*(fabs(x[i]-x[j])))/lscale + (5*(fabs(x[i]-x[j]))^2)/(3*lscale^2)) * exp(-sqrt(5)*(fabs(x[i]-x[j]))/lscale);
				}
			}
		}
		
		for (i in 1:n) {
			cov[i, i] = cov[i, i] + 1e-12;
		}
		return cholesky_decompose(cov) * zgp;
	}
	
	vector gp_SE_rng(real[] x2,
					 vector f1, real[] x1,
					 real sdgp, real lscale, real delta) {
	int N1 = rows(f1);
	int N2 = size(x2);
	vector[N2] f2;
	{
	  matrix[N1, N1] K =   cov_exp_quad(x1, sdgp, lscale)
						 + diag_matrix(rep_vector(delta, N1));
	  matrix[N1, N1] L_K = cholesky_decompose(K);

	  vector[N1] L_K_div_f1 = mdivide_left_tri_low(L_K, f1);
	  vector[N1] K_div_f1 = mdivide_right_tri_low(L_K_div_f1', L_K)';
	  matrix[N1, N2] k_x1_x2 = cov_exp_quad(x1, x2, sdgp, lscale);
	  vector[N2] f2_mu = (k_x1_x2' * K_div_f1);
	  matrix[N1, N2] v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
	  matrix[N2, N2] cov_f2 =   cov_exp_quad(x2, sdgp, lscale) - v_pred' * v_pred
							  + diag_matrix(rep_vector(delta, N2));
	  f2 = multi_normal_rng(f2_mu, cov_f2);
	}
	return f2;
	}

	vector gp_Matern_rng(real[] x2, vector f1, real[] x1, real sdgp, real lscale, real delta, real v) {
	int N1 = rows(f1);
	int N2 = size(x2);
	vector[N2] f2;
	matrix[N1, N1] K;
	matrix[N1, N2] k_x1_x2;
	matrix[N2, N2] cov_f2;
	matrix[N1, N1] L_K;
	vector[N1] L_K_div_f1;
	vector[N1] K_div_f1;
	vector[N2] f2_mu;
	matrix[N1, N2] v_pred;
	{
	  for (i in 1:N1) {
		for (j in i:N1) {
			if (v==3) {
			K[i,j] = sdgp^2 * (1 + (sqrt(3)*(fabs(x1[i]-x1[j])))/lscale) * exp(-sqrt(3)*(fabs(x1[i]-x1[j]))/lscale);
			}
			if (v==5) {
			K[i,j] = sdgp^2 * (1 + (sqrt(3)*(fabs(x1[i]-x1[j])))/lscale + (5*(fabs(x1[i]-x1[j]))^2)/(3*lscale^2)) * exp(-sqrt(5)*(fabs(x1[i]-x1[j]))/lscale);
			}
		}
	  }
	  for (i in 1:N1) {
	     K[i,i] = K[i, i] + delta;
	     for (j in (i+1):N1) {
			K[j, i] = K[i, j];
		 }
	  }
	  
	  L_K = cholesky_decompose(K);
	  L_K_div_f1 = mdivide_left_tri_low(L_K, f1);
	  K_div_f1 = mdivide_right_tri_low(L_K_div_f1', L_K)';

	  for (i in 1:N1) {
		for (j in 1:N2) {
			if (v==3) {
			k_x1_x2[i,j] = sdgp^2 * (1 + (sqrt(3)*(fabs(x1[i]-x2[j])))/lscale) * exp(-sqrt(3)*(fabs(x1[i]-x2[j]))/lscale);
			}
			if (v==5) {
			k_x1_x2[i,j] = sdgp^2 * (1 + (sqrt(3)*(fabs(x1[i]-x2[j])))/lscale + (5*(fabs(x1[i]-x2[j]))^2)/(3*lscale^2)) * exp(-sqrt(5)*(fabs(x1[i]-x2[j]))/lscale);
			}
		}
	  }
	  
	  f2_mu = (k_x1_x2' * K_div_f1); //'
	  v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
	  
	  for (i in 1:N2) {
		for (j in i:N2) {
			if (v==3) {
			cov_f2[i,j] = sdgp^2 * (1 + (sqrt(3)*(fabs(x2[i]-x2[j])))/lscale) * exp(-sqrt(3)*(fabs(x2[i]-x2[j]))/lscale);
			}
			if (v==5) {
			cov_f2[i,j] = sdgp^2 * (1 + (sqrt(3)*(fabs(x2[i]-x2[j])))/lscale + (5*(fabs(x2[i]-x2[j]))^2)/(3*lscale^2)) * exp(-sqrt(5)*(fabs(x2[i]-x2[j]))/lscale);
			}
		}
	  }
	  for (i in 1:N2) {
	     for (j in (i+1):N2) {
			cov_f2[j, i] = cov_f2[i, j];
		 }
	  }	  
	  
	  cov_f2 =   cov_f2 - v_pred' * v_pred + diag_matrix(rep_vector(delta, N2)); //'
	  f2 = multi_normal_rng(f2_mu, cov_f2);
	}
	return f2;
	}
	
}

data {
	int<lower=1> n;			
	real x[n];			
	vector[n] y;		
}

transformed data{
}

parameters {
	real<lower=0> lscale;
	real<lower=0> sdgp;
	real<lower=0> sigma;
	vector[n] eta;
}

transformed parameters{
	vector[n] f;

	# f = gp_mattern(x, sdgp, lscale, eta, param_v);
	f = gp_SE(x, sdgp, lscale, eta);	
}

model{
	eta ~ normal(0, 1);
	lscale ~ gamma(2.5,50); 
	sigma ~ gamma(1,1); 
	sdgp ~ gamma(10,10); 
	
	y ~ normal(f, sigma); 
}

generated quantities{
}
