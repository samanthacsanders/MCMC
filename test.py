from joint_dist_sampler import DistributionSampler
import node
import matplotlib.pyplot as plt
import matplotlib.pylab as mlab
import cPickle as pickle
from alarm_data import data
import math

def run_faculty(ds):
	num_samples = 1000
	
	#initialize
	init_faculty_evals(ds)
	ds.generate_samples(num_samples)
	
	mean_marginal = get_marginal(ds.samples, 'Mean')
	var_marginal = get_marginal(ds.samples, 'Var')
	
	plot_mixing(mean_marginal, 'Mean')
	plot_mixing(var_marginal, 'Var')
	
	plot_posterior(mean_marginal, ds.prior_pdfs['mean_prior_pdf'], 'Mean', 4.0, 7.0)
	plot_posterior(var_marginal, ds.prior_pdfs['var_prior_pdf'], 'Var', 0.0001, 1.0)

def init_faculty_evals(ds):
		
	# Faculty Evaluation Network
	eval_mean = node.NormalNode(last=2, obs=False, name='Mean', candsd=0.2, mean=5, var=1/9.0)
	eval_var = node.InvGammaNode(last=0.25, obs=False, name='Var', candsd=0.15, alpha=11, beta=2.5)
	ds.nodes = [eval_mean, eval_var]
		
	ds.prior_pdfs = {'mean_prior_pdf': eval_mean.get_prior_pdf, 'var_prior_pdf': eval_var.get_prior_pdf}
		
	data = [6.39, 6.32, 6.25, 6.24, 6.21, 6.18, 6.17, 6.13, 6.00, 6.00, 5.97, 5.82, \
			5.81, 5.71, 5.55, 5.50, 5.39, 5.37, 5.35, 5.30, 5.27, 4.94, 4.50]
		
	for i, d in enumerate(data):
		n = node.NormalNode(last=d, obs=True, name='X_'+str(i), candsd=None, mean=eval_mean, var=eval_var)
		eval_mean.children.append(n)
		eval_var.children.append(n)
		ds.nodes.append(n)

##########################################################################################

def run_golfers(ds):
	num_samples = 10000
	
	init_golfers(ds)
	ds.generate_samples(num_samples)

	hypertournmean_marginal = get_marginal(ds.samples, 'Tournament Hyper Mean')
	hypertournvar_marginal = get_marginal(ds.samples, 'Tournament Hyper Var')
	tournmean_marginal = get_marginal(ds.samples, 'Tournament 1')
	hypergolfervar_marginal = get_marginal(ds.samples, 'Golfer Hyper Var')
	golfervar_marginal = get_marginal(ds.samples, 'VijaySingh')
	obsvar_marginal = get_marginal(ds.samples, 'Observation Var')
		
	plot_mixing(hypertournmean_marginal, 'Tournament Hyper Mean')
	plot_mixing(hypertournvar_marginal, 'Tournament Hyper Var')
	plot_mixing(tournmean_marginal, 'Tournament 1')
	plot_mixing(hypergolfervar_marginal, 'Golfer Hyper Var')
	plot_mixing(golfervar_marginal, 'VijaySingh')
	plot_mixing(obsvar_marginal, 'Observation Var')
		
		
	#ds.plot_posterior(hypertournmean_marginal, ds.prior_pdfs['hypertourn_prior_pdf'], 'Tournament Hyper Mean', 65, 85)
	#ds.plot_posterior(hypertournvar_marginal, ds.prior_pdfs['hypertournvar_prior_pdf'], 'Tournament Hyper Var', 0, 3)
		
	ability = []
	for golfer in ds.golfermean:
		samples = get_marginal(ds.samples, golfer)
		#samples = ds.golfermean[golfer].samples[:]
		samples.sort()
		nsamples = len(samples)
		median = samples[nsamples//2]
		low = samples[int(.05 * nsamples)]
		high = samples[int(.95 * nsamples)]
		ability.append( (golfer, low, median, high) )
    
	ability.sort(lambda x, y: cmp(x[2], y[2]) )
	i = 1
	for golfer, low, median, high in ability:
		print '%d: %s %f; 90%% interval: (%f, %f)' % (i, golfer, median, low, high)
		i += 1
		
def init_golfers(ds):
	data = pickle.load(open('golfdata.p', 'rb'))
	tourns = pickle.load(open('tourns.p', 'rb'))
	golfers = pickle.load(open('golfers.p', 'rb'))
		
	hypertournmean = node.NormalNode(last=72.8, obs=False, name='Tournament Hyper Mean', candsd=2, mean=72, var=2)
	hypertournvar = node.InvGammaNode(last=1, obs=False, name='Tournament Hyper Var', candsd=1.2, alpha=18, beta=1/.015)
	ds.nodes = [hypertournmean, hypertournvar]
		
	tournmean={}
	for tourn in tourns:
		tournmean[tourn] = node.NormalNode(last=72, obs=False, name='Tournament %s'%tourn, candsd=8, mean=hypertournmean, var=hypertournvar)
		hypertournmean.children.append(tournmean[tourn])
		hypertournvar.children.append(tournmean[tourn])
		ds.nodes.append(tournmean[tourn])
			
	hypergolfervar = node.InvGammaNode(last=3.5, obs=False, name='Golfer Hyper Var', candsd=0.5, alpha=18, beta=1/.015)
	ds.nodes.append(hypergolfervar)
		
	ds.golfermean={}
	for golfer in golfers:
		ds.golfermean[golfer] = node.NormalNode(last=0, obs=False, name=golfer, candsd=1.5, mean=0, var=hypergolfervar)
		hypergolfervar.children.append(ds.golfermean[golfer])
		ds.nodes.append(ds.golfermean[golfer])
			
	obsvar = node.InvGammaNode(last=3.1, obs=False, name='Observation Var', candsd=0.1, alpha=83, beta=1/.0014)
	ds.nodes.append(obsvar)
		
	for i, (name, score, tourn) in enumerate(data):
		#print 'tourn: ', tourn
		#print 'tournmean[tourn].last: ', tournmean[tourn].last
		n = node.NormalNode(last=score, obs=True, name='OBS_'+str(i), candsd=None, mean=tournmean[tourn].last+ds.golfermean[name].last, var=obsvar)
		obsvar.children.append(n)
		ds.nodes.append(n)
			
			
	# How many of these nodes do we need to get prior/posterior graphs for?
	ds.prior_pdfs = {'hypertourn_prior_pdf': hypertournmean.get_prior_pdf, 'hypertournvar_prior_pdf': hypertournvar.get_prior_pdf,
					'hypergolfervar_prior_pdf': hypergolfervar.get_prior_pdf, 'obsvar_prior_pdf': obsvar.get_prior_pdf, 
					'golfermean_prior_pdf': ds.golfermean['VijaySingh'].get_prior_pdf, 'tournmean_prior_pdf': tournmean['1'].get_prior_pdf}

##########################################################################################

def run_wacky(ds):
	num_samples = 10000
	init_wacky(ds)
	ds.generate_samples(num_samples)
	
	a_marg = get_marginal(ds.samples, 'A')
	b_marg = get_marginal(ds.samples, 'B')
	c_marg = get_marginal(ds.samples, 'C')
	d_marg = get_marginal(ds.samples, 'D')
	e_marg = get_marginal(ds.samples, 'E')
	f_marg = get_marginal(ds.samples, 'F')
	g_marg = get_marginal(ds.samples, 'G')
	
	plot_mixing(a_marg, 'A')
	plot_mixing(b_marg, 'B')
	plot_mixing(c_marg, 'C')
	plot_mixing(d_marg, 'D')
	plot_mixing(e_marg, 'E')
	plot_mixing(f_marg, 'F')
	plot_mixing(g_marg, 'G')
	
	plot_posterior(a_marg, ds.prior_pdfs['a'], 'A', 10, 30)
	#plot_posterior(b_marg, ds.prior_pdfs['b'], 'B', 0, 50)
	#plot_posterior(c_marg, ds.prior_pdfs['c'], 'C', 0, 50)
	#plot_posterior(d_marg, ds.prior_pdfs['d'], 'D', 0, 50)
	plot_posterior(e_marg, ds.prior_pdfs['e'], 'E', 0, 1)
	#plot_posterior(f_marg, ds.prior_pdfs['f'], 'F', 0, 50)
	#plot_posterior(g_marg, ds.prior_pdfs['g'], 'G', 0, 50)

def init_wacky(ds):
	a = node.NormalNode(last=10, obs=False, name='A', candsd=17, mean=20, var=1)
	e = node.BetaNode(last=0.5, obs=False, name='E', candsd=2, alpha=1, beta=1)
	b = node.GammaPiNode(last=1, obs=False, name='B', candsd=1, alpha=a, beta=7)
	d = node.BetaNode(last=0.5, obs=False, name='D', candsd=0.05, alpha=a, beta=e)
	c = node.BinomialNode(last=0, obs=False, name='C', candsd=1, n=1, p=d)
	f = node.PoissonNode(last=1, obs=False, name='F', candsd=1, rate=d)
	g = node.NormalNode(last=5, obs=True, name='G', candsd=1, mean=e, var=f)
    	
	a.children = [b, d]
	e.children = [d, g]
	d.children = [f, c]
	f.children = [g]

	ds.nodes = [a, e, b, d, c, f, g]
    	
	ds.prior_pdfs = {'a': a.get_prior_pdf, 'b': b.get_prior_pdf, 'c': c.get_prior_pdf,
    				'e': e.get_prior_pdf, 'f': f.get_prior_pdf, 'g': g.get_prior_pdf}
    				
##########################################################################################

def run_beta_binomial_test(ds):
	num_samples = 10000
	init_beta_binomial_test(ds)
	ds.generate_samples(num_samples)
	
	b_marginal = get_marginal(ds.samples, 'B')
	x_marginal = get_marginal(ds.samples, 'X')
	plot_mixing(b_marginal, 'B')
	plot_posterior(b_marginal, ds.prior_pdfs['b'], 'B', 0, 1.0)

def init_beta_binomial_test(ds):
	b = node.BetaNode(last=0.5, obs=False, name='B', candsd=0.2, alpha=2, beta=3)
	x = node.BinomialNode(last=1, obs=True, name='X', candsd=0.2, n=1, p=b, parents=[b])
	ds.nodes=[b,x]
	b.children.append(x)
	ds.prior_pdfs={'b': b.get_prior_pdf, 'x': x.get_prior_pdf}
	
##########################################################################################

def run_gamma_poisson_test(ds):	
	num_samples = 10000
	init_gamma_poisson_test(ds)
	ds.generate_samples(num_samples)
	
	l_marginal = get_marginal(ds.samples, 'L')
	y_marginal = get_marginal(ds.samples, 'Y')
	plot_mixing(l_marginal, 'L')
	plot_posterior(l_marginal, ds.prior_pdfs['l'], 'L', 0, 5.0)
	
def init_gamma_poisson_test(ds):
	l = node.GammaNode(last=3, obs=False, name='L', candsd=0.2, alpha=2, beta=3)
	y = node.PoissonNode(last=3, obs=True, name='Y', candsd=0.2, rate=l)
	ds.nodes=[l, y]
	l.children.append(y)
	ds.prior_pdfs={'l': l.get_prior_pdf, 'y': y.get_prior_pdf}
	
##########################################################################################
	
def run_nozomu_net(ds):
	num_samples = 10000
	init_nozomu_net(ds)
	ds.generate_samples(num_samples)
	
	prog_marg = get_marginal(ds.samples, 'Progress')
	enc_marg = get_marginal(ds.samples, 'Encounters')
	
	plot_mixing(enc_marg, 'Encounters')
	plot_mixing(prog_marg, 'Progress')
	
	enc_marg.sort()
	nsamples = len(enc_marg)
	median = enc_marg[nsamples//2]
	print 'median: ', median
	
	
def init_nozomu_net(ds):
	progress = node.BetaNode(last=0.1, obs=False, name='Progress', candsd=0.2, alpha=2, beta=15)
	encounters = node.PoissonNode(last=1, obs=False, name='Encounters', candsd=1, rate=1/float(progress.last))
	ds.nodes=[progress, encounters]
	progress.children=[encounters]
	ds.prior_pdfs={'progress': progress.get_prior_pdf, 'encounters': encounters.get_prior_pdf}

##########################################################################################

def run_judge_net(ds):
	num_samples = 10000
	init_judge_net(ds)
	ds.generate_samples(num_samples)
	
	judge1_marg = get_marginal(ds.samples, 'Judge1')
	judge2_marg = get_marginal(ds.samples, 'Judge2')
	judge3_marg = get_marginal(ds.samples, 'Judge3')
	score_marg = get_marginal(ds.samples, 'Score')
	
	plot_mixing(judge1_marg, 'Judge1')
	plot_mixing(judge2_marg, 'Judge2')
	plot_mixing(judge3_marg, 'Judge3')
	plot_mixing(score_marg, 'Score')
	
	plot_posterior(judge1_marg, ds.prior_pdfs['judge1'], 'Judge1', 0, 10)
	#plot_posterior(judge2_marg, ds.prior_pdfs['judge2'], 'Judge2', 0, 15)
	plot_posterior(judge3_marg, ds.prior_pdfs['judge3'], 'Judge3', 0, 20)
	
def init_judge_net(ds):
	judge1 = node.NormalNode(last=5, obs=False, name='Judge1', candsd=0.5, mean=5, var=1)
	judge2 = node.PoissonNode(last=5, obs=False, name='Judge2', candsd=2, rate=6)
	judge3 = node.GammaNode(last=6, obs=False, name='Judge3', candsd=1, alpha=7.5, beta=1)
	score = node.NormalNode(last=5, obs=False, name='Score', candsd=0.3, mean=(judge1.last+judge2.last+judge3.last)/3.0, var=0.5)
	ds.nodes=[judge1, judge2, judge3, score]
	judge1.children=[score]
	judge2.children=[score]
	judge3.children=[score]
	ds.prior_pdfs={'judge1': judge1.get_prior_pdf, 'judge2': judge2.get_prior_pdf, 
					'judge3': judge3.get_prior_pdf, 'score': score.get_prior_pdf}

##########################################################################################

def run_my_net(ds):
	num_samples = 10000
	init_my_net(ds)
	ds.generate_samples(num_samples)
	
	a_marg = get_marginal(ds.samples, 'A')
	b_marg = get_marginal(ds.samples, 'B')

	plot_mixing(a_marg, 'A')
	plot_mixing(b_marg, 'B')
	
	#plot_posterior(a_marg, ds.prior_pdfs['a'], 'A', 7, 13)
	
	b_marg.sort()
	nsamples = len(b_marg)
	median = b_marg[nsamples//2]
	print 'median: ', median

def init_my_net(ds):
	a = node.PoissonNode(last=3, obs=False, name='A', candsd=1, rate=6)
	b = node.GammaNode(last=0.75, obs=False, name='B', candsd=1, alpha=3, beta=math.sqrt(a.last))
	ds.nodes=[a,b]
	ds.prior_pdfs={'a': a.get_prior_pdf}
	a.children=[b]
	
##########################################################################################

def run_hyper_alarm_model(ds):
	num_samples = 50000
	init_hyper_alarm_model(ds)
	ds.generate_samples(num_samples)
	
	e_hyper_hyper_marginal = get_marginal(ds.samples, 'Earthquake Hyper Hyper Success Probability')
	e_hyper_marginal = get_marginal(ds.samples, 'Earthquake Hyper Success Probability')
	b_hyper_marginal = get_marginal(ds.samples, 'Burglary Hyper Success Probability')
	a_hyper_00_marginal = get_marginal(ds.samples, 'Alarm B(f) E(f) Hyper Success Probability')
	a_hyper_01_marginal = get_marginal(ds.samples, 'Alarm B(f) E(t) Hyper Success Probability')
	a_hyper_10_marginal = get_marginal(ds.samples, 'Alarm B(t) E(f) Hyper Success Probability')
	a_hyper_11_marginal = get_marginal(ds.samples, 'Alarm B(t) E(t) Hyper Success Probability')
	#a_hyper_hyper_11_marginal = get_marginal(ds.samples, 'Alarm B(t) E(t) Hyper Hyper Success Probability')
	m_hyper_1_marginal = get_marginal(ds.samples, 'MaryCalls A(t) Hyper Success Probability')
	m_hyper_0_marginal = get_marginal(ds.samples, 'MaryCalls A(f) Hyper Success Probability')
	j_hyper_1_marginal = get_marginal(ds.samples, 'JohnCalls A(t) Hyper Success Probability')
	j_hyper_0_marginal = get_marginal(ds.samples, 'JohnCalls A(f) Hyper Success Probability')
	
	plot_mixing(e_hyper_hyper_marginal, 'Earthquake Hyper Hyper Success Probability')
	plot_mixing(e_hyper_marginal, 'Earthquake Hyper Success Probability')
	plot_mixing(b_hyper_marginal, 'Burglary Hyper Success Probability')
	plot_mixing(a_hyper_00_marginal, 'Alarm B(f) E(f) Hyper Success Probability')
	plot_mixing(a_hyper_01_marginal, 'Alarm B(f) E(t) Hyper Success Probability')
	plot_mixing(a_hyper_10_marginal, 'Alarm B(t) E(f) Hyper Success Probability')
	plot_mixing(a_hyper_11_marginal, 'Alarm B(t) E(t) Hyper Success Probability')
	#plot_mixing(a_hyper_hyper_11_marginal, 'Alarm B(t) E(t) Hyper Hyper Success Probability')
	plot_mixing(m_hyper_1_marginal, 'MaryCalls A(t) Hyper Success Probability')
	plot_mixing(m_hyper_0_marginal, 'MaryCalls A(f) Hyper Success Probability')
	plot_mixing(j_hyper_1_marginal, 'JohnCalls A(t) Hyper Success Probability')
	plot_mixing(j_hyper_0_marginal, 'JohnCalls A(f) Hyper Success Probability')
	
	#plot_posterior(e_hyper_marginal, ds.prior_pdfs['e_hyper'], 'Earthquake Hyper Success Probability', 0, 0.5)
	#plot_posterior(b_hyper_marginal, ds.prior_pdfs['b_hyper'], 'Burglary Hyper Success Probability', 0, 0.5)
	#plot_posterior(a_hyper_bf_ef_marginal, ds.prior_pdfs['a_hyper_00'], 'Alarm B(f) E(f) Hyper Success Probability', 0, 0.5)
	#plot_posterior(a_hyper_bf_ef_marginal, ds.prior_pdfs['a_hyper_00'], 'Alarm B(f) E(f) Hyper Success Probability', 0, 0.5)
	#plot_posterior(m_hyper_1_marginal, ds.prior_pdfs['m_hyper_1'], 'MaryCalls A(t) Hyper Success Probability', 0.3, 0.8)
	#plot_posterior(m_hyper_0_marginal, ds.prior_pdfs['m_hyper_0'], 'MaryCalls A(f) Hyper Success Probability', 0, 0.5)
	
	print 'e_hyper_hyper median: ', get_median(e_hyper_hyper_marginal)
	print 'e_hyper median: ', get_median(e_hyper_marginal)
	print 'b_hyper median: ', get_median(b_hyper_marginal)
	#print 'a_hyper_hyper_11 median: ', get_median(a_hyper_hyper_11_marginal)
	print 'a_hyper_11 median: ', get_median(a_hyper_11_marginal)
	print 'a_hyper_10 median: ', get_median(a_hyper_10_marginal)
	print 'a_hyper_01 median: ', get_median(a_hyper_01_marginal)
	print 'a_hyper_00 median: ', get_median(a_hyper_00_marginal)
	print 'm_hyper_1 median: ', get_median(m_hyper_1_marginal)
	print 'm_hyper_0 median: ', get_median(m_hyper_0_marginal)
	print 'j_hyper_1 median: ', get_median(j_hyper_1_marginal)
	print 'j_hyper_0 median: ', get_median(j_hyper_0_marginal)
	
def init_hyper_alarm_model(ds):

	e_hyper_hyper = node.BetaNode(last=0.5, obs=False, name='Earthquake Hyper Hyper Success Probability', candsd=0.1, alpha=1, beta=1)
	e_hyper = node.BetaNode(last=0.3, obs=False, name='Earthquake Hyper Success Probability', candsd=0.1, alpha=e_hyper_hyper, beta=1)
	b_hyper = node.BetaNode(last=0.2, obs=False, name='Burglary Hyper Success Probability', candsd=0.1, alpha=1, beta=1)
	#a_hyper_hyper_11 = node.BetaNode(last=1, obs=False, name='Alarm B(t) E(t) Hyper Hyper Success Probability', candsd=0.1, alpha=1, beta=1)
	a_hyper_11 = node.BetaNode(last=0.95, obs=False, name='Alarm B(t) E(t) Hyper Success Probability', candsd=0.1, alpha=1, beta=1)
	a_hyper_10 = node.BetaNode(last=0.94, obs=False, name='Alarm B(t) E(f) Hyper Success Probability', candsd=0.1, alpha=1, beta=1)
	a_hyper_01 = node.BetaNode(last=0.29, obs=False, name='Alarm B(f) E(t) Hyper Success Probability', candsd=0.1, alpha=1, beta=1)
	a_hyper_00 = node.BetaNode(last=0.2, obs=False, name='Alarm B(f) E(f) Hyper Success Probability', candsd=0.1, alpha=1, beta=1)
	m_hyper_1 = node.BetaNode(last=0.7, obs=False, name='MaryCalls A(t) Hyper Success Probability', candsd=0.15, alpha=1, beta=1)
	m_hyper_0 = node.BetaNode(last=0.3, obs=False, name='MaryCalls A(f) Hyper Success Probability', candsd=0.15, alpha=1, beta=1)
	j_hyper_1 = node.BetaNode(last=0.9, obs=False, name='JohnCalls A(t) Hyper Success Probability', candsd=0.1, alpha=1, beta=1)
	j_hyper_0 = node.BetaNode(last=0.2, obs=False, name='JohnCalls A(f) Hyper Success Probability', candsd=0.1, alpha=1, beta=1)
		
	#ds.nodes = [e_hyper, b_hyper, a_hyper_00, a_hyper_11, a_hyper_10, a_hyper_01]
	#ds.nodes = [e_hyper, b_hyper, m_hyper_1, m_hyper_0]
	ds.nodes = [e_hyper, e_hyper_hyper, b_hyper, m_hyper_1, m_hyper_0, a_hyper_00, a_hyper_11, a_hyper_10, a_hyper_01, j_hyper_1, j_hyper_0]
	
	e_hyper_hyper.children.append(e_hyper)
		
	for i, d in enumerate(data):
		p_a = None
		p_m = None
		p_j = None
		
		e = node.BinomialNode(last=d['E'], obs=True, name='E_'+str(i), candsd=None, n=1, p=e_hyper)
		b = node.BinomialNode(last=d['B'], obs=True, name='B_'+str(i), candsd=None, n=1, p=b_hyper)
		
		if d['B'] == 1 and d['E'] == 1:
			p_a = a_hyper_11
		elif d['B'] == 1 and d['E'] == 0:
			p_a = a_hyper_10
		elif d['B'] == 0 and d['E'] == 1:
			p_a = a_hyper_01
		else:
			p_a = a_hyper_00
			
		a = node.BinomialNode(last=d['A'], obs=True, name='A_'+str(i), candsd=None, n=1, p=p_a)
		
		if d['A'] == 1:
			p_m = m_hyper_1
			p_j = j_hyper_1
		else:
			p_m = m_hyper_0
			p_j = j_hyper_0
			
		m = node.BinomialNode(last=d['M'], obs=True, name='M_'+str(i), candsd=None, n=1, p=p_m)
		j = node.BinomialNode(last=d['J'], obs=True, name='J_'+str(i), candsd=None, n=1, p=p_j)
		
		e.children=[a]
		b.children=[a]
		a.children = [m, j]
		ds.nodes.extend([e, b, a, m, j])
			
		e_hyper.children.append(e)
		b_hyper.children.append(b)
		a_hyper_00.children.append(a)
		a_hyper_11.children.append(a)
		a_hyper_10.children.append(a)
		a_hyper_01.children.append(a)
		m_hyper_1.children.append(m)
		m_hyper_0.children.append(m)
		j_hyper_1.children.append(j)
		j_hyper_0.children.append(j)
			
			
	ds.prior_pdfs = {'e_hyper': e_hyper.get_prior_pdf, 'b_hyper': b_hyper.get_prior_pdf, 'a_hyper_11': a_hyper_11.get_prior_pdf,
					'a_hyper_10': a_hyper_10.get_prior_pdf, 'a_hyper_01': a_hyper_01.get_prior_pdf,
					'a_hyper_00': a_hyper_00.get_prior_pdf, 'm_hyper_1': m_hyper_1.get_prior_pdf,
					'm_hyper_0': m_hyper_0.get_prior_pdf, 'j_hyper_1': j_hyper_1.get_prior_pdf,
					'j_hyper_0': j_hyper_0.get_prior_pdf}
					
##########################################################################################

def get_median(samples):
	samples.sort()
	nsamples = len(samples)
	median = samples[nsamples//2]
	return median
	
def get_marginal(samples, node_name):
	return [s[node_name] for s in samples]
			
def plot_mixing(samples, name):
	xs, ys = zip(*enumerate(samples))
		
	plt.plot(xs, ys)
	plt.title('{} mixing'.format(name))
	plt.show()
		
def plot_posterior(samples, prior_pdf, name, xmin, xmax):
	xs = mlab.frange(xmin, xmax, (xmax-xmin) / 100.0)
	ys = [prior_pdf(x) for x in xs]
	plt.plot(xs, ys, label='Prior Dist')
		
	plt.hist(samples, bins=75, normed=True, label='Posterior Dist')
		
	plt.title('Prior and Posterior of {}'.format(name))
	plt.ylim(ymin=0)
	plt.xlim(xmin, xmax)
	plt.show()
	
def plot_posterior_hack(samples, prior_pdf, name, xmin, xmax):
	xs = mlab.frange(xmin, xmax, (xmax-xmin) / 100.0)
	ys = [prior_pdf(x) for x in xs]
	ts = stats.beta(3, 3)
		
	plt.plot(xs, ys, label='Prior Dist')
	plt.plot(xs, ts.pdf(xs))
		
	plt.hist(samples, bins=100, normed=True, label='Posterior Dist')
		
	plt.title('Prior and Posterior of {}'.format(name))
	plt.ylim(ymin=0)
	plt.xlim(xmin, xmax)
	plt.show()
	

def main():
	ds = DistributionSampler()
	
	#run_faculty(ds)
	#run_golfers(ds)
	#run_wacky(ds)
	#run_beta_binomial_test(ds)
	#run_gamma_poisson_test(ds)
	run_hyper_alarm_model(ds)
	#run_nozomu_net(ds)
	#run_judge_net(ds)
	#run_my_net(ds)

if __name__ == '__main__':
    main()