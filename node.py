import random, math
from scipy import special

class Node(object):
    
    def __init__(self, name, last, obs, candsd):
    	self.children = []
    	self.name = name
    	self.last = last
    	self.candsd = candsd
    	self.obs = obs
        
    def get_children_log_lh(self, children, cand):
    	log_lh_last = 0
    	log_lh_cand = 0
    
    	for child in children:
    		if isinstance(child, NormalNode):
    			if child.mean is self:
    				log_lh_last += child.get_logprob(mean=self.last, var=child.var, value=child.last)
    				log_lh_cand += child.get_logprob(mean=cand, var=child.var, value=child.last)
    			else: # child.var is self
    				log_lh_last += child.get_logprob(mean=child.mean, var=self.last, value=child.last)
    				log_lh_cand += child.get_logprob(mean=child.mean, var=cand, value=child.last)
    		elif isinstance(child, InvGammaNode) or isinstance(child, GammaNode) or isinstance(child, GammaPiNode) or isinstance(child, BetaNode):
    			if child.alpha is self:
    				log_lh_last += child.get_logprob(alpha=self.last, beta=child.beta, value=child.last)
    				log_lh_cand += child.get_logprob(alpha=cand, beta=child.beta, value=child.last)
    			else: # child.beta is self
    				log_lh_last += child.get_logprob(alpha=child.alpha, beta=self.last, value=child.last)
    				log_lh_cand += child.get_logprob(alpha=child.alpha, beta=cand, value=child.last)
    		elif isinstance(child, PoissonNode):
    			log_lh_last += child.get_logprob(rate=self.last, value=child.last)
    			log_lh_cand += child.get_logprob(rate=cand, value=child.last)
    		elif isinstance(child, BinomialNode):
    			if child.n is self:
    				log_lh_last += child.get_logprob(n=self.last, p=child.p, value=child.last)
    				log_lh_cand += child.get_logprob(n=cand, p=child.p, value=child.last)
    			else: # child.p is self
    				log_lh_last += child.get_logprob(n=child.n, p=self.last, value=child.last)
    				log_lh_cand += child.get_logprob(n=child.n, p=cand, value=child.last)
    	return log_lh_last, log_lh_cand

class NormalNode(Node):

    def __init__(self, last, obs, candsd, name, mean, var):
        Node.__init__(self, name, last, obs, candsd)
        self.mean = mean
        self.prior_mean = mean
        self.var = var
        self.prior_var = var

    def get_logprob(self, mean, var, value):
        var=var.last if isinstance(var, Node) else var
        mean=mean.last if isinstance(mean, Node) else mean
        #print 'get_logprob self.name: ', self.name
        #print 'mean: ', mean
        #print 'var: ', var
        #print 'value: ', value
        return -0.5 * (math.log(var) + (1/var * (value - mean)**2))
        
    def get_prior_pdf(self, x):
    	return ((1 / (2 * math.pi * self.prior_var) ** 0.5) * 
    		math.exp(-1 / (2 * self.prior_var) * (x - self.prior_mean) ** 2))
    		
    def sample(self):
    	cand = random.gauss(self.last, self.candsd)
    	#print 'self.name: ', self.name
    	#print 'self.last: ', self.last
    	#print 'self.candsd: ', self.candsd
    	#print 'cand: ', cand
    	
    	log_lh_last, log_lh_cand = self.get_children_log_lh(self.children, cand)
    	#print 'got children'
    	log_lh_last += self.get_logprob(mean=self.mean, var=self.var, value=self.last)
    	log_lh_cand += self.get_logprob(mean=self.mean, var=self.var, value=cand)
    	
    	u = math.log(random.random())
    	if u < log_lh_cand - log_lh_last:
    		self.last = cand
    		return cand
    	else:
    		return self.last
    		
class PoissonNode(Node):

	def __init__(self, last, obs, name, candsd, rate):
		Node.__init__(self, name, last, obs, candsd)
		self.rate = rate
		self.prior_rate = rate
		
	def get_logprob(self, rate, value):
		rate=rate.last if isinstance(rate, Node) else rate
		return value*math.log(rate)-rate-math.log(math.factorial(value))
		
	def get_prior_pdf(self, x):
		return (self.prior_rate**x * math.exp(-self.prior_rate)) / math.factorial(x)
		
	def sample(self):
		cand = math.floor(random.gauss(self.last, self.candsd))
		if cand <= 0:
			return self.last
		log_lh_last, log_lh_cand = self.get_children_log_lh(self.children, cand)
		log_lh_last += self.get_logprob(rate=self.rate, value=self.last)
		log_lh_cand += self.get_logprob(rate=self.rate, value=cand)
		u = math.log(random.random())
		if u < log_lh_cand - log_lh_last:
			self.last = cand
			return cand
		else:
			return self.last	
		
class GammaNode(Node):

	def __init__(self, last, obs, name, candsd, alpha, beta):
		Node.__init__(self, name, last, obs, candsd)
		self.alpha = alpha
		self.prior_alpha = alpha
		self.beta = beta
		self.prior_beta = beta
		
	def get_logprob(self, alpha, beta, value):
		alpha=alpha.last if isinstance(alpha, Node) else alpha
		beta=beta.last if isinstance(beta, Node) else beta
		return alpha*math.log(beta)-math.log(special.gamma(alpha))+(alpha-1)*math.log(value)-(beta*value)
		
	def get_prior_pdf(self, x):
		return self.prior_beta**self.prior_alpha / special.gamma(self.prior_alpha) * \
			x**(self.prior_alpha - 1) * math.exp(-self.prior_beta * x)
			
	def sample(self):
		cand = random.gauss(self.last, self.candsd)
		if cand <= 0:
			return self.last
		log_lh_last, log_lh_cand = self.get_children_log_lh(self.children, cand)
		log_lh_last += self.get_logprob(alpha=self.alpha, beta=self.beta, value=self.last)
		log_lh_cand += self.get_logprob(alpha=self.alpha, beta=self.beta, value=cand)
		u = math.log(random.random())
		if u < log_lh_cand - log_lh_last:
			self.last = cand
			return cand
		else:
			return self.last
			
class GammaPiNode(Node):

	def __init__(self, last, obs, name, candsd, alpha, beta):
		Node.__init__(self, name, last, obs, candsd)
		self.alpha = alpha
		self.prior_alpha = alpha
		self.beta = beta
		self.prior_beta = beta
		
	def get_logprob(self, alpha, beta, value):
		alpha=alpha.last if isinstance(alpha, Node) else alpha
		beta=beta.last if isinstance(beta, Node) else beta
		
		#print 'alpha: ', alpha
		#print 'beta: ', beta
		#print 'value: ', value
		#print 'special.gamma(alpha): ', special.gamma(alpha)
		return alpha*math.log(beta)-special.gammaln(alpha)+(alpha-1)*math.log(value)-(beta*value)
		
	def get_prior_pdf(self, x):
		return self.prior_beta**self.prior_alpha / special.gamma(self.prior_alpha) * \
			x**(self.prior_alpha - 1) * math.exp(-self.prior_beta * x)
			
	def sample(self):
		cand = random.gauss(self.last, self.candsd)
		if cand <= 0:
			return self.last
		log_lh_last, log_lh_cand = self.get_children_log_lh(self.children, cand)
		log_lh_last += self.get_logprob(alpha=self.alpha, beta=self.beta, value=self.last)
		log_lh_cand += self.get_logprob(alpha=self.alpha, beta=self.beta, value=cand)
		u = math.log(random.random())
		if u < log_lh_cand - log_lh_last:
			self.last = cand
			return cand
		else:
			return self.last
			
class InvGammaNode(Node):

    def __init__(self, last, obs, name, candsd, alpha, beta):
    	Node.__init__(self, name, last, obs, candsd)
        self.alpha = alpha
        self.prior_alpha = alpha
        self.beta = beta
        self.prior_beta = beta

    def get_logprob(self, alpha, beta, value):
        alpha=alpha.last if isinstance(alpha, Node) else alpha
        beta=beta.last if isinstance(beta, Node) else beta
        return alpha*math.log(beta)-math.log(special.gamma(alpha))+(-alpha-1)*math.log(value)-(beta/value)
        
    def get_prior_pdf(self, x):
    	return (self.prior_beta ** self.prior_alpha / math.gamma(self.prior_alpha) * \
    		x**(-self.prior_alpha - 1) * math.exp(-self.prior_beta / x))
    		
    def sample(self):
		cand = random.gauss(self.last, self.candsd)
		if cand <= 0:
			return self.last
		log_lh_last, log_lh_cand = self.get_children_log_lh(self.children, cand)
		log_lh_last += self.get_logprob(alpha=self.alpha, beta=self.beta, value=self.last)
		log_lh_cand += self.get_logprob(alpha=self.alpha, beta=self.beta, value=cand)
		u = math.log(random.random())
		if u < log_lh_cand - log_lh_last:
			self.last = cand
			return cand
		else:
			return self.last	
			
class BetaNode(Node):
	
	def __init__(self, last, obs, name, candsd, alpha, beta):
		Node.__init__(self, name, last, obs, candsd)
		self.alpha = alpha
		self.prior_alpha = alpha
		self.beta = beta
		self.prior_beta = beta
		
	def get_logprob(self, alpha, beta, value):
		alpha=alpha.last if isinstance(alpha, Node) else alpha
		beta=beta.last if isinstance(beta, Node) else beta
		return (alpha-1)*math.log(value) + (beta-1)*math.log(1-value) - special.gammaln(alpha) - \
			special.gammaln(beta) + special.gammaln(alpha+beta)
			
	def get_prior_pdf(self, x):
		return (x**(self.prior_alpha-1) * (1-x)**(self.prior_beta-1) * special.gamma(self.prior_alpha+self.prior_beta)) / \
			(special.gamma(self.prior_alpha) * special.gamma(self.prior_beta))
			
	def sample(self):
		cand = random.gauss(self.last, self.candsd)
		if cand <= 0 or cand >= 1:
			return self.last
		log_lh_last, log_lh_cand = self.get_children_log_lh(self.children, cand)
		log_lh_last += self.get_logprob(alpha=self.alpha, beta=self.beta, value=self.last)
		log_lh_cand += self.get_logprob(alpha=self.alpha, beta=self.beta, value=cand)
		u = math.log(random.random())
		#print 'u, log_lh_cand - log_lh_last: ', u, log_lh_cand - log_lh_last
		if u < log_lh_cand - log_lh_last:
			self.last = cand
			return cand
		else:
			return self.last
				
class BernoulliNode(Node):

	def __init__(self, last, obs, name, p):
		self.last = last
		self.obs = obs
		self.name = name
		self.p = p
		
	def get_probability(self):
		pass
		
	def sample(self):
		self.last = 1
		pos = self.get_logprob()
		for child in self.children:
			a *= child.get_probability()
		self.value = 0
		b = self.get_probability()
		for child in self.children:
			b *= child.get_probability()
		return a / (a + b)


# Bernoulli distribution if n=1			
class BinomialNode(Node):

	def __init__(self, last, obs, name, candsd, n, p):
		Node.__init__(self, name, last, obs, candsd)
		self.n = n
		#self.prior_n = n
		self.p = p
		#self.prior_p = p
		#self.parents = parents
		
	def get_logprob(self, n, p, value):
		#print 'self.name: ', self.name
		n=n.last if isinstance(n, Node) else n
		#if type(p) is dict:
			#for p in self.parents:
				
		if isinstance(p, Node):
			p=p.last

		return math.log(math.factorial(n)) - math.log(math.factorial(value)) - math.log(math.factorial(n-value)) + \
			value * math.log(p) + (n-value) * math.log(1.0-p)
			
	def get_prior_pdf(self, x):
		return (math.factorial(self.prior_n) / (math.factorial(x) * math.factorial(self.prior_n-x))) * \
			self.prior_p**x * (1-self.prior_p)**(self.prior_n-x)
			
	def sample(self):
		cand = round(random.uniform(0, self.n))
		if cand <= 0:
			return self.last
		log_lh_last, log_lh_cand = self.get_children_log_lh(self.children, cand)
		log_lh_last += self.get_logprob(n=self.n, p=self.p, value=self.last)
		log_lh_cand += self.get_logprob(n=self.n, p=self.p, value=cand)
		u = math.log(random.random())
		if u < log_lh_cand - log_lh_last:
			self.last = cand
			return cand
		else:
			return self.last
		