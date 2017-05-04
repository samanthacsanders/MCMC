import random, node
import cPickle as pickle

class DistributionSampler(object):

	def __init__(self):
		self.burn = 1000
		self.samples = []
		self.nodes = []
		self.prior_pdfs = {}
    	
	def generate_samples(self, num_samples):
		mcmc = self.get_sample()
		# burn period
		for i in range(self.burn):
			self.get_sample() 
		
		# record samples
		self.samples = [next(mcmc) for i in range(num_samples)]
		pickle.dump(self.samples, open('golf_samples.p', 'wb'))
            
	def get_sample(self):
		while True:
			sample = {}
			# iterate over all of the nodes
			for n in self.nodes:
				if n.obs:
					sample[n.name]=n.last
				else:
					sample[n.name]=n.sample()
				#print n.name, sample[n.name]
				
			yield sample
