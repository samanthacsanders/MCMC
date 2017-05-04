import cPickle as pickle

def read_golferdata():

	data = []
	golfers = set()
	tourns = set()

	for line in open('golfdataR.dat', 'rb'):
		#print line
		d = line.split()
		#print d[1]
		golfer = d[0]
		tourn = d[2]
		score = float(d[1])
		
		golfers.add(golfer)
		tourns.add(tourn)
		data.append((golfer, score, tourn))

	pickle.dump(data, open('golfdata.p', 'wb'))	
	pickle.dump(golfers, open('golfers.p', 'wb'))
	pickle.dump(tourns, open('tourns.p', 'wb'))
		
read_golferdata()