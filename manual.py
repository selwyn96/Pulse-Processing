#
#
# This Code is to manually construct the histogram from the pulse sequence.
#
#
import numpy as np
import util
from matplotlib import pyplot as plt



def main():
	num_examples = 10
	default_length = 150000
	# Importing the pulse sequence and histogram
	pulse , histos = util.get_examples(num_examples, default_length, True)

	# Extracting the pulse heights
	for i in range(num_examples):
		sim=pulse[i]
		k=0
		sim_temp= []
		for j in range(0,149980):
			if(sim[j+20]-sim[j]>=0.15):
				x=sim[j+20]-sim[j]
				sim_temp.extend(x)
		print(sim_temp)
		sim_temp = [x * 100 for x in sim_temp]
		weights = np.ones_like(sim_temp)/len(sim_temp)
		bins=np.linspace(0,100,101)
		plt.figure
		plt.hist(sim_temp, bins=bins, weights=weights, color="blue", alpha=0.5, normed=False)
		plt.plot(histos[i,:])
		plt.show() 



     
   

if __name__ == "__main__":
    main()


