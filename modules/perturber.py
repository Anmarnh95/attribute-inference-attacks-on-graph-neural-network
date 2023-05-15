import numpy as np
import copy

class Perturber():

    def perturb_all(self, samples, m, RAA = True):
        curr_samples = copy.deepcopy(samples)
        # Sample random indexes of features to perturb them
        idx = np.random.choice(samples.size(dim=1)-1,size=m,replace= False)
        # Perturb every sample
        for i in range(curr_samples.size()[0]):
            for j in idx:
                curr_samples[i][j] = np.NaN 
            # If RAA is wanted, resample features randomly
            if RAA:
                idx = np.random.choice(samples.size(dim=1)-1,size=m,replace= False)
        return curr_samples, np.isnan(curr_samples).bool()

    def perturb(self, candidates, sensetive_attr ,m, RAA = True, perturbation_ratio = 1):

        curr_samples = copy.deepcopy(candidates)

        # Perturb sensetive attributes
        idx = np.random.choice(curr_samples.size()[0],size= int(curr_samples.size()[0] * perturbation_ratio),replace= False)
        for i in idx:
            for j in sensetive_attr:
                curr_samples[i][j] = np.NaN

        # Perturb nonsensetive attributes
        if m > 0:
            attr = np.arange(curr_samples.size(dim=1)-1)
            attr_choices = np.setdiff1d(attr,sensetive_attr)
            idx = np.random.choice(attr_choices,size=m,replace= False)
            for i in range(curr_samples.size()[0]):
                for j in idx:
                    curr_samples[i][j] = np.NaN
                # If RAA is wanted, resample features randomly
                if RAA:
                    idx = np.random.choice(attr_choices,size=m,replace= False)

        # print("PERTURBER:")
        # print("Old Candidates")
        # print(candidates)
        # print("New Candidates")
        # print(f"RATIO: {perturbation_ratio}")
        # print(curr_samples)
        # print("Mask")
        # print(np.isnan(curr_samples).bool())

        return curr_samples, np.isnan(curr_samples).bool()