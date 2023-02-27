import numpy as np
import pandas as pd
import scipy
import scipy.spatial.ckdtree as ckdtree
from simulator.models.attack_detection.robustpcc import RobustPCC


class AnomalyDetection():
    def __init__(self, method = "IF", sensitivity_level=0.05):
        '''
        :params method: Kullback-Leibler (KL) Divergence; Isolation Forest (IF); Kmeans; MixGaussian; Spectral
        '''
        # load attack detection model.
        self.method = method
        self.sensitivity_level = sensitivity_level
        self.n_dim = 0
        self.model = None


    @staticmethod
    def estimateGaussian(X):
        m = X.shape[0]
        # compute mean of X
        sum_ = np.sum(X, axis=0)
        mu = (sum_ / m)
        # compute variance of X
        var = np.var(X, axis=0)
        # print(mu, var)
        return mu, var

    @staticmethod
    def multivariateGaussian(X, mu, sigma):
        k = len(mu)
        sigma = np.diag(sigma)
        X = X - mu.T
        p = 1 / ((2 * np.pi) ** (k / 2) * (np.linalg.det(sigma) ** 0.5)) * np.exp(
            -0.5 * np.sum(X @ np.linalg.pinv(sigma) * X, axis=1))
        return p
    @staticmethod
    def get_hour_sin_cos(hour):
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        return hour_sin, hour_cos

    def train(self, historical_data):
        '''
        historical_data: event over one-week period
        save trained model to a sav. file
        '''
        historical_data = np.array(historical_data)
        if self.method == "IF":
            from sklearn.ensemble import IsolationForest
            historical_data = pd.DataFrame(historical_data)
            historical_data[4], historical_data[5] = self.get_hour_sin_cos(historical_data[3])
            historical_data.drop(3, axis=1, inplace=True)
            if_model = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(self.sensitivity_level),
                                       max_features=1.0)
            if_model.fit(historical_data.iloc[:,1:]) # it can be pd.Dataframe
            self.model = if_model
        elif self.method == "KL_div":
            # only need to save a 3d density per hist bins
            self.n_bins = [5,5,6]
            self.ranges = [(0,0.5), (25, 35), (0,24)]  # start SoC, charging duration, charging start time.
            self.n_dim = len(historical_data[0])
            X_train_density = []
            for i in range(self.n_dim-1):
                x_density, _ = np.histogram(historical_data[:,i+1], bins=self.n_bins[i],
                                                 range= self.ranges[i],
                                                 density=True)
                X_train_density.append(x_density+1e-3) # add a small number to avoid zero probability
            self.model = X_train_density
        else:
            historical_data = pd.DataFrame(historical_data)
            historical_data[4], historical_data[5] = self.get_hour_sin_cos(historical_data[3])
            historical_data.drop(3, axis=1, inplace=True)

            if self.method == "Kmeans":
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=1, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(historical_data.iloc[:,1:])
                self.model = kmeans

            elif self.method == "MixGaussian":
                mu, sigma = self.estimateGaussian(historical_data.iloc[:,1:].to_numpy())
                self.model =[mu,sigma] # get the mean and var using a mix gaussian kernel

            elif self.method == "Spectral":
                rpcc = RobustPCC(historical_data.iloc[:,1:], gamma=self.sensitivity_level)
                self.model = rpcc
            else:
                print(f"method {self.method} is not supported yet!!!!!!!!!!!!")
                return

        print(f"Finished training the {self.method} model")

    def predict(self, testing_data, evcs_status):
        '''
        testing_data: evcs_id, start_SoC, charging duration, hour of day. (only for the EVCS in S and I)
        evcs_status: an array of evcs status, 0: S, 1: I , 2: R
        1. load model
        2. predict
        3. return the outliers
        '''
        # convert testing data to np array
        testing_data = pd.DataFrame(testing_data)
        all_evcs_id = testing_data[0].to_numpy() # the first column evcs_id
        # print(f"successfully load testing data: {len(all_evcs_id)} EVCS with id: {all_evcs_id}")
        # print(f"test data: {testing_data}")

        # conduct prediction (testing)
        if self.method == "IF": # threshold is embedded in model trainning.
            testing_data[4], testing_data[5] = self.get_hour_sin_cos(testing_data[3])
            testing_data.drop(3, axis=1, inplace=True)
            outliers = self.model.predict(testing_data.loc[:,1:])
        elif self.method == "KL_div": # introduce the predefined threshold.
            outliers = self.get_outliers_KLDiv(testing_data)
        elif self.method == "Kmeans":
            outliers = self.get_outliers_Kmeans(testing_data)
        elif self.method == "MixGaussian":
            outliers = self.get_outliers_MixGaussian(testing_data)
        elif self.method == "Spectral":
            outliers = self.get_outliers_Spectral(testing_data)
        else:
            print(f"method {self.method} is not supported yet!!!!!!!!!!!!")
            return

        evcs_id_outlier, evcs_id_inlier, Accuracy = self.get_performance(outliers, all_evcs_id ,evcs_status)
        return evcs_id_outlier, evcs_id_inlier, Accuracy # , Precision, Recall, F1]


    def get_outliers_MixGaussian(self,testing_data,epsilon = 1e-2):
        '''
        lower epsilon means less outliers
        epsilon = 0.01 means 1% significant
        '''
        mu, sigma = self.model
        outliers = []
        testing_data[4],testing_data[5] = self.get_hour_sin_cos(testing_data[3])
        testing_data.drop(3, axis=1, inplace=True) # drop the original time format.

        for id_x, testing_data_x in testing_data.groupby(0):  # index 0 is the evcs_id
            p = self.multivariateGaussian(testing_data_x.iloc[:,1:].to_numpy(), mu, sigma)  # p value
            outlier_percentage = len(np.nonzero(p<epsilon)[0])/len(p)  # 1-np.sum(p<epsilon)/len(p)
            if outlier_percentage>=self.sensitivity_level:
                print(f"outlier_percentage: {outlier_percentage}, sensitivity_level: {self.sensitivity_level}, p value: {p}")
                outliers.append(id_x)
        return outliers

    def get_outliers_Spectral(self,testing_data):
        """
        :param testing_data: evcs_id, start_SoC, charging duration, hour of day. (only for the EVCS in S and I)
        :return: outliers (an array of length all_evcs)
        """
        rpcc = self.model

        testing_data[4],testing_data[5] = self.get_hour_sin_cos(testing_data[3])
        testing_data.drop(3, axis=1, inplace=True)
        rpcc_result = rpcc.predict(testing_data.iloc[:,1:])
        # print(f"check robust PCC results:{rpcc_result}!!!!")
        return rpcc_result

    def get_outliers_Kmeans(self, testing_data):
        testing_data[4],testing_data[5] = self.get_hour_sin_cos(testing_data[3])
        testing_data.drop(3, axis=1, inplace=True)
        outliers = []
        print(f"check testing data: {testing_data.iloc[0]}")
        for id_x, testing_data_x in testing_data.groupby(0): # index 0 is the evcs_id
            outlier_percentage = self.get_outlier_percentage_to_cluster(testing_data_x.iloc[:,1:])
            # print(f"EVCS: {id_x}, Anomaly score: {outlier_percentage}")
            if outlier_percentage>=self.sensitivity_level:
                outliers.append(id_x)
        return outliers

    def get_outlier_percentage_to_cluster(self, X_test, outlier_threshold=None):
        # if outlier_threshold is None:
        #     outlier_threshold = [0.10, 2.5, 0.25, 0.25]
        # x,y,z_sin,z_cos = outlier_threshold
        tree = ckdtree.cKDTree(self.model.cluster_centers_)
        dist_, _ = tree.query(X_test.to_numpy())
        outlier_percentage = len(np.where(dist_ >= 2.5)[0]) / len(dist_)
        return outlier_percentage

    def get_outliers_KLDiv(self,testing_data):
        outliers = []
        # print(f"testing data:\n {testing_data}")
        for id_x, testing_data_x in testing_data.groupby(0): # index 0 is the evcs_id
            kl_div_tot = 0
            n_bins = self.n_bins # we shall use a kernel.
            ranges = self.ranges  # start SoC, charging duration, charging start time.
            n_dim = self.n_dim
            for i in range(n_dim-1):
                x_density,_ = np.histogram(testing_data_x.loc[:,i+1], bins=n_bins[i],range= ranges[i],density=True)

                vec = scipy.special.kl_div(self.model[i],x_density+1e-3) # add a small number to avoid zero probability
                kl_div = np.nansum(vec)
                kl_div_tot += kl_div
            print(f"EVCS: {id_x}, Anomaly score: {kl_div_tot}, sensitivity_level: {self.sensitivity_level}")
            if kl_div_tot>= self.sensitivity_level:
                outliers.append(id_x) # this evcs is detected to be an anomaly
        return outliers

    def get_model_path(self):
        return f"../../data/trained_model_{self.method}_sen_{self.sensitivity_level}.sav"



    def get_performance(self, outliers, all_evcs_id, evcs_status):
        """
        evcs_id_outlier: a list of detected evcs id
        evcs_status: a list of evcs status, 0: S, 1: I, 2: R, and the length is the number of EVCS
        """
        # ground truth information
        evcs_id_S = set(np.where(evcs_status==0)[0]) # IDs for EVCS in S
        evcs_id_I = set(np.where(evcs_status==1)[0]) # IDs for EVCS in I
        # results from anomaly detection.
        if self.method == "IF": # the output format is a numpy array
            evcs_id_outlier = set(all_evcs_id[outliers==-1])
        elif self.method == "Spectral": # the output format is a list
            evcs_id_outlier = set(all_evcs_id[outliers==1])
        else:
            evcs_id_outlier = set(outliers)

        evcs_id_inlier = set(all_evcs_id) - evcs_id_outlier
        print(f"the number of inliers: {len(evcs_id_inlier)}, and outliers: {len(evcs_id_outlier)}, total: {len(all_evcs_id)}")

        # print(f"inlier: {evcs_id_inlier}, outlier: {evcs_id_outlier}, S: {evcs_id_S}, I: {evcs_id_I}")
        TP = len(evcs_id_outlier&evcs_id_I) # detected as anomaly and it is.
        FP = len(evcs_id_outlier&evcs_id_S) # detected as anomaly but it is not.
        TN = len(evcs_id_inlier&evcs_id_S) # detected as normal and it is.
        FN = len(evcs_id_inlier&evcs_id_I) # detected as normal but it is not.

        # print(f"TP:{evcs_id_outlier&evcs_id_I}, FP:{evcs_id_outlier&evcs_id_S}, TN:{evcs_id_inlier&evcs_id_S}, FN:{evcs_id_inlier&evcs_id_I}")
        Accuracy = (TP+TN)/(TP+TN+FP+FN)
        # Precision = TP/(TP+FP)
        # Recall = TP/(TP+FN)
        # F1 = 2*Precision*Recall/(Precision+Recall)

        return evcs_id_outlier, evcs_id_inlier, Accuracy # , Precision, Recall, F1
