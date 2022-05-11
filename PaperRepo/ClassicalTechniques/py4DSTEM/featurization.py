import numpy as np
import py4DSTEM
from py4DSTEM.process.utils import tqdmnd
from py4DSTEM.io.datastructure import DataCube
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.decomposition import NMF, PCA, FastICA
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings("ignore")

class clustering(object):
    """
    A class for feature selection, modification, and unsupervised of 4D-STEM data based on a user defined
    set of input features for each pattern.
    
    Initialization methods:
        __init__:
            Creates dictionary to store features in "self.features".
    
    Feature Dictionary Modification Methods
        add_feature:
            Adds a matrix to be stored at self.features[key]
        remove_feature:
            Removes a key-value pair at the self.features[key]
        update_features:
            Updates an NMF, PCA, or ICA reduced feature set to the self.features location.
        concatenate_features:
            Concatenates all features within a list of keys into one matrix stored at self.features[output_key]        
    
    Feature Representation Methods
        MinMaxScaler:
            Performs sklearn MinMaxScaler operation on features stored at a key
        mean_feature:
            Takes the rowwise average of a matrix stored at a key, such that only one column is left,
            reducing a set of n features down to 1 feature per pattern.
        median_feature:
            Takes the rowwise median of a matrix stored at a key, such that only one column is left,
            reducing a set of n features down to 1 feature per pattern.
        max_feature:
            Takes the rowwise max of a matrix stored at a key, such that only one column is left,
            reducing a set of n features down to 1 feature per pattern.
        
    Classification Methods
        nmf:
            Nonnegative matrix Factorization to refine features. Performed iteratively by merging
            [add more details later]
        gmm:
            Gaussian mixture model to predict class labels. Fits a gaussian based on covariance of features
            [add more details later]
    
    Class Examination Methods
        get_class_DPs:
            Gets weighted class diffraction patterns (DPs) for an NMF or GMM operation
        get_class_ims:
            Gets weighted class images (ims) for an NMF or FMM operation
    """

    def __init__(self, keys, features):
        """
        Initializes classification instance.
        
        This method:
        1. Generates key:value pair to access input features
        2. Initializes the empty dictionaries for feature modification and classification
        
        Args:
        keys (list): A list of keys in which the features will be stored at in the self.features 
        dictionary. A key can be an int, float, or str but it is recommended to use strings with 
        features "names", such as 'Bragg Disks' for the Bragg disk array.
        features (list): A list of ndarrays which will each be associated with value stored at the key in the same index within the list
        """
        if not isinstance(features, list):
            raise TypeError('features must be a list')
        if not isinstance(keys, list):
            raise TypeError('keys must be a list')
        for i in range(len(features)):
            if not isinstance(features[i], np.ndarray):
                string = 'feature {} must be of type np.ndarray'
                raise TypeError(string.format(i))
            if not isinstance(keys[i], str):
                string = 'key {} must be of type string'
                raise TypeError(string.format(i))
        self.features = dict(zip(keys,features))
        self.pca = {}
        self.ica = {}
        self.nmf = {}
        self.nmf_comps = {}
        self.Ws = {}
        self.Hs = {}
        self.W = {}
        self.H = {}
        self.nmf_labels = {}
        self.gmm = {}
        self.gmm_labels = {}
        self.gmm_proba = {}
        self.class_DPs = {}
        self.class_ims = {}
        return

    def add_feature(self, key, feature):
        """
        Add a feature to the features dictionary
        
        Args:
        key (int, float, str): A key in which a feature can be accessed from
        feature (ndarray): The feature associated with the key
        """
        self.features[key] = feature
        return
    
    def remove_feature(self, key):
        """
        Removes a feature to the feature dictionary
        
        Args:
        Key (int, float, str): A key which will be removed
        """
        remove_key = self.features.pop(key, None)
        if remove_key is not None:
            print("The feature stored at", key, "has been removed.")
        else:
            print(key, "is not a key in the classfication.features dictionary")
        return
            
    def concatenate_features(self, keys, output_key):
        """
        Concatenates dataframes in 'key' and saves them to features with key 'output_key'
        
        Args
        keys (list) A list of keys to be concatenated into one array
        output_key (int, float, str) the key in which the concatenated array will be stored
        """
        self.features[output_key] = np.concatenate([self.features[keys[i]] for i in range(len(keys))], axis = 1)
        return
    
    def update_features(self, keys, classification_method):
        """
        Updates the features dictionary with dimensionality reduced components for use downstream.
        New keys will be called "key_location"
        
        Args:
        Key
        location
        """
        for i in range(len(keys)):
            if classification_method == 'nmf':
                self.features[keys[i] + '_nmf'] = self.W[keys[i]]
            elif classification_method == 'pca':
                self.features[keys[i] + '_pca'] = self.pca[keys[i]]
            elif classification_method == 'ica':
                self.features[keys[i] + '_ica'] = self.ica[keys[i]]
        return
    
    def mean_feature(self, keys, replace = False):
        """
        Takes the columnwise mean of the ndarray located 
        """
        if replace == True:
            for i in range(len(keys)):
                self.features[keys[i]] = np.mean(self.features[keys[i]], axis = 1).reshape(self.features[keys[i]].shape[0],1)
        elif replace == False:
            for i in range(len(keys)):
                self.features[keys[i] + '_mean'] = np.mean(self.features[keys[i]], axis = 1).reshape(self.features[keys[i]].shape[0],1)
        return

    def median_feature(self, keys, replace = False):
        """
        Takes median of feature in key. if replace = True, replaces value in key with
        median value over entire feature
        
        Args:
        key (str)
        """
        if replace == True:
            for i in range(len(keys)):
                self.features[keys[i]] = np.median(self.features[keys[i]], axis = 1).reshape(self.features[keys[i]].shape[0],1)
        elif replace == False:
            for i in range(len(keys)):
                self.features[keys[i] + '_median'] = np.median(self.features[keys[i]], axis = 1).reshape(self.features[keys[i]].shape[0],1)
        return
    
    def max_feature(self, keys, replace = False):
        """
        Takes the columnwise max of the ndarray located 
        """
        if replace == True:
            for i in range(len(keys)):
                self.features[keys[i]] = np.max(self.features[keys[i]], axis = 1).reshape(self.features[keys[i]].shape[0],1)
        elif replace == False:
            for i in range(len(keys)):
                self.features[keys[i] + '_max'] = np.max(self.features[keys[i]], axis = 1).reshape(self.features[keys[i]].shape[0],1)
        return

    def MinMaxScaler(self, keys):
        """
        Uses sklearn MinMaxScaler to scale a subset of the input features. Output will be a new feature
        in feature dictionary with key "key_mms"
        Accepts
        keys (list) list of feature names (same as in dict) to perform scaling on
        Returns
        """
        minmax = {}
        mms = MinMaxScaler()
        for i in range(len(keys)):
            self.features[keys[i] + '_mms'] = mms.fit_transform(self.features[keys[i]])
        return
    
    def RobustScaler(self, keys):
        """
        Uses sklearn RobustScaler to scale a subset of the input features. Output will be a new feature
        in feature dictionary with key "key_mms"
        Accepts
        keys (list) list of feature names (same as in dict) to perform scaling on
        Returns
        """
        for i in range(len(keys)):
            rs = RobustScaler()
            self.features[keys[i] + '_rs'] = rs.fit_transform(self.features[keys[i]])
        return
    
    def shift_positive(self, keys):
        """
        Replaces a feature stored at feature.key with the positive shifted array.
        """
        for i in range(len(keys)):
            self.features[keys[i]] += np.abs(self.features[keys[i]].min())
        return
    
    def PCA(self, keys, components):
        """
        Performs PCA on a set of keys
        
        Args:
        keys (int, float, str): keys to performa PCA on
        components (list): A list of ints for each key. This will be the output number of features
        """
        for i in range(len(keys)):
            pca = PCA(n_components = components[keys[i]])
            self.pca[keys[i]] = pca.fit_transform(self.features[keys[i]])
        return

    def ICA(self, keys, components):
        """
        Performs ICA on a set of keys
        
        Args:
        keys (int, float, str): keys to performa ICA on
        components (list): A list of ints for each key. This will be the output number of features
        """
        for i in range(len(keys)):
            ica = FastICA(n_components = components[keys[i]])
            self.ica[keys[i]] = ica.fit_transform(self.features[keys[i]])
        return
    
    #@ignore_warnings(category=ConvergenceWarning)
    def NMF(self, keys, n_components, iters, random_state = None):
        """
        Placeholder for docstrings
        """   
        if random_state == None:
            rng = np.random.RandomState(seed = 42)
        else:
            seed = random_state
        for i in range(len(keys)):
            recon_error = np.inf
            for n in range(len(iters)):
                if random_state == None:
                    seed = rng.randint(5000)
                nmf_temp = NMF(n_components = n_components[keys[i]], random_state = seed)
                nmf_ = nmf_temp.fit_transform(self.features[keys[i]])
                if nmf_temp.reconstruction_err_ < recon_error:
                    self.nmf[keys[i]] = nmf_
                    self.nmf_comps[keys[i]] = nmf_temp.components_
                    recon_error = nmf_temp.reconstruction_err_
        return
    
    def NMF_iterative(self, keys, max_components, merge_thresh, iters, random_state = None):
        """
        Performs nmf iteratively on input features
        
        Args:
        keys
        max_components
        merge_thresh
        iters
        random_state
        """
        Ws, Hs, W, H = {}, {}, {}, {}
        for i in range(len(keys)):
            self.Ws[keys[i]], self.Hs[keys[i]], self.W[keys[i]], self.H[keys[i]] = _nmf_single(
                self.features[keys[i]],
                max_components=max_components[keys[i]],
                merge_thresh = merge_thresh[keys[i]],
                iters = iters[keys[i]], random_state = random_state)
        return
        
    def GMM(self, keys, cv, components, iters, random_state = None):
        """
        Performs gaussian mixture model on input features
        Args:
        keys
        cv
        components
        iters
        random_state
        """
        gmm, gmm_labels, gmm_proba = {}, {}, {}
        for i in range(len(keys)):
            self.gmm[keys[i]], self.gmm_labels[keys[i]], self.gmm_proba[keys[i]] = _gmm_single(
                self.features[keys[i]],
                cv = cv[keys[i]],
                components = components[keys[i]],
                iters = iters[keys[i]],
                random_state = random_state)
        return

    def get_nmf_labels(self, keys):
        """[summary]

        Args:
            keys ([type]): [description]
        """
        for i in range(len(keys)):
            self.nmf_labels[keys[i]] = (self.W[keys[i]].max(axis = 1, 
                keepdims=1) == self.W[keys[i]])
        return
        
    def get_class_DPs(self, keys, classification_method, thresh, dc):
        """
        Returns weighted class patterns based on classification instance
        dc (datacube) must be vectorized in real space (shape = (R_Nx * R_Ny, 1, Q_Nx, Q_Ny)
        Accepts:
        keys
        classification_method
        dc (datacube) Vectorized in real space, with shape (R_Nx * R_Ny, 1, Q_Nx, Q_Ny)
        Returns:
        Class patterns
        """
        for i in range(len(keys)):
            class_patterns = []
            if classification_method[keys[i]] == 'nmf':
                self.get_nmf_labels([keys[i]])
                for l in range(self.W[keys[i]].shape[1]):
                    class_pattern = np.zeros((dc.data.shape[2], dc.data.shape[3]))
                    x_ = np.where(self.W[keys[i]][:,l] > thresh[keys[i]])[0]
                    for x in range(x_.shape[0]):
                        class_pattern += dc.data[x_[x],0] * self.W[keys[i]][x_[x],l]
                    class_patterns.append((class_pattern - np.min(class_pattern)) 
                                          / (np.max(class_pattern) - np.min(class_pattern)))
            elif classification_method[keys[i]] == 'gmm':
                for l in range(np.max(self.gmm_labels[keys[i]])+1):
                    class_pattern = np.zeros((dc.data.shape[2], dc.data.shape[3]))
                    x_ = np.where(self.gmm_proba[keys[i]][:,l] > thresh[keys[i]])[0]
                    for x in range(x_.shape[0]):
                        class_pattern += dc.data[x_[x],0] * self.gmm_proba[keys[i]][x_[x],l]
                    class_patterns.append((class_pattern - np.min(class_pattern)) 
                                          / (np.max(class_pattern) - np.min(class_pattern)))
            self.class_DPs[keys[i]] = class_patterns
        return
        
    def get_class_ims(self, keys, classification_method, dc):
        """
        Returns weighted class maps based on classification instance in either the NMF or GMM
        class.
        
        Args:
        keys
        classification_method
        dc
        
        Returns:
        """
        
        for i in range(len(keys)):
            class_maps = []
            if classification_method[keys[i]] == 'nmf':
                self.get_nmf_labels(keys[i])
                self.class_ims[keys[i]] = [(self.nmf_labels[keys[i]][:,i] * self.W[keys[i]]).reshape(dc.R_Nx, dc.R_Ny) 
                                           for i in range(self.nmf_labels[keys[i]].shape[1])]
            elif classification_method[keys[i]] == 'gmm':
                for l in range((np.max(self.gmm_labels[keys[i]])+1)):
                    R_vals = np.where(self.gmm_labels[keys[i]].reshape(dc.R_Nx,dc.R_Ny) == l, 1, 0)
                    class_maps.append(R_vals * self.gmm_proba[keys[i]][:,l].reshape(dc.R_Nx, dc.R_Ny))
                self.class_ims[keys[i]] = class_maps
        return

#@ignore_warnings(category=ConvergenceWarning)
#with warnings.catch_warnings():
def _nmf_single(x, max_components, merge_thresh, iters, random_state=None):
    """
    Args
    x
    max_components
    merge_thresh
    iters
    
    Returns
    Ws
    Hs
    W
    H
    """
    err = np.inf    
    if random_state == None:
        rng = np.random.RandomState(seed = 42)
    else:
        seed = random_state
    for i in range(iters):
        if random_state == None:
            seed = rng.randint(5000)
        n_comps = max_components
        recon_error, counter = 0, 0
        Hs, Ws = [], []
        for z in range(max_components):
            nmf = NMF(n_components = n_comps, random_state = seed)
            if counter == 0:
                nmf_temp = nmf.fit_transform(x)
            else:
                nmf_temp = nmf.fit_transform(nmf_temp)
            Ws.append(nmf_temp)
            Hs.append(np.transpose(nmf.components_))
            recon_error += nmf.reconstruction_err_
            counter += 1
            tril = np.tril(np.corrcoef(nmf_temp, rowvar = False), k = -1)
            if np.nanmax(tril) >= merge_thresh:
                inds = np.argwhere(tril >= merge_thresh)
                for n in range(inds.shape[0]):
                    nmf_temp[:, inds[n,0]] += nmf_temp[:,inds[n,1]]
                ys_sorted = np.sort(np.unique(inds[n,1]))[::-1]
                for n in range(ys_sorted.shape[0]):
                    np.delete(nmf_temp, ys_sorted[n], axis=1)
            else:
                break
            n_comps = nmf_temp.shape[1] - 1
            if n_comps <=1:
                break
        if (recon_error / counter) < err:
            err = (recon_error / counter)
            W_comps = Ws
            H_comps = Hs
            W = nmf_temp
            if len(Hs) >= 2:
                H = np.transpose(np.linalg.multi_dot(H_comps))
            else:
                H = Hs
    return W_comps, H_comps, W, H

#@ignore_warnings(category=ConvergenceWarning)
def _gmm_single(x, cv, components, iters, random_state=None):
    """
    Runs GMM several times and saves value with best BIC score
    Accepts:
    key list of strings
    cv list of strings
    components list of ints
    iters int
    Returns:
    """
    lowest_bic = np.infty
    bic = []
    bic_temp = 0
    if random_state == None:
        rng = np.random.RandomState(seed = 42)
    else:
        seed = random_state
    for n in range(iters):
        if random_state == None:
            seed = rng.randint(5000)
        for j in range(len(components)):
            for cv_type in cv:
                gmm = GaussianMixture(n_components=components[j],
                                      covariance_type=cv_type, random_state = seed)
                labels = gmm.fit_predict(x)
                bic_temp = gmm.bic(x)    
        if bic_temp < lowest_bic:
            lowest_bic = bic_temp
            best_gmm = gmm
            best_gmm_labels = labels
            best_gmm_proba = gmm.predict_proba(x)
    return best_gmm, best_gmm_labels, best_gmm_proba

def rasterize_peaks(
    pointlistarray,
    bins_x,
    bins_y,
    Q_Nx,
    Q_Ny,
    coords=None,
    mask=None):
    """
    Prepare an array of peak positions in evenly spaced binds for cluster classification in the 
    Args:
        pointlistarray (PointListArray)
        coords (coordinates)
        bins_x (int)
        bins_y (int)
        mask (bool)
    Returns:
        peak_data (ndarray)
    """
    if coords is not None:
        nx_bins = int(coords.Q_Nx / bins_x)
        ny_bins = int(coords.Q_Ny / bins_y)
        n_bins = nx_bins * ny_bins
        qx, qy = coords.get_origin()
    else:
        nx_bins = int(Q_Nx / bins_x)
        ny_bins = int(Q_Ny / bins_y)
        n_bins = nx_bins * ny_bins
    
    ##Intialize empty array
    peak_data = np.zeros((pointlistarray.shape[0], pointlistarray.shape[1], n_bins))
    
    for (Rx, Ry) in tqdmnd(pointlistarray.shape[0],pointlistarray.shape[1]):
        pointlist = pointlistarray.get_pointlist(Rx,Ry)

        if pointlist.data.shape[0] == 0:
            continue
        else:
            if coords is not None:
                if mask is not None:
                    deletemask = np.zeros(pointlist.length, dtype=bool)
                    for i in range(pointlist.length):
                        deletemask_ceil = np.where((mask[ np.ceil(pointlist.data['qx'] + qx[Rx, Ry]).astype(int),
                            np.ceil(pointlist.data['qy'] + qy[Rx,Ry]).astype(int) ] == False), True, False) 
                        pointlist.remove_points(deletemask_ceil)
                        deletemask_floor = np.where((mask[ np.floor(pointlist.data['qx'] + qx[Rx, Ry]).astype(int),
                            np.floor(pointlist.data['qy'] + qy[Rx,Ry]).astype(int) ] == False), True, False)
                        pointlist.remove_points(deletemask_floor)
                for i in range(pointlist.data.shape[0]):
                    if pointlist.data[i][0] - np.floor(pointlist.data[i][0]) < 0.5:
                        floor_x, ceil_x = np.floor((pointlist.data[i][0] + qx[Rx, Ry] - 1)/bins_x), np.ceil((pointlist.data[i][0] + qx[Rx, Ry] - 1)/bins_x)
                    else:
                        floor_x, ceil_x = np.floor((pointlist.data[i][0] + qx[Rx, Ry])/bins_x), np.ceil((pointlist.data[i][0] + qx[Rx, Ry])/bins_x)
                    if pointlist.data[i][1] - np.floor(pointlist.data[i][1]) < 0.5:
                        floor_y, ceil_y = np.floor((pointlist.data[i][1] + qy[Rx, Ry] - 1)/bins_y), np.ceil((pointlist.data[i][1] + qy[Rx, Ry] - 1)/bins_y)
                    else:
                        floor_y, ceil_y = np.floor((pointlist.data[i][1] + qy[Rx, Ry])//bins_y), np.ceil((pointlist.data[i][1] + qx[Rx, Ry])/bins_y)  
                    if (pointlist.data[i][0] + qx[Rx, Ry]) < 1:
                        floor_x, ceil_x = np.floor((pointlist.data[i][0] + qx[Rx, Ry])/bins_x), np.ceil((pointlist.data[i][0] + qx[Rx, Ry])/bins_x)   
                    if (pointlist.data[i][1] + qy[Rx, Ry]) < 1:
                        floor_y, ceil_y = np.floor((pointlist.data[i][1] + qy[Rx, Ry])/bins_y), np.ceil((pointlist.data[i][1] + qy[Rx, Ry])/bins_x)
                    binval_ff = int((floor_x * ny_bins) + floor_y)
                    binval_cf = int(((floor_x + 1) * ny_bins) + floor_y)
                    peak_data[Rx,Ry,binval_ff] += pointlist.data[i][2]
                    peak_data[Rx,Ry,binval_ff + 1] += pointlist.data[i][2]
                    peak_data[Rx,Ry,binval_cf] += pointlist.data[i][2]
                    peak_data[Rx,Ry,binval_cf + 1] += pointlist.data[i][2]   
            else:
                if mask is not None:
                    deletemask = np.zeros(pointlist.length, dtype=bool)
                    for i in range(pointlist.length):
                        deletemask_ceil = np.where((mask[ np.ceil(pointlist.data['qx']).astype(int),
                            np.ceil(pointlist.data['qy']).astype(int) ] == False), True, False) 
                        pointlist.remove_points(deletemask_ceil)
                        deletemask_floor = np.where((mask[ np.floor(pointlist.data['qx']).astype(int),
                            np.floor(pointlist.data['qy']).astype(int) ] == False), True, False)
                        pointlist.remove_points(deletemask_floor)
                for i in range(pointlist.data.shape[0]):
                    if pointlist.data[i][0] - np.floor(pointlist.data[i][0]) < 0.5:
                        floor_x, ceil_x = np.floor(pointlist.data[i][0]- 1), np.ceil(pointlist.data[i][0] - 1)
                    else:
                        floor_x, ceil_x = np.floor(pointlist.data[i][0]), np.ceil(pointlist.data[i][0])
                    if pointlist.data[i][1] - np.floor(pointlist.data[i][1]) < 0.5:
                        floor_y, ceil_y = np.floor(pointlist.data[i][1] - 1), np.ceil(pointlist.data[i][1] - 1)
                    else:
                        floor_y, ceil_y = np.floor(pointlist.data[i][1]), np.ceil(pointlist.data[i][1])  
                    if (pointlist.data[i][0]) < 1:
                        floor_x, ceil_x = np.floor(pointlist.data[i][0]) , np.ceil(pointlist.data[i][0])    
                    if (pointlist.data[i][1]) < 1:
                        floor_y, ceil_y = np.floor(pointlist.data[i][1]), np.ceil(pointlist.data[i][1])
                    peak_data[Rx,Ry,int((floor_x * ny_bins) + floor_y)] += pointlist.data[i][2]
                    peak_data[Rx,Ry,int((floor_x * ny_bins) + floor_y) + 1] += pointlist.data[i][2]
                    peak_data[Rx,Ry,int((ceil_x * ny_bins) + floor_y)] += pointlist.data[i][2]
                    peak_data[Rx,Ry,int((ceil_x * ny_bins) + floor_y) + 1] += pointlist.data[i][2]                        
    return peak_data