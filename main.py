import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
import math

# Helper functions and classes

# knn_lis: data structure which maintains a list of the number of highest data values where
# the number is specified using k_num(ordered datastructure)

class Node:
    def __init__(self, user, data):
        self.data = data
        self.user = user
        self.next = None

    def __repr__(self):
        return self.data

    def get_user(self):
        return self.user

    def get(self):
        return self.data


class knn_lis:
    def __init__(self, k_num):
        self.head = None
        self.tail = None
        self.size = 0
        self.k_num = k_num

    def append(self, user, data):
        if self.size == 0:
            self.head = Node(user, data)
            self.tail = self.head
            self.size += 1
        # keep adding data until the size is full
        elif self.size < self.k_num:
            moving_node = self.head.next
            prev_node = self.head
            # when data value is minimum
            if prev_node.get() > data:
                new_node = Node(user, data)
                new_node.next = prev_node
                self.head = new_node
                self.size += 1
                return
            while moving_node != None:
                if moving_node.get() > data:
                    break
                moving_node = moving_node.next
                prev_node = prev_node.next
            new_node = Node(user, data)
            prev_node.next = new_node
            new_node.next = moving_node
            if moving_node == None:
                self.tail = new_node
            self.size += 1
        # do not add any data when the size is max and data is lower than any of the current data
        elif self.head.get() > data:
            return
        # remove the first element and add new element as the tail
        elif self.tail.get() < data:
            self.head = self.head.next
            new_node = Node(user, data)
            self.tail.next = new_node
            self.tail = new_node
        # remove the first element and add new element to the appropriate position
        else:
            moving_node = self.head.next.next
            prev_node = self.head.next
            if prev_node.get() > data:
                new_node = Node(user, data)
                new_node.next = prev_node
                self.head = new_node
                return
            else:
                # assigning new head
                self.head = prev_node
            while moving_node != None:
                if moving_node.get() > data:
                    break
                moving_node = moving_node.next
                prev_node = prev_node.next
            new_node = Node(user, data)
            prev_node.next = new_node
            new_node.next = moving_node
    # conversion to python list
    def toArray(self):
        arr = []
        moving_node = self.head
        while moving_node != None:
            arr.append(moving_node.get_user())
            moving_node = moving_node.next
        return arr


def swap(num1, num2, unordered_list):
    temp = unordered_list[num1]
    unordered_list[num1] = unordered_list[num2]
    unordered_list[num2] = temp

# used as a helper function for quicksort(planned to use to create distribution matrix, but because of time constraint, not going to be used)
def partition(left, right, unordered_list):
    mid_num = unordered_list[right]
    part_num = left
    for i in range(left, right):
        if(mid_num > unordered_list[i]):
            swap(part_num, i, unordered_list)
            part_num += 1
    swap(part_num, right, unordered_list)
    return part_num


def quick_sort(left, right, unordered_list):
    if left < right:
        part_num = partition(left, right, unordered_list)
        quick_sort(left, part_num-1, unordered_list)
        quick_sort(part_num+1, right, unordered_list)


def cosine_similarity(x, y):
    mag_x = np.dot(x, x)**(1/2)
    mag_y = np.dot(y, y)**(1/2)
    if mag_x == 0 or mag_y == 0:
        return 0
    return np.dot(x, y)/(mag_x*mag_y)


def score(test_data, prediction_function, measure_num):
    # mean squared error
    if measure_num == 0:
        mse = 0
        for i in range(0, int(len(test_data[0]))):
            mse += (test_data[2][i] -
                    prediction_function(int(test_data[0][i]), int(test_data[1][i])))**2
        mse /= (len(test_data))
        return mse
    # mean absolute error
    elif measure_num == 1:
        mae = 0
        for i in range(0, int(len(test_data))):
            mae += abs(test_data[2][i] -
                       prediction_function(int(test_data[0][i]), int(test_data[1][i])))
        mae /= (len(test_data))
        return mae

# K-fold cross validation
def cross_validation(k, data, fit, prediction_function, measure_num):
    setRatio = 1.0/k
    data_size = len(data)
    rng = np.random.default_rng()
    numarr = rng.choice(np.arange(0, data_size), size=data_size, replace=False)
    res_list = np.array([])
    for r in range(0, k):
        data_list = np.zeros((3, int(data_size*setRatio*(k-1))))
        test_list = np.zeros((3, int(data_size*setRatio)))
        data_idx = 0
        test_idx = 0
        for s in range(0, k):
            start = data_size*(setRatio*s)
            end = data_size*(setRatio*(s+1))
            if r == s:
                for i in range(round(start), round(end)):
                    test_list[0][test_idx] = data['userid'][numarr[i]]
                    test_list[1][test_idx] = data['movieid'][numarr[i]]
                    test_list[2][test_idx] = data['ratings'][numarr[i]]
                    test_idx += 1
            else:
                for i in range(round(start), round(end)):
                    data_list[0][data_idx] = data['userid'][numarr[i]]
                    data_list[1][data_idx] = data['movieid'][numarr[i]]
                    data_list[2][data_idx] = data['ratings'][numarr[i]]
                    data_idx += 1
        fit(data_list)
        res_list = np.append(res_list, score(
            test_list, prediction_function, measure_num))
    return res_list

#Collaborative Filtering model class
class cf_alg:
    def __init__(self, k_num, influence, threshold, sim_threshold, data):
        self.k_num = k_num
        self.influence = influence
        self.sim_threshold = sim_threshold
        self.threshold = threshold
        self.max_user_id = max(data['userid'])
        self.max_movie_id = max(data['movieid'])

    # Prediction made using user-user-similarity matrix and user-item rating matrix
    def predict_sim(self, user_id, movie_id):
        numerator = 0
        denominator = 0
        for i in self.kn_list[user_id]:
            if self.user_movie_matrix[int(i)][movie_id] == 0:
                continue
            numerator += self.user_movie_matrix[int(
                i)][movie_id] * self.user_similarity_matrix[user_id][int(i)]
            denominator += self.user_similarity_matrix[user_id][int(i)]
        if denominator == 0:
            return 0
        return numerator/denominator
    #prediction using sign matrix
    def predict_gs(self, user_id, movie_id):
        numerator = 0
        denominator = 0
        inf_sign = self.inf_sign_list[user_id]
        for i in self.kn_list[user_id]:
            if self.user_movie_matrix[int(i)][movie_id] == 0:
                continue
            numerator += self.user_movie_matrix[int(i)][movie_id]*min(1, max(0, self.user_similarity_matrix[user_id][int(
                i)]-abs(inf_sign-self.inf_sign_list[int(i)])*self.influence))
            denominator += min(1, max(0, self.user_similarity_matrix[user_id][int(
                i)]-abs(inf_sign-self.inf_sign_list[int(i)])*self.influence))
        if denominator == 0:
            return 0
        return numerator/denominator

    def raw_cf_fit(self, data):
        # Constructing user-item matrix
        self.data = data
        self.user_movie_matrix = np.zeros(
            (self.max_user_id+1, self.max_movie_id+1))
        for i in range(0, int(len(data[0]))):
            self.user_movie_matrix[int(self.data[0][i])][int(
                self.data[1][i])] = self.data[2][i]
        # Initialization of user-user similarity matrix
        self.kn_list = np.zeros((self.max_user_id+1, self.k_num))
        # construct user-user similarity matrix
        self.user_similarity_matrix = np.zeros(
            (self.max_user_id+1, self.max_user_id+1))
        for i in range(1, self.max_user_id+1):
            temkn_list = knn_lis(self.k_num)
            for j in range(1, self.max_user_id+1):
                if i == j:
                    continue
                sim = cosine_similarity(
                    self.user_movie_matrix[i], self.user_movie_matrix[j])
                temkn_list.append(j, sim)
                self.user_similarity_matrix[i][j] = sim
            tem_arr = temkn_list.toArray()
            for s in range(0, len(tem_arr)):
                self.kn_list[i][s] = tem_arr[s]

    def gs_fit(self, data):
        self.raw_cf_fit(data)
        self.inf_sign_list = np.zeros(self.max_user_id+1)
        sim_order = knn_lis(self.max_user_id+1)
        for idx, sim_row in enumerate(self.user_similarity_matrix):
            count = 0
            for sim in sim_row:
                if sim > self.sim_threshold:
                    count += 1
            sim_order.append(idx, count)
        sim_order_list = sim_order.toArray()

        for gs_idx in range(0, (int)(self.max_user_id*self.threshold)):
            self.inf_sign_list[sim_order_list[gs_idx]] = -0.5
            #self.inf_sign_list[sim_order_list[gs_idx]] = -1
        for non_gs_idx in range((int)(self.max_user_id*self.threshold), self.max_user_id+1):
            #self.inf_sign_list[sim_order_list[non_gs_idx]] = 1
            self.inf_sign_list[sim_order_list[gs_idx]] = 0.5


def main():
    # Loading Data
    data = pd.read_csv('u.data', sep='\t')
    k_val = 25
    influence = 0.2
    thresh = 0.2
    sim_thresh = 0.5
    cross_val_num = 5
    cf_predictor = cf_alg(k_val, influence, thresh, sim_thresh, data)
    regular = 0
    for i in range(0, round(50/cross_val_num)):
        regular += cross_validation(cross_val_num, data, cf_predictor.raw_cf_fit,
                                   cf_predictor.predict_sim, 1)
    print("K-val: "+str(k_val))
    print("Regular: "+str(sum(regular)/50))
    gs = 0
    for i in range(0,round(50/cross_val_num)):
       gs += cross_validation(cross_val_num, data, cf_predictor.gs_fit, cf_predictor.predict_gs, 1)
    print("Hyper parameter Values: "+str(k_val)+" "+str(influence)+" " +
          str(thresh)+" "+str(sim_thresh)+" "+str(cross_val_num))
    print("Gray Sheep: "+str(sum(gs)/50))




if __name__ == '__main__':
    main()
