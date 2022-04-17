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

# used as a helper function for quicksort


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


class cf_alg:
    def __init__(self, data):
        self.data = data

    # Prediction made using user-user-similarity matrix and user-item rating matrix
    def predict_sim(self, user_id, movie_id):
        numerator = 0
        denominator = 0
        for i in self.kn_list[user_id]:
            if self.user_movie_matrix[int(i)][movie_id] == 0:
                continue
            numerator += self.user_movie_matrix[int(i)][movie_id] * \
                self.user_similarity_matrix[user_id][int(i)]
            denominator += self.user_similarity_matrix[user_id][int(i)]
        if denominator == 0:
            return 0
        return numerator/denominator

    # creating data structures that will be used in the score function
    def raw_cf_fit(self, k_num):
        # Constructing user-item matrix
        dataf = pd.DataFrame(data=self.data)
        self.max_user_id = max(dataf['userid'])
        self.max_movie_id = max(dataf['movieid'])
        self.user_movie_matrix = np.zeros(
            (self.max_user_id+1, self.max_movie_id+1))
        for i in range(0, len(dataf)):
            self.user_movie_matrix[self.data['userid'][i]
                                   ][self.data['movieid'][i]] = self.data['ratings'][i]
        # Initialization of user-user similarity matrix
        self.kn_list = np.zeros((1, k_num))
        # construct user-user similarity matrix
        self.user_similarity_matrix = np.zeros(
            (self.max_user_id+1, self.max_user_id+1))
        for i in range(1, self.max_user_id+1):
            temkn_list = knn_lis(k_num)
            for j in range(1, self.max_user_id+1):
                if i == j:
                    continue
                sim = cosine_similarity(
                    self.user_movie_matrix[i], self.user_movie_matrix[j])
                temkn_list.append(j, sim)
                self.user_similarity_matrix[i][j] = sim
            self.kn_list = np.vstack(
                (self.kn_list, np.array(temkn_list.toArray())))

    # Function which provides different kind of measurement
    def score(self, prediction_function, measure_num):
        # mean squared error
        if measure_num == 0:
            mse = 0
            for i in range(1, self.max_user_id+1):
                for j in range(1, self.max_movie_id+1):
                    mse += (self.user_movie_matrix[i][j] -
                            prediction_function(i, j))**2
            mse /= (self.max_user_id*self.max_movie_id)
            print(mse)
        # mean absolute error
        elif measure_num == 1:
            mae = 0
            for i in range(1, self.max_user_id+1):
                for j in range(1, self.max_movie_id+1):
                    mae += abs(self.user_movie_matrix[i]
                               [j]-prediction_function(i, j))
            mae /= (self.max_user_id*self.max_movie_id)
            print(mae)


# identify gray sheep based on this similarity matrix(probably using outlier detection)
# Using the above matrix and a variable which is based on gray sheep
# identification(Ex.This attribute will further increase the similarity when users are in the
# same type(gray sheep and gray sheep or non-gray sheep and non-gray sheep and decrease
def main():
    # Loading Data
    data = pd.read_csv('u.data', sep='\t')
    cf_predictor = cf_alg(data)
    cf_predictor.raw_cf_fit(5)
    cf_predictor.score(cf_predictor.predict_sim, 1)


if __name__ == '__main__':
    main()
