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
        test_list = np.zeros((3, int(data_size*setRatio*(k))))
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

    def predict_gs(self, user_id, movie_id):
        numerator = 0
        denominator = 0
        inf_sign = self.inf_sign_list[user_id]
        for i in self.kn_list[user_id]:
            if self.user_movie_matrix[int(i)][movie_id] == 0:
                continue
            numerator += self.user_movie_matrix[int(i)][movie_id]*min(1, max(0, self.user_similarity_matrix[user_id][int(
                i)]+(inf_sign-self.inf_sign_list[int(i)])*self.influence))
            denominator += min(1, max(0, self.user_similarity_matrix[user_id][int(
                i)]+(inf_sign-self.inf_sign_list[int(i)])*self.influence))
        if denominator == 0:
            return 0
        return numerator/denominator
    # creating data structures that will be used in the score function

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

    # Function which provides different kind of measurement


# identify gray sheep based on this similarity matrix(probably using outlier detection)
# Using the above matrix and a variable which is based on gray sheep
# identification(Ex.This attribute will further increase the similarity when users are in the
# same type(gray sheep and gray sheep or non-gray sheep and non-gray sheep and decrease
def main():
    # Loading Data
    data = pd.read_csv('u.data', sep='\t')
    k_val = 25
    influence = 0.2
    thresh = 0.2
    sim_thresh = 0.5
    cross_val_num = 25
    cf_predictor = cf_alg(k_val, influence, thresh, sim_thresh, data)
    regular = 0
    for i in range(0, round(50/cross_val_num)):
        #regular += cross_validation(cross_val_num, data, cf_predictor.gs_fit,
        #                            cf_predictor.predict_gs, 1)
        regular += cross_validation(cross_val_num, data, cf_predictor.raw_cf_fit,
                                   cf_predictor.predict_sim, 1)
    print("Values: "+str(k_val)+" "+str(influence)+" " +
          str(thresh)+" "+str(sim_thresh)+" "+str(cross_val_num))
    print("Regular: "+str(sum(regular)/50))
    #gs = 0
    # for i in range(0,4):
    #   gs += cross_validation(5, data, cf_predictor.gs_fit, cf_predictor.predict_gs, 1)
    #print("Gray Sheep: "+str(sum(gs)/25))

    #(Modified Version)#
    #----Standard----#
    # K-val influence thresh sim_thresh cross_val
    #  25      0.2      0.2     0.5         25
    #GS 0.8001691086548014 0.8361059937049579 0.8681000192583052
    #Regular  0.833964542741702 0.8540044518843996 0.8143368064600517 #Need recomputation
    #---------K-val--------#
    #          5
    #
    #          10
    #
    #          25
    #
    #          50
    #
    #          100
    #
    #--------influence--------#
    #           0.01
    #
    #           0.1
    #
    #           0.2
    #
    #           0.4
    #
    #           1.0
    #
    #--------threeshold------#
    #           0.05
    #
    #           0.1
    #
    #           0.2
    #
    #           0.3
    #
    #           0.5
    #
    #--------sim_thresh------#
    #           0.2
    #
    #           0.3
    #
    #           0.4
    #
    #           0.5
    #
    #           0.6
    #
    #--------cross-val-------#
    #           5
    #
    #Regular  0.8852301446786646 0.8650523608245106
    #           10
    #
    #Regular  0.8618333762828165 0.9075728110734727 
    #           50
    #
    #Regular
    #           100
    #

    #(Pre-Modified Version(Some Modified data included))
    #------------------ 5 - fold cross validation ------------#
    #      K-num influence threshold sim_threshold
    #        10,    0.5,     0.1,       0.5
    # GS
    # MAE  1.169775576902102 1.2148343410723625 1.3112807648232732
    #      K-num influence threshold sim_threshold
    #        5,    0.5,     0.1,       0.5
    # GS    1.5484654439448937 1.5293769788842067 1.5269780393421148
    #      K-num influence threshold sim_threshold
    #        5,    0.2,     0.1,       0.5
    # GS    1.0653914317196622  0.9482218930916114
    #      K-num influence threshold sim_threshold
    #        5,    0.7,     0.1,       0.5
    # GS    1.6182222222222222 1.4874116709676124 1.3882998864445932
    #      K-num influence threshold sim_threshold
    #        5,    1.0,     0.1,       0.5
    # GS     1.592888888888889 1.4896666666666665 1.5759999999999996
    #      K-num influence threshold sim_threshold
    #        5,    0.2,     0.05,       0.5
    # GS     1.3042101688670669  1.401732404960552 1.2417305205825968
    #      K-num influence threshold sim_threshold
    #        5,    0.2,     0.2,       0.5
    # GS     1.3964787329137456 1.2662301986084565 1.2779974508722478
    #      K-num influence threshold sim_threshold
    #        5,    0.2,     0.2,       0.6
    # GS     1.4209437428008445  1.393881669675412
    #BEST##########################
    #      K-num influence threshold sim_threshold
    #        5,    0.2,     0.2,       0.4
    # GS    1.1912288755500864 1.2453689737745148 1.3981474992415923
    #      K-num influence threshold sim_threshold
    #        5,    0.2,     0.2,       0.3
    # GS    1.2654981191058736 1.3217108977631955 1.376294871584944
    #      K-num influence threshold sim_threshold
    #        5,    0.3,     0.3,       0.4
    # GS   1.0073491496638403  1.2333692483891772 1.737605109805155
    #      K-num influence threshold sim_threshold
    #        5,    0.1,     0.2,       0.4
    # GS    1.3206353288535513 1.516499474819229 1.472538630605339  1.2695070176292176
    #      K-num influence threshold sim_threshold
    #        10,    0.2,     0.2,       0.4
    # GS    1.128260557725687  1.0297563938488867  1.1624633109400149 1.1491860623589851
    #      K-num influence threshold sim_threshold
    #        15,    0.2,     0.2,       0.4
    # GS    1.0710607534405645 0.8825639824407535 0.8961600472908968
    #      K-num influence threshold sim_threshold
    #        25,    0.2,     0.2,       0.4
    # GS    0.9422658904102121
    # Regular 0.8852301446786646 0.8650523608245106
    #      K-num influence threshold sim_threshold
    #        50,    0.2,     0.2,       0.4
    #      K-num influence threshold sim_threshold
    #        100,    0.2,     0.2,       0.4
    # GS  0.7821874642185898
    # Regular 0.7990886299722041
    # Regular
    # K-num 5
    # K-num 10 1.0776313799781196  1.1119050051661628 0.9878423272573827
    # K-num 25 0.8852301446786646 0.8650523608245106
    # K-num 50  0.7990886299722041
    # K-num 100 0.7537285975720767
    #----------------10-fold cross validation------------------_#
    #      K-num influence threshold sim_threshold
    #        100,    0.2,     0.2,       0.4
    # GS     0.8858412892342789 0.7748401611232404 0.8463350413153079
    #      K-num influence threshold sim_threshold
    #        50,    0.2,     0.2,       0.4
    # GS    0.7748255326325608  0.7868897333725302 0.808356922702703
    #      K-num influence threshold sim_threshold
    #        25,    0.2,     0.2,       0.4
    # GS    0.9505032418331447 1.0067545318919142
    #      K-num influence threshold sim_threshold
    #        10 0.2 0.1 0.5 10
    # GS     0.9812648874458121  0.9925736154728578 0.9088550731806845
    #      K-num influence threshold sim_threshold
    #        5,    0.2,     0.2,       0.4
    # GS     1.164495548013816  1.3740977088174555
    # Regular
    # K-num 100 0.8184255099872507 0.7497603448600236 11:00
    # K-num 50 0.7614220550994425  0.9203306095527016 8:58
    # K-num 25 0.8618333762828165 0.9075728110734727  8:18
    # K-num 10 1.1378865113551264 0.9726862850523521  7:56
    # K-num 5 1.2451636167110256 1.2652217112658106   7:34
    #--------------------25-fold cross validation--------------#
    #      K-num influence threshold sim_threshold
    #        100,    0.2,     0.2,       0.4
    # GS    0.7741695131344403  0.8008148531892957   0.8338582286259972    11:17
    #      K-num influence threshold sim_threshold
    #        50,    0.2,     0.2,       0.4
    # GS    0.9195466743767097 0.9288892025476264  0.7326148514183997 8:50
    #      K-num influence threshold sim_threshold
    #        25 0.2 0.1 0.5
    # GS    0.9826299314820378 0.894950394446532 0.879968315826838
    #      K-num influence threshold sim_threshold
    #        10 0.2 0.1 0.5
    # GS     0.9957700028869175
    #        25 0.2 0.2 0.5
    #         0.8688725294930921 0.8361059937049579 0.8681000192583052
    #        10,    0.01,     0.2,       0.4
    # GS      1.0273300098078741 0.9723765629578446 1.0539753621396093
    #        10,    0.01,     0.2,       0.5
    # GS     0.9546131635166053 0.9402864196847225
    #      K-num influence threshold sim_threshold
    #        5,    0.2,     0.2,       0.4
    # GS    1.1339289028959723 1.3107733189641424 1.096857505448239
    # Regular
    # K-num 100 0.7593295733341203 0.7361978445484426 0.885831894340043
    # K-num 50  0.8033511767373366 0.8200135832703588 0.9108250169424228
    # K-num 25  0.9836764772101584 0.8540044518843996 0.8143368064600517
    # K-num 10  0.9270324976425419 0.9713690928624138 1.0755074198758956
    # K-num 5   1.144891274552136 1.239971453271159 1.3512551597008793
    #--------------------50-fold cross validation--------------#
    #      K-num influence threshold sim_threshold
    #        100,    0.2,     0.2,       0.4
    # GS  0.8521902657801139
    #      K-num influence threshold sim_threshold
    #        50,    0.2,     0.2,       0.4
    # GS
    #      K-num influence threshold sim_threshold
    #        25,    0.2,     0.2,       0.4
    # GS
    #      K-num influence threshold sim_threshold
    #        10,    0.2,     0.2,       0.4
    # GS
    #        10,    0.01,     0.2,       0.4
    # GS  0.9254280407954643  0.9954281079897999
    #        10,    0.01,     0.2,       0.5
    # GS      0.94878309335933 1.1199107696839206
    #        10,    0.005,     0.2,       0.4
    # GS  0.9107025742672187 1.0623885110179747 0.9788990911494431
    #        10,    0.001,     0.2,       0.4
    # GS   0.9641066421069974 1.0374811157420771
    #      K-num influence threshold sim_threshold
    #        5,    0.2,     0.2,       0.4
    # GS
    #      10 0.2 0.05 0.5
    # GS    1.022522451412698
    # Regular
    # K-num 100 0.8187241634773668 0.8179849249071107
    # K-num 50
    # K-num 25
    # K-num 10
    # K-num 5


if __name__ == '__main__':
    main()
