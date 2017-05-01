import csv
import sys
import math
import random


def load_train_data_set(file_name, train_set):
    with open(file_name, 'r') as csv_file:
        lines = csv.reader(csv_file)
        data_set = list(lines)
        for x in range(1,len(data_set)):
            for y in range(3):
                data_set[x][y] = float(data_set[x][y])
            train_set.append(data_set[x])


def euclidean_distance(cluster, point):
    distance = 0
    for x in range(1,3):
        distance += pow((cluster[x] - point[x]), 2)
    return distance


def compute_sse(k, cluster_list):
    err_val = 0
    for x in range(0, len(k)):
        for y in range(0, len(cluster_list[x])):
            err_val += euclidean_distance(k[x], cluster_list[x][y])
    return err_val


def k_means(train_set, no_of_clusters):
    cluster_list = {}
    k = []
    for max_iter in range(0,25):
        k.clear()
        # Generate/Compute Clusters and add to list k
        for x in range(0, no_of_clusters):
            if max_iter == 0:
                if len(k)==0:
                    k=random.sample(train_set,no_of_clusters)
            else:
                x_val = 0
                y_val = 0
                for t in range(len(cluster_list[x])):
                    x_val += cluster_list[x][t][1]
                    y_val += cluster_list[x][t][2]
                k.append([0, x_val/len(cluster_list[x]), y_val/len(cluster_list[x])])
            cluster_list[x] = []
        # Find the appropriate cluster of each data point in trainingSet
        for x in range(len(train_set)):
            minimum = 1000
            min_cluster = 0
            for y in range(len(k)):
                dist = math.sqrt(euclidean_distance(k[y], train_set[x]))
                if dist < minimum:
                    minimum=dist
                    min_cluster = y
            cluster_list[min_cluster].append(train_set[x])
    sse = compute_sse(k,cluster_list)
    return cluster_list, sse, k


def main(argv):
    input_file = ''
    output_file = ''
    if len(argv) ==0:
        print('k-means.py <number_of_clusters> <input_file>  <output_file>')
        sys.exit(2)
    no_of_clusters=int(argv[1])
    input_file=argv[2]
    output_file=argv[3]

    # prepare Training data
    train_set=[]
    load_train_data_set(input_file, train_set)

    # generate the clusters, SSE and output the results
    final_clusters, sse, k = k_means(train_set,no_of_clusters)

    file = open(output_file, "w")
    for x in range(0, len(k)):
        file.write("Cluster " + str(x+1) + ": [" + str(round(k[x][1],4)) + str(round(k[x][2],4)) + "]" + '\n')
        file.write(str([int(index[0]) for index in final_clusters[x]]))
        file.write('\n\n')
    file.write("Sum of Squared Error (SSE): " + str(round(sse,4)))
    file.close()

if __name__ == "__main__":
   main(sys.argv)