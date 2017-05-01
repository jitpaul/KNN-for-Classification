import json
import sys
import csv


def load_train_data_set(file_name, train_set, id_set):
    for line in open(file_name, 'r'):
        tweet = json.loads(line)
        train_set.append(tweet["text"].split())
        id_set.append(str(tweet["id"]))


def load_initial_cluster_set(file_name, train_set,id_set,k):
    with open(file_name, 'r') as csv_file:
        lines = csv.reader(csv_file)
        clusters = list(lines)
        for x in range(len(clusters)):
            k.append(train_set[id_set.index(clusters[x][0])])


def jaccard_distance(tweet1, tweet2):
    i = len(set(tweet1) & set(tweet2))
    u = len(set(tweet1) | set(tweet2))
    return (u-i)/u


def compute_sse(k, cluster_list):
    err_val = 0
    for x in range(0, len(k)):
        for y in range(0, len(cluster_list[x])):
            err_val += pow(jaccard_distance(k[x], cluster_list[x][y]),2)
    return err_val


def find_minimum_tweet(cluster_list):
    minimum = 1000000
    if len(cluster_list)==0:
        return []
    if len(cluster_list)==1:
        return cluster_list[0]
    for x in range(len(cluster_list)):
        distance = 0
        for y in range(x+1,len(cluster_list)):
            distance += jaccard_distance(cluster_list[x], cluster_list[y])
            if distance < minimum:
                minimum = distance
                minimum_tweet = cluster_list[x]
    return minimum_tweet


def k_means(train_set, no_of_clusters,k):
    cluster_list = {}
    for x in range(no_of_clusters):
        cluster_list[x]=[]
    for max_iter in range(0,25):
        if max_iter != 0:
            # Generate/Compute Clusters and add to list k
            k.clear()
            for x in range(no_of_clusters):
                k.append(find_minimum_tweet(cluster_list[x]))
        else:
            # Find the appropriate cluster of each data point in trainingSet
            for x in range(len(train_set)):
                minimum = 1000
                min_cluster = 0
                for y in range(len(k)):
                    dist = jaccard_distance(k[y], train_set[x])
                    if dist < minimum:
                        minimum=dist
                        min_cluster = y
                cluster_list[min_cluster].append(train_set[x])
    sse = compute_sse(k,cluster_list)
    return cluster_list, sse, k


def main(argv):

    if len(argv) ==0:
        print('tweets-k-means <numberOfClusters> <initialSeedsFile> <TweetsDataFile> <outputFile>')
        sys.exit(2)
    no_of_clusters=25 # hardcoded as 25 since the initial_seeds_file contains 25 clusters.
    initial_seeds_file=argv[2]
    tweets_data_file = argv[3]
    output_file=argv[4]

    # prepare Training data
    train_set=[]
    id_set=[]
    k=[]
    load_train_data_set(tweets_data_file, train_set, id_set)
    load_initial_cluster_set(initial_seeds_file,train_set,id_set,k)

    # generate the clusters, SSE and output the results
    final_clusters, sse, k = k_means(train_set,no_of_clusters,k)

    file = open(output_file, "w")
    for x in range(0, len(k)):
        file.write("Cluster: " + str(x+1) + '\n' + "<")
        for y in range(len(final_clusters[x])):
            file.write(id_set[train_set.index(final_clusters[x][y])] + ", ")
        file.write(">" +'\n\n')
    file.write("Sum of Squared Error (SSE): " + str(round(sse,4)))
    file.close()

if __name__ == "__main__":
    main(sys.argv)