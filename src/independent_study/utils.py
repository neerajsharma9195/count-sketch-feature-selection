


def get_top_k_positions(top_k_file):
    with open(top_k_file, 'r') as f:
        topk_data = f.readlines()
        positions = []
        for line in topk_data:
            positions.append(int(line.split(":")[0]))
        return positions



if __name__ == '__main__':
    get_top_k_positions("/src/independent_study/results/topk_features_2000.txt")