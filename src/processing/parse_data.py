import os
import json

def process_data(filePath, batch_size=1):
    output = []
    with open(filePath, 'r') as f:
        for line in f.readlines():
            label, features = process_line(line)
            output.append((label, features))
    return output


def process_line(line):
    all_data = line.split(" ")
    features = all_data[1:]
    features_values = [item.split(":") for item in features]
    return all_data[0], features_values


if __name__ == '__main__':
    current_directory = (os.path.dirname(__file__))
    data_directory_path = os.path.join(current_directory, '..', 'data')
    print(data_directory_path)
    fileName = "rcv1_train.binary"
    filePath = os.path.join(data_directory_path, fileName)
    output = process_data(filePath)
    print("len of output {}".format(len(output)))

    '''
    fileName = "/Users/neerajsharma/my_work/umass/umass_study/1st_sem/CS689/final_project/src/data/rcv1_train.binary"
    all_data = process_data(fileName)
    with open("output.txt", 'w') as f:
        f.write(json.dumps(all_data))
        '''
