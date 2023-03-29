import torch
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Topic Model data process')
    parser.add_argument('--data_name', type=str, default="test")
    parser.add_argument('--vocabulary_size', type=int, default=2 ** 12)
    parser.add_argument('--input_path', type=str, default="./dataset/test-500/")
    parser.add_argument('--input_file', type=str, default="data.xlsx")
    parser.add_argument('--lang', type=str, default="en")
    parser.add_argument('--output_path', type=str, default="./output/test-500-gsm/")
    #### load config
    args = parser.parse_args()
    print(torch.__version__)
    print(2/0)
    print(args.data_name)
    print("process done")